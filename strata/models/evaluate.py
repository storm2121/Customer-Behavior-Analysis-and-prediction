"""The evaluation gate of the daily pipeline (ARCHITECTURE section 12).

After training and before the ledger and the report, ``strata evaluate`` checks that every model in
the zone's registry was trained on the data the warehouse holds now (same fingerprint, same
mapping), and that its headline clears a floor any working model clears. The floors are sanity
checks, not targets:

- silence: the served score (trees with Platt scaling) ranks the later cutoff better than chance,
  that is the lower end of its client-bootstrap AUC interval is above one half;
- adoption: for each target, the lower end of the trees' out-of-fold AUC interval is above one half;
- survival: the Cox model's concordance is above one half;
- segments: the bootstrap stability (mean adjusted Rand index) is above zero.

A result names the model and the check, never a value, so the gate can run in the private zone and
log to an orchestrator without carrying a number out of it.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from strata.config import Settings, Zone
from strata.ingest.load import load_manifest
from strata.ingest.manifest import Manifest
from strata.models import registry

HALF = 0.5


@dataclass(frozen=True, slots=True)
class Check:
    """One check of one model: what was checked, and whether it passed."""

    model: str
    check: str
    passed: bool


def _lower(metric: object) -> float | None:
    """The lower end of a ``{"value", "ci"}`` metric's interval, if it has one."""
    if not isinstance(metric, dict):
        return None
    interval = metric.get("ci")
    if isinstance(interval, list | tuple) and len(interval) == 2:
        return _number(interval[0])
    return None


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _above(value: float | None, floor: float) -> bool:
    return value is not None and math.isfinite(value) and value > floor


def _dig(headline: dict[str, Any], *keys: str) -> object:
    value: object = headline
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _silence(headline: dict[str, Any]) -> list[tuple[str, bool]]:
    auc = _dig(headline, "metrics", "lightgbm_platt", "auc")
    return [("served score ranks better than chance", _above(_lower(auc), HALF))]


def _adoption(headline: dict[str, Any]) -> list[tuple[str, bool]]:
    targets = _dig(headline, "targets")
    if not isinstance(targets, dict) or not targets:
        return [("adoption targets present", False)]
    return [
        (
            f"{target}: better than chance out of fold",
            _above(_lower(_dig(item, "metrics", "lightgbm", "auc")), HALF),
        )
        for target, item in sorted(targets.items())
        if isinstance(item, dict)
    ]


def _survival(headline: dict[str, Any]) -> list[tuple[str, bool]]:
    return [("concordance above chance", _above(_number(headline.get("concordance")), HALF))]


def _segments(headline: dict[str, Any]) -> list[tuple[str, bool]]:
    ari = _number(_dig(headline, "stability", "mean_ari"))
    return [("segments are stable under bootstrap refits", _above(ari, 0.0))]


FLOORS: dict[str, Callable[[dict[str, Any]], list[tuple[str, bool]]]] = {
    "silence": _silence,
    "survival": _survival,
    "adoption": _adoption,
    "segments": _segments,
}


def evaluate(models: registry.Registry, manifest: Manifest) -> list[Check]:
    """Every check of every expected model against the warehouse's current manifest."""
    checks: list[Check] = []
    for name, floors in FLOORS.items():
        try:
            entry = models.get(name)
        except KeyError:
            checks.append(Check(name, "in the registry", passed=False))
            continue
        current = (
            entry.data_fingerprint == manifest.fingerprint
            and entry.mapping_sha256 == manifest.mapping_sha256
        )
        checks.append(Check(name, "trained on the data in the warehouse", current))
        checks.extend(Check(name, label, passed) for label, passed in floors(entry.headline))
    return checks


def evaluate_zone(settings: Settings, zone: Zone) -> list[Check]:
    """The gate on one zone's workspace: its registry against its last ingest."""
    workspace = settings.workspace(zone)
    manifest = load_manifest(workspace)
    models = registry.load(workspace.root / "models" / registry.REGISTRY_FILE)
    return evaluate(models, manifest)
