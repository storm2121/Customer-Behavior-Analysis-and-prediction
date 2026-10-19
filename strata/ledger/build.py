"""``strata ledger build``: evaluate every registered claim and write the ledger files.

Where the files go depends on the zone (ADR 0008):

- public zone (synthetic data): ``.strata/ledger/claims.synthetic.json``; proves the claim
  functions run, feeds the public demo report, and never touches ``docs/``;
- private zone (real data): ``<STRATA_PRIVATE_DIR>/ledger/claims.private.json`` with every claim;
  with ``--public``, also ``docs/ledger/claims.public.json`` with the allow-listed claims only
  (full claims, verdicts and model-quality metrics), after the publication rules of
  :mod:`strata.ledger.allowlist` pass.

A claim that cannot be computed on the warehouse (for example because a group is empty) is
reported as skipped rather than silently dropped; ``--public`` refuses to run if an allow-listed
claim was skipped.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from strata import __version__
from strata.config import PrivacyBoundaryError, Settings, Zone
from strata.ingest.load import load_manifest
from strata.ledger.allowlist import (
    METRICS_ONLY_KEYS,
    METRICS_ONLY_META,
    MIN_PUBLIC_CELL_N,
    is_metrics_only,
    is_public,
    is_verdict_only,
    ledger_violations,
)
from strata.ledger.bootstrap import DEFAULT_RESAMPLES, claim_seed
from strata.ledger.model import (
    ClaimRecord,
    Ledger,
    PublishedVerdict,
    Robustness,
    SourceKind,
    Stratum,
)
from strata.ledger.registry import ClaimContext, ClaimError, ClaimSpec, registered
from strata.ledger.robustness import check
from strata.log import get_logger
from strata.warehouse import connect

log = get_logger(__name__)

PUBLIC_LEDGER = Path("docs/ledger/claims.public.json")
SIGNIFICANT_DIGITS = 6


@dataclass(frozen=True)
class BuildResult:
    ledger: Ledger
    path: Path
    public_path: Path | None = None
    skipped: dict[str, str] = field(default_factory=dict)


def _round(value: float) -> float:
    """Six significant digits: stable files across platforms, far below any reported precision."""
    if value == 0 or not math.isfinite(value):
        return value
    return float(f"{value:.{SIGNIFICANT_DIGITS}g}")


def _rounded_pairs(pairs: dict[str, tuple[float, float]]) -> dict[str, tuple[float, float]]:
    return {key: (_round(low), _round(high)) for key, (low, high) in pairs.items()}


def evaluate(
    spec: ClaimSpec,
    context: ClaimContext,
    *,
    fingerprint: str,
    source_kind: SourceKind,
    computed_at: datetime,
) -> ClaimRecord:
    """Compute one claim, run its robustness check, and attach provenance."""
    context.claim_id = spec.id
    result = spec.compute(context)
    robustness: Robustness | None = None
    if spec.robustness is not None:
        if result.robustness_frame is None:
            msg = f"headline claim {spec.id!r} returned no frame for its robustness check"
            raise ClaimError(msg)
        robustness = check(
            result.robustness_frame,
            spec.robustness,
            n_resamples=context.n_resamples,
            seed=claim_seed(context.seed, f"{spec.id}/robustness"),
            hint=result.mechanism_hint,
            public_hint=result.public_hint,
        )
    return ClaimRecord(
        id=spec.id,
        title=spec.title,
        question=spec.question,
        method=spec.method,
        values={key: _round(value) for key, value in result.values.items()},
        units=result.units,
        ci=_rounded_pairs(result.ci),
        interval_label=result.interval_label,
        value_n=result.value_n,
        n=result.n,
        n_events=result.n_events,
        caveats=result.caveats,
        meta={**result.meta, "effort": "quick"} if context.quick else result.meta,
        headline=spec.headline,
        public=is_public(spec.id),
        robustness=robustness,
        data_fingerprint=fingerprint,
        source_kind=source_kind,
        computed_at=computed_at,
        strata_version=__version__,
    )


def _without_small_values(record: ClaimRecord) -> ClaimRecord:
    """Withhold every value computed on fewer clients than the floor (and its table rows)."""
    small = {key for key, size in record.value_n.items() if size < MIN_PUBLIC_CELL_N}
    if not small:
        return record

    def rows(spec: str) -> str:
        return "|".join(item for item in spec.split("|") if item.split(":", 1)[-1] not in small)

    meta = {
        key: rows(value) if key.startswith("rows_") else value for key, value in record.meta.items()
    }
    meta["withheld_values"] = ", ".join(sorted(small))
    return record.model_copy(
        update={
            "values": {key: value for key, value in record.values.items() if key not in small},
            "units": {key: unit for key, unit in record.units.items() if key not in small},
            "ci": {key: pair for key, pair in record.ci.items() if key not in small},
            "value_n": {key: size for key, size in record.value_n.items() if key not in small},
            "meta": meta,
        }
    )


def public_view(record: ClaimRecord) -> ClaimRecord:
    """The record as it may be published. A value computed on fewer clients than the floor is
    withheld, with its table rows; a stratum keeps its numbers only if each group in it reaches the
    floor, otherwise only its label and sufficiency remain. Stratifier-level adjusted effects, the
    verdict and the explanation are aggregates over many clients and stay."""
    record = _without_small_values(record)
    if record.robustness is None:
        return record
    stratifiers = []
    for stratifier in record.robustness.stratifiers:
        strata = [
            stratum
            if min(stratum.n_treatment, stratum.n_reference) >= MIN_PUBLIC_CELL_N
            else Stratum(
                stratum=stratum.stratum,
                effect=None,
                ci=None,
                n_treatment=0,
                n_reference=0,
                sufficient=stratum.sufficient,
                suppressed=True,
            )
            for stratum in stratifier.strata
        ]
        stratifiers.append(stratifier.model_copy(update={"strata": strata}))
    robustness = record.robustness.model_copy(update={"stratifiers": stratifiers})
    return record.model_copy(update={"robustness": robustness})


def verdict_view(record: ClaimRecord) -> PublishedVerdict:
    """A headline as a ``verdict_only`` publication may show it: the verdict, the stratifier names,
    the stratifiers that reverse the effect and the number-free explanation; nothing else."""
    robustness = record.robustness
    if robustness is None or not robustness.public_mechanism:
        msg = f"claim {record.id!r} has no robustness result to publish as a verdict"
        raise ClaimError(msg)
    return PublishedVerdict(
        id=record.id,
        title=record.title,
        question=record.question,
        verdict=robustness.verdict,
        stratifiers=[result.stratifier for result in robustness.stratifiers],
        reversing_stratifiers=[
            result.stratifier for result in robustness.stratifiers if result.reverses
        ],
        mechanism=robustness.public_mechanism,
        data_fingerprint=record.data_fingerprint,
        source_kind=record.source_kind,
        computed_at=record.computed_at,
        strata_version=record.strata_version,
    )


def metrics_view(record: ClaimRecord) -> ClaimRecord:
    """A model-quality claim as a ``metrics_only`` publication may show it: its allow-listed
    metrics with their intervals and the clients behind them, and how they were measured; no other
    value, no robustness result, and only for a model trained at full effort."""
    allowed = METRICS_ONLY_KEYS.get(record.id)
    if allowed is None:
        msg = f"claim {record.id!r} is not allow-listed for metrics-only publication"
        raise ClaimError(msg)
    if record.meta.get("model_effort") != "full" or record.meta.get("effort") == "quick":
        msg = (
            f"claim {record.id!r} rests on a quick model or build; publish full-effort results only"
        )
        raise ClaimError(msg)
    if record.n < MIN_PUBLIC_CELL_N:
        msg = f"claim {record.id!r} rests on fewer clients than the publication floor"
        raise ClaimError(msg)
    return record.model_copy(
        update={
            "values": {key: value for key, value in record.values.items() if key in allowed},
            "units": {key: unit for key, unit in record.units.items() if key in allowed},
            "ci": {key: pair for key, pair in record.ci.items() if key in allowed},
            "value_n": {key: size for key, size in record.value_n.items() if key in allowed},
            "meta": {key: text for key, text in record.meta.items() if key in METRICS_ONLY_META},
            "robustness": None,
            "public": True,
        }
    )


def _dump(ledger: Ledger, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = ledger.model_dump(mode="json")
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )


def _selected(only: list[str] | None) -> list[ClaimSpec]:
    specs = registered()
    unknown = sorted(set(only or []) - set(specs))
    if unknown:
        msg = f"unknown claim ids: {unknown}"
        raise ClaimError(msg)
    return [specs[claim_id] for claim_id in only] if only else list(specs.values())


def _merged(
    previous: Ledger,
    records: list[ClaimRecord],
    fingerprint: str,
    attempted: set[str],
    mapping_sha256: str | None = None,
) -> list[ClaimRecord]:
    """``records`` plus the claims of ``previous`` that were not attempted, for the same data.

    A claim that was attempted and skipped loses its old record: a stale number must not survive
    the claim failing on the current code.
    """
    if previous.data_fingerprint != fingerprint or previous.mapping_sha256 != mapping_sha256:
        return records
    kept = [record for record in previous.claims if record.id not in attempted]
    # Registry order, whatever order the modules were recomputed in.
    order = {claim_id: rank for rank, claim_id in enumerate(registered())}
    return sorted([*kept, *records], key=lambda record: order.get(record.id, len(order)))


def build_ledger(
    settings: Settings,
    zone: Zone,
    *,
    public: bool = False,
    n_resamples: int = DEFAULT_RESAMPLES,
    only: list[str] | None = None,
    quick: bool = False,
) -> BuildResult:
    """Evaluate the registered claims (or ``only`` those) on the zone's warehouse.

    With ``only``, the new records replace their namesakes in the existing ledger of the same data
    (same fingerprint) and every other claim is kept, so modules can be recomputed one at a time.
    ``quick`` lowers the effort of model-based claims (tests and demos); each claim says so.
    """
    workspace = settings.workspace(zone)
    manifest = load_manifest(workspace)
    source_kind: SourceKind = manifest.source_kind
    if public and (zone is not Zone.PRIVATE or source_kind != "real"):
        msg = "--public writes real allow-listed claims: run it with --private on the real extract."
        raise PrivacyBoundaryError(msg)
    if public and quick:
        msg = "--public publishes claims: build them at full effort, without --quick."
        raise ClaimError(msg)
    computed_at = datetime.now(UTC).replace(microsecond=0)
    selected = _selected(only)
    records: list[ClaimRecord] = []
    skipped: dict[str, str] = {}
    con = connect(workspace.warehouse, read_only=True)
    try:
        context = ClaimContext(
            con,
            seed=settings.seed,
            n_resamples=n_resamples,
            quick=quick,
            models_dir=workspace.root / "models",
            fingerprint=manifest.fingerprint,
            mapping_sha256=manifest.mapping_sha256,
        )
        for spec in selected:
            try:
                records.append(
                    evaluate(
                        spec,
                        context,
                        fingerprint=manifest.fingerprint,
                        source_kind=source_kind,
                        computed_at=computed_at,
                    )
                )
            except ClaimError as error:
                skipped[spec.id] = str(error)
                log.warning("claim skipped", claim=spec.id, reason=str(error))
    finally:
        con.close()
    name = "claims.private.json" if workspace.is_private else "claims.synthetic.json"
    path = workspace.ledger_dir / name
    if workspace.is_private:
        settings.assert_private_output(path)
    if only and path.is_file():
        attempted = {spec.id for spec in selected}
        records = _merged(
            load_ledger(path), records, manifest.fingerprint, attempted, manifest.mapping_sha256
        )
    ledger = Ledger(
        source_kind=source_kind,
        data_fingerprint=manifest.fingerprint,
        mapping_sha256=manifest.mapping_sha256,
        computed_at=computed_at,
        claims=records,
    )
    _dump(ledger, path)
    public_path = None
    if public:
        public_path = _publish(settings, ledger, skipped)
    return BuildResult(ledger=ledger, path=path, public_path=public_path, skipped=skipped)


def _publish(settings: Settings, ledger: Ledger, skipped: dict[str, str]) -> Path:
    """Write the public ledger: full claims through :func:`public_view`, model-quality claims
    through :func:`metrics_view`, verdict-only headlines through :func:`verdict_view`, after the
    publication rules pass."""
    missing = sorted(
        claim_id
        for claim_id in skipped
        if is_public(claim_id) or is_verdict_only(claim_id) or is_metrics_only(claim_id)
    )
    if missing:
        msg = f"allow-listed claims could not be computed: {missing}"
        raise ClaimError(msg)
    shared = Ledger(
        source_kind=ledger.source_kind,
        data_fingerprint=ledger.data_fingerprint,
        mapping_sha256=ledger.mapping_sha256,
        computed_at=ledger.computed_at,
        claims=[
            metrics_view(record) if is_metrics_only(record.id) else public_view(record)
            for record in ledger.claims
            if record.public or is_metrics_only(record.id)
        ],
        verdicts=[verdict_view(record) for record in ledger.claims if is_verdict_only(record.id)],
    )
    problems = ledger_violations(shared.model_dump(mode="json"))
    if problems:
        raise PrivacyBoundaryError("; ".join(problems))
    public_path = settings.repo_root / PUBLIC_LEDGER
    _dump(shared, public_path)
    return public_path


def load_ledger(path: Path) -> Ledger:
    """Read a ledger file written by :func:`build_ledger`."""
    if not path.is_file():
        msg = f"No ledger at {path.name}: run `strata ledger build` first."
        raise FileNotFoundError(msg)
    return Ledger.model_validate_json(path.read_text(encoding="utf-8"))
