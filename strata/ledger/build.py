"""``strata ledger build``: evaluate every registered claim and write the ledger files.

Where the files go depends on the zone (ADR 0008):

- public zone (synthetic data): ``.strata/ledger/claims.synthetic.json``; proves the claim
  functions run, feeds the public demo report, and never touches ``docs/``;
- private zone (real data): ``<STRATA_PRIVATE_DIR>/ledger/claims.private.json`` with every claim;
  with ``--public``, also ``docs/ledger/claims.public.json`` with the allow-listed claims only,
  after the publication rules of :mod:`strata.ledger.allowlist` pass.

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
from strata.ledger.allowlist import MIN_PUBLIC_CELL_N, is_public, public_violations
from strata.ledger.bootstrap import DEFAULT_RESAMPLES, claim_seed
from strata.ledger.model import ClaimRecord, Ledger, Robustness, SourceKind, Stratum
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
        )
    return ClaimRecord(
        id=spec.id,
        title=spec.title,
        question=spec.question,
        method=spec.method,
        values={key: _round(value) for key, value in result.values.items()},
        units=result.units,
        ci=_rounded_pairs(result.ci),
        value_n=result.value_n,
        n=result.n,
        n_events=result.n_events,
        caveats=result.caveats,
        meta=result.meta,
        headline=spec.headline,
        public=is_public(spec.id),
        robustness=robustness,
        data_fingerprint=fingerprint,
        source_kind=source_kind,
        computed_at=computed_at,
        strata_version=__version__,
    )


def public_view(record: ClaimRecord) -> ClaimRecord:
    """The record as it may be published: a stratum keeps its numbers only if each group in it
    reaches the cell floor; otherwise only its label and sufficiency remain. Stratifier-level
    adjusted effects, the verdict and the explanation are aggregates over many clients and stay."""
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


def _dump(ledger: Ledger, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = ledger.model_dump(mode="json")
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )


def build_ledger(
    settings: Settings,
    zone: Zone,
    *,
    public: bool = False,
    n_resamples: int = DEFAULT_RESAMPLES,
    only: list[str] | None = None,
) -> BuildResult:
    """Evaluate the registered claims (or ``only`` those) on the zone's warehouse."""
    workspace = settings.workspace(zone)
    manifest = load_manifest(workspace)
    source_kind: SourceKind = manifest.source_kind
    if public and (zone is not Zone.PRIVATE or source_kind != "real"):
        msg = "--public writes real allow-listed claims: run it with --private on the real extract."
        raise PrivacyBoundaryError(msg)
    computed_at = datetime.now(UTC).replace(microsecond=0)
    specs = registered()
    selected = [specs[claim_id] for claim_id in only] if only else list(specs.values())
    records: list[ClaimRecord] = []
    skipped: dict[str, str] = {}
    con = connect(workspace.warehouse, read_only=True)
    try:
        context = ClaimContext(con, seed=settings.seed, n_resamples=n_resamples)
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
    ledger = Ledger(
        source_kind=source_kind,
        data_fingerprint=manifest.fingerprint,
        computed_at=computed_at,
        claims=records,
    )
    name = "claims.private.json" if workspace.is_private else "claims.synthetic.json"
    path = workspace.ledger_dir / name
    if workspace.is_private:
        settings.assert_private_output(path)
    _dump(ledger, path)
    public_path = None
    if public:
        missing = sorted(claim_id for claim_id in skipped if is_public(claim_id))
        if missing:
            msg = f"allow-listed claims could not be computed: {missing}"
            raise ClaimError(msg)
        shared = Ledger(
            source_kind=source_kind,
            data_fingerprint=manifest.fingerprint,
            computed_at=computed_at,
            claims=[public_view(record) for record in records if record.public],
        )
        problems = public_violations(record.model_dump(mode="json") for record in shared.claims)
        if problems:
            raise PrivacyBoundaryError("; ".join(problems))
        public_path = settings.repo_root / PUBLIC_LEDGER
        _dump(shared, public_path)
    return BuildResult(ledger=ledger, path=path, public_path=public_path, skipped=skipped)


def load_ledger(path: Path) -> Ledger:
    """Read a ledger file written by :func:`build_ledger`."""
    if not path.is_file():
        msg = f"No ledger at {path.name}: run `strata ledger build` first."
        raise FileNotFoundError(msg)
    return Ledger.model_validate_json(path.read_text(encoding="utf-8"))
