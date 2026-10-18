"""The ledger build: every claim computes, headlines carry verdicts, the boundary holds."""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from strata.cli import app
from strata.config import SYNTHETIC_MARKER, PrivacyBoundaryError, Settings, Zone, is_within
from strata.ledger.allowlist import MIN_PUBLIC_CELL_N, PUBLIC_CLAIM_IDS, public_violations
from strata.ledger.build import build_ledger, evaluate, load_ledger, public_view
from strata.ledger.model import ClaimRecord, Ledger, Unit
from strata.ledger.registry import (
    REGISTRY,
    ClaimContext,
    ClaimError,
    ClaimResult,
    ClaimSpec,
    claim,
    registered,
)
from strata.ledger.robustness import RobustnessSpec
from strata.warehouse import connect
from tests.conftest import build_warehouse


def test_every_registered_claim_computes_on_synthetic_data(synthetic_ledger: Ledger) -> None:
    assert sorted(synthetic_ledger.ids) == sorted(registered())
    assert synthetic_ledger.source_kind == "synthetic"
    for record in synthetic_ledger.claims:
        assert record.values
        assert set(record.units) == set(record.values)
        assert set(record.ci) <= set(record.values)
        assert record.n > 0
        assert record.public == (record.id in PUBLIC_CLAIM_IDS)
        for key, (low, high) in record.ci.items():
            assert low <= high, (record.id, key)


def test_every_headline_carries_a_robustness_verdict(synthetic_ledger: Ledger) -> None:
    headlines = [record for record in synthetic_ledger.claims if record.headline]
    assert {record.id for record in headlines} >= {"silence_by_channel", "age_digital_divide"}
    for record in headlines:
        assert record.robustness is not None
        assert record.robustness.verdict in {"holds", "weakens", "reverses", "insufficient"}
        assert record.robustness.mechanism


def test_the_synthetic_silence_finding_reverses_by_activity(synthetic_ledger: Ledger) -> None:
    """The generator encodes the documented mechanism, so the demo shows the reversal."""
    robustness = synthetic_ledger.get("silence_by_channel").robustness
    assert robustness is not None
    assert robustness.verdict == "reverses"
    reversing = {s.stratifier for s in robustness.stratifiers if s.reverses}
    assert "activity_tercile" in reversing
    assert robustness.overall_effect > 0


def test_shares_are_proportions(synthetic_ledger: Ledger) -> None:
    for record in synthetic_ledger.claims:
        for key, unit in record.units.items():
            if unit is Unit.SHARE:
                assert -1e-9 <= record.values[key] <= 1 + 1e-9, (record.id, key)
    mix = synthetic_ledger.get("channel_mix")
    assert sum(mix.values.values()) == pytest.approx(1.0)


def test_builds_are_deterministic(built_warehouse: Settings) -> None:
    first = build_ledger(built_warehouse, Zone.PUBLIC, n_resamples=60, only=["channel_mix"])
    second = build_ledger(built_warehouse, Zone.PUBLIC, n_resamples=60, only=["channel_mix"])
    assert first.ledger.claims[0].values == second.ledger.claims[0].values
    assert first.ledger.claims[0].ci == second.ledger.claims[0].ci
    assert first.path.name == "claims.synthetic.json"
    assert load_ledger(first.path).ids == ["channel_mix"]


def test_public_claims_cannot_be_built_from_synthetic_data(built_warehouse: Settings) -> None:
    with pytest.raises(PrivacyBoundaryError, match="--private"):
        build_ledger(built_warehouse, Zone.PUBLIC, public=True, only=["channel_mix"])


def test_load_ledger_explains_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="ledger build"):
        load_ledger(tmp_path / "absent.json")


def test_a_private_build_writes_private_claims_and_the_public_subset(
    tmp_path: Path, synthetic_dir: Path
) -> None:
    repo = tmp_path / "repo"
    shutil.copytree(
        Settings().repo_root / "dbt", repo / "dbt", ignore=shutil.ignore_patterns("target", "logs")
    )
    extract = tmp_path / "extract"
    shutil.copytree(synthetic_dir, extract)
    (extract / SYNTHETIC_MARKER).unlink()
    settings = Settings(_env_file=None, repo_root=repo, data_dir=extract)  # type: ignore[call-arg]
    build_warehouse(settings, Zone.PRIVATE, None)
    result = build_ledger(settings, Zone.PRIVATE, public=True, n_resamples=60)
    assert is_within(result.path, extract / "private_out")
    assert result.path.name == "claims.private.json"
    assert result.public_path == repo / "docs" / "ledger" / "claims.public.json"
    payload = json.loads(result.public_path.read_text(encoding="utf-8"))
    assert {record["id"] for record in payload["claims"]} == set(PUBLIC_CLAIM_IDS)
    assert public_violations(payload["claims"]) == []
    private = load_ledger(result.path)
    assert "silence_by_channel" in private.ids
    assert private.source_kind == "real"  # the folder had no synthetic marker


def record_with_strata(n_treatment: int, n_reference: int) -> ClaimRecord:
    from strata.ledger.model import Robustness, StratifierResult, Stratum

    stratum = Stratum(
        stratum="x",
        effect=0.1,
        ci=(0.0, 0.2),
        n_treatment=n_treatment,
        n_reference=n_reference,
        sufficient=True,
    )
    robustness = Robustness(
        group_col="g",
        treatment="a",
        reference="b",
        metric="m",
        overall_effect=0.1,
        overall_ci=(0.0, 0.2),
        n_treatment=200,
        n_reference=200,
        stratifiers=[
            StratifierResult(
                stratifier="s", strata=[stratum], adjusted_effect=0.1, sign_flips=0, reverses=False
            )
        ],
        sign_flips=0,
        shrinkage=0.0,
        verdict="holds",
        mechanism="text",
    )
    return ClaimRecord(
        id="channel_mix",
        title="t",
        question="q",
        method="m",
        values={"v": 0.5},
        units={"v": Unit.SHARE},
        value_n={"v": 500},
        n=500,
        public=True,
        robustness=robustness,
        data_fingerprint="0" * 64,
        source_kind="real",
        computed_at=datetime(2026, 1, 1, tzinfo=UTC),
        strata_version="test",
    )


def test_public_view_suppresses_small_strata() -> None:
    small = public_view(record_with_strata(60, 20))
    assert small.robustness is not None
    stratum = small.robustness.stratifiers[0].strata[0]
    assert stratum.suppressed
    assert stratum.effect is None
    assert stratum.n_treatment == stratum.n_reference == 0
    large = public_view(record_with_strata(MIN_PUBLIC_CELL_N, MIN_PUBLIC_CELL_N))
    assert not large.robustness.stratifiers[0].strata[0].suppressed  # type: ignore[union-attr]
    plain = record_with_strata(60, 60).model_copy(update={"robustness": None})
    assert public_view(plain) is plain


def test_publication_rules_catch_small_cells_and_unknown_claims() -> None:
    record = record_with_strata(60, 20).model_dump(mode="json")
    assert any("stratum" in p for p in public_violations([record]))
    record["robustness"]["n_reference"] = 10
    assert any("compares a group" in p for p in public_violations([record]))
    assert public_violations([{"id": "secret", "n": 900}]) == ["claim 'secret' is not allow-listed"]
    assert public_violations([{"n": 900}]) == ["claim #0 has no id"]
    flagged = public_violations([{"id": "channel_mix", "n": True}])
    assert flagged
    assert "minimum" in flagged[0]


def test_a_headline_without_a_robustness_spec_is_refused() -> None:
    with pytest.raises(ValueError, match="robustness"):
        ClaimSpec(
            id="x", title="t", question="q", method="m", compute=lambda c: None, headline=True
        )  # type: ignore[arg-type,return-value]


def test_claims_cannot_be_registered_twice() -> None:
    with pytest.raises(ValueError, match="twice"):
        claim("channel_mix", title="t", question="q", method="m")(lambda c: None)  # type: ignore[arg-type,return-value]


def test_a_headline_must_return_its_robustness_frame(built_warehouse: Settings) -> None:
    spec = ClaimSpec(
        id="broken_headline",
        title="t",
        question="q",
        method="m",
        compute=lambda context: ClaimResult(values={"v": 1.0}, units={"v": Unit.NUMBER}, n=60),
        headline=True,
        robustness=RobustnessSpec("g", "a", "b", "m"),
    )
    con = connect(built_warehouse.workspace(Zone.PUBLIC).warehouse, read_only=True)
    try:
        with pytest.raises(ClaimError, match="no frame"):
            evaluate(
                spec,
                ClaimContext(con),
                fingerprint="f",
                source_kind="synthetic",
                computed_at=datetime.now(UTC),
            )
    finally:
        con.close()


def test_failing_claims_are_skipped_not_hidden(
    built_warehouse: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(context: ClaimContext) -> ClaimResult:
        raise ClaimError("not enough clients")

    spec = ClaimSpec(id="always_fails", title="t", question="q", method="m", compute=fail)
    monkeypatch.setitem(REGISTRY, "always_fails", spec)
    result = build_ledger(built_warehouse, Zone.PUBLIC, n_resamples=50, only=["always_fails"])
    assert result.skipped == {"always_fails": "not enough clients"}
    assert result.ledger.claims == []


def test_records_reject_non_finite_values() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        ClaimRecord(
            id="x",
            title="t",
            question="q",
            method="m",
            values={"v": float("nan")},
            units={"v": Unit.NUMBER},
            n=1,
            data_fingerprint="f",
            source_kind="synthetic",
            computed_at=datetime.now(UTC),
            strata_version="test",
        )


def test_ledger_lookup_errors_are_explicit(synthetic_ledger: Ledger) -> None:
    with pytest.raises(KeyError, match="no claim"):
        synthetic_ledger.get("missing")
    with pytest.raises(KeyError, match="no value"):
        synthetic_ledger.get("channel_mix").value("missing")


def test_cli_ledger_build_lists_claims_and_verdicts(
    built_warehouse: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STRATA_DB", str(built_warehouse.db))
    monkeypatch.setenv("STRATA_WORK_DIR", str(built_warehouse.work_dir))
    result = CliRunner().invoke(
        app, ["ledger", "build", "--resamples", "60", "--only", "silence_by_channel"]
    )
    assert result.exit_code == 0, result.output
    assert "reverses" in result.output
    assert "claims.synthetic.json" in result.output
