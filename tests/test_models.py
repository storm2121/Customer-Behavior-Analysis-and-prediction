"""Risk models: leakage, metrics, the four model families, the registry and the model cards."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from strata.cli import app
from strata.config import PrivacyBoundaryError, Settings, Zone
from strata.dbt_runner import run_dbt
from strata.ledger.model import Ledger
from strata.models import (
    adoption,
    cards,
    churn,
    metrics,
    plots,
    registry,
    segments,
    splits,
    survival,
)
from strata.models.train import train
from strata.warehouse import connect, read_relation, replace_table
from tests.conftest import public_settings

OUTCOME_COLUMNS = {"n_events_outcome", "silent", "outcome_start", "outcome_end", "role"}


def _read(settings: Settings, name: str) -> pd.DataFrame:
    con = connect(settings.workspace().warehouse, read_only=True)
    try:
        return read_relation(con, name)
    finally:
        con.close()


@pytest.mark.slow
def test_no_feature_reads_past_its_cutoff(built_warehouse: Settings, tmp_path: Path) -> None:
    """Rewrite every event after the earliest cutoff; that cutoff's features must not move."""
    cutoff = pd.Timestamp(splits.earliest().cutoff) + pd.Timedelta(days=1)
    tables = {
        name: _read(built_warehouse, f"raw.{name}")
        for name in ("clients", "events", "client_aggregates")
    }
    events = tables["events"]
    after = pd.to_datetime(events["ts"]) >= cutoff
    later = events[after].iloc[::2].copy()
    later["amount"] = later["amount"] * 3 + 1
    tables["events"] = pd.concat([events[~after], later, later], ignore_index=True)
    perturbed = public_settings(tmp_path / "perturbed")
    con = connect(perturbed.workspace().warehouse)
    try:
        for name, frame in tables.items():
            replace_table(con, "raw", name, frame)
    finally:
        con.close()
    assert run_dbt(perturbed, Zone.PUBLIC, ["seed"]).success
    assert run_dbt(perturbed, Zone.PUBLIC, ["run", "--select", "+features_silence"]).success
    original, changed = (
        _read(built_warehouse, "marts.features_silence"),
        _read(perturbed, "marts.features_silence"),
    )
    columns = [c for c in original.columns if c not in OUTCOME_COLUMNS]
    early = splits.earliest().cutoff
    a = splits.rows_for(original, early)[columns]
    b = splits.rows_for(changed, early)[columns]
    pd.testing.assert_frame_equal(a, b, check_dtype=False)
    pd.testing.assert_frame_equal(
        splits.design(splits.rows_for(original, early)),
        splits.design(splits.rows_for(changed, early)),
    )
    # The perturbation was real: the later cutoffs' features did change.
    late = splits.primary().cutoff
    assert not splits.rows_for(original, late)[columns].equals(
        splits.rows_for(changed, late)[columns]
    )


def test_backtests_move_forward_in_time() -> None:
    tests = splits.backtests()
    assert len(tests) == 2
    assert all(t.train.cutoff < t.test.cutoff for t in tests)
    assert tests[-1].test.role == "primary"
    assert splits.earliest().cutoff < splits.primary().cutoff


def test_metrics_on_known_cases() -> None:
    y = np.array([0, 0, 1, 1])
    perfect = np.array([0.1, 0.2, 0.8, 0.9])
    assert metrics.auc(y, perfect) == 1.0
    assert metrics.auc(y, perfect[::-1]) == 0.0
    assert metrics.brier(y, y.astype(float)) == 0.0
    assert metrics.precision_at(y, perfect, 0.5) == 1.0
    assert metrics.lift_at(y, perfect, 0.5) == pytest.approx(2.0)
    assert np.isnan(metrics.auc(np.zeros(4, dtype=np.int64), perfect))
    calibrated = np.full(1000, 0.3)
    outcomes = (np.arange(1000) < 300).astype(np.int64)
    assert metrics.calibration_error(outcomes, calibrated) == pytest.approx(0.0)
    assert metrics.log_loss(y, np.array([0.0, 0.0, 1.0, 1.0])) < 1e-9
    curve = metrics.reliability(outcomes, calibrated)
    assert curve[0]["n"] == 1000
    assert curve[0]["observed"] == pytest.approx(0.3)


def test_bootstrap_intervals_and_paired_differences() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    p = np.clip(y * 0.6 + rng.normal(0.2, 0.2, 400), 0, 1)
    evaluated = metrics.evaluate(y, p, n_resamples=200, seed=1)
    for name, metric in evaluated.items():
        assert metric.ci[0] <= metric.value <= metric.ci[1], name
    same = metrics.paired_difference(y, p, p, metrics.auc, n_resamples=50)
    assert same.value == 0.0
    assert same.ci == (0.0, 0.0)


@pytest.fixture(scope="module")
def silence(built_warehouse: Settings) -> churn.ChurnResult:
    return churn.run(_read(built_warehouse, "marts.features_silence"), effort=churn.QUICK, seed=3)


def test_the_silence_models_are_compared_on_later_cutoffs(silence: churn.ChurnResult) -> None:
    assert [b.test_cutoff > b.train_cutoff for b in silence.backtests] == [True, True]
    primary = silence.primary
    assert set(primary.metrics) == {
        "majority",
        "activity_rule",
        "glm",
        "lightgbm",
        "lightgbm_isotonic",
        "lightgbm_platt",
    }
    assert primary.metrics["majority"]["auc"].value == pytest.approx(0.5)
    assert primary.metrics["lightgbm"]["auc"].value > 0.6
    # Platt scaling is monotone: it keeps the trees' ranking.
    assert primary.metrics["lightgbm_platt"]["auc"].value == pytest.approx(
        primary.metrics["lightgbm"]["auc"].value
    )
    assert all(item["odds_ratio"] > 0 for item in silence.odds_ratios.values())
    scored = silence.scored
    assert scored["probability"].is_monotonic_decreasing
    assert scored["probability"].between(0, 1).all()
    assert isinstance(scored["reasons"].iloc[0], list)
    assert list(silence.shap_global.values()) == sorted(silence.shap_global.values(), reverse=True)
    # SHAP is computed for every backtest, and the last one is the primary's.
    assert all(backtest.shap_global for backtest in silence.backtests)
    assert silence.backtests[-1].shap_global == silence.shap_global


def test_the_reliability_diagram_is_a_png_of_the_bins(tmp_path: Path) -> None:
    bins = [
        {"bin_low": 0.0, "bin_high": 0.1, "n": 400, "predicted": 0.05, "observed": 0.06},
        {"bin_low": 0.5, "bin_high": 0.6, "n": 40, "predicted": 0.55, "observed": 0.5},
    ]
    target = tmp_path / "models" / "silence" / plots.RELIABILITY_IMAGE
    image = plots.reliability_png({"lightgbm_platt": bins, "glm": bins}, target, "a title")
    assert image == target
    assert image.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_training_the_silence_model_draws_its_reliability_and_copies_it_beside_the_card(
    built_warehouse: Settings, synthetic_ledger: Ledger
) -> None:
    # The synthetic ledger fixture trains the silence model on this warehouse first.
    assert synthetic_ledger.get("silence_model_quality").n > 0
    folder = built_warehouse.workspace(Zone.PUBLIC).root / "models"
    assert (folder / "silence" / plots.RELIABILITY_IMAGE).is_file()
    assert (folder / "cards" / cards.RELIABILITY_CARD_IMAGE).is_file()
    card = (folder / "cards" / "silence.md").read_text(encoding="utf-8")
    assert f"]({cards.RELIABILITY_CARD_IMAGE})" in card
    assert "The three largest drivers of every backtest" in card


def test_survival_gaps_and_kaplan_meier() -> None:
    events = pd.DataFrame(
        {
            "client_id": ["a", "a", "a", "b"],
            "event_date": pd.to_datetime(["2024-09-01", "2024-09-01", "2024-09-11", "2024-09-30"]),
        }
    )
    frame = survival.gaps(events, end=datetime(2024, 10, 2, tzinfo=UTC).date())
    assert frame.to_dict(orient="list") == {
        "client_id": ["a", "a", "b"],
        "duration": [10.0, 21.0, 2.0],
        "observed": [True, False, False],
    }
    durations = np.array([1.0, 2.0, 3.0, 4.0])
    observed = np.array([True, True, True, True])
    assert survival.km_median(durations, observed) == 2.0
    weighted = survival.km_median(durations, observed, np.array([0.0, 0.0, 1.0, 1.0]) + 1e-12)
    assert weighted == 3.0
    assert np.isnan(survival.km_median(durations, np.zeros(4, dtype=bool)))


def test_survival_against_lifelines() -> None:
    from lifelines import KaplanMeierFitter

    rng = np.random.default_rng(5)
    durations = rng.integers(1, 60, 300).astype(float)
    observed = rng.random(300) < 0.7
    fitter = KaplanMeierFitter().fit(durations, observed)
    assert survival.km_median(durations, observed) == fitter.median_survival_time_
    assert survival.km_survival_at(durations, observed, 20) == pytest.approx(
        float(fitter.survival_function_at_times(20).iloc[0])
    )


def test_survival_summaries_on_synthetic_data(built_warehouse: Settings) -> None:
    result = survival.run(
        _read(built_warehouse, "intermediate.int_events"),
        _read(built_warehouse, "marts.client_360"),
        n_resamples=20,
    )
    groups = {g["group"] for g in result["gap_by_profile"]}
    assert groups <= {"branch_only", "digital_only", "mixed", "none"}
    for curve in result["gap_by_profile"]:
        low, high = curve["median_ci"]
        assert low <= curve["median"] <= high
    assert set(result["cox"]["hazard_ratios"]) >= {"age_decades", "log_events"}
    assert 0.5 < result["cox"]["concordance"] <= 1.0
    assert result["silence_clients"] > 0


def test_adoption_never_uses_the_online_counts(built_warehouse: Settings) -> None:
    sample = _read(built_warehouse, "marts.aggregate_clients")
    columns = adoption.features(sample[sample["client_id"].notna()]).columns
    assert not any(c.startswith(("ops_", "op_amount_")) for c in columns)
    assert not set(adoption.TARGETS) & set(columns)
    result = adoption.run(sample, n_resamples=30)
    for target in adoption.TARGETS:
        item = result[target]
        assert 0 < item["rate"] < 1
        assert sum(d["n"] for d in item["deciles"]) == item["n_clients"]
        assert item["deciles"][-1]["cumulative_gain"] == pytest.approx(1.0)
        assert sum(item["importance"].values()) == pytest.approx(1.0)


def test_segments_are_selected_and_profiled(built_warehouse: Settings) -> None:
    features = _read(built_warehouse, "marts.features_segments")
    x = segments.matrix(features.sort_values("client_id").reset_index(drop=True))
    assert np.allclose(x.mean(), 0, atol=1e-9)
    result = segments.run(features, effort=segments.QUICK, seed=2)
    assert 2 <= result["k"] <= segments.QUICK.k_max
    assert sum(s["share"] for s in result["segments"]) == pytest.approx(1.0)
    assert -1 <= result["stability"]["mean_ari"] <= 1
    assert all(" / " in s["name"] for s in result["segments"])
    assert segments.describe("card_recency_never", high=True) == "no card use"
    assert segments.describe("online_recency_days", high=False) == "recent online use"
    assert segments.describe("card_share", high=True) == "high card share"
    assert len(result["assignments"]) == result["n_clients"]


def test_segment_names_take_their_second_word_from_another_channel() -> None:
    ranked = ["branch_recency_never", "branch_monetary", "card_frequency", "online_share"]
    assert segments.distinctive(ranked) == ["branch_recency_never", "card_frequency"]
    assert segments.distinctive(["card_frequency", "card_monetary"]) == ["card_frequency"]
    assert segments.distinctive([]) == []


def _entry(source_kind: str = "synthetic") -> registry.ModelEntry:
    return registry.entry(
        name="segments",
        title="t",
        manifest_fingerprint="f" * 64,
        mapping_sha256=None,
        source_kind=source_kind,
        effort="quick",
        params={},
        headline={
            "k": 2,
            "stability": {"mean_ari": 0.9, "ci_low": 0.8, "ci_high": 1.0},
            "segments": [{"segment": 0, "name": "high a / low b", "n_clients": 10, "share": 1.0}],
            "bic": {"2": 1.0},
            "silhouette": {"2": 0.5},
        },
        run_id=None,
        results_file="segments/results.json",
    )


def test_promotion_refuses_models_trained_on_real_data(tmp_path: Path) -> None:
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None,
        repo_root=tmp_path / "repo",
        db=tmp_path / "w" / "s.duckdb",
        work_dir=tmp_path / "w",
    )
    path = settings.workspace(Zone.PUBLIC).root / "models" / registry.REGISTRY_FILE
    registry.write(
        path, registry.Registry(zone="public", source_kind="real", models=[_entry("real")])
    )
    with pytest.raises(PrivacyBoundaryError, match="synthetic"):
        registry.promote(settings)
    registry.write(
        path, registry.Registry(zone="public", source_kind="synthetic", models=[_entry()])
    )
    published = registry.promote(settings)
    assert published == settings.repo_root / registry.PUBLIC_REGISTRY
    assert registry.load(published).get("segments").name == "segments"
    with pytest.raises(KeyError):
        registry.load(published).get("silence")


def test_model_cards_render_and_detect_drift(tmp_path: Path) -> None:
    published = registry.Registry(zone="public", source_kind="synthetic", models=[_entry()])
    text = cards.render(published.models[0])
    assert "# Model card" in text
    assert "synthetic data" in text
    assert cards.card_problems(tmp_path, published)[0][1].startswith("missing")
    cards.write_cards(published, tmp_path / cards.CARDS_DIR)
    assert cards.card_problems(tmp_path, published) == []
    (tmp_path / cards.CARDS_DIR / "segments.md").write_text("edited", encoding="utf-8")
    (tmp_path / cards.CARDS_DIR / "stray.md").write_text("x", encoding="utf-8")
    found = dict(cards.card_problems(tmp_path, published))
    assert "drifted" in found["docs/model_cards/segments.md"]
    assert "no model" in found["docs/model_cards/stray.md"]


def test_training_registers_models_and_writes_cards(built_warehouse: Settings) -> None:
    result = train(built_warehouse, Zone.PUBLIC, ["survival", "segments"], quick=True)
    names = [entry.name for entry in result.models]
    assert {"survival", "segments"} <= set(names)
    folder = built_warehouse.workspace(Zone.PUBLIC).root / "models"
    saved = registry.load(folder / registry.REGISTRY_FILE)
    assert saved.source_kind == "synthetic"
    assert all(entry.mlflow_run_id for entry in saved.models)
    assert (folder / "cards" / "segments.md").is_file()
    assert json.loads((folder / "segments" / "results.json").read_text(encoding="utf-8"))["k"] >= 2
    assert (folder / "segments" / "assignments.json").is_file()


def test_cli_train_and_promote_errors(
    built_warehouse: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STRATA_DB", str(built_warehouse.db))
    monkeypatch.setenv("STRATA_WORK_DIR", str(built_warehouse.work_dir))
    result = CliRunner().invoke(app, ["train", "astrology"])
    assert result.exit_code == 2
    result = CliRunner().invoke(app, ["train", "segments", "--quick"])
    assert result.exit_code == 0, result.output
    assert "segments" in result.output
    result = CliRunner().invoke(app, ["--private", "train", "segments", "--promote"])
    assert result.exit_code != 0
