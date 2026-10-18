"""The monitoring layer: rules on hand-built cases, detectors, the queue, and the review store."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from strata.cli import app
from strata.config import Settings, Zone
from strata.models.anomaly import rules as engine
from strata.models.anomaly import scores
from strata.models.anomaly.run import run
from strata.review import db

if TYPE_CHECKING:
    from strata.synthetic.parametric import SyntheticExtract

RULES = engine.load_rules()


Rows = list[tuple[str, str, float]]


def _inputs(
    deposits: Rows | None = None,
    withdrawals: Rows | None = None,
    months: pd.DataFrame | None = None,
    card_spend: dict[str, float] | None = None,
) -> engine.Inputs:
    def frame(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
        out = pd.DataFrame(list(rows), columns=["client_id", "event_date", "amount"])
        out["event_date"] = pd.to_datetime(out["event_date"])
        return out.sort_values(["client_id", "event_date"]).reset_index(drop=True)

    return engine.Inputs(
        deposits=frame(deposits or []),
        withdrawals=frame(withdrawals or []),
        months=months
        if months is not None
        else pd.DataFrame(columns=["client_id", "month_start", "n_events", "gap_months_before"]),
        card_spend=pd.Series(card_spend or {}, dtype=float),
    )


def _clients(rule_id: str, data: engine.Inputs) -> list[str]:
    rule = RULES.get(rule_id)
    return [hit.client_id for hit in engine.ENGINES[rule.kind](rule, data)]


def test_rule_thresholds_are_the_dbt_variables() -> None:
    project = yaml.safe_load((Settings().repo_root / "dbt" / "dbt_project.yml").read_text("utf-8"))
    variables = project["vars"]
    assert (
        RULES.get("large_cash_deposit").params["threshold"] == variables["large_deposit_threshold"]
    )
    structuring = RULES.get("structuring").params
    assert structuring["threshold"] == variables["large_deposit_threshold"]
    assert structuring["window_days"] == variables["velocity_days"]
    assert RULES.get("deposit_velocity").params["window_days"] == variables["velocity_days"]
    passing = RULES.get("pass_through").params
    assert passing["min_deposit"] == variables["pass_through_min_deposit"]
    assert passing["window_days"] == variables["pass_through_days"]
    cash_only = RULES.get("cash_only_high_volume").params
    assert cash_only["min_deposit_amount"] == variables["large_deposit_threshold"]
    burst = RULES.get("dormant_then_burst").params
    assert burst["min_gap_months"] == variables["burst_min_gap_months"]
    assert burst["min_events"] == variables["burst_min_events"]


def test_structuring_needs_split_deposits_within_the_window() -> None:
    split = [("a", "2024-01-01", 40_000), ("a", "2024-01-10", 35_000), ("a", "2024-01-20", 30_000)]
    assert _clients("structuring", _inputs(split)) == ["a"]
    spread = [("b", "2024-01-01", 40_000), ("b", "2024-02-15", 35_000), ("b", "2024-03-30", 30_000)]
    assert _clients("structuring", _inputs(spread)) == []
    small = [("c", "2024-01-01", 30_000), ("c", "2024-01-05", 30_000), ("c", "2024-01-09", 30_000)]
    assert _clients("structuring", _inputs(small)) == []
    # A deposit at the threshold is a large deposit, not part of a split.
    large = [("d", "2024-01-01", 100_000), ("d", "2024-01-02", 10_000), ("d", "2024-01-03", 5_000)]
    assert _clients("structuring", _inputs(large)) == []
    hits = engine.structuring(RULES.get("structuring"), _inputs(split))
    assert hits[0].evidence["total"] == 105_000
    assert hits[0].evidence["deposits"] == 3


def test_pass_through_needs_a_quick_withdrawal_of_most_of_a_deposit() -> None:
    deposit = [("a", "2024-03-01", 100_000)]
    assert _clients("pass_through", _inputs(deposit, [("a", "2024-03-04", 95_000)])) == ["a"]
    assert _clients("pass_through", _inputs(deposit, [("a", "2024-03-11", 95_000)])) == []
    assert _clients("pass_through", _inputs(deposit, [("a", "2024-03-04", 10_000)])) == []
    assert _clients("pass_through", _inputs(deposit, [("a", "2024-03-01", 95_000)])) == []
    small = [("b", "2024-03-01", 10_000)]
    assert _clients("pass_through", _inputs(small, [("b", "2024-03-02", 9_000)])) == []


def test_velocity_counts_deposits_within_the_window() -> None:
    eight = [("a", f"2024-04-{day:02d}", 1_000.0) for day in range(1, 25, 3)]
    assert len(eight) == 8
    assert _clients("deposit_velocity", _inputs(eight)) == ["a"]
    assert _clients("deposit_velocity", _inputs(eight[:7])) == []
    spread = [("b", f"2024-{month:02d}-01", 1_000.0) for month in range(1, 9)]
    assert _clients("deposit_velocity", _inputs(spread)) == []


def test_large_deposits_dormancy_and_cash_intensity() -> None:
    assert _clients("large_cash_deposit", _inputs([("a", "2024-01-01", 100_000)])) == ["a"]
    assert _clients("large_cash_deposit", _inputs([("a", "2024-01-01", 99_999)])) == []
    months = pd.DataFrame(
        {
            "client_id": ["a", "b"],
            "month_start": pd.to_datetime(["2024-05-01", "2024-05-01"]),
            "n_events": [6, 9],
            "gap_months_before": [3, 2],
        }
    )
    assert _clients("dormant_then_burst", _inputs(months=months)) == ["a"]
    deposits = [
        ("a", "2024-01-01", 60_000),
        ("b", "2024-01-01", 60_000),
        ("c", "2024-01-01", 40_000),
        ("d", "2024-01-01", 60_000),
        ("e", "2024-01-01", 120_000),
    ]
    spend = {"a": 10_000.0, "b": 20_000.0, "c": 0.0, "d": 0.0, "e": 0.0}
    # A client without card spend is never cash-intensive; large sums make it cash-only instead.
    assert _clients("cash_intensive", _inputs(deposits, card_spend=spend)) == ["a"]
    assert _clients("cash_only_high_volume", _inputs(deposits, card_spend=spend)) == ["e"]


def test_pseudo_labels_need_two_families() -> None:
    hits = [
        engine.Hit("structuring", "splitting", "a"),
        engine.Hit("pass_through", "flow", "a"),
        engine.Hit("large_cash_deposit", "amount", "b"),
        engine.Hit("large_cash_deposit", "amount", "b"),
    ]
    assert engine.pseudo_labels(hits) == {"a"}


def test_the_amount_model_flags_unusual_amounts_not_usual_large_ones() -> None:
    rng = np.random.default_rng(0)
    # Client 1 always moves large sums; client 0 has one amount far above its usual level.
    rows = [
        (f"c{client}", day, float((50_000.0 if client == 1 else 200.0) * rng.lognormal(0, 0.2)))
        for client in range(60)
        for day in range(20)
    ]
    rows.append(("c0", 30, 40_000.0))
    events = pd.DataFrame(rows, columns=["client_id", "day", "amount"])
    events["event_date"] = pd.Timestamp("2024-01-01") + pd.to_timedelta(events["day"], unit="D")
    events["channel"], events["kind_group"], events["is_repeat"] = "branch", "deposit", False
    flags = scores.amount_scores(events)
    assert flags["c0"].z > 3
    assert flags["c0"].amount == 40_000.0
    assert flags["c1"].z < flags["c0"].z


def test_combine_ranks_by_score_and_explains() -> None:
    queue_input = pd.Series(["x", "y", "z"])
    hits = [engine.Hit("structuring", "splitting", "y", {"total": 1.0})]
    forest = pd.Series({"x": 0.1, "y": 0.2, "z": 0.3})
    queue = scores.combine(
        queue_input, hits=hits, rules=RULES, forest=forest, amounts={}, digits={}
    )
    assert queue.alerts["client_id"].tolist()[0] == "y"
    assert queue.alerts["rank"].tolist() == [1, 2, 3]
    assert queue.alerts.loc[0, "reasons"][0]["rule"] == "structuring"
    assert scores.precision_at(queue.alerts["client_id"], {"y"}, 1) == 1.0
    assert len(queue.top(2)) == 2


def test_the_queue_finds_the_planted_scenarios(
    built_warehouse: Settings, synthetic_extract: SyntheticExtract
) -> None:
    result = run(built_warehouse, Zone.PUBLIC, budget=50)
    truth = synthetic_extract.truth
    planted = set(truth.loc[truth["scenario"] != "none", "identifiant_client"])
    found = result.top["client_id"].isin(planted).sum()
    assert found / len(planted) >= 0.8
    precision = result.evaluation["precision_at_k"]["10"]
    assert precision["combined"] > result.evaluation["base_rate"]
    assert precision["statistical_only"] > result.evaluation["base_rate"]
    assert result.run_id >= 1
    assert result.queue_file.is_file()


def test_feedback_round_trips_through_the_store(tmp_path: Path) -> None:
    url = f"sqlite:///{(tmp_path / 'review.sqlite').as_posix()}"
    store = db.engine(url)
    alerts = [
        {"client_id": "a", "rank": 1, "score": 0.9, "rules": ["structuring"], "reasons": []},
        {"client_id": "b", "rank": 2, "score": 0.8, "rules": [], "reasons": []},
    ]
    run_id = db.store_run(
        store,
        zone="public",
        source_kind="synthetic",
        data_fingerprint="f" * 64,
        rules_digest="r" * 64,
        n_clients=2,
        alerts=alerts,
        budget=2,
    )
    assert db.latest_run(store) == run_id
    first, second = db.alerts_of(store, run_id)
    db.add_feedback(store, first["id"], "dismissed", "ana")
    db.add_feedback(store, first["id"], "confirmed", "ana", note="second look")
    db.add_feedback(store, second["id"], "dismissed", "bo")
    db.annotate(store, second["id"], "bo", "seasonal business")
    rows = db.alerts_of(store, run_id)
    assert [row["label"] for row in rows] == ["confirmed", "dismissed"]
    assert all(row["status"] == "reviewed" for row in rows)
    assert db.precision_from_feedback(store, run_id, 2)["precision"] == 0.5
    with pytest.raises(ValueError, match="label must be"):
        db.add_feedback(store, first["id"], "guilty", "ana")
    with pytest.raises(KeyError):
        db.add_feedback(store, 999, "confirmed", "ana")
    with pytest.raises(KeyError):
        db.annotate(store, 999, "ana", "x")
    store.dispose()
    db.downgrade(url)
    db.migrate(url)
    assert db.latest_run(db.engine(url)) is None


def test_private_alerts_never_leave_the_private_folder(tmp_path: Path) -> None:
    extract = tmp_path / "extract"
    extract.mkdir()
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None,
        data_dir=extract,
        pg_url="postgresql+psycopg://user:secret@db:5432/strata",
    )
    url = db.database_url(settings, Zone.PRIVATE)
    assert url.startswith("sqlite:///")
    assert (extract / "private_out").as_posix() in url
    assert db.database_url(settings, Zone.PUBLIC).startswith("postgresql")


def test_cli_anomaly_commands(built_warehouse: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STRATA_DB", str(built_warehouse.db))
    monkeypatch.setenv("STRATA_WORK_DIR", str(built_warehouse.work_dir))
    runner = CliRunner()
    result = runner.invoke(app, ["anomaly", "run", "--budget", "20"])
    assert result.exit_code == 0, result.output
    assert "precision at k" in result.output
    queue = runner.invoke(app, ["anomaly", "queue", "--limit", "3"])
    assert queue.exit_code == 0, queue.output
    store = db.engine(db.database_url(Settings(), Zone.PUBLIC))
    latest = db.latest_run(store)
    assert latest is not None
    alert_id = db.alerts_of(store, latest, limit=1)[0]["id"]
    store.dispose()
    feedback = runner.invoke(app, ["anomaly", "feedback", str(alert_id), "confirmed"])
    assert feedback.exit_code == 0, feedback.output
    assert "confirmed" in feedback.output
    bad = runner.invoke(app, ["anomaly", "feedback", str(alert_id), "guilty"])
    assert bad.exit_code != 0


def test_the_migration_renders_for_postgresql(capsys: pytest.CaptureFixture[str]) -> None:
    from alembic import command
    from alembic.config import Config

    config = Config()
    config.set_main_option("script_location", db.MIGRATIONS.as_posix())
    config.set_main_option("sqlalchemy.url", "postgresql+psycopg://user@localhost:5432/strata")
    command.upgrade(config, "head", sql=True)
    sql = capsys.readouterr().out
    for table in ("runs", "alerts", "feedback", "annotations"):
        assert f"CREATE TABLE {table}" in sql
    assert "TIMESTAMP WITH TIME ZONE" in sql
