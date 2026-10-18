"""Data-quality checks: on the synthetic warehouse, and on hand-built frames for each rule."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from kashf.cli import app
from kashf.config import Settings, Zone
from kashf.contracts.specs import EXTRACTS, Extract
from kashf.ingest import dq
from kashf.synthetic.parametric import SyntheticExtract
from kashf.warehouse import connect


def by_id(results: list[dq.DQResult]) -> dict[tuple[str, str], dq.DQResult]:
    return {(r.check_id, r.extract): r for r in results}


def test_the_synthetic_warehouse_has_no_errors_and_records_the_quirks(
    public_warehouse: Settings,
) -> None:
    con = connect(public_warehouse.workspace(Zone.PUBLIC).warehouse)
    try:
        results = dq.run(con)
        stored = con.table("raw.dq_report").df()
    finally:
        con.close()
    assert not dq.has_errors(results)
    found = by_id(results)
    assert found[("exact_repeats", "card_transactions")].count > 0
    assert found[("zero_amount", "online_operations")].count > 0
    assert found[("sub_one_mad", "card_transactions")].count > 0
    coverage = found[("aggregate_join_coverage", "aggregate_clients")]
    assert coverage.count == 0
    assert coverage.status == "warning"
    assert found[("zero_amount_outside_card_service", "online_operations")].status == "ok"
    assert len(stored) == len(results)


def test_reports_carry_no_identifier(public_warehouse: Settings) -> None:
    con = connect(public_warehouse.workspace(Zone.PUBLIC).warehouse)
    try:
        report = dq.run(con)
    finally:
        con.close()
    text = " ".join(f"{r.check_id} {r.extract} {r.description}" for r in report)
    assert "syn_" not in text


def card_frame(rows: list[tuple[str, str, float | None, str | None, str]]) -> pd.DataFrame:
    frame = pd.DataFrame(
        rows,
        columns=[
            "identifiant_client",
            "date_transaction",
            "montant_local",
            "libelle_nature_transaction",
            "type_origine",
        ],
    )
    frame["date_transaction"] = pd.to_datetime(frame["date_transaction"], format="ISO8601")
    return frame


def test_exact_repeats_count_second_and_later_copies_only() -> None:
    card = card_frame(
        [
            ("a", "2024-01-01", 10.0, "WITHDRAWAL", "National"),
            ("a", "2024-01-01", 10.0, "WITHDRAWAL", "National"),
            ("a", "2024-01-01", 10.0, "WITHDRAWAL", "National"),
            ("a", "2024-01-01", 10.0, "WITHDRAWAL", "International"),
        ]
    )
    result = dq.exact_repeats({Extract.CARD: card})[0]
    assert (result.count, result.denominator, result.status) == (2, 4, "warning")


def test_time_of_day_and_window_checks() -> None:
    card = card_frame(
        [
            ("a", "2024-01-01 13:00", 1.0, "WITHDRAWAL", "National"),
            ("a", "2022-01-01", 1.0, "WITHDRAWAL", "National"),
        ]
    )
    assert dq.time_of_day({Extract.CARD: card})[0].count == 1
    assert dq.outside_window({Extract.CARD: card})[0].count == 1


def test_amount_checks() -> None:
    card = card_frame(
        [
            ("a", "2024-01-01", 0.5, "PURCHASES & SERVICES", "National"),
            ("a", "2024-01-02", 150.0, "WITHDRAWAL", "National"),
            ("a", "2024-01-03", 200.0, "WITHDRAWAL", "National"),
        ]
    )
    online = pd.DataFrame(
        {
            "identifiant_client": ["a", "a"],
            "service": ["Recharge Binatna", "Opérations sur carte"],
            "operation_date_time": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "montant_transaction": [0.0, 0.0],
        }
    )
    found = by_id(dq.amounts({Extract.CARD: card, Extract.ONLINE: online}))
    assert found[("sub_one_mad", "card_transactions")].count == 1
    assert found[("withdrawal_not_multiple_of_100", "card_transactions")].count == 1
    assert found[("zero_amount_outside_card_service", "online_operations")].count == 1


def test_vocabulary_and_direction_checks() -> None:
    branch = pd.DataFrame(
        {
            "identifiant_client": ["a", "a", "a"],
            "date_operation": pd.to_datetime(["2024-01-01"] * 3),
            "montant_operation": [1.0, 2.0, 3.0],
            "sens_operation": ["D", "C", "X"],
            "operation": ["VERSEMENT SUR COMPTE", "NOT A LABEL", "RETRAIT CHEQUE"],
            "devise": ["MAD", "EUR", "MAD"],
        }
    )
    found = by_id(dq.vocabularies({Extract.BRANCH: branch}))
    assert found[("unknown_label", "branch_operations")].count == 1
    assert found[("unknown_direction", "branch_operations")].count == 1
    assert found[("currency_not_mad", "branch_operations")].count == 1
    assert found[("direction_contradicts_label", "branch_operations")].count == 1


def test_join_and_plausibility_checks(synthetic_extract: SyntheticExtract) -> None:
    frames = {name: frame.copy() for name, frame in synthetic_extract.frames.items()}
    frames[Extract.CARD].loc[frames[Extract.CARD].index[0], "identifiant_client"] = "syn_orphan"
    frames[Extract.AGGREGATE].loc[0, "identifier"] = frames[Extract.CLIENTS][
        "identifiant_client"
    ].iloc[0]
    clients = frames[Extract.CLIENTS]
    clients.loc[clients.index[0], "age"] = 12
    clients.loc[clients.index[1], "profession"] = pd.NA
    found = by_id(dq.run_checks(frames))
    assert found[("orphan_events", "card_transactions")].count == 1
    assert found[("aggregate_join_coverage", "aggregate_clients")].status == "info"
    assert found[("age_out_of_range", "clients")].count >= 1
    assert found[("profession_block_partial", "clients")].count >= 1


def test_aggregate_semantics_contradictions_are_counted(
    synthetic_extract: SyntheticExtract,
) -> None:
    aggregate = synthetic_extract.frames[Extract.AGGREGATE].copy()
    offline = aggregate.index[aggregate["client_has_cih_online"] == 2][0]
    aggregate.loc[offline, "nombre_operations"] = 5
    found = by_id(dq.aggregate_file({Extract.AGGREGATE: aggregate}))
    assert found[("online_flag_contradicts_operations", "aggregate_clients")].count == 1
    assert found[("constant_columns", "aggregate_clients")].count >= 1


def test_an_empty_file_is_an_error() -> None:
    empty = {Extract.CARD: card_frame([])}
    assert dq.row_counts(empty)[0].status == "error"
    assert dq.joins(empty) == []


def test_a_warehouse_without_a_manifest_is_an_error(tmp_path: Path) -> None:
    con = connect(tmp_path / "empty.duckdb")
    try:
        results = dq.reconciliation_results(con)
    finally:
        con.close()
    assert dq.has_errors(results)


def test_cli_pipeline_synth_ingest_dq(tmp_path: Path) -> None:
    runner = CliRunner()
    out = tmp_path / "synth"
    assert (
        runner.invoke(
            app, ["synth", "--out", str(out), "--clients", "120", "--seed", "3"]
        ).exit_code
        == 0
    )
    ingested = runner.invoke(app, ["ingest", "--data-dir", str(out)])
    assert ingested.exit_code == 0, ingested.output
    assert "reconciliation" in ingested.output
    checked = runner.invoke(app, ["dq"])
    assert checked.exit_code == 0, checked.output
    assert "exact_repeats" in checked.output
    assert all(
        spec.filename in runner.invoke(app, ["synth", "--out", str(out)]).output
        for spec in EXTRACTS[:1]
    )


def test_cli_refuses_real_looking_data_in_the_public_zone(
    tmp_path: Path, synthetic_dir: Path
) -> None:
    import shutil

    unmarked = tmp_path / "unmarked"
    shutil.copytree(synthetic_dir, unmarked)
    (unmarked / "SYNTHETIC.json").unlink()
    result = CliRunner().invoke(app, ["ingest", "--data-dir", str(unmarked)])
    assert result.exit_code != 0


def test_cli_dq_without_a_warehouse(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["dq"])
    assert result.exit_code == 2


@pytest.mark.private
def test_private_pipeline_runs_against_the_real_extract() -> None:
    """Local only: proves the private commands run; asserts on structure, never on values."""
    runner = CliRunner()
    assert runner.invoke(app, ["--private", "ingest"]).exit_code == 0
    assert runner.invoke(app, ["--private", "dq"]).exit_code == 0
