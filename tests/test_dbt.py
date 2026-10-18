"""The dbt warehouse, built once per session on the default synthetic extract.

dbt's own tests (generic and singular) run inside the build; these pytest checks restate the
milestone's definition of done from the outside: the build is green, the marts hold the rows the
manifest says they should, features never read past their cutoff, and a private build writes its
artefacts only under the private folder.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path

import duckdb
import pytest
from typer.testing import CliRunner

from strata.cli import app
from strata.config import SYNTHETIC_MARKER, Settings, Zone, is_within
from strata.contracts.vocab import silence_cutoffs
from strata.dbt_runner import dbt_arguments, run_dbt
from strata.ingest.load import load_manifest
from strata.warehouse import connect
from tests.conftest import build_warehouse, public_settings

MARTS = (
    "client_360",
    "features_silence",
    "features_segments",
    "features_anomaly",
    "daily_counts",
    "aggregate_clients",
)


@pytest.fixture
def con(built_warehouse: Settings) -> Iterator[duckdb.DuckDBPyConnection]:
    connection = connect(built_warehouse.workspace(Zone.PUBLIC).warehouse, read_only=True)
    yield connection
    connection.close()


def scalar(con: duckdb.DuckDBPyConnection, sql: str) -> int:
    row = con.execute(sql).fetchone()
    assert row is not None
    return int(row[0])


def test_every_mart_is_built(con: duckdb.DuckDBPyConnection) -> None:
    for mart in MARTS:
        assert scalar(con, f"select count(*) from marts.{mart}") > 0  # noqa: S608


def test_marts_row_counts_match_the_manifest(
    built_warehouse: Settings, con: duckdb.DuckDBPyConnection
) -> None:
    manifest = load_manifest(built_warehouse.workspace(Zone.PUBLIC))
    rows = {item.extract: item.rows for item in manifest.files}
    assert scalar(con, "select count(*) from marts.client_360") == rows["clients"]
    events = rows["card_transactions"] + rows["online_operations"] + rows["branch_operations"]
    assert scalar(con, "select count(*) from intermediate.int_events") == events
    non_empty = scalar(
        con, "select count(*) from raw.aggregate_clients where identifier is not null"
    )
    assert scalar(con, "select count(*) from marts.aggregate_clients") == non_empty


def test_silence_features_cover_every_cutoff_without_leakage(
    con: duckdb.DuckDBPyConnection,
) -> None:
    cutoffs = {
        row[0]
        for row in con.execute("select distinct cutoff_date from marts.features_silence").fetchall()
    }
    assert cutoffs == {cutoff.cutoff for cutoff in silence_cutoffs()}
    assert (
        scalar(
            con, "select count(*) from marts.features_silence where last_event_date > cutoff_date"
        )
        == 0
    )
    assert (
        scalar(
            con, "select count(*) from marts.features_silence where outcome_start <= cutoff_date"
        )
        == 0
    )


def test_channel_states_partition_the_clients_every_month(con: duckdb.DuckDBPyConnection) -> None:
    clients = scalar(con, "select count(*) from staging.stg_clients")
    per_month = con.execute(
        "select month_start, count(*) from intermediate.int_channel_state group by 1"
    ).fetchall()
    assert len(per_month) == 13
    assert {count for _, count in per_month} == {clients}


def test_daily_counts_are_dense(con: duckdb.DuckDBPyConnection) -> None:
    days = scalar(con, "select count(distinct day) from marts.daily_counts")
    series = scalar(con, "select count(distinct channel || '/' || kind) from marts.daily_counts")
    assert scalar(con, "select count(*) from marts.daily_counts") == days * series
    assert scalar(con, "select count(distinct day) from marts.daily_counts where is_ramadan") == 30


def test_repeat_flags_agree_with_the_quality_report(con: duckdb.DuckDBPyConnection) -> None:
    flagged = scalar(con, "select count(*) from staging.stg_card_transactions where is_repeat")
    reported = scalar(
        con,
        "select \"count\" from raw.dq_report where check_id = 'exact_repeats' "
        "and extract = 'card_transactions'",
    )
    assert flagged == reported > 0


def test_documentation_is_generated(built_warehouse: Settings) -> None:
    assert run_dbt(built_warehouse, Zone.PUBLIC, ["docs", "generate"]).success
    target = built_warehouse.workspace(Zone.PUBLIC).dbt_target_dir
    assert (target / "catalog.json").is_file()
    assert (target / "manifest.json").is_file()


def test_dbt_arguments_point_at_the_zone(built_warehouse: Settings) -> None:
    workspace = built_warehouse.workspace(Zone.PUBLIC)
    arguments = dbt_arguments(built_warehouse, workspace, ["run"])
    assert arguments[0] == "run"
    assert str(workspace.dbt_target_dir) in arguments
    assert str(workspace.dbt_log_dir) in arguments


def test_run_dbt_needs_a_command_and_a_warehouse(tmp_path: Path) -> None:
    settings = public_settings(tmp_path)
    with pytest.raises(ValueError, match="dbt command"):
        run_dbt(settings, Zone.PUBLIC, [])
    with pytest.raises(FileNotFoundError, match="strata ingest"):
        run_dbt(settings, Zone.PUBLIC, ["run"])


def test_cli_dbt_requires_a_command() -> None:
    assert CliRunner().invoke(app, ["dbt"]).exit_code == 2


def test_cli_dbt_reports_failure(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["dbt", "run"])
    assert result.exit_code != 0


def test_a_private_build_keeps_every_artefact_outside_the_repository(
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
    workspace = settings.workspace(Zone.PRIVATE)
    assert is_within(workspace.dbt_target_dir, extract / "private_out")
    assert (workspace.dbt_target_dir / "run_results.json").is_file()
    created = {path.relative_to(repo).parts[:2] for path in repo.rglob("*") if path.is_file()}
    assert all(parts[0] == "dbt" and parts[1] != "target" for parts in created)
