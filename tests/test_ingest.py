"""Ingest: zone rules, the raw layer, the manifest and reconciliation."""

from __future__ import annotations

import shutil
from pathlib import Path

import duckdb
import pytest

from kashf.config import SYNTHETIC_MARKER, PrivacyBoundaryError, Settings, Zone, is_within
from kashf.contracts.specs import CARD, EXTRACTS
from kashf.ingest.load import Reconciliation, ingest, load_manifest, resolve_source
from kashf.warehouse import connect, read_table, replace_table, row_count, table_exists
from tests.conftest import public_settings


def copy_without_marker(source: Path, destination: Path) -> Path:
    """Synthetic files with their marker removed: indistinguishable from a real extract."""
    shutil.copytree(source, destination)
    (destination / SYNTHETIC_MARKER).unlink()
    return destination


def test_public_runs_need_a_folder(tmp_path: Path) -> None:
    with pytest.raises(PrivacyBoundaryError, match="--data-dir"):
        resolve_source(public_settings(tmp_path), Zone.PUBLIC, None)


def test_public_runs_refuse_unmarked_data(tmp_path: Path, synthetic_dir: Path) -> None:
    unmarked = copy_without_marker(synthetic_dir, tmp_path / "looks_real")
    with pytest.raises(PrivacyBoundaryError, match="--private"):
        resolve_source(public_settings(tmp_path), Zone.PUBLIC, unmarked)


def test_public_runs_refuse_tampered_synthetic_folders(tmp_path: Path, synthetic_dir: Path) -> None:
    tampered = tmp_path / "tampered"
    shutil.copytree(synthetic_dir, tampered)
    with (tampered / CARD.filename).open("ab") as handle:
        handle.write(b"\r\n")
    with pytest.raises(PrivacyBoundaryError, match="no longer match"):
        resolve_source(public_settings(tmp_path), Zone.PUBLIC, tampered)


def test_private_runs_need_the_extract_location(tmp_path: Path) -> None:
    with pytest.raises(PrivacyBoundaryError, match="KASHF_DATA_DIR"):
        resolve_source(public_settings(tmp_path), Zone.PRIVATE, None)


def test_private_runs_refuse_unmarked_data_inside_the_repository(
    tmp_path: Path, synthetic_dir: Path
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    inside = copy_without_marker(synthetic_dir, repo / "data")
    settings = Settings(_env_file=None, repo_root=repo)  # type: ignore[call-arg]
    with pytest.raises(PrivacyBoundaryError, match="inside the repository"):
        resolve_source(settings, Zone.PRIVATE, inside)


def test_public_ingest_builds_the_raw_layer(public_warehouse: Settings) -> None:
    workspace = public_warehouse.workspace(Zone.PUBLIC)
    con = connect(workspace.warehouse, read_only=True)
    try:
        for spec in EXTRACTS:
            assert table_exists(con, "raw", spec.name.value)
            stored = read_table(con, "raw", spec.name.value)
            assert list(stored.columns) == [*spec.columns, "_source_row"]
        manifest = read_table(con, "raw", "manifest")
        assert set(manifest["source_kind"]) == {"synthetic"}
        assert manifest["fingerprint"].nunique() == 1
    finally:
        con.close()
    assert load_manifest(workspace).source_kind == "synthetic"


def test_fingerprint_is_stable_and_sensitive(tmp_path: Path, synthetic_dir: Path) -> None:
    settings = public_settings(tmp_path / "w")
    first = ingest(settings, Zone.PUBLIC, synthetic_dir)
    second = ingest(settings, Zone.PUBLIC, synthetic_dir)
    assert first.ok
    assert second.ok
    assert first.manifest.fingerprint == second.manifest.fingerprint
    assert first.manifest.loaded_at <= second.manifest.loaded_at
    assert first.manifest.file("card_transactions").line_terminator == "CRLF"


def test_private_ingest_writes_only_under_the_private_folder(
    tmp_path: Path, synthetic_dir: Path
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    extract = copy_without_marker(synthetic_dir, tmp_path / "extract")
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None, repo_root=repo, data_dir=extract, db=repo / ".kashf" / "kashf.duckdb"
    )
    report = ingest(settings, Zone.PRIVATE)
    assert report.ok
    assert report.manifest.source_kind == "real"
    assert is_within(report.workspace.warehouse, extract / "private_out")
    assert (extract / "private_out" / "manifest.json").is_file()
    assert not any(repo.rglob("*"))


def test_a_private_run_may_read_a_marked_synthetic_folder(
    tmp_path: Path, synthetic_dir: Path
) -> None:
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None, repo_root=tmp_path / "repo", private_dir=tmp_path / "private"
    )
    source = resolve_source(settings, Zone.PRIVATE, synthetic_dir)
    assert source.kind == "synthetic"


def test_load_manifest_explains_a_missing_ingest(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="kashf ingest"):
        load_manifest(public_settings(tmp_path).workspace(Zone.PUBLIC))


def test_reconciliation_tolerates_rounding_but_not_loss() -> None:
    assert Reconciliation("x", "sum", 1.0, 1.0 + 1e-12).ok
    assert not Reconciliation("x", "rows", 10.0, 9.0).ok


def test_warehouse_helpers_refuse_unsafe_identifiers(tmp_path: Path) -> None:
    con = connect(tmp_path / "w.duckdb")
    try:
        with pytest.raises(ValueError, match="identifier"):
            replace_table(con, "raw", "x; drop table y", duckdb.sql("select 1").df())
        replace_table(con, "raw", "small", duckdb.sql("select 1 as a").df())
        assert row_count(con, "raw", "small") == 1
        assert not table_exists(con, "raw", "absent")
    finally:
        con.close()
