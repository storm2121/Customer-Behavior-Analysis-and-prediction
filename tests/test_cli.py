"""CLI behaviour: help, version, the info screen, and sanitised failure reporting."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from kashf import __version__, cli
from kashf.config import PrivacyBoundaryError

FAKE_ID = "0f" * 20
runner = CliRunner()


def test_help_lists_the_commands() -> None:
    result = runner.invoke(cli.app, ["--help"])
    assert result.exit_code == 0
    assert "privacy-check" in result.output
    assert "info" in result.output


def test_version_flag() -> None:
    result = runner.invoke(cli.app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_info_in_the_public_zone_shows_repository_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KASHF_DB")  # `info` writes nothing, so the defaults are safe here
    monkeypatch.delenv("KASHF_WORK_DIR")
    result = runner.invoke(cli.app, ["info"])
    assert result.exit_code == 0
    assert "public" in result.output
    assert ".kashf/kashf.duckdb" in result.output


def test_info_in_the_private_zone_never_spells_out_private_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    extract = tmp_path / "VerySecretFolder"
    extract.mkdir()
    monkeypatch.setenv("KASHF_DATA_DIR", str(extract))
    result = runner.invoke(cli.app, ["--private", "info"])
    assert result.exit_code == 0
    assert "VerySecretFolder" not in result.output
    assert "<KASHF_PRIVATE_DIR>/warehouse/kashf.duckdb" in result.output


def test_info_without_a_private_folder_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    result = runner.invoke(cli.app, ["--zone", "private", "info"])
    assert result.exit_code == 0
    assert "not configured" in result.output


def test_describe_marks_paths_outside_the_private_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KASHF_DATA_DIR", str(tmp_path / "extract"))
    state = cli.AppState(settings=cli.Settings(), zone=cli.Zone.PRIVATE)
    assert cli._describe(tmp_path / "elsewhere", state) == "<outside KASHF_PRIVATE_DIR>"
    public = cli.AppState(settings=cli.Settings(), zone=cli.Zone.PUBLIC)
    assert cli._describe(tmp_path / "x", public) == (tmp_path / "x").as_posix()


def test_a_data_dir_inside_the_repository_stops_the_cli_with_code_three(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = cli.Settings().repo_root
    monkeypatch.setenv("KASHF_DATA_DIR", str(repo / "data"))
    monkeypatch.setattr(sys, "argv", ["kashf", "info"])
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    assert exit_info.value.code == 3


def test_unexpected_errors_are_reported_without_identifiers(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def explode() -> None:
        msg = f"bad row for {FAKE_ID}"
        raise ValueError(msg)

    monkeypatch.setattr(cli, "app", explode)
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    assert exit_info.value.code == 1
    err = capsys.readouterr().err
    assert "ValueError" in err
    assert FAKE_ID not in err


def test_private_failures_keep_the_traceback_in_the_private_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("KASHF_DATA_DIR", str(tmp_path / "extract"))
    monkeypatch.setattr(sys, "argv", ["kashf", "--private", "info"])

    def explode() -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(cli, "app", explode)
    with pytest.raises(SystemExit):
        cli.main()
    logs = list((tmp_path / "extract" / "private_out" / "logs").glob("kashf-error-*.log"))
    assert len(logs) == 1
    assert "RuntimeError" in logs[0].read_text(encoding="utf-8")
    assert "<KASHF_PRIVATE_DIR>/logs/" in capsys.readouterr().err


def test_public_failures_write_no_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KASHF_DATA_DIR", str(tmp_path / "extract"))
    monkeypatch.setattr(sys, "argv", ["kashf", "info"])

    def explode() -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(cli, "app", explode)
    with pytest.raises(SystemExit):
        cli.main()
    assert not (tmp_path / "extract" / "private_out").exists()


def test_report_failure_survives_an_unconfigured_environment(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cli._report_failure(RuntimeError("no private folder configured"))
    assert "RuntimeError" in capsys.readouterr().err


def test_boundary_errors_raised_by_commands_exit_with_code_three(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse() -> None:
        raise PrivacyBoundaryError("nope")

    monkeypatch.setattr(cli, "app", refuse)
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    assert exit_info.value.code == 3


def test_typer_exits_pass_through(monkeypatch: pytest.MonkeyPatch) -> None:
    def leave() -> None:
        raise SystemExit(0)

    monkeypatch.setattr(cli, "app", leave)
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    assert exit_info.value.code == 0
    assert isinstance(typer.Exit(), Exception)
