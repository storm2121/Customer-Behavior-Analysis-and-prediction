"""Reports: the public demo is labelled synthetic; the private one stays in the private folder."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from strata.cli import app
from strata.config import PrivacyBoundaryError, Settings, Zone, is_within
from strata.ledger.build import _dump
from strata.ledger.model import Ledger
from strata.report.build import build_report, report_path
from strata.report.figures import figures_for, robustness_figure, share_figure


def public_settings_with_ledger(tmp_path: Path, ledger: Ledger) -> Settings:
    settings = Settings(
        _env_file=None,
        repo_root=tmp_path / "repo",
        work_dir=tmp_path / "work",
        db=tmp_path / "w.duckdb",
    )  # type: ignore[call-arg]
    _dump(ledger, settings.workspace(Zone.PUBLIC).ledger_dir / "claims.synthetic.json")
    return settings


def test_the_public_report_is_the_labelled_synthetic_demo(
    tmp_path: Path, synthetic_ledger: Ledger
) -> None:
    settings = public_settings_with_ledger(tmp_path, synthetic_ledger)
    path = build_report(settings, "public")
    assert path == settings.work_dir / "reports" / "public_report.html"
    html = path.read_text(encoding="utf-8")
    assert "Synthetic data from the public generator" in html
    assert "Allow-listed claims on the real extract" not in html  # no public ledger in this repo
    assert "silence_by_channel" in html
    assert "plotly" in html.lower()


def test_the_public_report_adds_allow_listed_claims_when_present(
    tmp_path: Path, synthetic_ledger: Ledger
) -> None:
    settings = public_settings_with_ledger(tmp_path, synthetic_ledger)
    public = Ledger(
        source_kind="synthetic",
        data_fingerprint=synthetic_ledger.data_fingerprint,
        computed_at=synthetic_ledger.computed_at,
        claims=[record for record in synthetic_ledger.claims if record.public],
    )
    _dump(public, settings.repo_root / "docs" / "ledger" / "claims.public.json")
    html = build_report(settings, "public").read_text(encoding="utf-8")
    assert "Allow-listed claims on the real extract" in html


def test_the_private_report_needs_a_real_ledger_under_the_private_folder(
    tmp_path: Path, synthetic_ledger: Ledger
) -> None:
    settings = Settings(_env_file=None, repo_root=tmp_path / "repo", data_dir=tmp_path / "extract")  # type: ignore[call-arg]
    path = report_path(settings, "private")
    assert is_within(path, tmp_path / "extract" / "private_out")
    _dump(synthetic_ledger, settings.workspace(Zone.PRIVATE).ledger_dir / "claims.private.json")
    with pytest.raises(PrivacyBoundaryError, match="real extract"):
        build_report(settings, "private")
    real = synthetic_ledger.model_copy(update={"source_kind": "real"})
    _dump(real, settings.workspace(Zone.PRIVATE).ledger_dir / "claims.private.json")
    written = build_report(settings, "private")
    assert written == path
    assert "Private: every claim on the real extract" in written.read_text(encoding="utf-8")


def test_the_private_report_cannot_land_in_the_repository(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    settings = Settings(_env_file=None, repo_root=repo, data_dir=tmp_path / "extract")  # type: ignore[call-arg]
    object.__setattr__(settings, "private_dir", repo / "out")
    with pytest.raises(PrivacyBoundaryError):
        report_path(settings, "private")


def test_figures_come_from_records_only(synthetic_ledger: Ledger) -> None:
    mix = synthetic_ledger.get("channel_mix")
    assert share_figure(mix) is not None
    assert robustness_figure(mix) is None
    silence = synthetic_ledger.get("silence_by_channel")
    assert robustness_figure(silence) is not None
    assert len(figures_for(silence)) == 2
    assert share_figure(synthetic_ledger.get("aggregate_join_coverage")) is None


def test_cli_report_variants(
    tmp_path: Path, synthetic_ledger: Ledger, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = public_settings_with_ledger(tmp_path, synthetic_ledger)
    monkeypatch.setenv("STRATA_WORK_DIR", str(settings.work_dir))
    monkeypatch.setenv("STRATA_REPO_ROOT", str(settings.repo_root))
    runner = CliRunner()
    result = runner.invoke(app, ["report", "--variant", "public"])
    assert result.exit_code == 0, result.output
    assert "public_report.html" in result.output
    assert runner.invoke(app, ["report", "--variant", "nonsense"]).exit_code == 2
