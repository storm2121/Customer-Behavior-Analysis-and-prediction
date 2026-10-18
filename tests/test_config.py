"""The configuration is where the path half of the privacy boundary lives."""

from __future__ import annotations

from pathlib import Path

import pytest

from strata.config import (
    ENV_FILES,
    PRIVATE_SUBDIR,
    PrivacyBoundaryError,
    Settings,
    Zone,
    default_env_files,
    find_repo_root,
    get_settings,
    is_within,
)


def make(tmp_path: Path, **overrides: object) -> Settings:
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    return Settings(_env_file=None, repo_root=repo, **overrides)  # type: ignore[call-arg]


def test_relative_paths_resolve_against_the_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("STRATA_DB")
    monkeypatch.delenv("STRATA_WORK_DIR")
    settings = make(tmp_path)
    assert settings.db == settings.repo_root / ".strata" / "strata.duckdb"
    assert settings.work_dir == settings.repo_root / ".strata"


def test_environment_variables_are_read_with_the_strata_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "extract"
    data.mkdir()
    monkeypatch.setenv("STRATA_DATA_DIR", str(data))
    monkeypatch.setenv("STRATA_ZONE", "private")
    monkeypatch.setenv("STRATA_SEED", "7")
    settings = make(tmp_path)
    assert settings.data_dir == data.resolve()
    assert settings.zone is Zone.PRIVATE
    assert settings.seed == 7


def test_private_dir_defaults_to_a_folder_next_to_the_extract(tmp_path: Path) -> None:
    data = tmp_path / "extract"
    data.mkdir()
    settings = make(tmp_path, data_dir=data)
    assert settings.resolved_private_dir == data.resolve() / PRIVATE_SUBDIR


def test_explicit_private_dir_wins(tmp_path: Path) -> None:
    private = tmp_path / "elsewhere"
    settings = make(tmp_path, data_dir=tmp_path / "extract", private_dir=private)
    assert settings.resolved_private_dir == private.resolve()


def test_private_dir_is_required_for_private_runs(tmp_path: Path) -> None:
    settings = make(tmp_path)
    with pytest.raises(PrivacyBoundaryError, match="STRATA_PRIVATE_DIR"):
        _ = settings.resolved_private_dir
    with pytest.raises(PrivacyBoundaryError, match="STRATA_DATA_DIR"):
        settings.require_data_dir()


def test_data_dir_inside_the_repository_is_refused(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    with pytest.raises(PrivacyBoundaryError, match="inside the repository"):
        Settings(_env_file=None, repo_root=repo, data_dir=repo / "data")  # type: ignore[call-arg]


def test_private_dir_inside_the_repository_is_refused(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    with pytest.raises(PrivacyBoundaryError, match="inside the repository"):
        Settings(_env_file=None, repo_root=repo, private_dir=repo / "out")  # type: ignore[call-arg]


def test_dot_dot_cannot_smuggle_a_path_into_the_repository(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "sub").mkdir(parents=True)
    sneaky = repo / "sub" / ".." / "data"
    with pytest.raises(PrivacyBoundaryError):
        Settings(_env_file=None, repo_root=repo, data_dir=sneaky)  # type: ignore[call-arg]


def test_public_workspace_lives_under_the_work_dir(tmp_path: Path) -> None:
    settings = make(tmp_path)
    workspace = settings.workspace(Zone.PUBLIC)
    assert not workspace.is_private
    assert workspace.warehouse == settings.db
    for path in (workspace.ledger_dir, workspace.reports_dir, workspace.figures_dir):
        assert is_within(path, settings.work_dir)


def test_private_workspace_lives_entirely_outside_the_repository(tmp_path: Path) -> None:
    settings = make(tmp_path, data_dir=tmp_path / "extract")
    workspace = settings.workspace(Zone.PRIVATE)
    assert workspace.is_private
    for path in (
        workspace.warehouse,
        workspace.ledger_dir,
        workspace.reports_dir,
        workspace.figures_dir,
        workspace.dbt_target_dir,
        workspace.dbt_log_dir,
    ):
        assert is_within(path, settings.resolved_private_dir)
        assert not is_within(path, settings.repo_root)


def test_workspace_defaults_to_the_configured_zone(tmp_path: Path) -> None:
    settings = make(tmp_path, data_dir=tmp_path / "extract", zone=Zone.PRIVATE)
    assert settings.workspace().is_private


def test_private_outputs_must_stay_under_the_private_dir(tmp_path: Path) -> None:
    settings = make(tmp_path, data_dir=tmp_path / "extract")
    inside = settings.resolved_private_dir / "report.html"
    assert settings.assert_private_output(inside) == inside.resolve()
    with pytest.raises(PrivacyBoundaryError, match="inside the repository"):
        settings.assert_private_output(settings.repo_root / "report.html")
    with pytest.raises(PrivacyBoundaryError, match="under STRATA_PRIVATE_DIR"):
        settings.assert_private_output(tmp_path / "loose.html")


def test_repo_root_is_found_from_the_package() -> None:
    root = find_repo_root()
    assert (root / "pyproject.toml").is_file()
    assert (root / "strata" / "config.py").is_file()


def test_repo_root_falls_back_to_the_current_directory(tmp_path: Path) -> None:
    assert find_repo_root(tmp_path / "nowhere" / "file.py") == Path.cwd().resolve()


def test_get_settings_is_cached() -> None:
    get_settings.cache_clear()
    assert get_settings() is get_settings()
    get_settings.cache_clear()


def test_local_env_files_are_anchored_at_the_repository_root(tmp_path: Path) -> None:
    files = default_env_files(tmp_path)
    assert files == (str(tmp_path / ".env"), str(tmp_path / ".env.local"))
    assert default_env_files() == ENV_FILES


def test_env_local_supplies_private_locations_and_the_environment_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    extract = tmp_path / "extract"
    local = tmp_path / ".env.local"
    local.write_text(f"STRATA_DATA_DIR={extract}\nSTRATA_SEED=11\n", encoding="utf-8")
    repo = tmp_path / "repo"
    repo.mkdir()
    settings = Settings(_env_file=(str(tmp_path / ".env"), str(local)), repo_root=repo)  # type: ignore[call-arg]
    assert settings.data_dir == extract.resolve()
    assert settings.seed == 11
    monkeypatch.setenv("STRATA_SEED", "12")
    assert Settings(_env_file=str(local), repo_root=repo).seed == 12  # type: ignore[call-arg]


def test_the_test_suite_never_reads_the_machine_configuration() -> None:
    """The real repository may hold a `.env.local` naming the extract; tests must not see it."""
    assert Settings().data_dir is None
    assert Settings().private_dir is None
