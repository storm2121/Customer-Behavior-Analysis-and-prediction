"""Shared fixtures.

Every test runs on synthetic data, in an environment scrubbed of ``KASHF_*`` variables and from a
temporary working directory, so a developer's real configuration (or a stray ``.env``) can never
leak into a test run. Tests marked ``private`` are the only exception and are skipped unless
``--run-private`` is given on a machine where ``KASHF_DATA_DIR`` is set.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from kashf.config import Settings, Zone

if TYPE_CHECKING:
    from kashf.ledger.model import Ledger
    from kashf.synthetic.parametric import SyntheticExtract

# Fixtures import the pipeline lazily: collecting the suite stays fast, and a test module that
# needs only the guard or the settings never imports pandas, DuckDB or dbt.


def pytest_configure(config: pytest.Config) -> None:
    # pytest's shared temp root (``pytest-of-<user>``) can be left unreadable by another process on
    # Windows; a project-specific root outside the repository avoids it. Temp folders must stay
    # outside the repository because several tests assert that private paths resolve outside it.
    del config
    os.environ.setdefault("PYTEST_DEBUG_TEMPROOT", str(Path(tempfile.gettempdir()) / "kashf-tests"))
    Path(os.environ["PYTEST_DEBUG_TEMPROOT"]).mkdir(parents=True, exist_ok=True)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-private",
        action="store_true",
        default=False,
        help="run tests marked private against the real extract in KASHF_DATA_DIR",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-private") and os.environ.get("KASHF_DATA_DIR"):
        return
    skip = pytest.mark.skip(reason="private: needs --run-private and KASHF_DATA_DIR")
    for item in items:
        if "private" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _isolated_environment(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    if "private" in request.keywords:
        return
    for key in list(os.environ):
        if key.startswith("KASHF_"):
            monkeypatch.delenv(key, raising=False)
    # No test may write into the developer's public workspace.
    monkeypatch.setenv("KASHF_DB", str(tmp_path / "work" / "kashf.duckdb"))
    monkeypatch.setenv("KASHF_WORK_DIR", str(tmp_path / "work"))
    monkeypatch.chdir(tmp_path)


GitRunner = Callable[..., str]


@pytest.fixture(scope="session")
def synthetic_extract() -> SyntheticExtract:
    """The default public synthetic extract (1,000 clients, seed 42), in memory."""
    from kashf.synthetic.parametric import ParametricConfig, generate

    return generate(ParametricConfig())


@pytest.fixture(scope="session")
def synthetic_dir(
    tmp_path_factory: pytest.TempPathFactory, synthetic_extract: SyntheticExtract
) -> Path:
    """The default synthetic extract written to disk, exactly as `kashf synth` writes it."""
    from kashf.synthetic.writer import write_extract

    folder = tmp_path_factory.mktemp("synth")
    write_extract(synthetic_extract, folder)
    return folder


def public_settings(root: Path) -> Settings:
    """Settings whose public workspace lives under ``root`` (independent of the environment)."""
    return Settings(  # type: ignore[call-arg]
        _env_file=None,
        db=root / "kashf.duckdb",
        work_dir=root,
        data_dir=None,
        private_dir=None,
    )


@pytest.fixture(scope="session")
def public_warehouse(tmp_path_factory: pytest.TempPathFactory, synthetic_dir: Path) -> Settings:
    """A public warehouse ingested from :func:`synthetic_dir`, shared by read-only tests."""
    from kashf.ingest.load import ingest

    settings = public_settings(tmp_path_factory.mktemp("warehouse"))
    ingest(settings, Zone.PUBLIC, synthetic_dir)
    return settings


def build_warehouse(settings: Settings, zone: Zone, source: Path | None) -> None:
    """Ingest, measure quality and run the whole dbt build into ``settings``' warehouse."""
    from kashf.dbt_runner import run_dbt
    from kashf.ingest import dq
    from kashf.ingest.load import ingest
    from kashf.warehouse import connect

    ingest(settings, zone, source)
    con = connect(settings.workspace(zone).warehouse)
    try:
        dq.run(con)
    finally:
        con.close()
    assert run_dbt(settings, zone, ["build"]).success


@pytest.fixture(scope="session")
def built_warehouse(tmp_path_factory: pytest.TempPathFactory, synthetic_dir: Path) -> Settings:
    """A public warehouse with every dbt model built from the default synthetic extract."""
    settings = public_settings(tmp_path_factory.mktemp("built"))
    build_warehouse(settings, Zone.PUBLIC, synthetic_dir)
    return settings


@pytest.fixture(scope="session")
def synthetic_ledger(built_warehouse: Settings) -> Ledger:
    """Every registered claim computed on the built synthetic warehouse."""
    from kashf.ledger.build import build_ledger

    result = build_ledger(built_warehouse, Zone.PUBLIC)
    assert result.skipped == {}
    return result.ledger


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """An empty git repository with a deterministic identity and LF line endings."""
    repo = tmp_path / "repo"
    repo.mkdir()
    for args in (
        ["init", "-q"],
        ["config", "user.email", "tests@example.invalid"],
        ["config", "user.name", "tests"],
        ["config", "core.autocrlf", "false"],
        ["config", "commit.gpgsign", "false"],
    ):
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)
    return repo


@pytest.fixture
def git(git_repo: Path) -> GitRunner:
    """Run a git command inside :func:`git_repo` and return its stdout."""

    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(git_repo), *args], check=True, capture_output=True, text=True
        )
        return completed.stdout

    return run
