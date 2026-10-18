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

import pytest


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
    monkeypatch.chdir(tmp_path)


GitRunner = Callable[..., str]


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
