"""``kashf dbt``: run dbt against the active zone's warehouse.

dbt is invoked in-process through its programmatic runner. Everything that depends on the zone is
decided here, never in the dbt project: the warehouse path (through the ``KASHF_DBT_DB`` variable
the profile reads), and the ``target/`` and ``logs/`` folders, which on a private run hold compiled
SQL and run results about real data and therefore live under ``KASHF_PRIVATE_DIR`` (ADR 0008).
Anonymous usage statistics are switched off in the environment as well as in the project.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from kashf.config import Settings, Workspace, Zone

DBT_ENV_FLAGS = {
    "DBT_SEND_ANONYMOUS_USAGE_STATS": "false",
    "DO_NOT_TRACK": "1",
    "DBT_USE_COLORS": "false",
}


@dataclass(frozen=True, slots=True)
class DbtResult:
    """Outcome of one dbt command."""

    command: tuple[str, ...]
    success: bool


def project_dir(settings: Settings) -> Path:
    return settings.repo_root / "dbt"


@contextmanager
def _environment(values: dict[str, str]) -> Iterator[None]:
    saved = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def dbt_arguments(settings: Settings, workspace: Workspace, args: Sequence[str]) -> list[str]:
    """The full dbt command line: the caller's arguments plus the zone's folders."""
    folder = str(project_dir(settings))
    return [
        *args,
        "--project-dir",
        folder,
        "--profiles-dir",
        folder,
        "--target-path",
        str(workspace.dbt_target_dir),
        "--log-path",
        str(workspace.dbt_log_dir),
    ]


def run_dbt(settings: Settings, zone: Zone, args: Sequence[str]) -> DbtResult:
    """Run one dbt command (``run``, ``test``, ``build``, ``docs generate`` ...) in ``zone``."""
    from dbt.cli.main import dbtRunner  # noqa: PLC0415 - dbt is heavy; import it only to run it

    if not args:
        msg = "give a dbt command, for example `kashf dbt build`"
        raise ValueError(msg)
    workspace = settings.workspace(zone)
    if workspace.is_private:
        for path in (workspace.warehouse, workspace.dbt_target_dir, workspace.dbt_log_dir):
            settings.assert_private_output(path)
    if not workspace.warehouse.is_file():
        msg = "No warehouse yet: run `kashf ingest` (and `kashf dq`) first."
        raise FileNotFoundError(msg)
    workspace.dbt_target_dir.mkdir(parents=True, exist_ok=True)
    workspace.dbt_log_dir.mkdir(parents=True, exist_ok=True)
    command = dbt_arguments(settings, workspace, args)
    with _environment({"KASHF_DBT_DB": str(workspace.warehouse), **DBT_ENV_FLAGS}):
        try:
            result = dbtRunner().invoke(command)
        finally:
            release_duckdb()
    return DbtResult(command=tuple(args), success=bool(result.success))


def release_duckdb() -> None:
    """Close the DuckDB connection dbt-duckdb keeps open after a command.

    The adapter caches one database handle per process (so in-memory databases survive between
    nodes) and its ``close_all_connections`` only drops the reference. While that handle lives, the
    same process cannot open the warehouse read-only, which the ledger and the API need. The cache
    is an adapter internal, so a change in its shape is tolerated rather than fatal.
    """
    try:
        from dbt.adapters.duckdb.connections import (  # noqa: PLC0415 - see run_dbt
            DuckDBConnectionManager,
        )
    except ImportError:  # pragma: no cover - dbt-duckdb is a declared dependency
        return
    environment = getattr(DuckDBConnectionManager, "_ENV", None)
    close = getattr(environment, "close", None)
    if callable(close):
        close()
    forget = getattr(DuckDBConnectionManager, "close_all_connections", None)
    if callable(forget):
        forget()
