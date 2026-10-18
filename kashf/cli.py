"""Command-line interface: ``kashf <command>``.

Heavy libraries are imported inside the commands so that ``kashf --help`` and the privacy guard,
which runs on every commit, start instantly.

Errors are printed sanitised: exception type and a masked message, never a traceback with local
variables, because a local variable in a private run can hold a row of real data. The full
traceback of a private run is written to the private log folder instead.
"""

from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from kashf import __version__
from kashf.config import PrivacyBoundaryError, Settings, Workspace, Zone
from kashf.log import configure_logging, mask_identifiers

app = typer.Typer(
    name="kashf",
    help="Kashf: retail-banking behaviour analytics with an evidence ledger.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
    rich_markup_mode="rich",
)

console = Console()
err_console = Console(stderr=True)


@dataclass(slots=True)
class AppState:
    """Resolved once per invocation and handed to every command through the typer context."""

    settings: Settings
    zone: Zone

    @property
    def workspace(self) -> Workspace:
        return self.settings.workspace(self.zone)


def _state(ctx: typer.Context) -> AppState:
    state = ctx.obj
    if not isinstance(state, AppState):  # pragma: no cover - typer always runs the callback
        msg = "CLI state missing"
        raise TypeError(msg)
    return state


def _version(value: bool) -> None:
    if value:
        console.print(f"kashf {__version__}")
        raise typer.Exit


@app.callback()
def _root(
    ctx: typer.Context,
    private: Annotated[
        bool,
        typer.Option(
            "--private",
            help="Run in the private zone (real data; outputs only under KASHF_PRIVATE_DIR).",
        ),
    ] = False,
    zone: Annotated[
        Zone | None,
        typer.Option("--zone", envvar="KASHF_ZONE", help="public (synthetic) or private (real)."),
    ] = None,
    log_level: Annotated[str, typer.Option("--log-level", envvar="KASHF_LOG_LEVEL")] = "INFO",
    log_json: Annotated[bool, typer.Option("--log-json", envvar="KASHF_LOG_JSON")] = False,
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version, is_eager=True, help="Show the version."),
    ] = False,
) -> None:
    """Shared options. The zone is explicit: real data is only ever read with --private."""
    del version
    settings = Settings()
    chosen = Zone.PRIVATE if private else (zone or settings.zone)
    configure_logging(log_level, json=log_json)
    ctx.obj = AppState(settings=settings, zone=chosen)


def _describe(path: Path, state: AppState) -> str:
    """A path as it may be shown on screen: private locations are never spelled out."""
    settings = state.settings
    if state.zone is Zone.PRIVATE:
        try:
            rel = path.resolve().relative_to(settings.resolved_private_dir)
        except ValueError:
            return "<outside KASHF_PRIVATE_DIR>"
        return f"<KASHF_PRIVATE_DIR>/{rel.as_posix()}"
    try:
        return path.resolve().relative_to(settings.repo_root).as_posix()
    except ValueError:
        return path.as_posix()


def _private_dir_configured(settings: Settings) -> bool:
    try:
        _ = settings.resolved_private_dir
    except PrivacyBoundaryError:
        return False
    return True


@app.command()
def info(ctx: typer.Context) -> None:
    """Show the resolved configuration without revealing private locations."""
    state = _state(ctx)
    settings = state.settings
    table = Table(title=f"kashf {__version__}", show_header=False)
    table.add_row("zone", state.zone.value)
    table.add_row("KASHF_DATA_DIR", "set" if settings.data_dir else "not set")
    configured = _private_dir_configured(settings)
    private = "configured (outside the repository)" if configured else "not configured"
    table.add_row("KASHF_PRIVATE_DIR", private)
    if state.zone is Zone.PUBLIC or configured:
        workspace = state.workspace
        table.add_row("warehouse", _describe(workspace.warehouse, state))
        table.add_row("ledger", _describe(workspace.ledger_dir, state))
        table.add_row("reports", _describe(workspace.reports_dir, state))
    console.print(table)


@app.command("privacy-check")
def privacy_check(
    ctx: typer.Context,
    paths: Annotated[
        list[Path] | None, typer.Argument(help="Files to scan (as pre-commit passes them).")
    ] = None,
    staged: Annotated[
        bool, typer.Option("--staged", help="Scan the staged content of the index.")
    ] = False,
    all_files: Annotated[
        bool, typer.Option("--all-files", help="Scan every tracked file.")
    ] = False,
) -> None:
    """Block extract files, extract-shaped content, identifiers and non-allow-listed claims."""
    from kashf import privacy_guard

    repo = _state(ctx).settings.repo_root
    if staged:
        files = privacy_guard.staged_files(repo)
    elif all_files:
        files = privacy_guard.tracked_files(repo)
    elif paths:
        files = privacy_guard.disk_files(paths, repo)
    else:
        err_console.print("privacy-check: give paths, --staged or --all-files")
        raise typer.Exit(code=2)
    scanner = privacy_guard.Scanner.from_environment()
    findings = privacy_guard.scan(files, scanner)
    if findings:
        for finding in findings:
            err_console.print(f"[red]BLOCKED[/red] {escape(finding.render())}", highlight=False)
        source = scanner.matcher.source
        err_console.print(f"privacy-check: {len(findings)} finding(s); identifier rule: {source}")
        raise typer.Exit(code=1)
    console.print(f"privacy-check: clean (identifier rule: {scanner.matcher.source})")


def _report_failure(error: BaseException) -> None:
    """Print a sanitised error; keep the full traceback in the private log on private runs."""
    message = mask_identifiers(str(error))[:500]
    err_console.print(
        f"[red]error[/red] {type(error).__name__}: {escape(message)}", highlight=False
    )
    try:
        settings = Settings()
        log_dir = settings.resolved_private_dir / "logs"
    except Exception:  # noqa: BLE001 - reporting must never raise
        return
    if "--private" not in sys.argv and settings.zone is not Zone.PRIVATE:
        return
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        (log_dir / f"kashf-error-{stamp}.log").write_text(
            "".join(traceback.format_exception(error)), encoding="utf-8"
        )
        err_console.print("full traceback written to <KASHF_PRIVATE_DIR>/logs/")
    except OSError:
        return


def main() -> None:
    """Console-script entry point with sanitised error reporting."""
    try:
        app()
    except PrivacyBoundaryError as error:
        err_console.print(f"[red]privacy boundary[/red] {escape(str(error))}", highlight=False)
        raise SystemExit(3) from None
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as error:  # noqa: BLE001 - last line of defence for the terminal
        _report_failure(error)
        raise SystemExit(1) from None


if __name__ == "__main__":  # pragma: no cover
    main()
