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


@app.command()
def synth(
    ctx: typer.Context,
    out: Annotated[Path, typer.Option("--out", help="Destination folder.")] = Path(".kashf/synth"),
    clients: Annotated[int, typer.Option("--clients", min=1, help="Clients per sample.")] = 1000,
    seed: Annotated[int | None, typer.Option("--seed", help="Defaults to KASHF_SEED.")] = None,
) -> None:
    """Write the public synthetic extract: five files shaped exactly like the real ones."""
    from kashf.contracts.specs import EXTRACTS
    from kashf.synthetic.parametric import ParametricConfig, generate
    from kashf.synthetic.writer import write_extract

    settings = _state(ctx).settings
    config = ParametricConfig(n_clients=clients, seed=settings.seed if seed is None else seed)
    extract = generate(config)
    marker = write_extract(extract, out.resolve(), real_data_dir=settings.data_dir)
    table = Table(title=f"synthetic extract (seed {config.seed}, {clients} clients)")
    for column in ("file", "rows", "sha256"):
        table.add_column(column)
    for spec in EXTRACTS:
        rows = len(extract.frames[spec.name])
        table.add_row(spec.filename, str(rows), marker.files[spec.filename][:12])
    console.print(table)


@app.command()
def ingest(
    ctx: typer.Context,
    data_dir: Annotated[
        Path | None,
        typer.Option("--data-dir", help="Synthetic folder (public) or the extract (private)."),
    ] = None,
) -> None:
    """Validate the five files against their contracts and load the raw layer of the warehouse."""
    from kashf.ingest.load import ingest as run_ingest

    state = _state(ctx)
    report = run_ingest(state.settings, state.zone, data_dir)
    table = Table(title=f"ingest ({state.zone.value}, {report.manifest.source_kind} data)")
    for column in ("extract", "rows", "columns", "first date", "last date", "line ends"):
        table.add_column(column)
    for item in report.manifest.files:
        table.add_row(
            item.extract,
            str(item.rows),
            str(item.columns),
            str(item.date_min or ""),
            str(item.date_max or ""),
            item.line_terminator,
        )
    console.print(table)
    console.print(f"warehouse: {_describe(report.workspace.warehouse, state)}")
    console.print(f"fingerprint: {report.manifest.fingerprint[:16]}")
    failed = [check for check in report.reconciliations if not check.ok]
    for check in failed:
        err_console.print(f"[red]reconciliation failed[/red] {check.extract} {check.measure}")
    if failed:
        raise typer.Exit(code=1)
    console.print(f"reconciliation: {len(report.reconciliations)} checks passed")


@app.command()
def dq(ctx: typer.Context) -> None:
    """Measure data quality on the raw layer; counts only, never values. Exit 1 on errors."""
    from kashf.ingest import dq as quality
    from kashf.warehouse import connect

    state = _state(ctx)
    warehouse = state.workspace.warehouse
    if not warehouse.is_file():
        err_console.print("no warehouse yet: run `kashf ingest` first")
        raise typer.Exit(code=2)
    con = connect(warehouse)
    try:
        results = quality.run(con)
    finally:
        con.close()
    table = Table(title=f"data quality ({state.zone.value})")
    for column in ("check", "extract", "status", "count", "of"):
        table.add_column(column, overflow="fold")
    colours = {"ok": "green", "info": "cyan", "warning": "yellow", "error": "red"}
    for result in results:
        table.add_row(
            result.check_id,
            result.extract,
            f"[{colours[result.status]}]{result.status}[/]",
            str(result.count),
            "" if result.denominator is None else str(result.denominator),
        )
    console.print(table)
    if quality.has_errors(results):
        raise typer.Exit(code=1)


@app.command(
    "dbt",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def dbt_command(ctx: typer.Context) -> None:
    """Run dbt on the zone's warehouse: `kashf dbt build`, `kashf dbt docs generate`, ..."""
    from kashf.dbt_runner import run_dbt

    state = _state(ctx)
    if not ctx.args:
        err_console.print("kashf dbt: give a dbt command, for example `kashf dbt build`")
        raise typer.Exit(code=2)
    result = run_dbt(state.settings, state.zone, list(ctx.args))
    if not result.success:
        raise typer.Exit(code=1)


ledger_app = typer.Typer(
    name="ledger",
    help="The evidence ledger: build claims, render documents, check for drift.",
    no_args_is_help=True,
)
app.add_typer(ledger_app)


@ledger_app.command("build")
def ledger_build(
    ctx: typer.Context,
    public: Annotated[
        bool,
        typer.Option(
            "--public", help="Also write the allow-listed claims to docs/ledger (private zone)."
        ),
    ] = False,
    resamples: Annotated[
        int, typer.Option("--resamples", min=50, help="Bootstrap resamples.")
    ] = 1000,
    only: Annotated[
        list[str] | None, typer.Option("--only", help="Compute only these claim ids.")
    ] = None,
) -> None:
    """Evaluate every registered claim on the zone's warehouse and write the ledger."""
    from kashf.ledger.build import build_ledger

    state = _state(ctx)
    result = build_ledger(
        state.settings, state.zone, public=public, n_resamples=resamples, only=only
    )
    table = Table(title=f"ledger ({state.zone.value}, {result.ledger.source_kind} data)")
    for column in ("claim", "clients", "headline", "verdict", "public"):
        table.add_column(column, overflow="fold")
    for record in result.ledger.claims:
        verdict = record.robustness.verdict if record.robustness else ""
        table.add_row(
            record.id,
            str(record.n),
            "yes" if record.headline else "",
            verdict,
            "yes" if record.public else "",
        )
    console.print(table)
    console.print(f"ledger: {_describe(result.path, state)}")
    if result.public_path is not None:
        console.print(
            f"public claims: {_describe(result.public_path, AppState(state.settings, Zone.PUBLIC))}"
        )
    for claim_id, reason in result.skipped.items():
        err_console.print(f"[yellow]skipped[/yellow] {claim_id}: {escape(reason)}")


@ledger_app.command("render")
def ledger_render(ctx: typer.Context) -> None:
    """Render docs/findings and the README section from docs/ledger/claims.public.json."""
    from kashf.ledger.build import PUBLIC_LEDGER, load_ledger
    from kashf.ledger.render import render_all

    root = _state(ctx).settings.repo_root
    written = render_all(root, load_ledger(root / PUBLIC_LEDGER))
    for path in written:
        console.print(f"rendered {path.relative_to(root).as_posix()}")


@ledger_app.command("check")
def ledger_check(ctx: typer.Context) -> None:
    """Fail on drift, on hand-typed numbers in governed documents, or on non-allow-listed claims."""
    from kashf.ledger.check import check_repository

    problems = check_repository(_state(ctx).settings.repo_root)
    for problem in problems:
        err_console.print(f"[red]ledger[/red] {escape(problem.render())}", highlight=False)
    if problems:
        err_console.print(f"ledger check: {len(problems)} problem(s)")
        raise typer.Exit(code=1)
    console.print("ledger check: documents match the ledger; no hand-typed numbers")


@app.command()
def report(
    ctx: typer.Context,
    variant: Annotated[
        str, typer.Option("--variant", help="public (synthetic demo) or private.")
    ] = "public",
) -> None:
    """Render the HTML report: private (every real claim, private folder) or public (demo)."""
    from kashf.report.build import build_report

    state = _state(ctx)
    if variant not in {"public", "private"}:
        err_console.print("--variant must be public or private")
        raise typer.Exit(code=2)
    path = build_report(state.settings, "private" if variant == "private" else "public")
    shown = AppState(state.settings, Zone.PRIVATE if variant == "private" else Zone.PUBLIC)
    console.print(f"report: {_describe(path, shown)}")


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
