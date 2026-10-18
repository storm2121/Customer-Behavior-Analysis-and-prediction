"""``strata analyse``: run one analysis module (or all of them) on the zone's warehouse.

A module recomputes the ledger claims it owns (merged into the zone's ledger, so the other claims
stay), then draws its figures into the zone's figures folder. The Markov module also saves what the
API will need to serve the simulator (M7): the transition counts, the branch operations per state,
the starting state counts and the bands of the sample policy. Figures and the simulator file are
computed from the warehouse, so on a private run they are written under ``STRATA_PRIVATE_DIR`` and
nowhere else.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from strata.analysis import calendar, figures, markov
from strata.config import Settings, Workspace, Zone
from strata.ledger.build import build_ledger
from strata.ledger.claims.digits import BENFORD_SERIES, MIN_AMOUNTS
from strata.ledger.claims.markov import PolicyRun, run_policy
from strata.ledger.model import Ledger
from strata.ledger.registry import ClaimContext
from strata.warehouse import connect

SIMULATOR_FILE = Path("analysis") / "markov_simulator.json"


@dataclass(frozen=True)
class Analysis:
    """What a module draws from: its workspace, read access to the warehouse, its fresh claims."""

    workspace: Workspace
    context: ClaimContext
    ledger: Ledger

    @property
    def folder(self) -> Path:
        return self.workspace.figures_dir


@dataclass(frozen=True)
class Outputs:
    paths: list[Path]
    policy: PolicyRun | None = None


Step = Callable[[Analysis], Outputs]


@dataclass(frozen=True)
class Module:
    name: str
    claims: tuple[str, ...]
    outputs: Step


@dataclass(frozen=True)
class ModuleResult:
    name: str
    claims: list[str]
    outputs: list[Path]
    skipped: dict[str, str] = field(default_factory=dict)
    policy: PolicyRun | None = None


def _calendar(analysis: Analysis) -> Outputs:
    if "calendar_effects" not in analysis.ledger.ids:
        return Outputs([])
    fig = figures.calendar_forest(analysis.ledger.get("calendar_effects"))
    return Outputs([figures.write(fig, analysis.folder, "calendar_effects")])


def _seasonality(analysis: Analysis) -> Outputs:
    daily = analysis.context.table("marts.daily_counts")
    paths = []
    for series in calendar.CHANNEL_SERIES:
        frame = calendar.daily_series(daily, series)
        if frame["n_events"].sum() == 0:
            continue
        fig = figures.stl_components(frame, f"{series.channel} events")
        paths.append(figures.write(fig, analysis.folder, f"seasonality_stl_{series.channel}"))
    return Outputs(paths)


def _concentration(analysis: Analysis) -> Outputs:
    fig = figures.lorenz_curves(analysis.context.table("marts.client_360"))
    return Outputs([figures.write(fig, analysis.folder, "concentration_lorenz")])


def _benford(analysis: Analysis) -> Outputs:
    events = analysis.context.table("intermediate.int_events")
    events = events[~events["is_repeat"] & (events["amount"] > 0)]
    amounts = {}
    for name, (channel, kind) in BENFORD_SERIES.items():
        rows = events[(events["channel"] == channel) & (events["kind_group"] == kind)]
        if len(rows) >= MIN_AMOUNTS:
            amounts[name] = rows["amount"].to_numpy(dtype=float)
    fig = figures.benford_bars(amounts)
    return Outputs([figures.write(fig, analysis.folder, "benford_first_digits")])


def simulator_payload(run: PolicyRun) -> dict[str, object]:
    """The simulator's inputs and the sample policy's bands, as the API will serve them."""
    return {
        "states": list(markov.STATES),
        "transition_counts": run.counts.tolist(),
        "branch_rates": run.rates,
        "start_counts": run.start.tolist(),
        "posterior_samples": run.n_samples,
        "sample_policy": {
            "description": run.policy.description,
            "start_counts": run.policy_start.tolist(),
            "bands": markov.bands(run.policy_paths).quantiles,
            "baseline_bands": markov.bands(run.baseline_paths).quantiles,
        },
    }


def _markov(analysis: Analysis) -> Outputs:
    run = run_policy(analysis.context)
    fig = figures.simulation_fan(run.policy_paths, run.baseline_paths, run.policy.description)
    figure = figures.write(fig, analysis.folder, "markov_policy")
    path = analysis.workspace.root / SIMULATOR_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(simulator_payload(run), indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")
    return Outputs([figure, path], policy=run)


MODULES: dict[str, Module] = {
    module.name: module
    for module in (
        Module(
            "calendar",
            ("calendar_effects", "calendar_bayes", "weekday_pattern", "salary_day_withdrawals"),
            _calendar,
        ),
        Module(
            "seasonality",
            (
                "weekly_seasonality",
                "december_deposit_wave",
                "pre_eid_branch_deposits",
                "ramadan_card_effect",
            ),
            _seasonality,
        ),
        Module("concentration", ("spend_concentration", "rfm_by_channel"), _concentration),
        Module(
            "benford", ("benford_purchases", "benford_by_category", "benford_two_digits"), _benford
        ),
        Module("markov", ("channel_markov", "markov_policy_simulation"), _markov),
    )
}


@contextmanager
def _analysis(
    settings: Settings, workspace: Workspace, ledger: Ledger, *, quick: bool
) -> Iterator[Analysis]:
    con = connect(workspace.warehouse, read_only=True)
    try:
        context = ClaimContext(con, seed=settings.seed, quick=quick)
        yield Analysis(workspace=workspace, context=context, ledger=ledger)
    finally:
        con.close()


def run_module(settings: Settings, zone: Zone, name: str, *, quick: bool = False) -> ModuleResult:
    """Recompute the claims of module ``name`` and write its figures (and the simulator file)."""
    if name not in MODULES:
        msg = f"unknown module {name!r}; choose from {sorted(MODULES)}"
        raise ValueError(msg)
    module = MODULES[name]
    workspace = settings.workspace(zone)
    if workspace.is_private:
        settings.assert_private_output(workspace.figures_dir)
        settings.assert_private_output(workspace.root / SIMULATOR_FILE)
    built = build_ledger(settings, zone, only=list(module.claims), quick=quick)
    computed = [claim_id for claim_id in module.claims if claim_id not in built.skipped]
    if not computed:
        return ModuleResult(name=name, claims=[], outputs=[], skipped=built.skipped)
    with _analysis(settings, workspace, built.ledger, quick=quick) as analysis:
        outputs = module.outputs(analysis)
    return ModuleResult(
        name=name,
        claims=computed,
        outputs=outputs.paths,
        skipped=built.skipped,
        policy=outputs.policy,
    )


def run_all(settings: Settings, zone: Zone, *, quick: bool = False) -> list[ModuleResult]:
    return [run_module(settings, zone, name, quick=quick) for name in MODULES]
