"""Claims from the Markov chain of monthly channel states (M4)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from strata.analysis import markov
from strata.contracts.vocab import age_bands
from strata.ledger.bootstrap import claim_seed
from strata.ledger.claims._common import require
from strata.ledger.model import Unit
from strata.ledger.registry import ClaimContext, ClaimResult, claim

SAMPLE_POLICY = markov.younger_than(45, 0.3)
"""The sample policy of ARCHITECTURE section 9, used by the claim and the CLI demonstration."""

HORIZON = 12
INTERVAL = markov.BAND_INTERVAL


def _band(samples: np.ndarray) -> tuple[float, float]:
    return (float(np.percentile(samples, 5)), float(np.percentile(samples, 95)))


@dataclass(frozen=True)
class Chain:
    """Monthly states (with branch operations per client-month) and the posterior matrices."""

    states: pd.DataFrame
    counts: markov.FloatArray
    samples: markov.FloatArray

    @property
    def complete(self) -> pd.DataFrame:
        return self.states[~self.states["is_partial_month"].astype(bool)]

    @property
    def latest(self) -> pd.DataFrame:
        """Client states in the latest complete month: where the simulator starts."""
        complete = self.complete
        latest = complete["month_start"].eq(complete["month_start"].max())
        return complete.loc[latest]


def chain(context: ClaimContext) -> Chain:
    states = context.table("intermediate.int_channel_state")
    months = context.table("intermediate.int_client_month")[
        ["client_id", "month_start", "n_branch"]
    ]
    states = states.merge(months, on=["client_id", "month_start"], how="left")
    counts = markov.transition_counts(states)
    require(counts.sum() > 0, "no month-to-month transition")
    samples = markov.posterior_samples(
        counts, n_samples=context.posterior_samples, seed=claim_seed(context.seed, "markov")
    )
    return Chain(states=states, counts=counts, samples=samples)


@dataclass(frozen=True)
class PolicyRun:
    """One what-if and its baseline, projected with the same posterior samples (paired paths)."""

    policy: markov.Policy
    counts: markov.FloatArray
    rates: dict[str, float]
    start: markov.FloatArray
    policy_start: markov.FloatArray
    baseline_paths: markov.Paths
    policy_paths: markov.Paths
    n_clients: int
    n_samples: int
    start_by_age_band: dict[str, markov.FloatArray]


def run_policy(
    context: ClaimContext, policy: markov.Policy = SAMPLE_POLICY, horizon: int = HORIZON
) -> PolicyRun:
    """Shared by the claim and ``strata analyse markov`` so both report the same projection."""
    fitted = chain(context)
    latest = fitted.latest
    clients = context.table("staging.stg_clients")
    rates = markov.branch_rates(fitted.complete)
    start = markov.initial_counts(latest, clients, markov.NO_POLICY)
    policy_start = markov.initial_counts(latest, clients, policy)
    return PolicyRun(
        policy=policy,
        counts=fitted.counts,
        rates=rates,
        start=start,
        policy_start=policy_start,
        baseline_paths=markov.simulate_paths(fitted.samples, start, rates, horizon),
        policy_paths=markov.simulate_paths(fitted.samples, policy_start, rates, horizon),
        n_clients=int(latest["client_id"].nunique()),
        n_samples=int(fitted.samples.shape[0]),
        start_by_age_band=markov.counts_by_band(latest, clients, age_bands()),
    )


@claim(
    "channel_markov",
    title="How clients move between channels from month to month",
    question=(
        "How likely is each monthly channel state to follow each other, where would the mix "
        "settle, and how long does a move take?"
    ),
    method=(
        "Transitions between consecutive complete months of int_channel_state; each row of the "
        "transition matrix has a Dirichlet(1) prior and a Dirichlet(1 + counts) posterior; "
        "posterior means with 90% intervals from posterior samples, for the transition "
        "probabilities, the stationary distribution and the mean first-passage times (fundamental "
        "matrix)."
    ),
)
def channel_markov(context: ClaimContext) -> ClaimResult:
    fitted = chain(context)
    samples = fitted.samples
    values: dict[str, float] = {}
    intervals: dict[str, tuple[float, float]] = {}
    units: dict[str, Unit] = {}
    for i, origin in enumerate(markov.STATES):
        for j, destination in enumerate(markov.STATES):
            key = f"p_{origin}_to_{destination}"
            values[key] = float(samples[:, i, j].mean())
            intervals[key] = _band(samples[:, i, j])
            units[key] = Unit.SHARE
    stationary = np.array([markov.stationary(matrix) for matrix in samples])
    passage = np.array([markov.mean_first_passage(matrix) for matrix in samples])
    for j, state in enumerate(markov.STATES):
        values[f"stationary_{state}"] = float(stationary[:, j].mean())
        intervals[f"stationary_{state}"] = _band(stationary[:, j])
        units[f"stationary_{state}"] = Unit.SHARE
    for origin, destination in (
        ("branch_only", "mixed"),
        ("branch_only", "digital_only"),
        ("digital_only", "none"),
        ("mixed", "none"),
        ("none", "mixed"),
    ):
        i, j = markov.STATES.index(origin), markov.STATES.index(destination)
        key = f"months_{origin}_to_{destination}"
        values[key] = float(np.median(passage[:, i, j]))
        intervals[key] = _band(passage[:, i, j])
        units[key] = Unit.NUMBER
    n_clients = int(fitted.states["client_id"].nunique())
    return ClaimResult(
        values=values,
        units=units,
        ci=intervals,
        interval_label=INTERVAL,
        value_n=dict.fromkeys(values, n_clients),
        n=n_clients,
        n_events=int(fitted.counts.sum()),
        meta={"posterior_samples": str(samples.shape[0]), "states": ", ".join(markov.STATES)},
        caveats=[
            (
                "The chain is time-homogeneous: one transition matrix for the whole year, so "
                "seasonal moves (December, summer) are averaged into it."
            ),
            (
                "A state is a month's channel mix; a client with no event in a month is in the "
                "none state that month, which is not the same as having left the bank."
            ),
            "Months to a state are median expected months to first reach it under the chain.",
        ],
    )


@claim(
    "markov_policy_simulation",
    title="What if younger branch-only clients went digital?",
    question=(
        "If a share of branch-only clients under 45 moved to mixed use this month, how would "
        "branch load and the digital share differ from the baseline over the following year?"
    ),
    method=(
        "The latest complete month's client states, with the policy applied, projected twelve "
        "months with every posterior sample of the transition matrix; branch load uses the mean "
        "branch operations per client-month of each state. The baseline is the same projection "
        "without the policy; differences are taken sample by sample, with medians and 90% bands."
    ),
)
def markov_policy_simulation(context: ClaimContext) -> ClaimResult:
    run = run_policy(context)
    baseline, policy = run.baseline_paths, run.policy_paths
    values: dict[str, float] = {}
    intervals: dict[str, tuple[float, float]] = {}
    units: dict[str, Unit] = {}

    def record(key: str, draws: np.ndarray, unit: Unit) -> None:
        values[key] = float(np.median(draws))
        intervals[key] = _band(draws)
        units[key] = unit

    for month in (0, 1, 3, HORIZON):
        record(
            f"baseline_branch_load_month_{month}", baseline["branch_load"][:, month], Unit.NUMBER
        )
        record(
            f"baseline_digital_share_month_{month}", baseline["digital_share"][:, month], Unit.SHARE
        )
        record(
            f"baseline_silent_share_month_{month}", baseline["silent_share"][:, month], Unit.SHARE
        )
        record(
            f"digital_share_gain_month_{month}",
            policy["digital_share"][:, month] - baseline["digital_share"][:, month],
            Unit.DIFFERENCE,
        )
        record(
            f"branch_load_change_month_{month}",
            policy["branch_load"][:, month] - baseline["branch_load"][:, month],
            Unit.NUMBER,
        )
    saved = (baseline["branch_load"][:, 1:] - policy["branch_load"][:, 1:]).sum(axis=1)
    record("branch_operations_saved_over_horizon", saved, Unit.NUMBER)
    for state, rate in run.rates.items():
        values[f"branch_ops_per_client_month_{state}"] = rate
        units[f"branch_ops_per_client_month_{state}"] = Unit.NUMBER
    return ClaimResult(
        values=values,
        units=units,
        ci=intervals,
        interval_label=INTERVAL,
        value_n=dict.fromkeys(values, run.n_clients),
        n=run.n_clients,
        meta={
            "policy": run.policy.description,
            "horizon_months": str(HORIZON),
            "posterior_samples": str(run.n_samples),
        },
        caveats=[
            (
                "A what-if under the fitted chain, not a forecast: it assumes moved clients then "
                "behave like other mixed-use clients and that transition rates stay as observed."
            ),
            (
                "A one-off move fades as the chain returns to its stationary mix, so the effect is "
                "largest in the first months."
            ),
            (
                "Bands carry the uncertainty of the transition matrix, not the noise of a finite "
                "population or of the branch operation rates."
            ),
            (
                "Branch load rises under a move to mixed use whenever mixed-use clients make more "
                "branch operations per month than branch-only clients; the rates per state are "
                "part of the claim."
            ),
        ],
    )
