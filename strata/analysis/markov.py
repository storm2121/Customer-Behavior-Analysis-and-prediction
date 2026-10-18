"""A Markov chain of monthly channel states: a Bayesian transition matrix and a what-if simulator.

States are the monthly channel states of ``int_channel_state`` (none, branch_only, digital_only,
mixed). Each row of the transition matrix gets a Dirichlet(1) prior, so the posterior of row ``i``
is Dirichlet(1 + counts_i); posterior samples of the matrix carry every derived quantity's
uncertainty (stationary distribution, mean first-passage times, simulated futures). Equations are
in ``docs/methods/markov.md``.

The simulator answers questions of the form "what if a share ``p`` of branch-only clients matching
a filter moved to mixed use this month?": it projects the state distribution forward with every
posterior sample of the matrix and reports 5th, 50th and 95th percentiles of branch load, digital
share and silent share. It projects expected counts, so its bands carry parameter uncertainty, not
the extra noise of a finite population.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from strata.contracts.vocab import Band, band_of

FloatArray = npt.NDArray[np.float64]

STATES = ("none", "branch_only", "digital_only", "mixed")
DIGITAL_STATES = ("digital_only", "mixed")
BRANCH_STATES = ("branch_only", "mixed")
DEFAULT_SAMPLES = 2000


def transition_counts(states: pd.DataFrame) -> FloatArray:
    """Counts of month-to-month transitions between complete months.

    ``states`` has one row per client and month with ``channel_state``, ``previous_state`` and
    ``is_partial_month``; transitions into a partial month (the window's stub) are dropped.
    """
    usable = states[~states["is_partial_month"].astype(bool) & states["previous_state"].notna()]
    index = {state: i for i, state in enumerate(STATES)}
    counts = np.zeros((len(STATES), len(STATES)))
    origin = usable["previous_state"].map(index).to_numpy(dtype=np.int64)
    destination = usable["channel_state"].map(index).to_numpy(dtype=np.int64)
    np.add.at(counts, (origin, destination), 1.0)
    return counts


def posterior_samples(
    counts: FloatArray, n_samples: int = DEFAULT_SAMPLES, seed: int = 42, prior: float = 1.0
) -> FloatArray:
    """``(n_samples, k, k)`` transition matrices drawn row by row from Dirichlet(prior + counts)."""
    rng = np.random.default_rng(seed)
    k = counts.shape[0]
    samples = np.empty((n_samples, k, k))
    for row in range(k):
        samples[:, row, :] = rng.dirichlet(prior + counts[row], size=n_samples)
    return samples


def stationary(matrix: FloatArray) -> FloatArray:
    """The stationary distribution ``pi`` with ``pi P = pi`` and ``sum(pi) = 1``."""
    k = matrix.shape[0]
    system = np.vstack([matrix.T - np.eye(k), np.ones(k)])
    target = np.concatenate([np.zeros(k), [1.0]])
    solution, *_ = np.linalg.lstsq(system, target, rcond=None)
    return np.asarray(
        np.clip(solution, 0.0, None) / np.clip(solution, 0.0, None).sum(), dtype=np.float64
    )


def mean_first_passage(matrix: FloatArray) -> FloatArray:
    """Expected months to first reach state ``j`` from state ``i`` (diagonal: mean recurrence).

    From the fundamental matrix ``Z = (I - P + 1 pi)^-1``: ``m_ij = (z_jj - z_ij) / pi_j`` for
    ``i != j`` and ``m_jj = 1 / pi_j`` (Kemeny and Snell).
    """
    k = matrix.shape[0]
    pi = stationary(matrix)
    fundamental = np.linalg.inv(np.eye(k) - matrix + np.outer(np.ones(k), pi))
    with np.errstate(divide="ignore", invalid="ignore"):
        passage = (np.diag(fundamental)[None, :] - fundamental) / pi[None, :]
        np.fill_diagonal(passage, 1.0 / pi)
    return np.asarray(passage, dtype=np.float64)


def validate_move(share: float, target: str) -> None:
    """A policy moves a proportion of the branch-only clients to another state."""
    if not 0.0 <= share <= 1.0:
        msg = "a policy share is a proportion between zero and one"
        raise ValueError(msg)
    if target not in STATES or target == "branch_only":
        msg = "a policy moves branch-only clients to none, digital_only or mixed"
        raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class Policy:
    """Move ``share`` of the branch-only clients chosen by ``eligible`` to ``target`` at month 0."""

    share: float
    description: str
    eligible: Callable[[pd.DataFrame], pd.Series]
    target: str = "mixed"

    def __post_init__(self) -> None:
        validate_move(self.share, self.target)


TARGET_WORDS = {"mixed": "mixed use", "digital_only": "digital only", "none": "no activity"}


def younger_than(age: int, share: float, target: str = "mixed") -> Policy:
    """The sample policy of ARCHITECTURE section 9: younger branch-only clients adopt digital."""
    words = TARGET_WORDS.get(target, target)
    return Policy(
        share=share,
        description=f"{share:.0%} of branch-only clients under {age} move to {words}",
        eligible=lambda clients: clients["age"] < age,
        target=target,
    )


NO_POLICY = Policy(share=0.0, description="no intervention", eligible=lambda c: c["age"] < 0)


@dataclass(frozen=True)
class Bands:
    """Percentiles of each simulated quantity, one row per month of the horizon (month 0 first)."""

    months: list[int]
    quantiles: dict[str, dict[str, list[float]]]

    def at(self, quantity: str, month: int, level: str = "p50") -> float:
        return self.quantiles[quantity][level][self.months.index(month)]


def initial_counts(month_states: pd.DataFrame, clients: pd.DataFrame, policy: Policy) -> FloatArray:
    """Clients per state in the starting month, after the policy has moved its share."""
    joined = month_states.merge(clients[["client_id", "age"]], on="client_id", how="left")
    counts = np.array([(joined["channel_state"] == state).sum() for state in STATES], dtype=float)
    branch_only = joined["channel_state"] == "branch_only"
    eligible = branch_only & policy.eligible(joined).fillna(value=False).astype(bool)
    moved = policy.share * float(eligible.sum())
    counts[STATES.index("branch_only")] -= moved
    counts[STATES.index(policy.target)] += moved
    return counts


def counts_by_band(
    month_states: pd.DataFrame, clients: pd.DataFrame, bands: Sequence[Band]
) -> dict[str, FloatArray]:
    """Clients per state in the starting month, for every age band (and ``missing``).

    With these counts a policy filtered on age bands can be applied without client rows, which is
    what the API and the dashboard's sliders need.
    """
    joined = month_states.merge(clients[["client_id", "age"]], on="client_id", how="left")
    labels = pd.Series(
        [band_of(None if pd.isna(age) else float(age), tuple(bands)) for age in joined["age"]],
        index=joined.index,
        dtype="object",
    ).fillna("missing")
    names = [*(band.label for band in bands), "missing"]
    return {
        name: np.array(
            [float(((labels == name) & (joined["channel_state"] == s)).sum()) for s in STATES]
        )
        for name in names
    }


def banded_start(
    by_band: dict[str, FloatArray], eligible: Sequence[str], share: float, target: str = "mixed"
) -> FloatArray:
    """Starting counts after moving ``share`` of the branch-only clients of ``eligible`` bands."""
    validate_move(share, target)
    start = np.sum(list(by_band.values()), axis=0).astype(np.float64)
    moved = share * sum(float(by_band[band][STATES.index("branch_only")]) for band in eligible)
    start[STATES.index("branch_only")] -= moved
    start[STATES.index(target)] += moved
    return start


Paths = dict[str, FloatArray]
"""``(posterior samples, months)`` trajectories of each simulated quantity."""

QUANTITIES = ("branch_load", "digital_share", "silent_share")


def simulate_paths(
    samples: FloatArray,
    start: FloatArray,
    branch_rates: dict[str, float],
    horizon: int = 12,
) -> Paths:
    """Project ``start`` (clients per state) through every sampled matrix for ``horizon`` months.

    ``branch_rates`` is the mean number of branch operations per client-month in each state, so
    branch load is the expected number of branch operations in the month. Paths stay paired by
    posterior sample, so a policy and its baseline can be differenced sample by sample.
    """
    total = start.sum()
    distribution = np.repeat(start[None, :], samples.shape[0], axis=0)
    rates = np.array([branch_rates.get(state, 0.0) for state in STATES])
    digital = np.array([state in DIGITAL_STATES for state in STATES], dtype=float)
    silent = np.array([state == "none" for state in STATES], dtype=float)
    steps: dict[str, list[FloatArray]] = {name: [] for name in QUANTITIES}
    for _ in range(horizon + 1):
        steps["branch_load"].append(distribution @ rates)
        steps["digital_share"].append(distribution @ digital / total)
        steps["silent_share"].append(distribution @ silent / total)
        distribution = np.einsum("si,sij->sj", distribution, samples)
    return {name: np.column_stack(values) for name, values in steps.items()}


def bands(paths: Paths) -> Bands:
    """5th, 50th and 95th percentiles of every quantity at every month."""
    months = next(iter(paths.values())).shape[1]
    quantiles = {
        name: {
            level: [float(value) for value in np.percentile(values, q, axis=0)]
            for level, q in (("p05", 5), ("p50", 50), ("p95", 95))
        }
        for name, values in paths.items()
    }
    return Bands(months=list(range(months)), quantiles=quantiles)


def simulate(
    samples: FloatArray,
    start: FloatArray,
    branch_rates: dict[str, float],
    horizon: int = 12,
) -> Bands:
    """Bands of :func:`simulate_paths` (what the API serves and the dashboard draws)."""
    return bands(simulate_paths(samples, start, branch_rates, horizon))


def branch_rates(client_months: pd.DataFrame) -> dict[str, float]:
    """Mean branch operations per client-month in each channel state (complete months only)."""
    complete = client_months[~client_months["is_partial_month"].astype(bool)]
    means = complete.groupby("channel_state")["n_branch"].mean()
    return {state: float(means.get(state, 0.0)) for state in STATES}
