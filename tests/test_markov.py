"""The Markov chain of monthly channel states, its posterior and the what-if simulator."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from strata.analysis import markov

if TYPE_CHECKING:
    from strata.ledger.model import Ledger

TWO_STATES = np.array([[0.9, 0.1], [0.5, 0.5]])
STATE = markov.STATES.index
RATES = {"none": 0.0, "branch_only": 2.0, "digital_only": 0.0, "mixed": 1.0}
START = np.array([10.0, 30.0, 20.0, 40.0])

stochastic_matrices = arrays(np.float64, (4, 4), elements=st.floats(0.01, 1.0)).map(
    lambda weights: weights / weights.sum(axis=1, keepdims=True)
)


def test_transitions_skip_first_months_and_months_into_the_partial_stub() -> None:
    frame = pd.DataFrame(
        {
            "channel_state": ["mixed", "mixed", "none", "branch_only", "digital_only"],
            "previous_state": [None, "mixed", "mixed", "none", "branch_only"],
            "is_partial_month": [False, False, False, False, True],
        }
    )
    counts = markov.transition_counts(frame)
    assert counts.sum() == 3
    assert counts[STATE("mixed"), STATE("mixed")] == 1
    assert counts[STATE("mixed"), STATE("none")] == 1
    assert counts[STATE("none"), STATE("branch_only")] == 1


@settings(max_examples=30, deadline=None)
@given(arrays(np.float64, (4, 4), elements=st.integers(0, 50).map(float)), st.integers(0, 2**16))
def test_posterior_rows_are_probability_distributions(counts: np.ndarray, seed: int) -> None:
    samples = markov.posterior_samples(counts, n_samples=40, seed=seed)
    assert samples.shape == (40, 4, 4)
    assert np.all(samples >= 0)
    np.testing.assert_allclose(samples.sum(axis=2), 1.0)


def test_the_posterior_mean_is_the_dirichlet_mean() -> None:
    counts = np.array([[900.0, 100.0], [500.0, 500.0]])
    samples = markov.posterior_samples(counts, n_samples=4000, seed=1)
    expected = (counts + 1) / (counts + 1).sum(axis=1, keepdims=True)
    np.testing.assert_allclose(samples.mean(axis=0), expected, atol=0.005)


def test_posterior_samples_are_reproducible() -> None:
    counts = np.ones((4, 4))
    first = markov.posterior_samples(counts, n_samples=10, seed=3)
    np.testing.assert_array_equal(first, markov.posterior_samples(counts, n_samples=10, seed=3))


def test_the_stationary_distribution_of_a_known_chain() -> None:
    pi = markov.stationary(TWO_STATES)
    np.testing.assert_allclose(pi, [5 / 6, 1 / 6])


@settings(max_examples=40, deadline=None)
@given(stochastic_matrices)
def test_the_stationary_distribution_is_invariant(matrix: np.ndarray) -> None:
    pi = markov.stationary(matrix)
    assert pi.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(pi @ matrix, pi, atol=1e-9)


def test_mean_first_passage_times_of_a_known_chain() -> None:
    passage = markov.mean_first_passage(TWO_STATES)
    # Leaving state 0 is a geometric wait with success probability 0.1: ten months on average.
    assert passage[0, 1] == pytest.approx(10.0)
    assert passage[1, 0] == pytest.approx(2.0)
    # Mean recurrence times are the reciprocals of the stationary probabilities (Kac).
    assert passage[0, 0] == pytest.approx(6 / 5)
    assert passage[1, 1] == pytest.approx(6.0)


@settings(max_examples=40, deadline=None)
@given(stochastic_matrices)
def test_first_passage_times_solve_their_equations(matrix: np.ndarray) -> None:
    passage = markov.mean_first_passage(matrix)
    for target in range(matrix.shape[0]):
        elsewhere = passage[:, target].copy()
        elsewhere[target] = 0.0
        # One step, then the remaining time from wherever the step lands (unless on target).
        np.testing.assert_allclose(passage[:, target], 1.0 + matrix @ elsewhere, rtol=1e-6)


CLIENTS = pd.DataFrame({"client_id": ["a", "b", "c", "d", "e"], "age": [30, 40, 50, 60, None]})
MONTH = pd.DataFrame(
    {
        "client_id": ["a", "b", "c", "d", "e"],
        "channel_state": ["branch_only", "branch_only", "branch_only", "mixed", "branch_only"],
    }
)


def test_a_policy_moves_its_share_of_the_eligible_branch_only_clients() -> None:
    start = markov.initial_counts(MONTH, CLIENTS, markov.younger_than(45, 0.5))
    assert start[STATE("branch_only")] == pytest.approx(3.0)
    assert start[STATE("mixed")] == pytest.approx(2.0)
    assert start.sum() == pytest.approx(5.0)


def test_without_a_policy_the_month_is_kept_as_observed() -> None:
    start = markov.initial_counts(MONTH, CLIENTS, markov.NO_POLICY)
    np.testing.assert_array_equal(start, [0.0, 4.0, 0.0, 1.0])


def test_a_policy_can_move_clients_to_digital_only_use() -> None:
    policy = markov.younger_than(45, 1.0, target="digital_only")
    start = markov.initial_counts(MONTH, CLIENTS, policy)
    assert start[STATE("digital_only")] == pytest.approx(2.0)
    assert "digital only" in policy.description


def test_policies_are_validated() -> None:
    with pytest.raises(ValueError, match="proportion"):
        markov.younger_than(45, 1.5)
    with pytest.raises(ValueError, match="none, digital_only or mixed"):
        markov.younger_than(45, 0.3, target="branch_only")


def test_simulated_paths_start_from_the_observed_month() -> None:
    samples = markov.posterior_samples(np.full((4, 4), 20.0), n_samples=100, seed=0)
    paths = markov.simulate_paths(samples, START, RATES, horizon=6)
    assert paths["branch_load"].shape == (100, 7)
    np.testing.assert_allclose(paths["branch_load"][:, 0], 30 * 2.0 + 40 * 1.0)
    np.testing.assert_allclose(paths["digital_share"][:, 0], 0.6)
    np.testing.assert_allclose(paths["silent_share"][:, 0], 0.1)
    assert np.all(paths["digital_share"] + paths["silent_share"] <= 1.0 + 1e-12)


def test_a_one_off_move_fades_towards_the_stationary_mix() -> None:
    matrix = np.array(
        [
            [0.7, 0.1, 0.1, 0.1],
            [0.2, 0.6, 0.1, 0.1],
            [0.1, 0.0, 0.7, 0.2],
            [0.1, 0.1, 0.2, 0.6],
        ]
    )
    moved = START + np.array([0.0, -15.0, 0.0, 15.0])
    baseline = markov.simulate_paths(matrix[None], START, RATES, horizon=80)
    policy = markov.simulate_paths(matrix[None], moved, RATES, horizon=80)
    gap = np.abs(policy["digital_share"] - baseline["digital_share"])[0]
    assert gap[0] > 0.1
    assert gap[-1] < 1e-6
    np.testing.assert_allclose(
        baseline["digital_share"][0, -1], markov.stationary(matrix)[2:].sum(), atol=1e-6
    )


def test_bands_are_ordered_and_indexed_by_month() -> None:
    samples = markov.posterior_samples(np.full((4, 4), 20.0), n_samples=200, seed=0)
    result = markov.simulate(samples, START, RATES, horizon=12)
    assert result.months == list(range(13))
    for quantity in markov.QUANTITIES:
        levels = result.quantiles[quantity]
        for low, middle, high in zip(levels["p05"], levels["p50"], levels["p95"], strict=True):
            assert low <= middle <= high
    assert result.at("digital_share", 0) == pytest.approx(0.6)


def test_branch_rates_use_complete_months_only() -> None:
    frame = pd.DataFrame(
        {
            "channel_state": ["branch_only", "branch_only", "mixed", "none"],
            "n_branch": [2, 4, 1, 0],
            "is_partial_month": [False, True, False, False],
        }
    )
    assert markov.branch_rates(frame) == {
        "none": 0.0,
        "branch_only": 2.0,
        "digital_only": 0.0,
        "mixed": 1.0,
    }


def test_the_markov_claims_on_synthetic_data(synthetic_ledger: Ledger) -> None:
    chain = synthetic_ledger.get("channel_markov")
    assert chain.interval_label == "90% posterior interval"
    for origin in markov.STATES:
        row = sum(chain.values[f"p_{origin}_to_{state}"] for state in markov.STATES)
        assert row == pytest.approx(1.0, abs=1e-4)
    stationary = sum(chain.values[f"stationary_{state}"] for state in markov.STATES)
    assert stationary == pytest.approx(1.0, abs=1e-4)
    for key, (low, high) in chain.ci.items():
        assert low <= chain.values[key] <= high, key
    simulation = synthetic_ledger.get("markov_policy_simulation")
    gain = simulation.values["digital_share_gain_month_0"]
    assert gain > 0
    assert abs(simulation.values["digital_share_gain_month_12"]) < gain
    assert simulation.meta["effort"] == "quick"
