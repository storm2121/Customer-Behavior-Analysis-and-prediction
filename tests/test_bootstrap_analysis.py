"""The client bootstrap, Benford tests and concentration measures."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from strata.analysis import benford
from strata.analysis.concentration import gini, lorenz, top_share
from strata.ledger.bootstrap import (
    Bootstrap,
    claim_seed,
    percentile_interval,
    weighted_quantile,
)


def test_bootstrap_weights_are_multinomial_resamples() -> None:
    boot = Bootstrap(50, n_resamples=200, seed=3)
    assert boot.weights.shape == (200, 50)
    assert np.all(boot.weights.sum(axis=1) == 50)
    again = Bootstrap(50, n_resamples=200, seed=3)
    np.testing.assert_array_equal(boot.weights, again.weights)
    assert not np.array_equal(boot.weights, Bootstrap(50, n_resamples=200, seed=4).weights)


def test_bootstrap_needs_clients() -> None:
    with pytest.raises(ValueError, match="at least one client"):
        Bootstrap(0)


def test_ratio_intervals_cover_the_point_estimate() -> None:
    rng = np.random.default_rng(0)
    events = rng.integers(1, 20, size=300).astype(float)
    hits = rng.binomial(events.astype(int), 0.3).astype(float)
    boot = Bootstrap(300, n_resamples=500, seed=1)
    draws = boot.ratio(hits, events)
    low, high = percentile_interval(draws)
    point = hits.sum() / events.sum()
    assert low < point < high
    assert high - low < 0.1


def test_apply_and_intervals_for_nonlinear_statistics() -> None:
    values = np.arange(1, 101, dtype=float)
    boot = Bootstrap(100, n_resamples=300, seed=2)
    samples = boot.apply(lambda w: {"median": weighted_quantile(values, w, 0.5)})
    low, high = boot.intervals(samples)["median"]
    assert low <= 50 <= high


def test_percentile_interval_ignores_undefined_resamples() -> None:
    assert percentile_interval([np.nan, 1.0, 2.0, 3.0])[0] >= 1.0
    low, high = percentile_interval([np.nan, np.inf])
    assert np.isnan(low)
    assert np.isnan(high)


def test_weighted_quantile_handles_weights_and_empty_samples() -> None:
    assert weighted_quantile([1.0, 2.0, 3.0], [0, 0, 5], 0.5) == 3.0
    assert np.isnan(weighted_quantile([1.0], [0], 0.5))


def test_claim_seeds_are_stable_and_distinct() -> None:
    assert claim_seed(42, "channel_mix") == claim_seed(42, "channel_mix")
    assert claim_seed(42, "channel_mix") != claim_seed(42, "weekday_pattern")
    assert claim_seed(42, "x") != claim_seed(43, "x")


@pytest.mark.parametrize(
    ("amount", "first", "first_two"),
    [(1.0, 1, 10), (0.3, 3, 30), (1000.0, 1, 10), (9.99, 9, 99), (0.0123, 1, 12), (47.5, 4, 47)],
)
def test_significant_digits_are_exact(amount: float, first: int, first_two: int) -> None:
    assert benford.first_digits([amount])[0] == first
    assert benford.first_two_digits([amount])[0] == first_two


def test_non_positive_amounts_have_no_digits() -> None:
    assert benford.first_digits([0.0, -5.0, np.nan]).size == 0


def test_benford_expectations_sum_to_one() -> None:
    assert benford.EXPECTED_FIRST.sum() == pytest.approx(1.0)
    assert benford.EXPECTED_FIRST_TWO.sum() == pytest.approx(1.0)


def test_benford_distributed_amounts_conform() -> None:
    rng = np.random.default_rng(7)
    amounts = 10 ** rng.uniform(0, 5, size=20_000)
    first = benford.first_digit_test(amounts)
    assert first.conformity == "close conformity"
    assert first.p_value > 0.001
    assert benford.first_two_digit_test(amounts).mad < 0.0022


def test_uniform_leading_digits_do_not_conform() -> None:
    amounts = np.repeat(np.arange(1, 10, dtype=float), 500)
    test = benford.first_digit_test(amounts)
    assert test.conformity == "nonconformity"
    assert test.p_value < 1e-6


def test_conformity_bands_are_ordered() -> None:
    assert benford.conformity(0.004) == "close conformity"
    assert benford.conformity(0.010) == "acceptable conformity"
    assert benford.conformity(0.014) == "marginally acceptable conformity"
    assert benford.conformity(0.05) == "nonconformity"


def test_digit_tests_need_amounts() -> None:
    with pytest.raises(ValueError, match="no positive amounts"):
        benford.first_digit_test([])


def test_gini_extremes() -> None:
    assert gini(np.ones(100)) == pytest.approx(0.0, abs=1e-12)
    concentrated = np.zeros(100)
    concentrated[0] = 1.0
    assert gini(concentrated) == pytest.approx(0.99, abs=1e-9)
    assert np.isnan(gini(np.zeros(5)))


def test_weights_are_repetitions() -> None:
    values = np.array([1.0, 2.0, 5.0])
    weights = np.array([2.0, 1.0, 3.0])
    repeated = np.repeat(values, weights.astype(int))
    assert gini(values, weights) == pytest.approx(gini(repeated))
    assert top_share(values, 0.5, weights) == pytest.approx(top_share(repeated, 0.5))


def test_top_share_counts_whole_and_partial_clients() -> None:
    values = np.array([10.0, 30.0, 60.0])
    assert top_share(values, 1 / 3) == pytest.approx(0.6)
    assert top_share(values, 0.5) == pytest.approx((60 + 0.5 * 30) / 100)
    assert np.isnan(top_share(np.zeros(3), 0.1))


def test_lorenz_curve_runs_from_zero_to_one() -> None:
    population, share = lorenz([1.0, 2.0, 3.0])
    assert population[0] == share[0] == 0.0
    assert population[-1] == pytest.approx(1.0)
    assert share[-1] == pytest.approx(1.0)
    assert np.all(np.diff(share) >= 0)
    with pytest.raises(ValueError, match="positive"):
        lorenz([0.0, 0.0])


@settings(max_examples=40, deadline=None)
@given(st.lists(st.floats(min_value=0.01, max_value=1e6), min_size=2, max_size=200))
def test_gini_is_a_proportion(values: list[float]) -> None:
    coefficient = gini(np.array(values))
    assert -1e-9 <= coefficient < 1.0
