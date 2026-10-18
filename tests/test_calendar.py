"""Calendar regression, its Bayesian twin and the seasonal decompositions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from strata.analysis import calendar as cal
from strata.analysis import seasonality
from strata.analysis.calendar_bayes import SamplerSettings, fit_bayes
from strata.analysis.seasonality import (
    decompose_weighted,
    monthly_matrix,
    weekday_standardised_ratio,
    weekly_strength,
)
from strata.config import Settings
from strata.holidays import is_bank_holiday
from strata.ledger.bootstrap import Bootstrap
from strata.warehouse import connect, read_relation

if TYPE_CHECKING:
    from strata.ledger.model import Ledger

PLANTED = {"dom_01_05": 0.4, "dow_sun": -0.5, "ramadan": 0.3, "december": 0.6}


@pytest.fixture(scope="module")
def marts(built_warehouse: Settings) -> dict[str, pd.DataFrame]:
    con = connect(built_warehouse.workspace().warehouse, read_only=True)
    try:
        return {
            "daily": read_relation(con, "marts.daily_counts"),
            "events": read_relation(con, "intermediate.int_events"),
        }
    finally:
        con.close()


@pytest.fixture(scope="module")
def planted(marts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """The synthetic calendar with counts drawn from a negative binomial with known effects."""
    frame = cal.daily_series(marts["daily"], cal.SERIES[0]).copy()
    design = cal.design_matrix(frame)
    december = (frame["year_month"] == "2023-12").to_numpy(dtype=float)
    log_mean = (
        np.log(200.0)
        + PLANTED["dom_01_05"] * design["dom_01_05"].to_numpy()
        + PLANTED["dow_sun"] * design["dow_sun"].to_numpy()
        + PLANTED["ramadan"] * design["ramadan"].to_numpy()
        + PLANTED["december"] * december
    )
    rng = np.random.default_rng(11)
    alpha = 0.02
    # NB2 as a gamma-Poisson mixture: variance mu + alpha mu^2.
    frame["n_events"] = rng.poisson(rng.gamma(1 / alpha, alpha * np.exp(log_mean)))
    return frame


def test_the_design_has_references_and_sum_coded_months(planted: pd.DataFrame) -> None:
    design = cal.design_matrix(planted)
    assert "intercept" in design.columns
    assert "dow_mon" not in design.columns
    assert "dom_11_15" not in design.columns
    months = sorted(planted["year_month"].unique())
    month_columns = [name for name in design.columns if name.startswith("month_")]
    assert len(month_columns) == len(months) - 1
    last = (planted["year_month"] == months[-1]).to_numpy()
    assert (design.loc[last, month_columns] == -1).all().all()
    constant = [name for name in design.columns if name != "intercept" and design[name].std() == 0]
    assert constant == []


def test_the_bank_holiday_term_leaves_the_eids_to_their_own_terms(planted: pd.DataFrame) -> None:
    design = cal.design_matrix(planted)
    eid = design["eid_al_fitr"].astype(bool) | design["eid_al_adha"].astype(bool)
    assert not (design.loc[eid, "bank_holiday"] > 0).any()
    assert planted.loc[eid.to_numpy(), "is_bank_holiday"].astype(bool).all()


def test_the_last_month_effect_is_minus_the_sum_of_the_others() -> None:
    coefficients = pd.Series({"month_2024_01": 0.2, "month_2024_02": -0.5})
    months = ["2024-01", "2024-02", "2024-03"]
    assert cal.month_effect(coefficients, "2024-03", months) == pytest.approx(0.3)
    assert cal.month_effect(coefficients, "2024-01", months) == pytest.approx(0.2)


def test_the_regression_recovers_planted_effects(planted: pd.DataFrame) -> None:
    fit = cal.fit_series(planted, cal.SERIES[0])
    for effect, log_ratio in PLANTED.items():
        if effect == "december":
            continue
        low, high = fit.wald[effect]
        standard_error = (np.log(high) - np.log(low)) / (2 * 1.96)
        assert abs(np.log(fit.rate_ratios[effect]) - log_ratio) < 3 * standard_error, effect
        assert standard_error < 0.1, effect
    # December is planted against the other months, the model reads it against the average month.
    assert fit.rate_ratios["december"] > np.exp(PLANTED["december"]) * 0.9
    assert fit.alpha == pytest.approx(0.02, rel=0.5)
    assert fit.n_days == len(planted)


def test_client_matrices_add_up_to_the_mart(marts: dict[str, pd.DataFrame]) -> None:
    for series in (*cal.SERIES, *cal.CHANNEL_SERIES):
        frame = cal.daily_series(marts["daily"], series)
        matrix, n_clients = cal.client_daily_matrix(marts["events"], frame, series)
        assert matrix.shape == (n_clients, len(frame))
        np.testing.assert_array_equal(matrix.sum(axis=0), frame["n_events"].to_numpy())


def test_branch_series_keep_open_days_only(marts: dict[str, pd.DataFrame]) -> None:
    series = next(s for s in cal.SERIES if s.open_days_only)
    frame = cal.daily_series(marts["daily"], series)
    assert frame["day_of_week"].max() <= 5
    assert not any(is_bank_holiday(day) for day in pd.to_datetime(frame["day"]).dt.date)


def test_the_mart_and_python_agree_on_bank_holidays(marts: dict[str, pd.DataFrame]) -> None:
    days = marts["daily"].drop_duplicates("day")
    python = [is_bank_holiday(day) for day in pd.to_datetime(days["day"]).dt.date]
    assert days["is_bank_holiday"].astype(bool).tolist() == python


def test_bootstrap_refits_give_intervals_around_the_estimate(
    marts: dict[str, pd.DataFrame],
) -> None:
    series = next(s for s in cal.SERIES if s.key == "card_withdrawal")
    frame = cal.daily_series(marts["daily"], series)
    matrix, n_clients = cal.client_daily_matrix(marts["events"], frame, series)
    boot = Bootstrap(n_clients, n_resamples=20, seed=4)
    fit = cal.fit_series(frame, series, matrices=matrix, boot=boot)
    low, high = fit.bootstrap["dom_01_05"]
    assert low <= fit.rate_ratios["dom_01_05"] <= high
    assert fit.bootstrap.keys() == fit.rate_ratios.keys()


def test_the_bayesian_model_recovers_planted_effects(planted: pd.DataFrame) -> None:
    sampler = SamplerSettings(chains=2, tune=200, draws=200)
    fit = fit_bayes(planted, cal.SERIES[0], sampler, seed=3)
    assert fit.rate_ratios["dom_01_05"] == pytest.approx(np.exp(PLANTED["dom_01_05"]), rel=0.1)
    low, high = fit.intervals["dom_01_05"]
    assert low < fit.rate_ratios["dom_01_05"] < high
    assert fit.probability_december_doubles is not None
    assert 0.0 <= fit.probability_december_doubles <= 1.0
    assert 0.75 <= fit.ppc_coverage_90 <= 1.0
    assert 0.0 <= fit.ppc_sd_p_value <= 1.0
    assert fit.rhat_max < 1.1
    assert fit.divergences >= 0
    assert "2 chains" in fit.settings.describe()


def test_weekly_strength_separates_cycles_from_noise() -> None:
    days = np.arange(364)
    rng = np.random.default_rng(0)
    weekly = 100 + 30 * np.sin(2 * np.pi * days / 7) + rng.normal(0, 3, days.size)
    noise = 100 + rng.normal(0, 3, days.size)
    trend = 50 + 0.5 * days + rng.normal(0, 3, days.size)
    assert weekly_strength(weekly).weekly > 0.9
    assert weekly_strength(noise).weekly < 0.4
    assert weekly_strength(trend).trend > 0.9


def test_a_flat_component_has_no_strength() -> None:
    assert seasonality._strength(np.zeros(10), np.zeros(10)) == 0.0
    assert seasonality._strength(np.arange(10.0), np.zeros(10)) == pytest.approx(0.0)


def test_weekly_strength_needs_three_weeks() -> None:
    with pytest.raises(ValueError, match="three weeks"):
        weekly_strength(np.ones(20))


def test_the_december_wave_splits_into_breadth_and_depth() -> None:
    counts = np.array([[1, 1, 2], [1, 1, 2], [0, 0, 1], [0, 0, 1]], dtype=float)
    amounts = counts * 100.0
    ratios = decompose_weighted(np.ones((1, 4)), counts, amounts, december=2, others=[0, 1])
    # Six deposits against two, by four depositors against two, of 150 each against 100.
    np.testing.assert_allclose(ratios, [[3.0, 2.0, 1.5]])


def test_special_days_are_compared_with_the_same_weekdays() -> None:
    weekdays = np.array([1, 2, 1, 2, 1, 2, 2])
    totals = np.array([[10.0, 20.0, 10.0, 20.0, 15.0, 30.0, 30.0]])
    special = np.array([0, 0, 0, 0, 1, 1, 1], dtype=bool)
    ratio = weekday_standardised_ratio(totals, weekdays, special, ~special)
    # Every special day is half as busy again as its weekday; the raw ratio would say more.
    np.testing.assert_allclose(ratio, [1.5])
    assert totals[0, special].mean() / totals[0, ~special].mean() > 1.5
    with pytest.raises(ValueError, match="weekday"):
        weekday_standardised_ratio(totals, weekdays, special, np.zeros(7, dtype=bool))


def test_monthly_matrices_sum_values_by_client_and_month() -> None:
    months = pd.Series(["2024-01", "2024-02", "2024-01", "2023-01"])
    matrix = monthly_matrix(
        np.array([0, 0, 1, 1]), months, [5.0, 7.0, 1.0, 9.0], 2, ["2024-01", "2024-02"]
    )
    np.testing.assert_array_equal(matrix, [[5.0, 7.0], [1.0, 0.0]])
    counts = monthly_matrix(np.array([0, 0, 1, 1]), months, None, 2, ["2024-01", "2024-02"])
    np.testing.assert_array_equal(counts, [[1.0, 1.0], [1.0, 0.0]])


def test_the_calendar_claims_on_synthetic_data(synthetic_ledger: Ledger) -> None:
    effects = synthetic_ledger.get("calendar_effects")
    # The generator plants salary-day withdrawals and a December deposit wave.
    assert effects.values["card_withdrawal_dom_01_05"] > 1.2
    assert effects.values["card_deposit_december"] > 2.0
    bayes = synthetic_ledger.get("calendar_bayes")
    assert bayes.interval_label == "95% credible interval"
    assert bayes.values["card_deposit_p_december_doubles"] > 0.5
    assert bayes.meta["sampler"].startswith("2 chains")
    weekly = synthetic_ledger.get("weekly_seasonality")
    assert weekly.values["branch_weekly_strength"] > 0.5
