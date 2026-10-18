"""Calendar effects on daily activity: a negative-binomial regression per series.

Each series is a daily count (card events, card withdrawals, card deposits, branch deposits,
online operations) from the dense ``daily_counts`` mart. The design (``design_matrix``) is shared
with the Bayesian model:

- day of week, Monday as reference (branch series use open days only, Monday to Friday);
- five-day bins of the month, days 11 to 15 as reference, so the first bin is the salary-day effect;
- calendar month in sum-to-zero (deviation) coding, so a month's effect is against the average
  month. With a single year, month of year and a linear trend cannot be separated; the monthly
  effects absorb the trend, and the docs say so;
- indicators for Ramadan, Eid al-Fitr, Eid al-Adha, the week before Eid al-Adha and the other
  bank holidays (national and religious; the Eids are bank holidays too, but have their own terms).

The dispersion is estimated once by maximum likelihood (NB2); intervals come from a client
bootstrap that rebuilds every daily series from resampled clients and refits the regression at the
fitted dispersion (iteratively reweighted least squares). Equations: ``docs/methods/calendar.md``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import NegativeBinomial

from strata.ledger.bootstrap import Bootstrap, percentile_interval

FloatArray = npt.NDArray[np.float64]

WEEKDAY_NAMES = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
REFERENCE_BIN = "11-15"
PERIOD_COLUMNS = {
    "ramadan": "is_ramadan",
    "eid_al_fitr": "is_eid_al_fitr",
    "eid_al_adha": "is_eid_al_adha",
    "pre_eid_al_adha_week": "is_pre_eid_al_adha_week",
    "bank_holiday": "is_bank_holiday",
}


@dataclass(frozen=True, slots=True)
class Series:
    """One daily series: which events it counts and on which days it is modelled."""

    key: str
    label: str
    channel: str
    kinds: tuple[str, ...] | None = None
    open_days_only: bool = False


SERIES = (
    Series("card_all", "all card events", "card"),
    Series("card_withdrawal", "card withdrawals", "card", ("withdrawal",)),
    Series("card_deposit", "card deposits", "card", ("deposit",)),
    Series("branch_deposit", "branch deposits", "branch", ("deposit",), open_days_only=True),
    Series("online_all", "all online operations", "online"),
)

CHANNEL_SERIES = tuple(
    Series(f"{channel}_all", f"all {channel} events", channel)
    for channel in ("card", "online", "branch")
)
"""One series per channel with every event on every day, as the STL decomposition needs."""


def daily_series(daily: pd.DataFrame, series: Series) -> pd.DataFrame:
    """Calendar columns plus the series' daily count, on the days the series is modelled."""
    rows = daily[daily["channel"] == series.channel]
    if series.kinds is not None:
        rows = rows[rows["kind"].isin(series.kinds)]
    calendar_columns = [
        "day",
        "day_of_week",
        "day_of_month_bin",
        "year_month",
        "in_coverage",
        *PERIOD_COLUMNS.values(),
    ]
    grouped = rows.groupby(calendar_columns, as_index=False)["n_events"].sum()
    frame = pd.DataFrame(grouped).sort_values(by="day")
    frame = frame[frame["in_coverage"].astype(bool)]
    if series.open_days_only:
        frame = frame[(frame["day_of_week"] <= 5) & ~frame["is_bank_holiday"].astype(bool)]
    return pd.DataFrame(frame.reset_index(drop=True))


def _indicator(mask: pd.Series) -> FloatArray:
    return np.asarray(mask, dtype=np.float64)


def design_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Columns of the regression; constant columns (an absent weekday, say) are dropped."""
    columns: dict[str, FloatArray] = {"intercept": np.ones(len(frame))}
    for day in range(2, 8):
        columns[f"dow_{WEEKDAY_NAMES[day - 1]}"] = _indicator(frame["day_of_week"] == day)
    for label in sorted(frame["day_of_month_bin"].unique()):
        if label != REFERENCE_BIN:
            columns[f"dom_{label.replace('-', '_')}"] = (
                frame["day_of_month_bin"] == label
            ).to_numpy(dtype=float)
    months = sorted(frame["year_month"].unique())
    last = months[-1]
    for month in months[:-1]:
        # Sum-to-zero coding: each month's coefficient is its deviation from the average month.
        indicator = (frame["year_month"] == month).to_numpy(dtype=float)
        columns[f"month_{month.replace('-', '_')}"] = indicator - (
            frame["year_month"] == last
        ).to_numpy(dtype=float)
    for name, column in PERIOD_COLUMNS.items():
        columns[name] = _indicator(frame[column].astype(bool))
    # The mart's bank holidays include the two Eids (branches close on both); the Eids have their
    # own terms, so the bank-holiday term covers the other bank holidays and every effect is read
    # against ordinary days.
    eid = frame["is_eid_al_fitr"].astype(bool) | frame["is_eid_al_adha"].astype(bool)
    columns["bank_holiday"] = _indicator(frame["is_bank_holiday"].astype(bool) & ~eid)
    design = pd.DataFrame(columns)
    varying = [name for name in design.columns if name == "intercept" or design[name].std() > 0]
    return design[varying]


def month_effect(coefficients: pd.Series, month: str, months: list[str]) -> float:
    """A month's log rate ratio against the average month, the last month included."""
    names = [f"month_{m.replace('-', '_')}" for m in months[:-1]]
    present = [name for name in names if name in coefficients.index]
    if month == months[-1]:
        return float(-coefficients[present].sum())
    return float(coefficients.get(f"month_{month.replace('-', '_')}", np.nan))


MIN_DISPERSION = 1e-8
LOG_LIMIT = 30.0
"""Rate ratios are reported on a bounded log scale: a rare indicator (one holiday in the window)
can have an enormous standard error, and an unbounded exponent would overflow."""


def _ratio(log_value: float) -> float:
    return float(np.exp(np.clip(log_value, -LOG_LIMIT, LOG_LIMIT)))


@dataclass(frozen=True)
class CalendarFit:
    """A fitted series: rate ratios with model-based and client-bootstrap intervals."""

    series: Series
    n_days: int
    alpha: float
    rate_ratios: dict[str, float]
    wald: dict[str, tuple[float, float]]
    bootstrap: dict[str, tuple[float, float]] = field(default_factory=dict)


def _effects(coefficients: pd.Series, months: list[str]) -> dict[str, float]:
    effects = {
        str(name): float(value) for name, value in coefficients.items() if name != "intercept"
    }
    if "2023-12" in months:
        effects["december"] = month_effect(coefficients, "2023-12", months)
    return {name: value for name, value in effects.items() if not name.startswith("month_")}


def fit_series(
    frame: pd.DataFrame,
    series: Series,
    *,
    matrices: FloatArray | None = None,
    boot: Bootstrap | None = None,
) -> CalendarFit:
    """Fit one series; with ``matrices`` (clients by days) and ``boot``, add bootstrap intervals."""
    design = design_matrix(frame)
    y = frame["n_events"].to_numpy(dtype=float)
    months = sorted(frame["year_month"].unique())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mle = NegativeBinomial(y, design.to_numpy(), loglike_method="nb2").fit(disp=0, maxiter=500)
    # A dispersion at the boundary means the series is close to Poisson; keep the family valid.
    alpha = max(float(mle.params[-1]), MIN_DISPERSION)
    coefficients = pd.Series(mle.params[:-1], index=design.columns)
    errors = pd.Series(mle.bse[:-1], index=design.columns)
    effects = _effects(coefficients, months)
    if "month_2023_12" in errors.index:
        errors["december"] = errors["month_2023_12"]
    wald = {
        name: (
            _ratio(effects[name] - 1.96 * errors[name]),
            _ratio(effects[name] + 1.96 * errors[name]),
        )
        for name in effects
        if name in errors.index and np.isfinite(errors[name])
    }
    rate_ratios = {name: _ratio(value) for name, value in effects.items()}
    intervals: dict[str, tuple[float, float]] = {}
    if matrices is not None and boot is not None:
        draws: dict[str, list[float]] = {name: [] for name in rate_ratios}
        family = sm.families.NegativeBinomial(alpha=alpha)
        totals = boot.weights @ matrices
        for resampled in totals:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                refit = sm.GLM(resampled, design.to_numpy(), family=family).fit()
            refit_effects = _effects(pd.Series(refit.params, index=design.columns), months)
            for name, collected in draws.items():
                collected.append(_ratio(refit_effects.get(name, np.nan)))
        intervals = {name: percentile_interval(values) for name, values in draws.items()}
    return CalendarFit(
        series=series,
        n_days=len(frame),
        alpha=alpha,
        rate_ratios=rate_ratios,
        wald=wald,
        bootstrap=intervals,
    )


def client_daily_matrix(
    events: pd.DataFrame, frame: pd.DataFrame, series: Series
) -> tuple[FloatArray, int]:
    """``(clients, modelled days)`` event counts of ``series``, aligned with ``frame``'s days."""
    # Same definition as the daily_counts mart: events without a category are not in any series.
    rows = events[(events["channel"] == series.channel) & events["kind_group"].notna()]
    if series.kinds is not None:
        rows = rows[rows["kind_group"].isin(series.kinds)]
    clients = pd.Index(pd.unique(rows["client_id"]))
    days = pd.Index(pd.to_datetime(frame["day"]))
    client_codes = clients.get_indexer(pd.Index(rows["client_id"]))
    day_codes = days.get_indexer(pd.Index(pd.to_datetime(rows["event_date"])))
    keep = day_codes >= 0
    matrix = np.zeros((len(clients), len(days)))
    np.add.at(matrix, (client_codes[keep], day_codes[keep]), 1.0)
    return matrix, len(clients)
