"""Seasonality: weekly patterns by STL, the December deposit wave, and special-week ratios.

- ``weekly_strength`` decomposes a daily series with STL (period seven, robust) and measures the
  strength of the weekly component and of the trend: ``F = max(0, 1 - Var(R) / Var(X + R))`` where
  ``R`` is the remainder and ``X`` the component (Wang, Smith and Hyndman, 2006). A single year
  cannot carry a yearly STL period, so yearly effects are read from the calendar regression and
  from the decompositions below.
- ``decompose_weighted`` splits December's ratio to the average other month into breadth
  (distinct depositors) and depth (amount per depositor).
- ``weekday_standardised_ratio`` compares special days with ordinary days of the same weekday, so a
  period that happens to contain more Saturdays does not look busier.

Equations: ``docs/methods/seasonality.md``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from statsmodels.tsa.seasonal import STL

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

WEEK = 7


def _strength(remainder: FloatArray, component: FloatArray) -> float:
    total = np.var(component + remainder)
    if total <= 0:
        return 0.0
    return float(max(0.0, 1.0 - np.var(remainder) / total))


@dataclass(frozen=True, slots=True)
class Strength:
    weekly: float
    trend: float


def weekly_strength(series: npt.ArrayLike) -> Strength:
    """Strength of the weekly cycle and of the trend in a daily series."""
    values = np.asarray(series, dtype=np.float64)
    if values.size < 3 * WEEK:
        msg = "STL needs at least three weeks of days"
        raise ValueError(msg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = STL(values, period=WEEK, robust=True).fit()
    remainder = np.asarray(fit.resid, dtype=np.float64)
    return Strength(
        weekly=_strength(remainder, np.asarray(fit.seasonal, dtype=np.float64)),
        trend=_strength(remainder, np.asarray(fit.trend, dtype=np.float64)),
    )


def decompose_weighted(
    weights: FloatArray, counts: FloatArray, amounts: FloatArray, december: int, others: list[int]
) -> FloatArray:
    """December's count, breadth and depth ratios for each row of client ``weights``.

    ``counts`` and ``amounts`` are ``(clients, months)``; a client is a depositor in a month when
    they made at least one deposit in it.
    """
    count = weights @ counts
    breadth = weights @ (counts > 0).astype(np.float64)
    value = weights @ amounts
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = value / breadth
        ratios = np.column_stack(
            [
                count[:, december] / count[:, others].mean(axis=1),
                breadth[:, december] / breadth[:, others].mean(axis=1),
                depth[:, december] / np.nanmean(depth[:, others], axis=1),
            ]
        )
    return np.asarray(ratios, dtype=np.float64)


def weekday_standardised_ratio(
    totals: FloatArray, weekdays: npt.ArrayLike, special: BoolArray, baseline: BoolArray
) -> FloatArray:
    """Mean daily total on special days over the mean on baseline days of the same weekday.

    ``totals`` is ``(resamples, days)``; the per-weekday ratios are averaged with the number of
    special days on each weekday as weights.
    """
    weekday = np.asarray(weekdays)
    ratios = []
    weights = []
    for day in np.unique(weekday[special]):
        on_special = special & (weekday == day)
        on_baseline = baseline & (weekday == day)
        if on_baseline.sum() == 0:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios.append(totals[:, on_special].mean(axis=1) / totals[:, on_baseline].mean(axis=1))
        weights.append(on_special.sum())
    if not ratios:
        msg = "no weekday has both special and baseline days"
        raise ValueError(msg)
    stacked = np.column_stack(ratios)
    return np.asarray(stacked @ (np.array(weights) / np.sum(weights)), dtype=np.float64)


def monthly_matrix(
    codes: npt.ArrayLike,
    months: pd.Series,
    values: npt.ArrayLike | None,
    n_clients: int,
    order: list[str],
) -> FloatArray:
    """``(clients, months)`` sums of ``values`` (default: counts), months in ``order``."""
    index = months.map({month: i for i, month in enumerate(order)})
    keep = index.notna().to_numpy()
    matrix = np.zeros((n_clients, len(order)))
    weights = np.ones(len(index)) if values is None else np.asarray(values, dtype=np.float64)
    np.add.at(matrix, (np.asarray(codes)[keep], index[keep].astype(int).to_numpy()), weights[keep])
    return matrix
