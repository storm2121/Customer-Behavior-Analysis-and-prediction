"""Claims about when clients transact: weekdays, salary days, December, Ramadan and Eid.

Daily statistics are bootstrapped by client: each client's events are laid out as a row of daily
counts, a resample reweights the rows, and the statistic is recomputed from the weighted daily
totals. The calendar models of M4 refine these simple ratios with a full regression.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from strata.contracts.specs import WINDOW_END, WINDOW_START
from strata.holidays import (
    EID_AL_ADHA_2024,
    EID_AL_FITR_2024,
    PRE_EID_AL_ADHA_WEEK_2024,
    RAMADAN_2024,
    is_bank_holiday,
)
from strata.ledger.bootstrap import percentile_interval
from strata.ledger.claims._common import WEEKDAYS, FloatArray, per_client_matrix, require, shares
from strata.ledger.model import Unit
from strata.ledger.registry import ClaimContext, ClaimResult, claim

DAYS = pd.date_range(WINDOW_START, WINDOW_END, freq="D")
CARD_END = date(2024, 9, 29)


def _events(context: ClaimContext, channel: str) -> pd.DataFrame:
    events = context.table("intermediate.int_events")
    selected = events[(events["channel"] == channel) & ~events["is_repeat"]].copy()
    selected["event_date"] = pd.to_datetime(selected["event_date"])
    return selected.reset_index(drop=True)


def _daily_matrix(events: pd.DataFrame, values: str | None = None) -> tuple[FloatArray, int]:
    """``(clients, days)`` matrix of event counts (or summed ``values``) over the window."""
    clients = pd.Index(pd.unique(events["client_id"]))
    codes = clients.get_indexer(pd.Index(events["client_id"])).astype(np.int64)
    days = np.asarray((events["event_date"] - pd.Timestamp(WINDOW_START)).dt.days, dtype=np.int64)
    weights = None if values is None else events[values].fillna(0).to_numpy(dtype=np.float64)
    return per_client_matrix(codes, days, len(clients), len(DAYS), weights), len(clients)


@claim(
    "weekday_pattern",
    title="The week of a retail bank",
    question="How are card, online and branch events spread over the days of the week?",
    method=(
        "Share of each channel's events falling on each ISO weekday (int_events, exact repeats "
        "excluded); client bootstrap intervals."
    ),
)
def weekday_pattern(context: ClaimContext) -> ClaimResult:
    values: dict[str, float] = {}
    intervals: dict[str, tuple[float, float]] = {}
    value_n: dict[str, int] = {}
    total_events = 0
    clients: set[str] = set()
    for channel in ("card", "online", "branch"):
        events = _events(context, channel)
        if events.empty:
            continue
        codes = pd.Index(pd.unique(events["client_id"])).get_indexer(pd.Index(events["client_id"]))
        n_clients = int(codes.max()) + 1
        weekday = np.asarray(events["event_date"].dt.dayofweek, dtype=np.int64)
        matrix = per_client_matrix(codes.astype(np.int64), weekday, n_clients, 7)
        boot = context.bootstrap(n_clients, stream=channel)
        channel_values, channel_intervals = shares(
            boot, matrix, [f"{channel}_{d}" for d in WEEKDAYS]
        )
        values.update(channel_values)
        intervals.update(channel_intervals)
        value_n.update(dict.fromkeys(channel_values, n_clients))
        total_events += len(events)
        clients.update(events["client_id"])
    require("card_mon" in values, "no card events")
    card = [values[f"card_{d}"] for d in WEEKDAYS]
    values["card_peak_to_trough"] = max(card) / min(card)
    value_n["card_peak_to_trough"] = value_n["card_mon"]
    units = {key: Unit.RATIO if key.endswith("peak_to_trough") else Unit.SHARE for key in values}
    names = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
    return ClaimResult(
        values=values,
        units=units,
        ci=intervals,
        value_n=value_n,
        n=len(clients),
        n_events=total_events,
        meta={
            "rows_weekdays": "|".join(
                f"{name}:{key}" for name, key in zip(names, WEEKDAYS, strict=True)
            )
        },
        caveats=[
            (
                "Dates may be posting dates rather than transaction dates, so weekend activity can "
                "surface on the next working day."
            ),
            (
                "Branches are closed at weekends; branch shares on those days show where postings "
                "land."
            ),
        ],
    )


def _months(dates: pd.Series) -> pd.Series:
    return pd.Series(dates.dt.strftime("%Y-%m"), index=dates.index, dtype="str")


@claim(
    "december_deposit_wave",
    title="The December deposit wave on cards",
    question="How much larger is December's card deposit activity than a normal month, and why?",
    method=(
        "Card deposits per complete month (int_events, the stub of the last month dropped): "
        "December's count against the mean of the other months, decomposed into breadth (distinct "
        "depositors) and depth (amount per depositor); client bootstrap intervals."
    ),
)
def december_deposit_wave(context: ClaimContext) -> ClaimResult:
    events = _events(context, "card")
    deposits = events[events["kind"] == "deposit"].copy()
    require(len(deposits) > 0, "no card deposits")
    deposits["month"] = _months(deposits["event_date"])
    complete = sorted(m for m in deposits["month"].unique() if m != WINDOW_END.strftime("%Y-%m"))
    require("2023-12" in complete and len(complete) > 2, "December and other months are needed")
    clients = pd.Index(pd.unique(events["client_id"]))
    codes = clients.get_indexer(pd.Index(deposits["client_id"])).astype(np.int64)
    month_index = deposits["month"].map({m: i for i, m in enumerate(complete)})
    keep = month_index.notna().to_numpy()
    columns = np.asarray(month_index[keep], dtype=np.int64)
    counts = per_client_matrix(codes[keep], columns, len(clients), len(complete))
    amounts = per_client_matrix(
        codes[keep],
        columns,
        len(clients),
        len(complete),
        deposits["amount"].fillna(0).to_numpy()[keep],
    )
    december = complete.index("2023-12")
    others = [i for i in range(len(complete)) if i != december]
    boot = context.bootstrap(len(clients))

    def decompose(weights: np.ndarray) -> np.ndarray:
        """Count, breadth and depth ratios for each row of ``weights``."""
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

    point = decompose(np.ones((1, len(clients))))[0]
    draws = decompose(boot.weights)
    keys = ("count_ratio", "breadth_ratio", "depth_ratio")
    values = {key: float(point[i]) for i, key in enumerate(keys)}
    intervals = {key: percentile_interval(draws[:, i]) for i, key in enumerate(keys)}
    values["december_depositors"] = float((counts[:, december] > 0).sum())
    units = dict.fromkeys(keys, Unit.RATIO)
    units["december_depositors"] = Unit.COUNT
    return ClaimResult(
        values=values,
        units=units,
        ci=intervals,
        value_n=dict.fromkeys(values, len(clients)),
        n=len(clients),
        n_events=len(deposits),
        caveats=[
            "One December in the window: the wave cannot be told apart from a one-off event.",
            "Card deposits are a small category; a few clients move the ratios.",
        ],
    )


@claim(
    "salary_day_withdrawals",
    title="Salary days at the cash machine",
    question=(
        "Do cash withdrawals concentrate at the start of the month, and which amounts dominate?"
    ),
    method=(
        "Share of card withdrawal value on days 1 to 5 of the month, against the share of calendar "
        "days that are days 1 to 5 within card coverage (int_events); the two most frequent "
        "withdrawal amounts; client bootstrap intervals."
    ),
)
def salary_day_withdrawals(context: ClaimContext) -> ClaimResult:
    events = _events(context, "card")
    withdrawals = (
        events[events["kind"] == "withdrawal"].dropna(subset=["amount"]).reset_index(drop=True)
    )
    require(len(withdrawals) > 0, "no withdrawals")
    clients = pd.Index(pd.unique(withdrawals["client_id"]))
    codes = clients.get_indexer(pd.Index(withdrawals["client_id"])).astype(np.int64)
    early = (withdrawals["event_date"].dt.day <= 5).to_numpy(dtype=np.float64)
    amount = withdrawals["amount"].to_numpy(dtype=np.float64)
    early_value = np.bincount(codes, weights=amount * early, minlength=len(clients))
    all_value = np.bincount(codes, weights=amount, minlength=len(clients))
    boot = context.bootstrap(len(clients))
    coverage = DAYS[DAYS.date <= CARD_END]
    expected = float((coverage.day <= 5).mean())
    observed = float(early_value.sum() / all_value.sum())
    draws = boot.ratio(early_value, all_value)
    top = withdrawals["amount"].value_counts()
    values = {
        "early_month_value_share": observed,
        "uniform_expectation": expected,
        "early_month_ratio": observed / expected,
        "most_common_amount": float(top.index[0]),
        "second_most_common_amount": float(top.index[1]) if len(top) > 1 else float(top.index[0]),
    }
    intervals = {
        "early_month_value_share": percentile_interval(draws),
        "early_month_ratio": percentile_interval(draws / expected),
    }
    units = {
        "early_month_value_share": Unit.SHARE,
        "uniform_expectation": Unit.SHARE,
        "early_month_ratio": Unit.RATIO,
        "most_common_amount": Unit.AMOUNT,
        "second_most_common_amount": Unit.AMOUNT,
    }
    return ClaimResult(
        values=values,
        units=units,
        ci=intervals,
        value_n=dict.fromkeys(values, len(clients)),
        n=len(clients),
        n_events=len(withdrawals),
        caveats=["Salary dates are not in the data; days 1 to 5 are a proxy for pay day."],
    )


def _ratio_by_weekday(totals: np.ndarray, special: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """Day-of-week standardised ratio of mean daily totals, special days against baseline days.

    ``totals`` is ``(resamples, days)``; the weights are the number of special days per weekday.
    """
    weekday = DAYS.dayofweek.to_numpy()
    ratios = []
    weights = []
    for day in range(7):
        on_special = special & (weekday == day)
        on_baseline = baseline & (weekday == day)
        if on_special.sum() == 0 or on_baseline.sum() == 0:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios.append(totals[:, on_special].mean(axis=1) / totals[:, on_baseline].mean(axis=1))
        weights.append(on_special.sum())
    require(bool(ratios), "no comparable weekdays")
    stacked = np.column_stack(ratios)
    return np.asarray(stacked @ (np.array(weights) / np.sum(weights)), dtype=np.float64)


def _in(period_start: date, period_end: date) -> np.ndarray:
    return np.array([period_start <= day <= period_end for day in DAYS.date])


@claim(
    "ramadan_card_effect",
    title="Card activity during Ramadan",
    question="Do clients use their cards more or less during Ramadan than on comparable days?",
    method=(
        "Daily card events (int_events, repeats excluded) during Ramadan against the other days of "
        "card coverage, Eid days excluded, as a day-of-week standardised ratio of daily means; "
        "client "
        "bootstrap interval. The calendar regression of M4 adjusts for trend and month as well."
    ),
)
def ramadan_card_effect(context: ClaimContext) -> ClaimResult:
    events = _events(context, "card")
    require(len(events) > 0, "no card events")
    matrix, n_clients = _daily_matrix(events)
    coverage = DAYS.date <= CARD_END
    eids = _in(EID_AL_FITR_2024.start, EID_AL_FITR_2024.end) | _in(
        EID_AL_ADHA_2024.start, EID_AL_ADHA_2024.end
    )
    ramadan = _in(RAMADAN_2024.start, RAMADAN_2024.end) & coverage
    baseline = coverage & ~ramadan & ~eids
    boot = context.bootstrap(n_clients)
    point = float(_ratio_by_weekday(matrix.sum(axis=0)[None, :], ramadan, baseline)[0])
    draws = _ratio_by_weekday(boot.weights @ matrix, ramadan, baseline)
    totals = matrix.sum(axis=0)
    values = {
        "ramadan_ratio": point,
        "ramadan_daily_mean": float(totals[ramadan].mean()),
        "other_daily_mean": float(totals[baseline].mean()),
    }
    return ClaimResult(
        values=values,
        units={
            "ramadan_ratio": Unit.RATIO,
            "ramadan_daily_mean": Unit.NUMBER,
            "other_daily_mean": Unit.NUMBER,
        },
        ci={"ramadan_ratio": percentile_interval(draws)},
        value_n=dict.fromkeys(values, n_clients),
        n=n_clients,
        n_events=len(events),
        caveats=[
            (
                "One Ramadan in the window; a ratio close to one means no detectable change, not "
                "proof "
                "of none."
            ),
            "Ramadan dates follow the moon sighting and may differ by a day.",
        ],
    )


@claim(
    "pre_eid_branch_deposits",
    title="Branch deposits in the week before Eid al-Adha",
    question="Do branch deposits rise in the week before Eid al-Adha?",
    method=(
        "Daily branch deposit operations (int_events, deposit category) in the week before Eid "
        "al-Adha against other open weekdays (weekends, bank holidays and Eid days excluded), as a "
        "day-of-week standardised ratio of daily means, for counts and for amounts; client "
        "bootstrap intervals."
    ),
)
def pre_eid_branch_deposits(context: ClaimContext) -> ClaimResult:
    events = _events(context, "branch")
    deposits = events[events["kind_group"] == "deposit"].reset_index(drop=True)
    require(len(deposits) > 0, "no branch deposits")
    counts, n_clients = _daily_matrix(deposits)
    amounts, _ = _daily_matrix(deposits, values="amount")
    weekday = DAYS.dayofweek.to_numpy()
    open_days = (weekday < 5) & ~np.array([is_bank_holiday(day) for day in DAYS.date])
    pre_eid = _in(PRE_EID_AL_ADHA_WEEK_2024.start, PRE_EID_AL_ADHA_WEEK_2024.end) & open_days
    baseline = open_days & ~pre_eid
    boot = context.bootstrap(n_clients)
    values: dict[str, float] = {}
    intervals: dict[str, tuple[float, float]] = {}
    for name, matrix in (("count_ratio", counts), ("amount_ratio", amounts)):
        values[name] = float(_ratio_by_weekday(matrix.sum(axis=0)[None, :], pre_eid, baseline)[0])
        intervals[name] = percentile_interval(
            _ratio_by_weekday(boot.weights @ matrix, pre_eid, baseline)
        )
    return ClaimResult(
        values=values,
        units=dict.fromkeys(values, Unit.RATIO),
        ci=intervals,
        value_n=dict.fromkeys(values, n_clients),
        n=n_clients,
        n_events=len(deposits),
        caveats=[
            "One Eid al-Adha in the window; the week before it is five working days.",
            "Large deposits weigh heavily on the amount ratio.",
        ],
    )
