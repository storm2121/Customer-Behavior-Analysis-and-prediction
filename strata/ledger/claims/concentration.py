"""Claims about concentration beyond card spend: recency, frequency and value per channel (M4)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from strata.analysis.concentration import gini, rfm_scores
from strata.ledger.bootstrap import weighted_quantile
from strata.ledger.claims._common import require
from strata.ledger.model import Unit
from strata.ledger.registry import ClaimContext, ClaimResult, claim

RFM_CHANNELS = {
    "card": ("card_recency_days", "n_card", "card_spend"),
    "online": ("online_recency_days", "n_online", "online_amount"),
    "branch": ("branch_recency_days", "n_branch", "branch_credit_amount"),
}

_UNITS = {
    "median_recency_days": Unit.DAYS,
    "median_frequency": Unit.NUMBER,
    "median_monetary": Unit.AMOUNT,
    "best_value_share": Unit.SHARE,
    "frequency_gini": Unit.NUMBER,
}


def _statistics(
    weights: np.ndarray,
    recency: np.ndarray,
    frequency: np.ndarray,
    monetary: np.ndarray,
    best: np.ndarray,
) -> dict[str, float]:
    total = float(np.sum(weights * monetary))
    share = float(np.sum(weights * monetary * best) / total) if total > 0 else float("nan")
    return {
        "median_recency_days": weighted_quantile(recency, weights, 0.5),
        "median_frequency": weighted_quantile(frequency, weights, 0.5),
        "median_monetary": weighted_quantile(monetary, weights, 0.5),
        "best_value_share": share,
        "frequency_gini": gini(frequency, weights),
    }


def _bound(
    recency: np.ndarray, frequency: np.ndarray, monetary: np.ndarray, best: np.ndarray
) -> Callable[[np.ndarray], dict[str, float]]:
    def statistics(weights: np.ndarray) -> dict[str, float]:
        return _statistics(weights, recency, frequency, monetary, best)

    return statistics


@claim(
    "rfm_by_channel",
    title="Recency, frequency and value in each channel",
    question=(
        "How recent, frequent and valuable is a typical user of each channel, and how much of a "
        "channel's value comes from its best clients?"
    ),
    method=(
        "For the clients active in each channel (client_360): median recency (days to the end of "
        "the window), median frequency and median monetary value; quintile scores from one to "
        "five; the share of the channel's value held by clients scoring at least four on all "
        "three; the Gini coefficient of frequency. Client-bootstrap intervals."
    ),
)
def rfm_by_channel(context: ClaimContext) -> ClaimResult:
    clients = context.table("marts.client_360")
    values: dict[str, float] = {}
    intervals: dict[str, tuple[float, float]] = {}
    units: dict[str, Unit] = {}
    value_n: dict[str, int] = {}
    for channel, (recency_col, frequency_col, monetary_col) in RFM_CHANNELS.items():
        active = clients[clients[frequency_col] > 0]
        if len(active) < 10:
            continue
        recency = active[recency_col].to_numpy(dtype=np.float64)
        frequency = active[frequency_col].to_numpy(dtype=np.float64)
        monetary = active[monetary_col].to_numpy(dtype=np.float64)
        best = (rfm_scores(recency, frequency, monetary) >= 4).all(axis=1)
        point = _statistics(np.ones(len(active)), recency, frequency, monetary, best)
        boot = context.bootstrap(len(active), stream=channel)
        drawn = boot.intervals(boot.apply(_bound(recency, frequency, monetary, best)))
        for name, value in point.items():
            key = f"{channel}_{name}"
            values[key] = value
            intervals[key] = drawn[name]
            value_n[key] = len(active)
            units[key] = _UNITS[name]
    require(bool(values), "no channel with enough active clients")
    return ClaimResult(
        values=values,
        units=units,
        ci=intervals,
        value_n=value_n,
        n=len(clients),
        caveats=[
            (
                "Monetary value is card spend, online amounts and branch credits respectively, so "
                "values are not comparable across channels."
            ),
            "Recency is measured to the end of the window; the card file ends a few days earlier.",
            "Intervals fix the quintile cut-offs, and so the best clients, at the sample values.",
        ],
    )
