"""Time-based splits for client-level outcomes (ADR 0003).

Every cutoff of the ``silence_cutoffs`` seed defines one prediction problem: features from events
on or before the cutoff (``marts.features_silence``), the label from the three months after it. The
universe is every client with at least one event in the observation window. Backtests train on one
cutoff and test on the next, so a model is always evaluated on an origin later than the one it
learnt from; hyperparameters are tuned on the earliest cutoff only.

With one year of data the outcome windows of consecutive cutoffs overlap (they are three months
long and a month apart), so a backtest's training labels share some calendar weeks with its test
labels. The model cards state this limitation; no feature ever reads past its own cutoff.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from datetime import date

import numpy as np
import numpy.typing as npt
import pandas as pd

from strata.contracts.vocab import Cutoff, silence_cutoffs

FloatArray = npt.NDArray[np.float64]

NUMERIC_FEATURES = (
    "n_events",
    "n_card",
    "n_online",
    "n_branch",
    "n_events_30d",
    "n_events_90d",
    "active_months",
    "recency_days",
    "active_span_days",
    "card_spend",
    "online_amount",
    "branch_credit",
    "age",
    "tenure_years",
    "children",
)
"""Features read from ``features_silence``; each one is computed from events up to the cutoff."""

MISSABLE = ("age", "tenure_years", "children")
"""Client attributes a source may leave empty; each gets a missing-value flag."""

FLAG_FEATURES = ("profession_missing",)
PROFILES = ("branch_only", "digital_only", "mixed")
"""Channel profiles one-hot encoded (``none`` cannot occur: the universe has events)."""

SKEWED = (
    "n_events",
    "n_card",
    "n_online",
    "n_branch",
    "n_events_30d",
    "n_events_90d",
    "card_spend",
    "online_amount",
    "branch_credit",
)
"""Counts and amounts entered on a log scale (log1p) in the GLM, where scale matters."""


@dataclass(frozen=True)
class Backtest:
    """Train on one cutoff, test on a later one."""

    train: Cutoff
    test: Cutoff

    @property
    def name(self) -> str:
        return f"{self.train.cutoff.isoformat()}_to_{self.test.cutoff.isoformat()}"


def cutoffs() -> tuple[Cutoff, ...]:
    """The seed's cutoffs, earliest first."""
    return tuple(sorted(silence_cutoffs(), key=lambda cutoff: cutoff.cutoff))


def earliest() -> Cutoff:
    return cutoffs()[0]


def primary() -> Cutoff:
    return next(cutoff for cutoff in cutoffs() if cutoff.role == "primary")


def backtests() -> list[Backtest]:
    """Rolling origins: each cutoff trains the model tested on the next one."""
    ordered = cutoffs()
    return [Backtest(train, test) for train, test in itertools.pairwise(ordered)]


def rows_for(features: pd.DataFrame, cutoff: date) -> pd.DataFrame:
    """The universe of one cutoff, in a canonical client order."""
    rows = features[pd.to_datetime(features["cutoff_date"]).dt.date == cutoff]
    return rows.sort_values("client_id").reset_index(drop=True)


def design(rows: pd.DataFrame, *, log_skewed: bool = False) -> pd.DataFrame:
    """The model matrix of a cutoff's rows (missing numbers imputed with zero plus a flag)."""
    columns: dict[str, pd.Series] = {}
    for name in NUMERIC_FEATURES:
        values = pd.to_numeric(rows[name], errors="coerce").astype(float)
        if name in MISSABLE:
            columns[f"{name}_missing"] = values.isna().astype(float)
        values = values.fillna(0.0)
        columns[name] = (
            pd.Series(np.log1p(values.clip(lower=0)), index=values.index)
            if log_skewed and name in SKEWED
            else values
        )
    for name in FLAG_FEATURES:
        columns[name] = rows[name].astype(bool).astype(float)
    for profile in PROFILES[1:]:
        columns[f"profile_{profile}"] = (rows["channel_profile"] == profile).astype(float)
    return pd.DataFrame(columns, index=rows.index)


def labels(rows: pd.DataFrame) -> npt.NDArray[np.int64]:
    return np.asarray(rows["silent"].astype(bool), dtype=np.int64)
