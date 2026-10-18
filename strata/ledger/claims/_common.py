"""Helpers shared by claim functions: keys, per-client matrices and vectorised share intervals."""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd

from strata.ledger.bootstrap import Bootstrap, percentile_interval
from strata.ledger.registry import ClaimError

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
"""ISO weekday order (Monday is 1)."""


def slug(label: str) -> str:
    """A template-safe key from a band label: ``<25`` → ``under_25``, ``65+`` → ``65_plus``."""
    text = label.strip().lower().replace("<", "under ").replace("+", " plus")
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def client_codes(ids: pd.Series, universe: pd.Series | None = None) -> tuple[IntArray, int]:
    """Integer code of each row's client within ``universe`` (default: the clients present)."""
    categories = pd.Index(pd.unique(universe if universe is not None else ids))
    codes = categories.get_indexer(pd.Index(ids))
    if (codes < 0).any():
        msg = "rows refer to clients outside the universe"
        raise ClaimError(msg)
    return codes.astype(np.int64), len(categories)


def per_client_matrix(
    codes: IntArray,
    columns: IntArray,
    n_clients: int,
    n_columns: int,
    values: npt.ArrayLike | None = None,
) -> FloatArray:
    """``(n_clients, n_columns)`` sums of ``values`` (default: counts) per client and column."""
    matrix = np.zeros((n_clients, n_columns), dtype=np.float64)
    weights = np.ones(len(codes)) if values is None else np.asarray(values, dtype=np.float64)
    np.add.at(matrix, (codes, columns), weights)
    return matrix


def require(condition: object, message: str) -> None:
    """Raise :class:`ClaimError` with ``message`` unless ``condition`` holds."""
    if not bool(condition):
        raise ClaimError(message)


def shares(
    boot: Bootstrap, matrix: FloatArray, keys: Sequence[str]
) -> tuple[dict[str, float], dict[str, tuple[float, float]]]:
    """Point shares of each column of ``matrix`` and their client-bootstrap intervals."""
    totals = matrix.sum(axis=1)
    grand = totals.sum()
    require(grand > 0, "no events to share")
    point = matrix.sum(axis=0) / grand
    draws = boot.ratio(matrix, totals[:, None])
    values = {key: float(point[i]) for i, key in enumerate(keys)}
    intervals = {key: percentile_interval(draws[:, i]) for i, key in enumerate(keys)}
    return values, intervals


def client_index(ids: pd.Series) -> tuple[IntArray, pd.Index]:
    """Position of each row's client among the distinct clients present, and those clients."""
    clients = pd.Index(pd.unique(ids))
    return clients.get_indexer(pd.Index(ids)).astype(np.int64), clients


def activity_tercile(n_events: pd.Series) -> pd.Series:
    """Equal-sized terciles of activity; ties are broken by position so the thirds are balanced."""
    ranks = n_events.rank(method="first")
    return pd.qcut(ranks, 3, labels=["low", "mid", "high"]).astype(str)
