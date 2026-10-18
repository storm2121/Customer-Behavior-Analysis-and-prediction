"""Concentration of spend: Lorenz curve, Gini coefficient and top-k shares.

All functions accept optional weights so that the bootstrap can evaluate them on a resample without
materialising repeated rows (a weight is the number of times a client was drawn).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def _prepared(
    values: npt.ArrayLike, weights: npt.ArrayLike | None, *, descending: bool
) -> tuple[FloatArray, FloatArray]:
    x = np.asarray(values, dtype=np.float64)
    w = np.ones_like(x) if weights is None else np.asarray(weights, dtype=np.float64)
    keep = (w > 0) & np.isfinite(x)
    x, w = x[keep], w[keep]
    order = np.argsort(-x if descending else x, kind="stable")
    return x[order], w[order]


def lorenz(
    values: npt.ArrayLike, weights: npt.ArrayLike | None = None
) -> tuple[FloatArray, FloatArray]:
    """Cumulative population share and cumulative value share, from the smallest value up."""
    x, w = _prepared(values, weights, descending=False)
    if x.size == 0 or x.sum() <= 0:
        msg = "a Lorenz curve needs positive values"
        raise ValueError(msg)
    population = np.concatenate([[0.0], np.cumsum(w) / w.sum()])
    share = np.concatenate([[0.0], np.cumsum(x * w) / np.sum(x * w)])
    return population, share


def gini(values: npt.ArrayLike, weights: npt.ArrayLike | None = None) -> float:
    """Gini coefficient: one minus twice the area under the Lorenz curve."""
    try:
        population, share = lorenz(values, weights)
    except ValueError:
        return float("nan")
    area = np.sum((population[1:] - population[:-1]) * (share[1:] + share[:-1]) / 2)
    return float(1 - 2 * area)


def top_share(
    values: npt.ArrayLike, fraction: float, weights: npt.ArrayLike | None = None
) -> float:
    """Share of the total held by the top ``fraction`` of the (weighted) population."""
    x, w = _prepared(values, weights, descending=True)
    if x.size == 0 or np.sum(x * w) <= 0:
        return float("nan")
    cutoff = fraction * w.sum()
    cumulative = np.cumsum(w)
    # Whole clients up to the cutoff, and the fraction of the client who crosses it.
    taken = np.minimum(w, np.maximum(cutoff - (cumulative - w), 0.0))
    return float(np.sum(taken * x) / np.sum(w * x))
