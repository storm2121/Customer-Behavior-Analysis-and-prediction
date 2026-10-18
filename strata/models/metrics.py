"""Classification metrics with client-bootstrap intervals.

Every metric of a model card is a point estimate on the test clients with a 95% percentile interval
from resampling those clients with replacement, like every number of the ledger. Lift and
precision at k read the top ``k`` share of clients by score; ties at the cut are broken by order of
appearance after a stable sort, which is deterministic because test rows are in client order.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from sklearn.metrics import average_precision_score, roc_auc_score

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

TOP_SHARE = 0.10
CALIBRATION_BINS = 10
EPSILON = 1e-12


@dataclass(frozen=True)
class Metric:
    value: float
    ci: tuple[float, float]

    def as_dict(self) -> dict[str, float | list[float]]:
        return {"value": self.value, "ci": [self.ci[0], self.ci[1]]}


def _defined(y: IntArray) -> bool:
    return 0 < int(y.sum()) < len(y)


def auc(y: IntArray, p: FloatArray) -> float:
    return float(roc_auc_score(y, p)) if _defined(y) else float("nan")


def average_precision(y: IntArray, p: FloatArray) -> float:
    return float(average_precision_score(y, p)) if _defined(y) else float("nan")


def brier(y: IntArray, p: FloatArray) -> float:
    return float(np.mean((p - y) ** 2))


def log_loss(y: IntArray, p: FloatArray) -> float:
    clipped = np.clip(p, EPSILON, 1 - EPSILON)
    return float(-np.mean(y * np.log(clipped) + (1 - y) * np.log(1 - clipped)))


def _top(p: FloatArray, share: float) -> IntArray:
    count = max(1, round(share * len(p)))
    return np.argsort(-p, kind="stable")[:count]


def precision_at(y: IntArray, p: FloatArray, share: float = TOP_SHARE) -> float:
    return float(y[_top(p, share)].mean())


def lift_at(y: IntArray, p: FloatArray, share: float = TOP_SHARE) -> float:
    base = float(y.mean())
    return precision_at(y, p, share) / base if base > 0 else float("nan")


def calibration_error(y: IntArray, p: FloatArray, bins: int = CALIBRATION_BINS) -> float:
    """Expected calibration error, equal-width bins: the weighted mean |observed - predicted|."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        mask = index == b
        if mask.any():
            total += mask.mean() * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return total


def reliability(y: IntArray, p: FloatArray, bins: int = CALIBRATION_BINS) -> list[dict[str, float]]:
    """Mean predicted and observed rates per equal-width bin (the reliability diagram)."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    return [
        {
            "bin_low": float(edges[b]),
            "bin_high": float(edges[b + 1]),
            "n": int((index == b).sum()),
            "predicted": float(p[index == b].mean()),
            "observed": float(y[index == b].mean()),
        }
        for b in range(bins)
        if (index == b).any()
    ]


METRICS: dict[str, Callable[[IntArray, FloatArray], float]] = {
    "auc": auc,
    "average_precision": average_precision,
    "brier": brier,
    "log_loss": log_loss,
    "calibration_error": calibration_error,
    "precision_at_10": precision_at,
    "lift_at_10": lift_at,
}


def _interval(draws: list[float]) -> tuple[float, float]:
    finite = np.asarray([d for d in draws if np.isfinite(d)], dtype=np.float64)
    if finite.size == 0:
        return (float("nan"), float("nan"))
    low, high = np.percentile(finite, [2.5, 97.5])
    return (float(low), float(high))


def evaluate(
    y: IntArray,
    p: FloatArray,
    *,
    n_resamples: int = 1000,
    seed: int = 42,
    metrics: dict[str, Callable[[IntArray, FloatArray], float]] | None = None,
) -> dict[str, Metric]:
    """Every metric on ``(y, p)`` with a client-bootstrap 95% interval."""
    chosen = metrics or METRICS
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)
    rng = np.random.default_rng(seed)
    resamples = rng.integers(0, len(y), size=(n_resamples, len(y)))
    results: dict[str, Metric] = {}
    for name, function in chosen.items():
        draws = [function(y[rows], p[rows]) for rows in resamples]
        results[name] = Metric(value=function(y, p), ci=_interval(draws))
    return results


def paired_difference(
    y: IntArray,
    first: FloatArray,
    second: FloatArray,
    metric: Callable[[IntArray, FloatArray], float],
    *,
    n_resamples: int = 1000,
    seed: int = 42,
) -> Metric:
    """``metric(first) - metric(second)`` on the same resampled clients (a paired comparison)."""
    rng = np.random.default_rng(seed)
    resamples = rng.integers(0, len(y), size=(n_resamples, len(y)))
    draws = [metric(y[rows], first[rows]) - metric(y[rows], second[rows]) for rows in resamples]
    return Metric(value=metric(y, first) - metric(y, second), ci=_interval(draws))
