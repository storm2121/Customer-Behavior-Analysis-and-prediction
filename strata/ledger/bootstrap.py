"""Client-level bootstrap.

Events of one client are not independent (a heavy user contributes hundreds of rows), so every
interval in the ledger resamples *clients* with replacement and recomputes the statistic from all
the events of the drawn clients. A resample is represented by multinomial weights (how many times
each client was drawn); statistics that are ratios of per-client sums then vectorise across all
resamples as one matrix product, and the rest loop over resamples with weighted formulas.

Intervals are percentile intervals at 95%, from 1,000 resamples by default (ARCHITECTURE section 7).
"""

from __future__ import annotations

import zlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

DEFAULT_RESAMPLES = 1000
ALPHA = 0.05


def claim_seed(seed: int, claim_id: str) -> int:
    """A seed per claim, so adding or reordering claims never changes another claim's interval."""
    return (seed * 1_000_003 + zlib.crc32(claim_id.encode("utf-8"))) % (2**32)


def percentile_interval(samples: npt.ArrayLike, alpha: float = ALPHA) -> tuple[float, float]:
    """Equal-tailed percentile interval, ignoring resamples where the statistic is undefined."""
    values = np.asarray(samples, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (float("nan"), float("nan"))
    low, high = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(low), float(high))


@dataclass(frozen=True)
class Bootstrap:
    """Multinomial client weights for ``n_resamples`` resamples of ``n_clients`` clients."""

    n_clients: int
    n_resamples: int = DEFAULT_RESAMPLES
    seed: int = 42
    weights: FloatArray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.n_clients < 1:
            msg = "a bootstrap needs at least one client"
            raise ValueError(msg)
        rng = np.random.default_rng(self.seed)
        draws = rng.multinomial(
            self.n_clients, np.full(self.n_clients, 1.0 / self.n_clients), size=self.n_resamples
        )
        object.__setattr__(self, "weights", draws.astype(np.float64))

    def ratio(self, numerators: npt.ArrayLike, denominators: npt.ArrayLike) -> FloatArray:
        """Resampled ratios of per-client sums: ``(W @ num) / (W @ den)``, one row per resample.

        ``numerators`` is ``(n_clients,)`` or ``(n_clients, k)``; ``denominators`` broadcasts.
        """
        num = np.asarray(numerators, dtype=np.float64)
        den = np.asarray(denominators, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.asarray((self.weights @ num) / (self.weights @ den), dtype=np.float64)

    def apply(
        self, statistic: Callable[[FloatArray], Mapping[str, float]]
    ) -> dict[str, FloatArray]:
        """Evaluate ``statistic(client_weights)`` on every resample (for non-linear statistics)."""
        collected: dict[str, list[float]] = {}
        for row in self.weights:
            for key, value in statistic(row).items():
                collected.setdefault(key, []).append(float(value))
        return {key: np.asarray(values, dtype=np.float64) for key, values in collected.items()}

    def intervals(self, samples: Mapping[str, npt.ArrayLike]) -> dict[str, tuple[float, float]]:
        return {key: percentile_interval(values) for key, values in samples.items()}


def weighted_quantile(values: npt.ArrayLike, weights: npt.ArrayLike, q: float) -> float:
    """Quantile of a weighted sample (weights are counts in a bootstrap resample)."""
    x = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    keep = w > 0
    x, w = x[keep], w[keep]
    if x.size == 0:
        return float("nan")
    order = np.argsort(x, kind="stable")
    x, w = x[order], w[order]
    cumulative = np.cumsum(w)
    target = q * cumulative[-1]
    return float(x[np.searchsorted(cumulative, target, side="left")])
