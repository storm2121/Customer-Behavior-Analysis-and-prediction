"""Claims on the quality of the trained models, published in ``metrics_only`` mode (M11).

They read what ``strata train`` wrote for the zone (the results file and the registry entry of the
model trained on this warehouse's data) instead of recomputing anything, so the published metrics
are exactly the ones in the model card of the same zone. The allowlist publishes only aggregate
metrics with their client-bootstrap intervals: no score, no reason, no feature importance.
"""

from __future__ import annotations

import math
from typing import Any

from strata.ledger.claims._common import require
from strata.ledger.model import Unit
from strata.ledger.registry import ClaimContext, ClaimError, ClaimResult, claim
from strata.models.intervals import STABILITY_INTERVAL

SERVED_SILENCE_MODEL = "lightgbm_platt"
SILENCE_METRICS = {
    "brier": Unit.NUMBER,
    "calibration_error": Unit.NUMBER,
    "lift_at_10": Unit.RATIO,
}
OVERLAP_CAVEAT = (
    "One year of data: the outcome windows of consecutive cutoffs overlap, so the test labels "
    "share calendar weeks with the training labels."
)


def _finite(value: object, what: str) -> float:
    """``value`` as a float, or a skipped claim saying what is missing."""
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        msg = f"no finite {what}"
        raise ClaimError(msg)
    return float(value)


def _metric(metrics: dict[str, Any], key: str) -> tuple[float, tuple[float, float] | None]:
    item = metrics.get(key) or {}
    value = _finite(item.get("value"), f"{key} for the served score")
    bounds = list(item.get("ci") or [])
    if len(bounds) == 2 and all(bound is not None for bound in bounds):
        return value, (_finite(bounds[0], key), _finite(bounds[1], key))
    return value, None


@claim(
    "silence_model_quality",
    title="How good the served silence score is on a later cutoff",
    question=(
        "Trained on one cutoff and tested on the next, how accurate, how well calibrated and how "
        "selective is the served silence-risk score?"
    ),
    method=(
        "The primary backtest of `strata train silence`: the served score (LightGBM with Platt "
        "scaling) is fitted on the clients of one cutoff and scored on those of the next, whose "
        "outcome it never saw. Brier score, expected calibration error and lift in the top tenth "
        "of scores, each with a client-bootstrap interval."
    ),
)
def silence_model_quality(context: ClaimContext) -> ClaimResult:
    results, entry = context.model_results("silence")
    backtests = results.get("backtests") or []
    require(backtests, "the silence model has no backtest")
    primary = backtests[-1]
    metrics = (primary.get("metrics") or {}).get(SERVED_SILENCE_MODEL) or {}
    values: dict[str, float] = {}
    ci: dict[str, tuple[float, float]] = {}
    for key in SILENCE_METRICS:
        values[key], interval = _metric(metrics, key)
        if interval is not None:
            ci[key] = interval
    n = int(primary["n_test"])
    return ClaimResult(
        values=values,
        units=dict(SILENCE_METRICS),
        n=n,
        ci=ci,
        value_n=dict.fromkeys(values, n),
        caveats=[
            OVERLAP_CAVEAT,
            "Silence is a quarter without any event, not the closing of an account.",
        ],
        meta={
            "model": SERVED_SILENCE_MODEL,
            "train_cutoff": str(primary["train_cutoff"]),
            "test_cutoff": str(primary["test_cutoff"]),
            "model_effort": str(entry.get("effort", "")),
        },
    )


@claim(
    "segment_stability",
    title="How stable the behavioural segments are",
    question=(
        "Refitted on bootstrap resamples of the clients, how closely do the segments reproduce "
        "the original assignment?"
    ),
    method=(
        "Gaussian mixtures with the number of components chosen by BIC (`strata train segments`); "
        "the adjusted Rand index between the original assignment and refits on client bootstrap "
        "resamples, its mean with the range of the middle ninety-five per cent of refits."
    ),
)
def segment_stability(context: ClaimContext) -> ClaimResult:
    results, entry = context.model_results("segments")
    stability = results.get("stability") or {}
    value = _finite(stability.get("mean_ari"), "segment stability")
    low = _finite(stability.get("ci_low"), "lower end of the stability range")
    high = _finite(stability.get("ci_high"), "upper end of the stability range")
    n = int(results["n_clients"])
    return ClaimResult(
        values={"mean_ari": value},
        units={"mean_ari": Unit.NUMBER},
        n=n,
        ci={"mean_ari": (low, high)},
        interval_label=STABILITY_INTERVAL,
        value_n={"mean_ari": n},
        caveats=[
            (
                "Stability measures how reproducible the assignment is, not whether the segments "
                "are meaningful; the number of segments is chosen on the same clients."
            )
        ],
        meta={"model_effort": str(entry.get("effort", ""))},
    )
