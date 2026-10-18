"""The robustness checker (a Simpson guard) for headline claims.

A headline compares a client-level metric between two groups, for example the share of clients who
go silent among branch-only and digital-only clients. Such a comparison can be produced entirely by
a third variable that differs between the groups. The checker recomputes the difference inside
strata of each stratifier (activity tercile, age band, tenure band by default) and combines them.

Definitions (ARCHITECTURE section 8, made precise here and in ``docs/methods/robustness.md``):

- crude effect: difference of the metric's mean, treatment minus reference;
- stratum effect: the same difference inside one stratum; a stratum is *sufficient* when both groups
  have at least ``min_group`` clients in it, and only sufficient strata enter the verdict;
- adjusted effect of a stratifier: the Mantel-Haenszel weighted average of the sufficient stratum
  effects, weights ``n1 * n0 / (n1 + n0)``;
- ``sign_flips``: sufficient strata whose effect has the opposite sign to the crude effect;
- ``shrinkage``: ``1 - min(stratum effect / crude effect)`` over sufficient strata;
- verdict: ``reverses`` if any adjusted effect has the opposite sign to the crude effect; otherwise
  ``holds`` if no sufficient stratum flips and shrinkage is below the threshold; otherwise
  ``weakens``; ``insufficient`` when no stratifier has a sufficient stratum.

Intervals come from the same client bootstrap as every other number in the ledger.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from kashf.ledger.bootstrap import DEFAULT_RESAMPLES, Bootstrap, FloatArray, percentile_interval
from kashf.ledger.model import Robustness, StratifierResult, Stratum, Verdict

DEFAULT_STRATA = ("activity_tercile", "age_band", "tenure_band")
STRATIFIER_LABELS = {
    "activity_tercile": "activity level (tercile of events)",
    "age_band": "age band",
    "tenure_band": "tenure band",
    "gender": "gender code",
}


@dataclass(frozen=True, slots=True)
class RobustnessSpec:
    """What a headline compares, and within which strata to check it."""

    group_col: str
    treatment: str
    reference: str
    metric: str
    strata: tuple[str, ...] = DEFAULT_STRATA
    min_group: int = 10
    shrinkage_threshold: float = 0.5
    metric_is_share: bool = True
    metric_label: str = ""
    treatment_label: str = ""
    reference_label: str = ""

    def label(self, which: str) -> str:
        """Readable names for the explanation text (fall back to the raw values)."""
        if which == "metric":
            return self.metric_label or self.metric
        if which == "treatment":
            return self.treatment_label or self.treatment.replace("_", "-")
        return self.reference_label or self.reference.replace("_", "-")


def _sign(value: float | None) -> int:
    if value is None or not np.isfinite(value) or value == 0:
        return 0
    return 1 if value > 0 else -1


def _difference(
    weights: FloatArray, y: FloatArray, treated: FloatArray, reference: FloatArray
) -> FloatArray:
    with np.errstate(divide="ignore", invalid="ignore"):
        treated_mean = (weights @ (y * treated)) / (weights @ treated)
        reference_mean = (weights @ (y * reference)) / (weights @ reference)
    return np.asarray(treated_mean - reference_mean, dtype=np.float64)


def _format(value: float | None, share: bool) -> str:
    if value is None or not np.isfinite(value):
        return "undefined"
    return f"{100 * value:+.1f} points" if share else f"{value:+.3g}"


@dataclass(frozen=True, slots=True)
class _Stratification:
    result: StratifierResult
    insufficient: list[str]


def _stratify(
    frame: pd.DataFrame,
    spec: RobustnessSpec,
    stratifier: str,
    boot: Bootstrap,
    crude_sign: int,
) -> _Stratification:
    y = np.asarray(frame[spec.metric], dtype=np.float64)
    treated = np.asarray(frame[spec.group_col] == spec.treatment, dtype=np.float64)
    reference = np.asarray(1.0 - treated, dtype=np.float64)
    ones = np.ones((1, len(frame)))
    strata: list[Stratum] = []
    insufficient: list[str] = []
    resampled_effects: list[FloatArray] = []
    resampled_weights: list[FloatArray] = []
    point_effects: list[float] = []
    point_weights: list[float] = []
    labels = frame[stratifier]
    for level in sorted(labels.dropna().unique(), key=str):
        inside = np.asarray(labels == level, dtype=np.float64)
        n1, n0 = int((treated * inside).sum()), int((reference * inside).sum())
        sufficient = n1 >= spec.min_group and n0 >= spec.min_group
        effect = None
        ci = None
        if n1 and n0:
            effect = float(_difference(ones, y, treated * inside, reference * inside)[0])
        if sufficient and effect is not None:
            draws = _difference(boot.weights, y, treated * inside, reference * inside)
            ci = percentile_interval(draws)
            w1 = boot.weights @ (treated * inside)
            w0 = boot.weights @ (reference * inside)
            with np.errstate(divide="ignore", invalid="ignore"):
                mh = np.where(w1 + w0 > 0, w1 * w0 / (w1 + w0), 0.0)
            resampled_effects.append(np.nan_to_num(draws, nan=0.0))
            resampled_weights.append(np.where(np.isfinite(draws), mh, 0.0))
            point_effects.append(effect)
            point_weights.append(n1 * n0 / (n1 + n0))
        else:
            insufficient.append(str(level))
        strata.append(
            Stratum(
                stratum=str(level),
                effect=effect,
                ci=ci,
                n_treatment=n1,
                n_reference=n0,
                sufficient=sufficient,
            )
        )
    adjusted = adjusted_ci = None
    if point_effects:
        adjusted = float(np.average(point_effects, weights=point_weights))
        effects, weights = np.vstack(resampled_effects), np.vstack(resampled_weights)
        with np.errstate(divide="ignore", invalid="ignore"):
            draws = (effects * weights).sum(axis=0) / weights.sum(axis=0)
        adjusted_ci = percentile_interval(draws)
    flips = sum(
        1
        for stratum in strata
        if stratum.sufficient and _sign(stratum.effect) not in (0, crude_sign)
    )
    reverses = crude_sign != 0 and _sign(adjusted) == -crude_sign
    result = StratifierResult(
        stratifier=stratifier,
        strata=strata,
        adjusted_effect=adjusted,
        adjusted_ci=adjusted_ci,
        sign_flips=flips,
        reverses=reverses,
    )
    return _Stratification(result=result, insufficient=insufficient)


def check(
    frame: pd.DataFrame,
    spec: RobustnessSpec,
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = 42,
    hint: str = "",
) -> Robustness:
    """Run the Simpson guard on a client-level ``frame`` (one row per client)."""
    compared = frame[frame[spec.group_col].isin([spec.treatment, spec.reference])].reset_index(
        drop=True
    )
    treated = np.asarray(compared[spec.group_col] == spec.treatment, dtype=np.float64)
    reference = np.asarray(1.0 - treated, dtype=np.float64)
    n_treatment, n_reference = int(treated.sum()), int(reference.sum())
    if n_treatment == 0 or n_reference == 0:
        msg = f"robustness check needs clients in both '{spec.treatment}' and '{spec.reference}'"
        raise ValueError(msg)
    y = np.asarray(compared[spec.metric], dtype=np.float64)
    boot = Bootstrap(len(compared), n_resamples=n_resamples, seed=seed)
    crude = float(_difference(np.ones((1, len(compared))), y, treated, reference)[0])
    crude_ci = percentile_interval(_difference(boot.weights, y, treated, reference))
    crude_sign = _sign(crude)

    stratifications = [
        _stratify(compared, spec, stratifier, boot, crude_sign)
        for stratifier in spec.strata
        if stratifier != spec.group_col
    ]
    results = [item.result for item in stratifications]
    sufficient_effects = [
        stratum.effect
        for result in results
        for stratum in result.strata
        if stratum.sufficient and stratum.effect is not None
    ]
    sign_flips = sum(result.sign_flips for result in results)
    shrinkage = None
    if crude_sign != 0 and sufficient_effects:
        shrinkage = float(1 - min(effect / crude for effect in sufficient_effects))

    verdict: Verdict
    if crude_sign == 0 or not sufficient_effects:
        verdict = "insufficient"
    elif any(result.reverses for result in results):
        verdict = "reverses"
    elif sign_flips == 0 and shrinkage is not None and shrinkage < spec.shrinkage_threshold:
        verdict = "holds"
    else:
        verdict = "weakens"

    return Robustness(
        group_col=spec.group_col,
        treatment=spec.treatment,
        reference=spec.reference,
        metric=spec.metric,
        overall_effect=crude,
        overall_ci=crude_ci,
        n_treatment=n_treatment,
        n_reference=n_reference,
        stratifiers=results,
        sign_flips=sign_flips,
        shrinkage=shrinkage,
        verdict=verdict,
        mechanism=_mechanism(spec, crude, stratifications, verdict, hint),
    )


def _mechanism(
    spec: RobustnessSpec,
    crude: float,
    stratifications: list[_Stratification],
    verdict: Verdict,
    hint: str,
) -> str:
    share = spec.metric_is_share
    opening = (
        f"Crude difference in the {spec.label('metric')} ({spec.label('treatment')} minus "
        f"{spec.label('reference')}): {_format(crude, share)}."
    )
    sentences = [opening]
    for item in stratifications:
        result = item.result
        label = STRATIFIER_LABELS.get(result.stratifier, result.stratifier)
        if result.adjusted_effect is None:
            sentences.append(f"No stratum of {label} has enough clients in both groups.")
            continue
        change = " The sign reverses." if result.reverses else ""
        sentences.append(
            f"Holding {label} fixed, the adjusted difference is "
            f"{_format(result.adjusted_effect, share)}.{change}"
        )
        if item.insufficient:
            sentences.append(
                f"Strata of {label} without enough clients of one group to compare: "
                f"{', '.join(item.insufficient)}."
            )
    sentences.append(f"Verdict: {verdict}.")
    if hint:
        sentences.append(hint)
    return " ".join(sentences)
