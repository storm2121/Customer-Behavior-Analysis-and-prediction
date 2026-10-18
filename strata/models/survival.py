"""Survival views of activity: time to the next event and to silence (ARCHITECTURE section 10).

- **Gaps.** Every client's distinct event days give recurrent gaps: the days from one active day to
  the next. The last gap of a client is censored at the end of the window.
- **Kaplan-Meier** curves of the gap by channel profile and by age band, with medians whose 95%
  intervals come from a client bootstrap (gaps of one client are not independent, so Greenwood's
  formula would be too narrow), and log-rank tests.
- **Cox proportional hazards** of the gap with cluster-robust standard errors by client, and the
  scaled Schoenfeld residual test of proportional hazards for every covariate.
- **Time to silence**: from a client's first event to the start of the first gap of at least ninety
  days, censored when ninety days of observation no longer fit in the window.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats

from strata.contracts.specs import WINDOW_END

FloatArray = npt.NDArray[np.float64]

SILENCE_DAYS = 90


def gaps(events: pd.DataFrame, end: date = WINDOW_END) -> pd.DataFrame:
    """``client_id, duration, observed`` for every gap between a client's distinct active days."""
    days = (
        events[["client_id", "event_date"]]
        .assign(event_date=lambda frame: pd.to_datetime(frame["event_date"]))
        .drop_duplicates()
        .sort_values(["client_id", "event_date"])
    )
    following = days.groupby("client_id")["event_date"].shift(-1)
    observed = following.notna()
    finish = following.where(observed, pd.Timestamp(end))
    duration = (finish - days["event_date"]).dt.days.astype(float)
    frame = pd.DataFrame(
        {
            "client_id": days["client_id"].to_numpy(),
            "duration": duration.to_numpy(),
            "observed": observed.to_numpy(dtype=bool),
        }
    )
    # A last active day on the window's final day leaves nothing to observe.
    return frame[frame["duration"] > 0].reset_index(drop=True)


def time_to_silence(events: pd.DataFrame, end: date = WINDOW_END) -> pd.DataFrame:
    """``client_id, duration, observed``: days from the first event to the first long gap."""
    days = (
        events[["client_id", "event_date"]]
        .assign(event_date=lambda f: pd.to_datetime(f["event_date"]))
        .drop_duplicates()
        .sort_values(["client_id", "event_date"])
        .reset_index(drop=True)
    )
    first = days.groupby("client_id")["event_date"].min()
    horizon = pd.Timestamp(end) - pd.Timedelta(days=SILENCE_DAYS)
    rows = []
    starts = days.assign(gap=days.groupby("client_id")["event_date"].shift(-1) - days["event_date"])
    starts["gap"] = starts["gap"].fillna(pd.Timestamp(end) - starts["event_date"])
    long_gaps = starts[starts["gap"] >= pd.Timedelta(days=SILENCE_DAYS)]
    first_silence = long_gaps.groupby("client_id")["event_date"].min()
    for client, began in first.items():
        silence = first_silence.get(client)
        if silence is not None and silence <= horizon:
            rows.append((client, float((silence - began).days), True))
        else:
            rows.append((client, float(max((horizon - began).days, 0)), False))
    out = pd.DataFrame(rows, columns=["client_id", "duration", "observed"])
    return out[out["duration"] > 0].reset_index(drop=True)


def km_curve(
    durations: FloatArray, observed: npt.NDArray[np.bool_], weights: FloatArray | None = None
) -> tuple[FloatArray, FloatArray]:
    """Weighted Kaplan-Meier: event times and the survival function just after each."""
    w = np.ones_like(durations) if weights is None else weights
    if durations.size == 0:
        return np.empty(0), np.empty(0)
    order = np.argsort(durations, kind="stable")
    t, e, w = durations[order], observed[order], w[order]
    unique, first = np.unique(t, return_index=True)
    cumulative = np.concatenate([[0.0], np.cumsum(w)])
    at_risk = cumulative[-1] - cumulative[first]
    deaths = np.add.reduceat(w * e, first)
    keep = deaths > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        survival = np.cumprod(1 - deaths[keep] / at_risk[keep])
    return unique[keep], survival


def km_median(
    durations: FloatArray, observed: npt.NDArray[np.bool_], weights: FloatArray | None = None
) -> float:
    times, survival = km_curve(durations, observed, weights)
    below = np.flatnonzero(survival <= 0.5)
    return float(times[below[0]]) if below.size else float("nan")


def km_survival_at(durations: FloatArray, observed: npt.NDArray[np.bool_], day: float) -> float:
    times, survival = km_curve(durations, observed)
    before = np.flatnonzero(times <= day)
    return float(survival[before[-1]]) if before.size else 1.0


def _client_bootstrap_median(
    frame: pd.DataFrame, n_resamples: int, seed: int
) -> tuple[float, float]:
    codes, clients = pd.factorize(frame["client_id"])
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_resamples):
        counts = rng.multinomial(len(clients), np.full(len(clients), 1 / len(clients)))
        weights = counts[codes].astype(np.float64)
        keep = weights > 0
        draws.append(
            km_median(
                frame["duration"].to_numpy()[keep],
                frame["observed"].to_numpy()[keep],
                weights[keep],
            )
        )
    finite = np.asarray([d for d in draws if np.isfinite(d)])
    if finite.size == 0:
        return (float("nan"), float("nan"))
    low, high = np.percentile(finite, [2.5, 97.5])
    return (float(low), float(high))


@dataclass(frozen=True)
class GroupCurve:
    group: str
    n_clients: int
    n_spells: int
    median: float
    median_ci: tuple[float, float]
    survival_7: float
    survival_30: float
    survival_90: float


def curves_by(
    frame: pd.DataFrame, groups: pd.Series, *, n_resamples: int, seed: int
) -> tuple[list[GroupCurve], dict[str, float]]:
    """Kaplan-Meier summaries per group and the multivariate log-rank test across groups."""
    from lifelines.statistics import multivariate_logrank_test  # noqa: PLC0415

    labelled = frame.assign(group=frame["client_id"].map(groups)).dropna(subset=["group"])
    results = []
    for name, rows in labelled.groupby("group", sort=True):
        durations = rows["duration"].to_numpy(dtype=np.float64)
        observed = rows["observed"].to_numpy(dtype=bool)
        results.append(
            GroupCurve(
                group=str(name),
                n_clients=int(rows["client_id"].nunique()),
                n_spells=len(rows),
                median=km_median(durations, observed),
                median_ci=_client_bootstrap_median(rows, n_resamples, seed),
                survival_7=km_survival_at(durations, observed, 7),
                survival_30=km_survival_at(durations, observed, 30),
                survival_90=km_survival_at(durations, observed, 90),
            )
        )
    test = multivariate_logrank_test(labelled["duration"], labelled["group"], labelled["observed"])
    return results, {"statistic": float(test.test_statistic), "p_value": float(test.p_value)}


def cox(frame: pd.DataFrame, covariates: pd.DataFrame) -> dict[str, Any]:
    """Cox model of the gap with client-clustered robust errors, and a proportional-hazards test.

    statsmodels' ``PHReg`` (Breslow ties) gives the sandwich covariance clustered by client. The
    test of proportional hazards correlates each covariate's Schoenfeld residuals with the rank of
    the event time (the Grambsch-Therneau idea, unscaled): a small p-value means the covariate's
    effect changes over the gap.
    """
    from lifelines.utils import concordance_index  # noqa: PLC0415
    from statsmodels.duration.hazard_regression import PHReg  # noqa: PLC0415

    data = frame.merge(covariates, left_on="client_id", right_index=True, how="inner")
    columns = [c for c in covariates.columns if data[c].std() > 0]
    exog = data[columns].to_numpy(dtype=np.float64)
    duration = data["duration"].to_numpy(dtype=np.float64)
    status = data["observed"].to_numpy(dtype=np.int64)
    groups, _ = pd.factorize(data["client_id"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = PHReg(duration, exog, status=status, ties="breslow").fit(groups=groups)
    params = np.asarray(result.params)
    errors = np.asarray(result.bse)
    pvalues = np.asarray(result.pvalues)
    hazard_ratios = {
        name: {
            "hazard_ratio": float(np.exp(params[i])),
            "ci_low": float(np.exp(params[i] - 1.96 * errors[i])),
            "ci_high": float(np.exp(params[i] + 1.96 * errors[i])),
            "p_value": float(pvalues[i]),
        }
        for i, name in enumerate(columns)
    }
    residuals = np.asarray(result.schoenfeld_residuals)
    events = status == 1
    ranks = stats.rankdata(duration[events])
    ph: dict[str, float] = {}
    for i, name in enumerate(columns):
        column = residuals[events, i]
        finite = np.isfinite(column)
        if finite.sum() < 3 or np.std(column[finite]) == 0:
            continue
        correlation, p_value = stats.pearsonr(column[finite], ranks[finite])
        ph[name] = float(p_value)
        del correlation
    risk = exog @ params
    return {
        "hazard_ratios": hazard_ratios,
        "ph_test_p_values": ph,
        "concordance": float(concordance_index(duration, -risk, status)),
        "n_spells": len(data),
        "n_clients": int(data["client_id"].nunique()),
    }


def covariates(clients: pd.DataFrame) -> pd.DataFrame:
    """Client covariates of the Cox model, indexed by client: profile, age (decades), events."""
    frame = pd.DataFrame(index=clients["client_id"].to_numpy())
    profile = clients["channel_profile"].to_numpy()
    frame["profile_digital_only"] = (profile == "digital_only").astype(float)
    frame["profile_mixed"] = (profile == "mixed").astype(float)
    frame["age_decades"] = pd.to_numeric(clients["age"], errors="coerce").fillna(0).to_numpy() / 10
    frame["log_events"] = np.log1p(clients["n_events"].to_numpy(dtype=float))
    return frame


def run(
    events: pd.DataFrame, clients: pd.DataFrame, *, n_resamples: int = 200, seed: int = 42
) -> dict[str, Any]:
    """Every survival summary, ready for the registry and the model card."""
    events = events[~events["is_repeat"]] if "is_repeat" in events else events
    spells = gaps(events)
    silence = time_to_silence(events)
    profile = clients.set_index("client_id")["channel_profile"]
    age = clients.set_index("client_id")["age_band"]
    by_profile, logrank_profile = curves_by(spells, profile, n_resamples=n_resamples, seed=seed)
    by_age, logrank_age = curves_by(spells, age, n_resamples=n_resamples, seed=seed)
    silence_profile, logrank_silence = curves_by(
        silence, profile, n_resamples=n_resamples, seed=seed
    )
    return {
        "gap_by_profile": [vars(curve) for curve in by_profile],
        "gap_logrank_profile": logrank_profile,
        "gap_by_age_band": [vars(curve) for curve in by_age],
        "gap_logrank_age_band": logrank_age,
        "cox": cox(spells, covariates(clients)),
        "silence_by_profile": [vars(curve) for curve in silence_profile],
        "silence_logrank_profile": logrank_silence,
        "silence_events": int(silence["observed"].sum()),
        "silence_clients": len(silence),
    }
