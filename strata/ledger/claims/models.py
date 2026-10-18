"""Claims from the calendar models and the seasonal decomposition (M4)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from strata.analysis import calendar as cal
from strata.analysis.calendar_bayes import FULL, QUICK, fit_bayes
from strata.analysis.seasonality import weekly_strength
from strata.ledger.bootstrap import percentile_interval
from strata.ledger.claims._common import MONTH_TREND_CAVEAT, require
from strata.ledger.model import Unit
from strata.ledger.registry import ClaimContext, ClaimResult, claim

REPORTED_EFFECTS = (
    "dom_01_05",
    "december",
    "ramadan",
    "eid_al_fitr",
    "eid_al_adha",
    "pre_eid_al_adha_week",
    "bank_holiday",
    "dow_sat",
    "dow_sun",
)
EFFECT_LABELS = {
    "dom_01_05": "days 1 to 5 against days 11 to 15",
    "december": "December against the average month",
    "ramadan": "Ramadan days against other days",
    "eid_al_fitr": "Eid al-Fitr",
    "eid_al_adha": "Eid al-Adha",
    "pre_eid_al_adha_week": "the week before Eid al-Adha",
    "bank_holiday": "bank holidays other than the Eids (national and religious)",
    "dow_sat": "Saturday against Monday",
    "dow_sun": "Sunday against Monday",
}


def _calendar_inputs(context: ClaimContext) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = context.table("marts.daily_counts")
    events = context.table("intermediate.int_events")
    require(len(daily) > 0, "no daily series")
    return daily, events


@claim(
    "calendar_effects",
    title="Calendar effects on daily activity",
    question=(
        "How much do salary days, December, Ramadan, the Eids, the week before Eid al-Adha, bank "
        "holidays and weekends change daily card, online and branch activity?"
    ),
    method=(
        "Negative-binomial regression of each daily series (daily_counts) on day of week, five-day "
        "bins of the month, calendar month in sum-to-zero coding (one year cannot separate month "
        "from trend), and indicators for Ramadan, the Eids, the week before Eid al-Adha and the "
        "other bank holidays; branch series on open days only. Rate ratios with client-bootstrap "
        "intervals from refits at the fitted dispersion."
    ),
)
def calendar_effects(context: ClaimContext) -> ClaimResult:
    daily, events = _calendar_inputs(context)
    values: dict[str, float] = {}
    intervals: dict[str, tuple[float, float]] = {}
    value_n: dict[str, int] = {}
    meta: dict[str, str] = {"bootstrap_refits": str(context.model_resamples)}
    clients: set[str] = set()
    total_events = 0
    for series in cal.SERIES:
        frame = cal.daily_series(daily, series)
        if frame["n_events"].sum() == 0:
            continue
        matrix, n_clients = cal.client_daily_matrix(events, frame, series)
        boot = context.bootstrap(n_clients, stream=series.key, resamples=context.model_resamples)
        fit = cal.fit_series(frame, series, matrices=matrix, boot=boot)
        meta[f"{series.key}_label"] = series.label
        meta[f"{series.key}_dispersion"] = f"{fit.alpha:.4g}"
        meta[f"{series.key}_days"] = str(fit.n_days)
        for effect in REPORTED_EFFECTS:
            if effect not in fit.rate_ratios:
                continue
            key = f"{series.key}_{effect}"
            values[key] = fit.rate_ratios[effect]
            if effect in fit.bootstrap and all(np.isfinite(fit.bootstrap[effect])):
                intervals[key] = fit.bootstrap[effect]
            value_n[key] = n_clients
        clients.update(events.loc[events["channel"] == series.channel, "client_id"])
        total_events += int(frame["n_events"].sum())
    require(bool(values), "no estimable effect")
    meta.update({f"effect_{name}": label for name, label in EFFECT_LABELS.items()})
    return ClaimResult(
        values=values,
        units=dict.fromkeys(values, Unit.RATIO),
        ci=intervals,
        value_n=value_n,
        n=len(clients),
        n_events=total_events,
        meta=meta,
        caveats=[
            (
                "One year of data: each special period (one Ramadan, one December) is observed "
                "once, so an effect cannot be told apart from a one-off event in the same weeks."
            ),
            MONTH_TREND_CAVEAT,
            "Dates may be posting dates; weekend effects partly reflect posting calendars.",
            "Intervals hold the dispersion at its fitted value across bootstrap refits.",
        ],
    )


@claim(
    "calendar_bayes",
    title="The December deposit wave and salary days, in a Bayesian calendar model",
    question=(
        "How probable is it that December's card deposit rate is more than twice the average "
        "month's, and does the model reproduce the observed series?"
    ),
    method=(
        "The calendar regression in PyMC (negative binomial, log link) with weakly informative "
        "priors: Normal(0, 0.5) on log rate ratios, Normal(log mean, 1) on the intercept, "
        "Exponential(0.1) on the dispersion. Posterior medians with 95% equal-tailed credible "
        "intervals; posterior predictive checks (coverage of the 90% predictive interval and a "
        "check on the standard deviation of the series); convergence diagnostics."
    ),
)
def calendar_bayes(context: ClaimContext) -> ClaimResult:
    daily, events = _calendar_inputs(context)
    settings = QUICK if context.quick else FULL
    values: dict[str, float] = {}
    intervals: dict[str, tuple[float, float]] = {}
    value_n: dict[str, int] = {}
    meta = {"sampler": settings.describe()}
    clients: set[str] = set()
    for key in ("card_deposit", "card_withdrawal"):
        series = next(s for s in cal.SERIES if s.key == key)
        frame = cal.daily_series(daily, series)
        require(frame["n_events"].sum() > 0, f"no {series.label}")
        fit = fit_bayes(frame, series, settings, seed=context.seed)
        n_clients = int(
            events.loc[
                (events["channel"] == series.channel)
                & events["kind_group"].isin(series.kinds or ()),
                "client_id",
            ].nunique()
        )
        clients.update(events.loc[events["channel"] == series.channel, "client_id"])
        for effect in ("december", "dom_01_05", "ramadan", "pre_eid_al_adha_week"):
            if effect in fit.rate_ratios:
                values[f"{key}_{effect}"] = fit.rate_ratios[effect]
                intervals[f"{key}_{effect}"] = fit.intervals[effect]
        if fit.probability_december_doubles is not None:
            values[f"{key}_p_december_doubles"] = fit.probability_december_doubles
        values[f"{key}_ppc_coverage_90"] = fit.ppc_coverage_90
        values[f"{key}_ppc_sd_p_value"] = fit.ppc_sd_p_value
        values[f"{key}_rhat_max"] = fit.rhat_max
        values[f"{key}_ess_bulk_min"] = fit.ess_bulk_min
        values[f"{key}_divergences"] = float(fit.divergences)
        value_n.update({name: n_clients for name in values if name.startswith(key)})
    units: dict[str, Unit] = {}
    for name in values:
        if name.endswith(("p_december_doubles", "ppc_coverage_90")):
            units[name] = Unit.SHARE
        elif name.endswith(("ess_bulk_min", "divergences")):
            units[name] = Unit.COUNT
        elif name.endswith(("ppc_sd_p_value", "rhat_max")):
            units[name] = Unit.NUMBER
        else:
            units[name] = Unit.RATIO
    return ClaimResult(
        values=values,
        units=units,
        ci=intervals,
        interval_label="95% credible interval",
        value_n=value_n,
        n=len(clients),
        meta=meta,
        caveats=[
            MONTH_TREND_CAVEAT,
            (
                "Posterior probabilities depend on the priors; with a single December the data "
                "cannot separate a seasonal wave from a one-off event."
            ),
            "A predictive-check value near zero or one means the model misses the feature checked.",
        ],
    )


@claim(
    "weekly_seasonality",
    title="How weekly is the week?",
    question="How strong is the weekly cycle in daily card, online and branch activity?",
    method=(
        "STL decomposition (period seven, robust) of each channel's daily event count: strength of "
        "the weekly component and of the trend, max(0, 1 - Var(remainder) / Var(component + "
        "remainder)); client-bootstrap intervals."
    ),
)
def weekly_seasonality(context: ClaimContext) -> ClaimResult:
    daily, events = _calendar_inputs(context)
    values: dict[str, float] = {}
    intervals: dict[str, tuple[float, float]] = {}
    value_n: dict[str, int] = {}
    clients: set[str] = set()
    for series in cal.CHANNEL_SERIES:
        channel = series.channel
        frame = cal.daily_series(daily, series)
        if frame["n_events"].sum() == 0:
            continue
        matrix, n_clients = cal.client_daily_matrix(events, frame, series)
        point = weekly_strength(frame["n_events"].to_numpy(dtype=float))
        boot = context.bootstrap(n_clients, stream=channel, resamples=context.model_resamples)
        draws = [weekly_strength(totals) for totals in boot.weights @ matrix]
        values[f"{channel}_weekly_strength"] = point.weekly
        values[f"{channel}_trend_strength"] = point.trend
        intervals[f"{channel}_weekly_strength"] = percentile_interval([d.weekly for d in draws])
        intervals[f"{channel}_trend_strength"] = percentile_interval([d.trend for d in draws])
        value_n.update(
            {f"{channel}_weekly_strength": n_clients, f"{channel}_trend_strength": n_clients}
        )
        clients.update(events.loc[events["channel"] == channel, "client_id"])
    require(bool(values), "no daily series")
    return ClaimResult(
        values=values,
        units=dict.fromkeys(values, Unit.NUMBER),
        ci=intervals,
        value_n=value_n,
        n=len(clients),
        meta={"bootstrap_resamples": str(context.model_resamples)},
        caveats=[
            (
                "A strength near one means the component dominates the noise; branch activity is "
                "weekly by construction because branches close at weekends."
            ),
        ],
    )
