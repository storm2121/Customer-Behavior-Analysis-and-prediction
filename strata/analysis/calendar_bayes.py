"""The calendar regression as a Bayesian model (PyMC), with posterior predictive checks.

Same likelihood and design as :mod:`strata.analysis.calendar` (negative binomial, log link), with
weakly informative priors: ``Normal(log mean, 1)`` for the intercept, ``Normal(0, 0.5)`` for every
log rate ratio (a ratio between about 0.37 and 2.7 with 95% prior probability) and an
``Exponential(0.1)`` prior on the dispersion. The posterior answers direct questions such as "what
is the probability that December's deposit rate is more than twice the average month's?", and
posterior predictive checks compare replicated series with the observed one. ADR 0007 records why
PyMC is the backend on this platform.
"""

from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from strata.analysis.calendar import Series, design_matrix

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class SamplerSettings:
    """MCMC effort. ``FULL`` for real runs, ``QUICK`` for tests and demos."""

    chains: int
    tune: int
    draws: int

    def describe(self) -> str:
        return f"{self.chains} chains of {self.draws} draws after {self.tune} tuning steps"


FULL = SamplerSettings(chains=4, tune=1000, draws=1000)
QUICK = SamplerSettings(chains=2, tune=300, draws=300)


@dataclass(frozen=True)
class BayesFit:
    """Posterior summaries of one series."""

    series: Series
    settings: SamplerSettings
    rate_ratios: dict[str, float]
    intervals: dict[str, tuple[float, float]]
    probability_december_doubles: float | None
    ppc_coverage_90: float
    ppc_sd_p_value: float
    rhat_max: float
    ess_bulk_min: float
    divergences: int


def _december_column(design: pd.DataFrame, months: list[str]) -> FloatArray | None:
    """Weights that turn the month coefficients into December's sum-coded effect."""
    names = [name for name in design.columns if name.startswith("month_")]
    if "2023-12" not in months:
        return None
    weights = np.zeros(len(design.columns))
    if months[-1] == "2023-12":
        for name in names:
            weights[design.columns.get_loc(name)] = -1.0
    else:
        weights[design.columns.get_loc("month_2023_12")] = 1.0
    return weights


def fit_bayes(
    frame: pd.DataFrame, series: Series, settings: SamplerSettings = FULL, seed: int = 42
) -> BayesFit:
    """Sample the posterior of ``series`` and summarise it."""
    # No C++ compiler is assumed (none on the reference Windows machine): PyTensor then uses its
    # own backend, and saying so explicitly keeps it from warning on every import (ADR 0007).
    os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")
    import arviz as az  # noqa: PLC0415 - heavy imports only when a Bayesian fit is requested
    import pymc as pm  # noqa: PLC0415

    logging.getLogger("pymc").setLevel(logging.ERROR)
    design = design_matrix(frame)
    y = frame["n_events"].to_numpy(dtype=np.int64)
    months = sorted(frame["year_month"].unique())
    x = design.drop(columns="intercept").to_numpy(dtype=float)
    names = [name for name in design.columns if name != "intercept"]
    with pm.Model():
        intercept = pm.Normal("intercept", mu=float(np.log(max(y.mean(), 0.1))), sigma=1.0)
        beta = pm.Normal("beta", mu=0.0, sigma=0.5, shape=len(names))
        alpha = pm.Exponential("alpha", lam=0.1)
        mu = pm.math.exp(intercept + pm.math.dot(x, beta))
        pm.NegativeBinomial("y", mu=mu, alpha=alpha, observed=y)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            idata = pm.sample(
                draws=settings.draws,
                tune=settings.tune,
                chains=settings.chains,
                cores=1,
                random_seed=seed,
                progressbar=False,
                compute_convergence_checks=False,
            )
            predictive = pm.sample_posterior_predictive(idata, random_seed=seed, progressbar=False)
    beta_draws = idata.posterior["beta"].stack(sample=("chain", "draw")).to_numpy().T
    rate_ratios: dict[str, float] = {}
    intervals: dict[str, tuple[float, float]] = {}
    for index, name in enumerate(names):
        if name.startswith("month_"):
            continue
        ratios = np.exp(beta_draws[:, index])
        rate_ratios[name] = float(np.median(ratios))
        intervals[name] = (float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5)))
    probability = None
    weights = _december_column(design, months)
    if weights is not None:
        december = np.exp(beta_draws @ weights[1:])
        rate_ratios["december"] = float(np.median(december))
        intervals["december"] = (
            float(np.percentile(december, 2.5)),
            float(np.percentile(december, 97.5)),
        )
        probability = float((december > 2.0).mean())
    replicated = predictive.posterior_predictive["y"].stack(sample=("chain", "draw")).to_numpy()
    low, high = np.percentile(replicated, [5, 95], axis=1)
    coverage = float(((y >= low) & (y <= high)).mean())
    sd_p_value = float((replicated.std(axis=0) >= y.std()).mean())
    summary = az.summary(idata, var_names=["intercept", "beta", "alpha"], kind="diagnostics")
    divergences = int(idata.sample_stats["diverging"].sum())
    return BayesFit(
        series=series,
        settings=settings,
        rate_ratios=rate_ratios,
        intervals=intervals,
        probability_december_doubles=probability,
        ppc_coverage_90=coverage,
        ppc_sd_p_value=sd_p_value,
        rhat_max=float(summary["r_hat"].max()),
        ess_bulk_min=float(summary["ess_bulk"].min()),
        divergences=divergences,
    )
