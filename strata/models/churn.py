"""Silence risk: who has no event in the three months after a cutoff (ARCHITECTURE section 10).

The comparison is between models of increasing complexity, each evaluated on a later cutoff than
the one it learnt from (``strata.models.splits``):

- ``majority``: the training silence rate for everyone (no ranking at all);
- ``activity_rule``: the training silence rate of the client's recent-activity bucket;
- ``glm``: a logistic regression with odds ratios and Wald intervals per standard deviation;
- ``lightgbm``: gradient-boosted trees tuned with Optuna on the earliest cutoff only;
- ``lightgbm_isotonic`` and ``lightgbm_platt``: the same trees, calibrated on a held-out fifth of
  the training clients.

Every metric carries a client-bootstrap interval (``strata.models.metrics``); SHAP values explain
the trees globally and per client.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import statsmodels.api as sm
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split

from strata.models import metrics, splits

SERVED = "lightgbm_platt"
"""The calibrated score the API serves and the backtest table of the model card reports."""

ACTIVITY_BUCKETS = (0, 1, 3, 6, 12)
"""Lower bounds of the buckets of events in the last ninety days used by the activity rule."""


@dataclass(frozen=True)
class Effort:
    trials: int
    resamples: int
    folds: int = 3


FULL = Effort(trials=40, resamples=1000)
QUICK = Effort(trials=4, resamples=100)


def _bucket(values: pd.Series) -> np.ndarray:
    return np.searchsorted(np.asarray(ACTIVITY_BUCKETS), values.to_numpy(), side="right") - 1


def activity_rule(train: pd.DataFrame, y_train: np.ndarray, test: pd.DataFrame) -> np.ndarray:
    """Silence rate of each recent-activity bucket in training, applied to the test clients."""
    buckets = _bucket(train["n_events_90d"])
    rates = {b: float(y_train[buckets == b].mean()) for b in np.unique(buckets)}
    overall = float(y_train.mean())
    return np.array([rates.get(b, overall) for b in _bucket(test["n_events_90d"])])


@dataclass(frozen=True)
class GlmFit:
    columns: list[str]
    means: dict[str, float]
    scales: dict[str, float]
    result: Any

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        x = _standardised(splits.design(rows, log_skewed=True), self.means, self.scales)
        return np.asarray(self.result.predict(sm.add_constant(x[self.columns], has_constant="add")))

    def odds_ratios(self) -> dict[str, dict[str, float]]:
        params, intervals = self.result.params, self.result.conf_int()
        return {
            name: {
                "odds_ratio": float(np.exp(params[name])),
                "ci_low": float(np.exp(intervals.loc[name, 0])),
                "ci_high": float(np.exp(intervals.loc[name, 1])),
                "p_value": float(self.result.pvalues[name]),
            }
            for name in self.columns
        }


def _standardised(
    x: pd.DataFrame, means: dict[str, float], scales: dict[str, float]
) -> pd.DataFrame:
    out = x.copy()
    for name, mean in means.items():
        out[name] = (out[name] - mean) / scales[name]
    return out


def fit_glm(rows: pd.DataFrame, y: np.ndarray) -> GlmFit:
    """Logistic regression; continuous features standardised so odds ratios are per one SD."""
    x = splits.design(rows, log_skewed=True)
    varying = [name for name in x.columns if x[name].std() > 0]
    x = x[varying]
    binary = {name for name in varying if set(np.unique(x[name])) <= {0.0, 1.0}}
    means = {name: float(x[name].mean()) for name in varying if name not in binary}
    scales = {name: float(x[name].std()) for name in means}
    design = sm.add_constant(_standardised(x, means, scales), has_constant="add")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = sm.GLM(y, design, family=sm.families.Binomial()).fit()
    return GlmFit(columns=varying, means=means, scales=scales, result=result)


def _lgbm(params: dict[str, Any], seed: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        random_state=seed,
        n_jobs=1,
        deterministic=True,
        force_col_wise=True,
        verbose=-1,
        **params,
    )


def positive(model: lgb.LGBMClassifier, x: pd.DataFrame) -> np.ndarray:
    """Predicted probability of the positive class."""
    return np.asarray(model.predict_proba(x), dtype=np.float64)[:, 1]


def tune(rows: pd.DataFrame, y: np.ndarray, effort: Effort, seed: int) -> dict[str, Any]:
    """Optuna search on one cutoff only: mean cross-validated log loss over stratified folds."""
    x = splits.design(rows)
    folds = list(StratifiedKFold(effort.folds, shuffle=True, random_state=seed).split(x, y))

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 400),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 4, 48),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 80),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
        losses = []
        for train_index, valid_index in folds:
            model = _lgbm(params, seed).fit(x.iloc[train_index], y[train_index])
            p = positive(model, x.iloc[valid_index])
            losses.append(metrics.log_loss(y[valid_index], p))
        return float(np.mean(losses))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=effort.trials)
    return {**study.best_params, "subsample_freq": 1}


@dataclass
class Calibrated:
    model: lgb.LGBMClassifier
    isotonic: IsotonicRegression
    platt: LogisticRegression

    def scores(self, rows: pd.DataFrame) -> dict[str, np.ndarray]:
        raw = positive(self.model, splits.design(rows))
        logit = np.log(np.clip(raw, 1e-6, 1 - 1e-6) / np.clip(1 - raw, 1e-6, 1))
        return {
            "lightgbm": raw,
            "lightgbm_isotonic": self.isotonic.predict(raw),
            "lightgbm_platt": self.platt.predict_proba(logit.reshape(-1, 1))[:, 1],
        }


def fit_trees(rows: pd.DataFrame, y: np.ndarray, params: dict[str, Any], seed: int) -> Calibrated:
    """Trees on four fifths of the training clients, both calibrators on the remaining fifth."""
    fit_rows, calibration_rows, fit_y, calibration_y = train_test_split(
        rows, y, test_size=0.2, stratify=y, random_state=seed
    )
    model = _lgbm(params, seed).fit(splits.design(fit_rows), fit_y)
    raw = positive(model, splits.design(calibration_rows))
    isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(
        raw, calibration_y
    )
    logit = np.log(np.clip(raw, 1e-6, 1 - 1e-6) / np.clip(1 - raw, 1e-6, 1))
    platt = LogisticRegression().fit(logit.reshape(-1, 1), calibration_y)
    return Calibrated(model=model, isotonic=isotonic, platt=platt)


def shap_values(model: lgb.LGBMClassifier, x: pd.DataFrame) -> np.ndarray:
    """SHAP values on the log-odds scale, ``(clients, features)``."""
    import shap  # noqa: PLC0415 - heavy import, only for explanations

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        values = shap.TreeExplainer(model).shap_values(x)
    array = np.asarray(values[1] if isinstance(values, list) else values, dtype=np.float64)
    return array[:, :, 1] if array.ndim == 3 else array


def reasons(
    values: np.ndarray, columns: list[str], top: int = 3
) -> list[list[dict[str, float | str]]]:
    """For each client, the features pushing its risk up the most, with their SHAP values."""
    order = np.argsort(-values, axis=1)[:, :top]
    return [
        [{"feature": columns[j], "shap": float(values[i, j])} for j in order[i] if values[i, j] > 0]
        for i in range(values.shape[0])
    ]


@dataclass
class BacktestResult:
    name: str
    train_cutoff: str
    test_cutoff: str
    n_train: int
    n_test: int
    silence_rate_train: float
    silence_rate_test: float
    metrics: dict[str, dict[str, metrics.Metric]]
    reliability: dict[str, list[dict[str, float]]]
    shap_global: dict[str, float] = field(default_factory=dict)
    """Mean absolute SHAP value of each feature for the trees of this backtest, on its test rows."""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "train_cutoff": self.train_cutoff,
            "test_cutoff": self.test_cutoff,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "silence_rate_train": self.silence_rate_train,
            "silence_rate_test": self.silence_rate_test,
            "metrics": {
                model: {name: value.as_dict() for name, value in scores.items()}
                for model, scores in self.metrics.items()
            },
            "reliability": self.reliability,
            "shap_global": self.shap_global,
        }


@dataclass
class ChurnResult:
    params: dict[str, Any]
    backtests: list[BacktestResult]
    odds_ratios: dict[str, dict[str, float]]
    shap_global: dict[str, float]
    scored: pd.DataFrame
    auc_gain_over_glm: metrics.Metric
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def primary(self) -> BacktestResult:
        return self.backtests[-1]


def _mean_abs_shap(contributions: np.ndarray, columns: list[str]) -> dict[str, float]:
    """Mean absolute SHAP value per feature, largest first."""
    means = np.abs(contributions).mean(axis=0)
    pairs = ((column, float(value)) for column, value in zip(columns, means, strict=True))
    return dict(sorted(pairs, key=lambda item: -item[1]))


def run(features: pd.DataFrame, *, effort: Effort = FULL, seed: int = 42) -> ChurnResult:
    """Tune on the earliest cutoff, evaluate every backtest, explain the primary one."""
    first = splits.rows_for(features, splits.earliest().cutoff)
    params = tune(first, splits.labels(first), effort, seed)
    results: list[BacktestResult] = []
    for backtest in splits.backtests():
        train = splits.rows_for(features, backtest.train.cutoff)
        test = splits.rows_for(features, backtest.test.cutoff)
        y_train, y_test = splits.labels(train), splits.labels(test)
        glm = fit_glm(train, y_train)
        trees = fit_trees(train, y_train, params, seed)
        scores = {
            "majority": np.full(len(test), float(y_train.mean())),
            "activity_rule": activity_rule(train, y_train, test),
            "glm": glm.predict(test),
            **trees.scores(test),
        }
        evaluated = {
            name: metrics.evaluate(y_test, p, n_resamples=effort.resamples, seed=seed)
            for name, p in scores.items()
        }
        x_backtest = splits.design(test)
        backtest_shap = _mean_abs_shap(
            shap_values(trees.model, x_backtest), list(x_backtest.columns)
        )
        results.append(
            BacktestResult(
                name=backtest.name,
                train_cutoff=backtest.train.cutoff.isoformat(),
                test_cutoff=backtest.test.cutoff.isoformat(),
                n_train=len(train),
                n_test=len(test),
                silence_rate_train=float(y_train.mean()),
                silence_rate_test=float(y_test.mean()),
                metrics=evaluated,
                reliability={name: metrics.reliability(y_test, p) for name, p in scores.items()},
                shap_global=backtest_shap,
            )
        )
    # The primary backtest's models, kept for explanation and scoring.
    # Platt scaling is monotone, so the served score keeps the trees' ranking (isotonic
    # regression ties scores into steps); the choice was made before any model was evaluated.
    explained = trees.scores(test)[SERVED]
    x_test = splits.design(test)
    contributions = shap_values(trees.model, x_test)
    columns = list(x_test.columns)
    shap_global = _mean_abs_shap(contributions, columns)
    scored = pd.DataFrame(
        {
            "client_id": test["client_id"],
            "cutoff_date": backtest.test.cutoff.isoformat(),
            "probability": explained,
            "reasons": reasons(contributions, columns),
        }
    ).sort_values("probability", ascending=False, kind="stable")
    gain = metrics.paired_difference(
        y_test,
        scores["lightgbm"],
        scores["glm"],
        metrics.auc,
        n_resamples=effort.resamples,
        seed=seed,
    )
    return ChurnResult(
        params=params,
        backtests=results,
        odds_ratios=glm.odds_ratios(),
        shap_global=shap_global,
        scored=scored.reset_index(drop=True),
        auc_gain_over_glm=gain,
    )


logging.getLogger("lightgbm").setLevel(logging.ERROR)
