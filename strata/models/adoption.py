"""Online adoption on the aggregate sample (ARCHITECTURE section 10).

The aggregate file describes a different sample of clients, pre-aggregated over the year, so there
is no time axis to split on: models are evaluated with out-of-fold predictions from stratified
five-fold cross-validation, and every metric carries a client-bootstrap interval. Two targets:

- ``has_online``: the client has online banking (the file's access flag);
- ``active_online``: at least the configured number of online operations in the year.

The online operation counts and amounts define the second target and follow from the first, so they
are never features. Age and e-mail are expected to dominate: the model card lists the features in
order of importance, whatever they turn out to be.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import StratifiedKFold

from strata.models import metrics

TARGETS = ("target_has_online", "target_active_online")
EXCLUDED_PREFIXES = ("ops_", "op_amount_", "months_since_first_op", "months_since_last_op")
EXCLUDED = {"client_id", "age_band", "has_online", *TARGETS}
GLM_FEATURES = (
    "age",
    "tenure_years",
    "children",
    "has_email",
    "has_active_contract",
    "has_international_txn",
    "log_txn_total",
)


def features(sample: pd.DataFrame) -> pd.DataFrame:
    """Every allowed column as a number (codes stay codes: trees split them, the GLM skips them)."""
    columns = [
        c for c in sample.columns if c not in EXCLUDED and not c.startswith(EXCLUDED_PREFIXES)
    ]
    frame = pd.DataFrame(index=sample.index)
    for column in columns:
        values = sample[column]
        if values.dtype == bool or str(values.dtype) == "boolean":
            frame[column] = values.astype(float)
        else:
            frame[column] = pd.to_numeric(values, errors="coerce").astype(float)
    frame["log_txn_total"] = np.log1p(frame["txn_total"].fillna(0).clip(lower=0))
    return frame


def _glm_design(x: pd.DataFrame) -> pd.DataFrame:
    design = x[list(GLM_FEATURES)].fillna(0.0).copy()
    design["age"] = design["age"] / 10
    return design


@dataclass(frozen=True)
class TargetResult:
    target: str
    n_clients: int
    rate: float
    metrics: dict[str, dict[str, metrics.Metric]]
    odds_ratios: dict[str, dict[str, float]]
    importance: dict[str, float]
    deciles: list[dict[str, float]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "n_clients": self.n_clients,
            "rate": self.rate,
            "metrics": {
                model: {name: value.as_dict() for name, value in scores.items()}
                for model, scores in self.metrics.items()
            },
            "odds_ratios": self.odds_ratios,
            "importance": self.importance,
            "deciles": self.deciles,
        }


def deciles(y: np.ndarray, p: np.ndarray) -> list[dict[str, float]]:
    """The decile table: share of the target in each tenth of clients by score, lift and gain."""
    order = np.argsort(-p, kind="stable")
    groups = np.array_split(order, 10)
    base, total = float(y.mean()), float(y.sum())
    rows, cumulative = [], 0.0
    for index, members in enumerate(groups, start=1):
        hits = float(y[members].sum())
        cumulative += hits
        rate = hits / len(members) if len(members) else float("nan")
        rows.append(
            {
                "decile": float(index),
                "n": float(len(members)),
                "rate": rate,
                "lift": rate / base if base > 0 else float("nan"),
                "cumulative_gain": cumulative / total if total > 0 else float("nan"),
            }
        )
    return rows


def _odds_ratios(x: pd.DataFrame, y: np.ndarray) -> dict[str, dict[str, float]]:
    design = sm.add_constant(_glm_design(x), has_constant="add")
    keep = [c for c in design.columns if c == "const" or design[c].std() > 0]
    result = sm.GLM(y, design[keep], family=sm.families.Binomial()).fit()
    intervals = result.conf_int()
    return {
        name: {
            "odds_ratio": float(np.exp(result.params[name])),
            "ci_low": float(np.exp(intervals.loc[name, 0])),
            "ci_high": float(np.exp(intervals.loc[name, 1])),
            "p_value": float(result.pvalues[name]),
        }
        for name in keep
        if name != "const"
    }


def run_target(
    sample: pd.DataFrame, target: str, *, n_resamples: int = 1000, seed: int = 42
) -> TargetResult:
    rows = sample[sample[target].notna()].sort_values("client_id").reset_index(drop=True)
    y = np.asarray(rows[target].astype(bool), dtype=np.int64)
    x = features(rows)
    oof = {"glm": np.zeros(len(rows)), "lightgbm": np.zeros(len(rows))}
    folds = StratifiedKFold(5, shuffle=True, random_state=seed)
    gains = np.zeros(x.shape[1])
    for train, test in folds.split(x, y):
        design = sm.add_constant(_glm_design(x.iloc[train]), has_constant="add")
        keep = [c for c in design.columns if c == "const" or design[c].std() > 0]
        glm = sm.GLM(y[train], design[keep], family=sm.families.Binomial()).fit()
        test_design = sm.add_constant(_glm_design(x.iloc[test]), has_constant="add")[keep]
        oof["glm"][test] = np.asarray(glm.predict(test_design))
        trees = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=20,
            random_state=seed,
            n_jobs=1,
            deterministic=True,
            force_col_wise=True,
            verbose=-1,
        ).fit(x.iloc[train], y[train])
        oof["lightgbm"][test] = np.asarray(trees.predict_proba(x.iloc[test]), dtype=np.float64)[
            :, 1
        ]
        gains += trees.booster_.feature_importance(importance_type="gain")
    importance = dict(
        sorted(
            ((c, float(g / gains.sum())) for c, g in zip(x.columns, gains, strict=True)),
            key=lambda item: -item[1],
        )
    )
    chosen = {
        name: metrics.METRICS[name] for name in ("auc", "average_precision", "brier", "lift_at_10")
    }
    return TargetResult(
        target=target,
        n_clients=len(rows),
        rate=float(y.mean()),
        metrics={
            name: metrics.evaluate(y, p, n_resamples=n_resamples, seed=seed, metrics=chosen)
            for name, p in oof.items()
        },
        odds_ratios=_odds_ratios(x, y),
        importance=importance,
        deciles=deciles(y, oof["lightgbm"]),
    )


def run(sample: pd.DataFrame, *, n_resamples: int = 1000, seed: int = 42) -> dict[str, Any]:
    clients = sample[sample["client_id"].notna()]
    return {
        target: run_target(clients, target, n_resamples=n_resamples, seed=seed).as_dict()
        for target in TARGETS
    }
