"""Statistical detectors and the combined triage score (ARCHITECTURE section 10).

- **Isolation Forest** on the per-client features of ``marts.features_anomaly`` (log-scaled).
- **Empirical-Bayes amount model.** For each category (channel and kind) the log amounts of a
  client are compared with the client's own typical amount, shrunk toward the category's: with
  ``n_i`` amounts of client ``i``, within-client variance ``s2`` and between-client variance ``t2``
  of the category, the shrunk mean is ``(t2 m_i + (s2 / n_i) mu) / (t2 + s2 / n_i)``. A robust z
  (residual over 1.4826 times the category's median absolute residual) scores each amount; a
  client's score is its largest z, so a few amounts far above the client's (and the category's)
  usual level stand out, while a client who always moves large sums does not.
- **Benford per client**: the mean absolute deviation of first digits from Benford's law for clients
  with enough amounts.
- **Combined score**: the weighted sum of the rule score (the weights of the rules that fire over
  the total), and the percentile ranks of the three detectors. Every alert carries its reasons.

Everything here ranks clients for a person to review. Nothing here decides anything about anyone.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from strata.analysis import benford
from strata.models.anomaly.rules import Hit, RuleSet, pseudo_labels

FOREST_FEATURES = (
    "n_events",
    "card_spend",
    "n_branch_deposits",
    "branch_deposit_amount",
    "max_branch_deposit",
    "n_large_deposits",
    "max_deposits_in_window",
    "max_deposit_sum_in_window",
    "n_pass_through",
    "cash_to_card_ratio",
    "max_month_events",
)
ROBUST = 1.4826


def _rank(values: pd.Series) -> pd.Series:
    """Percentile rank in [0, 1] (missing values rank lowest)."""
    return values.fillna(-np.inf).rank(pct=True, method="average")


def forest_scores(features: pd.DataFrame, seed: int) -> pd.Series:
    """Isolation Forest anomaly score per client (higher means more isolated)."""
    x = np.log1p(
        features[list(FOREST_FEATURES)]
        .apply(pd.to_numeric, errors="coerce")
        .astype(float)
        .fillna(0)
        .clip(lower=0)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = IsolationForest(n_estimators=200, random_state=seed).fit(x)
    return pd.Series(-model.score_samples(x), index=features["client_id"].to_numpy())


@dataclass(frozen=True)
class AmountFlag:
    z: float
    amount: float
    category: str
    event_date: str


def amount_scores(events: pd.DataFrame) -> dict[str, AmountFlag]:
    """The most unusual amount of every client under the empirical-Bayes model."""
    rows = events[events["amount"].notna() & (events["amount"] > 0) & ~events["is_repeat"]].copy()
    rows["category"] = rows["channel"] + ":" + rows["kind_group"].fillna("other")
    rows["log_amount"] = np.log(rows["amount"].astype(float))
    flags: dict[str, AmountFlag] = {}
    for category, group in rows.groupby("category", sort=True):
        stats = group.groupby("client_id")["log_amount"].agg(["mean", "count"])
        mu = float(group["log_amount"].mean())
        deviations = group["log_amount"] - group["client_id"].map(stats["mean"])
        within = float(np.var(deviations, ddof=1)) if len(group) > 1 else 0.0
        between = (
            max(
                float(stats["mean"].var(ddof=1)) - within * float((1 / stats["count"]).mean()), 1e-6
            )
            if len(stats) > 1
            else 1e-6
        )
        noise = within / stats["count"]
        shrunk = (between * stats["mean"] + noise * mu) / (between + noise)
        residual = group["log_amount"] - group["client_id"].map(shrunk)
        scale = ROBUST * float(np.median(np.abs(residual - np.median(residual))))
        if scale <= 0:
            continue
        z = residual / scale
        best = z.groupby(group["client_id"]).idxmax()
        for client, index in best.items():
            value = float(z[index])
            if client not in flags or value > flags[str(client)].z:
                flags[str(client)] = AmountFlag(
                    z=value,
                    amount=float(group.loc[index, "amount"]),
                    category=str(category),
                    event_date=pd.Timestamp(group.loc[index, "event_date"]).date().isoformat(),
                )
    return flags


def benford_scores(events: pd.DataFrame, min_amounts: int) -> dict[str, tuple[float, str, int]]:
    """``client -> (MAD, conformity, amounts)`` for clients with ``min_amounts`` amounts or more."""
    rows = events[events["amount"].notna() & (events["amount"] > 0) & ~events["is_repeat"]]
    out = {}
    for client, group in rows.groupby("client_id", sort=True):
        if len(group) >= min_amounts:
            test = benford.first_digit_test(group["amount"].to_numpy(dtype=np.float64))
            out[str(client)] = (test.mad, test.conformity, test.n)
    return out


@dataclass(frozen=True)
class Queue:
    """Every client's combined score and reasons, ranked, plus what the evaluation needs."""

    alerts: pd.DataFrame
    pseudo_labels: set[str]
    hits: list[Hit]

    def top(self, budget: int) -> pd.DataFrame:
        """The alert-budget selector: the ``budget`` highest-scoring clients."""
        return self.alerts.head(budget)


def _reasons(
    client: str,
    hits: list[Hit],
    amount: AmountFlag | None,
    digits: tuple[float, str, int] | None,
    forest: float,
) -> list[dict[str, Any]]:
    reasons: list[dict[str, Any]] = [
        {"source": "rule", "rule": hit.rule, "evidence": hit.evidence} for hit in hits
    ]
    if amount is not None and amount.z >= 3:
        reasons.append(
            {
                "source": "amount",
                "detail": f"unusually large {amount.category} amount",
                "amount": amount.amount,
                "date": amount.event_date,
                "robust_z": round(amount.z, 2),
            }
        )
    if digits is not None and digits[1] == "nonconformity":
        reasons.append(
            {
                "source": "benford",
                "detail": "leading digits far from Benford's law",
                "mad": round(digits[0], 4),
                "amounts": digits[2],
            }
        )
    if forest >= 0.99:
        reasons.append(
            {"source": "isolation_forest", "detail": "isolated by the forest (top percentile)"}
        )
    return reasons


def combine(
    clients: pd.Series,
    *,
    hits: list[Hit],
    rules: RuleSet,
    forest: pd.Series,
    amounts: dict[str, AmountFlag],
    digits: dict[str, tuple[float, str, int]],
) -> Queue:
    """Rank every client by the combined score; ties broken by client id."""
    index = pd.Index(clients.astype(str).sort_values().unique(), name="client_id")
    total = sum(rule.weight for rule in rules.rules)
    by_client: dict[str, list[Hit]] = {}
    for hit in hits:
        by_client.setdefault(hit.client_id, []).append(hit)
    rule_score = pd.Series(
        [sum(rules.get(h.rule).weight for h in by_client.get(c, [])) / total for c in index],
        index=index,
    )
    forest_rank = _rank(forest.reindex(index))
    amount_rank = _rank(pd.Series({c: f.z for c, f in amounts.items()}).reindex(index))
    benford_rank = _rank(pd.Series({c: d[0] for c, d in digits.items()}).reindex(index)).where(
        pd.Series(index.isin(list(digits)), index=index), 0.0
    )
    weights = rules.combine
    score = (
        weights.rules * rule_score
        + weights.isolation_forest * forest_rank
        + weights.amounts * amount_rank
        + weights.benford * benford_rank
    )
    statistical = (
        weights.isolation_forest * forest_rank
        + weights.amounts * amount_rank
        + weights.benford * benford_rank
    )
    frame = pd.DataFrame(
        {
            "client_id": index,
            "score": score.to_numpy(),
            "statistical_score": statistical.to_numpy(),
            "rule_score": rule_score.to_numpy(),
            "rules": [sorted({h.rule for h in by_client.get(c, [])}) for c in index],
            "reasons": [
                _reasons(
                    c, by_client.get(c, []), amounts.get(c), digits.get(c), float(forest_rank[c])
                )
                for c in index
            ],
        }
    )
    frame = frame.sort_values(["score", "client_id"], ascending=[False, True], kind="stable")
    frame["rank"] = np.arange(1, len(frame) + 1)
    return Queue(alerts=frame.reset_index(drop=True), pseudo_labels=pseudo_labels(hits), hits=hits)


def precision_at(ranked: pd.Series, positives: set[str], k: int) -> float:
    """Share of the top ``k`` clients that are positives."""
    top = ranked.head(k).astype(str)
    return float(top.isin(positives).mean()) if len(top) else float("nan")
