"""The rules engine: triage rules configured in YAML, each returning hits with their evidence.

Rules read the event table (branch deposits and withdrawals, ``int_events``), the client-month
panel (``int_client_month``) and the client table (``client_360``). Every hit names the rule, the
client and the evidence a reviewer needs (dates, amounts, counts), so an alert can always be
explained. Rule kinds:

- ``large_deposit``: a deposit at or above ``threshold``;
- ``structuring``: at least ``min_deposits`` deposits each below ``threshold`` and together at or
  above it within ``window_days``;
- ``velocity``: at least ``min_deposits`` deposits within ``window_days``;
- ``pass_through``: a deposit of at least ``min_deposit`` followed, within ``window_days``, by a
  branch withdrawal of at least ``min_share`` of it;
- ``dormant_then_burst``: a month with at least ``min_events`` events after at least
  ``min_gap_months`` months without any;
- ``cash_intensive``: branch deposits of at least ``min_deposit_amount`` and at least ``ratio``
  times the client's card spend.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field

RULES_FILE = Path(__file__).with_name("rules.yaml")

Kind = Literal[
    "large_deposit",
    "structuring",
    "velocity",
    "pass_through",
    "dormant_then_burst",
    "cash_intensive",
]


class Rule(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    kind: Kind
    family: str
    weight: float = Field(gt=0)
    description: str
    params: dict[str, float]


class Combine(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    rules: float
    isolation_forest: float
    amounts: float
    benford: float


class BenfordSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    min_amounts: int


class RuleSet(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    rules: list[Rule]
    combine: Combine
    benford: BenfordSettings

    def get(self, rule_id: str) -> Rule:
        return next(rule for rule in self.rules if rule.id == rule_id)


def load_rules(path: Path = RULES_FILE) -> RuleSet:
    return RuleSet.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


@dataclass(frozen=True)
class Hit:
    rule: str
    family: str
    client_id: str
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Inputs:
    """What the rules read: branch deposits and withdrawals, the monthly panel, card spend."""

    deposits: pd.DataFrame
    withdrawals: pd.DataFrame
    months: pd.DataFrame
    card_spend: pd.Series

    @classmethod
    def from_tables(
        cls, events: pd.DataFrame, months: pd.DataFrame, clients: pd.DataFrame
    ) -> Inputs:
        branch = events[(events["channel"] == "branch") & events["amount"].notna()]
        columns = ["client_id", "event_date", "amount"]

        def pick(kind: str) -> pd.DataFrame:
            rows = branch[branch["kind_group"] == kind][columns].copy()
            rows["event_date"] = pd.to_datetime(rows["event_date"])
            return rows.sort_values(["client_id", "event_date", "amount"]).reset_index(drop=True)

        return cls(
            deposits=pick("deposit"),
            withdrawals=pick("withdrawal"),
            months=months,
            card_spend=clients.set_index("client_id")["card_spend"].astype(float),
        )


def _day(value: Any) -> str:
    return pd.Timestamp(value).date().isoformat()


def _windows(rows: pd.DataFrame, days: int) -> list[tuple[int, int]]:
    """For each row (sorted by date), the slice of rows within ``days`` from it, as index pairs."""
    dates = rows["event_date"].to_numpy(dtype="datetime64[D]")
    ends = np.searchsorted(dates, dates + np.timedelta64(days - 1, "D"), side="right")
    return [(start, int(end)) for start, end in enumerate(ends)]


def large_deposit(rule: Rule, data: Inputs) -> list[Hit]:
    threshold = rule.params["threshold"]
    big = data.deposits[data.deposits["amount"] >= threshold]
    return [
        Hit(
            rule.id,
            rule.family,
            str(client),
            {
                "deposits": len(rows),
                "largest": float(rows["amount"].max()),
                "first_date": _day(rows["event_date"].min()),
                "threshold": threshold,
            },
        )
        for client, rows in big.groupby("client_id", sort=True)
    ]


def structuring(rule: Rule, data: Inputs) -> list[Hit]:
    threshold, minimum = rule.params["threshold"], int(rule.params["min_deposits"])
    days = int(rule.params["window_days"])
    hits = []
    below = data.deposits[data.deposits["amount"] < threshold]
    for client, group in below.groupby("client_id", sort=True):
        rows = group.reset_index(drop=True)
        amounts = rows["amount"].to_numpy()
        best: tuple[int, int] | None = None
        for start, end in _windows(rows, days):
            if end - start >= minimum and amounts[start:end].sum() >= threshold:
                best = (start, end)
                break
        if best is not None:
            start, end = best
            hits.append(
                Hit(
                    rule.id,
                    rule.family,
                    str(client),
                    {
                        "deposits": end - start,
                        "total": float(amounts[start:end].sum()),
                        "largest": float(amounts[start:end].max()),
                        "from": _day(rows["event_date"].iloc[start]),
                        "to": _day(rows["event_date"].iloc[end - 1]),
                        "threshold": threshold,
                    },
                )
            )
    return hits


def velocity(rule: Rule, data: Inputs) -> list[Hit]:
    minimum, days = int(rule.params["min_deposits"]), int(rule.params["window_days"])
    hits = []
    for client, group in data.deposits.groupby("client_id", sort=True):
        rows = group.reset_index(drop=True)
        counts = [(end - start, start, end) for start, end in _windows(rows, days)]
        count, start, end = max(counts)
        if count >= minimum:
            hits.append(
                Hit(
                    rule.id,
                    rule.family,
                    str(client),
                    {
                        "deposits": count,
                        "from": _day(rows["event_date"].iloc[start]),
                        "to": _day(rows["event_date"].iloc[end - 1]),
                        "window_days": days,
                    },
                )
            )
    return hits


def pass_through(rule: Rule, data: Inputs) -> list[Hit]:
    minimum, days = rule.params["min_deposit"], int(rule.params["window_days"])
    share = rule.params["min_share"]
    deposits = data.deposits[data.deposits["amount"] >= minimum]
    pairs = deposits.merge(data.withdrawals, on="client_id", suffixes=("_in", "_out"))
    lag = (pairs["event_date_out"] - pairs["event_date_in"]).dt.days
    pairs = pairs[(lag > 0) & (lag <= days) & (pairs["amount_out"] >= share * pairs["amount_in"])]
    hits = []
    for client, rows in pairs.groupby("client_id", sort=True):
        first = rows.sort_values(["event_date_in", "event_date_out"]).iloc[0]
        hits.append(
            Hit(
                rule.id,
                rule.family,
                str(client),
                {
                    "deposit": float(first["amount_in"]),
                    "deposit_date": _day(first["event_date_in"]),
                    "withdrawal": float(first["amount_out"]),
                    "withdrawal_date": _day(first["event_date_out"]),
                    "pairs": len(rows),
                },
            )
        )
    return hits


def dormant_then_burst(rule: Rule, data: Inputs) -> list[Hit]:
    gap, minimum = rule.params["min_gap_months"], rule.params["min_events"]
    months = data.months
    bursts = months[(months["n_events"] >= minimum) & (months["gap_months_before"] >= gap)]
    return [
        Hit(
            rule.id,
            rule.family,
            str(client),
            {
                "month": _day(rows["month_start"].min()),
                "events": int(rows.sort_values("month_start")["n_events"].iloc[0]),
                "silent_months_before": int(
                    rows.sort_values("month_start")["gap_months_before"].iloc[0]
                ),
            },
        )
        for client, rows in bursts.groupby("client_id", sort=True)
    ]


def cash_intensive(rule: Rule, data: Inputs) -> list[Hit]:
    ratio, minimum = rule.params["ratio"], rule.params["min_deposit_amount"]
    totals = data.deposits.groupby("client_id")["amount"].sum()
    spend = data.card_spend.reindex(totals.index).fillna(0.0)
    flagged = totals[(totals >= minimum) & (totals >= ratio * spend)]
    spend_of = {str(client): float(value) for client, value in spend.items()}
    return [
        Hit(
            rule.id,
            rule.family,
            str(client),
            {
                "branch_deposits": float(total),
                "card_spend": spend_of[str(client)],
                "ratio": ratio,
            },
        )
        for client, total in flagged.sort_index().items()
    ]


ENGINES: dict[str, Callable[[Rule, Inputs], list[Hit]]] = {
    "large_deposit": large_deposit,
    "structuring": structuring,
    "velocity": velocity,
    "pass_through": pass_through,
    "dormant_then_burst": dormant_then_burst,
    "cash_intensive": cash_intensive,
}


def evaluate(rules: RuleSet, data: Inputs) -> list[Hit]:
    """Every hit of every rule, in rule order and then client order."""
    return [hit for rule in rules.rules for hit in ENGINES[rule.kind](rule, data)]


def pseudo_labels(hits: list[Hit]) -> set[str]:
    """Clients flagged by rules of at least two different families."""
    families: dict[str, set[str]] = {}
    for hit in hits:
        families.setdefault(hit.client_id, set()).add(hit.family)
    return {client for client, found in families.items() if len(found) >= 2}
