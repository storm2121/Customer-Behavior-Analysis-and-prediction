"""Claims about leading digits beyond the public first-digit test of purchases (M4)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from strata.analysis import benford
from strata.ledger.bootstrap import percentile_interval
from strata.ledger.claims._common import client_index, require
from strata.ledger.model import Unit
from strata.ledger.registry import ClaimContext, ClaimResult, claim

BENFORD_SERIES = {
    "card_purchase": ("card", "purchase"),
    "card_ecommerce": ("card", "ecommerce"),
    "card_withdrawal": ("card", "withdrawal"),
    "card_deposit": ("card", "deposit"),
    "online_bill_pay": ("online", "bill_pay"),
    "branch_deposit": ("branch", "deposit"),
}
MIN_AMOUNTS = 100
"""Fewest amounts for a first-digit test of a category."""
MIN_TWO_DIGIT_AMOUNTS = 1000
"""Fewest amounts for a first-two-digit test: ninety cells need far more amounts than nine."""


def _amounts(context: ClaimContext) -> pd.DataFrame:
    events = context.table("intermediate.int_events")
    return events[~events["is_repeat"] & (events["amount"] > 0)].reset_index(drop=True)


@dataclass
class _Collected:
    values: dict[str, float] = field(default_factory=dict)
    intervals: dict[str, tuple[float, float]] = field(default_factory=dict)
    units: dict[str, Unit] = field(default_factory=dict)
    value_n: dict[str, int] = field(default_factory=dict)
    meta: dict[str, str] = field(default_factory=dict)

    def add_test(
        self,
        prefix: str,
        test: benford.BenfordTest,
        mad_draws: np.ndarray,
        n_clients: int,
    ) -> None:
        self.values[f"{prefix}_mad"] = test.mad
        self.intervals[f"{prefix}_mad"] = percentile_interval(mad_draws)
        self.values[f"{prefix}_p_value"] = test.p_value
        self.units[f"{prefix}_mad"] = Unit.NUMBER
        self.units[f"{prefix}_p_value"] = Unit.PVALUE
        self.value_n[f"{prefix}_mad"] = self.value_n[f"{prefix}_p_value"] = n_clients
        self.meta[f"{prefix}_conformity"] = test.conformity
        self.meta[f"{prefix}_amounts"] = str(test.n)


def _mad_draws(
    context: ClaimContext,
    codes: np.ndarray,
    n_clients: int,
    *,
    digits: np.ndarray,
    support: np.ndarray,
    expected: np.ndarray,
    stream: str,
) -> benford.FloatArray:
    """Client-bootstrap draws of the mean absolute deviation from ``expected``."""
    matrix = benford.digit_matrix(codes, digits, n_clients, support)
    draws = context.bootstrap(n_clients, stream=stream).ratio(matrix, matrix.sum(axis=1)[:, None])
    return np.asarray(np.abs(draws - expected).mean(axis=1), dtype=np.float64)


@claim(
    "benford_by_category",
    title="Leading digits across categories",
    question="Which categories of amounts follow Benford's law, and which are shaped by rules?",
    method=(
        "First significant digit of every positive amount in each category with at least "
        "a hundred amounts, and first two digits where there are at least a thousand (int_events, "
        "exact repeats excluded): mean absolute deviation from Benford's law with Nigrini's "
        "conformity bands for each test and a chi-square test; client-bootstrap intervals on the "
        "deviation."
    ),
)
def benford_by_category(context: ClaimContext) -> ClaimResult:
    events = _amounts(context)
    collected = _Collected()
    for name, (channel, kind) in BENFORD_SERIES.items():
        rows = events[(events["channel"] == channel) & (events["kind_group"] == kind)]
        if len(rows) < MIN_AMOUNTS:
            continue
        amounts = rows["amount"].to_numpy(dtype=np.float64)
        codes, clients = client_index(rows["client_id"])
        draws = _mad_draws(
            context,
            codes,
            len(clients),
            digits=benford.first_digits(amounts),
            support=benford.FIRST_DIGITS,
            expected=benford.EXPECTED_FIRST,
            stream=name,
        )
        collected.add_test(name, benford.first_digit_test(amounts), draws, len(clients))
        if len(rows) >= MIN_TWO_DIGIT_AMOUNTS:
            draws = _mad_draws(
                context,
                codes,
                len(clients),
                digits=benford.first_two_digits(amounts),
                support=benford.FIRST_TWO_DIGITS,
                expected=benford.EXPECTED_FIRST_TWO,
                stream=f"{name}/two",
            )
            test = benford.first_two_digit_test(amounts)
            collected.add_test(f"{name}_two_digit", test, draws, len(clients))
    require(bool(collected.values), "no category with enough amounts")
    return ClaimResult(
        values=collected.values,
        units=collected.units,
        ci=collected.intervals,
        value_n=collected.value_n,
        n=int(events["client_id"].nunique()),
        meta=collected.meta,
        caveats=[
            (
                "Benford's law describes amounts produced by many multiplicative processes; "
                "withdrawals (banknote multiples) and set prices are not, so their nonconformity "
                "is expected and harmless."
            ),
            "First-digit and first-two-digit tests have their own conformity bands (Nigrini).",
        ],
    )


@claim(
    "benford_two_digits",
    title="The first two digits of card purchases",
    question="Where do purchase amounts depart from Benford's law at the level of two digits?",
    method=(
        "First two significant digits of positive card purchase amounts (int_events, exact repeats "
        "excluded) against Benford's law: mean absolute deviation with Nigrini's two-digit bands, "
        "and the three most over-represented leading pairs; client-bootstrap intervals."
    ),
)
def benford_two_digits(context: ClaimContext) -> ClaimResult:
    events = _amounts(context)
    purchases = events[(events["channel"] == "card") & (events["kind_group"] == "purchase")]
    require(len(purchases) >= MIN_AMOUNTS, "too few purchases")
    amounts = purchases["amount"].to_numpy(dtype=np.float64)
    test = benford.first_two_digit_test(amounts)
    codes, clients = client_index(purchases["client_id"])
    pairs = benford.first_two_digits(amounts)
    matrix = benford.digit_matrix(codes, pairs, len(clients), benford.FIRST_TWO_DIGITS)
    boot = context.bootstrap(len(clients))
    draws = boot.ratio(matrix, matrix.sum(axis=1)[:, None])
    excess = test.observed - test.expected
    values = {"mad": test.mad, "p_value": test.p_value}
    intervals = {
        "mad": percentile_interval(np.abs(draws - benford.EXPECTED_FIRST_TWO).mean(axis=1))
    }
    units = {"mad": Unit.NUMBER, "p_value": Unit.PVALUE}
    meta = {"conformity": test.conformity}
    for rank, index in enumerate(np.argsort(-excess)[:3], start=1):
        key = f"excess_rank_{rank}"
        values[key] = float(excess[index])
        intervals[key] = percentile_interval(draws[:, index] - test.expected[index])
        units[key] = Unit.DIFFERENCE
        meta[f"pair_rank_{rank}"] = str(int(benford.FIRST_TWO_DIGITS[index]))
    return ClaimResult(
        values=values,
        units=units,
        ci=intervals,
        value_n=dict.fromkeys(values, len(clients)),
        n=len(clients),
        n_events=len(purchases),
        meta=meta,
        caveats=["Two-digit tests need many amounts; the bands are Nigrini's for this test."],
    )
