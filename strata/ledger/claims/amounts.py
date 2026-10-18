"""Claims about amounts: leading digits, online card spend abroad, and concentration of spend."""

from __future__ import annotations

import numpy as np
import pandas as pd

from strata.analysis import benford
from strata.analysis.concentration import gini, top_share
from strata.ledger.bootstrap import percentile_interval, weighted_quantile
from strata.ledger.claims._common import per_client_matrix, require
from strata.ledger.model import Unit
from strata.ledger.registry import ClaimContext, ClaimResult, claim


def _card(context: ClaimContext) -> pd.DataFrame:
    events = context.table("intermediate.int_events")
    card = events[(events["channel"] == "card") & ~events["is_repeat"]]
    return card.reset_index(drop=True)


@claim(
    "benford_purchases",
    title="Leading digits of card purchases",
    question="Do card purchase amounts follow Benford's law, and where do they depart from it?",
    method=(
        "First significant digit of every positive card purchase amount (int_events, physical "
        "merchants, exact repeats excluded) against Benford's law: mean absolute deviation with "
        "Nigrini's conformity bands, chi-square test, and the excess of leading nines; client "
        "bootstrap intervals."
    ),
)
def benford_purchases(context: ClaimContext) -> ClaimResult:
    card = _card(context)
    purchases = card[(card["kind"] == "purchase") & (card["amount"] > 0)].reset_index(drop=True)
    require(len(purchases) > 0, "no purchases")
    digits = benford.first_digits(purchases["amount"].to_numpy(dtype=np.float64))
    test = benford.first_digit_test(purchases["amount"].to_numpy(dtype=np.float64))
    clients = pd.Index(pd.unique(purchases["client_id"]))
    codes = clients.get_indexer(pd.Index(purchases["client_id"])).astype(np.int64)
    matrix = per_client_matrix(codes, digits - 1, len(clients), 9)
    boot = context.bootstrap(len(clients))
    draws = boot.ratio(matrix, matrix.sum(axis=1)[:, None])
    mad_draws = np.abs(draws - benford.EXPECTED_FIRST).mean(axis=1)
    values: dict[str, float] = {}
    intervals: dict[str, tuple[float, float]] = {}
    for digit in range(1, 10):
        values[f"share_digit_{digit}"] = float(test.observed[digit - 1])
        values[f"expected_digit_{digit}"] = float(test.expected[digit - 1])
        intervals[f"share_digit_{digit}"] = percentile_interval(draws[:, digit - 1])
    values["mad"] = test.mad
    intervals["mad"] = percentile_interval(mad_draws)
    values["digit_9_excess"] = float(test.observed[8] - test.expected[8])
    intervals["digit_9_excess"] = percentile_interval(draws[:, 8] - test.expected[8])
    values["chi2"] = test.chi2
    values["p_value"] = test.p_value
    units = dict.fromkeys(values, Unit.SHARE)
    units.update(
        {
            "mad": Unit.NUMBER,
            "digit_9_excess": Unit.DIFFERENCE,
            "chi2": Unit.NUMBER,
            "p_value": Unit.PVALUE,
        }
    )
    return ClaimResult(
        values=values,
        units=units,
        ci=intervals,
        value_n=dict.fromkeys(values, len(clients)),
        n=len(clients),
        n_events=len(purchases),
        meta={
            "conformity": test.conformity,
            "rows_digits": "|".join(f"{digit}:{digit}" for digit in range(1, 10)),
        },
        caveats=[
            (
                "With thousands of amounts the chi-square test rejects for tiny deviations; the "
                "MAD band is the recommended reading."
            ),
            "Prices are set, not drawn: amounts just under a round number produce an excess of "
            "leading nines (psychological pricing), which is not a sign of manipulation."
            if values["digit_9_excess"] > 0
            else "No excess of leading nines is visible, although set prices often produce one.",
        ],
    )


@claim(
    "ecommerce_international_share",
    title="Online card spend goes abroad, in small amounts",
    question="Which share of e-commerce card purchases is international, and how large are they?",
    method=(
        "Share of e-commerce purchases with an international origin, median e-commerce amount, and "
        "the share of card users with at least one e-commerce purchase (int_events, exact repeats "
        "excluded); client bootstrap intervals."
    ),
)
def ecommerce_international_share(context: ClaimContext) -> ClaimResult:
    card = _card(context)
    require(len(card) > 0, "no card events")
    clients = pd.Index(pd.unique(card["client_id"]))
    codes = clients.get_indexer(pd.Index(card["client_id"])).astype(np.int64)
    ecommerce = (card["kind"] == "ecommerce").to_numpy()
    require(ecommerce.any(), "no e-commerce purchases")
    international = (card["origin"] == "international").to_numpy(dtype=np.float64)
    n = len(clients)
    per_client_ecommerce = np.bincount(codes, weights=ecommerce.astype(np.float64), minlength=n)
    per_client_international = np.bincount(codes, weights=international * ecommerce, minlength=n)
    boot = context.bootstrap(n)
    amounts = card.loc[ecommerce, "amount"].to_numpy(dtype=np.float64)
    ecommerce_codes = codes[ecommerce]
    finite = np.isfinite(amounts)
    medians = [
        weighted_quantile(amounts[finite], weights[ecommerce_codes[finite]], 0.5)
        for weights in boot.weights
    ]
    users = (per_client_ecommerce > 0).astype(np.float64)
    values = {
        "international_share": float(per_client_international.sum() / per_client_ecommerce.sum()),
        "median_amount": float(np.nanmedian(amounts)),
        "card_users_with_ecommerce": float(users.mean()),
    }
    intervals = {
        "international_share": percentile_interval(
            boot.ratio(per_client_international, per_client_ecommerce)
        ),
        "median_amount": percentile_interval(medians),
        "card_users_with_ecommerce": percentile_interval(boot.ratio(users, np.ones(n))),
    }
    return ClaimResult(
        values=values,
        units={
            "international_share": Unit.SHARE,
            "median_amount": Unit.AMOUNT,
            "card_users_with_ecommerce": Unit.SHARE,
        },
        ci=intervals,
        value_n=dict.fromkeys(values, n),
        n=n,
        n_events=int(ecommerce.sum()),
        caveats=["Origin is the card network's flag; a foreign merchant may sell in dirhams."],
    )


TOPS = (1, 10, 20)
"""Top groups of spenders, in per cent of the clients with any card spend."""


@claim(
    "spend_concentration",
    title="How concentrated card spend is",
    question="Which share of card spend comes from the heaviest spenders, and what is the Gini?",
    method=(
        "Card spend (physical and online purchases) per client with any spend (client_360): share "
        "of the top 1%, 10% and 20% of spenders and the Gini coefficient; client bootstrap "
        "intervals."
    ),
)
def spend_concentration(context: ClaimContext) -> ClaimResult:
    clients = context.table("marts.client_360")
    spend = clients.loc[clients["card_spend"] > 0, "card_spend"].to_numpy(dtype=np.float64)
    require(spend.size >= 10, "too few spenders")
    boot = context.bootstrap(spend.size)

    def statistics(weights: np.ndarray) -> dict[str, float]:
        tops = {
            f"top_{percent}_share": top_share(spend, percent / 100, weights) for percent in TOPS
        }
        return {**tops, "gini": gini(spend, weights)}

    values = statistics(np.ones(spend.size))
    intervals = boot.intervals(boot.apply(statistics))
    units = dict.fromkeys(values, Unit.SHARE)
    units["gini"] = Unit.NUMBER
    # A top share describes the clients in the top group, so that group's size is the number of
    # clients behind the value (it decides whether the value may be published).
    value_n = {f"top_{percent}_share": -(-percent * spend.size // 100) for percent in TOPS}
    value_n["gini"] = int(spend.size)
    return ClaimResult(
        values=values,
        units=units,
        ci=intervals,
        value_n=value_n,
        n=int(spend.size),
        meta={"rows_tops": "|".join(f"top {p}%:top_{p}_share" for p in TOPS)},
        caveats=[
            (
                "Spend outside the bank (other cards, cash) is invisible; concentration is of this "
                "bank's card spend."
            )
        ],
    )
