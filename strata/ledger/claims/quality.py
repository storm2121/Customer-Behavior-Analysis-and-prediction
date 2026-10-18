"""Claims about the data itself: undecidable repeats and a join that cannot work."""

from __future__ import annotations

import numpy as np
import pandas as pd

from strata.ledger.bootstrap import percentile_interval
from strata.ledger.claims._common import require
from strata.ledger.model import Unit
from strata.ledger.registry import ClaimContext, ClaimResult, claim


@claim(
    "repeat_rows",
    title="Exact repeats that no one can classify",
    question="How many event rows repeat an earlier row on every field?",
    method=(
        "Rows flagged as exact repeats in staging (same client, date, amount, label and origin; "
        "the definition shared with the data-quality report), per channel; client bootstrap "
        "interval on the card share."
    ),
)
def repeat_rows(context: ClaimContext) -> ClaimResult:
    events = context.table("intermediate.int_events")
    require(len(events) > 0, "no events")
    values: dict[str, float] = {}
    units: dict[str, Unit] = {}
    for channel in ("card", "online", "branch"):
        rows = events[events["channel"] == channel]
        if rows.empty:
            continue
        values[f"{channel}_repeat_share"] = float(rows["is_repeat"].mean())
        values[f"{channel}_repeat_rows"] = float(rows["is_repeat"].sum())
        units[f"{channel}_repeat_share"] = Unit.SHARE
        units[f"{channel}_repeat_rows"] = Unit.COUNT
    card = events[events["channel"] == "card"]
    intervals: dict[str, tuple[float, float]] = {}
    if not card.empty:
        clients = pd.Index(pd.unique(card["client_id"]))
        codes = clients.get_indexer(pd.Index(card["client_id"]))
        repeats = np.bincount(
            codes, weights=card["is_repeat"].to_numpy(dtype=np.float64), minlength=len(clients)
        )
        card_rows = np.bincount(codes, minlength=len(clients)).astype(np.float64)
        boot = context.bootstrap(len(clients))
        intervals["card_repeat_share"] = percentile_interval(boot.ratio(repeats, card_rows))
    n = int(events["client_id"].nunique())
    return ClaimResult(
        values=values,
        units=units,
        ci=intervals,
        value_n=dict.fromkeys(values, n),
        n=n,
        n_events=len(events),
        caveats=[
            (
                "Without a transaction id or a time of day a repeat is either a double posting or "
                "a real second purchase of the same amount on the same day; rows are flagged, "
                "never dropped."
            ),
        ],
    )


@claim(
    "aggregate_join_coverage",
    title="The join the 2024 analysis relied on matches nothing",
    question="How many identifiers of the aggregate file appear in the client file?",
    method="Identifiers of the aggregate sample (stg_aggregate_clients) found in stg_clients.",
)
def aggregate_join_coverage(context: ClaimContext) -> ClaimResult:
    aggregate = context.table("staging.stg_aggregate_clients")
    clients = context.table("staging.stg_clients")
    require(len(aggregate) > 0, "empty aggregate sample")
    matches = int(aggregate["client_id"].isin(set(clients["client_id"])).sum())
    values = {"matching_identifiers": float(matches), "matching_share": matches / len(aggregate)}
    return ClaimResult(
        values=values,
        units={"matching_identifiers": Unit.COUNT, "matching_share": Unit.SHARE},
        value_n=dict.fromkeys(values, len(aggregate)),
        n=len(aggregate),
        caveats=[
            (
                "The two files are different samples; the warehouse never joins them, and the "
                "adoption models use the aggregate sample on its own."
            ),
        ],
    )
