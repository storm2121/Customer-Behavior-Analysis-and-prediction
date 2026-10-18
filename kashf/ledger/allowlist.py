"""Which claims computed on real data may enter the public repository.

This list is the whole of the public surface. A claim id that is not here can be computed, rendered
in the private report and discussed in an interview, but its numbers never reach a tracked file.
Adding an id is a privacy decision: it needs aggregate values only, no identifiers, and every value
computed on at least ``MIN_PUBLIC_CELL_N`` clients, which ``kashf ledger build --public`` and the
privacy guard both enforce.
"""

from __future__ import annotations

PUBLIC_CLAIM_IDS: frozenset[str] = frozenset(
    {
        "channel_mix",
        "age_digital_divide",
        "weekday_pattern",
        "benford_purchases",
    }
)

MIN_PUBLIC_CELL_N = 50
"""Smallest number of clients behind any published value (a k-anonymity style floor)."""


def is_public(claim_id: str) -> bool:
    """True if ``claim_id`` may be written to ``docs/ledger/claims.public.json``."""
    return claim_id in PUBLIC_CLAIM_IDS
