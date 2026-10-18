"""Which claims computed on real data may enter the public repository, and on what terms.

This list is the whole of the public surface. A claim id that is not here can be computed, rendered
in the private report and discussed in an interview, but its numbers never reach a tracked file.
Adding an id is a privacy decision: it needs aggregate values only, no identifiers, and every value
computed on at least ``MIN_PUBLIC_CELL_N`` clients. :func:`public_violations` states those terms
once; ``strata ledger build --public`` refuses to write and the privacy guard refuses to commit any
record that breaks them.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

PUBLIC_CLAIM_IDS: frozenset[str] = frozenset(
    {
        "channel_mix",
        "age_digital_divide",
        "weekday_pattern",
        "benford_purchases",
    }
)

MIN_PUBLIC_CELL_N = 50
"""Smallest number of clients behind any published value or stratum (a k-anonymity style floor)."""


def is_public(claim_id: str) -> bool:
    """True if ``claim_id`` may be written to ``docs/ledger/claims.public.json``."""
    return claim_id in PUBLIC_CLAIM_IDS


def _small(size: object) -> bool:
    return not isinstance(size, int) or isinstance(size, bool) or size < MIN_PUBLIC_CELL_N


def _stratum_violations(claim_id: str, robustness: object) -> list[str]:
    """Group sizes of a published robustness result, overall and per stratum, reach the floor."""
    if not isinstance(robustness, Mapping):
        return []
    problems = []
    if _small(robustness.get("n_treatment")) or _small(robustness.get("n_reference")):
        problems.append(
            f"claim '{claim_id}' compares a group of fewer than the minimum number of clients"
        )
    for stratifier in robustness.get("stratifiers", []) or []:
        if not isinstance(stratifier, Mapping):
            continue
        for stratum in stratifier.get("strata", []) or []:
            if not isinstance(stratum, Mapping) or stratum.get("suppressed"):
                continue
            if _small(stratum.get("n_treatment")) or _small(stratum.get("n_reference")):
                problems.append(
                    f"claim '{claim_id}' publishes a stratum with a group of fewer than "
                    "the minimum number of clients; suppress it"
                )
    return problems


def public_violations(records: Iterable[Any]) -> list[str]:
    """Reasons why ``records`` (claim records as JSON-like mappings) may not be published."""
    problems: list[str] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping) or "id" not in record:
            problems.append(f"claim #{index} has no id")
            continue
        claim_id = str(record["id"])
        if not is_public(claim_id):
            problems.append(f"claim '{claim_id}' is not allow-listed")
            continue
        sizes: list[object] = [record.get("n")]
        value_n = record.get("value_n")
        if isinstance(value_n, Mapping):
            sizes.extend(value_n.values())
        if any(_small(size) for size in sizes):
            problems.append(
                f"claim '{claim_id}' has a value computed on fewer than "
                "the minimum number of clients for publication"
            )
        problems.extend(_stratum_violations(claim_id, record.get("robustness")))
    return problems
