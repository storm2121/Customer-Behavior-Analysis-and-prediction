"""Which claims computed on real data may enter the public repository, and on what terms.

This mapping is the whole of the public surface. A claim id that is not here can be computed,
rendered in the private report and discussed in an interview, but nothing of it reaches a tracked
file. Adding an id is a privacy decision, made by the architect, in one of two modes:

- ``full``: the claim's aggregate values, intervals and robustness result, every value computed on
  at least ``MIN_PUBLIC_CELL_N`` clients and small strata suppressed;
- ``verdict_only``: what the robustness checker concluded (the verdict, the stratifier names, the
  stratifiers that reverse the effect, and a number-free explanation), never a value, an interval
  or a group size. It shows the checker working without publishing what it measured.

:func:`ledger_violations` states the terms once; ``strata ledger build --public`` refuses to write,
``strata ledger check`` fails and the privacy guard refuses to commit a public ledger that breaks
them.
"""

from __future__ import annotations

import enum
import re
from collections.abc import Iterable, Mapping
from typing import Any


class Publication(enum.StrEnum):
    FULL = "full"
    VERDICT_ONLY = "verdict_only"


PUBLICATION: Mapping[str, Publication] = {
    "channel_mix": Publication.FULL,
    "age_digital_divide": Publication.FULL,
    "weekday_pattern": Publication.FULL,
    "benford_purchases": Publication.FULL,
    "spend_concentration": Publication.FULL,
    "silence_by_channel": Publication.VERDICT_ONLY,
}

PUBLIC_CLAIM_IDS: frozenset[str] = frozenset(
    claim_id for claim_id, mode in PUBLICATION.items() if mode is Publication.FULL
)
"""Claims published with their numbers."""

VERDICT_ONLY_IDS: frozenset[str] = frozenset(
    claim_id for claim_id, mode in PUBLICATION.items() if mode is Publication.VERDICT_ONLY
)
"""Headlines published by their robustness verdict only."""

MIN_PUBLIC_CELL_N = 50
"""Smallest number of clients behind any published value or stratum (a k-anonymity style floor)."""

VERDICT_KEYS = frozenset(
    {
        "id",
        "title",
        "question",
        "publication",
        "verdict",
        "stratifiers",
        "reversing_stratifiers",
        "mechanism",
        "data_fingerprint",
        "source_kind",
        "computed_at",
        "strata_version",
    }
)
"""Everything a verdict-only record may hold; provenance fields are the only ones with digits."""

_WORDS = ("title", "question", "verdict", "mechanism", "stratifiers", "reversing_stratifiers")
_DIGIT = re.compile(r"\d")


def is_public(claim_id: str) -> bool:
    """True if ``claim_id`` may be published with its numbers."""
    return claim_id in PUBLIC_CLAIM_IDS


def is_verdict_only(claim_id: str) -> bool:
    """True if ``claim_id`` may be published by its robustness verdict only."""
    return claim_id in VERDICT_ONLY_IDS


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
    """Reasons why ``records`` (full claim records as JSON-like mappings) may not be published."""
    problems: list[str] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping) or "id" not in record:
            problems.append(f"claim #{index} has no id")
            continue
        claim_id = str(record["id"])
        if is_verdict_only(claim_id):
            problems.append(f"claim '{claim_id}' may only be published as a verdict")
            continue
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


def _strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    return [] if value is None else [str(value)]


def verdict_violations(verdicts: Iterable[Any]) -> list[str]:
    """Reasons why ``verdicts`` (verdict-only records as JSON-like mappings) may not be published.

    A verdict-only record holds nothing but the keys of :data:`VERDICT_KEYS`, and none of its
    words (title, question, verdict, explanation, stratifier names) carries a digit, so no value,
    interval or group size can travel inside a sentence either.
    """
    problems: list[str] = []
    for index, record in enumerate(verdicts):
        if not isinstance(record, Mapping) or "id" not in record:
            problems.append(f"verdict #{index} has no id")
            continue
        claim_id = str(record["id"])
        if not is_verdict_only(claim_id):
            problems.append(
                f"verdict '{claim_id}' is not allow-listed for verdict-only publication"
            )
            continue
        extra = sorted(set(record) - VERDICT_KEYS)
        if extra:
            problems.append(f"verdict '{claim_id}' carries fields beyond the verdict: {extra}")
        if record.get("publication") != Publication.VERDICT_ONLY.value:
            problems.append(f"verdict '{claim_id}' is not marked verdict_only")
        words = [text for key in _WORDS for text in _strings(record.get(key))]
        if any(_DIGIT.search(text) for text in words):
            problems.append(f"verdict '{claim_id}' has a number in its words")
    return problems


def ledger_violations(payload: object) -> list[str]:
    """Every reason why a public ledger file (parsed JSON) may not be published."""
    if not isinstance(payload, Mapping):
        return ["ledger JSON is not an object"]
    claims = payload.get("claims", [])
    verdicts = payload.get("verdicts", [])
    if not isinstance(claims, list) or not isinstance(verdicts, list):
        return ["ledger JSON has no list of claims or of verdicts"]
    return [*public_violations(claims), *verdict_violations(verdicts)]
