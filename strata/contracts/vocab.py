"""Code vocabularies, bands and cutoffs, read from the dbt seeds.

The seeds in ``dbt/seeds/`` are the single definition of every categorical value, every band and
every time cut: dbt joins them in SQL, and this module exposes them to the contracts, the
data-quality checks, the synthetic generator and the ledger. Keeping one definition means a label,
a band edge or a cutoff changes in exactly one reviewed file.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from functools import cache
from itertools import pairwise
from pathlib import Path

from strata.config import find_repo_root


def seeds_dir() -> Path:
    return find_repo_root() / "dbt" / "seeds"


def seed_rows(name: str) -> list[dict[str, str]]:
    """The rows of ``dbt/seeds/<name>.csv`` as dictionaries."""
    path = seeds_dir() / f"{name}.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


_rows = seed_rows


def _flag(value: str) -> bool:
    return value.strip().lower() == "true"


@dataclass(frozen=True, slots=True)
class CardCategory:
    label: str
    key: str
    direction: str
    is_spend: bool


@dataclass(frozen=True, slots=True)
class OnlineService:
    label: str
    key: str
    direction: str
    zero_amount_expected: bool


@dataclass(frozen=True, slots=True)
class BranchOperation:
    label: str
    key: str
    category: str
    assumed_direction: str
    is_cash_deposit: bool
    semantics: str


@cache
def card_categories() -> tuple[CardCategory, ...]:
    return tuple(
        CardCategory(
            label=row["label"],
            key=row["category_key"],
            direction=row["direction"],
            is_spend=_flag(row["is_spend"]),
        )
        for row in _rows("codes_card_category")
    )


@cache
def card_origins() -> tuple[str, ...]:
    return tuple(row["label"] for row in _rows("codes_card_origin"))


@cache
def online_services() -> tuple[OnlineService, ...]:
    return tuple(
        OnlineService(
            label=row["label"],
            key=row["service_key"],
            direction=row["direction"],
            zero_amount_expected=_flag(row["zero_amount_expected"]),
        )
        for row in _rows("codes_online_service")
    )


@cache
def branch_operations() -> tuple[BranchOperation, ...]:
    return tuple(
        BranchOperation(
            label=row["label"],
            key=row["operation_key"],
            category=row["category"],
            assumed_direction=row["assumed_direction"],
            is_cash_deposit=_flag(row["is_cash_deposit"]),
            semantics=row["semantics"],
        )
        for row in _rows("codes_branch_operation")
    )


@cache
def branch_directions() -> tuple[str, ...]:
    return tuple(row["code"] for row in _rows("codes_branch_direction"))


@cache
def client_codes(column: str) -> tuple[int, ...]:
    """Known integer codes of a client-file column (``genre``, ``situation_familiale``)."""
    return tuple(int(row["code"]) for row in _rows("codes_client") if row["column_name"] == column)


def card_label(key: str) -> str:
    """The extract's label for a card category key (``purchase`` → ``PURCHASES & SERVICES``)."""
    return next(category.label for category in card_categories() if category.key == key)


def online_label(key: str) -> str:
    return next(service.label for service in online_services() if service.key == key)


def branch_labels(category: str) -> tuple[str, ...]:
    return tuple(op.label for op in branch_operations() if op.category == category)


@dataclass(frozen=True, slots=True)
class Band:
    """A half-open interval ``[lower, upper)`` with the label used in every report."""

    label: str
    lower: int
    upper: int

    def __contains__(self, value: object) -> bool:
        return isinstance(value, int | float) and self.lower <= value < self.upper


def _bands(name: str) -> tuple[Band, ...]:
    bands = tuple(
        Band(label=row["band"], lower=int(row["lower"]), upper=int(row["upper"]))
        for row in _rows(name)
    )
    for previous, current in pairwise(bands):
        if previous.upper != current.lower:
            msg = f"{name}: bands must be contiguous"
            raise ValueError(msg)
    return bands


@cache
def age_bands() -> tuple[Band, ...]:
    return _bands("bands_age")


@cache
def tenure_bands() -> tuple[Band, ...]:
    return _bands("bands_tenure")


def band_of(value: float | None, bands: tuple[Band, ...]) -> str | None:
    """The label of the band holding ``value``, or ``None`` when missing or out of range."""
    if value is None:
        return None
    return next((band.label for band in bands if value in band), None)


@dataclass(frozen=True, slots=True)
class Cutoff:
    """A time cut for silence: features up to ``cutoff``, label over the outcome window."""

    cutoff: date
    outcome_start: date
    outcome_end: date
    role: str


@cache
def silence_cutoffs() -> tuple[Cutoff, ...]:
    return tuple(
        Cutoff(
            cutoff=date.fromisoformat(row["cutoff_date"]),
            outcome_start=date.fromisoformat(row["outcome_start"]),
            outcome_end=date.fromisoformat(row["outcome_end"]),
            role=row["role"],
        )
        for row in _rows("silence_cutoffs")
    )


def primary_cutoff() -> Cutoff:
    return next(cutoff for cutoff in silence_cutoffs() if cutoff.role == "primary")


@cache
def calendar_rows() -> tuple[tuple[str, str, date, date], ...]:
    """``(key, label, start, end)`` of the special periods (read by :mod:`strata.holidays`)."""
    return tuple(
        (
            row["period_key"],
            row["label"],
            date.fromisoformat(row["start_date"]),
            date.fromisoformat(row["end_date"]),
        )
        for row in _rows("calendar_periods")
    )


@cache
def holiday_rows() -> tuple[tuple[date, str], ...]:
    """``(date, label)`` of the bank holidays other than the Eids (national and religious)."""
    return tuple(
        (date.fromisoformat(row["holiday_date"]), row["label"]) for row in _rows("bank_holidays")
    )
