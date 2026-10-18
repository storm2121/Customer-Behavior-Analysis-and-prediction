"""Code vocabularies, read from the dbt seeds.

The seeds in ``dbt/seeds/codes_*.csv`` are the single definition of every categorical value: dbt
staging joins them, and this module exposes them to the contracts, the data-quality checks and the
synthetic generator. Keeping one definition means a new label is added in exactly one reviewed file.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from kashf.config import find_repo_root


def seeds_dir() -> Path:
    return find_repo_root() / "dbt" / "seeds"


def _rows(name: str) -> list[dict[str, str]]:
    path = seeds_dir() / f"{name}.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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
