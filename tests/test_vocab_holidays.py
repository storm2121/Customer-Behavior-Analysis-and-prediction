"""Code vocabularies (from the dbt seeds) and the shared calendar."""

from __future__ import annotations

from datetime import date

import pytest

from kashf.contracts import vocab
from kashf.holidays import (
    EID_AL_ADHA_2024,
    NATIONAL_HOLIDAYS,
    PRE_EID_AL_ADHA_WEEK_2024,
    RAMADAN_2024,
    SPECIAL_PERIODS,
    Period,
    day_of_month_bin,
    is_bank_holiday,
)


def test_card_categories_cover_the_seven_labels_with_unique_keys() -> None:
    categories = vocab.card_categories()
    assert len({c.label for c in categories}) == len(categories)
    assert len({c.key for c in categories}) == len(categories)
    assert vocab.card_label("withdrawal") == "WITHDRAWAL"
    assert {c.direction for c in categories} == {"debit", "credit"}


def test_online_services_mark_card_service_as_zero_amount() -> None:
    services = vocab.online_services()
    zero = [s for s in services if s.zero_amount_expected]
    assert [s.key for s in zero] == ["card_service"]
    assert vocab.online_label("card_service").startswith("Op")


def test_branch_operations_have_known_categories_and_directions() -> None:
    operations = vocab.branch_operations()
    assert len({op.label for op in operations}) == len(operations)
    assert {op.assumed_direction for op in operations} <= set(vocab.branch_directions())
    assert {op.semantics for op in operations} <= {"label", "assumed"}
    assert "VERSEMENT SUR COMPTE" in vocab.branch_labels("deposit")
    assert all(op.category == "deposit" for op in operations if op.is_cash_deposit)


def test_client_codes_and_origins() -> None:
    assert vocab.client_codes("genre") == (1, 2)
    assert vocab.client_codes("situation_familiale") == (1, 2, 3, 4)
    assert set(vocab.card_origins()) == {"National", "International"}


def test_seed_labels_keep_their_exact_spelling() -> None:
    labels = {op.label for op in vocab.branch_operations()}
    assert "COMMISSION SOUSCRIPTION CARTE BINATNA  PAR DB CPTE" in labels  # double space kept
    assert "PAIEMENT VIGNETTE PAR DÉBIT DE COMPTE" in labels


def test_periods_are_ordered_and_contain_their_days() -> None:
    for period in SPECIAL_PERIODS:
        assert period.start <= period.end
        assert period.start in period
        assert len(period.days()) == (period.end - period.start).days + 1
    assert RAMADAN_2024.end < EID_AL_ADHA_2024.start
    assert PRE_EID_AL_ADHA_WEEK_2024.end < EID_AL_ADHA_2024.start
    assert "not a date" not in RAMADAN_2024


def test_a_period_cannot_end_before_it_starts() -> None:
    with pytest.raises(ValueError, match="ends before"):
        Period("x", "x", date(2024, 1, 2), date(2024, 1, 1))


def test_bank_holidays_include_fixed_holidays_and_eids() -> None:
    assert all(is_bank_holiday(day) for day, _ in NATIONAL_HOLIDAYS)
    assert is_bank_holiday(EID_AL_ADHA_2024.start)
    assert not is_bank_holiday(date(2024, 3, 13))


@pytest.mark.parametrize(
    ("day", "expected"),
    [(1, "01-05"), (5, "01-05"), (6, "06-10"), (25, "21-25"), (26, "26-31"), (31, "26-31")],
)
def test_day_of_month_bins(day: int, expected: str) -> None:
    assert day_of_month_bin(date(2024, 1, day)) == expected
