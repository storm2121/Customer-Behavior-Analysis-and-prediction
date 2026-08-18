"""Feature engineering tests.

These run without the bank datasets, which is the whole point of keeping
features.py free of file access and globals.
"""

import pandas as pd
import pytest

from customer_analytics import features


def test_days_since_last_transaction_counts_whole_days():
    frame = pd.DataFrame({"date_transaction": ["2024-01-01", "2024-01-11"]})
    days = features.days_since_last_transaction(frame, today=pd.Timestamp("2024-01-21"))
    assert list(days) == [20, 10]


def test_days_since_last_transaction_fills_unparseable_dates_with_the_mean():
    frame = pd.DataFrame({"date_transaction": ["2024-01-01", "2024-01-11", "not a date"]})
    days = features.days_since_last_transaction(frame, today=pd.Timestamp("2024-01-21"))
    assert days.iloc[2] == pytest.approx(15.0)


def test_avg_transaction_amount_divides_by_tenure():
    frame = pd.DataFrame({"montant_local": [1000.0, 500.0], "anciennete_client": [10.0, 5.0]})
    assert list(features.avg_transaction_amount(frame)) == [100.0, 100.0]


def test_zero_tenure_does_not_poison_the_column_mean():
    """Dividing by zero tenure yields inf; it must not leak into the mean."""
    frame = pd.DataFrame({"montant_local": [1000.0, 500.0, 300.0], "anciennete_client": [10.0, 5.0, 0.0]})
    result = features.avg_transaction_amount(frame)
    assert result.iloc[2] == pytest.approx(100.0)
    assert result.notna().all()


@pytest.mark.parametrize(
    "age,expected",
    [(25, "18-30"), (30, "18-30"), (31, "31-45"), (50, "46-60"), (70, "61-75"), (90, "76+")],
)
def test_age_group_boundaries(age, expected):
    frame = pd.DataFrame({"age": [age]})
    assert features.age_group(frame).iloc[0] == expected


def test_age_below_the_first_bin_is_unclassified():
    """Bins start at 18, so a younger client is NaN rather than silently 18-30."""
    frame = pd.DataFrame({"age": [15]})
    assert pd.isna(features.age_group(frame).iloc[0])


def test_engineer_attaches_every_derived_column():
    frame = pd.DataFrame(
        {
            "date_transaction": ["2024-01-01"],
            "montant_local": [1000.0],
            "anciennete_client": [10.0],
            "age": [35],
        }
    )
    result = features.engineer(frame)
    for column in ("days_since_last_transaction", "avg_transaction_amount", "age_group"):
        assert column in result.columns


def test_engineer_does_not_mutate_its_input():
    frame = pd.DataFrame(
        {"date_transaction": ["2024-01-01"], "montant_local": [1.0], "anciennete_client": [1.0], "age": [30]}
    )
    features.engineer(frame)
    assert "age_group" not in frame.columns
