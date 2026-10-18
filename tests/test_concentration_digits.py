"""RFM scores, per-client digit matrices and the M4 concentration and Benford claims."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from strata.analysis import benford
from strata.analysis.concentration import quintile_scores, rfm_scores

if TYPE_CHECKING:
    from strata.ledger.model import Ledger


def test_quintile_scores_run_from_one_to_five() -> None:
    assert quintile_scores(np.arange(10)).tolist() == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    reverse = quintile_scores(np.arange(10), higher_is_better=False)
    assert reverse.tolist() == [5, 5, 4, 4, 3, 3, 2, 2, 1, 1]
    assert quintile_scores([]).size == 0


def test_tied_values_share_a_score() -> None:
    scores = quintile_scores([1, 1, 1, 1, 5])
    assert len(set(scores[:4].tolist())) == 1
    assert scores[4] > scores[0]


@settings(max_examples=50, deadline=None)
@given(st.lists(st.integers(0, 20), min_size=1, max_size=60))
def test_quintile_scores_are_bounded_and_monotone(values: list[int]) -> None:
    scores = quintile_scores(values)
    assert scores.min() >= 1
    assert scores.max() <= 5
    order = np.argsort(values, kind="stable")
    assert np.all(np.diff(scores[order]) >= 0)


def test_recent_frequent_valuable_clients_score_high() -> None:
    scores = rfm_scores([1, 100], [5, 1], [10.0, 1.0])
    assert scores.shape == (2, 3)
    assert np.all(scores[0] > scores[1])


def test_digit_matrices_count_digits_per_client() -> None:
    matrix = benford.digit_matrix([0, 0, 1], [1, 9, 1], 2, benford.FIRST_DIGITS)
    assert matrix.shape == (2, 9)
    assert matrix[0, 0] == 1
    assert matrix[0, 8] == 1
    assert matrix[1, 0] == 1
    assert matrix.sum() == 3


def test_the_concentration_and_digit_claims_on_synthetic_data(synthetic_ledger: Ledger) -> None:
    rfm = synthetic_ledger.get("rfm_by_channel")
    for channel in ("card", "online", "branch"):
        assert 0.0 <= rfm.values[f"{channel}_best_value_share"] <= 1.0
        assert 0.0 <= rfm.values[f"{channel}_frequency_gini"] <= 1.0
    categories = synthetic_ledger.get("benford_by_category")
    assert "card_purchase_mad" in categories.values
    assert "card_purchase_two_digit_mad" in categories.values
    assert categories.meta["card_purchase_two_digit_conformity"]
    two = synthetic_ledger.get("benford_two_digits")
    pairs = [int(two.meta[f"pair_rank_{rank}"]) for rank in (1, 2, 3)]
    assert all(10 <= pair <= 99 for pair in pairs)
    assert two.values["excess_rank_1"] >= two.values["excess_rank_2"] >= two.values["excess_rank_3"]
