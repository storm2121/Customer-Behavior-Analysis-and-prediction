"""The Simpson guard on frames whose answer is known by construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from strata.ledger.robustness import RobustnessSpec, check

SPEC = RobustnessSpec(
    group_col="group",
    treatment="a",
    reference="b",
    metric="outcome",
    strata=("stratum",),
    metric_label="outcome rate",
)


def frame(cells: dict[tuple[str, str], tuple[int, int]]) -> pd.DataFrame:
    """Rows from ``{(group, stratum): (n, successes)}``."""
    rows = []
    for (group, stratum), (n, successes) in cells.items():
        rows += [{"group": group, "stratum": stratum, "outcome": 1.0}] * successes
        rows += [{"group": group, "stratum": stratum, "outcome": 0.0}] * (n - successes)
    return pd.DataFrame(rows)


def simpson() -> pd.DataFrame:
    """Group a looks worse overall because it sits mostly in the high-risk stratum, yet within
    each stratum it does better: the textbook reversal."""
    return frame(
        {
            ("a", "high"): (200, 120),  # 60%
            ("b", "high"): (40, 28),  # 70%
            ("a", "low"): (40, 4),  # 10%
            ("b", "low"): (200, 40),  # 20%
        }
    )


def test_a_textbook_reversal_is_reported_as_reverses() -> None:
    result = check(simpson(), SPEC, n_resamples=300, seed=1)
    assert result.overall_effect > 0
    assert result.verdict == "reverses"
    stratifier = result.stratifiers[0]
    assert stratifier.reverses
    assert stratifier.adjusted_effect is not None
    assert stratifier.adjusted_effect < 0
    assert result.sign_flips == 2
    assert result.shrinkage is not None
    assert result.shrinkage > 1
    assert "The sign reverses." in result.mechanism
    assert "outcome rate" in result.mechanism


def test_a_constant_effect_holds() -> None:
    data = frame(
        {
            ("a", "high"): (100, 60),
            ("b", "high"): (100, 40),
            ("a", "low"): (100, 30),
            ("b", "low"): (100, 10),
        }
    )
    result = check(data, SPEC, n_resamples=200, seed=2)
    assert result.verdict == "holds"
    assert result.sign_flips == 0
    assert result.shrinkage == pytest.approx(0.0, abs=1e-9)
    low, high = result.overall_ci
    assert low < result.overall_effect < high


def test_an_effect_that_shrinks_inside_strata_weakens() -> None:
    data = frame(
        {
            ("a", "high"): (150, 90),  # 60%
            ("b", "high"): (50, 29),  # 58%: almost no effect here
            ("a", "low"): (50, 15),  # 30%
            ("b", "low"): (150, 15),  # 10%
        }
    )
    result = check(data, SPEC, n_resamples=200, seed=3)
    assert result.verdict == "weakens"
    assert result.sign_flips == 0
    assert result.shrinkage is not None
    assert result.shrinkage >= 0.5


def test_strata_with_too_few_clients_are_listed_and_excluded() -> None:
    data = pd.concat(
        [simpson(), frame({("a", "rare"): (30, 15), ("b", "rare"): (3, 1)})], ignore_index=True
    )
    result = check(data, SPEC, n_resamples=200, seed=4)
    rare = next(s for s in result.stratifiers[0].strata if s.stratum == "rare")
    assert not rare.sufficient
    assert rare.ci is None
    assert "rare" in result.mechanism
    assert result.verdict == "reverses"


def test_no_sufficient_stratum_is_insufficient() -> None:
    data = frame({("a", "x"): (5, 2), ("b", "y"): (5, 1)})
    result = check(data, SPEC, n_resamples=50, seed=5)
    assert result.verdict == "insufficient"
    assert "No stratum" in result.mechanism


def test_a_zero_crude_difference_is_insufficient() -> None:
    data = frame({("a", "x"): (50, 10), ("b", "x"): (50, 10)})
    assert check(data, SPEC, n_resamples=50, seed=6).verdict == "insufficient"


def test_both_groups_are_required() -> None:
    with pytest.raises(ValueError, match="both"):
        check(frame({("a", "x"): (20, 5)}), SPEC, n_resamples=20)


def test_the_group_column_is_never_a_stratifier() -> None:
    spec = RobustnessSpec("group", "a", "b", "outcome", strata=("group", "stratum"))
    result = check(simpson(), spec, n_resamples=50, seed=7)
    assert [s.stratifier for s in result.stratifiers] == ["stratum"]


def test_labels_fall_back_to_raw_values() -> None:
    spec = RobustnessSpec("group", "branch_only", "digital_only", "silent")
    assert spec.label("metric") == "silent"
    assert spec.label("treatment") == "branch-only"
    assert spec.label("reference") == "digital-only"


def test_non_share_metrics_are_described_without_points() -> None:
    spec = RobustnessSpec("group", "a", "b", "outcome", strata=("stratum",), metric_is_share=False)
    result = check(simpson(), spec, n_resamples=50, seed=8)
    assert "points" not in result.mechanism


@settings(max_examples=15, deadline=None)
@given(
    base=st.floats(min_value=0.05, max_value=0.35),
    lift=st.floats(min_value=0.1, max_value=0.35),
    sizes=st.lists(st.integers(min_value=60, max_value=150), min_size=4, max_size=4),
)
def test_a_uniform_lift_never_reverses(base: float, lift: float, sizes: list[int]) -> None:
    """If the treatment adds the same lift in every stratum, stratification cannot reverse it."""
    cells = {}
    for (group, stratum), n in zip(
        [("a", "x"), ("b", "x"), ("a", "y"), ("b", "y")], sizes, strict=True
    ):
        rate = base + (lift if group == "a" else 0.0) + (0.2 if stratum == "y" else 0.0)
        cells[(group, stratum)] = (n, round(rate * n))
    result = check(frame(cells), SPEC, n_resamples=50, seed=9)
    assert result.verdict != "reverses"
    assert result.stratifiers[0].adjusted_effect is not None
    assert result.stratifiers[0].adjusted_effect > 0
    assert np.isfinite(result.overall_effect)
