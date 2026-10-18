"""Benford's law: first-digit and first-two-digit tests with Nigrini's conformity bands.

Amounts that arise from many multiplicative processes follow Benford's law; prices are set rather
than drawn, so deviations in purchases usually reflect pricing habits (amounts just below a round
number) rather than anything suspicious. With thousands of amounts a chi-square test rejects for
tiny deviations, which is why the mean absolute deviation (MAD) and its conformity bands are the
recommended reading (Nigrini, *Benford's Law*, 2012).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import stats

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

FIRST_DIGITS = np.arange(1, 10)
FIRST_TWO_DIGITS = np.arange(10, 100)
EXPECTED_FIRST: FloatArray = np.log10(1 + 1 / FIRST_DIGITS)
EXPECTED_FIRST_TWO: FloatArray = np.log10(1 + 1 / FIRST_TWO_DIGITS)

# Nigrini's MAD thresholds: (upper bound, label), checked in order.
FIRST_DIGIT_BANDS = (
    (0.006, "close conformity"),
    (0.012, "acceptable conformity"),
    (0.015, "marginally acceptable conformity"),
)
FIRST_TWO_DIGIT_BANDS = (
    (0.0012, "close conformity"),
    (0.0018, "acceptable conformity"),
    (0.0022, "marginally acceptable conformity"),
)


def _significand(amounts: npt.ArrayLike) -> list[str]:
    values = np.asarray(amounts, dtype=np.float64)
    positive = values[np.isfinite(values) & (values > 0)]
    # Scientific notation puts the first significant digit first and is immune to the floating
    # point drift of log10-based formulas (0.3 is not 2.999...).
    return [f"{value:.12e}" for value in positive]


def first_digits(amounts: npt.ArrayLike) -> IntArray:
    """First significant digit (1 to 9) of every positive amount."""
    return np.array([int(text[0]) for text in _significand(amounts)], dtype=np.int64)


def first_two_digits(amounts: npt.ArrayLike) -> IntArray:
    """First two significant digits (10 to 99) of every positive amount."""
    return np.array([int(text[0] + text[2]) for text in _significand(amounts)], dtype=np.int64)


def digit_counts(digits: npt.ArrayLike, support: npt.ArrayLike) -> FloatArray:
    labels = np.asarray(support, dtype=np.int64)
    values = np.asarray(digits, dtype=np.int64)
    return np.array([(values == label).sum() for label in labels], dtype=np.float64)


def mad(observed: npt.ArrayLike, expected: npt.ArrayLike) -> float:
    """Mean absolute deviation between observed and expected proportions."""
    return float(np.mean(np.abs(np.asarray(observed) - np.asarray(expected))))


def conformity(value: float, bands: tuple[tuple[float, str], ...] = FIRST_DIGIT_BANDS) -> str:
    for bound, label in bands:
        if value <= bound:
            return label
    return "nonconformity"


@dataclass(frozen=True, slots=True)
class BenfordTest:
    """One test of observed digit proportions against Benford's law."""

    observed: FloatArray
    expected: FloatArray
    n: int
    mad: float
    chi2: float
    p_value: float
    conformity: str


def test_digits(
    counts: npt.ArrayLike, expected: FloatArray, bands: tuple[tuple[float, str], ...]
) -> BenfordTest:
    """Chi-square and MAD for digit ``counts`` against ``expected`` proportions."""
    observed_counts = np.asarray(counts, dtype=np.float64)
    n = int(observed_counts.sum())
    if n == 0:
        msg = "no positive amounts to test"
        raise ValueError(msg)
    observed = observed_counts / n
    chi2, p_value = stats.chisquare(observed_counts, expected * n)
    deviation = mad(observed, expected)
    return BenfordTest(
        observed=observed,
        expected=expected,
        n=n,
        mad=deviation,
        chi2=float(chi2),
        p_value=float(p_value),
        conformity=conformity(deviation, bands),
    )


def first_digit_test(amounts: npt.ArrayLike) -> BenfordTest:
    digits = first_digits(amounts)
    return test_digits(digit_counts(digits, FIRST_DIGITS), EXPECTED_FIRST, FIRST_DIGIT_BANDS)


def first_two_digit_test(amounts: npt.ArrayLike) -> BenfordTest:
    digits = first_two_digits(amounts)
    return test_digits(
        digit_counts(digits, FIRST_TWO_DIGITS), EXPECTED_FIRST_TWO, FIRST_TWO_DIGIT_BANDS
    )


def digit_matrix(
    codes: npt.ArrayLike, digits: npt.ArrayLike, n_clients: int, support: npt.ArrayLike
) -> FloatArray:
    """``(clients, digits)`` counts of leading digits, for client-bootstrap intervals."""
    labels = np.asarray(support, dtype=np.int64)
    column = {int(label): i for i, label in enumerate(labels)}
    matrix = np.zeros((n_clients, len(labels)))
    columns = np.array([column[int(d)] for d in np.asarray(digits)], dtype=np.int64)
    np.add.at(matrix, (np.asarray(codes, dtype=np.int64), columns), 1.0)
    return matrix
