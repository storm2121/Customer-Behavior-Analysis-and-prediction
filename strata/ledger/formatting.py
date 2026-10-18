"""How ledger values are written in prose. One formatter per unit, used by documents and reports."""

from __future__ import annotations

import math

from strata.ledger.model import Unit


def _share(value: float) -> str:
    percent = 100 * value
    return f"{percent:.0f}%" if abs(percent) >= 10 else f"{percent:.1f}%"


def _number(value: float) -> str:
    magnitude = abs(value)
    if magnitude >= 1000:
        return f"{value:,.0f}"
    if magnitude >= 10:
        return f"{value:.1f}"
    if magnitude >= 1:
        return f"{value:.2f}"
    return f"{value:.2g}"


def _amount(value: float) -> str:
    text = f"{value:,.0f}" if abs(value) >= 10 else f"{value:,.2f}"
    return f"{text} MAD"


TIMES = "\N{MULTIPLICATION SIGN}"

_FORMATTERS = {
    Unit.SHARE: _share,
    Unit.DIFFERENCE: lambda value: f"{100 * value:+.1f} percentage points",
    Unit.COUNT: lambda value: f"{value:,.0f}",
    Unit.AMOUNT: _amount,
    Unit.RATIO: lambda value: f"{value:.2f}{TIMES}",
    Unit.PVALUE: lambda value: "p < 0.001" if value < 0.001 else f"p = {value:.3f}",
    Unit.DAYS: lambda value: f"{value:,.0f} days",
    Unit.NUMBER: _number,
}


def format_value(value: float, unit: Unit) -> str:
    """Render ``value`` in ``unit`` (a share of 0.362 becomes ``36%``)."""
    if not math.isfinite(value):
        return "undefined"
    return _FORMATTERS[unit](value)


def format_interval(interval: tuple[float, float], unit: Unit) -> str:
    """``95% CI 31% to 40%``; differences keep their sign."""
    low, high = interval
    if unit is Unit.DIFFERENCE:
        return f"95% CI {100 * low:+.1f} to {100 * high:+.1f} points"
    return f"95% CI {format_value(low, unit)} to {format_value(high, unit)}"
