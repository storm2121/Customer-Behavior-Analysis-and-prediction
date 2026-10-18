"""Plotly figures for the report, built only from ledger records (never from rows)."""

from __future__ import annotations

import plotly.graph_objects as go

from strata.ledger.model import ClaimRecord, Unit

INK = "#1f2933"
ACCENT = "#2f6f9f"
MUTED = "#9aa5b1"


def _layout(fig: go.Figure, title: str, height: int = 360) -> go.Figure:
    fig.update_layout(
        title={"text": title, "x": 0, "font": {"size": 14}},
        height=height,
        margin={"l": 40, "r": 20, "t": 50, "b": 40},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": INK, "family": "system-ui, sans-serif", "size": 12},
        showlegend=False,
    )
    return fig


def share_figure(record: ClaimRecord) -> go.Figure | None:
    """Bars for every share that has an interval, with the interval as error bars."""
    keys = [key for key, unit in record.units.items() if unit is Unit.SHARE and key in record.ci]
    if not keys:
        return None
    values = [100 * record.values[key] for key in keys]
    lows = [100 * (record.values[key] - record.ci[key][0]) for key in keys]
    highs = [100 * (record.ci[key][1] - record.values[key]) for key in keys]
    fig = go.Figure(
        go.Bar(
            x=[key.replace("_", " ") for key in keys],
            y=values,
            marker_color=ACCENT,
            error_y={"type": "data", "array": highs, "arrayminus": lows, "color": INK},
            hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
        )
    )
    fig.update_yaxes(title_text="percent", gridcolor="#e4e7eb")
    return _layout(fig, f"{record.title}: shares with 95% intervals")


def robustness_figure(record: ClaimRecord) -> go.Figure | None:
    """A forest plot of the crude effect, every stratum effect and every adjusted effect."""
    robustness = record.robustness
    if robustness is None:
        return None
    labels, effects, lows, highs, colours = [], [], [], [], []

    def add(label: str, effect: float | None, ci: tuple[float, float] | None, colour: str) -> None:
        if effect is None:
            return
        labels.append(label)
        effects.append(100 * effect)
        low, high = ci if ci is not None else (effect, effect)
        lows.append(100 * (effect - low))
        highs.append(100 * (high - effect))
        colours.append(colour)

    add("crude", robustness.overall_effect, robustness.overall_ci, INK)
    for stratifier in robustness.stratifiers:
        for stratum in stratifier.strata:
            if stratum.sufficient:
                add(
                    f"{stratifier.stratifier}: {stratum.stratum}", stratum.effect, stratum.ci, MUTED
                )
        add(
            f"{stratifier.stratifier}: adjusted",
            stratifier.adjusted_effect,
            stratifier.adjusted_ci,
            ACCENT,
        )
    fig = go.Figure(
        go.Scatter(
            x=effects,
            y=labels,
            mode="markers",
            marker={"color": colours, "size": 9},
            error_x={"type": "data", "array": highs, "arrayminus": lows, "color": MUTED},
            hovertemplate="%{y}: %{x:+.1f} points<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_dash="dot", line_color=INK)
    fig.update_xaxes(title_text="difference, percentage points", gridcolor="#e4e7eb")
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, f"Robustness: {robustness.verdict}", height=120 + 26 * len(labels))


def figures_for(record: ClaimRecord) -> list[str]:
    """The record's figures as HTML fragments (plotly.js is included once by the page)."""
    built = [
        figure for figure in (share_figure(record), robustness_figure(record)) if figure is not None
    ]
    return [figure.to_html(full_html=False, include_plotlyjs=False) for figure in built]
