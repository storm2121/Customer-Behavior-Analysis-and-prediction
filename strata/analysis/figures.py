"""Figures for the analysis modules, written as HTML pages into the zone's figures folder.

Figures of a private run are drawn from real data, so they live under ``STRATA_PRIVATE_DIR``;
public runs write them under ``.strata/figures`` (gitignored). The pages of one folder share a
single ``plotly.min.js`` next to them, so they open offline without each embedding the library.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import STL

from strata.analysis import benford, markov
from strata.analysis.concentration import lorenz
from strata.ledger.model import ClaimRecord

INK = "#1f2933"
ACCENT = "#2f6f9f"
MUTED = "#9aa5b1"
WARM = "#b5651d"
BAND_LEVELS = (5, 50, 95)


def _style(fig: go.Figure, title: str, height: int = 420) -> go.Figure:
    fig.update_layout(
        title={"text": title, "x": 0},
        height=height,
        margin={"l": 60, "r": 20, "t": 60, "b": 40},
        font={"family": "system-ui, sans-serif", "size": 12, "color": INK},
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


def write(fig: go.Figure, folder: Path, name: str) -> Path:
    """Write ``fig`` as ``<folder>/<name>.html`` (plotly.js shared from the same folder)."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.html"
    fig.write_html(path, include_plotlyjs="directory", full_html=True)
    return path


def calendar_forest(record: ClaimRecord) -> go.Figure:
    """Rate ratios of every series with their bootstrap intervals, on a log axis."""
    keys = [key for key in record.values if key in record.ci]
    estimates = [record.values[key] for key in keys]
    fig = go.Figure(
        go.Scatter(
            x=estimates,
            y=[key.replace("_", " ") for key in keys],
            mode="markers",
            marker={"color": ACCENT, "size": 8},
            error_x={
                "type": "data",
                "symmetric": False,
                "array": [
                    record.ci[key][1] - value for key, value in zip(keys, estimates, strict=True)
                ],
                "arrayminus": [
                    value - record.ci[key][0] for key, value in zip(keys, estimates, strict=True)
                ],
                "color": MUTED,
            },
        )
    )
    fig.add_vline(x=1, line_dash="dot", line_color=INK)
    fig.update_xaxes(type="log", title_text="rate ratio (log scale)")
    fig.update_yaxes(autorange="reversed")
    return _style(fig, "Calendar effects (negative-binomial regression)", 160 + 18 * len(keys))


def stl_components(frame: pd.DataFrame, label: str) -> go.Figure:
    """A daily series with its STL trend and weekly component (period seven, robust fit)."""
    values = frame["n_events"].to_numpy(dtype=float)
    fit = STL(values, period=7, robust=True).fit()
    days = pd.to_datetime(frame["day"])
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, subplot_titles=("observed", "trend", "weekly")
    )
    fig.add_trace(go.Scatter(x=days, y=values, line={"color": MUTED, "width": 1}), row=1, col=1)
    fig.add_trace(go.Scatter(x=days, y=fit.trend, line={"color": ACCENT}), row=2, col=1)
    fig.add_trace(
        go.Scatter(x=days, y=fit.seasonal, line={"color": WARM, "width": 1}), row=3, col=1
    )
    fig.update_layout(showlegend=False)
    return _style(fig, f"STL decomposition of daily {label}", 600)


def lorenz_curves(clients: pd.DataFrame) -> go.Figure:
    """Lorenz curves of card spend, online amounts and branch credits across clients."""
    fig = go.Figure(
        go.Scatter(x=[0, 1], y=[0, 1], line={"color": MUTED, "dash": "dot"}, name="equality")
    )
    for column, label, colour in (
        ("card_spend", "card spend", ACCENT),
        ("online_amount", "online amounts", WARM),
        ("branch_credit_amount", "branch credits", INK),
    ):
        values = clients.loc[clients[column] > 0, column].to_numpy(dtype=float)
        if values.size < 2:
            continue
        population, share = lorenz(values)
        fig.add_trace(go.Scatter(x=population, y=share, name=label, line={"color": colour}))
    fig.update_xaxes(title_text="share of clients (smallest first)")
    fig.update_yaxes(title_text="share of value")
    return _style(fig, "Concentration of value by channel")


def benford_bars(amounts: dict[str, benford.FloatArray]) -> go.Figure:
    """Observed first-digit shares of each category against Benford's law."""
    digits = [int(digit) for digit in benford.FIRST_DIGITS]
    fig = go.Figure(
        go.Scatter(
            x=digits,
            y=[float(share) for share in benford.EXPECTED_FIRST],
            mode="lines+markers",
            name="Benford's law",
            line={"color": INK, "dash": "dot"},
        )
    )
    for name, values in amounts.items():
        test = benford.first_digit_test(values)
        observed = [float(share) for share in test.observed]
        fig.add_trace(go.Bar(x=digits, y=observed, name=name.replace("_", " "), opacity=0.7))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title_text="first significant digit", dtick=1)
    fig.update_yaxes(title_text="share of amounts")
    return _style(fig, "Leading digits against Benford's law")


def simulation_fan(policy: markov.Paths, baseline: markov.Paths, description: str) -> go.Figure:
    """Digital share and branch load under a policy and the baseline, with 90% bands."""
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("digital share", "branch operations per month")
    )
    months = list(range(policy["digital_share"].shape[1]))
    for column, quantity in ((1, "digital_share"), (2, "branch_load")):
        for paths, label, colour in ((baseline, "baseline", MUTED), (policy, "policy", ACCENT)):
            low, middle, high = np.percentile(paths[quantity], BAND_LEVELS, axis=0)
            band = go.Scatter(
                x=months + months[::-1],
                y=[*high, *low[::-1]],
                fill="toself",
                line={"width": 0},
                fillcolor=colour,
                opacity=0.2,
                hoverinfo="skip",
                showlegend=False,
            )
            line = go.Scatter(
                x=months, y=middle, line={"color": colour}, name=label, showlegend=column == 1
            )
            fig.add_trace(band, row=1, col=column)
            fig.add_trace(line, row=1, col=column)
    fig.update_xaxes(title_text="months from now")
    return _style(fig, f"What if {description}?", 420)
