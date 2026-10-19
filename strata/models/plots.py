"""Static images of the models, drawn with matplotlib for the model cards (M11).

Only aggregates are drawn (the reliability bins of a backtest: mean predicted and observed silence
per bin of the score), never a client. The image of a zone stays in that zone's model folder;
`strata promote` copies the synthetic one next to the public card.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt

RELIABILITY_IMAGE = "reliability.png"
INK = "#1f2933"
ACCENT = "#2f6f9f"
MUTED = "#9aa5b1"
SECOND = "#c26a2a"
CURVES = {
    "lightgbm_platt": ("served score (trees, Platt)", ACCENT),
    "glm": ("logistic GLM", SECOND),
}


def reliability_png(reliability: dict[str, list[dict[str, Any]]], path: Path, title: str) -> Path:
    """A reliability diagram of the served score and the GLM, with the diagonal of perfect
    calibration; each point is one bin, its size the number of clients in it."""
    figure, axis = plt.subplots(figsize=(5.2, 4.4), dpi=120)
    axis.plot([0, 1], [0, 1], linestyle="--", color=MUTED, linewidth=1, label="perfect calibration")
    for name, (label, colour) in CURVES.items():
        bins = reliability.get(name) or []
        if not bins:
            continue
        sizes = [max(12.0, 4.0 * float(b["n"]) ** 0.5) for b in bins]
        xs = [float(b["predicted"]) for b in bins]
        ys = [float(b["observed"]) for b in bins]
        axis.plot(xs, ys, color=colour, linewidth=1.5, label=label)
        axis.scatter(xs, ys, s=sizes, color=colour, alpha=0.8, edgecolors="none")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xlabel("predicted probability of silence", color=INK)
    axis.set_ylabel("observed share silent", color=INK)
    axis.set_title(title, color=INK, fontsize=10, loc="left")
    axis.grid(color="#e4e7eb", linewidth=0.6)
    axis.legend(frameon=False, fontsize=8, loc="upper left")
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, format="png", metadata={"Software": None})
    plt.close(figure)
    return path
