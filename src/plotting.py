"""Shared matplotlib styling and figure export.

All figures are written to ``figures/`` at publication resolution so they can be
dropped straight into the report rather than re-screenshotted from notebooks.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

from config import FIGURES

# A single categorical palette used across every figure, so the same strategy
# keeps the same colour from one plot to the next.
PALETTE = {
    "random": "#9aa0a6",
    "naive 1.00x": "#6b7280",
    "pace heuristic": "#8172b3",
    "best fixed": "#55a868",
    "MILP schedule": "#4c72b0",
    "Double Q-learning": "#dd8452",
    "DP optimal": "#2a4d7a",
}

SEQUENTIAL = "RdYlGn_r"
UNVISITED = "#e9e9e9"  # states a learner never reached, masked out of policy maps


def use_style() -> None:
    """Apply the project's plotting defaults."""
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "figure.constrained_layout.use": True,
    })


def save_fig(name: str, fig=None):
    """Write the current (or given) figure to ``figures/<name>.png``."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig = fig or plt.gcf()
    path = FIGURES / f"{name}.png"
    fig.savefig(path)
    return path


def colour(strategy: str) -> str:
    """Palette lookup tolerant of the ``best fixed 1.15x`` style suffixes."""
    if strategy in PALETTE:
        return PALETTE[strategy]
    for key, value in PALETTE.items():
        if strategy.startswith(key):
            return value
    return "#9aa0a6"
