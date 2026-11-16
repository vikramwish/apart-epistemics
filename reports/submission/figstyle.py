"""Publication figure styling for the Apart Research Epistemics submission.

All figures render to vector PDF at true print dimensions (5.5 in full-width or
2.7 in half-width) with consistent serif typography (~9-10 pt), disabled top
and right spines, light dashed gridlines, and a unified Okabe-Ito palette.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# Single colorblind-safe Okabe-Ito palette shared across every paper figure.
OKABE_ITO = {
    "blue": "#0072B2",  # P2 / Counter-Anchored / E_crowd / 0.36 E_1
    "vermilion": "#D55E00",  # P1 / Herding Failure / 0.48 Cross
    "green": "#009E73",  # P3 / Polyphony / D_between
    "purple": "#CC79A7",  # P4 / Combined (Polyphony + Counter-Anchoring)
    "amber": "#E69F00",  # D_within / 0.16 E_{a,2} / Secondary highlight
    "skyblue": "#56B4E9",  # Assistant series / accent
    "slate": "#5A6577",  # P0 Baseline (Isolated Homogeneous)
    "ink": "#1A202C",  # Primary text / zero line / direct labels
    "muted": "#4A5568",  # Subdued notes / reference lines
    "grid": "#CBD5E1",  # Subtle gridlines
}

CONDITION_COLORS = {
    "P0": OKABE_ITO["slate"],
    "P1": OKABE_ITO["vermilion"],
    "P2": OKABE_ITO["blue"],
    "P3": OKABE_ITO["green"],
    "P4": OKABE_ITO["purple"],
}

MODEL_COLORS = {
    "gemma-4-26b-a4b-it": OKABE_ITO["vermilion"],
    "qwen3.8-27b": OKABE_ITO["amber"],
    "glm-5.2": OKABE_ITO["green"],
    "deepseek-v4-flash": OKABE_ITO["blue"],
}


def apply_paper_style() -> None:
    """Configure global Matplotlib rcParams for 5.5 in vector PDF output."""
    mpl.use("pdf")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "STIX Two Text", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 7.5,
            "axes.edgecolor": OKABE_ITO["ink"],
            "axes.linewidth": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": OKABE_ITO["grid"],
            "grid.linestyle": "--",
            "grid.linewidth": 0.45,
            "grid.alpha": 0.6,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.color": OKABE_ITO["ink"],
            "ytick.color": OKABE_ITO["ink"],
            "text.color": OKABE_ITO["ink"],
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def panel_tag(ax: plt.Axes, label: str, x: float = -0.14, y: float = 1.06) -> None:
    """Place a uniform bold subpanel identifier ((a), (b)) outside the plot box."""
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=9.5,
        fontweight="bold",
        color=OKABE_ITO["ink"],
        va="bottom",
        ha="left",
    )
