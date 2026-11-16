"""Primary-outcome replication figure from the compact full-precision task data."""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
for font in (HERE / "fonts").glob("OldStandard-*.otf"):
    font_manager.fontManager.addfont(font)
plt.rcParams.update(
    {
        "font.family": "Old Standard",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.edgecolor": "#b9b9b9",
        "xtick.color": "#555555",
        "ytick.color": "#262626",
        "pdf.fonttype": 3,
    }
)
data = json.loads((HERE / "task_metrics.json").read_text())
errors = {c: np.array(v) for c, v in data["E2"].items()}
results = json.loads((HERE / "s2_results.json").read_text())
contrasts = {(r["a"], r["b"]): r for group in results["contrasts"].values() for r in group}
rng = np.random.default_rng(42)
indices = rng.integers(0, 40, size=(10000, 40))
INK, RUST, TEAL = "#303a43", "#b64b32", "#24766a"
fig, (left, right) = plt.subplots(1, 2, figsize=(7.4, 3.4))
fig.subplots_adjust(left=0.105, right=0.985, bottom=0.18, top=0.78, wspace=0.94)

families = [
    ("Gemma", "P0", "P1"),
    ("DeepSeek", "P0_DS", "P1_DS"),
    ("Qwen", "P0_QWEN", "P1_QWEN"),
    ("GLM", "P0_GLM", "P1_GLM"),
    ("Mixed panel", "P0_POLY", "P3"),
]
for y, (_, silent, shown) in enumerate(families):
    left.plot([errors[silent].mean(), errors[shown].mean()], [y, y], color="#d5d9dc", lw=2)
    for arm, offset, color, marker in ((silent, -0.10, INK, "o"), (shown, 0.10, RUST, "s")):
        m = errors[arm].mean()
        lo, hi = np.percentile(errors[arm][indices].mean(axis=1), [2.5, 97.5])
        left.errorbar(
            m,
            y + offset,
            xerr=[[m - lo], [hi - m]],
            fmt=marker,
            color=color,
            ms=4.2,
            capsize=2,
            elinewidth=1.2,
            markeredgewidth=0.8,
        )
left.set_yticks(range(5), [f[0] for f in families])
left.set_xlim(0, 0.041)
left.set_xticks([0, 0.01, 0.02, 0.03, 0.04], ["0", ".01", ".02", ".03", ".04"])
left.set_xlabel("Mean final squared log error", labelpad=8)
left.set_title(
    "A   Consensus increases error", loc="left", fontsize=10.2, fontweight="bold", pad=33
)
left.legend(
    handles=[
        Line2D([], [], marker="o", color=INK, lw=0, label="No telemetry", markersize=4),
        Line2D([], [], marker="s", color=RUST, lw=0, label="Median + IQR", markersize=4),
    ],
    loc="lower left",
    bbox_to_anchor=(-0.025, 1.01),
    ncol=2,
    frameon=False,
    handletextpad=0.35,
    columnspacing=0.7,
    fontsize=7.5,
    borderaxespad=0,
)

rows = [
    ("Gemma critique", "P1", "P2"),
    ("Gemma reflection", "P1", "P1_REFLECT"),
    ("Critique beyond\nreflection", "P1_REFLECT", "P2"),
    ("Mixed-panel critique", "P3", "P4"),
    ("Mixing beyond\nstatic averaging", "VIRTUAL_SOCIAL", "P3"),
]
for y, (_, a, b) in enumerate(rows):
    row = contrasts[a, b]
    m, (lo, hi) = row["mean"], row["ci"]
    right.errorbar(
        m, y, xerr=[[m - lo], [hi - m]], fmt="D", color=TEAL, ms=4.2, capsize=2, elinewidth=1.2
    )
right.axvline(0, color="#777777", lw=0.8, ls=(0, (3, 3)))
right.set_yticks(range(5), [r[0] for r in rows], fontsize=8.1)
right.set_xlim(-0.013, 0.001)
right.set_xticks([-0.012, -0.008, -0.004, 0], ["\u2212.012", "\u2212.008", "\u2212.004", "0"])
right.set_xlabel("Change in squared log error", labelpad=8)
right.set_title(
    "B   Interventions partly help", loc="left", fontsize=10.2, fontweight="bold", pad=33
)
right.text(
    0, 1.035, "← lower error with intervention", transform=right.transAxes, fontsize=8, color=TEAL
)
for ax in (left, right):
    ax.set_ylim(4.55, -0.55)
    ax.tick_params(axis="y", length=0, pad=5)
    ax.tick_params(axis="x", labelsize=8, length=3)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#eeeeee", lw=0.7)
fig.savefig(HERE / "fig5.pdf", bbox_inches="tight", pad_inches=0.08)
