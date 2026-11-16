import matplotlib

matplotlib.use("Agg")
import glob as _g

import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

[_fm.fontManager.addfont(_p) for _p in _g.glob("fonts/OldStandard-*.otf")]
plt.rcParams.update({"font.family": "Old Standard", "font.size": 8.5})
plt.rcParams.update(
    {
        "mathtext.fontset": "custom",
        "mathtext.rm": "Old Standard",
        "mathtext.it": "Old Standard:italic",
        "mathtext.bf": "Old Standard:bold",
        "mathtext.sf": "Old Standard",
    }
)
fig, ax = plt.subplots(figsize=(7.0, 2.5))
ax.set_xlim(0, 100)
ax.set_ylim(-3, 37)
ax.axis("off")


def box(x, y, w, h, txt, fc, ec="#262626", fs=7.4, bold=False):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=0.9
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        txt,
        ha="center",
        va="center",
        fontsize=fs,
        fontweight="bold" if bold else None,
    )


def arr(x1, y1, x2, y2, c="#262626", ls="-", rad=0):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=9,
            color=c,
            lw=1,
            ls=ls,
            connectionstyle=f"arc3,rad={rad}",
        )
    )


box(0.5, 9, 17, 14, "8 agents with\nbiased priors\n(shared \u00d71.5\nmisconception)", "#f4f4f4")
box(26, 9, 24, 14, "LLM assistant\n(one per agent)\nsees question +\nagent's estimate", "#e8e8e8")
box(59.5, 9, 18, 14, "Agent update\n$x\\leftarrow0.6x+0.4a$", "#f4f4f4")
box(83.5, 9, 16, 14, "Crowd mean\n$\\bar x$ vs truth\n→ error $E$", "#f4f4f4")
arr(18.3, 16, 25.2, 16)
arr(50.8, 16, 58.7, 16)
arr(78.3, 16, 82.7, 16)
box(22.5, 27.5, 31, 5.5, "+ peer median & IQR  (P1\u2013P4)", "#fee2e2", ec="#b91c1c", fs=7.6)
box(22.5, -1.5, 31, 5.5, "+ two-sided critique  (P2, P4)", "#dcfce7", ec="#047857", fs=7.6)
arr(38, 27.3, 38, 23.6, "#b91c1c")
arr(38, 4.2, 38, 8.4, "#047857")
ax.add_patch(
    FancyArrowPatch(
        (68.5, 23.8),
        (54.3, 30.2),
        arrowstyle="-|>",
        mutation_scale=9,
        color="#b91c1c",
        lw=0.9,
        ls="--",
        connectionstyle="arc3,rad=0.25",
    )
)
ax.text(
    79,
    31.5,
    "round 2 shows the median\nof the updated estimates",
    fontsize=6.8,
    color="#b91c1c",
    ha="center",
)
ax.text(
    91.5,
    2.5,
    "2 rounds · Study 1: 16 tasks · Study 2: 40 tasks \u00d7 3",
    fontsize=6.6,
    ha="center",
    color="#555555",
)
fig.savefig("fig0.pdf", bbox_inches="tight")
