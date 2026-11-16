import matplotlib
import numpy as np
from verify import T4, baseline_asst, d, p3, p4

matplotlib.use("Agg")
import glob as _g

import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm

[_fm.fontManager.addfont(_p) for _p in _g.glob("fonts/OldStandard-*.otf")]
plt.rcParams.update(
    {
        "font.family": "Old Standard",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 9.5,
    }
)
plt.rcParams.update(
    {
        "mathtext.fontset": "custom",
        "mathtext.rm": "Old Standard",
        "mathtext.it": "Old Standard:italic",
        "mathtext.bf": "Old Standard:bold",
        "mathtext.sf": "Old Standard",
    }
)

C = ["P0", "P1", "P2", "P3", "P4"]
lab = [
    "P0\nno telemetry",
    "P1\n+ median/IQR",
    "P2\n+ critique",
    "P3\n4 models",
    "P4\n4 models\n+ critique",
]
col = ["#6b7280", "#b91c1c", "#374151", "#a16207", "#047857"]
# Fig2 decomposition

fig, ax = plt.subplots(figsize=(3.3, 3.0))
b = np.zeros(5)
for k, (nm, cc) in enumerate(
    zip(
        [r"$0.36\,E_1$ carry-over", r"$0.16\,E_{a,2}$ assistant", r"$0.48\,$same-direction term"],
        ["#d1d5db", "#6b7280", "#c2410c"],
        strict=True,
    )
):
    ax.bar(C, T4[:, k], bottom=b, color=cc, label=nm, width=0.65)
    b += T4[:, k]
for j in range(5):
    ax.text(j, b[j] + 0.0004, f"{b[j]:.4f}", ha="center", fontsize=7)
ax.set_ylabel(r"$E_2$ components")
ax.legend(fontsize=6.6, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1)
ax.set_ylim(0, 0.026)
fig.tight_layout()
fig.savefig("fig2.pdf")
# Fig3 interaction
m = d.mean(0)
fig, ax = plt.subplots(figsize=(3.1, 3.0))
NAVY = "#262626"
RED = "#c2410c"
GREEN = "#15803d"
ax.plot([0, 1], [m[1], m[2]], "-", color=NAVY, lw=1.6, label="1 model (Gemma \u00d78)", zorder=1)
ax.plot([0, 1], [m[3], m[4]], "--", color=NAVY, lw=1.6, label="4 models (\u00d72 each)", zorder=1)
for x, y, c in [(0, m[1], RED), (1, m[2], GREEN), (0, m[3], RED), (1, m[4], GREEN)]:
    ax.scatter([x], [y], s=46, color=c, zorder=3, edgecolor="white", linewidth=0.8)
ax.axhline(m[0], ls=":", color="#9ca3af")
ax.text(-0.27, m[0] + 0.00035, "P0 (no telemetry)", fontsize=6.5, color="#6b7280")
ax.set_xticks([0, 1])
ax.set_xticklabels(["median shown", "+ critique"])
ax.set_xlim(-0.3, 1.3)
ax.set_ylabel(r"Mean $E_2$")
ax.legend(fontsize=7, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))
for x, y, t, dy in [
    (0, m[1], "P1", 0),
    (1, m[2], "P2", -0.0006),
    (0, m[3], "P3", 0),
    (1, m[4], "P4", 0),
]:
    ax.text(
        x - 0.08 if x == 0 else x + 0.07,
        y + dy,
        t,
        fontsize=7,
        va="center",
        ha="right" if x == 0 else "left",
    )
fig.tight_layout()
fig.savefig("fig3.pdf")
# Fig4 capability
mods = ["Gemma", "Qwen", "GLM", "DeepSeek"]
fig, ax = plt.subplots(figsize=(3.3, 3.0))
x = np.arange(4)
ax.bar(
    x - 0.18,
    p3,
    0.36,
    color="#c2410c",
    hatch="///",
    edgecolor="white",
    lw=0,
    label="in P3 (median)",
)
ax.bar(
    x + 0.18,
    p4,
    0.36,
    color="#15803d",
    hatch="///",
    edgecolor="white",
    lw=0,
    label="in P4 (+ critique)",
)
ax.axhline(baseline_asst, ls=":", color="#9ca3af")
ax.text(3.5, 0.033, "Gemma in P0", fontsize=6.5, ha="right", color="#6b7280")
ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels(mods, fontsize=7.5)
ax.set_ylabel("Assistant MSE (R1+R2)")
ax.set_ylim(3e-4, 8e-2)
ax.legend(fontsize=7, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
fig.tight_layout()
fig.savefig("fig4.pdf")
