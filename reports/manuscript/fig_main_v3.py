import matplotlib
import numpy as np
from verify import d

matplotlib.use("Agg")
import glob as _g

import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm
from matplotlib.patches import Patch

[_fm.fontManager.addfont(_p) for _p in _g.glob("fonts/OldStandard-*.otf")]
plt.rcParams.update(
    {
        "font.family": "Old Standard",
        "font.size": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
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


def pct(e):
    return (10 ** np.sqrt(e) - 1) * 100


rng = np.random.RandomState(42)
m = d.mean(0)
lo = []
hi = []
for j in range(5):
    bs = np.sort([d[rng.randint(0, 16, 16), j].mean() for _ in range(2000)])
    lo.append(pct(bs[50]))
    hi.append(pct(bs[1949]))
P = pct(m)
GREY, RED, GREEN = "#9ca3af", "#c2410c", "#15803d"
order = [0, 1, 2, 3, 4]
xs = [0, 1.15, 2.3, 4.0, 5.15]
colors = [GREY, RED, GREEN, RED, GREEN]
hatch = [None, None, None, "///", "///"]
fig, (a, b) = plt.subplots(1, 2, figsize=(7.6, 3.4), gridspec_kw={"width_ratios": [1.3, 1]})
for x, j, c, h in zip(xs, order, colors, hatch, strict=True):
    a.bar(x, P[j], width=0.78, color=c, alpha=0.9, hatch=h, edgecolor="white", lw=0)
    a.errorbar(x, P[j], yerr=[[P[j] - lo[j]], [hi[j] - P[j]]], color="#1f2937", lw=0.8, capsize=2.5)
    a.text(
        x,
        4.5,
        f"{P[j]:.0f}%",
        ha="center",
        fontsize=10,
        fontweight="bold",
        color="white" if c != GREY else "#1f2937",
        zorder=5,
        bbox=None if c == GREY else dict(boxstyle="round,pad=0.18", fc=c, ec="none"),
    )
a.axhline(P[0], ls="--", color="#4b5563", lw=0.9)
a.text(
    5.7,
    P[0] + 1.0,
    "baseline: assistant\nsees nothing",
    fontsize=7.2,
    color="#4b5563",
    ha="right",
    va="bottom",
)
a.set_xticks(xs)
a.set_xticklabels(
    [
        "sees\nnothing",
        "sees crowd\nmedian",
        "median\n+ critique",
        "sees crowd\nmedian",
        "median\n+ critique",
    ],
    fontsize=7.3,
)
for x0, x1, t in [(-0.4, 2.7, "one model (Gemma \u00d78)"), (3.6, 5.6, "four different models")]:
    a.annotate(
        "",
        xy=(x0, -0.30),
        xytext=(x1, -0.30),
        xycoords=("data", "axes fraction"),
        textcoords=("data", "axes fraction"),
        arrowprops=dict(arrowstyle="-", lw=0.8, color="#6b7280"),
    )
    a.text(
        (x0 + x1) / 2,
        -0.37,
        t,
        transform=a.get_xaxis_transform(),
        ha="center",
        fontsize=8,
        color="#374151",
    )
a.set_ylim(0, 60)
a.set_xlim(-0.6, 5.75)
a.set_ylabel("Transformed log RMSE (%)")
a.set_title("(a) Error magnitude", fontsize=9.5, loc="left", fontweight="bold")
# panel b
names = ["crowd median", "median + critique", "4 models, median", "4 models, median\n+ critique"]
idx = [1, 2, 3, 4]
cs = [RED, GREEN, RED, GREEN]
y = np.arange(4)[::-1]
for yi, j in zip(y, idx, strict=True):
    x = d[:, j] - d[:, 0]
    bt = (x < -1e-12).sum()
    ti = (abs(x) <= 1e-12).sum()
    wo = (x > 1e-12).sum()
    b.barh(yi, bt, color="#15803d", height=0.62)
    b.barh(yi, ti, left=bt, color="#e5e7eb", height=0.62)
    b.barh(yi, wo, left=bt + ti, color="#c2410c", height=0.62)
    b.text(
        bt / 2, yi, str(bt), ha="center", va="center", color="white", fontsize=9, fontweight="bold"
    )
    b.text(
        bt + ti + wo / 2,
        yi,
        str(wo),
        ha="center",
        va="center",
        color="white",
        fontsize=9,
        fontweight="bold",
    )
b.set_yticks(y)
b.set_yticklabels(names, fontsize=8)
b.set_xlim(0, 16)
b.set_xticks([0, 8, 16])
b.axvline(8, color="#6b7280", lw=0.7, ls=":")
b.set_xlabel("questions (of 16)", fontsize=8.5)
b.set_title("(b) Each question vs baseline", fontsize=9.5, loc="left", fontweight="bold")
b.legend(
    handles=[
        Patch(color="#15803d", label="better"),
        Patch(color="#e5e7eb", label="tie"),
        Patch(color="#c2410c", label="worse"),
    ],
    fontsize=7.5,
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(0.4, -0.2),
    ncol=3,
)
fig.tight_layout()
fig.subplots_adjust(bottom=0.3)
fig.savefig("fig1.pdf")
print([round(p, 1) for p in P])
