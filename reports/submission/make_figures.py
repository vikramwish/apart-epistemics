"""Generate all four vector-PDF empirical figures for the Apart submission.

Usage:
    /tmp/plot_venv/bin/python reports/submission/make_figures.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from figstyle import (
    CONDITION_COLORS,
    MODEL_COLORS,
    OKABE_ITO,
    apply_paper_style,
    panel_tag,
)

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "results" / "polyphony-p0-p4"
TASKS_PATH = ROOT / "configs" / "polyphony_benchmark_tasks.json"
OUT_DIR = Path(__file__).resolve().parent / "figures"

CONDS = ["P0", "P1", "P2", "P3", "P4"]
COND_SHORT = ["P0\nIso", "P1\nTel", "P2\nCA", "P3\nPoly", "P4\nPoly+CA"]


def load_data() -> tuple[dict, dict, dict[str, dict[str, np.ndarray]]]:
    """Load frozen analysis JSON, audit JSON, and per-task R2 error arrays."""
    analysis = json.loads((DATA_DIR / "numeric_analysis.json").read_text())
    audit = json.loads((DATA_DIR / "audit_reproduction_report.json").read_text())
    tasks = json.loads(TASKS_PATH.read_text())
    truth_map = {t["id"]: np.log10(float(t["numeric_truth"])) for t in tasks}

    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    with (DATA_DIR / "numeric_runs.jsonl").open() as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["round_idx"] == 2:
                cells[(row["condition_id"], row["question_id"])].append(row)

    per_task: dict[str, dict[str, np.ndarray]] = {"crowd": {}, "asst": {}}
    for cond in CONDS:
        c_errs = []
        a_errs = []
        for qid, z_star in truth_map.items():
            rs = cells[(cond, qid)]
            c_errs.append((np.mean([r["agent_log10_after"] for r in rs]) - z_star) ** 2)
            a_errs.append((np.mean([r["assistant_log10"] for r in rs]) - z_star) ** 2)
        per_task["crowd"][cond] = np.array(c_errs, dtype=float)
        per_task["asst"][cond] = np.array(a_errs, dtype=float)

    return analysis, audit, per_task


def make_fig1_headline(audit: dict, per_task: dict[str, dict[str, np.ndarray]]) -> None:
    """Figure 1: (a) R2 crowd & assistant MSE with 95% bootstrap CIs; (b) paired contrasts."""
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(5.5, 2.15), gridspec_kw={"width_ratios": [1.18, 1.0]}
    )
    rng = np.random.default_rng(42)

    x = np.arange(len(CONDS))
    w = 0.34
    agent_means, agent_lo, agent_hi = [], [], []
    asst_means, asst_lo, asst_hi = [], [], []
    for cond in CONDS:
        ce = per_task["crowd"][cond]
        ae = per_task["asst"][cond]
        bc = np.array([ce[rng.integers(0, len(ce), len(ce))].mean() for _ in range(2000)])
        ba = np.array([ae[rng.integers(0, len(ae), len(ae))].mean() for _ in range(2000)])
        m_c, lo_c, hi_c = ce.mean(), np.percentile(bc, 2.5), np.percentile(bc, 97.5)
        m_a, lo_a, hi_a = ae.mean(), np.percentile(ba, 2.5), np.percentile(ba, 97.5)
        agent_means.append(m_c)
        agent_lo.append(m_c - lo_c)
        agent_hi.append(hi_c - m_c)
        asst_means.append(m_a)
        asst_lo.append(m_a - lo_a)
        asst_hi.append(hi_a - m_a)

    bar_colors = [CONDITION_COLORS[c] for c in CONDS]
    ax1.bar(
        x - w / 2,
        agent_means,
        width=w,
        color=bar_colors,
        edgecolor=OKABE_ITO["ink"],
        linewidth=0.5,
        zorder=3,
    )
    ax1.errorbar(
        x - w / 2,
        agent_means,
        yerr=[agent_lo, agent_hi],
        fmt="none",
        ecolor=OKABE_ITO["ink"],
        elinewidth=0.75,
        capsize=2.0,
        zorder=4,
    )
    ax1.bar(
        x + w / 2,
        asst_means,
        width=w,
        color="white",
        edgecolor=bar_colors,
        linewidth=1.0,
        hatch="///",
        zorder=3,
    )
    ax1.errorbar(
        x + w / 2,
        asst_means,
        yerr=[asst_lo, asst_hi],
        fmt="none",
        ecolor=OKABE_ITO["muted"],
        elinewidth=0.65,
        capsize=1.8,
        zorder=4,
    )

    # Compact direct labels positioned clear of error bar caps
    for i, (m_c, hi_c) in enumerate(zip(agent_means, agent_hi, strict=True)):
        ax1.text(
            x[i] - w / 2,
            m_c + hi_c + 0.0011,
            f"{m_c:.4f}",
            ha="center",
            va="bottom",
            fontsize=6.5,
            fontweight="bold",
            color=bar_colors[i],
        )

    # Tiny inline bar-type indicator at top-right of panel (a) (solid vs hatched)
    ax1.bar(
        [-1],
        [0],
        color=OKABE_ITO["slate"],
        edgecolor=OKABE_ITO["ink"],
        linewidth=0.5,
        label="Blended crowd $E_2$",
    )
    ax1.bar(
        [-1],
        [0],
        color="white",
        edgecolor=OKABE_ITO["slate"],
        linewidth=0.9,
        hatch="///",
        label="Direct assistant $E_{a,2}$",
    )
    ax1.legend(
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.9,
        fontsize=6.8,
        handlelength=1.2,
        borderpad=0.25,
    )

    ax1.set_xticks(x)
    ax1.set_xticklabels(COND_SHORT, fontsize=7.5)
    ax1.set_xlim(-0.55, 4.55)
    ax1.set_ylim(0, 0.0495)
    ax1.set_ylabel("Round-2 MSE ($\\log_{10}^2$)")
    panel_tag(ax1, "(a)", x=-0.16, y=1.03)

    # Panel (b): Pre-specified contrasts with exact bootstrap 95% CIs & Holm p_H9,ex
    crows = {r["contrast_id"]: r for r in audit["exact_vs_pipeline_wilcoxon_matrix"]["crowd_error"]}
    specs = [
        ("P4 vs P3", "p4_vs_p3_crowd_error", OKABE_ITO["muted"], False),
        ("P4 vs P0", "p4_vs_p0_crowd_error", OKABE_ITO["muted"], False),
        ("P4 vs P1", "p4_vs_p1_crowd_error", CONDITION_COLORS["P4"], True),
        ("P3 vs P1", "p3_vs_p1_crowd_error", CONDITION_COLORS["P3"], True),
        ("P2 vs P1", "p2_vs_p1_crowd_error", CONDITION_COLORS["P2"], True),
        ("P1 vs P0", "p1_vs_p0_crowd_error", CONDITION_COLORS["P1"], True),
    ]
    ax2.yaxis.grid(False)
    ax2.xaxis.grid(True, color=OKABE_ITO["grid"], linestyle="--", linewidth=0.45, alpha=0.6)
    ax2.axvline(0.0, color=OKABE_ITO["ink"], linestyle="--", linewidth=0.75, zorder=2)

    ys = np.arange(len(specs))
    for yi, (_lbl, cid, col, sig) in enumerate(specs):
        row = crows[cid]
        md, lo, hi = row["mean_diff"], row["ci_low"], row["ci_high"]
        hp = row["holm9_exact_nz"]
        ax2.errorbar(
            md,
            yi,
            xerr=[[md - lo], [hi - md]],
            fmt="o" if sig else "s",
            color=col,
            ecolor=col,
            elinewidth=1.15,
            markersize=4.2,
            capsize=2.3,
            zorder=4,
        )
        p_str = f"p={hp:.3f}"
        if md >= 0 or cid == "p4_vs_p0_crowd_error" or hi <= -0.003:
            txt_x = hi + 0.0012
            ha = "left"
        else:
            txt_x = lo - 0.0012
            ha = "right"
        ax2.text(
            txt_x,
            yi,
            p_str,
            va="center",
            ha=ha,
            fontsize=6.5,
            fontweight="bold" if sig else "normal",
            color=col if sig else OKABE_ITO["muted"],
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92, "pad": 0.6},
            zorder=5,
        )

    ax2.set_yticks(ys)
    ax2.set_yticklabels([s[0] for s in specs], fontsize=7.5)
    ax2.set_xlim(-0.032, 0.0315)
    ax2.set_xlabel("Paired $\\Delta E_2$ ($\\log_{10}^2$, 95% boot CI)")
    panel_tag(ax2, "(b)", x=-0.16, y=1.03)

    fig.tight_layout(w_pad=1.6)
    fig.savefig(OUT_DIR / "fig1_headline_herding.pdf")
    plt.close(fig)


def make_fig3_hierarchical(analysis: dict, audit: dict) -> None:
    """Figure 3: Two-Level Hierarchical Page ANOVA decomposition (Agents vs Assistants)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 2.25), sharey=True)
    by_final = analysis["by_condition_final"]
    perm = audit["exact_random_label_permutation_audit"]
    x = np.arange(len(CONDS))
    w = 0.56

    panels = [
        (
            ax1,
            "(a)",
            "crowd_error",
            "within_model_diversity",
            "between_model_diversity",
            "avg_individual_error",
            "Blended agents (R2)",
        ),
        (
            ax2,
            "(b)",
            "assistant_crowd_error",
            "assistant_within_model_diversity",
            "assistant_between_model_diversity",
            "assistant_avg_individual_error",
            "Direct assistants (R2)",
        ),
    ]
    for ax, tag, ke, kwin, kbet, kind, plabel in panels:
        e_vals = np.array([by_final[c][ke] for c in CONDS])
        w_vals = np.array([by_final[c][kwin] for c in CONDS])
        b_vals = np.array([by_final[c][kbet] for c in CONDS])
        totals = np.array([by_final[c][kind] for c in CONDS])

        ax.bar(
            x,
            e_vals,
            width=w,
            color=OKABE_ITO["blue"],
            edgecolor="white",
            linewidth=0.5,
            label="Crowd error $E$",
            zorder=3,
        )
        ax.bar(
            x,
            w_vals,
            width=w,
            bottom=e_vals,
            color=OKABE_ITO["amber"],
            edgecolor="white",
            linewidth=0.5,
            label="$D_{\\mathrm{within}}$",
            zorder=3,
        )
        ax.bar(
            x,
            b_vals,
            width=w,
            bottom=e_vals + w_vals,
            color=OKABE_ITO["green"],
            edgecolor="white",
            linewidth=0.5,
            label="$D_{\\mathrm{between}}$",
            zorder=3,
        )

        for i in range(len(CONDS)):
            ax.text(
                x[i],
                totals[i] + 0.0008,
                f"{totals[i]:.4f}",
                ha="center",
                va="bottom",
                fontsize=6.8,
                color=OKABE_ITO["ink"],
            )
            if CONDS[i] in ("P3", "P4"):
                # Exact null baseline E + D_within + 0.75 D_within
                null_top = e_vals[i] + w_vals[i] + 0.75 * w_vals[i]
                ax.plot(
                    [x[i] - w / 2 - 0.05, x[i] + w / 2 + 0.05],
                    [null_top, null_top],
                    color=OKABE_ITO["ink"],
                    linestyle="--",
                    linewidth=1.0,
                    zorder=5,
                )

        # Direct callout on P3/P4 permutation survival
        if ke == "crowd_error":
            p3_k = perm["P3_agent_R2"]["n_tasks_perm_p_le_0_05"]
            p4_k = perm["P4_agent_R2"]["n_tasks_perm_p_le_0_05"]
        else:
            p3_k = perm["P3_assistant_R2"]["n_tasks_perm_p_le_0_05"]
            p4_k = perm["P4_assistant_R2"]["n_tasks_perm_p_le_0_05"]

        ax.annotate(
            f"Perm $p\\leq .05$:\nP3 {p3_k}/16, P4 {p4_k}/16",
            xy=(3.5, totals[4] + 0.003),
            xytext=(3.3, 0.0305),
            ha="center",
            fontsize=6.5,
            color=OKABE_ITO["muted"],
            arrowprops={"arrowstyle": "-", "color": OKABE_ITO["grid"], "lw": 0.6},
        )
        ax.set_xticks(x)
        ax.set_xticklabels(CONDS, fontsize=7.8)
        ax.set_ylim(0, 0.0395)
        panel_tag(ax, f"{tag} {plabel}", x=-0.04 if ax is ax1 else 0.0, y=1.03)

    ax1.set_ylabel("Individual MSE $\\bar{e}$ ($\\log_{10}^2$)")
    handles, labels = ax1.get_legend_handles_labels()
    null_line = plt.Line2D([0], [0], color=OKABE_ITO["ink"], linestyle="--", linewidth=1.0)
    handles.append(null_line)
    labels.append("Null $0.75\\,D_{\\mathrm{within}}$")
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=7.3,
        bbox_to_anchor=(0.52, -0.02),
    )
    fig.tight_layout(rect=[0, 0.07, 1, 1], w_pad=0.9)
    fig.savefig(OUT_DIR / "fig3_hierarchical_page.pdf")
    plt.close(fig)


def make_fig4_factorial(analysis: dict, audit: dict) -> None:
    """Figure 4: (a) 2x2 factorial interaction profile; (b) per-model capability confound."""
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(5.5, 2.20), gridspec_kw={"width_ratios": [1.0, 1.25]}
    )
    by_final = analysis["by_condition_final"]
    did = audit["exact_vs_pipeline_2x2_did"]["crowd_error"]

    # Panel (a): 2x2 interaction lines
    p0_e = by_final["P0"]["crowd_error"]
    ax1.axhline(p0_e, color=OKABE_ITO["slate"], linestyle="--", linewidth=0.9, zorder=2)
    ax1.text(
        1.05,
        p0_e + 0.00045,
        f"P0: {p0_e:.4f}",
        fontsize=6.8,
        color=OKABE_ITO["slate"],
        ha="left",
        va="bottom",
    )

    tel = [by_final["P1"]["crowd_error"], by_final["P3"]["crowd_error"]]
    ca = [by_final["P2"]["crowd_error"], by_final["P4"]["crowd_error"]]
    kx = [0, 1]
    ax1.plot(kx, tel, "-o", color=OKABE_ITO["vermilion"], lw=1.6, ms=4.8, zorder=4)
    ax1.plot(kx, ca, "-s", color=OKABE_ITO["blue"], lw=1.6, ms=4.5, zorder=4)

    # Direct endpoint & line labels placed clear of crossing slopes
    ax1.text(
        -0.06,
        tel[0] + 0.0010,
        f"P1: {tel[0]:.4f}",
        color=OKABE_ITO["vermilion"],
        fontsize=7.0,
        fontweight="bold",
        ha="left",
    )
    ax1.text(
        1.05,
        tel[1] + 0.0002,
        f"P3: {tel[1]:.4f}",
        color=OKABE_ITO["vermilion"],
        fontsize=7.0,
        fontweight="bold",
        ha="left",
        va="center",
    )
    ax1.text(
        0.0,
        ca[0] - 0.0015,
        f"P2: {ca[0]:.4f}",
        color=OKABE_ITO["blue"],
        fontsize=7.0,
        fontweight="bold",
        ha="center",
        va="top",
    )
    ax1.text(
        1.05,
        ca[1],
        f"P4: {ca[1]:.4f}",
        color=OKABE_ITO["blue"],
        fontsize=7.0,
        fontweight="bold",
        ha="left",
        va="center",
    )
    ax1.text(
        0.48,
        0.0178,
        "Telemetry",
        color=OKABE_ITO["vermilion"],
        fontsize=7.0,
        rotation=-27,
        ha="center",
    )
    ax1.text(
        0.58,
        0.0072,
        "Counter-Anchor",
        color=OKABE_ITO["blue"],
        fontsize=7.0,
        rotation=-10,
        ha="center",
    )
    ax1.text(
        0.98,
        0.95,
        f"DiD = {did['mean_diff']:+.4f}\n$p_{{H5}}={did['holm5_exact_nz']:.4f}$",
        transform=ax1.transAxes,
        ha="right",
        va="top",
        fontsize=6.8,
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": "#F8FAFC",
            "edgecolor": OKABE_ITO["grid"],
            "lw": 0.6,
        },
    )

    ax1.set_xticks(kx)
    ax1.set_xticklabels(["Homog. ($K{=}1$)", "Polyphonic ($K{=}4$)"], fontsize=7.5)
    ax1.set_xlim(-0.24, 1.42)
    ax1.set_ylim(0.0032, 0.0272)
    ax1.set_ylabel("Round-2 Crowd MSE ($\\log_{10}^2$)")
    panel_tag(ax1, "(a)", x=-0.18, y=1.03)

    # Panel (b): Per-model in-panel R1+R2 assistant MSE
    pmb = audit["per_model_breakdown"]
    groups = ["P0", "P1", "P2", "P3", "P4"]
    models = [
        ("gemma-4-26b-a4b-it", "Gemma-4"),
        ("qwen3.8-27b", "Qwen3.8"),
        ("glm-5.2", "GLM-5.2"),
        ("deepseek-v4-flash", "DeepSeek-v4"),
    ]
    gx = np.arange(len(groups))
    bw = 0.18
    for mi, (mslug, mshort) in enumerate(models):
        vals = [pmb[g][mslug]["mse_combined"] if mslug in pmb[g] else np.nan for g in groups]
        # Place Gemma alone cleanly at center for P0-P2, 4 bars for P3-P4
        if mslug == "gemma-4-26b-a4b-it":
            pos = [0.0, 1.0, 2.0, 3.0 - 1.5 * bw, 4.0 - 1.5 * bw]
            v = [vals[0], vals[1], vals[2], vals[3], vals[4]]
        else:
            pos = [3.0 + (mi - 1.5) * bw, 4.0 + (mi - 1.5) * bw]
            v = [vals[3], vals[4]]
        ax2.bar(
            pos,
            v,
            width=bw * 1.45 if mslug == "gemma-4-26b-a4b-it" else bw,
            color=MODEL_COLORS[mslug],
            edgecolor="white",
            linewidth=0.4,
            label=mshort,
            zorder=3,
        )
        for px, pv in zip(pos, v, strict=True):
            if px >= 2.5:
                rot = 90 if pv < 0.03 else 0
                ax2.text(
                    px + (0.01 if rot == 90 else 0.0),
                    pv + 0.0012,
                    f"{pv:.4f}" if pv < 0.01 else f"{pv:.3f}",
                    ha="center",
                    va="bottom",
                    rotation=rot,
                    fontsize=5.9,
                    color=MODEL_COLORS[mslug],
                )
            else:
                ax2.text(
                    px,
                    pv + 0.0012,
                    f"{pv:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=6.4,
                    color=MODEL_COLORS[mslug],
                )

    ax2.set_xticks(gx)
    ax2.set_xticklabels(groups, fontsize=7.6)
    ax2.set_xlim(-0.5, 4.55)
    ax2.set_ylim(0, 0.0535)
    ax2.set_ylabel("Pooled Asst. MSE ($\\log_{10}^2$)")
    ax2.legend(
        loc="upper left",
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.9,
        fontsize=6.5,
        columnspacing=0.8,
        handlelength=1.0,
    )
    panel_tag(ax2, "(b)", x=-0.15, y=1.03)

    fig.tight_layout(w_pad=1.5)
    fig.savefig(OUT_DIR / "fig4_factorial_capability.pdf")
    plt.close(fig)


def make_fig5_loto_crossterm(audit: dict) -> None:
    """Figure 5: (a) Leave-One-Task-Out (n=15) bounds; (b) R2 cross-term breakdown."""
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(5.5, 2.25), gridspec_kw={"width_ratios": [1.18, 1.0]}
    )
    loto = audit["loto_summary"]
    crows = {r["contrast_id"]: r for r in audit["exact_vs_pipeline_wilcoxon_matrix"]["crowd_error"]}
    cterm = audit["r2_signed_error_cross_term_decomposition"]

    specs = [
        ("P4 vs P1", "P4_vs_P1", "p4_vs_p1_crowd_error", CONDITION_COLORS["P4"]),
        ("P3 vs P1", "P3_vs_P1", "p3_vs_p1_crowd_error", CONDITION_COLORS["P3"]),
        ("P2 vs P1", "P2_vs_P1", "p2_vs_p1_crowd_error", CONDITION_COLORS["P2"]),
        ("P4 vs P0", "P4_vs_P0", "p4_vs_p0_crowd_error", OKABE_ITO["muted"]),
        ("P4 vs P3", "P4_vs_P3", "p4_vs_p3_crowd_error", OKABE_ITO["muted"]),
        ("P1 vs P0", "P1_vs_P0", "p1_vs_p0_crowd_error", CONDITION_COLORS["P1"]),
    ]
    ax1.yaxis.grid(False)
    ax1.xaxis.grid(True, color=OKABE_ITO["grid"], linestyle="--", linewidth=0.45, alpha=0.6)
    ax1.axvline(0.0, color=OKABE_ITO["ink"], linestyle="--", linewidth=0.75, zorder=2)

    ys = np.arange(len(specs))
    for yi, (_lbl, lk, ck, col) in enumerate(specs):
        lo, hi = loto[lk]["diff_min_max"]
        full_m = crows[ck]["mean_diff"]
        drop_ls = loto[lk]["drop_lake_superior"]["mean_diff"]
        n_surv = loto[lk]["n_subsets_holm9_exact_lt_005"]
        ax1.hlines(yi, lo, hi, color=col, linewidth=2.2, alpha=0.55, zorder=3)
        ax1.scatter(
            [full_m],
            [yi],
            color=col,
            s=22,
            zorder=4,
            label="Full ($n{=}16$)" if yi == 0 else None,
        )
        ax1.scatter(
            [drop_ls],
            [yi],
            facecolors="white",
            edgecolors=OKABE_ITO["ink"],
            marker="D",
            s=22,
            linewidths=0.9,
            zorder=5,
            label="Drop Lake Sup." if yi == 0 else None,
        )
        ax1.text(
            0.0143,
            yi,
            f"{n_surv}/16",
            va="center",
            ha="left",
            fontsize=6.8,
            fontweight="bold" if n_surv == 16 else "normal",
            color=col if n_surv >= 7 else OKABE_ITO["muted"],
        )

    ax1.text(
        0.0143,
        5.55,
        "Holm\n$p<.05$",
        fontsize=6.2,
        color=OKABE_ITO["muted"],
        va="bottom",
        ha="left",
    )
    ax1.set_yticks(ys)
    ax1.set_yticklabels([s[0] for s in specs], fontsize=7.4)
    ax1.set_xlim(-0.0185, 0.0215)
    ax1.set_ylim(-0.65, 6.35)
    ax1.set_xlabel("LOTO $\\Delta E_2$ Range ($\\log_{10}^2$)")
    ax1.legend(
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.95,
        fontsize=6.4,
        handletextpad=0.3,
        borderpad=0.25,
    )
    panel_tag(ax1, "(a)", x=-0.18, y=1.03)

    # Panel (b): Cross-term stacked bar breakdown
    x = np.arange(len(CONDS))
    w = 0.56
    c_r1 = np.array([cterm[c]["0.36_E_crowd_R1"] for c in CONDS])
    c_a2 = np.array([cterm[c]["0.16_E_asst_crowd_R2"] for c in CONDS])
    c_xx = np.array([cterm[c]["0.48_signed_cross_term"] for c in CONDS])
    tot = np.array([cterm[c]["E_crowd_R2_sum"] for c in CONDS])

    ax2.bar(
        x,
        c_r1,
        width=w,
        color=OKABE_ITO["blue"],
        edgecolor="white",
        lw=0.45,
        label="$0.36\\,E_1$",
        zorder=3,
    )
    ax2.bar(
        x,
        c_a2,
        width=w,
        bottom=c_r1,
        color=OKABE_ITO["amber"],
        edgecolor="white",
        lw=0.45,
        label="$0.16\\,E_{a,2}$",
        zorder=3,
    )
    ax2.bar(
        x,
        c_xx,
        width=w,
        bottom=c_r1 + c_a2,
        color=OKABE_ITO["vermilion"],
        edgecolor="white",
        lw=0.45,
        label="$0.48\\,\\mathrm{Cross}$",
        zorder=3,
    )

    for i in range(len(CONDS)):
        ax2.text(
            x[i],
            tot[i] + 0.0006,
            f"{tot[i]:.4f}",
            ha="center",
            va="bottom",
            fontsize=6.7,
            fontweight="bold",
        )
    ax2.annotate(
        "40.3% Cross\n(+.00923)",
        xy=(1.28, c_r1[1] + c_a2[1] + c_xx[1] * 0.55),
        xytext=(2.18, 0.0168),
        ha="center",
        fontsize=6.3,
        color=OKABE_ITO["vermilion"],
        arrowprops={"arrowstyle": "->", "color": OKABE_ITO["vermilion"], "lw": 0.7},
    )
    ax2.annotate(
        "Cross drops\n$-$68% vs P3",
        xy=(4, tot[4] + 0.0020),
        xytext=(3.75, 0.0155),
        ha="center",
        fontsize=6.4,
        color=OKABE_ITO["ink"],
        arrowprops={"arrowstyle": "->", "color": OKABE_ITO["ink"], "lw": 0.7},
    )

    ax2.set_xticks(x)
    ax2.set_xticklabels(CONDS, fontsize=7.5)
    ax2.set_ylim(0, 0.0310)
    ax2.set_ylabel("Round-2 Crowd MSE $E_2$")
    ax2.legend(
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.92,
        fontsize=6.5,
        handlelength=1.0,
        borderpad=0.2,
    )
    panel_tag(ax2, "(b)", x=-0.16, y=1.03)

    fig.tight_layout(w_pad=1.5)
    fig.savefig(OUT_DIR / "fig5_loto_crossterm.pdf")
    plt.close(fig)


def main() -> None:
    apply_paper_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    analysis, audit, per_task = load_data()
    make_fig1_headline(audit, per_task)
    make_fig3_hierarchical(analysis, audit)
    make_fig4_factorial(analysis, audit)
    make_fig5_loto_crossterm(audit)
    print("Generated vector PDFs:", sorted(p.name for p in OUT_DIR.glob("*.pdf")))


if __name__ == "__main__":
    main()
