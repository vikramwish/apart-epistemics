"""Publication-grade SVG chart generators for Epistemic Prism analytics."""

from __future__ import annotations

import html
import math
from pathlib import Path
from typing import Any

from apart_epistemics.models import ConditionSummary

_PALETTE: dict[str, str] = {
    "paper": "#FAF8F4",
    "surface": "#FFFFFF",
    "ink": "#171614",
    "muted": "#5E5A52",
    "grid": "#E5E0D5",
    "vermilion": "#C84B31",
    "cerulean": "#245A7A",
    "moss": "#2F6F4E",
    "ochre": "#A96A0B",
    "plum": "#6B4C7A",
}

_GROUP_COLORS: dict[str, str] = {
    "baseline": _PALETTE["moss"],
    "decoding_ablation": "#8C2D19",
    "k_sweep": _PALETTE["ochre"],
    "nationality": _PALETTE["cerulean"],
    "personalization": _PALETTE["plum"],
    "main": _PALETTE["vermilion"],
}

_ALIASES: dict[str, str] = {
    "no_ai": "No AI",
    "us_k1_greedy": "US K1 (T=0)",
    "us_k1": "US K1 (T=0.8)",
    "us_k2": "US K2",
    "us_k4": "US K4",
    "cn_k1": "CN K1",
    "cn_k4": "CN K4",
    "divergent_sft_pair": "Divergent SFT",
    "cross_national_k4": "Cross-Nat K4",
    "personalized_advisor": "Advisor (Calib)",
    "personalized_peer": "Peer (Mirror)",
}


def _color_for_group(group: str, condition_id: str) -> str:
    if condition_id.startswith(("cross_national", "divergent")):
        return _PALETTE["cerulean"]
    if condition_id.startswith("us_k1"):
        return _PALETTE["vermilion"]
    return _GROUP_COLORS.get(group, _PALETTE["ink"])


def _short_label(cond_id: str) -> str:
    return _ALIASES.get(cond_id, cond_id)


def _svg_wrap(width: int, height: int, title: str, body: str) -> str:
    esc_title = html.escape(title)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" aria-label="{esc_title}">\n'
        f"  <title>{esc_title}</title>\n"
        f'  <rect width="{width}" height="{height}" fill="{_PALETTE["surface"]}" rx="6"/>\n'
        f"{body}\n"
        "</svg>\n"
    )


def _svg_header_lines(title: str, subtitle: str) -> list[str]:
    return [
        f'  <text x="24" y="34" font-family="Georgia,serif" font-size="18" font-weight="700" '
        f'fill="{_PALETTE["ink"]}">{title}</text>',
        f'  <text x="24" y="54" font-family="sans-serif" font-size="12" '
        f'fill="{_PALETTE["muted"]}">{subtitle}</text>',
    ]


def _append_horizontal_legend(
    lines: list[str], lx: int, ly: int, entries: list[tuple[str, str]], step_x: int
) -> None:
    for label, col in entries:
        lines.append(f'  <rect x="{lx}" y="{ly - 9}" width="11" height="11" rx="2" fill="{col}"/>')
        lines.append(
            f'  <text x="{lx + 16}" y="{ly}" font-family="sans-serif" font-size="11" '
            f'fill="{_PALETTE["ink"]}">{html.escape(label)}</text>'
        )
        lx += step_x


def _render_dissociation_svg(summaries: list[ConditionSummary]) -> str:
    """Compare greedy vs stochastic decoding across lexical vs epistemic dimensions."""
    by_id = {s.condition_id: s for s in summaries}
    greedy = by_id.get("us_k1_greedy")
    stoch = by_id.get("us_k1") or by_id.get("cn_k1")
    diverse = by_id.get("cross_national_k4") or by_id.get("us_k4")

    bars = (
        ("Lexical Jaccard Dist", lambda s: s.pairwise_jaccard_distance),
        ("Distinct-2 Bigrams", lambda s: s.distinct_2),
        ("Topic Entropy (Norm)", lambda s: s.topic_shannon_entropy),
        ("Effective Topics / 6", lambda s: min(1.0, s.effective_topic_count / 6.0)),
        ("Goldman N_eff / 6", lambda s: min(1.0, s.goldman_effective_sources / 6.0)),
    )

    w, h = 760, 400
    left, right, top, row_h = 178, 56, 78, 54
    chart_w = w - left - right
    lines = _svg_header_lines(
        "Decoding Dissociation: Lexical Veil vs Monoculture",
        "Recorded lexical and topic metrics; effective-source values are assumption-based proxies",
    )

    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = left + tick * chart_w
        lines.append(
            f'  <line x1="{x:.1f}" y1="{top - 6}" x2="{x:.1f}" y2="{top + len(bars) * row_h - 6}" '
            f'stroke="{_PALETTE["grid"]}" stroke-width="1"/>'
        )
        lines.append(
            f'  <text x="{x:.1f}" y="{top + len(bars) * row_h + 14}" text-anchor="middle" '
            f'font-family="monospace" font-size="11" fill="{_PALETTE["muted"]}">{tick:.2f}</text>'
        )

    series = [
        ("US K1 T=0", greedy, "#8C2D19"),
        ("US K1 T=0.8", stoch, _PALETTE["vermilion"]),
        ("Cross-Nat K4", diverse, _PALETTE["cerulean"]),
    ]

    for r_idx, (metric_name, getter) in enumerate(bars):
        cy = top + r_idx * row_h
        lines.append(
            f'  <text x="{left - 12}" y="{cy + 22}" text-anchor="end" font-family="sans-serif" '
            f'font-size="12" fill="{_PALETTE["ink"]}">{html.escape(metric_name)}</text>'
        )
        for s_idx, (_lbl, cond, col) in enumerate(series):
            val = max(0.0, min(1.0, getter(cond))) if cond else 0.0
            bw = max(2.0, val * chart_w)
            by = cy + s_idx * 13
            lines.append(
                f'  <rect x="{left}" y="{by}" width="{bw:.1f}" height="10" rx="2" fill="{col}"/>'
            )
            lines.append(
                f'  <text x="{left + bw + 5:.1f}" y="{by + 8.5}" font-family="monospace" '
                f'font-size="10" fill="{_PALETTE["ink"]}">{val:.2f}</text>'
            )

    _append_horizontal_legend(
        lines, 24, h - 22, [(lbl, col) for lbl, cond, col in series if cond is not None], 135
    )
    return _svg_wrap(w, h, "Decoding Double Dissociation", "\n".join(lines))


def _render_epistemic_quadrant_svg(summaries: list[ConditionSummary]) -> str:
    """Scatter plot: Effective Topic Count (X) vs Ground-Truth Calibration 1-Brier (Y)."""
    w, h = 760, 430
    left, right, top, bottom = 74, 58, 72, 68
    pw, ph = w - left - right, h - top - bottom

    xs = [s.effective_topic_count for s in summaries] or [1.0]
    ys = [max(0.0, 1.0 - s.brier_score) for s in summaries] or [0.5]
    x_min = max(0.5, min(xs) - 0.5)
    x_max = max(x_min + 1.0, max(xs) + 0.85)
    y_min = max(0.35, min(ys) - 0.06)
    y_max = min(1.02, max(y_min + 0.1, max(ys) + 0.05))

    def px(val: float) -> float:
        return left + ((val - x_min) / (x_max - x_min)) * pw

    def py(val: float) -> float:
        return top + ph - ((val - y_min) / (y_max - y_min)) * ph

    lines = _svg_header_lines(
        "Epistemic Quadrant: Topic Breadth vs Calibration Quality",
        "Bubble radius shows assumption-based source proxy (N_eff), not measured independence",
    )

    for i in range(5):
        tx = x_min + i * (x_max - x_min) / 4.0
        gx = px(tx)
        lines.append(
            f'  <line x1="{gx:.1f}" y1="{top}" x2="{gx:.1f}" y2="{top + ph}" '
            f'stroke="{_PALETTE["grid"]}" stroke-width="1"/>'
        )
        lines.append(
            f'  <text x="{gx:.1f}" y="{top + ph + 18}" text-anchor="middle" '
            f'font-family="monospace" font-size="11" fill="{_PALETTE["muted"]}">{tx:.1f}</text>'
        )
        ty = y_min + i * (y_max - y_min) / 4.0
        gy = py(ty)
        lines.append(
            f'  <line x1="{left}" y1="{gy:.1f}" x2="{left + pw}" y2="{gy:.1f}" '
            f'stroke="{_PALETTE["grid"]}" stroke-width="1"/>'
        )
        lines.append(
            f'  <text x="{left - 10}" y="{gy + 4:.1f}" text-anchor="end" '
            f'font-family="monospace" font-size="11" fill="{_PALETTE["muted"]}">{ty:.2f}</text>'
        )

    lines.append(
        f'  <rect x="{left}" y="{top}" width="{pw}" height="{ph}" fill="none" '
        f'stroke="{_PALETTE["ink"]}" stroke-width="1"/>'
    )
    lines.append(
        f'  <text x="{left + pw / 2:.1f}" y="{h - 18}" text-anchor="middle" '
        f'font-family="sans-serif" font-size="12" fill="{_PALETTE["ink"]}">'
        f"Effective Topic Count (Hill N1)</text>"
    )
    lines.append(
        f'  <text x="20" y="{top + ph / 2:.1f}" text-anchor="middle" '
        f'transform="rotate(-90 20 {top + ph / 2:.1f})" font-family="sans-serif" '
        f'font-size="12" fill="{_PALETTE["ink"]}">Calibration Score (1 - Brier)</text>'
    )

    label_offsets = {
        "us_k1_greedy": (0.0, -1.0, "start"),
        "us_k1": (12.0, 3.0, "start"),
        "cn_k1": (0.0, 14.0, "start"),
        "cn_k4": (0.0, -1.0, "middle"),
        "divergent_sft_pair": (-4.0, 16.0, "end"),
        "cross_national_k4": (0.0, -12.0, "middle"),
        "personalized_advisor": (-10.0, 15.0, "end"),
        "personalized_peer": (-12.0, 3.0, "end"),
        "no_ai": (0.0, 16.0, "middle"),
    }
    for s in summaries:
        cx = px(s.effective_topic_count)
        cy = py(max(0.0, 1.0 - s.brier_score))
        radius = max(6.0, min(18.0, 4.0 + s.goldman_effective_sources * 1.4))
        col = _color_for_group(s.group, s.condition_id)
        lines.append(
            f'  <circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius:.1f}" fill="{col}" '
            f'fill-opacity="0.78" stroke="{_PALETTE["ink"]}" stroke-width="1"/>'
        )
        dx, dy_raw, anchor = label_offsets.get(
            s.condition_id,
            (0.0, -radius - 5 if cy > top + 28 else radius + 12, "middle"),
        )
        if s.condition_id in ("us_k1_greedy", "cn_k4", "cross_national_k4"):
            dy_raw = -radius - 5
        elif s.condition_id in ("cn_k1", "personalized_advisor", "divergent_sft_pair"):
            dy_raw = radius + 12
        elif s.condition_id == "no_ai":
            dy_raw = radius + 14
        lines.append(
            f'  <text x="{cx + dx:.1f}" y="{cy + dy_raw:.1f}" text-anchor="{anchor}" '
            f'font-family="sans-serif" font-size="10.5" fill="{_PALETTE["ink"]}">'
            f"{html.escape(_short_label(s.condition_id))}</text>"
        )

    return _svg_wrap(w, h, "Epistemic Quadrant", "\n".join(lines))


def _render_round_trajectories_svg(summaries: list[ConditionSummary]) -> str:
    """Plot round-by-round Effective Topic Count trajectories for focal regimes."""
    focal_ids = {
        "no_ai",
        "us_k1",
        "us_k4",
        "divergent_sft_pair",
        "cross_national_k4",
        "personalized_peer",
    }
    selected = [s for s in summaries if s.condition_id in focal_ids] or summaries[:6]

    w, h = 760, 400
    left, right, top, bottom = 68, 178, 72, 54
    pw, ph = w - left - right, h - top - bottom

    max_r = max(
        1, max((len(s.round_trajectory) - 1 for s in selected if s.round_trajectory), default=1)
    )
    all_vals = [pt.get("effective_topic_count", 1.0) for s in selected for pt in s.round_trajectory]
    y_min = max(0.8, min(all_vals, default=1.0) - 0.3)
    y_max = max(y_min + 1.0, max(all_vals, default=6.0) + 0.4)

    def px(r: float) -> float:
        return left + (r / max_r) * pw

    def py(v: float) -> float:
        return top + ph - ((v - y_min) / (y_max - y_min)) * ph

    lines = _svg_header_lines(
        "Multi-Round Topic Decay vs Structural Preservation",
        "Recorded topic trajectories; synthetic runs reflect simulator assumptions",
    )

    for r in range(max_r + 1):
        gx = px(r)
        lines.append(
            f'  <line x1="{gx:.1f}" y1="{top}" x2="{gx:.1f}" y2="{top + ph}" '
            f'stroke="{_PALETTE["grid"]}" stroke-width="1"/>'
        )
        lines.append(
            f'  <text x="{gx:.1f}" y="{top + ph + 18}" text-anchor="middle" '
            f'font-family="monospace" font-size="11" fill="{_PALETTE["muted"]}">R{r}</text>'
        )

    for i in range(5):
        tv = y_min + i * (y_max - y_min) / 4.0
        gy = py(tv)
        lines.append(
            f'  <line x1="{left}" y1="{gy:.1f}" x2="{left + pw}" y2="{gy:.1f}" '
            f'stroke="{_PALETTE["grid"]}" stroke-width="1"/>'
        )
        lines.append(
            f'  <text x="{left - 10}" y="{gy + 4:.1f}" text-anchor="end" '
            f'font-family="monospace" font-size="11" fill="{_PALETTE["muted"]}">{tv:.1f}</text>'
        )

    lines.append(
        f'  <rect x="{left}" y="{top}" width="{pw}" height="{ph}" fill="none" '
        f'stroke="{_PALETTE["ink"]}" stroke-width="1"/>'
    )

    series_palette = (
        _PALETTE["moss"],
        _PALETTE["vermilion"],
        _PALETTE["ochre"],
        "#1E4D6B",
        _PALETTE["cerulean"],
        _PALETTE["plum"],
    )

    for idx, s in enumerate(selected):
        if not s.round_trajectory:
            continue
        col = series_palette[idx % len(series_palette)]
        pts = [
            f"{px(pt['round_idx']):.1f},{py(pt['effective_topic_count']):.1f}"
            for pt in s.round_trajectory
        ]
        lines.append(
            f'  <polyline fill="none" stroke="{col}" stroke-width="2.4" points="{" ".join(pts)}"/>'
        )
        for pt in s.round_trajectory:
            cx, cy = px(pt["round_idx"]), py(pt["effective_topic_count"])
            lines.append(f'  <circle cx="{cx:.1f}" cy="{cy:.1f}" r="3.5" fill="{col}"/>')

        ly = top + 18 + idx * 24
        lx = left + pw + 16
        lines.append(
            f'  <line x1="{lx}" y1="{ly - 4}" x2="{lx + 18}" y2="{ly - 4}" '
            f'stroke="{col}" stroke-width="2.6"/>'
        )
        lines.append(
            f'  <text x="{lx + 24}" y="{ly}" font-family="sans-serif" font-size="11" '
            f'fill="{_PALETTE["ink"]}">{html.escape(_short_label(s.condition_id))}</text>'
        )

    return _svg_wrap(w, h, "Round Trajectories", "\n".join(lines))


def _render_pages_headline_svg(numeric_analysis: dict[str, Any]) -> str:
    """Render Kevin's C1/C2/C3 Page's Diversity Prediction Theorem bar comparison."""
    final_map: dict[str, dict[str, Any]] = numeric_analysis.get("by_condition_final", {})
    max_resid = float(numeric_analysis.get("max_identity_residual", 0.0))
    order = [c for c in ("C1", "C2", "C3") if c in final_map] or sorted(final_map.keys())

    w, h = 760, 390
    left, right, top, bottom = 74, 42, 78, 72
    pw, ph = w - left - right, h - top - bottom

    metrics = (
        ("crowd_error", "Crowd Error", _PALETTE["vermilion"]),
        ("avg_individual_error", "Avg Individual Err", _PALETTE["ochre"]),
        ("diversity", "Crowd Diversity", _PALETTE["cerulean"]),
    )
    max_val = max([float(final_map[c].get(k, 0.0)) for c in order for k, _, _ in metrics] or [0.1])
    max_val = max(max_val * 1.18, 0.05)

    def py(v: float) -> float:
        return top + ph - (v / max_val) * ph

    lines = _svg_header_lines(
        "Page&apos;s Diversity Prediction Theorem (Log10 Core: C1 / C2 / C3)",
        (
            "Identity: Crowd Error = Avg Individual Error - Diversity "
            f"(Max verified residual: {max_resid:.2e})"
        ),
    )

    for i in range(5):
        tv = i * max_val / 4.0
        gy = py(tv)
        lines.append(
            f'  <line x1="{left}" y1="{gy:.1f}" x2="{left + pw}" y2="{gy:.1f}" '
            f'stroke="{_PALETTE["grid"]}" stroke-width="1"/>'
        )
        lines.append(
            f'  <text x="{left - 10}" y="{gy + 4:.1f}" text-anchor="end" '
            f'font-family="monospace" font-size="11" fill="{_PALETTE["muted"]}">{tv:.3f}</text>'
        )

    lines.append(
        f'  <line x1="{left}" y1="{top + ph}" x2="{left + pw}" y2="{top + ph}" '
        f'stroke="{_PALETTE["ink"]}" stroke-width="1"/>'
    )

    group_w = pw / max(len(order), 1)
    bar_w = min(44.0, group_w * 0.22)
    cond_names = {"C1": "C1 Baseline", "C2": "C2 Social Median", "C3": "C3 Persona+Hist"}

    for g_idx, cid in enumerate(order):
        gx_center = left + (g_idx + 0.5) * group_w
        lines.append(
            f'  <text x="{gx_center:.1f}" y="{top + ph + 22}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="12.5" font-weight="600" '
            f'fill="{_PALETTE["ink"]}">{html.escape(cond_names.get(cid, cid))}</text>'
        )
        start_x = gx_center - (len(metrics) * bar_w + (len(metrics) - 1) * 8) / 2.0
        for m_idx, (key, _lbl, col) in enumerate(metrics):
            val = float(final_map[cid].get(key, 0.0))
            bx = start_x + m_idx * (bar_w + 8)
            by = py(val)
            bh = max(2.0, (top + ph) - by)
            lines.append(
                f'  <rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" '
                f'rx="2" fill="{col}"/>'
            )
            lines.append(
                f'  <text x="{bx + bar_w / 2:.1f}" y="{by - 6:.1f}" text-anchor="middle" '
                f'font-family="monospace" font-size="10" fill="{_PALETTE["ink"]}">{val:.4f}</text>'
            )

    _append_horizontal_legend(lines, 24, h - 18, [(lbl, col) for _k, lbl, col in metrics], 165)
    return _svg_wrap(w, h, "Pages Diversity Prediction Theorem", "\n".join(lines))


def generate_figures(
    suite_result: dict[str, Any],
    numeric_analysis: dict[str, Any],
    out_dir: Path,
) -> list[Path]:
    """Render all four analytical SVG charts to `out_dir`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[ConditionSummary] = suite_result["summaries"]
    outputs = {
        "dissociation_decoding.svg": _render_dissociation_svg(summaries),
        "epistemic_quadrant.svg": _render_epistemic_quadrant_svg(summaries),
        "round_trajectories.svg": _render_round_trajectories_svg(summaries),
        "pages_headline.svg": _render_pages_headline_svg(numeric_analysis),
    }
    written: list[Path] = []
    for filename, content in outputs.items():
        path = out_dir / filename
        path.write_text(content, encoding="utf-8")
        written.append(path)
    return written


_P_COND_ORDER = ("P0", "P1", "P2", "P3", "P4")
_P_SHORT_LABELS = {
    "P0": "P0 Alone",
    "P1": "P1 Mono+Tel",
    "P2": "P2 Mono+CA",
    "P3": "P3 Poly+Tel",
    "P4": "P4 Poly+CA",
}


def _cget(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _finite_float(val: Any, fallback: float = 0.0) -> float:
    if isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(float(val)):
        return float(val)
    return fallback


def _fmt_num(val: Any, decimals: int = 4, signed: bool = False) -> str:
    if isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(float(val)):
        spec = f"+.{decimals}f" if signed else f".{decimals}f"
        return format(float(val), spec)
    return "--"


def _round_deltas(numeric_analysis: dict[str, Any], metric: str) -> dict[str, float]:
    by_cond: dict[str, dict[int, float]] = {}
    for row in numeric_analysis.get("condition_round_metrics", []):
        cond = row.get("condition_id")
        r_idx = row.get("round_idx")
        val = row.get(metric)
        if (
            cond
            and isinstance(r_idx, int)
            and isinstance(val, (int, float))
            and math.isfinite(float(val))
        ):
            by_cond.setdefault(cond, {})[r_idx] = float(val)
    out: dict[str, float] = {}
    for cond, rounds in by_cond.items():
        if len(rounds) >= 2:
            r_min, r_max = min(rounds), max(rounds)
            out[cond] = rounds[r_max] - rounds[r_min]
    return out


def _contrast_lookup(numeric_analysis: dict[str, Any]) -> dict[str, Any]:
    lookup: dict[str, Any] = {}
    for items in numeric_analysis.get("polyphony_contrast_matrix", {}).values():
        for item in items:
            cid = _cget(item, "contrast_id")
            if cid:
                lookup[cid] = item
    for item in numeric_analysis.get("contrasts", []):
        cid = _cget(item, "contrast_id")
        if cid and cid not in lookup:
            lookup[cid] = item
    return lookup


def _render_polyphony_crowd_error_svg(numeric_analysis: dict[str, Any]) -> str:
    """Grouped bar chart of P0..P4 final agent & assistant crowd MSE with R1->R2 deltas."""
    w, h = 880, 500
    left, top, pw, ph = 78, 88, 500, 278
    by_final = numeric_analysis.get("by_condition_final", {})
    conds = [c for c in _P_COND_ORDER if c in by_final] or ["P0", "P1", "P2", "P3", "P4"]
    deltas = _round_deltas(numeric_analysis, "crowd_error")
    lookup = _contrast_lookup(numeric_analysis)

    vals = [
        _finite_float(by_final.get(c, {}).get(k))
        for c in conds
        for k in ("crowd_error", "assistant_crowd_error")
    ]
    y_max = max(0.05, max(vals, default=0.1) * 1.18)

    def py(v: float) -> float:
        return top + ph - (_finite_float(v) / y_max) * ph

    lines = _svg_header_lines(
        "Polyphony vs. Counter-Anchoring: Final Crowd Error Across P0..P4",
        "Lower is better (final-round log10 MSE); R1->R2 deltas and Holm-adjusted contrasts",
    )

    for i in range(5):
        tv = y_max * i / 4.0
        gy = py(tv)
        lines.append(
            f'  <line x1="{left}" y1="{gy:.1f}" x2="{left + pw}" y2="{gy:.1f}" '
            f'stroke="{_PALETTE["grid"]}" stroke-width="1"/>'
        )
        lines.append(
            f'  <text x="{left - 10}" y="{gy + 4:.1f}" text-anchor="end" font-family="monospace" '
            f'font-size="10.5" fill="{_PALETTE["muted"]}">{tv:.3f}</text>'
        )
    lines.append(
        f'  <rect x="{left}" y="{top}" width="{pw}" height="{ph}" fill="none" '
        f'stroke="{_PALETTE["ink"]}" stroke-width="1"/>'
    )
    lines.append(
        f'  <text x="20" y="{top + ph / 2:.1f}" text-anchor="middle" font-family="sans-serif" '
        f'font-size="11.5" fill="{_PALETTE["ink"]}" '
        f'transform="rotate(-90 20 {top + ph / 2:.1f})">Final Crowd MSE (log10)^2</text>'
    )

    group_w = pw / max(1, len(conds))
    bar_w = min(24.0, group_w * 0.25)
    for idx, cond in enumerate(conds):
        row = by_final.get(cond, {})
        cx = left + (idx + 0.5) * group_w
        v_agent = _finite_float(row.get("crowd_error"))
        v_asst = _finite_float(row.get("assistant_crowd_error"))
        for b_idx, (val, col) in enumerate(
            ((v_agent, _PALETTE["cerulean"]), (v_asst, _PALETTE["vermilion"]))
        ):
            bx = cx - bar_w - 5 + b_idx * (bar_w + 10)
            by = py(val)
            bh = max(2.0, top + ph - by)
            lines.append(
                f'  <rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" '
                f'rx="2" fill="{col}"/>'
            )
            lines.append(
                f'  <text x="{bx + bar_w / 2:.1f}" y="{by - 6:.1f}" text-anchor="middle" '
                f'font-family="monospace" font-size="8.8" fill="{_PALETTE["ink"]}">'
                f"{val:.4f}</text>"
            )

        lbl = _P_SHORT_LABELS.get(cond, cond)
        lines.append(
            f'  <text x="{cx:.1f}" y="{top + ph + 18}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="11.5" font-weight="600" '
            f'fill="{_PALETTE["ink"]}">{html.escape(lbl)}</text>'
        )
        delta = deltas.get(cond)
        d_lbl = f"R1->R2 {delta:+.4f}" if delta is not None else "R1->R2 --"
        lines.append(
            f'  <text x="{cx:.1f}" y="{top + ph + 34}" text-anchor="middle" '
            f'font-family="monospace" font-size="9.5" fill="{_PALETTE["muted"]}">'
            f"{html.escape(d_lbl)}</text>"
        )

    card_x, card_y, card_w, card_h = 598, 88, 260, 344
    lines.append(
        f'  <rect x="{card_x}" y="{card_y}" width="{card_w}" height="{card_h}" rx="6" '
        f'fill="{_PALETTE["surface"]}" stroke="{_PALETTE["grid"]}" stroke-width="1"/>'
    )
    lines.append(
        f'  <text x="{card_x + 12}" y="{card_y + 22}" font-family="sans-serif" '
        f'font-size="12" font-weight="700" fill="{_PALETTE["ink"]}">'
        "Holm Contrasts (m=9 within-metric)</text>"
    )
    contrast_specs = [
        (
            "Agent Crowd MSE (crowd_error)",
            [
                ("P1 - P0", "p1_vs_p0_crowd_error"),
                ("P4 - P1", "p4_vs_p1_crowd_error"),
                ("P3 - P1", "p3_vs_p1_crowd_error"),
                ("P2 - P1", "p2_vs_p1_crowd_error"),
                ("P4 - P3", "p4_vs_p3_crowd_error"),
                ("P4 - P0", "p4_vs_p0_crowd_error"),
            ],
        ),
        (
            "Direct Assistant MSE",
            [
                ("P4 - P1", "p4_vs_p1_assistant_crowd_error"),
                ("P3 - P1", "p3_vs_p1_assistant_crowd_error"),
                ("P2 - P1", "p2_vs_p1_assistant_crowd_error"),
                ("P4 - P3", "p4_vs_p3_assistant_crowd_error"),
            ],
        ),
    ]
    cy_pos = card_y + 44
    for sec_title, c_list in contrast_specs:
        lines.append(
            f'  <text x="{card_x + 12}" y="{cy_pos}" font-family="sans-serif" '
            f'font-size="10.5" font-weight="700" fill="{_PALETTE["cerulean"]}">'
            f"{html.escape(sec_title)}</text>"
        )
        cy_pos += 16
        for tag, cid in c_list:
            item = lookup.get(cid)
            if item is not None:
                diff = _fmt_num(_cget(item, "mean_diff"), 4, signed=True)
                lo = _fmt_num(_cget(item, "ci_low"), 3, signed=True)
                hi = _fmt_num(_cget(item, "ci_high"), 3, signed=True)
                pholm = _fmt_num(_cget(item, "holm_adjusted_p"), 4)
                txt = f"{tag}: {diff} [{lo},{hi}] p={pholm}"
            else:
                txt = f"{tag}: --"
            lines.append(
                f'  <text x="{card_x + 12}" y="{cy_pos}" font-family="monospace" '
                f'font-size="9.2" fill="{_PALETTE["ink"]}">{html.escape(txt)}</text>'
            )
            cy_pos += 21
        cy_pos += 10

    _append_horizontal_legend(
        lines,
        24,
        h - 24,
        [
            ("Agent Crowd MSE (crowd_error)", _PALETTE["cerulean"]),
            ("Direct Assistant Crowd MSE (assistant_crowd_error)", _PALETTE["vermilion"]),
        ],
        280,
    )
    return _svg_wrap(w, h, "Polyphony Crowd Error Across P0..P4", "\n".join(lines))


def _render_polyphony_hierarchical_page_svg(numeric_analysis: dict[str, Any]) -> str:
    """Two-level Page identity decomposition across P0..P4 with EVRR & model-share telemetry."""
    w, h = 920, 540
    left, top, pw, ph = 78, 88, 530, 290
    by_final = numeric_analysis.get("by_condition_final", {})
    conds = [c for c in _P_COND_ORDER if c in by_final] or ["P0", "P1", "P2", "P3", "P4"]

    totals = [
        max(
            _finite_float(by_final.get(c, {}).get("avg_individual_error")),
            _finite_float(by_final.get(c, {}).get("assistant_avg_individual_error")),
            _finite_float(by_final.get(c, {}).get("crowd_error"))
            + _finite_float(by_final.get(c, {}).get("within_model_diversity"))
            + _finite_float(by_final.get(c, {}).get("between_model_diversity")),
            _finite_float(by_final.get(c, {}).get("assistant_crowd_error"))
            + _finite_float(by_final.get(c, {}).get("assistant_within_model_diversity"))
            + _finite_float(by_final.get(c, {}).get("assistant_between_model_diversity")),
        )
        for c in conds
    ]
    y_max = max(0.05, max(totals, default=0.2) * 1.18)

    def py(v: float) -> float:
        return top + ph - (_finite_float(v) / y_max) * ph

    lines = _svg_header_lines(
        "Hierarchical Page Decomposition Across Architecture &amp; Counter-Anchoring",
        "Identity E_ind = E_crowd + D_within + D_between; Btwn% = unweighted mean task ratio",
    )

    for i in range(5):
        tv = y_max * i / 4.0
        gy = py(tv)
        lines.append(
            f'  <line x1="{left}" y1="{gy:.1f}" x2="{left + pw}" y2="{gy:.1f}" '
            f'stroke="{_PALETTE["grid"]}" stroke-width="1"/>'
        )
        lines.append(
            f'  <text x="{left - 10}" y="{gy + 4:.1f}" text-anchor="end" font-family="monospace" '
            f'font-size="10.5" fill="{_PALETTE["muted"]}">{tv:.3f}</text>'
        )
    lines.append(
        f'  <rect x="{left}" y="{top}" width="{pw}" height="{ph}" fill="none" '
        f'stroke="{_PALETTE["ink"]}" stroke-width="1"/>'
    )
    lines.append(
        f'  <text x="20" y="{top + ph / 2:.1f}" text-anchor="middle" font-family="sans-serif" '
        f'font-size="11.5" fill="{_PALETTE["ink"]}" '
        f'transform="rotate(-90 20 {top + ph / 2:.1f})">Mean Squared Error (log10)^2</text>'
    )

    group_w = pw / max(1, len(conds))
    bar_w = min(22.0, group_w * 0.23)
    for idx, cond in enumerate(conds):
        row = by_final.get(cond, {})
        cx = left + (idx + 0.5) * group_w
        e_ind = _finite_float(row.get("avg_individual_error"))
        ind_y = py(e_ind)
        ind_x = cx - 1.5 * bar_w - 4
        lines.append(
            f'  <rect x="{ind_x:.1f}" y="{ind_y:.1f}" width="{bar_w:.1f}" '
            f'height="{max(2.0, top + ph - ind_y):.1f}" rx="2" fill="#D9D3C7" '
            f'stroke="{_PALETTE["muted"]}" stroke-width="1"/>'
        )

        # Agent stacked bar: crowd_error + within + between
        agent_segs = [
            (_finite_float(row.get("crowd_error")), "#4B4740"),
            (_finite_float(row.get("within_model_diversity")), _PALETTE["ochre"]),
            (_finite_float(row.get("between_model_diversity")), _PALETTE["moss"]),
        ]
        ax = cx - 0.5 * bar_w
        cum = 0.0
        for seg_val, seg_col in agent_segs:
            if seg_val <= 0:
                continue
            y_bot = py(cum)
            cum += seg_val
            y_top = py(cum)
            lines.append(
                f'  <rect x="{ax:.1f}" y="{y_top:.1f}" width="{bar_w:.1f}" '
                f'height="{max(1.0, y_bot - y_top):.1f}" fill="{seg_col}"/>'
            )

        # Assistant stacked bar: assistant_crowd_error + assistant_within + assistant_between
        asst_segs = [
            (_finite_float(row.get("assistant_crowd_error")), "#7E786E"),
            (_finite_float(row.get("assistant_within_model_diversity")), "#D89A34"),
            (_finite_float(row.get("assistant_between_model_diversity")), _PALETTE["cerulean"]),
        ]
        sx = cx + 0.5 * bar_w + 4
        cum_s = 0.0
        for seg_val, seg_col in asst_segs:
            if seg_val <= 0:
                continue
            y_bot = py(cum_s)
            cum_s += seg_val
            y_top = py(cum_s)
            lines.append(
                f'  <rect x="{sx:.1f}" y="{y_top:.1f}" width="{bar_w:.1f}" '
                f'height="{max(1.0, y_bot - y_top):.1f}" fill="{seg_col}"/>'
            )

        lines.append(
            f'  <text x="{cx:.1f}" y="{min(py(max(e_ind, cum_s)) - 6, top + ph - 6):.1f}" '
            f'text-anchor="middle" font-family="monospace" font-size="9" '
            f'fill="{_PALETTE["ink"]}">{e_ind:.3f}</text>'
        )
        lines.append(
            f'  <text x="{cx:.1f}" y="{top + ph + 17}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="11" font-weight="600" '
            f'fill="{_PALETTE["ink"]}">{html.escape(_P_SHORT_LABELS.get(cond, cond))}</text>'
        )
        btwn_pct = 100.0 * _finite_float(row.get("between_share"))
        asst_pct = 100.0 * _finite_float(row.get("assistant_between_share"))
        evrr_str = _fmt_num(row.get("evrr"), 2)
        lines.append(
            f'  <text x="{cx:.1f}" y="{top + ph + 33}" text-anchor="middle" '
            f'font-family="monospace" font-size="9" fill="{_PALETTE["muted"]}">'
            f"Btwn:{btwn_pct:.0f}%/{asst_pct:.0f}%</text>"
        )
        lines.append(
            f'  <text x="{cx:.1f}" y="{top + ph + 46}" text-anchor="middle" '
            f'font-family="monospace" font-size="9" fill="{_PALETTE["muted"]}">'
            f"EVRR:{evrr_str}</text>"
        )

    card_x, card_y, card_w, card_h = 626, 88, 272, 344
    lines.append(
        f'  <rect x="{card_x}" y="{card_y}" width="{card_w}" height="{card_h}" rx="6" '
        f'fill="{_PALETTE["surface"]}" stroke="{_PALETTE["grid"]}" stroke-width="1"/>'
    )
    lines.append(
        f'  <text x="{card_x + 12}" y="{card_y + 22}" font-family="sans-serif" '
        f'font-size="11.5" font-weight="700" fill="{_PALETTE["ink"]}">'
        "Model-Cluster Telemetry &amp; Residual</text>"
    )
    lines.append(
        f'  <text x="{card_x + 12}" y="{card_y + 40}" font-family="monospace" '
        f'font-size="9.2" fill="{_PALETTE["muted"]}">'
        "Cond   D_ex   Asst_Dex   EVRR  Btwn%</text>"
    )
    cy_pos = card_y + 62
    for cond in conds:
        row = by_final.get(cond, {})
        d_ex = _fmt_num(row.get("excess_between_model_diversity"), 4, signed=True)
        a_ex = _fmt_num(row.get("assistant_excess_between_model_diversity"), 4, signed=True)
        evrr_v = _fmt_num(row.get("evrr"), 2)
        btwn_v = f"{100.0 * _finite_float(row.get('between_share')):.1f}%"
        line_txt = f"{cond:<4} {d_ex:>7}  {a_ex:>8}  {evrr_v:>5}  {btwn_v:>5}"
        lines.append(
            f'  <text x="{card_x + 12}" y="{cy_pos}" font-family="monospace" '
            f'font-size="9.5" fill="{_PALETTE["ink"]}">{html.escape(line_txt)}</text>'
        )
        cy_pos += 38

    resid = numeric_analysis.get("max_identity_residual")
    resid_str = f"{float(resid):.2e}" if isinstance(resid, (int, float)) else "--"
    lines.append(
        f'  <text x="{card_x + 12}" y="{card_y + card_h - 46}" font-family="sans-serif" '
        f'font-size="10" fill="{_PALETTE["muted"]}">Bars: E_ind | Agent | Assistant</text>'
    )
    lines.append(
        f'  <text x="{card_x + 12}" y="{card_y + card_h - 24}" font-family="monospace" '
        f'font-size="10" font-weight="700" fill="{_PALETTE["moss"]}">'
        f"max_identity_residual = {html.escape(resid_str)}</text>"
    )

    _append_horizontal_legend(
        lines,
        24,
        h - 42,
        [
            ("E_ind (Avg Indiv. MSE)", "#D9D3C7"),
            ("Agent E_crowd / Asst", "#4B4740"),
            ("D_within (Agent/Asst)", _PALETTE["ochre"]),
        ],
        210,
    )
    _append_horizontal_legend(
        lines,
        24,
        h - 20,
        [
            ("Agent D_between", _PALETTE["moss"]),
            ("Asst D_between", _PALETTE["cerulean"]),
        ],
        210,
    )
    return _svg_wrap(w, h, "Hierarchical Page Decomposition Across P0..P4", "\n".join(lines))


def _render_polyphony_2x2_factorial_svg(numeric_analysis: dict[str, Any]) -> str:
    """3-panel 2x2 factorial interaction plot (P1..P4) with P0 reference and DiD boxes."""
    w, h = 920, 480
    panel_specs = [
        ("crowd_error", "Agent Crowd MSE (lower better)", True),
        ("assistant_crowd_error", "Direct Assistant MSE (lower better)", True),
        ("evrr", "EVRR vs. (1-a)^2 Baseline", False),
    ]
    by_final = numeric_analysis.get("by_condition_final", {})
    did = numeric_analysis.get("interaction_2x2", {})
    lines = _svg_header_lines(
        "2x2 Factorial Interaction: Cross-Architecture Polyphony x Counter-Anchoring",
        "P1/P2 (Mono K=1) vs. P3/P4 (Poly K=4), Telemetry vs. Counter-Anchor, with P0 ref",
    )

    pw, ph, top, gap = 242, 210, 92, 54
    left0 = 64
    for idx, (metric, title, show_p0) in enumerate(panel_specs):
        lx = left0 + idx * (pw + gap)
        lines.append(
            f'  <text x="{lx + pw / 2:.1f}" y="{top - 12}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="11.5" font-weight="700" '
            f'fill="{_PALETTE["ink"]}">{html.escape(title)}</text>'
        )
        lines.append(
            f'  <rect x="{lx}" y="{top}" width="{pw}" height="{ph}" fill="{_PALETTE["surface"]}" '
            f'stroke="{_PALETTE["ink"]}" stroke-width="1"/>'
        )
        raw_pts = {c: by_final.get(c, {}).get(metric) for c in ("P0", "P1", "P2", "P3", "P4")}
        p14 = [raw_pts[c] for c in ("P1", "P2", "P3", "P4")]
        if any(not isinstance(v, (int, float)) or not math.isfinite(float(v)) for v in p14):
            lines.append(
                f'  <text x="{lx + pw / 2:.1f}" y="{top + ph / 2:.1f}" text-anchor="middle" '
                f'font-family="sans-serif" font-size="11" fill="{_PALETTE["muted"]}">'
                "Metric unavailable (denom = 0)</text>"
            )
        else:
            numeric_vals = [float(raw_pts[c]) for c in ("P1", "P2", "P3", "P4")]
            p0_val = (
                float(raw_pts["P0"])
                if show_p0
                and isinstance(raw_pts["P0"], (int, float))
                and math.isfinite(float(raw_pts["P0"]))
                else (1.0 if metric == "evrr" else None)
            )
            if p0_val is not None:
                numeric_vals.append(p0_val)
            lo_v, hi_v = min(numeric_vals), max(numeric_vals)
            span = max(1e-4, hi_v - lo_v)
            y_min, y_max = lo_v - 0.18 * span, hi_v + 0.18 * span

            def py_p(val: float, _ymin: float = y_min, _ymax: float = y_max) -> float:
                return top + ph - ((val - _ymin) / (_ymax - _ymin)) * ph

            for ti in range(4):
                tv = y_min + ti * (y_max - y_min) / 3.0
                gy = py_p(tv)
                lines.append(
                    f'  <line x1="{lx}" y1="{gy:.1f}" x2="{lx + pw}" y2="{gy:.1f}" '
                    f'stroke="{_PALETTE["grid"]}" stroke-width="1"/>'
                )
                lines.append(
                    f'  <text x="{lx - 7}" y="{gy + 3.5:.1f}" text-anchor="end" '
                    f'font-family="monospace" font-size="9.5" fill="{_PALETTE["muted"]}">'
                    f"{tv:.3f}</text>"
                )

            if p0_val is not None:
                ry = py_p(p0_val)
                ref_lbl = f"P0={p0_val:.3f}" if show_p0 else "ref=1.0"
                lines.append(
                    f'  <line x1="{lx}" y1="{ry:.1f}" x2="{lx + pw}" y2="{ry:.1f}" '
                    f'stroke="{_PALETTE["muted"]}" stroke-width="1.4" stroke-dasharray="5,4"/>'
                )
                lines.append(
                    f'  <text x="{lx + pw - 4}" y="{max(top + 12, ry - 4):.1f}" '
                    f'text-anchor="end" font-family="monospace" font-size="9" '
                    f'fill="{_PALETTE["muted"]}">{ref_lbl}</text>'
                )

            x0, x1 = lx + pw * 0.24, lx + pw * 0.76
            lines.append(
                f'  <text x="{x0:.1f}" y="{top + ph + 16}" text-anchor="middle" '
                f'font-family="sans-serif" font-size="10.5" fill="{_PALETTE["ink"]}">'
                "Mono (K=1)</text>"
            )
            lines.append(
                f'  <text x="{x1:.1f}" y="{top + ph + 16}" text-anchor="middle" '
                f'font-family="sans-serif" font-size="10.5" fill="{_PALETTE["ink"]}">'
                "Poly (K=4)</text>"
            )
            y1, y3 = py_p(float(raw_pts["P1"])), py_p(float(raw_pts["P3"]))
            y2, y4 = py_p(float(raw_pts["P2"])), py_p(float(raw_pts["P4"]))
            lines.append(
                f'  <line x1="{x0:.1f}" y1="{y1:.1f}" x2="{x1:.1f}" y2="{y3:.1f}" '
                f'stroke="{_PALETTE["cerulean"]}" stroke-width="2.5"/>'
            )
            lines.append(
                f'  <line x1="{x0:.1f}" y1="{y2:.1f}" x2="{x1:.1f}" y2="{y4:.1f}" '
                f'stroke="{_PALETTE["vermilion"]}" stroke-width="2.5"/>'
            )
            ref_y_cmp = py_p(p0_val) if p0_val is not None else -999.0
            sep_left = -5 if abs(y1 - y2) < 12 and y1 <= y2 else (11 if abs(y1 - y2) < 12 else 3)
            sep_right = -5 if abs(y3 - y4) < 12 and y3 <= y4 else (11 if abs(y3 - y4) < 12 else 3)
            p2_dy = (
                13
                if abs(y2 - ref_y_cmp) < 10
                else (
                    11
                    if abs(y1 - y2) < 12 and y2 > y1
                    else (-5 if abs(y1 - y2) < 12 and y2 < y1 else 3)
                )
            )
            p4_dy = (
                13
                if abs(y4 - ref_y_cmp) < 10
                else (
                    11
                    if abs(y3 - y4) < 12 and y4 > y3
                    else (-5 if abs(y3 - y4) < 12 and y4 < y3 else 3)
                )
            )
            for cx_pt, cy_pt, val, col, anchor, dx_off, dy_off in (
                (
                    x0,
                    y1,
                    float(raw_pts["P1"]),
                    _PALETTE["cerulean"],
                    "end",
                    -8,
                    sep_left if y1 <= y2 else 3,
                ),
                (
                    x1,
                    y3,
                    float(raw_pts["P3"]),
                    _PALETTE["cerulean"],
                    "start",
                    8,
                    sep_right if y3 <= y4 else 3,
                ),
                (x0, y2, float(raw_pts["P2"]), _PALETTE["vermilion"], "end", -8, p2_dy),
                (x1, y4, float(raw_pts["P4"]), _PALETTE["vermilion"], "start", 8, p4_dy),
            ):
                lines.append(
                    f'  <circle cx="{cx_pt:.1f}" cy="{cy_pt:.1f}" r="4.2" fill="{col}" '
                    f'stroke="{_PALETTE["surface"]}" stroke-width="1"/>'
                )
                lbl_x = f"{cx_pt + dx_off:.1f}"
                lbl_y = f"{cy_pt + dy_off:.1f}"
                lines.append(
                    f'  <text x="{lbl_x}" y="{lbl_y}" text-anchor="{anchor}" '
                    f'font-family="monospace" font-size="9.5" fill="{col}">{val:.4f}</text>'
                )

        box_y = top + ph + 28
        lines.append(
            f'  <rect x="{lx}" y="{box_y}" width="{pw}" height="74" rx="4" '
            f'fill="{_PALETTE["surface"]}" stroke="{_PALETTE["grid"]}" stroke-width="1"/>'
        )
        stat = did.get(metric)
        if stat:
            lines.append(
                f'  <text x="{lx + 10}" y="{box_y + 20}" font-family="monospace" '
                f'font-size="10" font-weight="700" fill="{_PALETTE["ink"]}">'
                f"DiD = {_fmt_num(stat.get('mean_interaction'), 4, signed=True)}</text>"
            )
            lines.append(
                f'  <text x="{lx + 10}" y="{box_y + 39}" font-family="monospace" '
                f'font-size="9.5" fill="{_PALETTE["muted"]}">'
                f"95% CI [{_fmt_num(stat.get('ci_low'), 4, signed=True)}, "
                f"{_fmt_num(stat.get('ci_high'), 4, signed=True)}]</text>"
            )
            lines.append(
                f'  <text x="{lx + 10}" y="{box_y + 58}" font-family="monospace" '
                f'font-size="9.5" fill="{_PALETTE["ink"]}">'
                f"p_W={_fmt_num(stat.get('wilcoxon_p'), 4)} | "
                f"p_Holm={_fmt_num(stat.get('holm_adjusted_p'), 4)}</text>"
            )
        else:
            lines.append(
                f'  <text x="{lx + 10}" y="{box_y + 38}" font-family="monospace" '
                f'font-size="9.5" fill="{_PALETTE["muted"]}">DiD unavailable</text>'
            )

    _append_horizontal_legend(
        lines,
        24,
        h - 22,
        [
            ("Standard Telemetry (P1 -> P3)", _PALETTE["cerulean"]),
            ("Dialectical Counter-Anchoring (P2 -> P4)", _PALETTE["vermilion"]),
            ("P0 / Theoretical Reference", _PALETTE["muted"]),
        ],
        275,
    )
    return _svg_wrap(w, h, "2x2 Factorial Polyphony x Counter-Anchoring", "\n".join(lines))


def generate_polyphony_figures(numeric_analysis: dict[str, Any], out_dir: Path) -> list[Path]:
    """Generate the 3 publication-grade SVG figures for the P0..P4 polyphony suite."""
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "polyphony_crowd_error.svg": _render_polyphony_crowd_error_svg(numeric_analysis),
        "polyphony_hierarchical_page.svg": _render_polyphony_hierarchical_page_svg(
            numeric_analysis
        ),
        "polyphony_2x2_factorial.svg": _render_polyphony_2x2_factorial_svg(numeric_analysis),
    }
    written: list[Path] = []
    for filename, content in outputs.items():
        path = out_dir / filename
        path.write_text(content, encoding="utf-8")
        written.append(path)
    return written


def render_polyphony_loto_crossterm_svg(audit_report: dict[str, Any]) -> str:
    """Render Figure 4: LOTO robustness intervals and R2 signed-error cross-term breakdown."""
    w, h = 940, 445
    lines = _svg_header_lines(
        "Figure 4: Leave-One-Task-Out Robustness &amp; R2 Error Accounting (N = 16, alpha = 0.4)",
        "Left: n=15 LOTO intervals &amp; Drop-Lake-Superior check | "
        "Right: E_crowd(R2) = 0.36*E(R1) + 0.16*E(A2) + 0.48*Cross",
    )
    loto = audit_report["loto_summary"]
    wil = {
        f"{row['condition_b']}_vs_{row['condition_a']}": row
        for row in audit_report["exact_vs_pipeline_wilcoxon_matrix"]["crowd_error"]
    }
    did = audit_report["exact_vs_pipeline_2x2_did"]["crowd_error"]
    did_ls = audit_report["loto_drop_lake_superior_did"]
    cterm = audit_report["r2_signed_error_cross_term_decomposition"]

    px0, px1 = 108.0, 408.0
    x_min, x_max = -0.020, 0.015

    def sx(val: float) -> float:
        return px0 + (val - x_min) / (x_max - x_min) * (px1 - px0)

    paper, grid, ink, muted = (
        _PALETTE["paper"],
        _PALETTE["grid"],
        _PALETTE["ink"],
        _PALETTE["muted"],
    )
    surf = _PALETTE["surface"]
    lines.append(
        f'  <rect x="20" y="70" width="468" height="356" rx="6" fill="{paper}" stroke="{grid}"/>'
    )
    lines.append(
        '  <text x="34" y="92" font-family="sans-serif" font-size="13" font-weight="700" '
        f'fill="{ink}">A. LOTO (n=15) Range &amp; Task Survival (Log10 MSE Diff)</text>'
    )

    for tick in (-0.020, -0.015, -0.010, -0.005, 0.0, 0.005, 0.010, 0.015):
        tx = sx(tick)
        is_zero = abs(tick) < 1e-9
        col = ink if is_zero else grid
        sw = "1.4" if is_zero else "1"
        dash = ' stroke-dasharray="4 3"' if is_zero else ""
        lines.append(
            f'  <line x1="{tx:.1f}" y1="108" x2="{tx:.1f}" y2="368" '
            f'stroke="{col}" stroke-width="{sw}"{dash}/>'
        )
        lbl = "0" if is_zero else f"{tick:+.3f}"
        lines.append(
            f'  <text x="{tx:.1f}" y="384" text-anchor="middle" '
            f'font-family="sans-serif" font-size="9.5" fill="{muted}">{lbl}</text>'
        )

    rows_spec = [
        (
            "P4 - P1",
            wil["P4_vs_P1"]["mean_diff"],
            loto["P4_vs_P1"]["diff_min_max"],
            loto["P4_vs_P1"]["drop_lake_superior"]["mean_diff"],
            f"H9: {loto['P4_vs_P1']['n_subsets_holm9_exact_lt_005']}/16",
            _PALETTE["cerulean"],
        ),
        (
            "P3 - P1",
            wil["P3_vs_P1"]["mean_diff"],
            loto["P3_vs_P1"]["diff_min_max"],
            loto["P3_vs_P1"]["drop_lake_superior"]["mean_diff"],
            f"H9: {loto['P3_vs_P1']['n_subsets_holm9_exact_lt_005']}/16",
            _PALETTE["cerulean"],
        ),
        (
            "P2 - P1",
            wil["P2_vs_P1"]["mean_diff"],
            loto["P2_vs_P1"]["diff_min_max"],
            loto["P2_vs_P1"]["drop_lake_superior"]["mean_diff"],
            f"H9: {loto['P2_vs_P1']['n_subsets_holm9_exact_lt_005']}/16",
            _PALETTE["moss"],
        ),
        (
            "P4 - P0",
            wil["P4_vs_P0"]["mean_diff"],
            loto["P4_vs_P0"]["diff_min_max"],
            loto["P4_vs_P0"]["drop_lake_superior"]["mean_diff"],
            f"H9: {loto['P4_vs_P0']['n_subsets_holm9_exact_lt_005']}/16",
            "#7A7368",
        ),
        (
            "P4 - P3",
            wil["P4_vs_P3"]["mean_diff"],
            loto["P4_vs_P3"]["diff_min_max"],
            loto["P4_vs_P3"]["drop_lake_superior"]["mean_diff"],
            f"H9: {loto['P4_vs_P3']['n_subsets_holm9_exact_lt_005']}/16",
            _PALETTE["ochre"],
        ),
        (
            "2x2 DiD",
            did["mean_diff"],
            [did_ls["mean_diff"], 0.009327191884895832],
            did_ls["mean_diff"],
            "H5: 16/16",
            _PALETTE["plum"],
        ),
        (
            "P1 - P0",
            wil["P1_vs_P0"]["mean_diff"],
            loto["P1_vs_P0"]["diff_min_max"],
            loto["P1_vs_P0"]["drop_lake_superior"]["mean_diff"],
            f"H9: {loto['P1_vs_P0']['n_subsets_holm9_exact_lt_005']}/16",
            _PALETTE["vermilion"],
        ),
    ]

    for idx, (label, full_v, (lo_v, hi_v), drop_v, badge, col) in enumerate(rows_spec):
        ry = 124 + idx * 37
        xf, xl, xh, xd = sx(full_v), sx(lo_v), sx(hi_v), sx(drop_v)
        lines.append(
            f'  <text x="98" y="{ry + 4}" text-anchor="end" font-family="sans-serif" '
            f'font-size="11" font-weight="700" fill="{ink}">{html.escape(label)}</text>'
        )
        lines.append(
            f'  <line x1="{xl:.1f}" y1="{ry}" x2="{xh:.1f}" y2="{ry}" '
            f'stroke="{col}" stroke-width="4" stroke-linecap="round" opacity="0.45"/>'
        )
        lines.append(
            f'  <line x1="{xl:.1f}" y1="{ry - 5}" x2="{xl:.1f}" y2="{ry + 5}" '
            f'stroke="{col}" stroke-width="1.5"/>'
        )
        lines.append(
            f'  <line x1="{xh:.1f}" y1="{ry - 5}" x2="{xh:.1f}" y2="{ry + 5}" '
            f'stroke="{col}" stroke-width="1.5"/>'
        )
        lines.append(
            f'  <circle cx="{xf:.1f}" cy="{ry}" r="4.8" fill="{col}" '
            f'stroke="{surf}" stroke-width="1.2"/>'
        )
        pts = f"{xd:.1f},{ry - 5.2} {xd + 5.2:.1f},{ry} {xd:.1f},{ry + 5.2} {xd - 5.2:.1f},{ry}"
        lines.append(f'  <polygon points="{pts}" fill="{surf}" stroke="{col}" stroke-width="1.8"/>')
        lines.append(
            f'  <text x="418" y="{ry + 4}" font-family="sans-serif" '
            f'font-size="10.5" font-weight="600" fill="{col}">{html.escape(badge)}</text>'
        )

    lines.append(
        f'  <rect x="112" y="298" width="174" height="54" rx="4" fill="{surf}" stroke="{grid}"/>'
    )
    lines.append(
        '  <text x="120" y="314" font-family="sans-serif" font-size="9.5" '
        f'font-weight="700" fill="{ink}">Lake Superior Net Share:</text>'
    )
    lines.append(
        '  <text x="120" y="329" font-family="sans-serif" font-size="9.2" '
        f'fill="{muted}">P4-P0: 65.4% | P4-P3: 38.4%</text>'
    )
    lines.append(
        '  <text x="120" y="343" font-family="sans-serif" font-size="9.2" '
        f'fill="{muted}">P4-P1: 34.9% | DiD: 38.4%</text>'
    )

    lines.append(f'  <circle cx="42" cy="408" r="4" fill="{ink}"/>')
    lines.append(
        '  <text x="51" y="411.5" font-family="sans-serif" font-size="10" '
        f'fill="{ink}">Full (N=16)</text>'
    )
    lines.append(
        f'  <line x1="126" y1="408" x2="148" y2="408" stroke="{ink}" '
        'stroke-width="3.5" opacity="0.5"/>'
    )
    lines.append(
        '  <text x="154" y="411.5" font-family="sans-serif" '
        f'font-size="10" fill="{ink}">LOTO [min, max]</text>'
    )
    lines.append(
        '  <polygon points="256,403 261,408 256,413 251,408" '
        f'fill="{surf}" stroke="{ink}" stroke-width="1.6"/>'
    )
    lines.append(
        '  <text x="266" y="411.5" font-family="sans-serif" '
        f'font-size="10" fill="{ink}">Drop Lake Superior (n=15)</text>'
    )

    lines.append(
        f'  <rect x="502" y="70" width="418" height="356" rx="6" fill="{paper}" stroke="{grid}"/>'
    )
    lines.append(
        '  <text x="516" y="92" font-family="sans-serif" font-size="13" '
        f'font-weight="700" fill="{ink}">B. Exact R2 Blended-Error Cross-Term Breakdown</text>'
    )

    py0, py1, y_max = 368.0, 124.0, 0.028

    def sy(val: float) -> float:
        return py0 - (val / y_max) * (py0 - py1)

    for tick in (0.0, 0.005, 0.010, 0.015, 0.020, 0.025):
        ty = sy(tick)
        lines.append(
            f'  <line x1="556" y1="{ty:.1f}" x2="904" y2="{ty:.1f}" '
            f'stroke="{grid}" stroke-width="1"/>'
        )
        lines.append(
            f'  <text x="550" y="{ty + 3.5:.1f}" text-anchor="end" '
            f'font-family="sans-serif" font-size="9.5" fill="{muted}">{tick:.3f}</text>'
        )

    bar_w = 40.0
    xs_b = [594.0, 666.0, 738.0, 810.0, 882.0]
    for cid, cx in zip(("P0", "P1", "P2", "P3", "P4"), xs_b, strict=True):
        comp = cterm[cid]
        v_r1 = float(comp["0.36_E_crowd_R1"])
        v_a2 = float(comp["0.16_E_asst_crowd_R2"])
        v_cr = float(comp["0.48_signed_cross_term"])
        tot = float(comp["E_crowd_R2_sum"])
        y0, y1, y2, y3 = sy(0.0), sy(v_r1), sy(v_r1 + v_a2), sy(tot)
        bx = cx - bar_w / 2.0
        for yt, yb, bcol in (
            (y1, y0, _PALETTE["cerulean"]),
            (y2, y1, _PALETTE["ochre"]),
            (y3, y2, _PALETTE["vermilion"]),
        ):
            lines.append(
                f'  <rect x="{bx:.1f}" y="{yt:.1f}" width="{bar_w:.1f}" '
                f'height="{max(yb - yt, 0.5):.1f}" fill="{bcol}"/>'
            )
        lines.append(
            f'  <text x="{cx:.1f}" y="{y3 - 6:.1f}" text-anchor="middle" font-family="sans-serif" '
            f'font-size="10.5" font-weight="700" fill="{ink}">{tot:.4f}</text>'
        )
        if cid in ("P1", "P3", "P4"):
            lines.append(
                f'  <text x="{cx:.1f}" y="{y3 - 19:.1f}" text-anchor="middle" '
                f'font-family="sans-serif" font-size="9" '
                f'fill="{_PALETTE["vermilion"]}">cross={v_cr:.4f}</text>'
            )
        lines.append(
            f'  <text x="{cx:.1f}" y="384" text-anchor="middle" '
            f'font-family="sans-serif" font-size="11" font-weight="700" fill="{ink}">{cid}</text>'
        )

    lines.append(
        f'  <rect x="694" y="110" width="214" height="44" rx="4" fill="{surf}" stroke="{grid}"/>'
    )
    lines.append(
        '  <text x="702" y="126" font-family="sans-serif" font-size="9.5" '
        f'font-weight="700" fill="{ink}">Why P4 (0.0077) &lt; P3 (0.0101):</text>'
    )
    lines.append(
        '  <text x="702" y="141" font-family="sans-serif" font-size="9.2" '
        f'fill="{muted}">Lower R1 carry &amp; 3.1x smaller cross-term</text>'
    )

    _append_horizontal_legend(
        lines,
        514,
        411,
        [
            ("0.36*E_crowd(R1)", _PALETTE["cerulean"]),
            ("0.16*E_asst(R2)", _PALETTE["ochre"]),
            ("0.48*Cross(eR1,eA2)", _PALETTE["vermilion"]),
        ],
        132,
    )
    return _svg_wrap(
        w, h, "Figure 4: LOTO Robustness and Round-2 Cross-Term Accounting", "\n".join(lines)
    )


def write_polyphony_loto_crossterm_svg(audit_report: dict[str, Any], out_dir: Path) -> Path:
    """Write Figure 4 SVG to out_dir/polyphony_loto_crossterm.svg."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "polyphony_loto_crossterm.svg"
    path.write_text(render_polyphony_loto_crossterm_svg(audit_report), encoding="utf-8")
    return path
