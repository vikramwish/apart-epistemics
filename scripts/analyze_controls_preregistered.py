#!/usr/bin/env python3
"""Pre-registered confirmatory analysis for the polyphony control arms.

Commit this file (and cite the commit hash) BEFORE reading any control-arm results.

Why this exists alongside numeric_core._compute_control_contrasts:
  * POOLED_K1_SOCIAL (numeric_core.py L1136-1145) averages the four single-model *crowd errors*.
    P3's crowd mean is the average of four family means, so by Jensen's inequality
    (mean of biases)^2 <= mean of (bias^2): P3 beats the pooled baseline even if mixing does
    nothing beyond static averaging. That contrast is reported here only as a descriptive bound.
  * The confirmatory diversity test instead builds a VIRTUAL MIXED PANEL: for each P3 seat
    (same agent_id, hence the same seeded prior), take that agent's trajectory from the
    single-model social arm of the same model family, and recompute the crowd error.
    Static averaging is identical on both sides, so P3 - VIRTUAL isolates the effect of
    seeing a heterogeneous (cross-model) median instead of a homogeneous one.
  * Silent agents never see each other, so P0_POLY must match VIRTUAL_SILENT (built the same
    way from P0, P0_DS, P0_QWEN, P0_GLM) up to sampling noise. This is a harness sanity check.
  * Non-significance is not equivalence. Equivalence claims use a pre-set margin (TOST via a
    90% bootstrap CI).
  * p-values are exact Wilcoxon signed-rank (tie-aware), not the normal approximation.

Usage:
  python -I analyze_controls_preregistered.py \
      results/polyphony-p0-p4/numeric_runs.jsonl <control_runs.jsonl> \
      --out results/polyphony-controls/preregistered_analysis.json

Only the Python standard library is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# ----------------------------------------------------------------------------------------
# PRE-REGISTERED CONSTANTS (do not edit after control results exist)
# ----------------------------------------------------------------------------------------
# Equivalence margin for P1_REFLECT vs P2: 25% of the baseline P1->P2 rescue
# (mean P2 - P1 crowd error = -0.011081 in results/polyphony-p0-p4/numeric_analysis.json).
DELTA_REFLECT = 0.25 * 0.011081  # = 0.00277
# Equivalence margin for the P0_POLY vs VIRTUAL_SILENT harness check: 25% of baseline P0 mean
# crowd error (0.011944).
DELTA_SANITY = 0.25 * 0.011944  # = 0.00299
ALPHA = 0.05
N_BOOT = 2000
SEED = 42

FAMILY_TOKENS = ("gemma", "deepseek", "qwen", "glm")
K1_SOCIAL = {"gemma": "P1", "deepseek": "P1_DS", "qwen": "P1_QWEN", "glm": "P1_GLM"}
K1_SILENT = {"gemma": "P0", "deepseek": "P0_DS", "qwen": "P0_QWEN", "glm": "P0_GLM"}
VIRTUAL_SOCIAL = "VIRTUAL_SOCIAL"
VIRTUAL_SILENT = "VIRTUAL_SILENT"
POOLED = "POOLED_K1_SOCIAL_JENSEN_BOUND"
BEST_K1_SOCIAL = "BEST_K1_SOCIAL"

# Confirmatory family (Holm, m = 8). Contrast = mean(b - a); negative means b is better.
CONFIRMATORY = [
    ("herding_ds", "P0_DS", "P1_DS"),
    ("herding_qwen", "P0_QWEN", "P1_QWEN"),
    ("herding_glm", "P0_GLM", "P1_GLM"),
    ("reflect_vs_p1", "P1", "P1_REFLECT"),
    ("critique_vs_reflect", "P1_REFLECT", "P2"),
    ("p3_vs_silent_poly", "P0_POLY", "P3"),
    ("p4_vs_silent_poly", "P0_POLY", "P4"),
    ("p3_vs_virtual_social", VIRTUAL_SOCIAL, "P3"),
]
# Secondary family (Holm, m = 2).
SECONDARY = [
    ("p3_vs_best_single_model_social", BEST_K1_SOCIAL, "P3"),
    ("p4_vs_best_single_model_social", BEST_K1_SOCIAL, "P4"),
]
# Descriptive only (no inferential claim).
DESCRIPTIVE = [
    ("p3_vs_pooled_jensen_bound", POOLED, "P3"),
    ("silent_poly_vs_virtual_silent", VIRTUAL_SILENT, "P0_POLY"),
]
# Pre-registered replication contrasts on the P-series (Holm, m = 4).
REPLICATION_P_SERIES = [
    ("p1_vs_p0", "P0", "P1"),
    ("p2_vs_p1", "P1", "P2"),
    ("p4_vs_p3", "P3", "P4"),
    ("p4_vs_p0", "P0", "P4"),
]
SPEC_KEYS_ALLOWED_TO_DIFFER = {"conditions", "numeric_models"}


# ----------------------------------------------------------------------------------------
# Loading and auditing
# ----------------------------------------------------------------------------------------
def family_of(slug: str) -> str:
    low = slug.lower()
    hits = [t for t in FAMILY_TOKENS if t in low]
    if len(hits) != 1:
        raise ValueError(f"Cannot map model slug {slug!r} to exactly one family.")
    return hits[0]


def load_records(paths: list[Path]) -> tuple[dict[tuple, dict], list[tuple[Path, dict]]]:
    """Later lines win for a repeated cell key, mirroring _load_existing_numeric_records."""
    records: dict[tuple, dict] = {}
    specs: list[tuple[Path, dict]] = []
    for path in paths:
        seen_specs: dict[str, dict] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # truncated tail, same tolerance as the pipeline
            key = (
                rec["condition_id"],
                rec["question_id"],
                int(rec["replicate"]),
                int(rec["round_idx"]),
                rec["agent_id"],
            )
            records[key] = rec
            spec = rec.get("run_spec") or {}
            seen_specs.setdefault(json.dumps(spec, sort_keys=True), spec)
        specs.extend((path, s) for s in seen_specs.values())
    return records, specs


def audit_specs(specs: list[tuple[Path, dict]]) -> dict[str, Any]:
    """Report every run_spec key that differs across files beyond conditions/models."""
    keys = sorted({k for _, s in specs for k in s} - SPEC_KEYS_ALLOWED_TO_DIFFER)
    diffs = {}
    for k in keys:
        vals = {}
        for path, s in specs:
            v = s.get(k)
            rendered = json.dumps(v, sort_keys=True)
            if len(rendered) > 120:
                rendered = "sha256:" + hashlib.sha256(rendered.encode()).hexdigest()[:12]
            vals.setdefault(rendered, []).append(path.name)
        if len(vals) > 1:
            diffs[k] = vals
    return {
        "n_distinct_specs": len(specs),
        "differing_keys": diffs,
        "pipeline_merge_will_succeed": not diffs,
        "note": (
            "If differing_keys is non-empty, numeric_core._merge_compatible_run_specs returns {} "
            "and analyze_numeric_core silently reports 'incomplete_or_unverified' with no "
            "control contrasts. A differing current_date also changes the system prompt."
        ),
    }


def truths_from_specs(specs: list[tuple[Path, dict]]) -> dict[str, float]:
    truth: dict[str, float] = {}
    for _, s in specs:
        for t in s.get("tasks", []):
            val = math.log10(float(t["numeric_truth"]))
            if t["id"] in truth and abs(truth[t["id"]] - val) > 1e-12:
                raise RuntimeError(f"Ground truth for {t['id']} differs across run specs.")
            truth[t["id"]] = val
    return truth


# ----------------------------------------------------------------------------------------
# Crowd errors, including virtual panels
# ----------------------------------------------------------------------------------------
def final_round(records: dict[tuple, dict]) -> int:
    return max(k[3] for k in records)


def index_by_condition(records: dict[tuple, dict]) -> dict[str, dict[tuple, dict[str, dict]]]:
    """cond -> (qid, rep, round) -> agent_id -> record."""
    out: dict[str, dict[tuple, dict[str, dict]]] = defaultdict(lambda: defaultdict(dict))
    for (c, q, rep, rnd, a), rec in records.items():
        out[c][(q, rep, rnd)][a] = rec
    return out


def crowd_error(logs: list[float], y: float) -> float:
    return (sum(logs) / len(logs) - y) ** 2


def abs_log_error(logs: list[float], y: float) -> float:
    return abs(sum(logs) / len(logs) - y)


def crowd_metric(logs: list[float], y: float, metric: str = "squared_log_error") -> float:
    if metric == "squared_log_error":
        return crowd_error(logs, y)
    if metric == "abs_log_error":
        return abs_log_error(logs, y)
    raise ValueError(f"Unsupported metric: {metric!r}")


def condition_errors(
    idx: dict,
    cond: str,
    truth: dict[str, float],
    rnd: int,
    n_agents: int,
    metric: str = "squared_log_error",
) -> dict[tuple, float]:
    out = {}
    for (q, rep, r), agents in idx.get(cond, {}).items():
        if r == rnd and len(agents) == n_agents:
            out[(q, rep)] = crowd_metric(
                [a["agent_log10_after"] for a in agents.values()], truth[q], metric=metric
            )
    return out


def arm_family(idx: dict, cond: str) -> str | None:
    fams = {family_of(rec["model_slug"]) for cells in idx.get(cond, {}).values() for rec in cells.values()}
    if not fams:
        return None
    if len(fams) != 1:
        raise RuntimeError(f"Single-model arm {cond} contains several model families: {fams}")
    return fams.pop()


def virtual_panel_errors(
    idx: dict,
    template_cond: str,
    source_by_family: dict[str, str],
    truth: dict[str, float],
    rnd: int,
    n_agents: int,
    metric: str = "squared_log_error",
) -> tuple[dict[tuple, float], list[str]]:
    """Seat-matched mix: each template seat's agent_id taken from its family's single-model arm."""
    problems: list[str] = []
    for fam, cond in source_by_family.items():
        actual = arm_family(idx, cond)
        if actual is not None and actual != fam:
            problems.append(f"{cond} expected family {fam}, found {actual}")
    out: dict[tuple, float] = {}
    for (q, rep, r), seats in idx.get(template_cond, {}).items():
        if r != 1 or len(seats) != n_agents:
            continue
        if len(idx.get(template_cond, {}).get((q, rep, rnd), {})) != n_agents:
            problems.append(f"incomplete template {template_cond}/{q}/rep{rep}/round{rnd}")
            continue
        logs = []
        for agent_id, rec in seats.items():
            src = source_by_family.get(family_of(rec["model_slug"]))
            src_rec = idx.get(src, {}).get((q, rep, rnd), {}).get(agent_id) if src else None
            prior_rec = idx.get(src, {}).get((q, rep, 1), {}).get(agent_id) if src else None
            if src_rec is None or prior_rec is None:
                miss_r = rnd if src_rec is None else 1
                problems.append(f"missing {src}/{q}/rep{rep}/round{miss_r}/{agent_id}")
                break
            prior = prior_rec["agent_log10_before"]
            if abs(prior - rec["agent_log10_before"]) > 1e-9:
                problems.append(
                    f"prior mismatch {template_cond} vs {src} at {q}/rep{rep}/{agent_id}"
                )
                break
            logs.append(src_rec["agent_log10_after"])
        else:
            out[(q, rep)] = crowd_metric(logs, truth[q], metric=metric)
    return out, problems


def collapse_replicates(errors: dict[tuple, float]) -> dict[str, float]:
    """Task is the unit of analysis: average replicates within task first."""
    by_q: dict[str, list[float]] = defaultdict(list)
    for (q, _rep), v in errors.items():
        by_q[q].append(v)
    return {q: sum(v) / len(v) for q, v in by_q.items()}


# ----------------------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------------------
def exact_wilcoxon_p(diffs: list[float]) -> tuple[float, int]:
    """Two-sided exact signed-rank p with average ranks for ties; zeros dropped."""
    nz = [d for d in diffs if abs(d) > 1e-15]
    n = len(nz)
    if n == 0:
        return 1.0, 0
    order = sorted(range(n), key=lambda i: abs(nz[i]))
    ranks2 = [0] * n  # doubled ranks, so tied averages stay integers
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(abs(nz[order[j + 1]]) - abs(nz[order[i]])) <= 1e-15:
            j += 1
        for k in range(i, j + 1):
            ranks2[order[k]] = i + j + 2  # 2 * average rank of positions i..j (1-based)
        i = j + 1
    total = sum(ranks2)
    counts = {0: 1}
    for r in ranks2:
        nxt = defaultdict(int)
        for s, c in counts.items():
            nxt[s] += c
            nxt[s + r] += c
        counts = nxt
    obs = sum(r for r, d in zip(ranks2, nz) if d > 0)
    dev = abs(2 * obs - total)
    extreme = sum(c for s, c in counts.items() if abs(2 * s - total) >= dev)
    return min(1.0, extreme / 2**n), n


def bootstrap_ci(diffs: list[float], level: float, seed: int = SEED) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(rng.choice(diffs) for _ in range(n)) / n for _ in range(N_BOOT))
    lo = means[int(math.floor((1 - level) / 2 * N_BOOT))]
    hi = means[min(N_BOOT - 1, int(math.ceil((1 + level) / 2 * N_BOOT)) - 1)]
    return lo, hi


def holm(ps: list[float]) -> list[float]:
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj = [0.0] * len(ps)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def contrast(name: str, a: str, b: str, err: dict[str, dict[str, float]]) -> dict[str, Any]:
    ea, eb = err.get(a, {}), err.get(b, {})
    units = sorted(set(ea) & set(eb))
    if not units:
        return {"name": name, "a": a, "b": b, "n_tasks": 0, "status": "missing data"}
    diffs = [eb[u] - ea[u] for u in units]
    mean_a = statistics.fmean(ea[u] for u in units)
    p, n_nz = exact_wilcoxon_p(diffs)
    lo95, hi95 = bootstrap_ci(diffs, 0.95)
    lo90, hi90 = bootstrap_ci(diffs, 0.90)
    md = statistics.fmean(diffs)
    return {
        "name": name,
        "a": a,
        "b": b,
        "n_tasks": len(units),
        "n_nonzero": n_nz,
        "mean_a": round(mean_a, 6),
        "mean_b": round(statistics.fmean(eb[u] for u in units), 6),
        "mean_diff_b_minus_a": round(md, 6),
        "relative_change": round(md / mean_a, 4) if mean_a else None,
        "ci95": [round(lo95, 6), round(hi95, 6)],
        "ci90": [round(lo90, 6), round(hi90, 6)],
        "b_better_worse_tied": [
            sum(d < -1e-15 for d in diffs),
            sum(d > 1e-15 for d in diffs),
            sum(abs(d) <= 1e-15 for d in diffs),
        ],
        "p_exact": round(p, 6),
    }


def apply_holm(rows: list[dict]) -> None:
    live = [r for r in rows if r.get("n_tasks")]
    for r, adj in zip(live, holm([r["p_exact"] for r in live])):
        r["p_holm"] = round(adj, 6)


def equivalent(row: dict, delta: float) -> bool | None:
    if not row.get("n_tasks"):
        return None
    lo, hi = row["ci90"]
    return -delta < lo and hi < delta


# ----------------------------------------------------------------------------------------
# Pre-registered interpretation rules
# ----------------------------------------------------------------------------------------
def interpret(conf: dict[str, dict], sec: dict[str, dict], desc: dict[str, dict]) -> dict[str, str]:
    def sig(r: dict) -> bool:
        return bool(r.get("n_tasks")) and r.get("p_holm", 1.0) < ALPHA

    out: dict[str, str] = {}
    herd = [k for k in ("herding_ds", "herding_qwen", "herding_glm") if sig(conf[k]) and conf[k]["mean_diff_b_minus_a"] > 0]
    out["herding_generalises"] = (
        f"Telemetry significantly worsens crowd error in {len(herd)}/3 additional models "
        f"({', '.join(herd) or 'none'}), plus Gemma from P0-P4."
    )

    cr = conf["critique_vs_reflect"]
    if sig(cr) and cr["mean_diff_b_minus_a"] < 0:
        out["critique_mechanism"] = "Two-sided critique beats length-matched generic reflection: 'dialectical' is supported."
    elif sig(cr) and cr["mean_diff_b_minus_a"] > 0:
        out["critique_mechanism"] = "Generic reflection beats the two-sided critique."
    elif equivalent(cr, DELTA_REFLECT):
        out["critique_mechanism"] = (
            f"Generic reflection is equivalent to the critique within +/-{DELTA_REFLECT:.5f}: "
            "drop 'dialectical'; the effect is structured pre-commitment reasoning."
        )
    else:
        out["critique_mechanism"] = "Inconclusive: neither a difference nor equivalence is shown. Report the CI."

    vs = conf["p3_vs_virtual_social"]
    if sig(vs) and vs["mean_diff_b_minus_a"] < 0:
        out["cross_model_social_benefit"] = "A heterogeneous median herds less than a homogeneous one: genuine cross-architecture social benefit."
    elif sig(vs) and vs["mean_diff_b_minus_a"] > 0:
        out["cross_model_social_benefit"] = "Mixing models under shared telemetry is worse than a static mix of single-model panels."
    else:
        out["cross_model_social_benefit"] = "No evidence that mixing adds anything beyond static averaging of model biases."

    sp = [conf[k] for k in ("p3_vs_silent_poly", "p4_vs_silent_poly")]
    out["social_vs_silent_mixed_panel"] = "; ".join(
        f"{r['b']} vs P0_POLY: {'better' if r['mean_diff_b_minus_a'] < 0 else 'worse'}"
        f" ({'significant' if sig(r) else 'n.s.'}, p_holm={r.get('p_holm')})"
        for r in sp if r.get("n_tasks")
    )

    sanity = desc["silent_poly_vs_virtual_silent"]
    eq = equivalent(sanity, DELTA_SANITY)
    out["harness_sanity"] = (
        "PASS: silent mixed panel matches its seat-matched virtual panel."
        if eq
        else "CHECK: P0_POLY does not demonstrably match VIRTUAL_SILENT; inspect routing/backends before trusting P3/P4 comparisons."
        if eq is False
        else "Not computable (missing arms)."
    )
    best = sec.get("p3_vs_best_single_model_social", {})
    if best.get("n_tasks"):
        out["practical"] = (
            f"P3 vs the best single model's social arm: diff {best['mean_diff_b_minus_a']:+.6f}, p_holm={best.get('p_holm')}."
        )
    return out


# ----------------------------------------------------------------------------------------
def _evaluate_metric(
    idx: dict,
    truth: dict[str, float],
    rnd: int,
    n_agents: int,
    conds: list[str],
    metric: str,
) -> tuple[dict[str, Any], list[str], list[str], dict[str, dict[tuple, float]]]:
    per_cell = {
        c: condition_errors(idx, c, truth, rnd, n_agents, metric=metric) for c in conds
    }

    v_social, prob_social = virtual_panel_errors(
        idx, "P3", K1_SOCIAL, truth, rnd, n_agents, metric=metric
    )
    v_silent, prob_silent = virtual_panel_errors(
        idx, "P0_POLY", K1_SILENT, truth, rnd, n_agents, metric=metric
    )
    per_cell[VIRTUAL_SOCIAL] = v_social
    per_cell[VIRTUAL_SILENT] = v_silent

    err = {c: collapse_replicates(v) for c, v in per_cell.items()}

    # Jensen bound: mean of the four single-model social crowd errors (the original POOLED).
    social = [err.get(c, {}) for c in K1_SOCIAL.values()]
    common = set.intersection(*(set(m) for m in social)) if all(social) else set()
    err[POOLED] = {q: statistics.fmean(m[q] for m in social) for q in common}

    # Best single model chosen on the SILENT arms (independent of the social data it is compared to).
    silent_means = {
        fam: statistics.fmean(err[c].values()) for fam, c in K1_SILENT.items() if err.get(c)
    }
    best_family = min(silent_means, key=silent_means.get) if len(silent_means) == 4 else None
    if best_family:
        err[BEST_K1_SOCIAL] = err.get(K1_SOCIAL[best_family], {})

    conf = [contrast(*c, err) for c in CONFIRMATORY]
    sec = [contrast(*c, err) for c in SECONDARY]
    desc = [contrast(*c, err) for c in DESCRIPTIVE]
    repl = [contrast(*c, err) for c in REPLICATION_P_SERIES]
    apply_holm(conf)
    apply_holm(sec)
    apply_holm(repl)

    summary = {
        "n_tasks_per_condition": {c: len(v) for c, v in err.items()},
        "mean_by_condition": {
            c: round(statistics.fmean(v.values()), 6) for c, v in err.items() if v
        },
        "best_single_model_by_silent_arm": best_family,
        "silent_arm_means": {k: round(v, 6) for k, v in silent_means.items()},
        "confirmatory_holm_m8": conf,
        "secondary_holm_m2": sec,
        "descriptive": desc,
        "replication_p_series_holm_m4": repl,
    }
    return summary, prob_social, prob_silent, per_cell


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path, help="numeric_runs.jsonl files (baseline + controls)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--n-agents", type=int, default=8)
    args = ap.parse_args(argv)

    records, specs = load_records(args.runs)
    if not records:
        raise RuntimeError("No records found in input runs.")
    audit = audit_specs(specs)
    truth = truths_from_specs(specs)
    rnd = final_round(records)
    idx = index_by_condition(records)

    conds = sorted(idx)
    primary, prob_social, prob_silent, primary_cells = _evaluate_metric(
        idx, truth, rnd, args.n_agents, conds, metric="squared_log_error"
    )
    secondary_abs, _, _, _ = _evaluate_metric(
        idx, truth, rnd, args.n_agents, conds, metric="abs_log_error"
    )

    tokens: dict[str, float | None] = {}
    for c in conds:
        outs = [r.get("output_tokens", 0) or 0 for cells in idx[c].values() for r in cells.values()]
        tokens[c] = round(statistics.fmean(outs), 1) if outs else None

    replicate_counts = {
        c: {
            "tasks": primary["n_tasks_per_condition"].get(c, 0),
            "complete_task_replicates": len(primary_cells.get(c, {})),
        }
        for c in [*conds, VIRTUAL_SOCIAL, VIRTUAL_SILENT]
    }

    conf = primary["confirmatory_holm_m8"]
    sec = primary["secondary_holm_m2"]
    desc = primary["descriptive"]

    result = {
        "preregistered_constants": {
            "DELTA_REFLECT": DELTA_REFLECT,
            "DELTA_SANITY": DELTA_SANITY,
            "alpha": ALPHA,
            "n_boot": N_BOOT,
            "seed": SEED,
        },
        "spec_audit": audit,
        "final_round": rnd,
        "conditions_found": conds,
        "n_tasks_per_condition": primary["n_tasks_per_condition"],
        "replicate_counts": replicate_counts,
        "mean_crowd_error": primary["mean_by_condition"],
        "best_single_model_by_silent_arm": primary["best_single_model_by_silent_arm"],
        "silent_arm_means": primary["silent_arm_means"],
        "virtual_panel_problems": {"social": prob_social[:20], "silent": prob_silent[:20]},
        "mean_output_tokens": tokens,
        "confirmatory_holm_m8": conf,
        "secondary_holm_m2": sec,
        "descriptive": desc,
        "replication_p_series_holm_m4": primary["replication_p_series_holm_m4"],
        "interpretation": interpret(
            {r["name"]: r for r in conf}, {r["name"]: r for r in sec}, {r["name"]: r for r in desc}
        ),
        "secondary_metric_abs_log_error": {
            "metric_definition": "|xbar_2 - y*|",
            **secondary_abs,
        },
    }
    text = json.dumps(result, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    if audit["differing_keys"]:
        print("\nWARNING: run specs differ beyond conditions/models; see spec_audit.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
