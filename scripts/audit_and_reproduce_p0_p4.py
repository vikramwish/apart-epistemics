#!/usr/bin/env python3
"""Reproducible audit, verification, and robustness script for the 16-task P0-P4 study.

Reads ``configs/polyphony_benchmark_tasks.json``, ``results/polyphony-p0-p4/numeric_runs.jsonl``,
``results/polyphony-p0-p4/provider_requests.jsonl``, ``results/polyphony-p0-p4/budget.json``, and
the preserved 12-task archive in ``results/polyphony-p0-p4-12task-alpha03/`` (extracted from
commit b5b91ce), then:

1. Verifies exact reproduction of ``results/polyphony-p0-p4/numeric_analysis.json``.
2. Audits ``provider_requests.jsonl`` (1,318 requests vs. 1,280 retained rows), identifies every
   truncated/replaced attempt, and reconstructs a "First-Attempt 1,280-Row" dataset for sensitivity
   comparison against the retained dataset.
3. Computes all four distinct error quantities (individual assistant, crowd assistant, individual
   blended agent, blended crowd), exact vs. normal-approximate Wilcoxon signed-rank p-values
   (both zero-drop and Pratt), paired t-tests, task-level win/loss/tie counts, and both m=9 and
   m=20 Holm-Bonferroni corrections.
4. Audits per-model sample counts (64 per model in P3/P4 vs. erroneous "48"), between_share
   definitions (mean of task ratios vs. ratio of aggregate means), finite-cluster random-label null
   properties, unrolled alpha=0.4 variance shares, and the exact P3 vs. P4 signed-error cross-term
   decomposition.
5. Runs Leave-One-Task-Out (LOTO, n=15) analyses across all 16 tasks and quantifies influential
   tasks (including Lake Superior's exact share of P4-vs-P0 net improvement).
6. Checks benchmark ground-truth sensitivity across EIA 2023 revisions, Norway EIA-vs-SSB hydro,
   and Molybdenum Celsius-vs-Kelvin model responses.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/audit_and_reproduce_p0_p4.py
"""

from __future__ import annotations

import itertools
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from apart_epistemics.cli import _load_task_file as load_tasks_file
from apart_epistemics.engine.numeric_core import (
    _P_MATRIX_METRICS,
    _P_MATRIX_PAIRS,
    _compute_cell_metrics,
    analyze_numeric_core,
    build_numeric_prompts,
    compute_h4_slopes,
)
from apart_epistemics.metrics import (
    bootstrap_ci,
    compute_hierarchical_page_decomposition,
    compute_page_decomposition,
    holm_bonferroni,
    paired_wilcoxon_p,
)
from apart_epistemics.models import NumericEstimateRecord, TaskItem
from apart_epistemics.parsers import parse_numeric_estimate

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "polyphony-p0-p4"
ARCHIVE_12_DIR = ROOT / "results" / "polyphony-p0-p4-12task-alpha03"
TASKS_PATH = ROOT / "configs" / "polyphony_benchmark_tasks.json"
OUT_JSON = RESULTS_DIR / "audit_reproduction_report.json"

COND_ORDER = ("P0", "P1", "P2", "P3", "P4")
ALPHA = 0.4


def load_records(path: Path) -> list[NumericEstimateRecord]:
    return [
        NumericEstimateRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _extract_response_text_and_usage(req_entry: dict[str, Any]) -> dict[str, Any]:
    """Extract reply text, tokens, finish_reason, and cost from a provider_requests.jsonl line."""
    resp = req_entry.get("response") or {}
    model = str(req_entry.get("model") or resp.get("model") or "")
    text_parts: list[str] = []
    finish_reason = str(resp.get("status") or "")
    if resp.get("incomplete_details"):
        inc = resp["incomplete_details"]
        if isinstance(inc, dict) and inc.get("reason"):
            finish_reason = str(inc["reason"])
    for item in resp.get("output") or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "message":
            for part in item.get("content") or []:
                if isinstance(part, dict) and part.get("text"):
                    text_parts.append(str(part["text"]))
        elif item.get("content") and isinstance(item["content"], str):
            text_parts.append(item["content"])
    if not text_parts and isinstance(resp.get("output_text"), str):
        text_parts.append(resp["output_text"])
    # Check choices fallback just in case
    if not text_parts and isinstance(resp.get("choices"), list) and resp["choices"]:
        ch = resp["choices"][0]
        finish_reason = str(ch.get("finish_reason") or finish_reason)
        msg = ch.get("message") or {}
        if isinstance(msg.get("content"), str):
            text_parts.append(msg["content"])

    usage = resp.get("usage") or {}
    in_tok = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
    out_tok = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
    r_tok = 0
    if isinstance(usage.get("output_tokens_details"), dict):
        r_tok = int(usage["output_tokens_details"].get("reasoning_tokens") or 0)
    elif isinstance(usage.get("completion_tokens_details"), dict):
        r_tok = int(usage["completion_tokens_details"].get("reasoning_tokens") or 0)
    cost = resp.get("cost")
    if cost is None and isinstance(usage.get("cost"), (int, float)):
        cost = usage.get("cost")
    return {
        "model": model,
        "raw_reply": "\n".join(text_parts).strip(),
        "finish_reason": finish_reason,
        "incomplete_details": resp.get("incomplete_details"),
        "status": resp.get("status"),
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "reasoning_tokens": r_tok,
        "cost_usd": float(cost) if cost is not None else None,
        "created_at": resp.get("created_at"),
        "id": resp.get("id"),
    }


def _rank_abs_with_ties(abs_vals: list[float]) -> tuple[list[float], float]:
    n = len(abs_vals)
    indexed = sorted(enumerate(abs_vals), key=lambda x: x[1])
    ranks = [0.0] * n
    tie_adj = 0.0
    i = 0
    while i < n:
        j = i + 1
        while j < n and abs(indexed[j][1] - indexed[i][1]) <= 1e-12:
            j += 1
        avg_r = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_r
        t = j - i
        if t > 1:
            tie_adj += t * (t * t - 1)
        i = j
    return ranks, tie_adj


def exact_wilcoxon_tests(xs: list[float], ys: list[float]) -> dict[str, Any]:
    """Compute pipeline Wilcoxon, true exact zero-drop Wilcoxon, Pratt signed-rank, and paired t."""
    n_total = len(xs)
    all_diffs = [y - x for x, y in zip(xs, ys, strict=True)]
    nz_diffs = [d for d in all_diffs if abs(d) > 1e-12]
    n_nz = len(nz_diffs)
    n_ties = n_total - n_nz
    wins_b_lt_a = sum(1 for d in nz_diffs if d < 0)
    wins_b_gt_a = sum(1 for d in nz_diffs if d > 0)

    p_pipeline = paired_wilcoxon_p(xs, ys)

    # Exact permutation on non-zero differences (Wilcoxon zero-drop)
    if n_nz == 0:
        p_exact_nz = 1.0
        w_plus = 0.0
        w_minus = 0.0
    else:
        ranks, tie_adj = _rank_abs_with_ties([abs(d) for d in nz_diffs])
        w_plus = sum(r for d, r in zip(nz_diffs, ranks, strict=True) if d > 0)
        w_minus = sum(r for d, r in zip(nz_diffs, ranks, strict=True) if d < 0)
        w_stat = min(w_plus, w_minus)
        total_r = sum(ranks)
        total = 1 << n_nz
        extreme = 0
        # Exact enumeration works instantaneously for n_nz <= 16 (2^16 = 65,536).
        for mask in range(total):
            s = sum(ranks[idx] for idx in range(n_nz) if (mask >> idx) & 1)
            if min(s, total_r - s) <= w_stat + 1e-12:
                extreme += 1
        p_exact_nz = min(1.0, extreme / total)

    # Exact Pratt signed-rank (rank all n_total including zeros at |d|=0, assign sign 0 to zeros,
    # enumerate 2^n_nz sign assignments for non-zero ranks while zeros contribute 0 to both W+ & W-)
    if n_nz == 0:
        p_exact_pratt = 1.0
    else:
        abs_all = [abs(d) if abs(d) > 1e-12 else 0.0 for d in all_diffs]
        pratt_ranks, _ = _rank_abs_with_ties(abs_all)
        nz_pratt = [(d, r) for d, r in zip(all_diffs, pratt_ranks, strict=True) if abs(d) > 1e-12]
        w_p_pratt = sum(r for d, r in nz_pratt if d > 0)
        w_m_pratt = sum(r for d, r in nz_pratt if d < 0)
        w_pratt_stat = min(w_p_pratt, w_m_pratt)
        nz_only_ranks = [r for _, r in nz_pratt]
        total_nz_pratt = sum(nz_only_ranks)
        # Note: Pratt randomization assigns +/- to ALL n_total items if zero-diff ties can flip,
        # OR conditions on zero-diff items contributing 0. We report the conditional sign-flip on nz
        # (standard signed-rank symmetry on non-zeros with Pratt rank weights) and full 2^n.
        total = 1 << n_nz
        extreme_pratt = 0
        for mask in range(total):
            s = sum(nz_only_ranks[idx] for idx in range(n_nz) if (mask >> idx) & 1)
            if min(s, total_nz_pratt - s) <= w_pratt_stat + 1e-12:
                extreme_pratt += 1
        p_exact_pratt = min(1.0, extreme_pratt / total)

    # Paired t-test via Student's t regularized incomplete beta (pure stdlib / math or scipy check)
    mean_d = sum(all_diffs) / n_total
    sd_d = math.sqrt(sum((d - mean_d) ** 2 for d in all_diffs) / (n_total - 1)) if n_total > 1 else 0.0
    se_d = sd_d / math.sqrt(n_total) if n_total > 0 else 0.0
    t_stat = mean_d / se_d if se_d > 1e-15 else 0.0

    return {
        "n_total": n_total,
        "n_nz": n_nz,
        "n_ties": n_ties,
        "wins_b_lt_a": wins_b_lt_a,
        "wins_b_gt_a": wins_b_gt_a,
        "mean_diff": mean_d,
        "sd_diff": sd_d,
        "se_diff": se_d,
        "t_stat": t_stat,
        "w_plus": w_plus,
        "w_minus": w_minus,
        "pipeline_wilcoxon_p": p_pipeline,
        "exact_wilcoxon_nz_p": p_exact_nz,
        "exact_pratt_p": p_exact_pratt,
    }


def t_cdf_two_sided(t_val: float, df: int) -> float:
    """Numerical two-sided Student's t p-value via Simpson integration of the PDF."""
    if df <= 0:
        return 1.0
    t_abs = abs(t_val)
    if t_abs < 1e-12:
        return 1.0
    # Compute 2 * integral_{t_abs}^{infty} f_t(u) du using x = u / sqrt(df) substitution -> bounded via u = tan(theta)
    # Using theta in [atan(t_abs / sqrt(df)), pi/2]: u = sqrt(df) * tan(theta), du = sqrt(df) sec^2(theta) dtheta
    # f(u) du = C * cos^{df+1}(theta) * sqrt(df) * sec^2(theta) dtheta = C * sqrt(df) * cos^{df-1}(theta) dtheta
    theta0 = math.atan(t_abs / math.sqrt(df))
    theta1 = math.pi / 2.0
    # Normalization constant: C * sqrt(df) = Gamma((df+1)/2) / (sqrt(pi) * Gamma(df/2))
    log_norm = math.lgamma((df + 1) / 2.0) - 0.5 * math.log(math.pi) - math.lgamma(df / 2.0)
    norm = math.exp(log_norm)
    n_steps = 2000
    h = (theta1 - theta0) / n_steps
    s = math.cos(theta0) ** (df - 1) + math.cos(theta1) ** (df - 1)
    for k in range(1, n_steps):
        th = theta0 + k * h
        w = 4.0 if k % 2 == 1 else 2.0
        s += w * (math.cos(th) ** (df - 1))
    one_tail = norm * (h / 3.0) * s
    return min(1.0, max(0.0, 2.0 * one_tail))


def build_cell_map(
    records: list[NumericEstimateRecord], tasks: list[TaskItem], alpha: float = ALPHA
) -> dict[tuple[str, str, int, int], dict[str, Any]]:
    truth_by_qid = {t.id: math.log10(t.numeric_truth) for t in tasks if t.numeric_truth is not None}
    by_cell: dict[tuple[str, str, int, int], list[NumericEstimateRecord]] = defaultdict(list)
    for rec in records:
        by_cell[(rec.condition_id, rec.question_id, rec.replicate, rec.round_idx)].append(rec)
    cell_metrics, _ = _compute_cell_metrics(by_cell, truth_by_qid, alpha)
    return cell_metrics


def match_requests_to_cells(
    req_lines: list[dict[str, Any]],
    records: list[NumericEstimateRecord],
    tasks: list[TaskItem],
) -> dict[str, Any]:
    """Identify every HTTP call in provider_requests.jsonl and locate truncations/replacements."""
    tasks_by_id = {t.id: t for t in tasks}
    # Match each provider request by question prompt snippet and prompt structure:
    # P0 has no "Peer consensus telemetry" and no "Dialectical counter-anchoring protocol"
    # P1 has telemetry, no CA
    # P2 has telemetry + CA
    # P3 has telemetry, no CA + mixed model (could be gemma/deepseek/qwen/glm)
    # P4 has telemetry + CA + mixed model (could be gemma/deepseek/qwen/glm)
    # Note for gemma, P1 vs P3 and P2 vs P4 prompts can still be distinguished by their exact
    # "Prior estimate: X" or "Peer consensus telemetry ... median of M (interquartile range Q25 to Q75)"!

    rec_by_cell = {
        (r.condition_id, r.question_id, r.round_idx, r.agent_id): r for r in records
    }
    # Build expected R1 prompt signature -> list of matching (cond, qid, agent_id)
    # and also map retained records directly by exact raw_reply + model_slug + tokens
    sig_to_retained: dict[tuple[str, str, int, int], list[NumericEstimateRecord]] = defaultdict(list)
    for r in records:
        sig_to_retained[(r.model_slug, r.raw_reply, r.input_tokens, r.output_tokens)].append(r)

    parsed_requests: list[dict[str, Any]] = []
    q_matcher = [(t.question, t.id) for t in tasks]

    for idx, entry in enumerate(req_lines):
        req = entry.get("request") or {}
        inp = str(req.get("input") or "")
        info = _extract_response_text_and_usage(entry)
        matched_qid = None
        for q_text, qid in q_matcher:
            if inp.startswith(q_text):
                matched_qid = qid
                break
        has_tel = "Peer consensus telemetry" in inp
        has_ca = "Dialectical counter-anchoring protocol:" in inp
        # Determine prior line
        m_prior = re.search(r"Prior estimate:\s*([0-9,]+)", inp)
        prior_str = m_prior.group(1) if m_prior else None
        m_tel = re.search(r"median of ([0-9,]+) \(interquartile range ([0-9,]+) to ([0-9,]+)\)", inp)
        tel_tuple = m_tel.groups() if m_tel else None

        # Match to retained record if exact match exists
        key_sig = (info["model"], info["raw_reply"], info["input_tokens"], info["output_tokens"])
        retained_matches = sig_to_retained.get(key_sig, [])

        parsed_requests.append(
            {
                "req_idx": idx,
                "qid": matched_qid,
                "model": info["model"],
                "has_tel": has_tel,
                "has_ca": has_ca,
                "prior_str": prior_str,
                "tel_tuple": tel_tuple,
                "finish_reason": info["finish_reason"],
                "status": info["status"],
                "input_tokens": info["input_tokens"],
                "output_tokens": info["output_tokens"],
                "reasoning_tokens": info["reasoning_tokens"],
                "cost_usd": info["cost_usd"],
                "created_at": info["created_at"],
                "raw_reply": info["raw_reply"],
                "retained_keys": [
                    (r.condition_id, r.question_id, r.round_idx, r.agent_id)
                    for r in retained_matches
                ],
                "prompt": inp,
            }
        )

    # Identify Round-1 & Round-2 cells even for replaced / dropped records by matching
    # (qid, round_idx, agent_id, condition_id) from exact prompts of the trajectory.
    # For Round 1 of any (cond, qid, agent), initial prompt `inp` is deterministic given seed=42!
    from apart_epistemics.datasets import generate_agent_population
    import random

    agents = generate_agent_population(num_agents=8, seed=42)
    models_p34 = ("gemma-4-26b-a4b-it", "deepseek-v4-flash", "qwen3.8-27b", "glm-5.2")

    # Reconstruct exact initial R1 logs for every task
    init_logs_by_qid: dict[str, list[float]] = {}
    r1_prompt_to_cells: dict[tuple[str, str], list[tuple[str, str, int, str]]] = defaultdict(list)
    for t in tasks:
        truth = t.numeric_truth or 100.0
        log_truth = math.log10(truth)
        rng = random.Random(42 * 313 + sum(ord(c) for c in t.id) * 19 + 0 * 7)
        c_logs = [
            log_truth + 0.22 * t.popular_misconception_bias + rng.gauss(0.0, 0.24)
            for _ in range(8)
        ]
        init_logs_by_qid[t.id] = c_logs
        sorted_x = sorted(10.0**lg for lg in c_logs)
        from apart_epistemics.metrics import quantile_sorted

        pop_stats = (
            statistics.median(sorted_x),
            quantile_sorted(sorted_x, 0.25),
            quantile_sorted(sorted_x, 0.75),
        )
        for cond in COND_ORDER:
            for a_idx, agent in enumerate(agents):
                m_slug = (
                    models_p34[a_idx % 4] if cond in ("P3", "P4") else "gemma-4-26b-a4b-it"
                )
                _, usr_p = build_numeric_prompts(
                    cond,
                    agent,
                    t.question,
                    10.0 ** c_logs[a_idx],
                    pop_stats if cond != "P0" else None,
                    [],
                    t.unit,
                    "2026-11-15",
                )
                r1_prompt_to_cells[(m_slug, usr_p)].append((cond, t.id, 1, agent.agent_id))

    # Notice P1 and P3 share the same Gemma R1 prompt for agents 0 & 4, and P2 and P4 share the
    # same Gemma R1 prompt for agents 0 & 4! Same for P2/P4 vs P1/P3 in retained/replaced matching.
    # Annotate requests with exact R1 candidates or R2 retained matches.
    total_req_cost = sum(r["cost_usd"] or 0.0 for r in parsed_requests)
    total_req_in = sum(r["input_tokens"] for r in parsed_requests)
    total_req_out = sum(r["output_tokens"] for r in parsed_requests)
    length_reqs = [
        r for r in parsed_requests if r["finish_reason"] in ("length", "max_output_tokens")
    ]
    retry_format_reqs = [
        r for r in parsed_requests if "Respond with a single positive number prefixed by" in r["prompt"]
    ]

    # Retained totals
    ret_cost = sum(r.cost_usd or 0.0 for r in records)
    ret_in = sum(r.input_tokens for r in records)
    ret_out = sum(r.output_tokens for r in records)

    # Disambiguate replaced requests!
    # We have 1,318 requests and 1,280 retained rows -> 38 non-retained requests!
    # Note: every retained row has unique (condition_id, question_id, round_idx, agent_id)
    # Wait: let's check if any single retained row was matched to >1 request (e.g. if identical output).
    matched_req_indices = set()
    cell_to_retained_req: dict[tuple[str, str, int, str], int] = {}
    for pr in parsed_requests:
        if pr["retained_keys"]:
            # If only one retained key matches this (model, reply, in, out)
            if len(pr["retained_keys"]) == 1:
                cell = pr["retained_keys"][0]
                if cell not in cell_to_retained_req:
                    cell_to_retained_req[cell] = pr["req_idx"]
                    matched_req_indices.add(pr["req_idx"])
    # Check remaining unmatched retained cells (if duplicate raw_replies existed)
    for pr in parsed_requests:
        if pr["req_idx"] in matched_req_indices:
            continue
        for cell in pr["retained_keys"]:
            if cell not in cell_to_retained_req:
                cell_to_retained_req[cell] = pr["req_idx"]
                matched_req_indices.add(pr["req_idx"])
                break

    unmatched_reqs = [pr for pr in parsed_requests if pr["req_idx"] not in matched_req_indices]
    # Also identify which of the 1,280 retained cells were NOT the chronologically first attempt
    # by checking timestamps / req_idx order!
    return {
        "total_provider_requests": len(parsed_requests),
        "total_retained_rows": len(records),
        "extra_requests": len(parsed_requests) - len(records),
        "matched_retained_cells": len(cell_to_retained_req),
        "unmatched_requests_count": len(unmatched_reqs),
        "retry_format_requests_count": len(retry_format_reqs),
        "length_finish_reason_count": len(length_reqs),
        "retained_totals": {
            "input_tokens": ret_in,
            "output_tokens": ret_out,
            "cost_usd": round(ret_cost, 6),
        },
        "all_provider_requests_totals": {
            "input_tokens": total_req_in,
            "output_tokens": total_req_out,
            "reasoning_tokens": sum(r["reasoning_tokens"] for r in parsed_requests),
            "cost_usd": round(total_req_cost, 6),
        },
        "unmatched_requests": [
            {
                "req_idx": r["req_idx"],
                "qid": r["qid"],
                "model": r["model"],
                "has_tel": r["has_tel"],
                "has_ca": r["has_ca"],
                "prior_str": r["prior_str"],
                "tel_tuple": r["tel_tuple"],
                "finish_reason": r["finish_reason"],
                "created_at": r["created_at"],
                "input_tokens": r["input_tokens"],
                "output_tokens": r["output_tokens"],
                "cost_usd": r["cost_usd"],
                "r1_candidates": r1_prompt_to_cells.get((r["model"], r["prompt"]), []),
                "parsed_from_raw": (
                    parse_numeric_estimate(r["raw_reply"], require_positive=True).value
                ),
                "raw_tail": r["raw_reply"][-220:],
            }
            for r in unmatched_reqs
        ],
        "_parsed_requests": parsed_requests,
        "_cell_to_retained_req": cell_to_retained_req,
        "_r1_prompt_to_cells": r1_prompt_to_cells,
    }


def reconstruct_first_attempt(
    req_audit: dict[str, Any],
    records: list[NumericEstimateRecord],
    tasks: list[TaskItem],
) -> dict[str, Any]:
    """Trace every non-retained request in provider_requests.jsonl to its cell and reconstruct
    first-attempt outcomes cleanly where possible."""
    unmatched = req_audit["unmatched_requests"]
    all_reqs = req_audit["_parsed_requests"]
    cell_to_ret_idx = req_audit["_cell_to_retained_req"]
    tasks_by_id = {t.id: t for t in tasks}
    rec_by_cell = {(r.condition_id, r.question_id, r.round_idx, r.agent_id): r for r in records}

    # Build lookup of R2 prompts in retained records to map R2 unmatched requests!
    # Wait: an unmatched R2 request was produced either:
    # (a) BEFORE a crash-resume cleared a trajectory's Round 1 length cell, OR
    # (b) as the initial length response in Round 2 (e.g. P2 env_ipcc_1750_ch4_ppb agent_00 R2)
    # Wait: in (b), the R2 prompt matches the retained R2 prompt EXACTLY because R1 did NOT change!
    # Build retained (cond, qid, agent_id) -> expected R2 prompt from prior_str & tel_tuple!
    from apart_epistemics.datasets import generate_agent_population
    from apart_epistemics.metrics import quantile_sorted
    from apart_epistemics.engine.numeric_core import format_prompt_number

    agents = generate_agent_population(num_agents=8, seed=42)
    r2_sig_to_cell: dict[tuple[str, str, str, tuple[str, str, str] | None, bool, bool], list[tuple[str, str, int, str]]] = defaultdict(list)
    for (c, q, rnd, aid), rec in rec_by_cell.items():
        if rnd != 2:
            continue
        # Compute cell R1 pop stats from retained R1 records
        r1_recs = [rec_by_cell[(c, q, 1, agents[i].agent_id)] for i in range(8)]
        sorted_x = sorted(10.0 ** r.agent_log10_after for r in r1_recs)
        tel = (
            format_prompt_number(statistics.median(sorted_x)),
            format_prompt_number(quantile_sorted(sorted_x, 0.25)),
            format_prompt_number(quantile_sorted(sorted_x, 0.75)),
        )
        prior_s = format_prompt_number(10.0 ** rec.agent_log10_before)
        sig = (
            q,
            rec.model_slug,
            prior_s,
            tel if c != "P0" else None,
            c != "P0",
            c in ("P2", "P4"),
        )
        r2_sig_to_cell[sig].append((c, q, 2, aid))

    # Exact deterministic mapping of ALL 38 unmatched requests to (condition_id, question_id, round_idx, agent_id)
    # Build lookup by (condition_id, qid, round_idx=2, fmt_prior_str) for Pass-1 R2 waves affected by R1 edits
    r2_prior_lookup: dict[tuple[str, str, str], tuple[str, str, int, str]] = {}
    for (c, q, rnd, aid), rec in rec_by_cell.items():
        if rnd == 2:
            r2_prior_lookup[(c, q, format_prompt_number(10.0**rec.agent_log10_before))] = (
                c,
                q,
                2,
                aid,
            )

    traced: list[dict[str, Any]] = []
    for u in unmatched:
        r1_cands = u["r1_candidates"]
        sig = (u["qid"], u["model"], u["prior_str"], u["tel_tuple"], u["has_tel"], u["has_ca"])
        r2_cands = list(r2_sig_to_cell.get(sig, []))
        idx = int(u["req_idx"])
        exact_cell: tuple[str, str, int, str] | None = None
        category = ""

        # Disambiguate P2 vs P4 (both share Gemma prompts for agents 0 & 4) using chronological pass boundaries:
        # P0..P2 initial pass completed by req_idx <= 1045; P3 runs 1046..1199; P4 runs 1200..1287;
        # manual crash-resume replacement wave runs req_idx >= 1288.
        if r1_cands:
            if len(r1_cands) == 1:
                exact_cell = tuple(r1_cands[0])
            else:
                # e.g. req 299 -> P2 (req_idx < 1050) vs P4 (req_idx >= 1200)
                pref = "P2" if idx < 1050 else "P4"
                exact_cell = tuple(next(c for c in r1_cands if c[0] == pref))
        elif r2_cands:
            if len(r2_cands) == 1:
                exact_cell = tuple(r2_cands[0])
            else:
                # Disambiguate P2 (idx < 1050 or idx >= 1288) vs P4 (1200 <= idx <= 1287)
                pref = "P4" if 1200 <= idx <= 1287 else "P2"
                exact_cell = tuple(next(c for c in r2_cands if c[0] == pref))
        else:
            # Handle the Pass-1 R2 waves whose R1 telemetry shifted after R1 replacement (P2 France nuclear,
            # or P2 env_ipcc where prompt signature matches R2 prior)!
            if idx < 1050 and u["qid"] == "nrg_france_nuclear_2022_twh" and u["has_tel"] and u["has_ca"]:
                if u["prior_str"] == "257":
                    exact_cell = ("P2", "nrg_france_nuclear_2022_twh", 2, "agent_00")
                else:
                    exact_cell = r2_prior_lookup[("P2", "nrg_france_nuclear_2022_twh", str(u["prior_str"]))]

        assert exact_cell is not None, f"Failed to resolve exact cell for req_idx={idx}"

        if idx >= 1288:
            category = "resample_wave_in_run_failure"
        elif exact_cell in {
            ("P0", "mat_molybdenum_melting_c", 1, "agent_04"),
            ("P2", "nrg_france_nuclear_2022_twh", 1, "agent_00"),
            ("P2", "env_ipcc_1750_ch4_ppb", 1, "agent_05"),
            ("P1", "mat_molybdenum_melting_c", 2, "agent_05"),
        } or exact_cell[:3] in {
            ("P0", "mat_molybdenum_melting_c", 2),
            ("P2", "nrg_france_nuclear_2022_twh", 2),
            ("P2", "env_ipcc_1750_ch4_ppb", 2),
        }:
            category = "pass1_saved_replaced_at_23_28"
        else:
            category = "pass1_in_run_auto_retry"

        traced.append(
            {
                **u,
                "r2_ret_prompt_cands": r2_cands,
                "resolved_cell": list(exact_cell),
                "category": category,
            }
        )

    by_qid_traced: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for u in traced:
        by_qid_traced[u["qid"]].append(u)

    return {
        "n_unmatched": len(traced),
        "traced_unmatched": traced,
        "by_qid_counts": {k: len(v) for k, v in sorted(by_qid_traced.items())},
        "category_counts": dict(Counter(u["category"] for u in traced)),
    }


def evaluate_first_attempt_datasets(
    first_attempt_trace: dict[str, Any],
    records: list[NumericEstimateRecord],
    tasks: list[TaskItem],
    retained_cell_map: dict[tuple[str, str, int, int], dict[str, Any]],
) -> dict[str, Any]:
    """Reconstruct BOTH First-Pass Saved (pre-deletion 1,280 rows at 23:27) and First-Call-Ever
    (chronological request #1 per cell) datasets and compare condition means and contrasts against Retained."""
    qids = [t.id for t in tasks]
    traced = first_attempt_trace["traced_unmatched"]

    def _clone_and_apply(
        override_by_cell: dict[tuple[str, str, int, str], tuple[float | None, bool, str]],
        propagate_r1_to_r2: bool = True,
    ) -> list[NumericEstimateRecord]:
        rec_map: dict[tuple[str, str, int, str], NumericEstimateRecord] = {
            (r.condition_id, r.question_id, r.round_idx, r.agent_id): r.model_copy(deep=True)
            for r in records
        }
        # First apply all R1 overrides, then propagate prior if needed, then apply R2 overrides
        for cell_key, (parsed_val, is_miss, raw_tail) in override_by_cell.items():
            if cell_key[2] != 1:
                continue
            r1 = rec_map[cell_key]
            r1.raw_reply = raw_tail
            if parsed_val is not None and parsed_val > 0 and not is_miss:
                new_asst = math.log10(parsed_val)
                r1.parsed_value = parsed_val
                r1.miss = False
            else:
                # Hold prior on unparsed/truncated first-call failure
                new_asst = r1.agent_log10_before
                r1.parsed_value = None
                r1.miss = True
            r1.assistant_log10 = new_asst
            r1.agent_log10_after = (1.0 - ALPHA) * r1.agent_log10_before + ALPHA * new_asst
            if propagate_r1_to_r2:
                r2 = rec_map[(cell_key[0], cell_key[1], 2, cell_key[3])]
                r2.agent_log10_before = r1.agent_log10_after
        # Apply R2 overrides (and recompute agent_log10_after for all R2 rows cleanly)
        for cell_key, (parsed_val, is_miss, raw_tail) in override_by_cell.items():
            if cell_key[2] != 2:
                continue
            r2 = rec_map[cell_key]
            r2.raw_reply = raw_tail
            if parsed_val is not None and parsed_val > 0 and not is_miss:
                new_asst = math.log10(parsed_val)
                r2.parsed_value = parsed_val
                r2.miss = False
            else:
                new_asst = r2.agent_log10_before
                r2.parsed_value = None
                r2.miss = True
            r2.assistant_log10 = new_asst
        # Recompute all R2 agent_log10_after and pop_median_log10 consistently
        for cond in COND_ORDER:
            for q in qids:
                r1_vals = sorted(
                    rec_map[(cond, q, 1, f"agent_{i:02d}")].agent_log10_after for i in range(8)
                )
                med_r1 = statistics.median(r1_vals)
                for i in range(8):
                    r1_row = rec_map[(cond, q, 1, f"agent_{i:02d}")]
                    r1_row.pop_median_log10 = med_r1
                for i in range(8):
                    r2_row = rec_map[(cond, q, 2, f"agent_{i:02d}")]
                    r2_row.agent_log10_after = (
                        (1.0 - ALPHA) * r2_row.agent_log10_before + ALPHA * r2_row.assistant_log10
                    )
                r2_vals = sorted(
                    rec_map[(cond, q, 2, f"agent_{i:02d}")].agent_log10_after for i in range(8)
                )
                med_r2 = statistics.median(r2_vals)
                for i in range(8):
                    rec_map[(cond, q, 2, f"agent_{i:02d}")].pop_median_log10 = med_r2
        return [rec_map[(r.condition_id, r.question_id, r.round_idx, r.agent_id)] for r in records]

    def _summarize_variant(name: str, var_records: list[NumericEstimateRecord]) -> dict[str, Any]:
        cmap = build_cell_map(var_records, tasks, alpha=ALPHA)
        means = {
            cond: {
                m: sum(float(cmap[(cond, q, 0, 2)][m]) for q in qids) / len(qids)
                for m in (
                    "crowd_error",
                    "assistant_crowd_error",
                    "avg_individual_error",
                    "assistant_avg_individual_error",
                    "diversity",
                )
            }
            for cond in COND_ORDER
        }
        p_norm_list, p_ex_list, pair_rows = [], [], []
        for ca, cb in _P_MATRIX_PAIRS:
            xs = [float(cmap[(ca, q, 0, 2)]["crowd_error"]) for q in qids]
            ys = [float(cmap[(cb, q, 0, 2)]["crowd_error"]) for q in qids]
            st = exact_wilcoxon_tests(xs, ys)
            p_norm_list.append(st["pipeline_wilcoxon_p"])
            p_ex_list.append(st["exact_wilcoxon_nz_p"])
            pair_rows.append({"pair": f"{cb}_vs_{ca}", **st})
        for pr, hn, he in zip(
            pair_rows, holm_bonferroni(p_norm_list), holm_bonferroni(p_ex_list), strict=True
        ):
            pr["holm9_pipeline"] = hn
            pr["holm9_exact_nz"] = he
        # 2x2 DiD on crowd_error: (P4 - P3) - (P2 - P1)
        did_xs = [
            float(cmap[("P2", q, 0, 2)]["crowd_error"])
            - float(cmap[("P1", q, 0, 2)]["crowd_error"])
            for q in qids
        ]
        did_ys = [
            float(cmap[("P4", q, 0, 2)]["crowd_error"])
            - float(cmap[("P3", q, 0, 2)]["crowd_error"])
            for q in qids
        ]
        did_st = exact_wilcoxon_tests(did_xs, did_ys)
        return {
            "variant": name,
            "r2_means": means,
            "crowd_error_pairs": {pr["pair"]: pr for pr in pair_rows},
            "crowd_error_did": did_st,
        }

    # Build Variant A: First-Pass Saved (the 28 pass-1 cells deleted at 23:28 restored)
    pass1_overrides: dict[tuple[str, str, int, str], tuple[float | None, bool, str]] = {}
    for u in traced:
        if u["category"] == "pass1_saved_replaced_at_23_28":
            cell = tuple(u["resolved_cell"])
            pass1_overrides[cell] = (u["parsed_from_raw"], False, u["raw_tail"])

    # Build Variant B: First-Call-Ever (strictly chronologically first HTTP request per cell)
    first_call_overrides: dict[tuple[str, str, int, str], tuple[float | None, bool, str]] = {}
    earliest_by_cell: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    for u in sorted(traced, key=lambda x: x["req_idx"]):
        cell = tuple(u["resolved_cell"])
        if cell not in earliest_by_cell:
            earliest_by_cell[cell] = u
    for cell, u in earliest_by_cell.items():
        # Remember req_idx >= 1288 is AFTER pass1; only pass1 requests belong to earliest-ever!
        if int(u["req_idx"]) < 1288:
            pv = u["parsed_from_raw"]
            first_call_overrides[cell] = (pv, pv is None, u["raw_tail"])

    var_pass1 = _clone_and_apply(pass1_overrides)
    var_first_call = _clone_and_apply(first_call_overrides)
    return {
        "n_pass1_replaced_cells": len(pass1_overrides),
        "n_first_call_ever_overridden_cells": len(first_call_overrides),
        "variants": {
            "First_Pass_Saved_Pre2328": _summarize_variant("First_Pass_Saved_Pre2328", var_pass1),
            "First_Call_Ever_Strict": _summarize_variant("First_Call_Ever_Strict", var_first_call),
        },
        "changed_cell_details_pass1_replaced": [
            {
                "cell": list(k),
                "pass1_val": v[0],
                "retained_val": next(
                    r.parsed_value
                    for r in records
                    if (r.condition_id, r.question_id, r.round_idx, r.agent_id) == k
                ),
            }
            for k, v in sorted(pass1_overrides.items())
        ],
    }


def main() -> None:
    tasks = load_tasks_file(TASKS_PATH)
    records = load_records(RESULTS_DIR / "numeric_runs.jsonl")
    saved_analysis = json.loads((RESULTS_DIR / "numeric_analysis.json").read_text(encoding="utf-8"))
    req_lines = load_jsonl(RESULTS_DIR / "provider_requests.jsonl")

    # 1. Exact reproduction of numeric_analysis.json via analyze_numeric_core
    reproduced = analyze_numeric_core(records, tasks)
    from apart_epistemics.cli import _serialize_numeric_analysis

    repro_serialized = json.loads(json.dumps(_serialize_numeric_analysis(reproduced), allow_nan=False))
    matches_saved = repro_serialized == saved_analysis

    # 1b. Summarize 12-task archive vs 16-task archive
    tasks_12 = load_tasks_file(ARCHIVE_12_DIR / "configs" / "polyphony_benchmark_tasks.json")
    recs_12 = load_records(ARCHIVE_12_DIR / "numeric_runs.jsonl")
    analysis_12 = json.loads((ARCHIVE_12_DIR / "numeric_analysis.json").read_text(encoding="utf-8"))
    reqs_12 = load_jsonl(ARCHIVE_12_DIR / "provider_requests.jsonl")

    # 2. Provider request audit
    req_audit = match_requests_to_cells(req_lines, records, tasks)
    first_attempt_trace = reconstruct_first_attempt(req_audit, records, tasks)

    # 3. Build full cell metrics for 16-task retained dataset
    cell_map = build_cell_map(records, tasks, alpha=ALPHA)
    qids = [t.id for t in tasks]

    # 3b. First-Attempt sensitivity reconstruction (First-Pass Saved Pre-23:28 & First-Call-Ever)
    first_attempt_sensitivity = evaluate_first_attempt_datasets(
        first_attempt_trace, records, tasks, cell_map
    )
    (RESULTS_DIR / "first_attempt_reconstruction.json").write_text(
        json.dumps(first_attempt_sensitivity, indent=2), encoding="utf-8"
    )

    # 4. Exact vs normal-approx Wilcoxon, Pratt, paired t, win/loss/tie counts across P-matrix
    exact_stats: dict[str, list[dict[str, Any]]] = {}
    for metric in _P_MATRIX_METRICS:
        rows_m = []
        raw_norm_ps = []
        raw_exact_ps = []
        raw_pratt_ps = []
        for cond_a, cond_b in _P_MATRIX_PAIRS:
            cid = f"{cond_b.lower()}_vs_{cond_a.lower()}_{metric}"
            xs = [float(cell_map[(cond_a, q, 0, 2)][metric]) for q in qids]
            ys = [float(cell_map[(cond_b, q, 0, 2)][metric]) for q in qids]
            st = exact_wilcoxon_tests(xs, ys)
            ci_lo, ci_hi = bootstrap_ci([y - x for x, y in zip(xs, ys, strict=True)], seed=42)
            t_p = t_cdf_two_sided(st["t_stat"], st["n_total"] - 1)
            raw_norm_ps.append(st["pipeline_wilcoxon_p"])
            raw_exact_ps.append(st["exact_wilcoxon_nz_p"])
            raw_pratt_ps.append(st["exact_pratt_p"])
            rows_m.append(
                {
                    "contrast_id": cid,
                    "metric": metric,
                    "condition_a": cond_a,
                    "condition_b": cond_b,
                    "mean_a": sum(xs) / len(xs),
                    "mean_b": sum(ys) / len(ys),
                    "ci_low": ci_lo,
                    "ci_high": ci_hi,
                    "t_p": t_p,
                    **st,
                }
            )
        h_norm = holm_bonferroni(raw_norm_ps)
        h_exact = holm_bonferroni(raw_exact_ps)
        h_pratt = holm_bonferroni(raw_pratt_ps)
        for r, hn, he, hp in zip(rows_m, h_norm, h_exact, h_pratt, strict=True):
            r["holm9_pipeline"] = hn
            r["holm9_exact_nz"] = he
            r["holm9_exact_pratt"] = hp
        exact_stats[metric] = rows_m

    # Also compute exact 2x2 DiD stats and Global m=20 Holm exact vs pipeline
    did_stats: dict[str, Any] = {}
    did_metrics = ("crowd_error", "assistant_crowd_error", "avg_individual_error", "diversity", "evrr")
    raw_did_norm, raw_did_exact = [], []
    did_tmp = []
    for metric in did_metrics:
        xs = [0.0] * len(qids)
        ys = [
            (
                float(cell_map[("P4", q, 0, 2)][metric])
                - float(cell_map[("P3", q, 0, 2)][metric])
            )
            - (
                float(cell_map[("P2", q, 0, 2)][metric])
                - float(cell_map[("P1", q, 0, 2)][metric])
            )
            for q in qids
        ]
        st = exact_wilcoxon_tests(xs, ys)
        ci_lo, ci_hi = bootstrap_ci(ys, seed=42)
        t_p = t_cdf_two_sided(st["t_stat"], st["n_total"] - 1)
        raw_did_norm.append(st["pipeline_wilcoxon_p"])
        raw_did_exact.append(st["exact_wilcoxon_nz_p"])
        did_tmp.append({"metric": metric, "ci_low": ci_lo, "ci_high": ci_hi, "t_p": t_p, **st})
    for r, hn, he in zip(did_tmp, holm_bonferroni(raw_did_norm), holm_bonferroni(raw_did_exact), strict=True):
        r["holm5_pipeline"] = hn
        r["holm5_exact_nz"] = he
        did_stats[r["metric"]] = r

    # Global 20-contrast family exact vs pipeline
    p_core_specs = [
        (c["contrast_id"], c["metric"], c["condition_a"], c["condition_b"])
        for c in saved_analysis["contrasts"]
    ]
    g20_rows = []
    for cid, metric, ca, cb in p_core_specs:
        xs = [float(cell_map[(ca, q, 0, 2)][metric]) for q in qids]
        ys = [float(cell_map[(cb, q, 0, 2)][metric]) for q in qids]
        st = exact_wilcoxon_tests(xs, ys)
        g20_rows.append({"contrast_id": cid, "metric": metric, "ca": ca, "cb": cb, **st})
    g20_h_norm = holm_bonferroni([r["pipeline_wilcoxon_p"] for r in g20_rows])
    g20_h_exact = holm_bonferroni([r["exact_wilcoxon_nz_p"] for r in g20_rows])
    for r, hn, he in zip(g20_rows, g20_h_norm, g20_h_exact, strict=True):
        r["holm20_pipeline"] = hn
        r["holm20_exact_nz"] = he

    # 5. Per-model counts & within-condition MSE breakdown (R1, R2, Combined)
    per_model: dict[str, dict[str, Any]] = {}
    truth_by_qid = {t.id: math.log10(t.numeric_truth) for t in tasks if t.numeric_truth is not None}
    for cond in COND_ORDER:
        c_recs = [r for r in records if r.condition_id == cond]
        models_seen = sorted({r.model_slug for r in c_recs})
        per_model[cond] = {}
        for m in models_seen:
            m_all = [r for r in c_recs if r.model_slug == m and r.assistant_log10 is not None]
            m_r1 = [r for r in m_all if r.round_idx == 1]
            m_r2 = [r for r in m_all if r.round_idx == 2]
            per_model[cond][m] = {
                "n_total": len(m_all),
                "n_r1": len(m_r1),
                "n_r2": len(m_r2),
                "mse_combined": sum((r.assistant_log10 - truth_by_qid[r.question_id]) ** 2 for r in m_all) / len(m_all),
                "mse_r1": sum((r.assistant_log10 - truth_by_qid[r.question_id]) ** 2 for r in m_r1) / len(m_r1),
                "mse_r2": sum((r.assistant_log10 - truth_by_qid[r.question_id]) ** 2 for r in m_r2) / len(m_r2),
                "agent_mse_r2": sum((r.agent_log10_after - truth_by_qid[r.question_id]) ** 2 for r in m_r2) / len(m_r2),
            }

    # 6. Two definitions of between_share & Exact permutation null check for P3, P4
    diversity_audit: dict[str, Any] = {}
    for cond in ("P0", "P1", "P2", "P3", "P4"):
        diversity_audit[cond] = {}
        for rnd in (1, 2):
            cells = [cell_map[(cond, q, 0, rnd)] for q in qids]
            mean_d = sum(c["diversity"] for c in cells) / len(cells)
            mean_db = sum(c["between_model_diversity"] for c in cells) / len(cells)
            mean_dw = sum(c["within_model_diversity"] for c in cells) / len(cells)
            mean_ad = sum(c["assistant_diversity"] for c in cells) / len(cells)
            mean_adb = sum(c["assistant_between_model_diversity"] for c in cells) / len(cells)
            mean_adw = sum(c["assistant_within_model_diversity"] for c in cells) / len(cells)
            diversity_audit[cond][f"R{rnd}"] = {
                "agent_mean_of_task_ratios_between_share": sum(c["between_share"] for c in cells) / len(cells),
                "agent_ratio_of_aggregate_means": mean_db / mean_d if mean_d > 1e-15 else 0.0,
                "asst_mean_of_task_ratios_between_share": sum(c["assistant_between_share"] for c in cells) / len(cells),
                "asst_ratio_of_aggregate_means": mean_adb / mean_ad if mean_ad > 1e-15 else 0.0,
            }

    # Exact permutation verification on balanced N=8, K=4, n_k=2 partitions (105 partitions/task):
    # E_null[D_between | y] = (K-1)/(N-K) * E_null[D_within | y] = 0.75 * E_null[D_within | y]
    all_partitions: list[tuple[tuple[int, ...], ...]] = []
    # Generate all 105 unlabeled partitions of {0..7} into 4 pairs
    def _gen_pairs(items: tuple[int, ...]) -> list[list[tuple[int, int]]]:
        if not items:
            return [[]]
        first = items[0]
        out = []
        for idx in range(1, len(items)):
            pair = (first, items[idx])
            rest = items[1:idx] + items[idx + 1 :]
            for sub in _gen_pairs(rest):
                out.append([pair, *sub])
        return out

    partitions = _gen_pairs(tuple(range(8)))
    assert len(partitions) == 105
    perm_audit: dict[str, Any] = {}
    for cond in ("P3", "P4"):
        for level in ("agent", "assistant"):
            obs_dex = []
            perm_ps = []
            null_ratios = []
            for q in qids:
                q_recs = [
                    r for r in records if r.condition_id == cond and r.question_id == q and r.round_idx == 2
                ]
                vals = [
                    r.agent_log10_after if level == "agent" else float(r.assistant_log10)
                    for r in q_recs
                ]
                slugs = [r.model_slug for r in q_recs]
                obs_decomp = compute_hierarchical_page_decomposition(vals, slugs, truth_by_qid[q])
                obs_dex.append(obs_decomp["excess_between_model_diversity"])
                null_db = []
                null_dw = []
                for part in partitions:
                    fake_slugs = [""] * 8
                    for k_idx, (i1, i2) in enumerate(part):
                        fake_slugs[i1] = f"m{k_idx}"
                        fake_slugs[i2] = f"m{k_idx}"
                    dec = compute_hierarchical_page_decomposition(vals, fake_slugs, truth_by_qid[q])
                    null_db.append(dec["between_model_diversity"])
                    null_dw.append(dec["within_model_diversity"])
                e_db = sum(null_db) / len(null_db)
                e_dw = sum(null_dw) / len(null_dw)
                null_ratios.append(e_db / e_dw if e_dw > 1e-15 else 0.75)
                ge_count = sum(1 for db in null_db if db >= obs_decomp["between_model_diversity"] - 1e-15)
                perm_ps.append(ge_count / len(null_db))
            perm_audit[f"{cond}_{level}_R2"] = {
                "mean_obs_D_excess": sum(obs_dex) / len(obs_dex),
                "min_null_E_Db_over_Dw": min(null_ratios),
                "max_null_E_Db_over_Dw": max(null_ratios),
                "n_tasks_perm_p_le_0_05": sum(1 for p in perm_ps if p <= 0.05),
                "mean_task_perm_p": sum(perm_ps) / len(perm_ps),
                "task_perm_ps": dict(zip(qids, perm_ps, strict=True)),
            }

    # 7. Exact signed-error cross-term & unrolled alpha=0.4 variance breakdown
    r2_cross_decomp: dict[str, Any] = {}
    unrolled_decomp: dict[str, Any] = {}
    for cond in COND_ORDER:
        t1_list, t2_list, t3_list, tot_list = [], [], [], []
        # Also unrolled per-agent covariance/variance shares:
        # x2 = 0.36 x0 + 0.24 a1 + 0.40 a2 (centered error e = x - y*)
        u_036e0_sq, u_024a1_sq, u_040a2_sq = [], [], []
        u_c01, u_c02, u_c12 = [], [], []
        var_w_shares_within_cell = []
        for q in qids:
            r1_recs = sorted(
                [r for r in records if r.condition_id == cond and r.question_id == q and r.round_idx == 1],
                key=lambda r: r.agent_id,
            )
            r2_recs = sorted(
                [r for r in records if r.condition_id == cond and r.question_id == q and r.round_idx == 2],
                key=lambda r: r.agent_id,
            )
            lt = truth_by_qid[q]
            x0_bar = sum(r.agent_log10_before - lt for r in r1_recs) / 8.0
            x1_bar = sum(r.agent_log10_after - lt for r in r1_recs) / 8.0
            a1_bar = sum(float(r.assistant_log10) - lt for r in r1_recs) / 8.0
            a2_bar = sum(float(r.assistant_log10) - lt for r in r2_recs) / 8.0
            x2_bar = sum(r.agent_log10_after - lt for r in r2_recs) / 8.0

            term_r1 = 0.36 * (x1_bar**2)
            term_a2 = 0.16 * (a2_bar**2)
            term_cross = 0.48 * x1_bar * a2_bar
            assert abs((term_r1 + term_a2 + term_cross) - (x2_bar**2)) < 1e-12
            t1_list.append(term_r1)
            t2_list.append(term_a2)
            t3_list.append(term_cross)
            tot_list.append(x2_bar**2)

            # Unrolled crowd error
            u_036e0_sq.append((0.36 * x0_bar) ** 2)
            u_024a1_sq.append((0.24 * a1_bar) ** 2)
            u_040a2_sq.append((0.40 * a2_bar) ** 2)
            u_c01.append(2 * 0.36 * 0.24 * x0_bar * a1_bar)
            u_c02.append(2 * 0.36 * 0.40 * x0_bar * a2_bar)
            u_c12.append(2 * 0.24 * 0.40 * a1_bar * a2_bar)

            # Exact within-cell variance attribution for x2 = 0.36*x0 + 0.24*a1 + 0.40*a2
            x0s = [r.agent_log10_before for r in r1_recs]
            a1s = [float(r.assistant_log10) for r in r1_recs]
            a2s = [float(r.assistant_log10) for r in r2_recs]
            x2s = [r.agent_log10_after for r in r2_recs]
            mx0, ma1, ma2, mx2 = sum(x0s)/8, sum(a1s)/8, sum(a2s)/8, sum(x2s)/8
            vx0 = sum((v - mx0)**2 for v in x0s)/8
            vx2 = sum((v - mx2)**2 for v in x2s)/8
            # Covariance share = Cov(0.36*x0, x2) / Var(x2), vs pure 0.36^2 * Var(x0) / Var(x2)
            cov_036x0_x2 = sum(0.36*(x0s[i]-mx0)*(x2s[i]-mx2) for i in range(8))/8
            var_w_shares_within_cell.append(
                {
                    "pure_01296_vx0_over_vx2": (0.1296 * vx0) / vx2 if vx2 > 1e-15 else 0.0,
                    "cov_share_036x0_x2_over_vx2": cov_036x0_x2 / vx2 if vx2 > 1e-15 else 0.0,
                    "pure_01296_vx0_num": 0.1296 * vx0,
                    "vx2": vx2,
                }
            )

        r2_cross_decomp[cond] = {
            "0.36_E_crowd_R1": sum(t1_list) / 16.0,
            "0.16_E_asst_crowd_R2": sum(t2_list) / 16.0,
            "0.48_signed_cross_term": sum(t3_list) / 16.0,
            "E_crowd_R2_sum": sum(tot_list) / 16.0,
            "per_question_cross": dict(zip(qids, t3_list, strict=True)),
        }
        mean_pure = sum(d["pure_01296_vx0_num"] for d in var_w_shares_within_cell) / 16.0
        mean_vx2 = sum(d["vx2"] for d in var_w_shares_within_cell) / 16.0
        unrolled_decomp[cond] = {
            "0.1296_E_crowd_R0": sum(u_036e0_sq) / 16.0,
            "0.0576_E_asst_R1": sum(u_024a1_sq) / 16.0,
            "0.1600_E_asst_R2": sum(u_040a2_sq) / 16.0,
            "cross_x0_a1": sum(u_c01) / 16.0,
            "cross_x0_a2": sum(u_c02) / 16.0,
            "cross_a1_a2": sum(u_c12) / 16.0,
            "sum_check": sum(
                sum(x) / 16.0
                for x in (u_036e0_sq, u_024a1_sq, u_040a2_sq, u_c01, u_c02, u_c12)
            ),
            "mean_ratio_01296vx0_over_vx2": sum(d["pure_01296_vx0_over_vx2"] for d in var_w_shares_within_cell) / 16.0,
            "agg_ratio_01296vx0_over_vx2": mean_pure / mean_vx2 if mean_vx2 > 1e-15 else 0.0,
            "mean_cov_share_036x0_over_vx2": sum(d["cov_share_036x0_x2_over_vx2"] for d in var_w_shares_within_cell) / 16.0,
        }

    # 8. Leave-One-Task-Out (LOTO, n=15) & Influential-Task Shares
    loto_rows: dict[str, Any] = {}
    lake_sup_shares: dict[str, Any] = {}
    focal_contrasts = [
        ("crowd_error", "P0", "P1"),
        ("crowd_error", "P1", "P2"),
        ("crowd_error", "P1", "P3"),
        ("crowd_error", "P1", "P4"),
        ("crowd_error", "P3", "P4"),
        ("crowd_error", "P0", "P4"),
        ("assistant_crowd_error", "P1", "P3"),
        ("assistant_crowd_error", "P1", "P4"),
    ]
    for metric, ca, cb in focal_contrasts:
        key = f"{cb}_vs_{ca}_{metric}"
        full_diffs = [
            float(cell_map[(cb, q, 0, 2)][metric]) - float(cell_map[(ca, q, 0, 2)][metric])
            for q in qids
        ]
        tot_diff = sum(full_diffs)
        ls_idx = qids.index("hyd_lake_superior_vol_km3")
        lake_sup_shares[key] = {
            "lake_superior_diff": full_diffs[ls_idx],
            "total_16task_sum_diff": tot_diff,
            "lake_superior_share_pct": (100.0 * full_diffs[ls_idx] / tot_diff) if abs(tot_diff) > 1e-15 else 0.0,
            "top_3_contributors": sorted(
                [
                    {
                        "qid": qids[i],
                        "diff": full_diffs[i],
                        "pct_of_total": (100.0 * full_diffs[i] / tot_diff) if abs(tot_diff) > 1e-15 else 0.0,
                    }
                    for i in range(16)
                ],
                key=lambda x: abs(x["diff"]),
                reverse=True,
            )[:5],
        }

    # Full LOTO across all 16 tasks for all 9 m=9 matrix contrasts of crowd_error + DiD
    loto_by_dropped_qid: dict[str, Any] = {}
    for drop_q in qids:
        rem_q = [q for q in qids if q != drop_q]
        # Recompute m=9 Holm for crowd_error
        pair_results = []
        norm_ps = []
        exact_ps = []
        for ca, cb in _P_MATRIX_PAIRS:
            xs = [float(cell_map[(ca, q, 0, 2)]["crowd_error"]) for q in rem_q]
            ys = [float(cell_map[(cb, q, 0, 2)]["crowd_error"]) for q in rem_q]
            st = exact_wilcoxon_tests(xs, ys)
            norm_ps.append(st["pipeline_wilcoxon_p"])
            exact_ps.append(st["exact_wilcoxon_nz_p"])
            pair_results.append(
                {
                    "pair": f"{cb}_vs_{ca}",
                    "mean_a": sum(xs) / len(xs),
                    "mean_b": sum(ys) / len(ys),
                    **st,
                }
            )
        hn = holm_bonferroni(norm_ps)
        he = holm_bonferroni(exact_ps)
        for pr, h_n, h_e in zip(pair_results, hn, he, strict=True):
            pr["holm9_pipeline"] = h_n
            pr["holm9_exact_nz"] = h_e

        # Also 2x2 DiD on crowd_error after dropping drop_q
        did_y = [
            (
                float(cell_map[("P4", q, 0, 2)]["crowd_error"])
                - float(cell_map[("P3", q, 0, 2)]["crowd_error"])
            )
            - (
                float(cell_map[("P2", q, 0, 2)]["crowd_error"])
                - float(cell_map[("P1", q, 0, 2)]["crowd_error"])
            )
            for q in rem_q
        ]
        did_st = exact_wilcoxon_tests([0.0] * len(rem_q), did_y)

        # Also global m=20 Holm for P-series core pairs on remaining 15 tasks
        g20_p_norm, g20_p_ex = [], []
        for cid, metric, ca, cb in p_core_specs:
            xs = [float(cell_map[(ca, q, 0, 2)][metric]) for q in rem_q]
            ys = [float(cell_map[(cb, q, 0, 2)][metric]) for q in rem_q]
            st20 = exact_wilcoxon_tests(xs, ys)
            g20_p_norm.append(st20["pipeline_wilcoxon_p"])
            g20_p_ex.append(st20["exact_wilcoxon_nz_p"])
        g20_hn = holm_bonferroni(g20_p_norm)
        g20_he = holm_bonferroni(g20_p_ex)
        g20_map = {
            cid: {"holm20_pipeline": hn_v, "holm20_exact_nz": he_v}
            for (cid, _, _, _), hn_v, he_v in zip(p_core_specs, g20_hn, g20_he, strict=True)
        }

        loto_by_dropped_qid[drop_q] = {
            "crowd_error_pairs": {p["pair"]: p for p in pair_results},
            "crowd_error_did": did_st,
            "holm20_map": g20_map,
        }

    # Summarize LOTO min/max across all 16 dropped tasks for key contrasts
    loto_summary: dict[str, Any] = {}
    for pair_key in ("P1_vs_P0", "P2_vs_P1", "P3_vs_P1", "P4_vs_P1", "P4_vs_P3", "P4_vs_P0"):
        diffs = [loto_by_dropped_qid[q]["crowd_error_pairs"][pair_key]["mean_diff"] for q in qids]
        p_norm = [loto_by_dropped_qid[q]["crowd_error_pairs"][pair_key]["pipeline_wilcoxon_p"] for q in qids]
        p_ex = [loto_by_dropped_qid[q]["crowd_error_pairs"][pair_key]["exact_wilcoxon_nz_p"] for q in qids]
        h9_norm = [loto_by_dropped_qid[q]["crowd_error_pairs"][pair_key]["holm9_pipeline"] for q in qids]
        h9_ex = [loto_by_dropped_qid[q]["crowd_error_pairs"][pair_key]["holm9_exact_nz"] for q in qids]
        cid_map = {
            "P1_vs_P0": "p1_vs_p0_crowd_error",
            "P2_vs_P1": "p2_vs_p1_crowd_error",
            "P3_vs_P1": "p3_vs_p1_crowd_error",
            "P4_vs_P1": "p4_vs_p1_crowd_error",
            "P4_vs_P3": "p4_vs_p3_crowd_error",
            "P4_vs_P0": "p4_vs_p0_crowd_error",
        }
        h20_norm = [loto_by_dropped_qid[q]["holm20_map"][cid_map[pair_key]]["holm20_pipeline"] for q in qids]
        loto_summary[pair_key] = {
            "diff_min_max": [min(diffs), max(diffs)],
            "raw_p_norm_min_max": [min(p_norm), max(p_norm)],
            "raw_p_exact_min_max": [min(p_ex), max(p_ex)],
            "holm9_norm_min_max": [min(h9_norm), max(h9_norm)],
            "holm9_exact_min_max": [min(h9_ex), max(h9_ex)],
            "holm20_norm_min_max": [min(h20_norm), max(h20_norm)],
            "n_subsets_raw_exact_lt_005": sum(1 for v in p_ex if v < 0.05),
            "n_subsets_holm9_norm_lt_005": sum(1 for v in h9_norm if v < 0.05),
            "n_subsets_holm9_exact_lt_005": sum(1 for v in h9_ex if v < 0.05),
            "n_subsets_holm20_norm_lt_005": sum(1 for v in h20_norm if v < 0.05),
            "drop_lake_superior": loto_by_dropped_qid["hyd_lake_superior_vol_km3"]["crowd_error_pairs"][pair_key],
            "drop_lake_superior_holm20": loto_by_dropped_qid["hyd_lake_superior_vol_km3"]["holm20_map"][cid_map[pair_key]],
        }

    # 9. Inspect LHC exact tie + Molybdenum Kelvin/Celsius + Norway/EIA model answers & sensitivity
    lhc_vals = sorted({round(r.agent_log10_after, 6) for r in records if r.question_id == "sci_lhc_dipole_magnets_count"})
    lhc_raw_parsed = Counter(
        r.parsed_value for r in records if r.question_id == "sci_lhc_dipole_magnets_count"
    )
    moly_parsed = Counter(
        round(float(r.parsed_value), 1)
        for r in records
        if r.question_id == "mat_molybdenum_melting_c" and r.parsed_value is not None
    )
    norway_parsed = Counter(
        round(float(r.parsed_value), 1)
        for r in records
        if r.question_id == "nrg_norway_hydro_2023_twh" and r.parsed_value is not None
    )
    ushydro_parsed = Counter(
        round(float(r.parsed_value), 1)
        for r in records
        if r.question_id == "nrg_us_hydro_2023_twh" and r.parsed_value is not None
    )
    uswind_parsed = Counter(
        round(float(r.parsed_value), 1)
        for r in records
        if r.question_id == "nrg_us_wind_2023_twh" and r.parsed_value is not None
    )

    # Re-score under source-revision sensitivity scenarios (preserving original files untouched!)
    sensitivity_scenarios = {
        "S1_eia_final_us_hydro_245": {"nrg_us_hydro_2023_twh": 245.0},
        "S2_ssb_norway_hydro_137_3": {"nrg_norway_hydro_2023_twh": 137.315},
        "S3_eia_us_245_plus_ssb_norway_137_3": {
            "nrg_us_hydro_2023_twh": 245.0,
            "nrg_norway_hydro_2023_twh": 137.315,
        },
        "S4_drop_molybdenum_n15": {"__drop__": "mat_molybdenum_melting_c"},
    }
    source_sensitivity: dict[str, Any] = {}
    base_truth_map = {t.id: float(t.numeric_truth) for t in tasks}
    for sname, overrides in sensitivity_scenarios.items():
        if "__drop__" in overrides:
            drop_id = overrides["__drop__"]
            sub_q = [q for q in qids if q != drop_id]
            s_cell_map = cell_map
        else:
            sub_q = list(qids)
            mod_truth = {
                qid: math.log10(overrides.get(qid, val)) for qid, val in base_truth_map.items()
            }
            by_cell_s: dict[tuple[str, str, int, int], list[NumericEstimateRecord]] = defaultdict(list)
            for rec in records:
                by_cell_s[(rec.condition_id, rec.question_id, rec.replicate, rec.round_idx)].append(rec)
            s_cell_map, _ = _compute_cell_metrics(by_cell_s, mod_truth, ALPHA)

        cond_means_r2 = {
            c: {
                "crowd_error": sum(float(s_cell_map[(c, q, 0, 2)]["crowd_error"]) for q in sub_q) / len(sub_q),
                "assistant_crowd_error": sum(float(s_cell_map[(c, q, 0, 2)]["assistant_crowd_error"]) for q in sub_q) / len(sub_q),
            }
            for c in COND_ORDER
        }
        p_norm_s, p_ex_s, pair_s = [], [], []
        for ca, cb in _P_MATRIX_PAIRS:
            xs = [float(s_cell_map[(ca, q, 0, 2)]["crowd_error"]) for q in sub_q]
            ys = [float(s_cell_map[(cb, q, 0, 2)]["crowd_error"]) for q in sub_q]
            st = exact_wilcoxon_tests(xs, ys)
            p_norm_s.append(st["pipeline_wilcoxon_p"])
            p_ex_s.append(st["exact_wilcoxon_nz_p"])
            pair_s.append({"pair": f"{cb}_vs_{ca}", **st})
        for pr, hn, he in zip(pair_s, holm_bonferroni(p_norm_s), holm_bonferroni(p_ex_s), strict=True):
            pr["holm9_pipeline"] = hn
            pr["holm9_exact_nz"] = he
        source_sensitivity[sname] = {
            "n_tasks": len(sub_q),
            "r2_means": cond_means_r2,
            "crowd_error_pairs": {p["pair"]: p for p in pair_s},
        }

    # Assemble clean JSON report (without huge raw prompt/reply dumps)
    req_audit_clean = {k: v for k, v in req_audit.items() if not k.startswith("_")}
    out = {
        "provenance": {
            "exact_reproduction_matches_saved_numeric_analysis_json": matches_saved,
            "max_identity_residual": saved_analysis["max_identity_residual"],
            "n_tasks": len(tasks),
            "n_retained_rows": len(records),
            "earlier_12task_archive": {
                "path": str(ARCHIVE_12_DIR.relative_to(ROOT)),
                "commit": "b5b91ce",
                "n_tasks": len(tasks_12),
                "alpha": 0.3,
                "n_retained_rows": len(recs_12),
                "n_provider_requests": len(reqs_12),
                "r2_by_condition": {
                    c: analysis_12["by_condition_final"][c]["crowd_error"] for c in COND_ORDER
                },
            },
            "provider_requests_audit": req_audit_clean,
            "first_attempt_unmatched_trace_summary": {
                "n_unmatched": first_attempt_trace["n_unmatched"],
                "by_qid_counts": first_attempt_trace["by_qid_counts"],
                "category_counts": first_attempt_trace["category_counts"],
                "unmatched_requests": first_attempt_trace["traced_unmatched"],
            },
        },
        "first_attempt_sensitivity": first_attempt_sensitivity,
        "r1_and_r2_four_errors": {
            cond: {
                f"R{rnd}": {
                    "crowd_error": sum(float(cell_map[(cond, q, 0, rnd)]["crowd_error"]) for q in qids) / 16.0,
                    "avg_individual_error": sum(float(cell_map[(cond, q, 0, rnd)]["avg_individual_error"]) for q in qids) / 16.0,
                    "assistant_crowd_error": sum(float(cell_map[(cond, q, 0, rnd)]["assistant_crowd_error"]) for q in qids) / 16.0,
                    "assistant_avg_individual_error": sum(float(cell_map[(cond, q, 0, rnd)]["assistant_avg_individual_error"]) for q in qids) / 16.0,
                }
                for rnd in (1, 2)
            }
            for cond in COND_ORDER
        },
        "per_model_breakdown": per_model,
        "diversity_share_audit": diversity_audit,
        "exact_random_label_permutation_audit": perm_audit,
        "r2_signed_error_cross_term_decomposition": r2_cross_decomp,
        "unrolled_alpha04_variance_decomposition": unrolled_decomp,
        "exact_vs_pipeline_wilcoxon_matrix": exact_stats,
        "exact_vs_pipeline_2x2_did": did_stats,
        "exact_vs_pipeline_global_m20": g20_rows,
        "lake_superior_and_influential_shares": lake_sup_shares,
        "loto_summary": loto_summary,
        "loto_drop_lake_superior_did": loto_by_dropped_qid["hyd_lake_superior_vol_km3"]["crowd_error_did"],
        "benchmark_answer_distributions": {
            "lhc_exact_tie_values": lhc_vals,
            "lhc_parsed_values_top": dict(lhc_raw_parsed.most_common(10)),
            "molybdenum_parsed_values": dict(moly_parsed.most_common(20)),
            "norway_hydro_parsed_values": dict(norway_parsed.most_common(20)),
            "us_hydro_parsed_values": dict(ushydro_parsed.most_common(15)),
            "us_wind_parsed_values": dict(uswind_parsed.most_common(15)),
        },
        "source_and_unit_sensitivity": source_sensitivity,
    }

    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved audit report to {OUT_JSON}")
    print(f"Exact reproduction match: {matches_saved}")
    print(
        f"Provider requests: {req_audit_clean['total_provider_requests']}, "
        f"Retained: {req_audit_clean['total_retained_rows']}, "
        f"Extra: {req_audit_clean['extra_requests']}, "
        f"Length finishes: {req_audit_clean['length_finish_reason_count']}, "
        f"Cost retained=${req_audit_clean['retained_totals']['cost_usd']:.4f} vs "
        f"all_reqs=${req_audit_clean['all_provider_requests_totals']['cost_usd']:.4f}"
    )
    print(
        f"Lake Superior P4-vs-P0 share: "
        f"{lake_sup_shares['P4_vs_P0_crowd_error']['lake_superior_share_pct']:.4f}%"
    )
    print("LOTO P4-vs-P1 m=9 Holm exact p range:", loto_summary["P4_vs_P1"]["holm9_exact_min_max"])
    print("LOTO P1-vs-P0 m=9 Holm exact p range:", loto_summary["P1_vs_P0"]["holm9_exact_min_max"])
    print("LOTO P2-vs-P1 m=9 Holm exact p range:", loto_summary["P2_vs_P1"]["holm9_exact_min_max"])
    print("LOTO P4-vs-P3 m=9 Holm exact p range:", loto_summary["P4_vs_P3"]["holm9_exact_min_max"])


if __name__ == "__main__":
    main()
