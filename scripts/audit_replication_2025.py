#!/usr/bin/env python3
"""Post-run integrity audit (Step 7), parity check (Step 8), and exploratory summaries (Step 9)."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from apart_epistemics.cli import _load_task_file as load_tasks_file
from apart_epistemics.engine.numeric_core import analyze_numeric_core
from apart_epistemics.models import NumericEstimateRecord

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "polyphony-replication-2025"
TASKS_PATH = ROOT / "configs" / "polyphony_replication_tasks_2025.filtered.json"


def _audit_runs_jsonl(runs_path: Path) -> tuple[list[NumericEstimateRecord], dict[str, Any]]:
    records: list[NumericEstimateRecord] = []
    seen_keys: set[tuple[str, str, int, int, str]] = set()
    duplicates = 0
    malformed_lines = 0
    misses: list[dict[str, Any]] = []
    truncations: list[dict[str, Any]] = []
    retries_used_present = False
    by_cond: Counter[str] = Counter()
    by_model: Counter[str] = Counter()

    with runs_path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, 1):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                raw_obj = json.loads(stripped)
                rec = NumericEstimateRecord.model_validate(raw_obj)
            except Exception as exc:
                malformed_lines += 1
                truncations.append({"line": line_no, "error": str(exc)})
                continue
            key = (rec.condition_id, rec.question_id, rec.replicate, rec.round_idx, rec.agent_id)
            if key in seen_keys:
                duplicates += 1
            seen_keys.add(key)
            records.append(rec)
            by_cond[rec.condition_id] += 1
            by_model[rec.model_slug] += 1
            if rec.miss:
                misses.append(
                    {
                        "condition_id": rec.condition_id,
                        "question_id": rec.question_id,
                        "replicate": rec.replicate,
                        "round_idx": rec.round_idx,
                        "agent_id": rec.agent_id,
                        "model_slug": rec.model_slug,
                        "finish_reason": rec.finish_reason,
                        "parsed_value": rec.parsed_value,
                        "raw_reply_tail": rec.raw_reply[-200:],
                    }
                )
            if rec.finish_reason and rec.finish_reason != "stop":
                truncations.append(
                    {
                        "condition_id": rec.condition_id,
                        "question_id": rec.question_id,
                        "replicate": rec.replicate,
                        "round_idx": rec.round_idx,
                        "agent_id": rec.agent_id,
                        "model_slug": rec.model_slug,
                        "finish_reason": rec.finish_reason,
                        "output_tokens": rec.output_tokens,
                        "miss": rec.miss,
                    }
                )
            if "retries_used" in raw_obj:
                retries_used_present = True

    return records, {
        "total_records": len(records),
        "unique_keys": len(seen_keys),
        "duplicates": duplicates,
        "malformed_lines": malformed_lines,
        "by_condition": dict(sorted(by_cond.items())),
        "by_model_slug": dict(sorted(by_model.items())),
        "miss_count": len(misses),
        "misses": misses,
        "non_stop_finish_count": len(truncations),
        "non_stop_finishes": truncations,
        "retries_used_field_in_records": retries_used_present,
    }


def _audit_provider_requests(req_path: Path, budget_path: Path) -> dict[str, Any]:
    http_successes = 0
    malformed = 0
    total_cost = 0.0
    missing_cost = 0
    cost_by_model: defaultdict[str, float] = defaultdict(float)
    calls_by_requested_model: Counter[str] = Counter()
    calls_by_response_model: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    incomplete_reasons: Counter[str] = Counter()
    format_retry_by_model: Counter[str] = Counter()
    in_tok = 0
    out_tok = 0
    reason_tok = 0

    with req_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                entry = json.loads(stripped)
            except json.JSONDecodeError:
                malformed += 1
                continue
            http_successes += 1
            req_m = str(entry.get("model") or "")
            resp = entry.get("response") or {}
            resp_m = str(resp.get("model") or req_m)
            calls_by_requested_model[req_m] += 1
            calls_by_response_model[resp_m] += 1
            req = entry.get("request") or {}
            req_input = str(req.get("input") or "")
            if "Respond with a single positive number prefixed by 'Final estimate: '." in req_input:
                format_retry_by_model[req_m] += 1
            statuses[str(resp.get("status") or "unknown")] += 1
            inc = resp.get("incomplete_details")
            if isinstance(inc, dict) and inc.get("reason"):
                incomplete_reasons[str(inc["reason"])] += 1
            usage = resp.get("usage") or {}
            in_tok += int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
            out_tok += int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
            det = usage.get("output_tokens_details") or usage.get("completion_tokens_details") or {}
            if isinstance(det, dict):
                reason_tok += int(det.get("reasoning_tokens") or 0)
            c = usage.get("cost")
            if isinstance(c, (int, float)):
                total_cost += float(c)
                cost_by_model[req_m] += float(c)
            else:
                missing_cost += 1

    budget = json.loads(budget_path.read_text(encoding="utf-8"))
    reserved_requests = int(budget["requests"])
    return {
        "http_successes": http_successes,
        "malformed_request_lines": malformed,
        "budget_reserved_requests": reserved_requests,
        "transient_http_retry_attempts": reserved_requests - http_successes,
        "budget_reserved_usd": budget["reserved_usd"],
        "budget_cap_usd": budget["cap_usd"],
        "per_request_allowance_usd": budget["per_request_allowance_usd"],
        "actual_cost_usd": round(total_cost, 6),
        "missing_cost_count": missing_cost,
        "cost_by_model_usd": {m: round(v, 6) for m, v in sorted(cost_by_model.items())},
        "calls_by_requested_model": dict(sorted(calls_by_requested_model.items())),
        "calls_by_response_model": dict(sorted(calls_by_response_model.items())),
        "statuses": dict(statuses),
        "incomplete_reasons": dict(incomplete_reasons),
        "format_retry_http_calls": sum(format_retry_by_model.values()),
        "format_retry_by_model": dict(sorted(format_retry_by_model.items())),
        "total_input_tokens": in_tok,
        "total_output_tokens": out_tok,
        "total_reasoning_tokens": reason_tok,
    }


def _compute_parity(
    prereg: dict[str, Any],
    pipeline: dict[str, Any],
) -> dict[str, Any]:
    """Compare preregistered_analysis.json against analyze_numeric_core output."""
    cond_diffs: dict[str, dict[str, Any]] = {}
    pr_means = prereg["mean_crowd_error"]
    pl_final = pipeline["by_condition_final"]
    for cond in sorted(set(pr_means) & set(pl_final)):
        a = round(float(pr_means[cond]), 6)
        b = round(float(pl_final[cond]["crowd_error"]), 6)
        cond_diffs[cond] = {
            "preregistered": a,
            "analyze_numeric_core": b,
            "diff": round(b - a, 8),
        }

    # Index pipeline crowd_error contrasts across all buckets
    pl_contrasts: dict[tuple[str, str], dict[str, Any]] = {}
    for bucket in (
        pipeline.get("contrasts", []),
        pipeline.get("control_contrasts", []),
        (pipeline.get("polyphony_contrast_matrix") or {}).get("crowd_error", []),
    ):
        for item in bucket:
            if item.get("metric") == "crowd_error":
                pl_contrasts[(item["condition_a"], item["condition_b"])] = item

    pr_contrasts = (
        prereg["confirmatory_holm_m8"]
        + prereg["secondary_holm_m2"]
        + prereg["descriptive"]
        + prereg["replication_p_series_holm_m4"]
    )
    contrast_comparisons: list[dict[str, Any]] = []
    for item in pr_contrasts:
        pair = (item["a"], item["b"])
        pl = pl_contrasts.get(pair)
        if pl is None and pair == ("POOLED_K1_SOCIAL_JENSEN_BOUND", "P3"):
            pl = pl_contrasts.get(("POOLED_K1_SOCIAL", "P3"))
        if pl is None:
            contrast_comparisons.append(
                {
                    "name": item["name"],
                    "pair": list(pair),
                    "in_analyze_numeric_core": False,
                    "reason": (
                        "Not computed by analyze_numeric_core (preregistered virtual/best panel)"
                    ),
                }
            )
            continue
        diffs = {}
        for fld_pr, fld_pl in (
            ("n_tasks", "n_cells"),
            ("mean_diff_b_minus_a", "mean_diff"),
            ("ci95_low", "ci_low"),
            ("ci95_high", "ci_high"),
            ("p_exact", "wilcoxon_p"),
            ("p_holm", "holm_adjusted_p"),
        ):
            if fld_pr == "ci95_low":
                v_pr = item["ci95"][0]
            elif fld_pr == "ci95_high":
                v_pr = item["ci95"][1]
            else:
                v_pr = item.get(fld_pr)
            v_pl = pl.get(fld_pl)
            if v_pr is not None and v_pl is not None and v_pr != v_pl:
                diffs[f"{fld_pr}_vs_{fld_pl}"] = {
                    "preregistered": v_pr,
                    "analyze_numeric_core": v_pl,
                    "delta": round(float(v_pl) - float(v_pr), 6),
                }
        contrast_comparisons.append(
            {
                "name": item["name"],
                "pair": list(pair),
                "in_analyze_numeric_core": True,
                "differing_fields": diffs,
            }
        )

    return {
        "condition_mean_crowd_error_parity": cond_diffs,
        "all_condition_means_identical": all(v["diff"] == 0 for v in cond_diffs.values()),
        "contrast_comparisons": contrast_comparisons,
    }


def _exploratory_stratified_summary(records: list[NumericEstimateRecord], tasks: list[Any]) -> Any:
    """Compute exploratory stratum-by-condition E2 & |xbar_2 - y*| summary."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import analyze_controls_preregistered as acp

    strata_by_qid = {t.id: t.selection_stratum for t in tasks}
    recs, specs = acp.load_records([RESULTS_DIR / "numeric_runs.jsonl"])
    truth = acp.truths_from_specs(specs)
    idx = acp.index_by_condition(recs)
    n_agents = 8
    rnd = 2
    out: dict[str, Any] = {}
    for stratum in ("control", "low_spread", "high_spread"):
        qids = {q for q, s in strata_by_qid.items() if s == stratum}
        stratum_out: dict[str, Any] = {"n_tasks": len(qids), "metrics": {}}
        for metric in ("squared_log_error", "abs_log_error"):
            errs: dict[str, dict[str, float]] = {}
            for cond in sorted(idx):
                ce = acp.condition_errors(idx, cond, truth, rnd, n_agents, metric=metric)
                ce_sub = {k: v for k, v in ce.items() if k[0] in qids}
                errs[cond] = acp.collapse_replicates(ce_sub)
            v_soc, _ = acp.virtual_panel_errors(
                idx, "P3", acp.K1_SOCIAL, truth, rnd, n_agents, metric=metric
            )
            v_sil, _ = acp.virtual_panel_errors(
                idx, "P0_POLY", acp.K1_SILENT, truth, rnd, n_agents, metric=metric
            )
            errs[acp.VIRTUAL_SOCIAL] = acp.collapse_replicates(
                {k: v for k, v in v_soc.items() if k[0] in qids}
            )
            errs[acp.VIRTUAL_SILENT] = acp.collapse_replicates(
                {k: v for k, v in v_sil.items() if k[0] in qids}
            )
            best_sil = min(
                ("P0", "P0_DS", "P0_QWEN", "P0_GLM"),
                key=lambda c: sum(errs[c].values()) / len(errs[c]),
            )
            best_map = {"P0": "P1", "P0_DS": "P1_DS", "P0_QWEN": "P1_QWEN", "P0_GLM": "P1_GLM"}
            errs[acp.BEST_K1_SOCIAL] = errs[best_map[best_sil]]
            stratum_out["metrics"][metric] = {
                "arm_means": {c: round(sum(v.values()) / len(v), 6) for c, v in errs.items()},
                "key_contrasts": [
                    acp.contrast("p1_vs_p0", "P0", "P1", errs),
                    acp.contrast("p2_vs_p1", "P1", "P2", errs),
                    acp.contrast("critique_vs_reflect", "P1_REFLECT", "P2", errs),
                    acp.contrast("p3_vs_silent_poly", "P0_POLY", "P3", errs),
                    acp.contrast("p4_vs_p3", "P3", "P4", errs),
                    acp.contrast("p3_vs_virtual_social", acp.VIRTUAL_SOCIAL, "P3", errs),
                    acp.contrast("p4_vs_best_k1_social", acp.BEST_K1_SOCIAL, "P4", errs),
                ],
            }
        out[stratum] = stratum_out
    del records
    return out


def main() -> int:
    tasks = load_tasks_file(TASKS_PATH)
    runs_path = RESULTS_DIR / "numeric_runs.jsonl"
    req_path = RESULTS_DIR / "provider_requests.jsonl"
    budget_path = RESULTS_DIR / "budget.json"
    prereg_path = RESULTS_DIR / "preregistered_analysis.json"

    records, runs_audit = _audit_runs_jsonl(runs_path)
    req_audit = _audit_provider_requests(req_path, budget_path)

    pipeline_raw = analyze_numeric_core(records, tasks)
    pipeline = json.loads(json.dumps(pipeline_raw, default=lambda o: o.model_dump()))
    (RESULTS_DIR / "numeric_analysis.json").write_text(
        json.dumps(pipeline, indent=2) + "\n", encoding="utf-8"
    )
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    parity = _compute_parity(prereg, pipeline)
    exploratory_strata = _exploratory_stratified_summary(records, tasks)

    cold_req_audit = _audit_provider_requests(
        RESULTS_DIR / "cold_probe" / "provider_requests.jsonl",
        RESULTS_DIR / "cold_probe" / "budget.json",
    )

    report = {
        "completion_status": pipeline["completion_status"],
        "missing_or_incomplete_cells": pipeline["missing_or_incomplete_cells"],
        "max_identity_residual": pipeline["max_identity_residual"],
        "runs_audit": runs_audit,
        "provider_requests_audit": req_audit,
        "cold_probe_requests_audit": cold_req_audit,
        "parity_with_preregistered_analysis": parity,
        "exploratory_by_stratum": exploratory_strata,
    }
    out_path = RESULTS_DIR / "audit_replication_report.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "completion_status": report["completion_status"],
                "missing_or_incomplete_cells": report["missing_or_incomplete_cells"],
                "max_identity_residual": report["max_identity_residual"],
                "total_records": runs_audit["total_records"],
                "miss_count": runs_audit["miss_count"],
                "non_stop_finish_count": runs_audit["non_stop_finish_count"],
                "format_retry_http_calls": req_audit["format_retry_http_calls"],
                "format_retry_by_model": req_audit["format_retry_by_model"],
                "main_http_successes": req_audit["http_successes"],
                "main_actual_cost_usd": req_audit["actual_cost_usd"],
                "cold_probe_cost_usd": cold_req_audit["actual_cost_usd"],
                "all_condition_means_identical": parity["all_condition_means_identical"],
                "calls_by_response_model": req_audit["calls_by_response_model"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
