#!/usr/bin/env python3
"""Run Step 5 single-shot cold probe across 4 models and filter replication tasks."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from apart_epistemics.backends.base import BudgetExceededError
from apart_epistemics.backends.providers import AdaptionBackend
from apart_epistemics.engine.numeric_core import p_series_system_prompt
from apart_epistemics.models import TaskItem
from apart_epistemics.parsers import parse_numeric_estimate
from apart_epistemics.run_state import configure_budget

MODELS = (
    "gemma-4-26b-a4b-it",
    "deepseek-v4-flash",
    "qwen3.8-27b",
    "glm-5.2",
)
RECALL_LOG10_THRESHOLD = 0.0086
D_CUTOFF_LOW = 0.000065
D_CUTOFF_HIGH = 0.00195
DEFAULT_CURRENT_DATE = "2026-11-15"


def build_cold_probe_prompts(task: TaskItem, current_date: str) -> tuple[str, str]:
    """Build question-only cold-probe prompts using p_series_system_prompt."""
    sys_p = p_series_system_prompt(current_date)
    in_unit = f" in {task.unit}" if task.unit else ""
    usr_p = (
        f"{task.question}\n"
        "What is your best estimate of the true quantity? "
        f"End with 'Final estimate: <plain number{in_unit}>' and nothing after the number."
    )
    return sys_p, usr_p


def assign_stratum(d_cold: float) -> str:
    """Assign descriptive stratum using pre-registered fixed D_cold midpoint cutoffs."""
    if d_cold < D_CUTOFF_LOW:
        return "control"
    if d_cold < D_CUTOFF_HIGH:
        return "low_spread"
    return "high_spread"


def _probe_one_cell(
    backend: Any,
    task: TaskItem,
    model_slug: str,
    current_date: str,
) -> dict[str, Any]:
    sys_p, usr_p = build_cold_probe_prompts(task, current_date)
    y_star = math.log10(float(task.numeric_truth))
    costs: list[float | None] = []
    out_tokens = 0
    in_tokens = 0
    retries_used = 0
    try:
        resp = backend.generate_numeric_reply(
            system_prompt=sys_p,
            user_prompt=usr_p,
            max_tokens=8192,
        )
        costs.append(resp.get("cost_usd"))
        in_tokens += int(resp.get("input_tokens", 0) or 0)
        out_tokens += int(resp.get("output_tokens", 0) or 0)
        parsed = parse_numeric_estimate(str(resp.get("raw_reply", "")), require_positive=True)
        if parsed.miss:
            retries_used = 1
            retry_prompt = (
                f"{usr_p}\n\nRespond with a single positive number prefixed by 'Final estimate: '."
            )
            resp = backend.generate_numeric_reply(
                system_prompt=sys_p,
                user_prompt=retry_prompt,
                max_tokens=8192,
            )
            costs.append(resp.get("cost_usd"))
            in_tokens += int(resp.get("input_tokens", 0) or 0)
            out_tokens += int(resp.get("output_tokens", 0) or 0)
            parsed = parse_numeric_estimate(str(resp.get("raw_reply", "")), require_positive=True)
    except BudgetExceededError:
        raise
    except Exception as exc:
        return {
            "task_id": task.id,
            "model_slug": model_slug,
            "parsed_value": None,
            "log10_estimate": None,
            "log10_truth": y_star,
            "log10_error": None,
            "abs_log10_error": None,
            "miss": True,
            "error": f"{type(exc).__name__}: {exc}",
            "retries_used": retries_used,
            "input_tokens": in_tokens,
            "output_tokens": out_tokens,
            "finish_reason": "error",
            "cost_usd": (
                sum(float(c) for c in costs if c is not None)
                if any(c is not None for c in costs)
                else None
            ),
            "raw_reply": "",
        }

    log_est = parsed.log10_value if not parsed.miss else None
    err = (log_est - y_star) if log_est is not None else None
    total_cost = (
        sum(float(c) for c in costs if c is not None) if all(c is not None for c in costs) else None
    )
    return {
        "task_id": task.id,
        "model_slug": model_slug,
        "parsed_value": parsed.value if not parsed.miss else None,
        "log10_estimate": log_est,
        "log10_truth": y_star,
        "log10_error": err,
        "abs_log10_error": abs(err) if err is not None else None,
        "miss": bool(parsed.miss or log_est is None),
        "retries_used": retries_used,
        "input_tokens": in_tokens,
        "output_tokens": out_tokens,
        "finish_reason": str(resp.get("finish_reason", "") or ""),
        "cost_usd": total_cost,
        "raw_reply": str(resp.get("raw_reply", "")),
    }


def _load_existing(records_path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    existing: dict[tuple[str, str], dict[str, Any]] = {}
    if not records_path.exists():
        return existing
    for line in records_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        existing[(rec["task_id"], rec["model_slug"])] = rec
    return existing


def evaluate_and_filter(
    tasks: list[TaskItem],
    records: dict[tuple[str, str], dict[str, Any]],
    models: tuple[str, ...] = MODELS,
) -> tuple[list[TaskItem], list[dict[str, Any]], dict[str, int]]:
    """Apply Step 5 exclusion rules (a) -> (b) -> (c) in exact order."""
    surviving: list[TaskItem] = []
    rows_summary: list[dict[str, Any]] = []
    stratum_counts = {"control": 0, "low_spread": 0, "high_spread": 0}

    for task in tasks:
        y_star = math.log10(float(task.numeric_truth))
        per_model = [records.get((task.id, m)) for m in models]
        if any(r is None or r.get("miss") or r.get("log10_estimate") is None for r in per_model):
            missing_models = [
                m
                for m, r in zip(models, per_model, strict=True)
                if r is None or r.get("miss") or r.get("log10_estimate") is None
            ]
            rows_summary.append(
                {
                    "task_id": task.id,
                    "status": "DROPPED",
                    "rule": "(a) parse_failure",
                    "reason": f"parse/API miss in: {', '.join(missing_models)}",
                    "selection_stratum": None,
                    "d_cold": None,
                    "e_cold": None,
                    "max_abs_err": None,
                    "per_model": {
                        m: r for m, r in zip(models, per_model, strict=True) if r is not None
                    },
                }
            )
            continue

        ests = [float(r["log10_estimate"]) for r in per_model if r is not None]
        errs = [e - y_star for e in ests]
        abs_errs = [abs(e) for e in errs]
        mean_est = statistics.fmean(ests)
        d_cold = sum((e - mean_est) ** 2 for e in ests) / len(ests)
        e_cold = (mean_est - y_star) ** 2

        if all(ae < RECALL_LOG10_THRESHOLD for ae in abs_errs):
            rows_summary.append(
                {
                    "task_id": task.id,
                    "status": "DROPPED",
                    "rule": "(b) recall_within_2pct",
                    "reason": (
                        f"all 4 models |log10 err| < {RECALL_LOG10_THRESHOLD} "
                        f"(max={max(abs_errs):.6f})"
                    ),
                    "selection_stratum": None,
                    "d_cold": d_cold,
                    "e_cold": e_cold,
                    "max_abs_err": max(abs_errs),
                    "per_model": {
                        m: r for m, r in zip(models, per_model, strict=True) if r is not None
                    },
                }
            )
            continue

        stratum = assign_stratum(d_cold)
        stratum_counts[stratum] += 1
        kept_task = task.model_copy(update={"selection_stratum": stratum})
        surviving.append(kept_task)
        rows_summary.append(
            {
                "task_id": task.id,
                "status": "KEPT",
                "rule": "(c) survived",
                "reason": f"assigned {stratum} (D_cold={d_cold:.6f})",
                "selection_stratum": stratum,
                "d_cold": d_cold,
                "e_cold": e_cold,
                "max_abs_err": max(abs_errs),
                "per_model": {
                    m: r for m, r in zip(models, per_model, strict=True) if r is not None
                },
            }
        )

    return surviving, rows_summary, stratum_counts


def _format_cell_err(per_model: dict[str, Any], slug: str) -> str:
    rec = per_model.get(slug)
    if rec is None or rec.get("miss") or rec.get("log10_error") is None:
        return "MISS"
    return f"{float(rec['log10_error']):+.5f}"


def format_markdown_report(
    rows_summary: list[dict[str, Any]],
    surviving: list[TaskItem],
    stratum_counts: dict[str, int],
    records: dict[tuple[str, str], dict[str, Any]],
    models: tuple[str, ...] = MODELS,
) -> str:
    kept_n = len(surviving)
    dropped_a = sum(1 for r in rows_summary if r["rule"] == "(a) parse_failure")
    dropped_b = sum(1 for r in rows_summary if r["rule"] == "(b) recall_within_2pct")
    total_cost = sum(
        float(r["cost_usd"]) for r in records.values() if r.get("cost_usd") is not None
    )
    total_retries = sum(int(r.get("retries_used", 0)) for r in records.values())

    lines = [
        "# Cold-Probe Results & Per-Model Errors (Step 5)",
        "",
        (
            f"- **Candidates probed:** `{len(rows_summary)}` tasks x `{len(models)}` models "
            f"(`{len(records)}` cell records, `{total_retries}` format retries)."
        ),
        (
            f'- **Prompt:** `p_series_system_prompt("{DEFAULT_CURRENT_DATE}")` + question-only '
            "user prompt (no prior estimate, no telemetry), `min_output_tokens=8192`, "
            '`reasoning={"effort": "none"}` on DeepSeek/Qwen/GLM.'
        ),
        f"- **Total probe API cost:** `${total_cost:.6f}`.",
        f"- **Dropped under rule (a) parse failure:** `{dropped_a}`",
        (
            "- **Dropped under rule (b) recall check "
            f"(`|log10 err| < {RECALL_LOG10_THRESHOLD}` for all 4 models):** `{dropped_b}`"
        ),
        f"- **Surviving tasks (`>= 25` required):** **`{kept_n}`**",
        (
            "- **Stratum counts (`D_cold` fixed midpoint cutoffs):** "
            f"`control`: `{stratum_counts['control']}`, "
            f"`low_spread`: `{stratum_counts['low_spread']}`, "
            f"`high_spread`: `{stratum_counts['high_spread']}`."
        ),
        "",
        (
            "| # | Task ID | Status | Stratum | `D_cold` | `E_cold` | Gemma err | DeepSeek err | "
            "Qwen err | GLM err | `max|err|` | Rule / Note |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for idx, item in enumerate(rows_summary, 1):
        pm = item["per_model"]
        d_str = f"{item['d_cold']:.6f}" if item["d_cold"] is not None else "NA"
        e_str = f"{item['e_cold']:.6f}" if item["e_cold"] is not None else "NA"
        m_str = f"{item['max_abs_err']:.5f}" if item["max_abs_err"] is not None else "NA"
        strat = item["selection_stratum"] or "-"
        lines.append(
            f"| {idx} | `{item['task_id']}` | {item['status']} | {strat} | {d_str} | {e_str} | "
            f"{_format_cell_err(pm, models[0])} | {_format_cell_err(pm, models[1])} | "
            f"{_format_cell_err(pm, models[2])} | {_format_cell_err(pm, models[3])} | "
            f"{m_str} | {item['reason']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None, base_backend: Any | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--tasks",
        type=Path,
        default=Path("configs/polyphony_replication_tasks_2025.json"),
    )
    ap.add_argument(
        "--filtered-out",
        type=Path,
        default=Path("configs/polyphony_replication_tasks_2025.filtered.json"),
    )
    ap.add_argument(
        "--probe-dir",
        type=Path,
        default=Path("results/polyphony-replication-2025/cold_probe"),
    )
    ap.add_argument(
        "--table-out",
        type=Path,
        default=Path("results/polyphony-replication-2025/cold_probe_errors.md"),
    )
    ap.add_argument("--current-date", default=DEFAULT_CURRENT_DATE)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--budget", type=float, default=5.0)
    ap.add_argument("--request-allowance", type=float, default=0.015369)
    args = ap.parse_args(argv)

    raw = json.loads(args.tasks.read_text(encoding="utf-8"))
    tasks = [TaskItem.model_validate(x) for x in raw]

    args.probe_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.probe_dir / "cold_probe_records.jsonl"
    existing = _load_existing(records_path)

    pending = [(task, m) for task in tasks for m in MODELS if (task.id, m) not in existing]

    if pending:
        if base_backend is None:
            base_backend = AdaptionBackend(model=MODELS[0], min_output_tokens=8192)
        configure_budget(base_backend, args.probe_dir, args.budget, args.request_allowance)
        routed = {m: base_backend.for_model(m) for m in MODELS}
        write_lock = threading.Lock()
        with (
            records_path.open("a", encoding="utf-8") as fh,
            ThreadPoolExecutor(max_workers=args.workers) as pool,
        ):
            futs = {
                pool.submit(_probe_one_cell, routed[m], task, m, args.current_date): (task.id, m)
                for task, m in pending
            }
            for fut in as_completed(futs):
                rec = fut.result()
                with write_lock:
                    existing[(rec["task_id"], rec["model_slug"])] = rec
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()

    surviving, rows_summary, stratum_counts = evaluate_and_filter(tasks, existing, MODELS)
    args.filtered_out.write_text(
        json.dumps([t.model_dump(mode="json") for t in surviving], indent=2) + "\n",
        encoding="utf-8",
    )
    summary_payload = {
        "n_candidates": len(tasks),
        "n_surviving": len(surviving),
        "stratum_counts": stratum_counts,
        "cutoffs": {"control_lt": D_CUTOFF_LOW, "high_spread_ge": D_CUTOFF_HIGH},
        "recall_log10_threshold": RECALL_LOG10_THRESHOLD,
        "rows": rows_summary,
    }
    (args.probe_dir / "cold_probe_summary.json").write_text(
        json.dumps(summary_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    md = format_markdown_report(rows_summary, surviving, stratum_counts, existing, MODELS)
    args.table_out.write_text(md, encoding="utf-8")
    print(
        f"Cold probe complete: {len(surviving)}/{len(tasks)} survived "
        f"(control={stratum_counts['control']}, low_spread={stratum_counts['low_spread']}, "
        f"high_spread={stratum_counts['high_spread']})."
    )
    if len(surviving) < 25:
        print("STOP: fewer than 25 tasks survived cold probe!", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
