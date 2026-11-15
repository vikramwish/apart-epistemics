"""Tests for pre-registered controls + replication analysis extensions (Step 3)."""

from __future__ import annotations

import importlib.util
import json
import math
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = [
    "google/gemma-4-26b-a4b-it",
    "deepseek/deepseek-v4-flash",
    "qwen/qwen3.8-27b",
    "z-ai/glm-5.2",
]
COND_MODEL = {
    "P0": [0] * 8,
    "P1": [0] * 8,
    "P2": [0] * 8,
    "P1_REFLECT": [0] * 8,
    "P0_DS": [1] * 8,
    "P1_DS": [1] * 8,
    "P0_QWEN": [2] * 8,
    "P1_QWEN": [2] * 8,
    "P0_GLM": [3] * 8,
    "P1_GLM": [3] * 8,
    "P3": [i % 4 for i in range(8)],
    "P4": [i % 4 for i in range(8)],
    "P0_POLY": [i % 4 for i in range(8)],
}
SOCIAL = {"P1", "P2", "P1_REFLECT", "P1_DS", "P1_QWEN", "P1_GLM", "P3", "P4"}
HERD = {MODELS[0]: 0.5, MODELS[1]: 0.2, MODELS[2]: 0.35, MODELS[3]: 0.05}
ALL_CONDS = [
    "P0",
    "P1",
    "P2",
    "P3",
    "P4",
    "P0_POLY",
    "P1_REFLECT",
    "P0_DS",
    "P1_DS",
    "P0_QWEN",
    "P1_QWEN",
    "P0_GLM",
    "P1_GLM",
]


def _load_acp():
    path = ROOT / "scripts" / "analyze_controls_preregistered.py"
    spec = importlib.util.spec_from_file_location("analyze_controls_preregistered", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _generate_synthetic_null(
    path: Path,
    *,
    seed: int,
    n_tasks: int,
    replicates: int,
    drop_cell: tuple[str, str, int, int, str] | None = None,
) -> None:
    tasks = [{"id": f"t{i}", "numeric_truth": 10 ** (2 + i * 0.15)} for i in range(n_tasks)]
    fam_rng = random.Random(seed * 17 + 1)
    fam_bias = {
        m: [fam_rng.gauss(0, 0.08 if j == 0 else 0.03) for _ in range(n_tasks)]
        for j, m in enumerate(MODELS)
    }
    base_spec = {
        "seed": 42,
        "alpha": 0.4,
        "n_agents": 8,
        "n_rounds": 2,
        "replicates": replicates,
        "current_date": "2026-11-15",
        "tasks": tasks,
        "conditions": ALL_CONDS,
    }
    rng = random.Random(seed)
    lines = []
    for cond in ALL_CONDS:
        for ti, task in enumerate(tasks):
            y = math.log10(task["numeric_truth"])
            b = 0.176 * (1 if ti % 2 else -1)
            for rep in range(replicates):
                pr = random.Random(42 * 313 + sum(ord(c) for c in task["id"]) * 19 + rep * 7)
                x = [y + b + pr.gauss(0, 0.24) for _ in range(8)]
                for r in (1, 2):
                    med = statistics.median(x)
                    nxt = []
                    for a in range(8):
                        m = MODELS[COND_MODEL[cond][a]]
                        h = HERD[m] if cond in SOCIAL else 0.0
                        if cond in ("P2", "P4"):
                            h *= 0.2
                        if cond == "P1_REFLECT":
                            h *= 0.6
                        ans = y + fam_bias[m][ti] + h * (med - y) + rng.gauss(0, 0.05)
                        after = 0.6 * x[a] + 0.4 * ans
                        out_toks = 300 if cond in ("P2", "P4", "P1_REFLECT") else 400
                        if drop_cell != (cond, task["id"], rep, r, f"agent_{a}"):
                            lines.append(
                                json.dumps(
                                    {
                                        "run_spec": base_spec,
                                        "condition_id": cond,
                                        "question_id": task["id"],
                                        "replicate": rep,
                                        "round_idx": r,
                                        "agent_id": f"agent_{a}",
                                        "model_slug": m,
                                        "assistant_log10": ans,
                                        "agent_log10_before": x[a],
                                        "agent_log10_after": after,
                                        "miss": False,
                                        "output_tokens": out_toks,
                                    }
                                )
                            )
                        nxt.append(after)
                    x = nxt
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_secondary_abs_log_error_replication_contrasts_and_replicates_with_missing_cell(
    tmp_path: Path,
) -> None:
    acp = _load_acp()
    in_jsonl = tmp_path / "sim_r3.jsonl"
    out_json = tmp_path / "analysis.json"
    _generate_synthetic_null(
        in_jsonl,
        seed=777,
        n_tasks=10,
        replicates=3,
        drop_cell=("P1_DS", "t0", 1, 2, "agent_1"),
    )
    assert acp.main([str(in_jsonl), "--out", str(out_json)]) == 0
    res = json.loads(out_json.read_text(encoding="utf-8"))

    assert res["spec_audit"]["pipeline_merge_will_succeed"] is True
    assert res["replicate_counts"]["P1_DS"] == {"tasks": 10, "complete_task_replicates": 29}
    assert res["replicate_counts"]["VIRTUAL_SOCIAL"] == {
        "tasks": 10,
        "complete_task_replicates": 29,
    }
    assert res["virtual_panel_problems"]["social"] == ["missing P1_DS/t0/rep1/round2/agent_1"]

    # Primary E2 replication Holm m=4 family present alongside original m=8, m=2, descriptive.
    assert [r["name"] for r in res["replication_p_series_holm_m4"]] == [
        "p1_vs_p0",
        "p2_vs_p1",
        "p4_vs_p3",
        "p4_vs_p0",
    ]

    # Secondary |xbar_2 - y*| reports exact same families and contrasts separately.
    sec = res["secondary_metric_abs_log_error"]
    assert sec["metric_definition"] == "|xbar_2 - y*|"
    assert [r["name"] for r in sec["confirmatory_holm_m8"]] == [
        r["name"] for r in res["confirmatory_holm_m8"]
    ]
    assert [r["name"] for r in sec["secondary_holm_m2"]] == [
        r["name"] for r in res["secondary_holm_m2"]
    ]
    assert [r["name"] for r in sec["descriptive"]] == [r["name"] for r in res["descriptive"]]
    assert [r["name"] for r in sec["replication_p_series_holm_m4"]] == [
        "p1_vs_p0",
        "p2_vs_p1",
        "p4_vs_p3",
        "p4_vs_p0",
    ]
    # Jensen bound checks linear |xbar - y| and E2 match positive means.
    assert sec["mean_by_condition"]["P0"] > res["mean_crowd_error"]["P0"]


def test_virtual_panel_null_false_positive_rate_20_seeds(tmp_path: Path) -> None:
    acp = _load_acp()
    fp_e2 = 0
    fp_abs = 0
    for s in range(1, 21):
        p = tmp_path / f"null_s{s}.jsonl"
        out = tmp_path / f"null_s{s}.json"
        _generate_synthetic_null(p, seed=s * 100, n_tasks=16, replicates=3)
        assert acp.main([str(p), "--out", str(out)]) == 0
        res = json.loads(out.read_text(encoding="utf-8"))
        e2_vs = next(
            r for r in res["confirmatory_holm_m8"] if r["name"] == "p3_vs_virtual_social"
        )
        abs_vs = next(
            r
            for r in res["secondary_metric_abs_log_error"]["confirmatory_holm_m8"]
            if r["name"] == "p3_vs_virtual_social"
        )
        fp_e2 += int(e2_vs["p_exact"] < 0.05)
        fp_abs += int(abs_vs["p_exact"] < 0.05)
    assert fp_e2 <= 2
    assert fp_abs <= 2
