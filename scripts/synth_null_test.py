import json
import math
import random
import statistics
import sys
from pathlib import Path

out = Path(sys.argv[1])
out.mkdir(exist_ok=True)
tasks = [{"id": f"t{i}", "numeric_truth": 10 ** (1 + i * 0.3)} for i in range(16)]
models = [
    "google/gemma-4-26b-a4b-it",
    "deepseek/deepseek-v4-flash",
    "qwen/qwen3.8-27b",
    "z-ai/glm-5.2",
]
fam_bias = {
    m: [random.Random(hash((m, i)) % 999).gauss(0, 0.08 if j == 0 else 0.03) for i in range(16)]
    for j, m in enumerate(models)
}
herd = {models[0]: 0.5, models[1]: 0.2, models[2]: 0.35, models[3]: 0.05}
cond_model = {
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
social = {"P1", "P2", "P1_REFLECT", "P1_DS", "P1_QWEN", "P1_GLM", "P3", "P4"}


def run(conds, spec, path, seed):
    rng = random.Random(seed)
    lines = []
    for c in conds:
        for ti, t in enumerate(tasks):
            y = math.log10(t["numeric_truth"])
            b = 0.176 * (1 if ti % 2 else -1)
            pr = random.Random(42 * 313 + ti)
            x = [y + b + pr.gauss(0, 0.24) for _ in range(8)]
            for r in (1, 2):
                med = statistics.median(x)
                new = []
                for a in range(8):
                    m = models[cond_model[c][a]]
                    h = herd[m] if c in social else 0
                    if c in ("P2", "P4"):
                        h *= 0.2
                    if c == "P1_REFLECT":
                        h *= 0.6
                    ans = y + fam_bias[m][ti] + h * (med - y) + rng.gauss(0, 0.05)
                    after = 0.6 * x[a] + 0.4 * ans
                    lines.append(
                        {
                            "run_spec": {**spec, "conditions": conds},
                            "condition_id": c,
                            "question_id": t["id"],
                            "replicate": 0,
                            "round_idx": r,
                            "agent_id": f"agent_{a}",
                            "model_slug": m,
                            "assistant_log10": ans,
                            "agent_log10_before": x[a],
                            "agent_log10_after": after,
                            "miss": False,
                            "output_tokens": 300 if c in ("P2", "P4", "P1_REFLECT") else 400,
                        }
                    )
                    new.append(after)
                x = new
    path.write_text("\n".join(json.dumps(l) for l in lines) + "\n")


base = {
    "seed": 42,
    "alpha": 0.4,
    "n_agents": 8,
    "n_rounds": 2,
    "current_date": "2026-11-15",
    "tasks": tasks,
    "backend": "openrouter",
}
run(["P0", "P1", "P2", "P3", "P4"], base, out / "base.jsonl", 1)
run(
    ["P0_POLY", "P1_REFLECT", "P0_DS", "P1_DS", "P0_QWEN", "P1_QWEN", "P0_GLM", "P1_GLM"],
    base,
    out / "ctrl.jsonl",
    2,
)
run(["P0_POLY"], {**base, "current_date": "2026-11-16"}, out / "ctrl_baddate.jsonl", 3)
