"""Study 1 figure inputs derived from the frozen raw log, without rounded tables."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CONDITIONS = ["P0", "P1", "P2", "P3", "P4"]
cells = defaultdict(list)
model_errors = defaultdict(list)
truth = {}
with (ROOT / "results/polyphony-p0-p4/numeric_runs.jsonl").open() as stream:
    for line in stream:
        record = json.loads(line)
        if not truth:
            truth = {
                task["id"]: np.log10(task["numeric_truth"]) for task in record["run_spec"]["tasks"]
            }
        condition, task = record["condition_id"], record["question_id"]
        cells[condition, task, record["round_idx"]].append(record)
        model_errors[condition, record["model_slug"]].append(
            (record["assistant_log10"] - truth[task]) ** 2
        )

if len(cells) != 160 or any(len(rows) != 8 for rows in cells.values()):
    raise ValueError("Study 1 must contain 16 tasks \u00d7 5 arms \u00d7 2 rounds \u00d7 8 agents")

max_diversity_residual = 0.0
for (_, task, _), rows in cells.items():
    estimates = np.array([r["agent_log10_after"] for r in rows])
    centre = estimates.mean()
    groups = defaultdict(list)
    for row in rows:
        groups[row["model_slug"]].append(row["agent_log10_after"])
    within = sum(sum((v - np.mean(g)) ** 2 for v in g) for g in groups.values()) / 8
    between = sum(len(g) * (np.mean(g) - centre) ** 2 for g in groups.values()) / 8
    individual_error = np.mean((estimates - truth[task]) ** 2)
    crowd_error = (centre - truth[task]) ** 2
    residual = abs(crowd_error - (individual_error - within - between))
    max_diversity_residual = max(max_diversity_residual, residual)
if max_diversity_residual > 1e-12:
    raise ValueError(f"Study 1 hierarchical diversity identity failed: {max_diversity_residual}")

names = sorted(truth)
d = np.empty((len(names), len(CONDITIONS)))
T4 = np.zeros((5, 3))
for j, condition in enumerate(CONDITIONS):
    for i, task in enumerate(names):
        rows = cells[condition, task, 2]
        x1 = np.mean([r["agent_log10_before"] for r in rows])
        x2 = np.mean([r["agent_log10_after"] for r in rows])
        a2 = np.mean([r["assistant_log10"] for r in rows])
        y = truth[task]
        d[i, j] = (x2 - y) ** 2
        terms = np.array([0.36 * (x1 - y) ** 2, 0.16 * (a2 - y) ** 2, 0.48 * (x1 - y) * (a2 - y)])
        if abs(terms.sum() - d[i, j]) > 1e-12:
            raise ValueError("Study 1 error decomposition failed")
        T4[j] += terms / len(names)

models = ["gemma-4-26b-a4b-it", "qwen3.8-27b", "glm-5.2", "deepseek-v4-flash"]
p3 = [np.mean(model_errors["P3", model]) for model in models]
p4 = [np.mean(model_errors["P4", model]) for model in models]
baseline_asst = np.mean(model_errors["P0", models[0]])

if __name__ == "__main__":
    print(dict(zip(CONDITIONS, d.mean(axis=0), strict=True)))
    print({"diversity_identity_panels": len(cells), "max_residual": max_diversity_residual})
