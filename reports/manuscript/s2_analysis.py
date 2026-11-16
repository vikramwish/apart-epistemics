"""Independent re-analysis of Study 2 from the raw run log.

Reads numeric_runs.jsonl or .jsonl.gz (optional path as argv[1]) and writes s2_results.json next to
this script. Uses only numpy/scipy and the standard library; no project code.
"""

import gzip
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
LOG = (
    Path(sys.argv[1])
    if len(sys.argv) > 1
    else ROOT / "results/polyphony-replication-2025/numeric_runs.jsonl"
)
OUT = Path(__file__).with_name("s2_results.json")
ALPHA = 0.4
RNG_SEED = 42
N_BOOT = 10000
N_PERM = 200000

# ---------------------------------------------------------------- load
cells = defaultdict(dict)  # (cond, task, rep) -> {(round, agent): record}
truth = {}
domains = {}
models = defaultdict(dict)  # cond -> agent -> model
with gzip.open(LOG, "rt") if LOG.suffix == ".gz" else LOG.open() as f:
    for line in f:
        d = json.loads(line)
        spec = d.pop("run_spec")
        d.pop("raw_reply", None)
        key = (d["condition_id"], d["question_id"], d["replicate"])
        if (d["round_idx"], d["agent_id"]) in cells[key]:
            raise ValueError("Duplicate cell in frozen run log")
        cells[key][(d["round_idx"], d["agent_id"])] = d
        models[d["condition_id"]][d["agent_id"]] = d["model_slug"]
        if not truth:
            for t in spec["tasks"]:
                truth[t["id"]] = math.log10(t["numeric_truth"])
                domains[t["id"]] = t["domain"]

conds = sorted({k[0] for k in cells})
tasks = sorted(truth)
reps = sorted({k[2] for k in cells})
if len(cells) != 13 * 40 * 3 or any(len(c) != 16 for c in cells.values()):
    raise ValueError("Expected all 24,960 cells of the completed replication")


def traj(cond, task, rep, drop_miss=False):
    """Return per-agent arrays x0, x1, a2_eff, x2 for one task-replicate."""
    c = cells[(cond, task, rep)]
    agents = sorted({a for (_, a) in c})
    x0 = np.array([c[(1, a)]["agent_log10_before"] for a in agents])
    x1 = np.array([c[(1, a)]["agent_log10_after"] for a in agents])
    x2 = np.array([c[(2, a)]["agent_log10_after"] for a in agents])
    a2 = (x2 - (1 - ALPHA) * x1) / ALPHA  # effective assistant answer (miss -> carry-forward)
    miss = any(c[(r, a)]["miss"] for r in (1, 2) for a in agents)
    if drop_miss and miss:
        return None
    return dict(agents=agents, x0=x0, x1=x1, a2=a2, x2=x2, miss=miss)


def metrics(t, y):
    xb0, xb1, xb2, ab2 = t["x0"].mean(), t["x1"].mean(), t["x2"].mean(), t["a2"].mean()
    D1 = ((t["x1"] - xb1) ** 2).mean()
    D2 = ((t["x2"] - xb2) ** 2).mean()
    return dict(
        E0=(xb0 - y) ** 2,
        E1=(xb1 - y) ** 2,
        E2=(xb2 - y) ** 2,
        abs2=abs(xb2 - y),
        carry=0.36 * (xb1 - y) ** 2,
        asst=0.16 * (ab2 - y) ** 2,
        samedir=0.48 * (xb1 - y) * (ab2 - y),
        evrr=D2 / ((1 - ALPHA) ** 2 * D1) if D1 > 0 else float("nan"),
        miss_pct=10 ** abs(xb2 - y) - 1,
    )


def task_level(cond, drop_miss=False):
    """Replicate-averaged metrics per task: dict metric -> array over tasks."""
    out = defaultdict(list)
    for q in tasks:
        per = [metrics(t, truth[q]) for r in reps if (t := traj(cond, q, r, drop_miss)) is not None]
        if not per:
            raise ValueError(f"No valid replicates remain for {cond}/{q}")
        for k in per[0]:
            out[k].append(float(np.nanmean([p[k] for p in per])))
    return {k: np.array(v) for k, v in out.items()}


TL = {c: task_level(c) for c in conds}
residual = max(
    abs(m["carry"] + m["asst"] + m["samedir"] - m["E2"])
    for c in conds
    for q in tasks
    for r in reps
    for m in [metrics(traj(c, q, r), truth[q])]
)
if residual > 1e-12:
    raise ValueError(f"Error identity failed: {residual}")


# ------------------------------------------------- seat-matched virtual social panel
def virtual(mixed_cond, k1_arm_for_model):
    """E2 per task for a panel built seat-by-seat from single-model arms."""
    vals = []
    for q in tasks:
        per = []
        for r in reps:
            seat_models = models[mixed_cond]
            x2 = []
            for a, m in sorted(seat_models.items()):
                arm = k1_arm_for_model[m]
                x2.append(cells[(arm, q, r)][(2, a)]["agent_log10_after"])
            for a, m in sorted(seat_models.items()):
                source = cells[(k1_arm_for_model[m], q, r)][(1, a)]
                target = cells[(mixed_cond, q, r)][(1, a)]
                if abs(source["agent_log10_before"] - target["agent_log10_before"]) > 1e-12:
                    raise ValueError("Virtual panel priors do not match")
            per.append((np.mean(x2) - truth[q]) ** 2)
        vals.append(np.mean(per))
    return np.array(vals)


SOCIAL = {
    "gemma-4-26b-a4b-it": "P1",
    "deepseek-v4-flash": "P1_DS",
    "qwen3.8-27b": "P1_QWEN",
    "glm-5.2": "P1_GLM",
}
SILENT = {
    "gemma-4-26b-a4b-it": "P0",
    "deepseek-v4-flash": "P0_DS",
    "qwen3.8-27b": "P0_QWEN",
    "glm-5.2": "P0_GLM",
}
VIRT_SOCIAL = virtual("P3", SOCIAL)
VIRT_SILENT = virtual("P0_POLY", SILENT)

# ------------------------------------------------- contrast statistics
rng = np.random.default_rng(RNG_SEED)
boot_idx = rng.integers(0, len(tasks), size=(N_BOOT, len(tasks)))
signs = rng.choice([-1.0, 1.0], size=(N_PERM, len(tasks)))


def contrast(a, b):
    """b - a per task: mean, paired bootstrap 95% CI, sign-flip permutation p, Wilcoxon p."""
    d = b - a
    boots = d[boot_idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    obs = abs(d.mean())
    perm = np.abs((signs * d).mean(axis=1))
    p_perm = (1 + (perm >= obs - 1e-15).sum()) / (N_PERM + 1)
    p_wil = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided", method="exact").pvalue
    return dict(
        mean=float(d.mean()),
        rel=float(d.mean() / a.mean()),
        ci=[float(lo), float(hi)],
        better=int((d < 0).sum()),
        worse=int((d > 0).sum()),
        ties=int((d == 0).sum()),
        p_perm=float(p_perm),
        p_wilcoxon=float(p_wil),
    )


def holm(ps):
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * ps[i]))
        adj[i] = run
    return adj.tolist()


E = {c: TL[c]["E2"] for c in conds}
E["VIRTUAL_SOCIAL"] = VIRT_SOCIAL
E["VIRTUAL_SILENT"] = VIRT_SILENT
# Compact, full-precision analysis inputs let readers reproduce task-level
# contrasts without downloading the repeated prompts in the 637 MB raw log.
with LOG.open("rb") as source:
    source_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
Path(__file__).with_name("task_metrics.json").write_text(
    json.dumps(
        {
            "source_file": LOG.name,
            "source_sha256": source_sha256,
            "unit": "task; mean of three replicate squared log10 crowd errors",
            "tasks": tasks,
            "domains": [domains[q] for q in tasks],
            "E2": {c: values.tolist() for c, values in E.items()},
        },
        indent=2,
    )
    + "\n"
)
families = {
    "pseries": [("P0", "P1"), ("P1", "P2"), ("P3", "P4"), ("P0", "P4")],
    "controls": [
        ("P0_DS", "P1_DS"),
        ("P0_QWEN", "P1_QWEN"),
        ("P0_GLM", "P1_GLM"),
        ("P0_POLY", "P3"),
        ("P0_POLY", "P4"),
        ("P1", "P1_REFLECT"),
        ("P1_REFLECT", "P2"),
        ("VIRTUAL_SOCIAL", "P3"),
    ],
    "secondary": [("P1_DS", "P3"), ("P1_DS", "P4")],
}
results = {}
for fam, pairs in families.items():
    rows = [dict(a=a, b=b, **contrast(E[a], E[b])) for a, b in pairs]
    for key in ("p_wilcoxon", "p_perm"):
        for r, h in zip(rows, holm([r[key] for r in rows]), strict=True):
            r[key + "_holm"] = h
    results[fam] = rows

# ------------------------------------------------- parse-miss sensitivity
TLd = {c: task_level(c, drop_miss=True) for c in conds}
sens = []
for pairs in families.values():
    for a, b in pairs:
        if a.startswith("VIRTUAL"):
            continue
        r = contrast(TLd[a]["E2"], TLd[b]["E2"])
        sens.append(dict(a=a, b=b, mean=r["mean"], p_wilcoxon=r["p_wilcoxon"]))
n_miss = sum(1 for k in cells if traj(*k)["miss"])

# ------------------------------------------------- arm summaries (incl. GLM)
arms = {}
for c in conds:
    t = TL[c]
    bt = t["miss_pct"][boot_idx].mean(axis=1)
    arms[c] = dict(
        E0=float(t["E0"].mean()),
        E2=float(t["E2"].mean()),
        abs2=float(t["abs2"].mean()),
        miss_pct=float(t["miss_pct"].mean() * 100),
        miss_ci=[float(np.percentile(bt, 2.5) * 100), float(np.percentile(bt, 97.5) * 100)],
        carry=float(t["carry"].mean()),
        asst=float(t["asst"].mean()),
        samedir=float(t["samedir"].mean()),
        evrr=float(np.nanmean(t["evrr"])),
    )

OUT.write_text(
    json.dumps(
        dict(
            n_cells=sum(len(v) for v in cells.values()),
            n_tasks=len(tasks),
            n_task_reps_with_miss=n_miss,
            max_identity_residual=residual,
            contrasts=results,
            parse_miss_sensitivity=sens,
            arms=arms,
        ),
        indent=1,
    )
    + "\n"
)

print("wrote", OUT)
