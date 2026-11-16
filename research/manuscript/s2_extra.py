"""Extra Study 2 analyses from the raw log (argv[1]) and cold-probe records (argv[2])."""

import gzip
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
PROBE = (
    Path(sys.argv[2])
    if len(sys.argv) > 2
    else ROOT / "results/polyphony-replication-2025/cold_probe/cold_probe_records.jsonl"
)
OUT = Path(__file__).with_name("s2_extra.json")
ALPHA = 0.4

cells = defaultdict(dict)
truth, domain = {}, {}
with gzip.open(LOG, "rt") if LOG.suffix == ".gz" else LOG.open() as f:
    for line in f:
        d = json.loads(line)
        spec = d.pop("run_spec")
        d.pop("raw_reply", None)
        if (d["round_idx"], d["agent_id"]) in cells[
            (d["condition_id"], d["question_id"], d["replicate"])
        ]:
            raise ValueError("Duplicate cell in frozen run log")
        cells[(d["condition_id"], d["question_id"], d["replicate"])][
            (d["round_idx"], d["agent_id"])
        ] = d
        if not truth:
            for t in spec["tasks"]:
                truth[t["id"]] = math.log10(t["numeric_truth"])
                domain[t["id"]] = t["domain"]
tasks = sorted(truth)
reps = sorted({k[2] for k in cells})
conds = sorted({k[0] for k in cells})
if len(cells) != 13 * 40 * 3 or any(len(c) != 16 for c in cells.values()):
    raise ValueError("Expected all 24,960 cells of the completed replication")


def cell_stats(cond, q, r):
    c = cells[(cond, q, r)]
    ag = sorted({a for (_, a) in c})
    x0 = np.array([c[(1, a)]["agent_log10_before"] for a in ag])
    x1 = np.array([c[(1, a)]["agent_log10_after"] for a in ag])
    x2 = np.array([c[(2, a)]["agent_log10_after"] for a in ag])
    a1 = (x1 - (1 - ALPHA) * x0) / ALPHA
    a2 = (x2 - (1 - ALPHA) * x1) / ALPHA
    y = truth[q]
    return dict(E0=(x0.mean() - y) ** 2, E2=(x2.mean() - y) ** 2, sd_a1=a1.std(), sd_a2=a2.std())


T = {
    c: {
        k: np.array([np.mean([cell_stats(c, q, r)[k] for r in reps]) for q in tasks])
        for k in ("E0", "E2", "sd_a1", "sd_a2")
    }
    for c in conds
}

# 1) Effective assistant-answer spread; misses carry the prior forward.
spread = {
    c: dict(sd_a1=float(T[c]["sd_a1"].mean()), sd_a2=float(T[c]["sd_a2"].mean())) for c in conds
}
spread_tests = {}
for a, b in [
    ("P0", "P1"),
    ("P1", "P2"),
    ("P0_DS", "P1_DS"),
    ("P0_QWEN", "P1_QWEN"),
    ("P0_GLM", "P1_GLM"),
    ("P0_POLY", "P3"),
]:
    d = T[b]["sd_a2"] - T[a]["sd_a2"]
    spread_tests[f"{a}->{b}"] = dict(
        rel=float(d.mean() / T[a]["sd_a2"].mean()),
        lower=int((d < 0).sum()),
        p=float(stats.wilcoxon(d, method="exact").pvalue),
    )

# 2) domain-clustered bootstrap for headline contrasts
doms = sorted(set(domain.values()))
by_dom = {g: [i for i, q in enumerate(tasks) if domain[q] == g] for g in doms}
rng = np.random.default_rng(42)
clust = {}
for a, b in [
    ("P0", "P1"),
    ("P1", "P2"),
    ("P0_DS", "P1_DS"),
    ("P0_QWEN", "P1_QWEN"),
    ("P0_GLM", "P1_GLM"),
    ("P0_POLY", "P3"),
    ("P3", "P4"),
    ("P1_REFLECT", "P2"),
]:
    d = T[b]["E2"] - T[a]["E2"]
    boots = []
    for _ in range(10000):
        pick = rng.choice(doms, size=len(doms), replace=True)
        idx = [i for g in pick for i in by_dom[g]]
        boots.append(d[idx].mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    clust[f"{a}->{b}"] = dict(mean=float(d.mean()), ci=[float(lo), float(hi)])

# 3) tasks where the crowd ends worse than it started
worse_than_start = {c: int((T[c]["E2"] > T[c]["E0"]).sum()) for c in conds}

# 4) exploratory: does herding or recovery track task difficulty (cold-probe error)?
cold = defaultdict(dict)
for line in PROBE.read_text().splitlines():
    r = json.loads(line)
    cold[r["task_id"]][r["model_slug"]] = r["abs_log10_error"]
g_cold = np.array([cold[q].get("gemma-4-26b-a4b-it", np.nan) for q in tasks])
herd = T["P1"]["E2"] - T["P0"]["E2"]
rescue = T["P1"]["E2"] - T["P2"]["E2"]
frac = np.where(herd > 1e-6, rescue / herd, np.nan)
ok = ~np.isnan(g_cold)
explor = dict(
    n=int(ok.sum()),
    rho_cold_vs_herding=list(map(float, stats.spearmanr(g_cold[ok], herd[ok]))),
    rho_cold_vs_rescue=list(map(float, stats.spearmanr(g_cold[ok], rescue[ok]))),
    rho_cold_vs_recovery_fraction=list(
        map(float, stats.spearmanr(g_cold[ok & ~np.isnan(frac)], frac[ok & ~np.isnan(frac)]))
    ),
    n_domains=len(doms),
)

OUT.write_text(
    json.dumps(
        dict(
            spread=spread,
            spread_tests=spread_tests,
            domain_cluster_ci=clust,
            worse_than_start=worse_than_start,
            exploratory=explor,
            domains={g: len(v) for g, v in by_dom.items()},
        ),
        indent=1,
    )
    + "\n"
)

print(
    json.dumps(
        dict(
            spread_tests=spread_tests,
            worse_than_start=worse_than_start,
            exploratory=explor,
            domain_cluster_ci=clust,
        ),
        indent=1,
    )
)
