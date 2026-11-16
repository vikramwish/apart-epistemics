"""Post hoc sensitivity checks from compact task-level data; entirely offline.

Reuses the frozen tie-aware exact signed-rank and Holm implementations.
No change to the preregistered analysis or its multiplicity families.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location(
    "frozen_analysis", ROOT / "scripts/analyze_controls_preregistered.py"
)
frozen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frozen)


def audit():
    data = json.loads((HERE / "task_metrics.json").read_text())
    original = json.loads((HERE / "s2_results.json").read_text())
    tasks, domains = data["tasks"], np.array(data["domains"])
    assert len(tasks) == len(set(tasks)) == 40
    assert len(domains) == 40
    assert len(set(domains)) == 15
    errors = {c: np.array(v) for c, v in data["E2"].items()}
    assert len(errors) == 15
    assert all(
        v.shape == (40,) and np.isfinite(v).all() and (v >= 0).all() for v in errors.values()
    )
    families = original["contrasts"]
    rows = []
    for family, contrasts in families.items():
        for row in contrasts:
            a, b = row["a"], row["b"]
            d = errors[b] - errors[a]
            p, n = frozen.exact_wilcoxon_p(d.tolist())
            assert n == 40
            assert abs(p - row["p_wilcoxon"]) < 1e-12
            assert abs(float(d.mean()) - row["mean"]) < 1e-12
            rows.append(
                dict(
                    family=family,
                    a=a,
                    b=b,
                    mean=float(d.mean()),
                    p=p,
                    task_signs=[],
                    domain_signs=[],
                    task_p_holm=[],
                    domain_p_holm=[],
                )
            )
    assert len(rows) == 14
    for row, p in zip(rows, frozen.holm([r["p"] for r in rows]), strict=True):
        row["p_global_holm"] = p

    for kind, exclusions in (("task", range(40)), ("domain", sorted(set(domains)))):
        for excluded in exclusions:
            keep = np.arange(40) != excluded if kind == "task" else domains != excluded
            for family in families:
                group = [r for r in rows if r["family"] == family]
                ps = []
                for row in group:
                    delta = (errors[row["b"]] - errors[row["a"]])[keep]
                    row[kind + "_signs"].append(int(np.sign(delta.mean())))
                    ps.append(frozen.exact_wilcoxon_p(delta.tolist())[0])
                for row, p in zip(group, frozen.holm(ps), strict=True):
                    row[kind + "_p_holm"].append(p)
    for row in rows:
        for kind in ("task", "domain"):
            signs = row.pop(kind + "_signs")
            ps = row.pop(kind + "_p_holm")
            row[kind + "_omission"] = {
                "subsets": len(ps),
                "same_direction": sum(int(s == np.sign(row["mean"])) for s in signs),
                "p_holm_range": [min(ps), max(ps)],
                "significant_subsets": sum(p < 0.05 for p in ps),
            }
    rng = np.random.default_rng(20261116)
    indices = rng.integers(0, 40, size=(10000, 40))
    recovery = {}
    for name, silent, social, critique in (
        ("Gemma", "P0", "P1", "P2"),
        ("Mix", "P0_POLY", "P3", "P4"),
    ):
        numerator = errors[social] - errors[critique]
        denominator = errors[social] - errors[silent]
        denom_boot = denominator[indices].mean(axis=1)
        assert (denom_boot > 0).all(), "Recovery ratio bootstrap has nonpositive denominator"
        draws = numerator[indices].mean(axis=1) / denom_boot
        recovery[name] = {
            "fraction": float(numerator.mean() / denominator.mean()),
            "ci": np.percentile(draws, [2.5, 97.5]).tolist(),
            "nonpositive_denominator_resamples": int((denom_boot <= 0).sum()),
        }
    result = dict(
        source_sha256=data["source_sha256"],
        n_tasks=40,
        n_domains=15,
        bootstrap_seed=20261116,
        bootstrap_resamples=10000,
        scope="post hoc; task estimates conditional on fixed benchmark and model runs",
        contrasts=rows,
        recovery=recovery,
    )
    (HERE / "statistical_checks.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        r"\begin{tabular}{@{}lrrr@{}}\toprule",
        r"Contrast & Global Holm & Max. LOTO Holm & Max. LODO Holm\\\midrule",
    ]
    for r in rows:
        a = r["a"].replace("_", r"\_")
        b = r["b"].replace("_", r"\_")
        values = [
            r["p_global_holm"],
            r["task_omission"]["p_holm_range"][1],
            r["domain_omission"]["p_holm_range"][1],
        ]
        rendered = [r"$<0.0001$" if v < 0.0001 else f"${v:.4f}$" for v in values]
        lines.append(a + r" $\to$ " + b + " & " + " & ".join(rendered) + r"\\")
    lines.append(r"\bottomrule\end{tabular}")
    (HERE / "sensitivity_table.tex").write_text("\n".join(lines) + "\n")
    recovery_text = [
        r"\paragraph{Recovery fraction.} We estimate the ratio of task-mean error changes,",
        r"\[R=\frac{\overline E_{\mathrm{social}}-\overline E_{\mathrm{critique}}}"
        r"{\overline E_{\mathrm{social}}-\overline E_{\mathrm{silent}}}.\]",
        r"Paired task-bootstrap percentile intervals use 10,000 draws (seed 20261116), "
        r"resampling the same tasks across all three arms. "
        r"The denominator was positive in every resample. ",
    ]
    for name, r in recovery.items():
        lo, hi = r["ci"]
        recovery_text.append(
            f"{name}: ${100 * r['fraction']:.1f}\\%$ "
            f"(95\\% CI $[{100 * lo:.1f}\\%,{100 * hi:.1f}\\%]$). "
        )
    recovery_text.append(
        r"This ratio of mean changes is not a mean of per-task ratios. "
        r"Both intervals exclude full recovery ($R=1$), conditional on the sampled benchmark."
    )
    (HERE / "recovery_summary.tex").write_text(
        "\n".join(line.rstrip() for line in recovery_text) + "\n"
    )
    print(
        json.dumps(
            {
                "recovery": recovery,
                "all_mean_directions_preserved": all(
                    r[k]["same_direction"] == r[k]["subsets"]
                    for r in rows
                    for k in ("task_omission", "domain_omission")
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    audit()
