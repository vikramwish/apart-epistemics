# Repository Summary: Apart AI × Epistemics

**Owner / Author:** Vikram Viswanath (`vikramwish`)  
**Active Branch:** `replication-2025`  
**Latest Verified Submission Artifacts:**
- Submission Paper: `reports/submission/main.pdf` (source `reports/submission/main.tex`, figures via `reports/submission/make_figures.py`)
- Audited Experiment Technical Report: `reports/POLYPHONY_EXPERIMENT_REPORT.md`
- Claim-to-Evidence Audit Table: `reports/CLAIM_EVIDENCE_TABLE.md`
- Revision Changelog & Required Controls: `reports/REVISION_CHANGELOG.md`
- Reproduction Script & Audit Outputs: `scripts/audit_and_reproduce_p0_p4.py` -> `results/polyphony-p0-p4/audit_reproduction_report.json`

## Key Verified Findings (16-Task, alpha = 0.4 Study)

- **Version Resolution:** Resolved the earlier `12`-task, `alpha = 0.3` run (`b5b91ce`, `960` retained rows, preserved untouched in `results/polyphony-p0-p4-12task-alpha03/`) from the target `16`-task, `alpha = 0.4` experiment (`fdd30fe` / `3a6294c`, `1,280` retained rows across `1,318` provider requests, `$0.486271` retained / `$0.501372` total cost, in `results/polyphony-p0-p4/`).
- **Social Monoculture Amplification (`P1 vs P0`):** Exposing homogeneous `gemma-4-26b-a4b-it` agents (`M = 8`) to the shared crowd median (`alpha = 0.4`, two rounds) nearly doubles Round-2 crowd `log10` MSE from `0.011944` (`P0`) to `0.022899` (`P1`: `+0.010955`, W/L/T `13/2/1`, exact Wilcoxon `p = 0.005371`, within-metric Holm-9 `p = 0.032227` exact / `0.049594` asymptotic), though leave-one-task-out (`n = 15`) survival under Holm-9 is `7/16` (`Drop Lake Superior`: Holm-9 `0.064453`).
- **Two Robust Mitigations vs. `P1`:**
  - Counter-anchoring prompt package (`P2 vs P1`): `-0.011081` (`14/1/1`, exact Holm-9 `p = 0.001099`, global Holm-20 `p = 0.002441`, **`16/16` LOTO survival**).
  - Multi-family panel (`P3 vs P1` and `P4 vs P1`): `-0.012841` (`12/3/1`, exact Holm-9 `p = 0.009277`, **`16/16` LOTO**) and `-0.015207` (`14/1/1`, exact Holm-9 `p = 0.009277`, **`16/16` LOTO**).
- **Sub-Additive `2 × 2` Factorial Interaction:** Across social conditions, `DiD = (P4 - P3) - (P2 - P1) = +0.008715` (W/L/T `1/14/1`, exact `p = 0.000183`, Holm-5 `p = 0.000916`, **`16/16` LOTO survival**, and surviving both first-attempt reconstructions), establishing that counter-anchoring and model mixing act as partial substitutes rather than additive complements.
- **Explicit Boundaries & Non-Significant Comparisons:**
  - **`P4 vs P3` is NOT significant after multiplicity correction** (`-0.002366`, raw exact `p = 0.025574`, Holm-9 exact `p = 0.127869`, Holm-20 `p = 0.383606`), and **`P4 vs P0` is null** (`-0.004252`, `p = 0.3894`, `65.44%` driven by `hyd_lake_superior_vol_km3`).
  - `P3`/`P4 vs P1` conflates model diversity with constituent capability (`deepseek-v4-flash` and `glm-5.2` have ~`10x`–`40x` lower in-condition assistant MSE than `gemma-4-26b-a4b-it`), so existing arms cannot isolate diversity from capability.

## Controls and Out-of-Sample Replication (Post-Sprint)

The follow-up controls listed as unrun in the 16-task paper (`P0_POLY`, `P1_REFLECT`,
homogeneous DeepSeek/Qwen/GLM arms) were run on the 16 tasks
(`results/polyphony-controls/`, analysis pre-registered in
`docs/PREREGISTRATION_controls.md`) and then replicated on `40` post-cutoff tasks
x `3` replicates (`24,960` cells, `$17.28`; pre-registration
`docs/PREREGISTRATION_replication_2025.md`). Full results:
[`results/polyphony-replication-2025/REPORT.md`](../results/polyphony-replication-2025/REPORT.md).

- Gemma herding (`P1 > P0`) and counter-anchoring rescue (`P2 < P1`) replicate.
- Herding generalizes to all `3/3` other model families; two-sided critique beats
  length-matched reflection (`p_holm = 0.00048`); `P4 < P3` is now significant
  (`p_holm = 0.000904`).
- Boundaries: silent `P0_POLY` beats both social mixed panels (`P3 +64.0%`,
  `P4 +40.6%` E2), and `P4` still does not beat silent Gemma `P0` (`p_holm = 0.6368`).
