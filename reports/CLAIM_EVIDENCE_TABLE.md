# Claim-to-Evidence Audit Table (`polyphony-p0-p4`, 16-Task, α = 0.4)

**Reproduction Command (verifies every quantitative entry below end-to-end):**
```bash
PYTHONPATH=src .venv/bin/python scripts/audit_and_reproduce_p0_p4.py
```
**Primary Artifact Files:**
- `results/polyphony-p0-p4/numeric_runs.jsonl` (`1,280` retained rows)
- `results/polyphony-p0-p4/provider_requests.jsonl` (`1,318` total logged API requests)
- `results/polyphony-p0-p4/numeric_analysis.json` (`exact_reproduction_matches_saved_numeric_analysis_json = true`, max residual `1.3877787807814457e-16`)
- `results/polyphony-p0-p4/audit_reproduction_report.json` & `results/polyphony-p0-p4/first_attempt_reconstruction.json`
- `results/polyphony-p0-p4-12task-alpha03/` (preserved `12`-task, `alpha=0.3` archive from commit `b5b91ce`)

---

## 1. Provenance, Versioning, Cost & Sampling Claims

| Claim | Status | Exact Evidence Path & Verified Value | Audit Notes / Revision |
|---|---|---|---|
| Target dataset has 16 tasks, 5 conditions, 8 agents, 2 rounds, 1 replicate, `alpha=0.4`, `1,280` retained rows | **VERIFIED** | `configs/polyphony_p0_p4.yaml`; `audit_reproduction_report.json["provenance"]` (`n_tasks=16, n_retained_rows=1280`) | Exact match (`max_identity_residual = 1.3878e-16`). |
| Earlier `12`-task `alpha=0.3` run existed (`960` rows, `979` requests, commit `b5b91ce`) | **VERIFIED & SEPARATED** | `results/polyphony-p0-p4-12task-alpha03/` (`P0=0.009932..P4=0.009928`) | Preserved cleanly alongside `results/polyphony-p0-p4/`; never conflated. |
| "Preregistered" experimental design / primary contrasts | **INVALIDATED -> REVISED** | `src/apart_epistemics/metrics.py` (`b5b91ce`, `2026-11-15 21:51`); no external preregistration URI | Removed `"preregistered"`; replaced with `"pre-specified in the analysis codebase prior to 16-task data collection (commit b5b91ce)"`. |
| "48/48/48/48" observations per model in `P3`/`P4` | **INVALIDATED -> FIXED** | `audit_reproduction_report.json["per_model_breakdown"]`: `n_total = 64` (`32 R1 + 32 R2`) | Corrected to **`64` retained observations per model** (`16 tasks × 2 agents × 2 rounds`). |
| Model MSE in `P3`/`P4` reflects "standalone" accuracy | **INVALIDATED -> FIXED** | `results/polyphony-p0-p4/numeric_runs.jsonl` (`pop_median_log10` & prior round context present) | Relabeled as **in-condition assistant accuracy**; true standalone arms missing for 3 frontier models. |
| API cost `$0.486` is total experiment spend with zero retries/truncations | **REVISED / QUALIFIED** | `audit_reproduction_report.json["provenance"]["provider_requests_audit"]` | Retained `1,280` rows cost **`$0.486271`** (`229,653` in / `531,150` out); all `1,318` logged calls cost **`$0.501372`** (`237,144` in / `654,919` out); `14` hit `8192` token cap (`6` loop replacements [`28+1` cells], `8` auto-retried, `2` resample failures, `1` `0.26` unit-slip auto-retry). |
| Independent verification / external co-authorship claims | **UNREPRODUCED -> ALIGNED** | Single-author git log + `audit_reproduction_report.json` | Single author (`Vikram Viswanath`, Independent Researcher, Apart AI × Epistemics Sprint); verification rests strictly on deterministic automated scripts and tests. |

---

## 2. Core Quantitative & Inferential Claims (`N = 16` Task Units)

| Claim | Status | Exact Evidence Path & Verified Value | Audit Notes / Revision |
|---|---|---|---|
| **Social Monoculture (`P1 > P0`):** Homogeneous social sharing increases R2 crowd `log10` MSE | **VERIFIED (LOTO-sensitive under Holm-9)** | `audit_reproduction_report.json["exact_vs_pipeline_wilcoxon_matrix"]["crowd_error"]`: `P0=0.011944 -> P1=0.022899`, diff `+0.010955`, W/L/T `2/13/1` (`13` worse, `2` better, `1` tie), `p_ex=0.005371`, `H9e=0.032227`, `H9n=0.049594` | Survives `H9e < 0.05` in `7/16` LOTO `n=15` subsets (`Drop Lake Superior`: `p_ex=0.010742`, `H9e=0.064453`); sensitive to unguarded `req 648` unit slip (`0.26`) in `First_Call_Ever_Strict`. |
| **Counter-Anchoring (`P2 < P1`):** Prompt package reduces homogeneous social error | **VERIFIED (HIGHLY ROBUST)** | Same matrix: `P2=0.011818`, diff `-0.011081`, CI `[-0.020269, -0.004246]`, W/L/T `14/1/1`, `p_ex=0.000122`, `H9e=0.001099`, `H20e=0.002441` | Survives **`16/16` LOTO subsets**, `First_Pass_Saved_Pre2328` (`H9e=0.023499`), `First_Call_Ever_Strict` (`H9e=0.029907`), and `S1–S4`. Conflates prompt components (lacks neutral reflection control). |
| **Multi-Family Panel (`P3, P4 < P1`):** Mixed panel improves over homogeneous `gemma` | **VERIFIED AS PACKAGE (CONFOUNDED WITH CAPABILITY)** | `P3-P1`: `-0.012841`, `12/3/1`, `H9e=0.009277`, `H20e=0.022034`; `P4-P1`: `-0.015207`, `14/1/1`, `H9e=0.009277`, `H20e=0.022034` | Survives **`16/16` LOTO subsets**, first-attempt variants, and `S1–S4`. Does **NOT** isolate diversity from model capability (`deepseek-v4-flash` & `glm-5.2` MSEs are `0.000655–0.003922`). |
| **`2 × 2` Sub-Additive Interaction (`(P4-P3)-(P2-P1) > 0`):** Factors act as partial substitutes | **VERIFIED (HIGHLY ROBUST)** | `audit_reproduction_report.json["exact_vs_pipeline_2x2_did"]["crowd_error"]`: `+0.008715`, CI `[+0.003488, +0.015809]`, W/L/T `1/14/1`, `p_ex=0.000183`, `H5e=0.000916` | Survives **`16/16` LOTO subsets** (`Drop Lake Superior`: `+0.005725`, `p_ex=0.000366`), `First_Pass` (`p_ex=0.005371`), `First_Call_Ever` (`p_ex=0.003357`), and `S1–S4`. |
| **Superiority of `P4` over `P3` (`P4 < P3`)** | **INVALIDATED -> NOT SIGNIFICANT** | `P4-P3`: diff `-0.002366`, W/L/T `13/2/1`, `p_norm=0.028768`, `p_ex=0.025574`, **`H9n=0.143839`, `H9e=0.127869`, `H20e=0.383606`**, `t_p=0.067109` | Directional (`13/15` non-tied tasks), **NOT statistically significant** after multiplicity correction; survives only `2/16` LOTO subsets (`Drop LS`: `H9e=0.247192`). |
| **`P4` beats `P0` baseline (`P4 < P0`)** | **INVALIDATED -> NULL & SINGLE-TASK DRIVEN** | `P4-P0`: diff `-0.004252`, `9/6/1`, `p_ex=0.389404`, `H9=1.000000`; `lake_superior_and_influential_shares`: Lake Superior = **`65.4421%`** of net diff (`-0.044523/-0.068035`) | Non-significant across `16/16` LOTO subsets (`0/16`); without Lake Superior, diff shrinks to `-0.001567` (`p_ex=0.6257`). |
| **Refutes pure capability explanation** | **INVALIDATED & REMOVED** | `per_model_breakdown` shows huge capability gap (`gemma` `0.021–0.046` vs `deepseek/glm` `0.0007–0.0039`); missing `P0-Mix` & single-model frontier arms | Existing design cannot separate diversity from stronger constituent models. Explicitly labeled a confounding limitation. |
| **Restores calibration / deployment readiness** | **INVALIDATED & REMOVED** | Study measures static point-estimate `log10` MSE only (no probabilistic calibration or human-in-the-loop trial) | Removed completely. |

---

## 3. Mathematical & Benchmark Verification Claims

| Claim | Status | Exact Evidence Path & Verified Value | Audit Notes / Revision |
|---|---|---|---|
| Hierarchical identity `E_ind = E_crowd + D_within + D_between` holds identically | **VERIFIED (ATTRIBUTION REVISED)** | `max_identity_residual = 1.3878e-16` (`numeric_analysis.json`) | Attributed properly to **Krogh & Vedelsby (1995)** ensemble decomposition + standard within/between variance partitioning (cf. Page 2007; Chuang et al. 2025). |
| `D_excess > 0` proves systematic family specialization / rejects permutation null | **INVALIDATED -> CORRECTED** | `exact_random_label_permutation_audit`: `E_perm[Db/Dw] = 0.750000` (exact); only **`4/16` (`P4` agent)** and **`3/16` (`P3` agent)** tasks reach permutation `p <= 0.05` (`mean p = 0.358–0.464`) | Corrected expectation-operator formula `E_perm[D_between] = ((K-1)/(M-1)) E_perm[D_total]`; `D_between = 0` by construction in `P0..P2`. |
| Unrolled `alpha=0.4` equation `x2 = 0.36 x0 + 0.24 a1 + 0.40 a2` (`0.36^2 = 12.96%` variance share) | **PARTIALLY REVISED** | `unrolled_alpha04_variance_decomposition`: `0.36` is direct linear weight; `0.1296` is squared weight (`0.1296*0.048728 = 0.006315`), NOT variance share under positive covariance | Explained explicit distinction; covariance-inclusive variance share of `x0` is `81.9%–89.2%`. |
| `P4 < P3` crowd error explained via diversity-driven cancellation | **REVISED TO EXACT CROSS-TERM ACCOUNTING** | `r2_signed_error_cross_term_decomposition`: `P3` (`0.007599 + 0.000374 + 0.002085 = 0.010058`) vs `P4` (`0.006105 + 0.000923 + 0.000664 = 0.007692`) | Driven by lower R1 carry (`-0.001494`) and `3.1x` smaller R1/R2 signed-error cross term (`+0.000664` vs `+0.002085`, e.g. `inf_gbt_height_m` `-0.014126`), despite higher R2 assistant error (`0.005770` vs `0.002339`). |
| `between_share` (`53.66%` P3 / `47.41%` P4) equals ratio of aggregate means | **CLARIFIED** | `diversity_share_audit`: mean of per-task ratios = `53.66%` / `47.41%` vs. ratio of aggregate means = `53.68%` / `65.07%` | Clarified explicitly as **unweighted mean of per-task ratios** in report, paper, and Figure 2. |
| Benchmark truths & primary sources (US Hydro `240->245`, Norway Hydro `136.11` vs `137.3`, Molybdenum `2623 °C` vs `2896 K`) | **VERIFIED & SENSITIVITY-TESTED** | `source_and_unit_sensitivity["S1..S4"]` + verified primary bibliographic & agency records | Preserved original `16`-task benchmark intact; `S1–S4` sensitivity analyses yield identical `Holm-9` inferences. |
