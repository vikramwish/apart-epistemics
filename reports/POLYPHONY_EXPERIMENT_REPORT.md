# Polyphony P0–P4 Audited Technical Report (16-Task, α = 0.4)

**Author:** Vikram Viswanath (Independent Researcher; Apart AI × Epistemics Research Sprint, November 2026)  
**Repository / Branch:** `vikramwish/apart-epistemics` (`numeric-question-bank`)  
**Reproducibility Script:** `PYTHONPATH=src .venv/bin/python scripts/audit_and_reproduce_p0_p4.py`  
**Artifact Bundle:** `results/polyphony-p0-p4/` (target 16-task `alpha=0.4`) and `results/polyphony-p0-p4-12task-alpha03/` (preserved earlier 12-task `alpha=0.3` run from commit `b5b91ce`)

---

## 1. Version Resolution & Full Provenance

An audit of repository history resolved two distinct experimental runs previously conflated in project notes:

| Property | Earlier Archived Run (`b5b91ce`) | Target Audit Run (`fdd30fe` / `3a6294c`) |
|---|---|---|
| Storage directory | `results/polyphony-p0-p4-12task-alpha03/` | `results/polyphony-p0-p4/` |
| Git commit timestamp | `2026-11-15 21:51` (`b5b91ce`) | `2026-11-16 00:09–00:23` (`fdd30fe` / `3a6294c`) |
| Task bank & blending rate | 12 numeric tasks, `alpha = 0.3`, `seed = 42` | **16 numeric tasks, `alpha = 0.4`, `seed = 42`** |
| Retained observations | `960` agent-round rows (`12 × 5 × 8 × 2`) | **`1,280` retained agent-round rows (`16 × 5 × 8 × 2`)** |
| Logged API attempts (`provider_requests.jsonl`) | `979` requests | **`1,318` total provider requests** (`+38` extra attempts) |
| Round-2 `crowd_error` means | `P0=0.009932, P1=0.010231, P2=0.010143, P3=0.010340, P4=0.009928` | **`P0=0.011944, P1=0.022899, P2=0.011818, P3=0.010058, P4=0.007692`** |

All target results in `results/polyphony-p0-p4/` remain completely intact. Running `scripts/audit_and_reproduce_p0_p4.py` reproduces every value in `results/polyphony-p0-p4/numeric_analysis.json` identically (`exact_reproduction_matches_saved_numeric_analysis_json = true`, max decomposition identity residual `1.3877787807814457e-16`), and produces `results/polyphony-p0-p4/audit_reproduction_report.json` and `results/polyphony-p0-p4/first_attempt_reconstruction.json`.

### 1.1 Retained Observations, API Attempts, Repetition Loops, and Cost Accounting

- **Independent experimental units vs. sub-samples:** Although `numeric_runs.jsonl` contains `1,280` retained rows (`16 tasks × 5 conditions × 8 agents × 2 rounds × 1 replicate`), the **independent unit of statistical inference is the task (`N = 16`)**. Agents share social context (`pop_median_log10`) and assistants within each `(condition, task)` cell; treating `1,280` rows as independent samples would commit pseudoreplication.
- **Per-model sample counts:** In mixed-panel arms `P3` and `P4`, four models (`gemma-4-26b-a4b-it`, `deepseek-v4-flash`, `qwen3.8-27b`, `glm-5.2`) are each assigned `2` agents per task across `16` tasks and `2` rounds (`16 × 2 × 2 = 64` retained observations per model per condition; `32` in Round 1 and `32` in Round 2), **not** `48`. Single-model arms (`P0`, `P1`, `P2`) have `256` retained observations on `gemma-4-26b-a4b-it` (`128` per round).
- **Retained-row cost vs. total experiment cost:**
  - Retained `1,280` rows: `229,653` input tokens, `531,150` output tokens, **`$0.486271`** actual cost.
  - All `1,318` logged provider requests (`provider_requests.jsonl`): `237,144` input tokens, `654,919` output tokens, **`$0.501372`** total logged request cost (`0` billed reasoning tokens under `reasoning={"effort": "none"}`).
  - Conservative budget tracker (`budget.json`): `$6.196` reserved across `3,098` cumulative request allowances (accounting ceiling, not actual billing).
- **Truncations, repetition loops, and replacement procedure:** Across the initial and resample passes, `14` responses hit `finish_reason = "max_output_tokens"` at `8,192` tokens: `8` failed numeric parsing and triggered automatic in-run retries, while `6` initial-pass repetition loops parsed a number and were replaced afterward via targeted task resampling (affecting `28` pre-23:28 replaced cells in `P0`, `P1`, and `P2`, plus `1` resample-loop replacement on `mat_molybdenum_melting_c`, and `2` resample-wave in-run failures = `38` extra API attempts total). One completed unit-slip (`0.26` on `req 648`) was automatically retried by the runtime `MAX_LOG10_JUMP = 2.0` guard (`First_Call_Ever_Strict` sensitivity recovers all `36` first-ever call overrides). Both first-attempt reconstructions (`First_Pass_Saved_Pre2328` and `First_Call_Ever_Strict`) preserve the headline pattern reported in Section 4.

---

## 2. Experimental Design & Four Error Definitions

All five arms share identical Round-0 initial estimates `x_i^(0)` (`E_crowd(R0) = 0.048728`, generated from `gemma-4-26b-a4b-it` at `seed=42`, `max_output_tokens=8192`), `M = 8` agents, `2` social rounds, and linear deGroot/opinion-pooling update with learning rate `alpha = 0.4`:
- `x_i^(r) = (1 - alpha) x_i^(r-1) + alpha a_i^(r) = 0.6 x_i^(r-1) + 0.4 a_i^(r)`.
- Conditions span a `2 × 2` social factorial plus an independent no-social baseline:
  - **P0 (No Social Baseline):** Homogeneous `gemma-4-26b-a4b-it`, no shared crowd summary.
  - **P1 (Homogeneous + Standard Prompt):** Homogeneous `gemma-4-26b-a4b-it`, shown crowd median `pop_median_log10`.
  - **P2 (Homogeneous + Counter-Anchoring Package):** Same model as `P1`, with counter-anchoring prompt instructions (`adversarial critique + independent reasoning`). Note: `P2 vs P1` tests a **multi-component prompt package**, not an isolated cognitive-forcing toggle, and lacks a token/length-matched neutral reflection control.
  - **P3 (Mixed 4-Family Panel + Standard Prompt):** Evenly split (`K = 4`, `2` agents each) across `gemma-4-26b-a4b-it`, `deepseek-v4-flash`, `qwen3.8-27b`, and `glm-5.2`; standard social prompt.
  - **P4 (Mixed 4-Family Panel + Counter-Anchoring Package):** Same 4-model panel as `P3` with the `P2` counter-anchoring prompt package.

For every task `q` with truth `T_q`, `log10` squared errors distinguish four distinct quantities (do **not** conflate blended agent accuracy with raw assistant accuracy or individual MSE with crowd MSE):

1. **Blended Crowd Error (`crowd_error`):** `((1/M) sum_i x_i^(r) - log10 T_q)^2` — squared error of the equal-weight crowd mean of blended agent estimates.
2. **Blended Individual Error (`avg_individual_error`):** `(1/M) sum_i (x_i^(r) - log10 T_q)^2` — average MSE across individual agents after blending (`E_ind = E_crowd + D_total`).
3. **Assistant Crowd Error (`assistant_crowd_error`):** `((1/M) sum_i a_i^(r) - log10 T_q)^2` — squared error of the equal-weight mean of raw round-`r` assistant outputs (measured **within** social conditions, not standalone model accuracy).
4. **Assistant Individual Error (`assistant_avg_individual_error`):** `(1/M) sum_i (a_i^(r) - log10 T_q)^2` — average MSE of individual assistant outputs.

---

## 3. Main Results: Round-1, Round-2, Per-Model Breakdown & Contrasts

### 3.1 Mean Errors and Diversity Decomposition Across Conditions

| Cond | R1 Blended Crowd | R1 Blended Ind. | R1 Asst Crowd | R1 Asst Ind. | **R2 Blended Crowd** | **R2 Blended Ind.** | R2 Total Div. | R2 Within Div. | R2 Between Div. | R2 Asst Crowd | R2 Asst Ind. | EVRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **P0** | 0.019523 | 0.048079 | 0.019134 | 0.035051 | **0.011944** | 0.028750 | 0.016806 | 0.016806 | 0.000000 | 0.009646 | 0.023701 | 1.378811 |
| **P1** | 0.030271 | 0.055571 | 0.019033 | 0.033526 | **0.022899** | 0.032455 | 0.009557 | 0.009557 | 0.000000 | 0.017313 | 0.019924 | 1.069987 |
| **P2** | 0.019347 | 0.044821 | 0.022888 | 0.031461 | **0.011818** | 0.024672 | 0.012854 | 0.012854 | 0.000000 | 0.013480 | 0.027838 | 1.339442 |
| **P3** | 0.021108 | 0.044012 | 0.004024 | 0.015162 | **0.010058** | 0.019078 | 0.009021 | 0.004178 | 0.004842 | 0.002339 | 0.008772 | 1.105996 |
| **P4** | 0.016958 | 0.042940 | 0.004146 | 0.013590 | **0.007692** | 0.021534 | 0.013842 | 0.004835 | 0.009007 | 0.005770 | 0.017038 | 1.282554 |

### 3.2 Pairwise Contrasts on Round-2 Blended Crowd Error (`N = 16`, Signed `B - A`)

Both the pipeline two-sided asymptotic normal-approximation Wilcoxon (`pipeline_wilcoxon_p`) and the **exact $2^{n_{\text{nz}}}$ zero-drop signed-rank distribution (`exact_wilcoxon_nz_p`)** plus exact Pratt signed-rank (`exact_pratt_p`) and paired *t*-test are reported alongside **within-metric Holm (`m = 9`)** and **exploratory cross-metric global Holm (`m = 20`)**. Notice `sci_lhc_dipole_magnets_count` ties identically across all 5 arms (`crowd_error = 0.016786`), leaving `15` non-zero pairs for every pairwise contrast.

| Contrast (`B - A`) | Mean Diff (95% BCa/Boot CI) | W / L / T (`B<A / B>A / Tie`) | Pipeline `p` | Exact NZ `p` | Exact Pratt `p` | **Holm `m=9` (Pipe / Exact)** | **Holm `m=20` (Pipe / Exact)** | Paired `t`-test `p` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **P2 - P1** | `-0.011081` (`[-0.020269, -0.004246]`) | `14 / 1 / 1` | `0.000892` | `0.000122` | `0.000122` | **`0.008027` / `0.001099`** | **`0.017838` / `0.002441`** | `0.024519` |
| **P4 - P1** | `-0.015207` (`[-0.027624, -0.005123]`) | `14 / 1 / 1` | `0.002866` | `0.001160` | `0.000854` | **`0.022925` / `0.009277`** | `0.054446` / **`0.022034`** | `0.023652` |
| **P3 - P1** | `-0.012841` (`[-0.023138, -0.004497]`) | `12 / 3 / 1` | `0.002866` | `0.001160` | `0.001465` | **`0.022925` / `0.009277`** | `0.054446` / **`0.022034`** | `0.020696` |
| **P1 - P0** | `+0.010955` (`[+0.003685, +0.019050]`) | `2 / 13 / 1` | `0.008266` | `0.005371` | `0.004639` | **`0.049594` / `0.032227`** | `0.132250` / `0.085938` | `0.018259` |
| **P4 - P3** | `-0.002366` (`[-0.004763, -0.000253]`) | `13 / 2 / 1` | `0.028768` | `0.025574` | `0.019714` | `0.143839` / `0.127869` | `0.431518` / `0.383606` | `0.067109` |
| **P4 - P2** | `-0.004126` (`[-0.008579, -0.000468]`) | `9 / 6 / 1` | `0.147532` | `0.151428` | `0.166931` | `0.590130` / `0.605713` | `1.000000` / `1.000000` | `0.073755` |
| **P4 - P0** | `-0.004252` (`[-0.010893, +0.000461]`) | `9 / 6 / 1` | `0.378675` | `0.389404` | `0.389282` | `1.000000` / `1.000000` | `1.000000` / `1.000000` | `0.181550` |
| **P2 - P0** | `-0.000126` (`[-0.005137, +0.004426]`) | `5 / 10 / 1` | `0.443231` | `0.454285` | `0.418213` | `1.000000` / `1.000000` | `1.000000` / `1.000000` | `0.962100` |
| **P3 - P0** | `-0.001887` (`[-0.007044, +0.002147]`) | `8 / 7 / 1` | `0.798273` | `0.803955` | `0.801270` | `1.000000` / `1.000000` | `1.000000` / `1.000000` | `0.437658` |

*Notes on multiplicity and exact vs. asymptotic tests:* Within-metric Holm (`m = 9`, pre-specified for each primary endpoint family in `src/apart_epistemics/metrics.py` prior to 16-task data collection at commit `b5b91ce`, though **not formally preregistered** in an external registry) treats `P1..P4` pairwise tests on `crowd_error` as the primary inference family. Under exact Wilcoxon (`m = 9`), `P2-P1`, `P4-P1`, `P3-P1`, and `P1-P0` remain `< 0.05`. Under the broader exploratory 20-contrast family (`m = 20`, spanning `crowd_error`, `assistant_crowd_error`, `between_model_diversity`, `excess_between_model_diversity`, and `evrr`), only `P2-P1` (`p = 0.017838` pipeline / `0.002441` exact), `P4-P1` (exact `0.022034`), `P3-P1` (exact `0.022034`), `assistant_crowd_error P3-P1` (exact `0.025940`), and trivial single-vs-multi-model `between_model_diversity` contrasts remain `< 0.05`. Crucially, **`P4 vs P3` is NOT statistically significant under either family (`Holm-9 p = 0.143839` pipeline, `0.127869` exact; `Holm-20 p = 0.431518` pipeline, `0.383606` exact)**.

### 3.3 2 × 2 Factorial Interaction on Social Conditions (`DiD = (P4 - P3) - (P2 - P1)`)

| Metric | Mean DiD | 95% CI | W / L / T (`<0 / >0 / 0`) | Pipe `p` (Holm-5) | Exact NZ `p` (Holm-5) |
|---|---:|---:|---:|---:|---:|
| **`crowd_error`** | **`+0.008715`** | `[+0.003488, +0.015809]` | `1 / 14 / 1` | `0.001092` (**`0.005458`**) | `0.000183` (**`0.000916`**) |
| `assistant_crowd_error` | `+0.007264` | `[-0.005420, +0.020280]` | `3 / 12 / 1` | `0.015785` (`0.063142`) | `0.012451` (**`0.049805`**) |
| `avg_individual_error` | `+0.010239` | `[+0.001770, +0.021511]` | `4 / 11 / 1` | `0.038167` (`0.114500`) | `0.035339` (`0.106018`) |
| `diversity` | `+0.001524` | `[-0.005048, +0.010525]` | `8 / 7 / 1` | `0.932107` (`1.000000`) | `0.934082` (`1.000000`) |
| `evrr` | `-0.092897` | `[-0.552618, +0.281141]` | `7 / 8 / 1` | `0.977345` (`1.000000`) | `0.977966` (`1.000000`) |

Both counter-anchoring (`P2-P1 = -0.011081`) and model mixing (`P3-P1 = -0.012841`) strongly reduce social monoculture amplification, while combining them gives a further non-significant `-0.002366` step (`P4-P3`), producing a significant **positive (sub-additive) interaction** (`DiD = +0.008715`, `p = 0.000183` exact).

### 3.4 In-Condition Model Breakdown (`n = 64` per Model in `P3`/`P4`; NOT Standalone Accuracy)

| Condition | Model | Retained `n` | Assistant R1 MSE | Assistant R2 MSE | Assistant Combined MSE | Agent R2 MSE |
|---|---|---:|---:|---:|---:|---:|
| P0 | `gemma-4-26b-a4b-it` | 256 (`128+128`) | 0.035051 | 0.023701 | 0.029376 | 0.028750 |
| P1 | `gemma-4-26b-a4b-it` | 256 (`128+128`) | 0.033526 | 0.019924 | 0.026725 | 0.032455 |
| P2 | `gemma-4-26b-a4b-it` | 256 (`128+128`) | 0.031461 | 0.027838 | 0.029649 | 0.024672 |
| P3 | `deepseek-v4-flash` | 64 (`32+32`) | 0.002277 | 0.001677 | 0.001977 | 0.013535 |
| P3 | `glm-5.2` | 64 (`32+32`) | 0.005099 | 0.002745 | 0.003922 | 0.019099 |
| P3 | `qwen3.8-27b` | 64 (`32+32`) | 0.022743 | 0.018265 | 0.020504 | 0.015856 |
| P3 | `gemma-4-26b-a4b-it` | 64 (`32+32`) | 0.030529 | 0.012398 | 0.021464 | 0.027824 |
| P4 | `deepseek-v4-flash` | 64 (`32+32`) | 0.001988 | 0.000534 | 0.001261 | 0.012539 |
| P4 | `glm-5.2` | 64 (`32+32`) | 0.000286 | 0.001023 | 0.000655 | 0.016799 |
| P4 | `qwen3.8-27b` | 64 (`32+32`) | 0.008031 | 0.019393 | 0.013712 | 0.019332 |
| P4 | `gemma-4-26b-a4b-it` | 64 (`32+32`) | 0.044055 | 0.047203 | 0.045629 | 0.037468 |

Because `deepseek-v4-flash` and `glm-5.2` have ~`10x`–`40x` lower in-condition assistant MSE than `gemma-4-26b-a4b-it`, and this experiment lacks a heterogeneous no-social arm (`P0-Mix`) and homogeneous `deepseek`/`glm`/`qwen` arms, **`P3/P4 vs. P1` cannot separate inter-model diversity from constituent model capability** and **never refutes a pure capability explanation**.

---

## 4. Robustness Diagnostics (Existing Data)

1. **Leave-One-Task-Out (`n = 15`) and Lake Superior Dominance:**
   - Single-task share of `hyd_lake_superior_vol_km3`: **`65.4421%`** of net `P4-vs-P0` reduction (`-0.044523 / -0.068035`), **`38.4124%`** of `P4-vs-P3`, **`38.4201%`** of `2×2 DiD`, **`38.4185%`** of `P2-vs-P1`, and **`34.8870%`** of `P4-vs-P1`.
   - Under `LOTO`, `P2-vs-P1`, `P3-vs-P1`, `P4-vs-P1`, and the `2×2 DiD` achieve **`Holm < 0.05` on all `16/16` `n=15` subsets** (`Drop Lake Superior`: `DiD = +0.005725`, `13/1/1`, exact `p = 0.000366`; `P2-vs-P1` exact Holm-9 `p = 0.002197`; `P3-vs-P1` `0.018555`; `P4-vs-P1` `0.018555`).
   - By contrast, **`P1-vs-P0`** stays `p_raw <= 0.0107` on all `16/16` subsets but drops out of `Holm-9 < 0.05` on `9/16` subsets (`7/16` survive; `Drop Lake Superior` Holm-9 exact `p = 0.064453`), **`P4-vs-P3`** hits `Holm-9 < 0.05` on only `2/16` subsets (`Drop Lake Superior` mean diff `-0.001554`, `Holm-9 = 0.247192`), and **`P4-vs-P0`** hits `0/16` (`Drop Lake Superior` mean diff `-0.001567`, exact `p = 0.6257`).
2. **First-Attempt Sensitivity (`First_Pass_Saved_Pre2328` & `First_Call_Ever_Strict`):**
   - `First_Pass_Saved_Pre2328` (`28` restored pre-23:28 cells: `P0=0.011978, P1=0.025049, P2=0.012681, P3=0.010058, P4=0.007692`): `P2-P1` exact `H9 = 0.023499`, `P3-P1` `0.007690`, `P4-P1` `0.007690`, `P1-P0` `0.025635`, `P4-P3` `0.127869`, `DiD = +0.010002` (exact `p = 0.005371`).
   - `First_Call_Ever_Strict` (`36` first-ever calls including `req 648` `0.26` unit-slip on `P0`, raising `P0 = 0.016268`): `P2-P1` exact `H9 = 0.029907`, `P3-P1` `0.007690`, `P4-P1` `0.012207`, `P4-P3` `0.333008`, `DiD = +0.009998` (exact `p = 0.003357`), while `P1-P0` shifts to non-significant (`H9 = 0.180908`) due to the unguarded `0.26` unit slip inflating `P0`.
3. **Ground-Truth Source Revision & Celsius/Kelvin Sensitivity (`S1`–`S4`):**
   - Re-scoring `nrg_us_hydro_2023_twh` at finalized EIA *Electric Power Annual 2023* (`245.0 TWh`, `S1`), `nrg_norway_hydro_2023_twh` at Statistics Norway gross (`137.3 TWh`, `S2`), both together (`S3`), or dropping `mat_molybdenum_melting_c` (`S4`, `n = 15`, where `72/80` retained model answers are `2623 °C`, `6` are `2350`, `1` is `2620`, and `1` is `2896 K`) leaves **every Holm-9 conclusion unchanged** (`P2/P3/P4 vs P1` exact `Holm-9 <= 0.013672`; `P4-P3` exact `Holm-9 = 0.127869–0.176392`).

---

## 5. Corrected Mathematical Accounting & Methodological Clarifications

1. **Ensemble Attribution (`E_ind = E_crowd + D_within + D_between`):** This identity combines **Krogh & Vedelsby's (1995)** diversity-prediction squared-error identity (`E_ind = E_crowd + D_total`) with standard ANOVA/Rao within-between variance decomposition (`D_total = D_within + D_between`), echoing Page's (2007) diversity prediction theorem and Chuang et al.'s (2025) numerical crowd estimation setting. It is accounting, not a novel physical law.
2. **Expectation-Correct Random-Label Null:** Under random permutation of the `K = 4` family labels across `M = 8` fixed agents (`2` agents per family), exact algebra yields `E_perm[D_between] = ((K - 1) / (M - 1)) E_perm[D_total]` and **`E_perm[D_between] = (3 / 4) E_perm[D_within]`** (`0.75` exactly, verified across all `105` balanced partitions of `8` agents on all `16` tasks). Wait: positive `D_excess = D_between - 0.75 D_within` **does NOT certify true task knowledge, systematic specialization, or task-level rejection of the null**: across `16` tasks, only **`4/16` (`P4`) and `3/16` (`P3`)** reach per-task permutation `p <= 0.05` (`mean p = 0.358–0.464`). Moreover, `D_between = 0` by construction in single-model arms `P0..P2`, so comparing `P3/P4` against `0` proves nothing about accuracy gains.
3. **Unrolled `alpha = 0.4` Linear Coefficient vs. Variance Share:** Two rounds of pooling give `x_i^(2) = 0.36 x_i^(0) + 0.24 a_i^(1) + 0.40 a_i^(2)`. The direct linear weight of the Round-0 prior `x^(0)` is **`36%`**. Its squared coefficient in the expanded MSE is `0.36^2 = 12.96%` (`0.1296 * 0.048728 = 0.006315`), which **is not** the prior's share of total error or variance whenever errors correlate across rounds; including the covariance terms involving `x^(0)` places the empirical covariance-inclusive variance share at `81.9%–89.2%` across `P0..P4`.
4. **Why `P4` Blended Crowd Error (`0.007692`) Is Lower Than `P3` (`0.010058`) Despite Higher R2 Assistant Crowd Error (`0.005770` vs. `0.002339`):** Exact squared-error expansion at Round 2 (`E_crowd^(2) = 0.36 E_crowd^(1) + 0.16 E_asst,crowd^(2) + 0.48 eps_bar^(1) eps_Abar^(2)`) yields:
   - `P0`: `0.007028 + 0.001543 + 0.003373 = 0.011944`
   - `P1`: `0.010898 + 0.002770 + 0.009231 = 0.022899`
   - `P2`: `0.006965 + 0.002157 + 0.002696 = 0.011818`
   - `P3`: `0.007599 + 0.000374 + 0.002085 = 0.010058`
   - `P4`: `0.006105 + 0.000923 + 0.000664 = 0.007692`
   Thus `P4` trails `P3` in Round-2 crowd error purely through arithmetic: lower Round-1 carried-over error (`0.006105` vs. `0.007599`) and a `3.1x` smaller positive Round-1/Round-2 signed-error alignment term (`0.000664` vs. `0.002085`, especially negative offsetting error on `inf_gbt_height_m`, `-0.014126`), **not** via diversity-driven cancellation established at statistical significance.
5. **Diversity Ratio Definition (`between_share`) & EVRR Status:** `between_share` in `numeric_analysis.json` (`P3 R2 = 53.66%` agent / `57.72%` assistant; `P4 R2 = 47.41%` / `50.66%`) is the **unweighted mean of per-task ratios** `mean_q(D_between,q / D_total,q)`, which differs substantially from the ratio of aggregate means (`P4 R2 = 65.07%` / `77.02%`) when tasks have heterogeneous total diversity. Likewise, `EVRR = E_ind / (E_ref + ...)` is an empirical variance-ratio index against a pooled-variance reference baseline, **not** an information-theoretic lower bound or epistemic-quality guarantee.

---

## 6. Genuine Required Follow-Up Controls & Open Issues

1. **Missing `P0-Mix` (`K = 4` no-social panel):** Needed to separate constituent model strength from multi-model social exchange.
2. **Missing Homogeneous Frontier Controls (`P0`/`P1`/`P2` on `deepseek-v4-flash`, `glm-5.2`, `qwen3.8-27b`):** Required before comparing capability vs. diversity explanations cleanly.
3. **Missing Token/Structure-Matched Generic Reflection Control:** Needed to isolate adversarial counter-anchoring from generic second-pass self-reflection or prompt length.
4. **Static Numerical Error Scope:** The study measures static `log10` estimation error on `16` physical/energy facts with one replicate (`seed = 42`, `alpha = 0.4`); it makes no validated claims about user calibration restoration or production deployment readiness.
