# Replication Study Report — `polyphony-replication-2025`

Pre-registered 40-task, 3-replicate (`24,960`-cell) out-of-sample replication of the `P0`–`P4` + 8-control-arm polyphony study ([PREREGISTRATION_replication_2025.md](../../docs/PREREGISTRATION_replication_2025.md)) on `2025–2026` post-cutoff quantitative benchmark tasks (`current_date = "2026-11-15"`).

**Key headline:** Every pre-registered directional control question that was null or underpowered in the 16-task original controls run (`0/3` generalization, `Inconclusive` critique-vs-reflection, `n.s.` `P3 vs VIRTUAL_SOCIAL`, `n.s.` `P4 vs P3`) **replicates decisively at `N = 40` tasks (`3` replicates)**: social herding generalizes to **4/4** model families, two-sided dialectical critique beats compute/length-matched reflection (`p_holm = 0.00048`), mixed-architecture interaction (`P3`) herds significantly less than same-model medians (`VIRTUAL_SOCIAL`, `p_holm = 0.0111`), and `P4` outperforms both `P3` (`p_holm = 0.000904`) and the strongest single model's social arm (`P1_DS`, `p_holm = 0.0240`). Simultaneously, the negative boundary conditions hold: peer telemetry still harms mixed panels relative to silent polyphony (`P3/P4 vs P0_POLY`, `+64.0%`/`+40.6%` $E_2$), and `P4` still fails to beat Gemma's silent baseline `P0` (`p_holm = 0.6368`).

---

## 1. Provenance, Execution Accounting & Missing/Dropped Items

### 1.1 Commit Chain & Artifacts
| Order | Branch | Commit Hash | Description |
|---|---|---|---|
| 1 | `numeric-question-bank` | `5ad9f348ace1b49d5399b2aa38300b4bcb433944` | Control-arm harness as executed by original `polyphony-controls` run (`PID 24020`) |
| 2 | `numeric-question-bank` | `3d414924c07c6fcbb10d0cce5627496d5491950d` | Pre-registered controls analysis kit before reading `polyphony-controls` outputs (`replication-2025` base) |
| 3 | `replication-2025` | `9ccd45798315ea53a3cd9f425281cbf9e19ba29d` | Steps 1–4: 43 verified tasks, `|xbar_2 - y*|` + Holm `m=4` extensions, `PREREGISTRATION_replication_2025.md` |
| 4 | `replication-2025` | `6bab85eecd0cace2d170e49979da32a71d331ced` | Step 5: Cold-probe filter (`40/43` surviving tasks), `polyphony_replication_2025.yaml` |
| 5 | `replication-2025` | `3c4aedbf56fb83bc6ba3fc71fc734b41303c5445` | Step 6 fix: Honor `--model` on `SyntheticEpistemicBackend` for dry-run parity |
| 6 | `replication-2025` | `237e639a07de6e0215b8db60f62e9c3d1880e69c` | Pre-run `PROVENANCE.md` lock prior to any Step 7 non-probe live calls |
| 7 | `replication-2025` | *(final commit)* | Step 7/8/9 outputs (`audit_replication_2025.py`, `audit_replication_report.json`, `numeric_analysis.json`, `preregistered_analysis.json`, `REPORT.md`, `PROVENANCE.md`) |

- **Raw JSONL Checksums (kept locally on disk under `results/polyphony-replication-2025/`, excluded from git due to >100 MB file sizes):**
  - `numeric_runs.jsonl` (`637 MB`): `sha256:041a8170379b3be671de2aa6a1e1017b8cc64c950f8511b5e71c59968fef272a`
  - `provider_requests.jsonl` (`166 MB`): `sha256:78aa027773e3fbd52a26b7752809206a72ee89007dbc1ac031997e419008460a`
- **Tracked Primary Artifacts:**
  - `numeric_runs.manifest.json`: `sha256:669727d6c01cbaa669826c3db750038f87e879dfb7ac6edb14643af323466799`
  - `preregistered_analysis.json`: `sha256:d48c0d36f72aa6bfac233f5d12655cb434baf9f0a6dfd6c40be6d52045e88406`

### 1.2 Model Slugs, Calls & Spend Accounting
Requested model slugs matched `response.model` across **100%** of HTTP responses (zero routing drift):

| Model Slug | Cold Probe Calls / Cost | Main Saved Cells | Main HTTP Successes (`Format Retries`) | Main Spend (USD) |
|---|---|---|---|---|
| `gemma-4-26b-a4b-it` | `43` / `$0.003611` | `9,120` | `9,125` (`5`) | `$0.912479` |
| `deepseek-v4-flash` | `43` / `$0.000931` | `5,280` | `5,310` (`30`) | `$0.174761` |
| `qwen3.8-27b` | `43` / `$0.053080` | `5,280` | `5,287` (`7`) | `$10.395274` |
| `glm-5.2` | `43` / `$0.028689` | `5,280` | `5,386` (`106`) | `$5.711123` |
| **Total** | **`172` / `$0.086311`** | **`24,960`** | **`25,108` (`148`)** | **`$17.193638`** |

- **Combined actual API spend (Cold Probe + Main Run):** **`$17.279949`** (`0` missing `usage.cost` entries; under `$20`).
- **Token totals (Main Run `provider_requests.jsonl`):** `4,108,246` input tokens, `13,562,141` output tokens (`0` reported separate reasoning tokens).
- **Budget ledger reservations:** Cold probe used `277` reservations (`105` transient retries, `$4.257213 / $5.00`). Main run used `28,744` reservations (`3,636` transient HTTP retry attempts + `25,108` HTTP successes, `$441.766536 / $445.00` conservative reservation cap).

### 1.3 Dropped Tasks (`5 / 45`) and Main-Run Integrity / Misses / Truncations
- **Step 2 primary-source drops (`2` of `8` `"VERIFY"` tasks; see [TASK_DECISIONS.md](TASK_DECISIONS.md)):**
  1. `nrg_us_coal_power_2025_twh`: Dropped — EIA *Electric Power Monthly* primary table shows `734,928` thousand MWh all-sectors and `730,459` thousand MWh utilities+IPPs vs STEO `731.0`.
  2. `nrg_germany_public_net_2025_twh`: Dropped — primary Fraunhofer ISE Energy-Charts reports `465.8 TWh` net load with no verified `440 TWh` headline figure.
  - Note: `nrg_france_consumption_2025_twh` was disambiguated pre-probe to specify `"weather- and calendar-corrected"` (`451.0 TWh` per RTE *Bilan électrique 2025*).
- **Step 5 cold-probe drops (`3` of `43` probed tasks; see [cold_probe_errors.md](cold_probe_errors.md)):**
  - Rule `(a)` parse failures: **`0`** (`0` format retries in cold probe).
  - Rule `(b)` recall (`|log10 err| < 0.0086` for all 4 models): **`3` dropped** (`env_noaa_co2_2025_ppm` `max|err|=0.001144`, `env_wmo_n2o_2025_ppb` `max|err|=0.002054`, `dem_us_population_2025` `max|err|=0.002553`).
  - Rule `(c)` surviving task strata (`40 >= 25` required): **`control`: `5`**, **`low_spread`: `18`**, **`high_spread`: `17`**.
- **Main-run completeness, identity residuals, retries, truncations, and parse misses:**
  - **Cells:** `24,960 / 24,960` saved (`0` missing/incomplete cells, `0` duplicates, `0` malformed lines, `120 / 120` complete task-replicates per condition).
  - **Identity residual:** `max_identity_residual = 3.330669e-16` (both Page and hierarchical Page identities across agent and assistant estimates).
  - **Format retries:** `148` extra single format-retry calls across `24,960` cells (`0.59%`), accounting **1-to-1** for `25,108 - 24,960 = 148`.
  - **Length truncations:** `6` HTTP responses hit `incomplete: max_output_tokens` (`8,192` tokens: `2` `gemma-4-26b-a4b-it`, `4` `qwen3.8-27b`). `5` resolved cleanly via format retry; `1` remained on a saved valid cell (`P1 / ind_world_steel_2025_mt / rep 2 / round 2 / agent_04`, `gemma-4-26b-a4b-it`, `finish_reason="length"`, `output_tokens=8192`, `miss=false`).
  - **Saved cell misses (`5 / 24,960` = `0.020%`):** Left as misses with prior carried forward (`agent_log10_after = agent_log10_before`), never re-sampled:
    1. `P3 / cry_arctic_max_2026_kkm2 / rep 1 / r2 / agent_07 (glm-5.2)` — tail `"Final estimate: 14.05"`
    2. `P4 / aut_europe_ev_sales_2025_k / rep 0 / r1 / agent_05 (deepseek-v4-flash)` — tail `"Final estimate: 3,800 thousand cars"`
    3. `P4 / aut_europe_ev_sales_2025_k / rep 1 / r1 / agent_05 (deepseek-v4-flash)` — tail `"Final estimate: 3,100 thousand cars."`
    4. `P4 / aut_europe_ev_sales_2025_k / rep 1 / r2 / agent_01 (deepseek-v4-flash)` — tail `"Final estimate: 2850 thousand cars."`
    5. `P0_POLY / cry_arctic_min_2026_kkm2 / rep 1 / r2 / agent_03 (glm-5.2)` — tail `"Final estimate: 4.27"`

---

## 2. Updated Claims Ledger (Original `n=16` vs. Replication `N=40, rep=3`)

| Claim | Original Verdict (`n=16`, `rep=1`) | Replication Result (`N=40`, `rep=3`, Primary $E_2$) | Pre-Registered Decision / Updated Verdict |
|---|---|---|---|
| **Harness sanity (`P0_POLY` vs `VIRTUAL_SILENT`)** | **PASS** (`+0.001007`, `90% CI [-0.000031, +0.002616]`, inside `±0.00299`) | **PASS** (`+0.000063`, `90% CI [-0.000513, +0.000621]`, `p_exact = 0.7346`) | **VERIFIED:** Seat-matched routing and priors reproduce cleanly (`0` virtual-panel problems). |
| **Gemma social herding (`P1 > P0`)** | **Verified (LOTO-sensitive):** `+0.010955` (`+91.7%`), `p_exact = 0.0054`, `H9e = 0.0322` | **`+0.017035` (`+114.5%`)**, `36/40` worse, `p_exact = 0.0`, **`p_holm(m=4) = 0.0`** | **REPLICATED & STRENGTHENED:** Social telemetry severely degrades Gemma crowd accuracy. |
| **Dialectical critique rescue (`P2 < P1`)** | **Verified (robust):** `-0.011081` (`-48.4%`), `p_exact = 0.00012`, `H9e = 0.0011` | **`-0.008476` (`-26.6%`)**, `35/40` better, `p_exact = 0.0`, **`p_holm(m=4) = 0.0`** | **REPLICATED:** Dialectical counter-anchoring robustly mitigates homogeneous herding. |
| **Critique atop polyphony (`P4 < P3`)** | **Invalidated -> n.s.:** `-0.002366`, `p_exact = 0.0256`, `H9e = 0.1279` | **`-0.002219` (`-14.3%`)**, `30/40` better, `p_exact = 0.000452`, **`p_holm(m=4) = 0.000904`** | **NOW CONFIRMED:** At `N = 40`, dialectical critique yields a reliable incremental gain atop polyphony. |
| **`P4` beats Gemma silent baseline (`P4 < P0`)** | **Invalidated -> Null:** `-0.004252`, `p_exact = 0.389`, `H9 = 1.0` (`65.4%` Lake Superior) | `-0.001528` (`-10.3%`), `20` better / `20` worse, `p_exact = 0.6368`, **`p_holm(m=4) = 0.6368`** | **CONFIRMED NULL:** Even combined `P4` does not reliably outperform silent Gemma `P0`. |
| **Herding generalizes beyond Gemma (`herding_ds/qwen/glm`)** | **`0 / 3` significant** (`p_holm(m=8)`: `0.0996` DS, `1.0` Qwen, `0.3186` GLM) -> *"specific to weaker models / underpowered"* | **All `3 / 3` significant:** DS `+0.008350` (`+105.5%`, `p_holm=0.0`), Qwen `+0.002638` (`+29.2%`, `p_holm=2e-05`), GLM `+0.002340` (`+23.9%`, `p_holm=0.002462`) | **DECISION RULE MET (`>= 2 of 3`):** *"Assistants herd" generalizes beyond Gemma across all four architectures.* |
| **Generic reflection (`P1_REFLECT - P1`) & Critique vs Reflection (`P2 - P1_REFLECT`)** | **Inconclusive:** `P1_REFLECT - P1 = -0.003805` (`p_holm=0.304`), `P2 - P1_REFLECT = -0.007276` (`p_holm=0.319`, CI spans `±0.00277`) | `P1_REFLECT - P1`: **`-0.004733` (`-14.8%`, `p_holm=0.0`)**; **`P2 - P1_REFLECT`: `-0.003743` (`-13.8%`, `30/40` better, `p_holm=0.00048`)** | **DECISION RULE MET (`Row 5 significant, P2 better`):** *"Dialectical" supported: the two-sided structure matters.* |
| **Mixed panel vs Silent polyphony (`P3/P4` vs `P0_POLY`)** | `P3` worse n.s. (`+0.001946`, `p_holm=0.319`); `P4` slightly better n.s. (`-0.00042`, `p_holm=1.0`) | **`P3 - P0_POLY`: `+0.006075` (`+64.0%`, `36/40` worse, `p_holm=0.0`)**; **`P4 - P0_POLY`: `+0.003855` (`+40.6%`, `34/40` worse, `p_holm=1.5e-05`)** | **REPLICATED NEGATIVE BOUNDARY:** Telemetry severely hurts mixed panels; silent `P0_POLY` outperforms `P3` and `P4`. |
| **Cross-architecture social benefit (`P3 vs VIRTUAL_SOCIAL`)** | **Null after Holm:** `-0.002953`, `p_exact = 0.0155`, `p_holm = 0.1085` | **`-0.001413` (`-8.33%`)**, `95% CI [-0.002404, -0.000505]`, `26/40` better, `p_exact = 0.0111`, **`p_holm = 0.011074`** | **DECISION RULE MET (`Row 8 significant, P3 better`):** *Genuine cross-architecture social benefit: a heterogeneous median herds less than same-model medians.* |
| **Secondary: Mixed vs Best Single-Model Social (`BEST_K1_SOCIAL = P1_DS`)** | `P3` worse n.s. (`+0.002466`, `p_holm=0.0603`); `P4` null (`+0.000100`, `p_holm=0.4543`) | `P3 vs P1_DS`: `-0.000704` (`p_holm=0.7852`); **`P4 vs P1_DS`: `-0.002923` (`-18.0%`, `p_holm=0.024032`)** | **`P4` BEATS BEST `K=1` SOCIAL ARM:** `P3` matches `P1_DS`; `P4` significantly outperforms `P1_DS` on both $E_2$ and `\|xbar_2 - y*\|`. |

---

## 3. Pre-Registered Statistical Tables & Parity Audit (`N = 40` Tasks, `3` Replicates)

### 3.1 Condition Means (`E2`, `|xbar_2 - y*|`, and Output Tokens)
*(All 13 physical arms plus 4 derived control references have `n = 40` tasks and `120` complete task-replicates. Best single silent model: `deepseek` -> `BEST_K1_SOCIAL = P1_DS`.)*

| Condition | Mean $E_2 = (\bar{x}_2 - y^*)^2$ | Mean $\|\bar{x}_2 - y^*\|$ | Mean Output Tokens |
|---|---|---|---|
| `P0_DS` | **`0.007916`** | **`0.077528`** | `178.1` |
| `P0_QWEN` | `0.009033` | `0.080022` | `1419.1` |
| `VIRTUAL_SILENT` | `0.009425` | `0.083850` | — |
| `P0_POLY` | `0.009488` | `0.084171` | `534.8` |
| `P0_GLM` | `0.009774` | `0.084557` | `152.3` |
| `P1_QWEN` | `0.011671` | `0.092289` | `1812.5` |
| `P1_GLM` | `0.012114` | `0.095665` | `220.2` |
| `P4` | `0.013343` | `0.102137` | `526.0` |
| `P0` | `0.014872` | `0.103885` | `390.5` |
| `P3` | `0.015563` | `0.110974` | `639.1` |
| `P1_DS` (`BEST_K1_SOCIAL`) | `0.016267` | `0.112331` | `213.3` |
| `VIRTUAL_SOCIAL` | `0.016976` | `0.116000` | — |
| `POOLED_K1_SOCIAL_JENSEN_BOUND` | `0.017989` | `0.115062` | — |
| `P2` | `0.023431` | `0.134915` | `293.6` |
| `P1_REFLECT` | `0.027173` | `0.146522` | `289.7` |
| `P1` | `0.031906` | `0.159964` | `357.4` |

### 3.2 Confirmatory Family (`Holm m = 8`)

#### Primary Metric: Squared Log Error $E_2$
| Contrast (`b - a`) | $\Delta$ (`rel`) | `95% Bootstrap CI` | `90% Bootstrap CI` | `b` Better / Worse / Tied | `p_exact` | `p_holm` |
|---|---|---|---|---|---|---|
| `herding_ds` (`P1_DS - P0_DS`) | `+0.008350` (`+105.5%`) | `[0.006056, 0.010732]` | `[0.006468, 0.010394]` | `1 / 39 / 0` | `0.000000` | **`0.000000`** |
| `herding_qwen` (`P1_QWEN - P0_QWEN`) | `+0.002638` (`+29.2%`) | `[0.001613, 0.003752]` | `[0.001761, 0.003592]` | `8 / 32 / 0` | `0.000005` | **`0.000020`** |
| `herding_glm` (`P1_GLM - P0_GLM`) | `+0.002340` (`+23.9%`) | `[0.000824, 0.004146]` | `[0.001057, 0.003807]` | `9 / 31 / 0` | `0.001231` | **`0.002462`** |
| `reflect_vs_p1` (`P1_REFLECT - P1`) | `-0.004733` (`-14.8%`) | `[-0.006537, -0.003025]` | `[-0.006217, -0.003236]` | `32 / 8 / 0` | `0.000000` | **`0.000000`** |
| `critique_vs_reflect` (`P2 - P1_REFLECT`) | `-0.003743` (`-13.8%`) | `[-0.005708, -0.001863]` | `[-0.005317, -0.002193]` | `30 / 10 / 0` | `0.000160` | **`0.000480`** |
| `p3_vs_silent_poly` (`P3 - P0_POLY`) | `+0.006075` (`+64.0%`) | `[0.004649, 0.007668]` | `[0.004817, 0.007410]` | `4 / 36 / 0` | `0.000000` | **`0.000000`** |
| `p4_vs_silent_poly` (`P4 - P0_POLY`) | `+0.003855` (`+40.6%`) | `[0.002302, 0.005675]` | `[0.002441, 0.005419]` | `6 / 34 / 0` | `0.000003` | **`0.000015`** |
| `p3_vs_virtual_social` (`P3 - VIRTUAL_SOCIAL`) | `-0.001413` (`-8.3%`) | `[-0.002404, -0.000505]` | `[-0.002240, -0.000655]` | `26 / 14 / 0` | `0.011074` | **`0.011074`** |

#### Pre-Registered Secondary Metric: Absolute Log Error $|\bar{x}_2 - y^*|$
| Contrast (`b - a`) | $\Delta$ (`rel`) | `95% Bootstrap CI` | `90% Bootstrap CI` | `b` Better / Worse / Tied | `p_exact` | `p_holm` |
|---|---|---|---|---|---|---|
| `herding_ds` (`P1_DS - P0_DS`) | `+0.034803` (`+44.9%`) | `[0.026857, 0.043634]` | `[0.028208, 0.042315]` | `1 / 39 / 0` | `0.000000` | **`0.000000`** |
| `herding_qwen` (`P1_QWEN - P0_QWEN`) | `+0.012267` (`+15.3%`) | `[0.007440, 0.017241]` | `[0.008170, 0.016597]` | `7 / 33 / 0` | `0.000009` | **`0.000036`** |
| `herding_glm` (`P1_GLM - P0_GLM`) | `+0.011108` (`+13.1%`) | `[0.003927, 0.019729]` | `[0.005061, 0.018197]` | `9 / 31 / 0` | `0.001298` | **`0.002596`** |
| `reflect_vs_p1` (`P1_REFLECT - P1`) | `-0.013441` (`-8.4%`) | `[-0.018707, -0.008594]` | `[-0.017755, -0.009292]` | `33 / 7 / 0` | `0.000000` | **`0.000000`** |
| `critique_vs_reflect` (`P2 - P1_REFLECT`) | `-0.011607` (`-7.9%`) | `[-0.017541, -0.006126]` | `[-0.016344, -0.007096]` | `30 / 10 / 0` | `0.000160` | **`0.000480`** |
| `p3_vs_silent_poly` (`P3 - P0_POLY`) | `+0.026803` (`+31.8%`) | `[0.020883, 0.033521]` | `[0.021785, 0.032433]` | `2 / 38 / 0` | `0.000000` | **`0.000000`** |
| `p4_vs_silent_poly` (`P4 - P0_POLY`) | `+0.017966` (`+21.3%`) | `[0.012035, 0.023928]` | `[0.012891, 0.023050]` | `5 / 35 / 0` | `0.000000` | **`0.000000`** |
| `p3_vs_virtual_social` (`P3 - VIRTUAL_SOCIAL`) | `-0.005026` (`-4.3%`) | `[-0.008082, -0.002261]` | `[-0.007675, -0.002702]` | `28 / 12 / 0` | `0.003358` | **`0.003358`** |

### 3.3 Pre-Registered Replication P-Series Contrasts (`Holm m = 4`)
| Metric | Contrast | $\Delta$ (`rel`) | `95% Bootstrap CI` | `b` Better / Worse / Tied | `p_exact` | `p_holm` |
|---|---|---|---|---|---|---|
| **$E_2$** | `p1_vs_p0` (`P1 - P0`) | `+0.017035` (`+114.5%`) | `[0.012324, 0.021786]` | `4 / 36 / 0` | `0.000000` | **`0.000000`** |
| **$E_2$** | `p2_vs_p1` (`P2 - P1`) | `-0.008476` (`-26.6%`) | `[-0.011339, -0.005842]` | `35 / 5 / 0` | `0.000000` | **`0.000000`** |
| **$E_2$** | `p4_vs_p3` (`P4 - P3`) | `-0.002219` (`-14.3%`) | `[-0.003719, -0.000545]` | `30 / 10 / 0` | `0.000452` | **`0.000904`** |
| **$E_2$** | `p4_vs_p0` (`P4 - P0`) | `-0.001528` (`-10.3%`) | `[-0.005596, 0.002288]` | `20 / 20 / 0` | `0.636765` | `0.636765` |
| **$\|\bar{x}_2 - y^*\|$** | `p1_vs_p0` (`P1 - P0`) | `+0.056078` (`+54.0%`) | `[0.039417, 0.072500]` | `4 / 36 / 0` | `0.000000` | **`0.000000`** |
| **$\|\bar{x}_2 - y^*\|$** | `p2_vs_p1` (`P2 - P1`) | `-0.025049` (`-15.7%`) | `[-0.033532, -0.017222]` | `36 / 4 / 0` | `0.000000` | **`0.000000`** |
| **$\|\bar{x}_2 - y^*\|$** | `p4_vs_p3` (`P4 - P3`) | `-0.008837` (`-8.0%`) | `[-0.013845, -0.003576]` | `31 / 9 / 0` | `0.000538` | **`0.001076`** |
| **$\|\bar{x}_2 - y^*\|$** | `p4_vs_p0` (`P4 - P0`) | `-0.001748` (`-1.7%`) | `[-0.015015, 0.011187]` | `21 / 19 / 0` | `0.795448` | `0.795448` |

### 3.4 Secondary Family (`Holm m = 2`) & Descriptive Contrasts
| Family / Metric | Contrast (`b - a`) | $\Delta$ (`rel`) | `95% CI` (`90% CI` for descriptive) | `b` Better / Worse / Tied | `p_exact` | `p_holm` |
|---|---|---|---|---|---|---|
| Secondary ($E_2$) | `p3_vs_best_single_model_social` (`P3 - P1_DS`) | `-0.000704` (`-4.3%`) | `[-0.002438, 0.000888]` | `19 / 21 / 0` | `0.785203` | `0.785203` |
| Secondary ($E_2$) | `p4_vs_best_single_model_social` (`P4 - P1_DS`) | `-0.002923` (`-18.0%`) | `[-0.004951, -0.001084]` | `23 / 17 / 0` | `0.012016` | **`0.024032`** |
| Secondary ($\|\bar{x}_2 - y^*\|$) | `p3_vs_best_single_model_social` (`P3 - P1_DS`) | `-0.001357` (`-1.2%`) | `[-0.006823, 0.003770]` | `21 / 19 / 0` | `0.795448` | `0.795448` |
| Secondary ($\|\bar{x}_2 - y^*\|$) | `p4_vs_best_single_model_social` (`P4 - P1_DS`) | `-0.010194` (`-9.1%`) | `[-0.017411, -0.003617]` | `23 / 17 / 0` | `0.024076` | **`0.048152`** |
| Descriptive ($E_2$) | `p3_vs_pooled_jensen_bound` | `-0.002427` (`-13.5%`) | 90%: `[-0.003331, -0.001574]` | `29 / 11 / 0` | `0.000031` | — |
| Descriptive ($E_2$) | `silent_poly_vs_virtual_silent` (Sanity) | `+0.000063` (`+0.7%`) | 90%: `[-0.000513, +0.000621]` | `19 / 21 / 0` | `0.734557` | — |
| Descriptive ($\|\bar{x}_2 - y^*\|$) | `p3_vs_pooled_jensen_bound` | `-0.004088` (`-3.6%`) | 90%: `[-0.006256, -0.002076]` | `25 / 15 / 0` | `0.008264` | — |
| Descriptive ($\|\bar{x}_2 - y^*\|$) | `silent_poly_vs_virtual_silent` (Sanity) | `+0.000321` (`+0.4%`) | 90%: `[-0.002028, +0.002717]` | `20 / 20 / 0` | `0.931152` | — |

### 3.5 Step 8 Parity Comparison (`analyze_controls_preregistered.py` vs. `analyze_numeric_core`)
Full parity details are saved in [audit_replication_report.json](audit_replication_report.json) and [numeric_analysis.json](numeric_analysis.json):

- **Point estimates (`mean_crowd_error` & `mean_diff`) — `100%` IDENTICAL (`0.000000` difference):** Across all `13` physical conditions (`P0`–`P1_GLM`) and every overlapping contrast, `by_condition_final[cond]["crowd_error"]` and `mean_diff` match `preregistered_analysis.json` identically (`max |diff| = 0.0`).
- **Four systematic, fully explained differences on contrasts (`0` unexplained differences):**
  1. **Analysis unit (`n = 40` vs. `n = 120`):** `analyze_controls_preregistered.py` averages replicates within task first (`n = 40` independent tasks per pre-registration), whereas `analyze_numeric_core` treats `(question_id, replicate)` pairs (`n = 120`) as units (`metrics.py` / `numeric_core.py`). Because `n = 120` ignores within-task replicate correlation, `analyze_numeric_core` reports narrower bootstrap 95% CIs (e.g., `p4_vs_p0`: `[-0.004373, +0.000989]` vs. pre-registered `[-0.005596, +0.002288]`).
  2. **Wilcoxon implementation (`exact` vs. normal approximation):** `analyze_controls_preregistered.py` computes exact tie-aware signed-rank probabilities; `analyze_numeric_core` (`paired_wilcoxon_p`) uses a normal approximation on `(qid, rep)` pairs (`n = 120`), producing smaller raw p-values (`herding_glm`: `0.000061` vs. `0.001231`; `p4_vs_p3`: `0.000080` vs. `0.000452`; `p4_vs_p0`: `0.516875` vs. `0.636765`).
  3. **Holm multiplicity family size:** `analyze_controls_preregistered.py` applies Holm across pre-registered families (`m = 8`, `m = 2`, `m = 4`), whereas `analyze_numeric_core` adjusts `contrasts` across all active P-matrix pairs (`m = 20`), `polyphony_contrast_matrix["crowd_error"]` across `10` P-pairs, and `control_contrasts` across `8` pipeline control contrasts (`POOLED_K1_SOCIAL` instead of `VIRTUAL_SOCIAL`).
  4. **Virtual & best-single-model panels absent in `analyze_numeric_core`:** `VIRTUAL_SOCIAL`, `VIRTUAL_SILENT`, and `BEST_K1_SOCIAL` (four pre-registered contrasts: `p3_vs_virtual_social`, `silent_poly_vs_virtual_silent`, `p3_vs_best_single_model_social`, `p4_vs_best_single_model_social`) are intentionally unique to `analyze_controls_preregistered.py`.

---

## 4. Deviations from Pre-Registration

1. **Cold-probe prompt (`Step 5`):** Because no verbatim cold-probe system/user prompt survived from the original 16-task selection, `scripts/probe_replication_tasks.py` used `p_series_system_prompt("2026-11-15")` paired with a question-only user prompt (no prior estimate, no telemetry), `min_output_tokens=8192`, and `reasoning={"effort": "none"}` on DeepSeek/Qwen/GLM (recorded in [PREREGISTRATION_replication_2025.md](../../docs/PREREGISTRATION_replication_2025.md) prior to running the probe).
2. **Budget reservation cap mechanics (`BudgetLedger`):** Because `BudgetLedger` deducts `--request-allowance` before every HTTP attempt without refunding unused allowance upon receiving actual `usage.cost`, we followed Section E by setting `--request-allowance 0.015369` (`1.5 x $0.010246` max observed per-call cost in `polyphony-p0-p4`) and initial `--budget 421.97` (`24,960 x 0.015369 x 1.1`) instead of `--budget 15` / `--request-allowance 0.002`. Neither parameter enters `run_spec` or `numeric_runs.manifest.json`.
3. **Concurrency adjustment & owner-authorized `--budget 445` on resume:** Initial `--workers 8` run (`task-380`) hit an upstream `HTTP 502` outage on `deepseek-v4-flash` at `7,958 / 24,960` cells (`1.282x` retry reservations). Resuming with `--workers 4` (`task-410`, identical manifest/seeds/prompts) lowered retry overhead to `1.088x`, reaching `23,822 / 24,960` (`95.44%`, `$15.96` actual spend) before hitting the `$421.97` reservation ceiling inside `P1_GLM`. Per Hard Rule 6, execution stopped and awaited explicit owner authorization (`--budget 445`, `task-463`), finishing the final `1,138` `P1_GLM` cells cleanly at `$17.19` actual spend.
4. **Synthetic dry-run routing fix (`3c4aedbf`):** Step 6 dry run caught `create_backend("synthetic")` ignoring `--model` and tagging non-routed arms as `openai/gpt-frontier`; fixed pre-live execution with zero effect on `AdaptionBackend`.
5. **Ground-truth sensitivity:** No post-freeze primary-source revisions altered the 40 surviving truths, so no separate revision-sensitivity table was triggered.

---

## 5. Exploratory (Not Pre-Registered)

*(Computed post-completion in [audit_replication_2025.py](../../scripts/audit_replication_2025.py) and saved to [audit_replication_report.json](audit_replication_report.json) and [numeric_analysis.json](numeric_analysis.json); descriptive / hypothesis-generating only.)*

### 5.1 Per-Stratum Breakdown (`D_cold` Cutoffs: `control < 0.000065 <= low_spread < 0.00195 <= high_spread`)
| Stratum (`n` tasks) | Metric | `P0` | `P1` | `P2` | `P0_POLY` | `P3` | `P4` | `BEST_K1_SOCIAL` | `P1 vs P0` (`p_ex`) | `P2 vs P1` (`p_ex`) | `P4 vs P3` (`p_ex`) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `control` (`5`) | $E_2$ | `0.024825` | `0.041115` | `0.026456` | `0.009380` | `0.012631` | `0.009918` | `0.010103` | `+0.016291` (`0.125`) | `-0.014659` (`0.0625`) | `-0.002713` (`0.125`) |
| `low_spread` (`18`) | $E_2$ | `0.013787` | `0.033975` | `0.024834` | `0.008621` | `0.014766` | `0.011535` | `0.009942` | `+0.020187` (`2.3e-05`) | `-0.009141` (`0.000145`) | `-0.003231` (`0.001289`) |
| `high_spread` (`17`) | $E_2$ | `0.013092` | `0.027008` | `0.021055` | `0.010437` | `0.017268` | `0.016265` | `0.019484` | `+0.013915` (`0.000656`) | `-0.005953` (`0.000107`) | `-0.001003` (`0.284`) |
| `control` (`5`) | $\|\bar{x}_2 - y^*\|$ | `0.127450` | `0.174230` | `0.137160` | `0.081666` | `0.097251` | `0.085059` | `0.071031` | `+0.046780` (`0.125`) | `-0.037070` (`0.0625`) | `-0.012191` (`0.0625`) |
| `low_spread` (`18`) | $\|\bar{x}_2 - y^*\|$ | `0.103704` | `0.169286` | `0.142204` | `0.083162` | `0.111398` | `0.098905` | `0.089137` | `+0.065583` (`5.3e-05`) | `-0.027083` (`0.000145`) | `-0.012493` (`0.000252`) |
| `high_spread` (`17`) | $\|\bar{x}_2 - y^*\|$ | `0.097147` | `0.145897` | `0.126537` | `0.085975` | `0.114561` | `0.110582` | `0.122255` | `+0.048750` (`0.001343`) | `-0.019359` (`0.000214`) | `-0.003979` (`0.404`) |

- Notable exploratory pattern: Gemma herding (`P1 - P0`) and `P2 vs P1` / `P2 vs P1_REFLECT` improvements hold strongly across both `low_spread` (`critique_vs_reflect` $E_2$: `p_ex = 0.0208`) and `high_spread` (`p_ex = 0.0150`), whereas the incremental `P4 vs P3` gain concentrates in `low_spread` (`-21.9%, p_ex = 0.0013`) while `P4 vs BEST_K1_SOCIAL` achieves its largest edge (`-16.5%, p_ex = 0.0448` on $E_2$) inside `high_spread` where single-model social herding worsens `P1_DS` (`0.019484`).

### 5.2 Hierarchical Page Decomposition, EVRR, `2 × 2` Interaction & `H4` Regression Slopes
- **Hierarchical Page decomposition (`Round 2`, `by_condition_final`):**
  - Diversity drops sharply under social herding (`P0 = 0.021863` -> `P1 = 0.008661`, `EVRR = 1.1897`), partially recovers under `P1_REFLECT` (`0.009396`) and `P2` (`0.011378`, `EVRR = 1.4220`).
  - In mixed panels, `between_share` rises modestly from `37.99%` (`P0_POLY`) to `39.96%` (`P3`) and `41.70%` (`P4`), while `excess_between_model_diversity` stays near zero (`-0.001145` `P0_POLY`, `-0.000373` `P3`, `+0.000092` `P4`).
- **`2 × 2` Sub-Additive Interaction (`(P4 - P3) - (P2 - P1)`):** Replicates the original `+0.008715` sub-additivity: `crowd_error` DiD = **`+0.006256`** (`95% CI [0.004102, 0.008270]`, `wilcoxon_p = 0.0`, `holm_adjusted_p = 0.0`, `n_cells = 120`); `diversity` DiD = `-0.002121` (`holm = 0.000105`); `evrr` DiD = `-0.157977` (`holm = 0.001496`).
- **Crowd-anchoring OLS slopes (`h4_slopes` across `n = 40` questions):** With `3` replicates (`48` observations per question), crowd-echoing slopes stabilize completely (`P1 slope_crowd = 0.8137` `[0.7063, 0.9085]` -> `P1_REFLECT = 0.6468` -> `P2 = 0.5145` -> `P3 = 0.3991` -> `P4 = 0.2525`; constituent `K=1` social arms: `P1_DS = 0.3025`, `P1_GLM = 0.2484`, `P1_QWEN = 0.2020`). User-prior anchoring falls from `P0 = 0.4169` to `0.0423–0.0907` under telemetry and rebounds partially under dialectical critique (`P2 = 0.1683`, `P4 = 0.0792`).
