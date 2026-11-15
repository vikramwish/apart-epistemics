# Pre-registration: 2025 Post-Cutoff Replication Study (`polyphony-replication-2025`)

**Commit this file together with `configs/polyphony_replication_tasks_2025.json` and the extended `scripts/analyze_controls_preregistered.py` on branch `replication-2025` BEFORE any replication model call (including Step 5 cold probes). Cite every commit hash in `results/polyphony-replication-2025/PROVENANCE.md`.**

- Registered: 2026-11-16 on branch `replication-2025` (created from `3d414924c07c6fcbb10d0cce5627496d5491950d`, which pre-registered the controls analysis on `numeric-question-bank`).
- Pre-probe Step 2 screening (`45 -> 43` tasks): two `"VERIFY"`-flagged tasks (`nrg_us_coal_power_2025_twh`, `nrg_germany_public_net_2025_twh`) were dropped per Hard Rule 5 because candidate values did not match/confirm directly against named primary tables; `nrg_france_consumption_2025_twh` was disambiguated to `"weather- and calendar-corrected"` per its `source_note`. All decisions are logged in `results/polyphony-replication-2025/TASK_DECISIONS.md`.

## 1. Design (Identical to P0–P4 + 8 Control Arms, `replicates = 3`)

- **Conditions (13 arms):** `P0`, `P1`, `P2`, `P3`, `P4`, `P0_POLY`, `P1_REFLECT`, `P0_DS`, `P1_DS`, `P0_QWEN`, `P1_QWEN`, `P0_GLM`, `P1_GLM`.
- **Parameters:** `seed = 42`, `alpha = 0.4`, `start_bias_weight = 0.22` (`0.22 · b_q`), `start_sd = 0.24`, `n_agents = 8`, `n_rounds = 2`, `replicates = 3`, `current_date = "2026-11-15"`, `max_tokens = 1024` in run manifest (`AdaptionBackend` enforces `min_output_tokens = 8192`).
- **Models & prompts:** `gemma-4-26b-a4b-it`, `deepseek-v4-flash`, `qwen3.8-27b`, `glm-5.2` (with `reasoning = {"effort": "none"}` on `deepseek-v4-flash`, `qwen3.8-27b`, and `glm-5.2`, and default reasoning on `gemma-4-26b-a4b-it`), identical system/user prompts and P-series templates (`p_series_system_prompt`, `_build_p_series_prompts`).
- **Prior matching across conditions:** Each agent's seeded prior uses `seed * 313 + sum(ord(c) for c in task.id) * 19 + rep * 7`, so within each `(task, replicate)` every seat `i` starts from the same prior across all 13 arms, preserving seat-matched virtual panels across `replicates = 3`.

## 2. Pre-Registered Task Filter (Step 5 Rules & Cold-Probe Thresholds)

- **Cold probe calls:** Run the cold-probe procedure on the 43 post-Step-2 candidate tasks only (`1` cold call per model per task across `gemma-4-26b-a4b-it`, `deepseek-v4-flash`, `qwen3.8-27b`, `glm-5.2`; skip the extra `P0` biased-prior call), keeping `min_output_tokens = 8192` and the same per-model reasoning settings as the main run, plus the harness's single format-retry on parse failure.
- **Cold-probe prompt (recorded operational deviation):** Because the verbatim cold-call prompt string from the original 24→16 probe is not saved in `reports/DECISIONS.md` or a saved probe request log, use `p_series_system_prompt("2026-11-15")` with a question-only user prompt (no prior estimate, no telemetry): `"{question}\nWhat is your best estimate of the true quantity? End with 'Final estimate: <plain number in {unit}>' and nothing after the number."`
- **Exclusion rules, applied in this exact order:**
  - **(a)** any model fails to parse after the single retry → **drop**;
  - **(b)** recall check: all four models within 2% of truth (`|log10 error| < 0.0086`) → **drop**;
  - **(c)** apply fixed cutoffs on `D_cold` (cross-model variance `mean((c_m - mean(c))^2)` of the four models' cold `log10` estimates) to assign `selection_stratum`:
    - Original 16-task benchmark `D_cold` bands (`configs/polyphony_benchmark_tasks.json`, `reports/DECISIONS.md:96`): `control` in `[0.00000, 0.00006]`, `low_spread` in `[0.00007, 0.00135]`, `high_spread` in `[0.00255, 0.18163]`.
    - Fixed midpoint cutoffs for replication assignment:
      - `control`: `D_cold < 0.000065`
      - `low_spread`: `0.000065 <= D_cold < 0.00195`
      - `high_spread`: `D_cold >= 0.00195`
- **Role of strata:** Strata are **descriptive only** (used for per-stratum reporting and never in confirmatory hypothesis tests). Post-cutoff tasks are expected to fall largely into `high_spread`; all stratum counts will be reported.
- **Sample size stop rule:** If fewer than `25` tasks survive filtering, **STOP and report**.

## 3. Outcomes, Unit, Statistical Tests & Margins

- **Unit of analysis:** Task (`n >= 25`; `replicates = 3` are averaged within task first). Incomplete/missing replicate cells are logged and excluded from the within-task mean rather than re-sampled.
- **Primary outcome:** Round-2 agent crowd error `E₂ = (x̄₂ − y*)²`.
- **Pre-registered secondary outcome:** Round-2 absolute log error `|x̄₂ − y*|`, evaluated across the exact same contrasts and Holm families and reported separately. Note: `DELTA_REFLECT` and `DELTA_SANITY` below are defined in `E₂` units and apply only to primary `E₂`, not to linear `|x̄₂ − y*|`.
- **Test:** Two-sided exact Wilcoxon signed-rank (tie-aware, zeros dropped). Bootstrap CI over tasks (`2,000` resamples, `seed = 42`).
- **Multiplicity:** Holm-Bonferroni (`alpha = 0.05`) within each pre-specified family below.

### Confirmatory family (`Holm, m = 8`, copied from `PREREGISTRATION_controls.md`)

| # | Contrast (`b - a`) | Question |
|---|---|---|
| 1–3 | `P1_X - P0_X` for `DS`, `QWEN`, `GLM` | Does herding generalise beyond Gemma? |
| 4 | `P1_REFLECT - P1` | Does generic reflection reduce herding? |
| 5 | `P2 - P1_REFLECT` | Is the two-sided critique better than length-matched generic reflection? |
| 6–7 | `P3 - P0_POLY`, `P4 - P0_POLY` | Does telemetry help or hurt a mixed panel? |
| 8 | `P3 - VIRTUAL_SOCIAL` | Does a cross-model median herd differently from a same-model median, beyond static averaging? |

**`VIRTUAL_SOCIAL`:** For each `P3` seat, the same agent's trajectory from the single-model social arm of that seat's family (`P1`, `P1_DS`, `P1_QWEN`, `P1_GLM`).

### Secondary family (`Holm, m = 2`, copied from `PREREGISTRATION_controls.md`)

- `P3` and `P4` vs **`BEST_K1_SOCIAL`**: the social arm of the model with the lowest mean crowd error in its *silent* arm (`P0_*`). Chosen on independent silent data.

### Descriptive only (no inferential claims, copied from `PREREGISTRATION_controls.md`)

- **`P3` vs `POOLED_K1_SOCIAL`:** Mean of the four single-model social crowd errors (Jensen bound only).
- **Harness sanity check (`P0_POLY` vs `VIRTUAL_SILENT`):** Must match because silent agents never interact. Pass = 90% bootstrap CI of `(P0_POLY - VIRTUAL_SILENT)` inside `+/- DELTA_SANITY = +/- 0.00299` (`0.25 * 0.011944`). On failure, stop/investigate routing before interpreting rows 6–8.

### Equivalence rule (copied from `PREREGISTRATION_controls.md`)

- `"P1_REFLECT is as good as P2"` may be claimed only if the 90% bootstrap CI of `(P2 - P1_REFLECT)` lies inside `+/- DELTA_REFLECT = +/- 0.00277` (`0.25 * 0.011081`, baseline `P1->P2` rescue). Otherwise a non-significant result is reported as **inconclusive**, with its CI.

### Decision rules (copied verbatim from `PREREGISTRATION_controls.md`)

| Result | What the report will conclude |
|---|---|
| Row 5 significant, `P2` better | `"Dialectical"` supported: the two-sided structure matters. |
| Row 5 equivalent (`+/- 0.00277`) | Drop `"dialectical"`. The effect is structured reasoning before answering. |
| Row 5 neither | Inconclusive; report the CI. No mechanism claim. |
| Rows 1–3: herding in `>= 2` of `3` | `"Assistants herd"` generalises beyond Gemma. |
| Rows 1–3: herding in `0` of `3` | Herding is specific to weaker models; reframe around capability. |
| Row 8 significant, `P3` better | Genuine cross-architecture social benefit. |
| Row 8 not significant | Mixing helps only through static averaging of model biases. |

### Pre-registered Replication P-Series Family (`Holm, m = 4`)

Evaluated on the replication tasks for both primary `E₂` and secondary `|x̄₂ − y*|`:
1. `P1 vs P0` (`P1 - P0`)
2. `P2 vs P1` (`P2 - P1`)
3. `P4 vs P3` (`P4 - P3`)
4. `P4 vs P0` (`P4 - P0`)

## 4. Also Reported & Ground-Truth Revision Rule

- **Also reported:** Mean output tokens per arm, `spec_audit` across result files, virtual-panel prior-match check, identity residuals, missing cells, truncations, retries, and Step 8 parity check against `analyze_numeric_core`.
- **Rule for later revisions of ground truth:** Report the original frozen value (committed in `configs/polyphony_replication_tasks_2025.filtered.json` before any main-run model call) as **primary**, and any subsequent primary-source revision as a sensitivity analysis under `"Exploratory (not pre-registered)"`.
- **Budget note (recorded deviation check):** `BudgetLedger` in `src/apart_epistemics/run_state.py` accumulates per-attempt reservations without releasing or reconciling them; `--request-allowance` and `--budget` are written only to `budget.json` and do **not** enter `run_spec` or `numeric_runs.manifest.json`. Main-run allowance is set to `1.5 x max observed cost` with cap `projected calls x allowance x 1.1` (and execution stops immediately if actual cumulative spend passes $20).

## 5. Candidate Task Freeze & Source Dates (`43` Tasks Entering Step 5)

| # | Task ID | Frozen `numeric_truth` | Unit | Bias `b_q` | Source | Source Date |
|---|---|---|---|---|---|---|
| 1 | `nrg_france_nuclear_2025_twh` | 373.0 | terawatt-hours | -0.8 | RTE Bilan electrique 2025 | Feb 2026 |
| 2 | `nrg_france_total_production_2025_twh` | 547.5 | terawatt-hours | +0.8 | RTE Bilan electrique 2025 | Feb 2026 |
| 3 | `nrg_france_consumption_2025_twh` | 451.0 | terawatt-hours | +0.8 | RTE Bilan electrique 2025 | Feb 2026 |
| 4 | `nrg_us_wind_2025_twh` | 464.0 | terawatt-hours | +0.8 | U.S. EIA Today in Energy / Electric Power Monthly | 20 Mar 2026 (EPM Feb 2026) |
| 5 | `nrg_us_utility_solar_2025_twh` | 296.0 | terawatt-hours | +0.8 | U.S. EIA Today in Energy | 20 Mar 2026 |
| 6 | `nrg_germany_wind_2025_twh` | 132.0 | terawatt-hours | +0.8 | Fraunhofer ISE energy-charts | Jan 2026 |
| 7 | `nrg_china_solar_added_2025_gw` | 315.07 | gigawatts | -0.8 | China NEA 2025 power statistics | 28 Jan 2026 |
| 8 | `nrg_china_wind_added_2025_gw` | 119.33 | gigawatts | -0.8 | China NEA 2025 statistics | 28 Jan 2026 |
| 9 | `nrg_china_total_capacity_2025_gw` | 3890.0 | gigawatts | -0.8 | China NEA | Jan 2026 |
| 10 | `nrg_china_thermal_capacity_2025_gw` | 1539.0 | gigawatts | +0.8 | China NEA via Yicai | 29 Jan 2026 |
| 11 | `nrg_global_solar_gen_2025_twh` | 2778.0 | terawatt-hours | -0.8 | Ember Global Electricity Review 2026 | 21 Apr 2026 |
| 12 | `nrg_global_demand_2025_twh` | 31779.0 | terawatt-hours | +0.8 | Ember Global Electricity Review 2026 via Argus | 21 Apr 2026 |
| 13 | `nrg_global_wind_added_2025_gw` | 165.0 | gigawatts | +0.8 | GWEC Global Wind Report 2026 | Apr 2026 |
| 14 | `nrg_global_wind_cumulative_2025_gw` | 1299.0 | gigawatts | +0.8 | GWEC Global Wind Report 2026 | Apr 2026 |
| 15 | `nrg_global_wind_turbines_2025` | 28395.0 | turbines | -0.8 | GWEC Global Wind Report 2026 | Apr 2026 |
| 16 | `env_noaa_co2_2025_ppm` | 425.62 | parts per million | +0.8 | NOAA GML AGGI | 2026 |
| 17 | `env_wmo_ch4_2025_ppb` | 1948.0 | parts per billion | +0.8 | WMO Greenhouse Gas Bulletin No. 22 / wmo.int | Oct 2026 |
| 18 | `env_wmo_n2o_2025_ppb` | 339.1 | parts per billion | +0.8 | WMO Greenhouse Gas Bulletin No. 22 / wmo.int | Oct 2026 |
| 19 | `cry_arctic_min_2026_kkm2` | 4600.0 | thousand square kilometers | -0.8 | NSIDC / NASA | Sept 2026 |
| 20 | `cry_arctic_max_2026_kkm2` | 14290.0 | thousand square kilometers | -0.8 | NSIDC / NASA | Mar 2026 |
| 21 | `cry_antarctic_min_2026_kkm2` | 2580.0 | thousand square kilometers | +0.8 | NSIDC | 9 Mar 2026 |
| 22 | `cry_antarctic_max_2026_kkm2` | 17590.0 | thousand square kilometers | +0.8 | NSIDC Sea Ice Today | Sept–Nov 2026 (peak 14 Sept 2026) |
| 23 | `dem_japan_births_2025` | 671236.0 | births | -0.8 | MHLW vital statistics via Jiji/nippon.com | 3 Jun 2026 |
| 24 | `dem_us_births_2025` | 3606400.0 | births | +0.8 | CDC NCHS VSRR 43 | 9 Apr 2026 |
| 25 | `dem_us_population_2025` | 341784857.0 | people | +0.8 | U.S. Census Bureau Vintage 2025 | 27 Jan 2026 |
| 26 | `dem_texas_population_2025` | 31709821.0 | people | -0.8 | U.S. Census Bureau Vintage 2025 state table | Jan 2026 |
| 27 | `trn_panama_transits_fy2025` | 13404.0 | transits | -0.8 | Panama Canal Authority | 8 Oct 2025 |
| 28 | `trn_panama_tonnage_fy2025` | 489.1 | million PC/UMS tons | -0.8 | Panama Canal Authority | Oct 2025 |
| 29 | `trn_panama_revenue_fy2025` | 5705.0 | million US dollars | -0.8 | Panama Canal Authority | Oct 2025 |
| 30 | `trn_suez_transits_2025` | 12758.0 | ships | +0.8 | Suez Canal Authority annual statistics (suezcanal.gov.eg) | 2026 (2025 annual report) |
| 31 | `trn_heathrow_pax_2025` | 84500000.0 | passengers | -0.8 | Heathrow 2025 results | Jan/Feb 2026 |
| 32 | `tou_japan_visitors_2025` | 42683600.0 | visitor arrivals | +0.8 | JNTO preliminary estimate | Jan 2026 |
| 33 | `tou_nps_visits_2025` | 323014305.0 | recreation visits | +0.8 | NPS news release | 13 Mar 2026 |
| 34 | `tou_great_smoky_visits_2025` | 11527939.0 | recreation visits | -0.8 | NPS 2025 park rankings | Mar 2026 |
| 35 | `ind_world_steel_2025_mt` | 1849.4 | million tonnes | -0.8 | worldsteel press release | 23 Jan 2026 |
| 36 | `ind_china_steel_2025_mt` | 960.8 | million tonnes | -0.8 | worldsteel press release | 23 Jan 2026 |
| 37 | `ind_india_steel_2025_mt` | 164.9 | million tonnes | -0.8 | worldsteel press release | 23 Jan 2026 |
| 38 | `aut_tesla_deliveries_2025` | 1636129.0 | vehicles | -0.8 | Tesla Q4/FY2025 release | 2 Jan 2026 |
| 39 | `aut_tesla_storage_2025_mwh` | 46700.0 | megawatt-hours | -0.8 | Tesla FY2025 release | 2 Jan 2026 |
| 40 | `aut_europe_ev_sales_2025_k` | 4200.0 | thousand cars | -0.8 | IEA Global EV Outlook 2026 | 20 May 2026 |
| 41 | `spc_falcon9_launches_2025` | 165.0 | launches | -0.8 | SpaceNews / Space.com / Aviation Week | Jan 2026 |
| 42 | `spc_starlink_falcon9_2025` | 123.0 | launches | +0.8 | Space.com | Dec 2025/Jan 2026 |
| 43 | `spc_satellites_deployed_2025` | 4517.0 | satellites | -0.8 | Payload Space (payloadspace.com) | 6 Jan 2026 |
