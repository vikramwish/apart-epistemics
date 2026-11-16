# Step 0 Orientation — `polyphony-replication-2025`

Status: **STOPPED at Step 0 under Hard Rule 6** (zero model calls made; zero existing files or results modified).

## 1. Repository & Active Process State

- Expected base branch `numeric-question-bank` HEAD: `3a6294c36b538bcb22d632b721be7ccb410acd9f`.
- That commit contains `P0`–`P4` only. The 8 control arms (`P0_POLY`, `P1_REFLECT`, `P0_DS`, `P1_DS`, `P0_QWEN`, `P1_QWEN`, `P0_GLM`, `P1_GLM`), `configs/polyphony_controls.yaml`, `scripts/analyze_polyphony_controls.py`, and `scripts/audit_and_reproduce_p0_p4.py` are present in the working tree as uncommitted edits/untracked files alongside report/submission edits.
- Active process observed during Step 0 (`PID 24020`, elapsed `54:58`):
  ```text
  /Users/vikramviswanath/apart_epistemics/.venv/bin/python .venv/bin/apart-epistemics run --config configs/polyphony_controls.yaml --backend adaption --model gemma-4-26b-a4b-it --current-date 2026-11-15 --numeric-only --workers 8 --budget 10 --request-allowance 0.002 --out results/polyphony-controls
  ```
- Branch `replication-2025` was **not** created yet to avoid interfering with the running process or presuming which uncommitted edits belong on `numeric-question-bank`.

## 2. Exact Config, CLI Flags & Harness Behavior Found

- **`configs/polyphony_p0_p4.yaml`**:
  - `seed: 42`, `num_agents: 8`, `num_rounds: 2`, `tasks_per_track: 16`, `numeric_tasks: 16`, `numeric_tasks_file: "configs/polyphony_benchmark_tasks.json"`
  - `numeric_start_bias_weight: 0.22`, `numeric_start_sd: 0.24`, `numeric_replicates: 1`, `alpha: 0.4`
  - `numeric_conditions: [P0, P1, P2, P3, P4]`
  - `numeric_models: [gemma-4-26b-a4b-it, deepseek-v4-flash, qwen3.8-27b, glm-5.2]`
- **`configs/polyphony_controls.yaml`**:
  - Identical parameters with `numeric_conditions: [P0_POLY, P1_REFLECT, P0_DS, P1_DS, P0_QWEN, P1_QWEN, P0_GLM, P1_GLM]`.
- **Recorded run-spec manifests (`results/polyphony-p0-p4/numeric_runs.manifest.json` & `results/polyphony-controls/numeric_runs.manifest.json`)**:
  - `version: 1`, `seed: 42`, `alpha: 0.4`, `n_agents: 8`, `n_rounds: 2`, `replicates: 1`, `max_tokens: 1024`, `start_bias_weight: 0.22`, `start_sd: 0.24`, `current_date: "2026-11-15"`
  - `backend`: `class: AdaptionBackend`, `model: gemma-4-26b-a4b-it`, `base_url: https://api.prod.adaptionlabs.ai/api/v1`
  - `numeric_models: ["gemma-4-26b-a4b-it", "deepseek-v4-flash", "qwen3.8-27b", "glm-5.2"]`
- **Backend behavior (`src/apart_epistemics/backends/providers.py:366-368, 410`)**:
  - `AdaptionBackend` enforces `min_output_tokens=8192` via `max(int(max_tokens), self.min_output_tokens)` on API calls (even though the manifest stores `max_tokens: 1024`), sets `reasoning={"effort": "none"}` on `qwen*`, `glm*`, `deepseek*` (leaving `gemma*` default), and retries with `max_retries=25`.
- **Budget files (`results/polyphony-p0-p4/budget.json` & `results/polyphony-controls/budget.json`)**:
  - `per_request_allowance_usd: 0.002`, `cap_usd: 10.0`.
- **Merge & Control Contrasts (`src/apart_epistemics/engine/numeric_core.py`)**:
  - `_merge_compatible_run_specs` compares run specs across runs while ignoring `"conditions"` and `"numeric_models"`.
  - `_CONTROL_CONTRAST_SPECS` (`numeric_core.py:1097-1106`) defines the 8 control contrasts (`p3_vs_p0_poly_crowd_error`, `p4_vs_p0_poly_crowd_error`, `p2_vs_p1_reflect_crowd_error`, `p1_reflect_vs_p1_crowd_error`, `p1_ds_vs_p0_ds_crowd_error`, `p1_qwen_vs_p0_qwen_crowd_error`, `p1_glm_vs_p0_glm_crowd_error`, `p3_vs_pooled_k1_social_crowd_error`).

## 3. Cold-Probe & Stratum Assignment Findings

- No standalone script for the 16-task Phase-3 cold probe exists in `scripts/` or git history.
- Built-in `src/apart_epistemics/datasets/selection.py` (`select-questions` / `run_cold_selection` with `rule="stratified"`) defaults to `repeats=8, keep=10, min_error=0.1, review_error=0.5, max_miss_rate=0.125`, uses `cold_prompts(task, current_date)` -> `shared_system_prompt(current_date)`, and does **not** assign a `"control"` stratum, run 4-model single-shot probing, or apply the `|log10 error| < 0.0086` recall filter.
- Per `docs/DECISIONS.md:95-98`, the 16-task `configs/polyphony_benchmark_tasks.json` selection ran an ad-hoc 5-call-per-task probe (4 cold calls across `gemma-4-26b-a4b-it`, `deepseek-v4-flash`, `qwen3.8-27b`, `glm-5.2` + 1 `P0` biased-prior call on Gemma, `min_output_tokens=8192`), kept items with `max(|c_m - log_truth|) <= 1.7` (all 24 had `<= 0.957`), selected 12 non-ceiling + top 4 ceiling-with-signal items ranked by `S = D_cold + E_cold + 0.25*|y_biased - c_gemma|`, and assigned strata with observed `D_cold` ranges:
  - `high_spread` (6 tasks): `D_cold` in `[0.00255, 0.18163]`
  - `low_spread` (6 tasks): `D_cold` in `[0.00007, 0.00135]`
  - `control` (4 tasks): `D_cold` in `[0.00000, 0.00006]`
- Each stratum in `configs/polyphony_benchmark_tasks.json` has `popular_misconception_bias` balanced 50/50 between `-0.8` and `0.8`.

## 4. Candidate Tasks Pre-Check (`/Users/vikramviswanath/Downloads/polyphony_replication_tasks_2025.json`)

Read-only validation before copying into `configs/`:
- **45 tasks**, all valid under `apart_epistemics.models.TaskItem.model_validate`.
- **45 unique IDs**, zero overlap with the 16 IDs in `configs/polyphony_benchmark_tasks.json`.
- **`numeric_truth` range**: `119.33` to `341784857.0` (all `>= 100`).
- **8 tasks containing `"VERIFY"` in `source_note`**:
  1. `nrg_france_consumption_2025_twh`
  2. `nrg_us_coal_power_2025_twh`
  3. `nrg_germany_public_net_2025_twh`
  4. `env_wmo_ch4_2025_ppb`
  5. `env_wmo_n2o_2025_ppb`
  6. `cry_antarctic_max_2026_kkm2`
  7. `trn_suez_transits_2025`
  8. `spc_satellites_deployed_2025`

## 5. Hard Rule 6 Blockers Requiring Owner Input

1. **Missing replication-kit files** referenced by Steps 1, 3, 4, 6, and 8:
   - `analyze_controls_preregistered.py` (needed for Steps 1, 3, 6, 8)
   - Controls pre-registration document containing `DELTA_REFLECT`, `DELTA_SANITY`, secondary family (`m=2`), descriptive contrasts, and decision rules (needed for Steps 1, 4)
   - `synth_null_test.py` (needed for Step 3 null calibration test)
   None of these three files exist in the repo, git history/worktrees/stashes, `~/Downloads/`, or elsewhere on the system.
2. **Git base state & active concurrent run**:
   - Control arms (`P0_POLY`, `P1_REFLECT`, `P0_DS`..`P1_GLM`, `configs/polyphony_controls.yaml`) are uncommitted on `numeric-question-bank` while `PID 24020` is actively writing to `results/polyphony-controls/`.
   - Clarify whether `replication-2025` should be branched after `polyphony-controls` finishes and the control-arm code is committed to `numeric-question-bank`, or how to isolate code vs manuscript edits.
3. **Cold-probe strata thresholds & system prompt (Step 5)**:
   - Clarify the exact `D_cold` (and/or `E_cold`) cutoffs for `high_spread` / `low_spread` / `control` on `N >= 25` surviving tasks (fixed thresholds vs quantiles) and whether the cold probe uses `shared_system_prompt(current_date)` (`selection.py:cold_prompts`) or `p_series_system_prompt`.
4. **Budget reservation allowance note (Step 7)**:
   - At `replicates=3` across `13` conditions (`8 agents * 2 rounds = 624` calls per task, or `>= 15,600` calls for `>= 25` tasks), using `--request-allowance 0.002` reserves `>= $31.20` and will trigger the `$15` budget cap unless a lower per-request allowance is specified or expected spend per request is much smaller.
