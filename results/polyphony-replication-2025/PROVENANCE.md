# Provenance Log — `polyphony-replication-2025`

- Base branch before replication: `numeric-question-bank` (`3a6294c36b538bcb22d632b721be7ccb410acd9f`).
- Replication branch: `replication-2025` (created from commit `3d414924c07c6fcbb10d0cce5627496d5491950d`).

## Commit Hashes (Strict Chronological Order Before Any Replication API Calls)

1. **`5ad9f348ace1b49d5399b2aa38300b4bcb433944`** (`numeric-question-bank`)
   - Message: `Control-arm harness as executed by run started Mon Nov 16 01:33:44 2026 AEDT`
   - Files: `configs/polyphony_controls.yaml`, `scripts/analyze_polyphony_controls.py`, `scripts/audit_and_reproduce_p0_p4.py`, `src/apart_epistemics/backends/synthetic.py`, `src/apart_epistemics/cli.py`, `src/apart_epistemics/engine/h4.py`, `src/apart_epistemics/engine/numeric_core.py`, `tests/test_engine_numeric.py` (`providers.py` had zero diff).
   - Verified pre-commit that all harness file mtimes (`<= 2026-11-16 01:12:03`) preceded PID `24020` start (`~2026-11-16 01:33:43–01:33:45`).

2. **`3d414924c07c6fcbb10d0cce5627496d5491950d`** (`numeric-question-bank`, base of `replication-2025`)
   - Message: `Pre-register controls analysis (before inspecting results)`
   - Files: `docs/PREREGISTRATION_controls.md`, `scripts/analyze_controls_preregistered.py`, `scripts/synth_null_test.py`.
   - Committed prior to reading any outputs or errors under `results/polyphony-controls/`.

3. **`9ccd45798315ea53a3cd9f425281cbf9e19ba29d`** (`replication-2025`)
   - Message: `Pre-register 2025 replication study (Steps 1-4)`
   - Files: `configs/polyphony_replication_tasks_2025.json` (`43` post-Step-2 verified tasks), `scripts/analyze_controls_preregistered.py` (added pre-registered secondary metric `|xbar_2 - y*|`, `REPLICATION_P_SERIES` Holm `m=4`, and `replicates > 1` missing-cell resilience), `tests/test_controls_preregistered.py`, `docs/PREREGISTRATION_replication_2025.md`, `results/polyphony-replication-2025/ORIENTATION.md`, `results/polyphony-replication-2025/TASK_DECISIONS.md`.
   - Committed prior to any cold-probe or main-run API calls.

4. **`6bab85eecd0cace2d170e49979da32a71d331ced`** (`replication-2025`)
   - Message: `Commit Step 5 cold-probe filter and 2025 replication config (40 tasks)`
   - Files: `configs/polyphony_replication_tasks_2025.filtered.json` (`40/43` surviving tasks: `control=5`, `low_spread=18`, `high_spread=17`; `3` dropped under rule (b) recall check, `0` parse misses), `configs/polyphony_replication_2025.yaml`, `scripts/probe_replication_tasks.py`, `tests/test_probe_replication_tasks.py`, `results/polyphony-replication-2025/cold_probe_errors.md`, `results/polyphony-replication-2025/cold_probe/*`.
   - Committed prior to Step 6 dry run and any non-probe API calls.

5. **`3c4aedbf56fb83bc6ba3fc71fc734b41303c5445`** (`replication-2025`)
   - Message: `Honor --model flag on SyntheticEpistemicBackend for dry-run parity`
   - Files: `src/apart_epistemics/backends/synthetic.py`, `src/apart_epistemics/backends/__init__.py`, `tests/test_backends.py`.
   - Caught by Step 6 dry run (without setting `self.model`, synthetic backend fell back to `openai/gpt-frontier` on `P0`/`P1`/`P2`/`P1_REFLECT`). Step 6 re-run with `--model gemma-4-26b-a4b-it` verified `24,960/24,960` cells, `diff_keys = []`, `pipeline_merge_will_succeed = true`, and identical manifest across all `13` conditions (`results/polyphony-replication-2025/dry_run/`).

6. **`237e639a07de6e0215b8db60f62e9c3d1880e69c`** (`replication-2025`)
   - Message: `Record Step 5 and Step 6 pre-run provenance hashes`
   - Final code/config/preregistration lock on `replication-2025` prior to any Step 7 non-probe live calls.

## Execution Runs & Final Artifact Checksums (Steps 7–9)

- **Main Run Segments (all sharing identical `numeric_runs.manifest.json`, zero re-sampling):**
  1. Initial launch (`--workers 8 --budget 421.97 --request-allowance 0.015369`): completed `7,958 / 24,960` cells (`$2.597242`) before an upstream `deepseek-v4-flash` HTTP 502 error exhausting `25` retries.
  2. Resumed (`--workers 4 --budget 421.97 --request-allowance 0.015369`): reached `23,822 / 24,960` cells (`$15.961529`) before hitting `BudgetExceededError` at `$421.956 / $421.97` (`27,455` reservations).
  3. Owner-authorized completion (`--workers 4 --budget 445 --request-allowance 0.015369`): completed all `24,960 / 24,960` cells (`25,108` HTTP successes including `148` format retries; `28,744` reservations = `$441.766536 / $445.00`; total main spend `$17.193638`; cold probe spend `$0.086311`; combined `$17.279949`).
- **Local Uncommitted Raw Logs (excluded from git due to >100 MB size limit):**
  - `results/polyphony-replication-2025/numeric_runs.jsonl` (`637 MB`): `sha256:041a8170379b3be671de2aa6a1e1017b8cc64c950f8511b5e71c59968fef272a`
  - `results/polyphony-replication-2025/provider_requests.jsonl` (`166 MB`): `sha256:78aa027773e3fbd52a26b7752809206a72ee89007dbc1ac031997e419008460a`
- **Tracked Primary Outputs (committed in Step 9):**
  - `results/polyphony-replication-2025/numeric_runs.manifest.json`: `sha256:669727d6c01cbaa669826c3db750038f87e879dfb7ac6edb14643af323466799`
  - `results/polyphony-replication-2025/preregistered_analysis.json`: `sha256:d48c0d36f72aa6bfac233f5d12655cb434baf9f0a6dfd6c40be6d52045e88406`
  - `results/polyphony-replication-2025/numeric_analysis.json`, `results/polyphony-replication-2025/audit_replication_report.json`, `results/polyphony-replication-2025/budget.json`, `results/polyphony-replication-2025/REPORT.md`, `scripts/audit_replication_2025.py`.
