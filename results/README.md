# Recorded live runs

Live runs on the Adaption API behind the submitted paper
([`reports/manuscript/main.pdf`](../reports/manuscript/main.pdf)) and its post-sprint
pre-registered replication. Earlier pilot, question-selection, and exploratory runs are kept
out of this branch (they remain in git history before the submission cleanup).

| Folder | What it is | Calls | Cost |
| --- | --- | --- | --- |
| `polyphony-p0-p4/` | **Paper's primary study.** 16 cold-probed tasks (`configs/polyphony_benchmark_tasks.json`) x `P0`–`P4` x 8 agents x 2 rounds x 1 replicate = `1,280` rows, `alpha = 0.4`. `gemma-4-26b-a4b-it` in `P0`–`P2`; 4-model split (`gemma-4-26b-a4b-it`, `deepseek-v4-flash`, `qwen3.8-27b`, `glm-5.2`) in `P3`/`P4`. `0` misses, `max_identity_residual = 1.388e-16`. Audit outputs: `audit_reproduction_report.json`, `first_attempt_reconstruction.json` (`scripts/audit_and_reproduce_p0_p4.py`). Analysis: [final manuscript](../reports/manuscript/main.pdf) | 1,318 | $0.5014 |
| `polyphony-p0-p4-12task-alpha03/` | Superseded earlier 12-task, `alpha = 0.3` run (`960` rows), preserved unmodified because the paper discloses it | 979 | — |
| `polyphony-controls/` | Follow-up control arms on the same 16 tasks (`P0_POLY`, `P1_REFLECT`, homogeneous DeepSeek/Qwen/GLM `P0`/`P1`), analysis pre-registered in [`docs/PREREGISTRATION_controls.md`](../docs/PREREGISTRATION_controls.md) | — | — |
| `polyphony-replication-2025/` | **Pre-registered out-of-sample replication** of `P0`–`P4` + controls: 40 tasks about 2025–2026 quantities (training cutoffs unverified) x 3 replicates = `24,960` cells. See [`REPORT.md`](polyphony-replication-2025/REPORT.md), [`PROVENANCE.md`](polyphony-replication-2025/PROVENANCE.md), and [`docs/PREREGISTRATION_replication_2025.md`](../docs/PREREGISTRATION_replication_2025.md) | 25,280 | $17.28 |

## What is in each folder

- **`numeric_runs.jsonl`**: one row per agent-round; **`numeric_runs.manifest.json`**: the exact
  run settings; **`numeric_analysis.json`** / **`preregistered_analysis.json`**: contrasts and
  per-round metrics.
- **`provider_requests.jsonl`**: every request sent and response received, one per line (the
  complete record; the other files are derived from it). Request headers and credentials are
  never written to it.
- **`budget.json`**: the request allowance ledger (`reserved_usd` is a conservative reservation,
  not the real bill; the real cost per call is in each response's `usage.cost`).

The replication's raw `numeric_runs.jsonl` (637 MB) and `provider_requests.jsonl` (166 MB) exceed
GitHub's file-size limit and are not committed; their SHA-256 checksums are recorded in
`polyphony-replication-2025/PROVENANCE.md`.

Configuration dry runs (`make smoke`) make no API calls. New experiment output goes to ignored `tmp/`.
