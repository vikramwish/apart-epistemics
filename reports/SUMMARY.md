# Apart Epistemics / Epistemic Prism (`numeric-question-bank`) Summary

- **Repository / Remote:** `https://github.com/vikramwish/apart-epistemics.git` (`origin`)
- **Branch:** `numeric-question-bank`
- **Standalone Research Report:** [`reports/POLYPHONY_EXPERIMENT_REPORT.md`](POLYPHONY_EXPERIMENT_REPORT.md)

## Shipped in Clean-Room Phase 2 (`P0`–`P4` Cross-Architecture Polyphony & Dialectical Counter-Anchoring)
- **Protocol, Benchmark & Engine Upgrades:**
  - Authored a zero-overlap 12-question Scientific, Engineering & Earth-Systems Fermi Benchmark ([`configs/polyphony_benchmark_tasks.json`](../configs/polyphony_benchmark_tasks.json); 4 `control`, 4 `low_spread`, 4 `high_spread`, balanced `+/-0.8` prior misconceptions, all `numeric_truth >= 305.0`) and suite config [`configs/polyphony_p0_p4.yaml`](../configs/polyphony_p0_p4.yaml) (`seed=42`, `num_agents=8`, `num_rounds=2`, `alpha=0.3`).
  - Added clean-room `P0`–`P4` prompts (`p_series_system_prompt`, `_P_COUNTER_ANCHOR_TEXT`), generalized `2x2` DiD interaction logic (`(P4 - P3) - (P2 - P1)`), pre-registered `polyphony_contrast_matrix` serialization (9 pairs $\times$ 5 metrics with per-metric Holm adjustment + `allow_nan=False`), H4 `P0 -> P1/P2` `crowd_pull_vs_c1_r1` support, and synthetic parity across `P0`–`P4` while keeping `C1`–`C6` intact.
  - Verified full suite: `ruff format --check src tests`, `ruff check src tests`, and `pytest` unit tests passing cleanly alongside machine-precision hierarchical Page residuals (`max_flat_res = 1.874e-16`, `max_hier_res = 1.943e-16` across all 120 cells).
- **Live 960-Row Adaption Run (`results/polyphony-p0-p4/`):**
  - Completed `960 / 960` rows across `P0`–`P4` (`192` homogeneous `gemma-4-26b-a4b-it` in each of `P0`, `P1`, `P2`; `48` each across `gemma-4-26b-a4b-it`, `deepseek-v4-flash`, `qwen3.8-27b`, and `glm-5.2` in `P3` and `P4` with `reasoning={"effort": "none"}` for non-Gemma prefixes), `0 / 960` misses, `0 / 960` final truncations (transparently re-sampling 3 initial Gemma `finish_reason="length"` repetition loops and their 2 downstream Round-2 trajectories = 19 rows via crash-resume), `$0.3840 USD` total provider cost (`2,352` HTTP requests reserved `$4.704`).
  - **Primary Findings:**
    - `between_model_diversity` increases from `0.0` (`P0`–`P2`) to `0.004750` (`48.85%` share, `excess = +0.001024`, Holm `p = 0.004395`) in `P3` and `0.004462` (`47.46%` share, `excess = +0.000622`, Holm `p = 0.004395`) in `P4`.
    - Even on canonical physical constants, unanchored cross-architecture social telemetry (`P3`) raises `crowd_error` relative to `P0` (`0.010340` vs. `0.009932`, `+0.000409`, Wilcoxon `p = 0.007812`, `95% CI [+0.000030, +0.001097]`, matrix Holm `p = 0.070312`) and increases Round-2 assistant-only MSE (`0.000616 -> 0.000940`).
    - Dialectical counter-anchoring (`P4`) cuts Round-2 assistant MSE by **92.4%** vs. `P3` (`0.000940 -> 0.000071`) and recovers the lowest final `crowd_error` of all five conditions (`0.009928`).

## Cross-Benchmark Replication (`C1`–`C6`, Phase 1)
- **Live 1,056-Row V3 Adaption Run (`results/numeric-polyphony-c1-c6/`):** Completed on `11` unsettled/municipal V3 questions (`configs/numeric_polyphony_c1_c6.yaml`), `0 / 1,056` misses, `0` truncations, `$0.3479 USD` (`1,221` requests), `max_identity_residual = 3.123e-16`. Under high baseline uncertainty (`C1` Round-2 assistant MSE `0.055792`), homogeneous social consensus collapses accuracy (`C2 = 0.039766` vs. `C1 = 0.021548`, `crowd_pull = 0.9550`), whereas combined polyphony + counter-anchoring (`C6`) cuts `crowd_error` by **47.6%** vs. `C2` (`0.020837`, Wilcoxon `p = 0.041992`) while elevating `evrr` (`1.414287` vs. `1.194963`, `p = 0.032227`).

## Open Risks & Next Steps
- **OpenRouter Credits (`reports/NEEDS_OWNER.md`):** OpenRouter account credits are exhausted (`HTTP 402`), halting full 17,820-row rounded sweeps on `openai/gpt-6-luna`; Adaption funded the completed 11-question V3 (`396` rows), 11-question `C1`–`C6` polyphony (`1,056` rows), clean-room 12-question `P0`–`P4` scientific polyphony (`960` rows), `deepseek-v4-flash` (`96` rows), `gemma-4-31b-it` (`24` numeric rows), and 4-model Track 4 (`80` rows) experiments.
- **Statistical Power & Prior-Inertia Horizon:** `P0`–`P4` uses `n = 12` questions (`min exact Wilcoxon p = 0.000488`) and `alpha = 0.3` over `2` rounds (`0.49` initial-prior weight), compressing crowd-level differences on solved canonical constants despite the `92.4%` assistant-MSE improvement in `P4`. Next steps: expand the scientific polyphony bank to `n >= 24` uncertain pre-cutoff constants and evaluate `num_rounds >= 4` / higher `alpha`.
