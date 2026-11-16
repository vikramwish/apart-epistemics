# Experiment Summary — Standalone `P0`–`P4` Cross-Architecture Polyphony & Counter-Anchoring Benchmark

- **Deliverable Scope:** Self-contained 16-task scientific quantitative estimation study (`P0`–`P4`, `alpha = 0.4`, `8` agents, `2` rounds, `1,280` live inference rows) in [`results/polyphony-p0-p4/`](../results/polyphony-p0-p4/) accompanied by 3 generated SVG figures in [`reports/figures/`](figures/) and full methodology/results in [`POLYPHONY_EXPERIMENT_REPORT.md`](POLYPHONY_EXPERIMENT_REPORT.md).
- **Task Bank & Integrity:** Zero-ceiling cold-probed 16-question benchmark ([`configs/polyphony_benchmark_tasks.json`](../configs/polyphony_benchmark_tasks.json): `6 high_spread`, `6 low_spread`, `4 control`, exact `+/-0.8` bias balance per stratum), isolated neutral baseline prompt (`p_series_system_prompt`), `1,280/1,280` rows completed, **`0/1,280` parse misses**, **`0/1,280` final truncations** (`6` total `finish_reason="length"` Gemma repetition loops causally re-sampled across `28 + 1` crash-resume cells), **`max_identity_residual = 1.3878e-16`**, and **`$0.4863 USD`** actual logged cost (`229,653` input / `531,150` output tokens; `0/1,280` `reasoning_tokens > 0`).

---

## Final Round-2 Summary (`P0`–`P4`, `n = 16` questions / cell mean)

| Condition | Setup | Agent `crowd_error` | Asst `crowd_error` | Agent `diversity` | `D_within` | `D_between` | `D_excess` | `EVRR` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **`P0`** | Mono Gemma, Alone | `0.011944` | `0.009646` | `0.016806` | `0.016806` | `0.000000` | `0.000000` | `1.3788` |
| **`P1`** | Mono Gemma + Peer Telemetry | `0.022899` | `0.017313` | `0.009557` | `0.009557` | `0.000000` | `0.000000` | `1.0700` |
| **`P2`** | Mono Gemma + Telemetry + CA | `0.011818` | `0.013480` | `0.012854` | `0.012854` | `0.000000` | `0.000000` | `1.3394` |
| **`P3`** | Poly `K=4` + Telemetry | `0.010058` | **`0.002339`** | `0.009021` | `0.004178` | `0.004842` | `0.001709` | `1.1060` |
| **`P4`** | Poly `K=4` + Telemetry + CA | **`0.007692`** | `0.005770` | `0.013842` | `0.004835` | `0.009007` | **`0.005381`** | `1.2826` |

## Key Empirical Conclusions
1. **Monoculture Peer Herding (`P1` vs. `P0`):** Exposing homogeneous agents to peer median/IQR nearly doubles blended crowd MSE (`0.011944 -> 0.022899` across `13/16` questions, `+0.010955`, Wilcoxon `p = 0.008266`, matrix Holm `p = 0.049594`), cuts direct assistant diversity by `81.4%` (`0.014055 -> 0.002611`), and induces a Round-1 crowd-pull slope of `+0.4489`.
2. **Counter-Anchoring Alone (`P2` vs. `P1`):** Symmetrical tail-testing on fixed $K=1$ `gemma-4-26b-a4b-it` (zero model swap) drops crowd error `-48.4%` (`0.022899 -> 0.011818`, `14/16` wins, Wilcoxon `p = 0.000892`, matrix Holm `p = 0.008027`, global 20-contrast Holm `p = 0.017838`) and eliminates crowd pull (`slope = -0.0254`).
3. **Polyphony Alone (`P3` vs. `P1`):** Mixing `gemma-4-26b-a4b-it`, `deepseek-v4-flash`, `qwen3.8-27b`, and `glm-5.2` cuts blended crowd error `-56.1%` (`0.010058`, `12/16` wins, matrix Holm `p = 0.022925`) and slashes **direct assistant crowd MSE by `86.5%`** (`0.017313 -> 0.002339`, matrix Holm `p = 0.031001`), generating `53.7%` between-model diversity (`0.004842`, matrix Holm `p = 0.004342`).
4. **Joint Polyphony + Counter-Anchoring (`P4`):** Delivers the lowest final blended crowd MSE (`0.007692`, `-66.4%` vs. `P1`, `14/16` wins, matrix Holm `p = 0.022925`; `P4 < P3` across all 3 strata, `p_W = 0.028768`), highest model-cluster variance (`between_model_diversity = 0.009007`, `excess = +0.005381`), and a statistically significant sub-additive `2x2` DiD (`+0.008715`, `95% CI [+0.003359, +0.016034]`, Holm `p = 0.005458`), proving diversity-driven error cancellation (`P4` beats `P3` even though `avg_individual_error` is `0.021534` vs. `0.019078`).

---

## Generated Figures & Code Verification

- [`figures/polyphony_crowd_error.svg`](figures/polyphony_crowd_error.svg) — P0–P4 agent vs. direct-assistant crowd MSE, R1->R2 deltas, and confirmatory Holm contrasts.
- [`figures/polyphony_hierarchical_page.svg`](figures/polyphony_hierarchical_page.svg) — Two-level Page identity decomposition (`E_ind = E_crowd + D_within + D_between`), `between_share`, `D_excess`, `EVRR`, and `1.39e-16` residual verification.
- [`figures/polyphony_2x2_factorial.svg`](figures/polyphony_2x2_factorial.svg) — 3-panel `2x2` interaction plot (`crowd_error`, `assistant_crowd_error`, `evrr`) with `P0` / `1.0` reference lines and exact `DiD` Holm statistics.
- **Verification Gates:** Strict JSON (`allow_nan=False` everywhere including `_cmd_analyze`), offline `ruff format --check` + `ruff check` + full `pytest` suite (`241 passed, 2 skipped` on live-key/external calibration skips), and secret-diff checks.

## Open Risks & Next Steps
- **Open Risks:** `n = 16` questions provides conservative power under the global `20`-contrast Holm family (`P4`/`P3` vs. `P1` reach `p_Holm = 0.0544` vs. `0.0229` within the `9`-contrast metric family); non-Gemma families ran with `reasoning={"effort": "none"}`.
- **Next Steps:** Expand replicate depth or compare adaptive trust weighting (`alpha` schedules) alongside enabled reasoning budgets in `P3`/`P4` to test whether `P4`'s variance-cancellation advantage persists when raw model accuracy further improves.
