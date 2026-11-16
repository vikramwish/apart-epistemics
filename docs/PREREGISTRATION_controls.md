# Pre-registration: polyphony control arms

**Commit this file and `scripts/analyze_controls_preregistered.py` before reading any control-arm output. Cite the commit hash in the paper.**

- Registered: 2026-11-16, while `configs/polyphony_controls.yaml` (8 arms, 2,048 calls) was running and before any control results had been inspected.
- Frozen baseline: `results/polyphony-p0-p4/` (P0–P4, 1,280 rows), unchanged.

## Design (identical to P0–P4)
16 tasks, 8 agents, 2 rounds, α = 0.4, seed 42, start bias 0.22·b_q, start SD 0.24. Each agent's seeded prior depends only on task, seed and replicate, so agent *i* starts from the same prior in every arm. That is what makes the seat-matched virtual panels below valid.

## Unit, outcome, tests
- **Unit:** task (n = 16; replicates averaged within task first). **Outcome:** Round-2 agent crowd error E₂ = (x̄₂ − y*)².
- **Test:** two-sided exact Wilcoxon signed-rank (tie-aware, zeros dropped). Bootstrap CI over tasks (2,000 resamples, seed 42).
- **Multiplicity:** Holm within each family below.

## Confirmatory family (Holm, m = 8)
| # | Contrast (b − a) | Question |
|---|---|---|
| 1–3 | P1_X − P0_X for DS, QWEN, GLM | Does herding generalise beyond Gemma? |
| 4 | P1_REFLECT − P1 | Does generic reflection reduce herding? |
| 5 | P2 − P1_REFLECT | Is the two-sided critique better than length-matched generic reflection? |
| 6–7 | P3 − P0_POLY, P4 − P0_POLY | Does telemetry help or hurt a mixed panel? |
| 8 | P3 − VIRTUAL_SOCIAL | Does a cross-model median herd differently from a same-model median, beyond static averaging? |

**VIRTUAL_SOCIAL:** for each P3 seat, the same agent's trajectory from the single-model social arm of that seat's family (P1, P1_DS, P1_QWEN, P1_GLM).

## Secondary family (Holm, m = 2)
P3 and P4 vs **BEST_K1_SOCIAL**: the social arm of the model with the lowest mean crowd error in its *silent* arm (P0_*). The model is chosen on independent data, to avoid picking the winner on the same numbers it is then compared on.

## Descriptive only (no inferential claims)
- **P3 vs the original `POOLED_K1_SOCIAL`.** This is the mean of the four single-model crowd errors. By Jensen's inequality P3 beats it whenever the models' biases differ, so it is reported as a bound, not as a test of diversity. In 20/20 null simulations, P3 "beat" it.
- **Harness check, P0_POLY vs VIRTUAL_SILENT.** These must match, because silent agents never interact. Pass = 90% CI of the difference within ±0.00299 (25% of baseline P0 error). On failure, investigate routing before interpreting rows 6–8.

## Equivalence rule
"P1_REFLECT is as good as P2" may be claimed only if the 90% bootstrap CI of (P2 − P1_REFLECT) lies inside ±0.00277, which is 25% of the baseline P1→P2 rescue of 0.011081. Otherwise a non-significant result is reported as **inconclusive**, with its CI.

## Decision rules (written before results)
| Result | What the paper will say |
|---|---|
| Row 5 significant, P2 better | "Dialectical" supported: the two-sided structure matters. |
| Row 5 equivalent | Drop "dialectical". The effect is structured reasoning before answering. |
| Row 5 neither | Inconclusive; report the CI. No mechanism claim. |
| Rows 1–3: herding in ≥ 2 of 3 | "Assistants herd" generalises beyond Gemma. |
| Rows 1–3: herding in 0 of 3 | Herding is specific to weaker models; reframe around capability. |
| Row 8 significant, P3 better | Genuine cross-architecture social benefit. |
| Row 8 not significant | Mixing helps only through static averaging of model biases. |

## Also reported
Mean output tokens per arm (P1, P2, P1_REFLECT), run-spec audit across result files, and the result of the virtual-panel prior-match check.

## Deviations
Any deviation from this plan will be listed in the paper under "Deviations from pre-registration".
