# Revision Changelog, Unresolved Issues & Necessary Follow-Up Experiments

## 1. Concise Changelog (7 Audit Domains)

1. **Version & Provenance Resolution (`b5b91ce` vs. `fdd30fe` / `3a6294c`):**
   - Separated and preserved the earlier `12`-task, `alpha = 0.3` run (`960` rows, `979` requests) into `results/polyphony-p0-p4-12task-alpha03/`, keeping the target `16`-task, `alpha = 0.4` run (`1,280` retained rows across `1,318` requests) completely untouched in `results/polyphony-p0-p4/`.
   - Added `scripts/audit_and_reproduce_p0_p4.py` to deterministically match all `1,318` provider calls (`38` extra attempts accounted for: `28` pre-23:28 replaced pass-1 cells, `8` in-run auto-retries, `2` resample-wave failures) and reconstruct `First_Pass_Saved_Pre2328` and `First_Call_Ever_Strict`.
2. **Statistical Inference & Robustness (`N = 16` Tasks):**
   - Replaced asymptotic-only interpretations with side-by-side **exact $2^{15}$ zero-drop Wilcoxon**, **exact Pratt signed-rank**, **asymptotic normal Wilcoxon**, and **paired $t$-tests**, preserving exact tie/zero counts (`1` tie across all arms on `sci_lhc_dipole_magnets_count`, `1,232.0`), task-level win/loss/tie counts, and both **within-metric Holm (`m = 9`)** and **global exploratory Holm (`m = 20`)** families.
   - Clarified that `m = 9` was pre-specified in `metrics.py` (`b5b91ce`) before 16-task collection rather than externally preregistered (removed all `"preregistered"` claims).
   - Added complete leave-one-task-out (`n = 15`) sensitivity diagnostics and exact full-precision task share accounting, revealing `hyd_lake_superior_vol_km3` accounts for **`65.4421%`** of the net `P4-vs-P0` reduction (`-0.044523 / -0.068035`), **`38.4124%`** of `P4-vs-P3`, **`38.4201%`** of `2×2 DiD`, and **`34.8870%`** of `P4-vs-P1`.
3. **Mathematical Interpretation & Attribution:**
   - Attributed `E_ind = E_crowd + D_total` properly to **Krogh & Vedelsby (1995)** combined with ANOVA/Rao within-between variance decomposition (`D_total = D_within + D_between`), alongside Page (2007) and Chuang et al. (2025).
   - Fixed the random-label permutation expectation identity (`E_perm[D_between] = ((K-1)/(M-1)) E_perm[D_total]`, so `E_perm[D_between] = 0.75 E_perm[D_within]`) and proved via exact 105-partition tests per task (`4/16` in `P4`, `3/16` in `P3` at `p <= 0.05`) that positive `D_excess` does **not** certify useful knowledge, systematic specialization, or null rejection.
   - Noted `D_between = 0` by construction in `P0..P2`, `EVRR` is an empirical variance ratio against a reference baseline (not a mathematical lower bound), `between_share` is an **unweighted mean of per-task ratios**, `0.36^2 = 12.96%` in `x2 = 0.36 x0 + 0.24 a1 + 0.40 a2` is a squared linear weight (not variance share under covariance), and `P4 < P3` arises via lower R1 carry plus a `3.1x` smaller R1/R2 signed-error cross term (`0.48 eps_R1 eps_A2`), not statistical proof of diversity-driven error cancellation.
4. **Causal & Scope Restraint:**
   - Removed claims that `P3/P4 vs. P1` `"refutes a pure capability explanation"`, that `P4` establishes superiority over `P3` (`H9e = 0.127869`, `H9n = 0.143839`), or that results generalize to `"calibration restoration"` or `"deployment readiness"`.
   - Reframed the paper around the verified robust results: social monoculture amplification (`P1 > P0`), strong mitigation under both `P2` and `P3`/`P4` vs `P1` (`16/16` LOTO), and the significant sub-additive $2\times 2$ interaction (`DiD = +0.008715`, `p_ex = 0.000183`).
5. **Concrete Reporting Errors Fixed:**
   - Corrected per-model mixed-panel sample count from `48` to **`64` retained observations per model** (`16 × 2 × 2`).
   - Relabeled in-condition assistant errors (no longer called `"standalone"` accuracy).
   - Separated retained spend (`$0.486271`, `1,280` rows) from total logged API attempt spend (`$0.501372`, `1,318` calls) and disclosed all `14` `max_output_tokens` hits, `6` repetition-loop replacements (`28+1` cells), and `1` unit-slip retry (`0.26`).
   - Aligned authorship and verification statements strictly with documented single-author work and automated audit scripts.
6. **Bibliography & Ground-Truth Audit:**
   - Corrected primary citations for **Denisov-Blanch et al. (2026)**, **Kim et al. (2025)**, **Kim et al. (2026)**, **Qiu et al. (2026)** (`18` LLMs across `25` interpretable metrics and `5` dimensions: semantic, lexical, syntactic, structural, stylistic; embedding similarity correlates only at $\rho = 0.140$ with human form judgments), **Schneider & Schramm (2025)**, and **Sourati et al. (2026)** (*Nature Human Behaviour*, `doi:10.1038/s41562-026-02550-0`).
   - Added **Krogh & Vedelsby (1995)** and **Chuang et al. (EMNLP 2025)**.
   - Documented exact sources, units, definitions, and publication/version years for all `16` benchmark items, reconciling `nrg_us_hydro_2023_twh` (`240.0` preliminary vs finalized EIA `245.0 TWh`), `nrg_norway_hydro_2023_twh` (EIA `136.11` net vs SSB `137.3 TWh` gross), and `mat_molybdenum_melting_c` (`2,623 °C` vs `2,896 K`) via sensitivity runs `S1–S4` without altering the original benchmark.
7. **Submission Figures & Manuscript Formatting:**
   - Updated `src/apart_epistemics/report/figures.py` (`polyphony_crowd_error`, `polyphony_hierarchical_page`, `polyphony_2x2_factorial`) and added Figure 4 (`polyphony_loto_crossterm.svg`).
   - Authored the submission LaTeX/PDF (now consolidated as `reports/submission/main.tex` / `main.pdf`; earlier Markdown and LaTeX drafts removed), strictly <=8 main-text pages + References + Appendices A–D, and synced Google Doc manuscript (`12PuoRXW3Yihle-fUpYvWbHqVz0zuqVJjYQPzSd2cjXs`).

---

## 2. Unresolved Issues (Cannot Be Settled Without New API Runs)

1. **Confounded Diversity vs. Model Capability in `P3`/`P4`:** Within `P3` and `P4`, `deepseek-v4-flash` and `glm-5.2` produce `10x–40x` lower in-condition MSE (`0.0007–0.0039`) than `gemma-4-26b-a4b-it` (`0.021–0.046`). Existing runs cannot isolate diversity effects from constituent model strength.
2. **Composite Prompt Package in `P2`/`P4`:** The counter-anchoring prompt bundles independent first-principles reasoning, critique of peer consensus, and reasoning length changes simultaneously; without a neutral reflection arm, mechanism remains unresolved.
3. **Underpowered Incremental Difference Between `P4` and `P3` (`n = 16`, 1 Replicate):** With `H9e = 0.127869` and `38.4%` sensitivity to Lake Superior, a larger task bank (`N >= 48`) and multi-seed replication (`R >= 3`) are required before judging whether combining counter-anchoring and heterogeneous panels yields real additive benefit over `P3` alone.

---

## 3. Genuinely Necessary Follow-Up Controls (Before Causal Claims About Polyphony)

1. **`P0-Mix` (No-Social Mixed Panel):** Run the exact 4-model panel (`gemma`, `deepseek`, `qwen`, `glm`; `2` agents each, same `x^(0)` or matched model priors) with `no social summary` (`no pop_median_log10`) to establish the capability baseline without inter-agent sharing.
2. **Homogeneous Frontier Arms (`P0`/`P1`/`P2` per Frontier Family):** Evaluate homogeneous `M = 8` panels of `deepseek-v4-flash`, `glm-5.2`, and `qwen3.8-27b` separately to measure true standalone accuracy, test whether frontier families exhibit monoculture amplification (`P1 > P0`), and benchmark a capability-matched homogeneous panel against `P3`/`P4`.
3. **Token/Structure-Matched Neutral Reflection Control (`P2-Neutral`):** Replace counter-anchoring instructions with an equal-length, neutral step-by-step verification prompt to separate epistemic forcing from generic extra reasoning / prompt structure.
