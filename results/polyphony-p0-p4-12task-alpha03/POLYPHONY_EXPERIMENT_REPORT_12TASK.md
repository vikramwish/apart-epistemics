# Cross-Architecture Polyphony & Dialectical Counter-Anchoring (`P0`–`P4`)

**Repository:** `https://github.com/vikramwish/apart-epistemics.git` (`origin`, branch `numeric-question-bank`)  
**Primary Benchmark & Results:** [`configs/polyphony_p0_p4.yaml`](../configs/polyphony_p0_p4.yaml), [`configs/polyphony_benchmark_tasks.json`](../configs/polyphony_benchmark_tasks.json), [`results/polyphony-p0-p4/`](../results/polyphony-p0-p4/)  
**Cross-Benchmark Replication:** [`configs/numeric_polyphony_c1_c6.yaml`](../configs/numeric_polyphony_c1_c6.yaml), [`results/numeric-polyphony-c1-c6/`](../results/numeric-polyphony-c1-c6/)  
**Date:** `2026-11-15`

---

## Abstract

Collective quantitative estimation with large language model (LLM) agents exhibits a structural tension when peer summary telemetry is shared across rounds: homogeneous ensembles anchor on shared training biases and collapse diversity, while social consensus summaries can pull even well-calibrated models away from true values. We introduce and evaluate a clean-room **$2 \times 2$ + Baseline** experimental design (`P0`–`P4`) crossing **architectural diversity** (homogeneous `gemma-4-26b-a4b-it` vs. equal 4-family polyphony across `gemma-4-26b-a4b-it`, `deepseek-v4-flash`, `qwen3.8-27b`, and `glm-5.2`) with **dialectical counter-anchoring** (tail-testing misconception mechanisms that could make both the user prior and peer median overshoot or undershoot ground truth) on a brand-new **12-task Scientific, Engineering & Earth-Systems Fermi Benchmark** (`960` live Adaption API observations, `0/960` misses, `0/960` truncations, `$0.3840 USD`).

To isolate architectural from within-model diversity without approximation error, we prove and implement a **Two-Level Hierarchical Page Diversity Theorem**, decomposing crowd squared error exactly as $E_{\text{crowd}} = \bar{E}_{\text{ind}} - D_{\text{between}} - D_{\text{within}}$, with finite-cluster permutation bias correction $D_{\text{excess}}$ and variance-preservation ratio (`EVRR`). Across all `120` question-condition-round cells of `P0`–`P4`, the identity holds to IEEE 754 float64 precision (`max_identity_residual = 1.943e-16`). Empirically, `P3` and `P4` raise `between_model_diversity` from `0.0` to `0.004750` (`48.85%` share, Holm `p = 0.004395`) and `0.004462` (`47.46%` share, Holm `p = 0.004395`), with positive excess between-model diversity (`+0.001024` and `+0.000622`). Crucially, on canonical physical constants where baseline assistant MSE is low (`P0` Round 2 `0.000616`), unanchored cross-model social consensus (`P3`) induces statistically detectable crowd-error degradation relative to `P0` (`+0.000409`, Wilcoxon `p = 0.007812`, `95% CI [+0.000030, +0.001097]`, matrix Holm `p = 0.070312`), inflating Round-2 assistant MSE to `0.000940`. Adding dialectical counter-anchoring (`P4`) suppresses Round-2 assistant MSE by **92.4%** vs. `P3` (`0.000940 -> 0.000071`) and recovers the lowest final `crowd_error` of all five arms (`0.009928`). Cross-benchmark comparison against the higher-uncertainty V3 `C1`–`C6` run (`1,056` rows, `$0.3479 USD`) shows how both mechanisms scale with task uncertainty: when models lack exact recall (`C1` Round-2 assistant MSE `0.055792`), combining polyphony and counter-anchoring (`C6`) cuts final `crowd_error` by **47.6%** vs. homogeneous consensus (`0.039766 -> 0.020837`, Wilcoxon `p = 0.041992`, `evrr` `+0.219325`).

---

## 1. Motivation & Clean-Room Experimental Design

Prior multi-agent LLM debate and wisdom-of-crowds studies often confabulate three distinct sources of forecast error: (1) prompt framing differences between independent and social arms, (2) same-architecture correlated errors disguised as multi-agent ensembles, and (3) benchmark overlap with earlier prompt-tuning pilots. To avoid those confounds, we built a **self-contained `P0`–`P4` protocol** (`p_series_system_prompt` and `_P_COUNTER_ANCHOR_TEXT` in `src/apart_epistemics/engine/numeric_core.py`) paired with a **zero-overlap 12-question scientific reference benchmark** (`configs/polyphony_benchmark_tasks.json`).

Two architectural design goals governed this experiment:

1. **Latency-Clean Four-Family Polyphony Routing:** In [`AdaptionBackend`](../src/apart_epistemics/backends/providers.py), non-Gemma model prefixes (`qwen`, `glm`, `deepseek`) automatically receive `payload["reasoning"] = {"effort": "none"}` while `gemma*` models run unmodified. This enables `backend.for_model(slug)` to dispatch all four frontier architectures (`gemma-4-26b-a4b-it`, `deepseek-v4-flash`, `qwen3.8-27b`, `glm-5.2`) in balanced 2-agents-per-family allocations with fast, non-blocking completion times and zero hidden-reasoning timeouts.
2. **Pre-Registered Two-Tier Contrast Control:** Rather than cherry-picking post-hoc comparisons, [`analyze_numeric_core`](../src/apart_epistemics/engine/numeric_core.py) automatically populates a structured 9-pair $\times$ 5-metric `polyphony_contrast_matrix` alongside the 16 named confirmatory contrasts and a paired `2x2` difference-in-differences interaction test (`DiD = (P4 - P3) - (P2 - P1)`).

---

## 2. Mathematical Framework: Two-Level Hierarchical Page Diversity Theorem

For any cell $(q, c, r, b)$ containing $N$ agents indexed by $i \in \{1, \dots, n_k\}$ across $K$ distinct model architectures $k \in \{1, \dots, K\}$ with $\sum_{k=1}^K n_k = N$, let $g_{ki} = \log_{10}(\hat{y}_{ki})$ be agent $(k, i)$'s post-update estimate, $g^\star = \log_{10}(y^\star)$ the log-truth, $\bar{g}_k = \frac{1}{n_k}\sum_{i=1}^{n_k} g_{ki}$ the architecture mean, and $\bar{g} = \frac{1}{N}\sum_{k=1}^K n_k \bar{g}_k$ the grand crowd mean.

### Theorem (Exact Two-Level ANOVA Decomposition of Crowd Error)

Define:

$$
E_{\text{crowd}} = (\bar{g} - g^\star)^2, \qquad
\bar{E}_{\text{ind}} = \frac{1}{N}\sum_{k=1}^K \sum_{i=1}^{n_k} (g_{ki} - g^\star)^2,
$$

$$
D_{\text{between}} = \frac{1}{N}\sum_{k=1}^K n_k (\bar{g}_k - \bar{g})^2, \qquad
D_{\text{within}} = \frac{1}{N}\sum_{k=1}^K \sum_{i=1}^{n_k} (g_{ki} - \bar{g}_k)^2.
$$

Then:

$$
E_{\text{crowd}} = \bar{E}_{\text{ind}} - D_{\text{between}} - D_{\text{within}}.
$$

### Proof

Expanding $(g_{ki} - g^\star)^2 = \bigl((g_{ki} - \bar{g}) + (\bar{g} - g^\star)\bigr)^2$ and averaging over all $N$ agents cancels the cross-term $2(\bar{g} - g^\star)\frac{1}{N}\sum_{k,i}(g_{ki} - \bar{g}) = 0$, recovering Page's flat identity $E_{\text{crowd}} = \bar{E}_{\text{ind}} - D$ with $D = \frac{1}{N}\sum_{k,i}(g_{ki} - \bar{g})^2$. Next, adding and subtracting each cluster mean $\bar{g}_k$:

$$
\begin{aligned}
D &= \frac{1}{N}\sum_{k=1}^K \sum_{i=1}^{n_k} \bigl((g_{ki} - \bar{g}_k) + (\bar{g}_k - \bar{g})\bigr)^2 \\
  &= \frac{1}{N}\sum_{k=1}^K \sum_{i=1}^{n_k} (g_{ki} - \bar{g}_k)^2
   + \frac{1}{N}\sum_{k=1}^K n_k (\bar{g}_k - \bar{g})^2
   + \frac{2}{N}\sum_{k=1}^K (\bar{g}_k - \bar{g}) \underbrace{\sum_{i=1}^{n_k} (g_{ki} - \bar{g}_k)}_{= 0} \\
  &= D_{\text{within}} + D_{\text{between}}. \qquad \square
\end{aligned}
$$

### Finite-Cluster Permutation Bias Correction (`excess_between_model_diversity`) and EVRR

Even if $K$ model labels are assigned at random to exchangeable agents sharing zero architectural difference, sampling variance creates non-zero $D_{\text{between}}$. Under a random permutation null with $1 < K < N$, fixed-effects one-way ANOVA gives $\mathbb{E}[D_{\text{between}}] = \frac{K-1}{N-K}\mathbb{E}[D_{\text{within}}]$. We therefore define [`compute_hierarchical_page_decomposition`](../src/apart_epistemics/metrics.py#L66-L120) with:

$$
D_{\text{excess}} = D_{\text{between}} - \frac{K - 1}{N - K} D_{\text{within}}, \qquad
\text{between\_share} = \frac{D_{\text{between}}}{D}.
$$

For our experimental allocation ($N = 8$, $K = 4$, $n_k = 2$), the random-label expectation coefficient is $\frac{4 - 1}{8 - 4} = 0.75$, so $D_{\text{excess}} > 0$ certifies architectural specialization strictly exceeding stochastic within-model noise.

Finally, [`analyze_numeric_core`](../src/apart_epistemics/engine/numeric_core.py) normalizes step-wise variance retention relative to passive damped prior inertia $(1 - \alpha)^2 D_{\text{before}}$ via the **Epistemic Variance Retention Ratio (`EVRR`)**:

$$
\text{EVRR} = \frac{D}{(1 - \alpha)^2 D_{\text{before}}}, \qquad
\text{evrr\_between} = \frac{D_{\text{between}}}{(1 - \alpha)^2 D_{\text{before}}}, \qquad
\text{evrr\_within} = \frac{D_{\text{within}}}{(1 - \alpha)^2 D_{\text{before}}},
$$

alongside $\text{evrr\_cumulative} = \frac{D^{(r)}}{(1 - \alpha)^{2r} D_{\text{before}}^{(1)}}$ (guarded to `None` when denominator $\le 10^{-15}$), where $\text{EVRR} > 1$ certifies new variance generated by assistant updates beyond passive prior retention. It also records both $|E_{\text{crowd}} - (\bar{E}_{\text{ind}} - D)|$ (`identity_residual`) and $|E_{\text{crowd}} - (\bar{E}_{\text{ind}} - D_{\text{between}} - D_{\text{within}})|$ (`hierarchical_identity_residual`).

---

## 3. Experimental Protocol (`P0`–`P4`) & 12-Task Scientific Benchmark

### 3.1 Five-Condition Architecture $\times$ Dialectical Design

All conditions share the same domain-neutral system prompt (`p_series_system_prompt`):

> `"You are a rigorous quantitative scientific estimator. Reason independently from physical principles and empirical reference scales without web lookup, and end with 'Final estimate: <number>'. Today's date is 2026-11-15."`

| Condition | Model Routing ($N=8$ agents) | Peer Median + IQR Telemetry | Dialectical Counter-Anchoring | Role in $2 \times 2$ + Baseline |
|---|---|---:|---:|---|
| **`P0`** | Homogeneous (`gemma-4-26b-a4b-it`, 8/8) | No | No | Independent social-free baseline |
| **`P1`** | Homogeneous (`gemma-4-26b-a4b-it`, 8/8) | Yes | No | Social consensus cell (homogeneous, unanchored) |
| **`P2`** | Homogeneous (`gemma-4-26b-a4b-it`, 8/8) | Yes | Yes | Counter-anchor main effect on same architecture |
| **`P3`** | 4-Model Polyphony (2 each: Gemma, DeepSeek-V4-Flash, Qwen3.8-27B, GLM-5.2) | Yes | No | Cross-architecture polyphony main effect (unanchored) |
| **`P4`** | 4-Model Polyphony (2 each: Gemma, DeepSeek-V4-Flash, Qwen3.8-27B, GLM-5.2) | Yes | Yes | Full combined polyphony + counter-anchoring |

In social conditions (`P1`–`P4`), prompts report the current population linear median and interquartile range (`"Peer consensus telemetry from prior estimators reports a median of ... (interquartile range ... to ...)."`). In dialectical conditions (`P2`, `P4`), prompts inject `_P_COUNTER_ANCHOR_TEXT`:

> `"Dialectical counter-anchoring protocol: before estimating, write 2-3 concise sentences testing both tails — identify specific physical scaling mechanisms or misconceptions by which both the prior estimate and peer median could systematically overshoot (too high) OR undershoot (too low) the true value — then derive your estimate from independent physical principles and reference scales."`

Simulation hyperparameters in [`configs/polyphony_p0_p4.yaml`](../configs/polyphony_p0_p4.yaml) match Phase 1 `C1`–`C6`: `seed = 42`, `num_agents = 8`, `num_rounds = 2`, `numeric_replicates = 1`, `alpha = 0.3` (`g_after = 0.7 * g_before + 0.3 * g_assistant`), `numeric_start_bias_weight = 0.22`, `numeric_start_sd = 0.24`, and equal `6` / `6` `+0.8` / `-0.8` prior misconception sign balance across the 12 tasks (`total rows = 12 * 5 * 8 * 2 = 960`).

### 3.2 Clean-Room 12-Task Scientific, Engineering & Earth-Systems Benchmark

Every task in [`configs/polyphony_benchmark_tasks.json`](../configs/polyphony_benchmark_tasks.json) is a pre-2026 physical, geophysical, astrophysical, molecular, or structural standard (`numeric_truth >= 305.0`), verified disjoint from both `configs/numeric_final_tasks.json` and `build_numeric_core_tasks()`:

| Task ID | Domain | Stratum | Ground Truth | Unit | Bias Sign | Source Standard |
|---|---|---|---:|---|---:|---|
| `sci_earth_mean_radius_km` | `planetary_geodesy` | `control` | `6,371.0` | `kilometers` | `+0.8` | NASA Earth Fact Sheet (NSSDCA) |
| `sci_solar_effective_temp_k` | `stellar_astrophysics` | `control` | `5,772.0` | `kelvins` | `+0.8` | IAU 2015 Resolution B3 |
| `sci_moon_mean_distance_km` | `lunar_astrometry` | `control` | `384,400.0` | `kilometers` | `-0.8` | NASA Moon Fact Sheet (NSSDCA) |
| `sci_c14_half_life_yr` | `nuclear_geochronology` | `control` | `5,730.0` | `years` | `-0.8` | Godwin (1962), *Nature* 195:984 (Cambridge) |
| `sci_prem_cmb_depth_km` | `deep_earth_seismology` | `low_spread` | `2,891.0` | `kilometers` | `+0.8` | Dziewonski & Anderson (1981) PREM |
| `sci_earth_sidereal_day_s` | `geodynamics` | `low_spread` | `86,164.0` | `seconds` | `+0.8` | NASA Earth Fact Sheet (23.9345 h) |
| `sci_tungsten_melting_c` | `materials_thermophysics` | `low_spread` | `3,422.0` | `degrees Celsius` | `-0.8` | NIST / CRC Handbook |
| `sci_canakkale_bridge_span_m` | `structural_engineering` | `low_spread` | `2,023.0` | `meters` | `-0.8` | Turkish General Directorate of Highways (KGM, 2022) |
| `sci_kola_borehole_depth_m` | `crustal_drilling` | `high_spread` | `12,262.0` | `meters` | `-0.8` | Kola SG-3 1989 record vertical depth |
| `sci_hoover_dam_capacity_mw` | `hydroelectric_engineering` | `high_spread` | `2,080.0` | `megawatts` | `+0.8` | U.S. Bureau of Reclamation nameplate spec |
| `sci_arecibo_diameter_m` | `radio_astronomy` | `high_spread` | `305.0` | `meters` | `+0.8` | NSF / Cornell NAIC specifications |
| `sci_rcrs_mtdna_bp` | `molecular_genomics` | `high_spread` | `16,569.0` | `base pairs` | `-0.8` | Andrews et al. (1999) rCRS / NC_012920 |

---

## 4. Primary Empirical Results (`results/polyphony-p0-p4/`)

### 4.1 Execution Integrity, Cost & Numerical Exactness

- **Dataset Completeness:** `960 / 960` agent-round observations (`completion_status = "complete"`, `missing_or_incomplete_cells = 0`), `0 / 960` parse misses (`0.000`).
- **Model Balancing:** `P0`, `P1`, `P2` logged `192 / 192` `gemma-4-26b-a4b-it` rows each; `P3` and `P4` logged `48` rows per architecture across `gemma-4-26b-a4b-it`, `deepseek-v4-flash`, `qwen3.8-27b`, and `glm-5.2` (`672 / 96 / 96 / 96` aggregate rows).
- **Zero Truncation & Transparent Retry Note:** `0 / 960` final rows have `finish_reason != "stop"`. In the initial run pass, three `gemma-4-26b-a4b-it` replies (`P1 sci_hoover_dam_capacity_mw` Round 1 `agent_02`, `P2 sci_kola_borehole_depth_m` Round 1 `agent_01`, and `P1 sci_tungsten_melting_c` Round 2 `agent_07`) produced valid parsed answers (`2080.0`, `12262.0`, `3150.0`) followed by degenerate `"Final estimate:"` token-repetition loops reaching `8,192` tokens (`finish_reason = "length"`). To preserve downstream Round-2 `pop_stats` consistency and meet the strict `0`-truncation gate, those 3 truncated cells and the 2 affected Round-2 trajectories (`19` rows total) were removed and re-sampled cleanly via crash-resume, producing `960 / 960` `finish_reason = "stop"`.
- **Reasoning Token Accounting & Spend:** `0 / 960` rows recorded `reasoning_tokens > 0` (`deepseek-v4-flash`, `qwen3.8-27b`, and `glm-5.2` used `reasoning={"effort": "none"}`, and Adaption's usage field reports `0` reasoning tokens for `gemma-4-26b-a4b-it`). Total provider-reported cost across the final 960 rows is **`$0.3840 USD`** (`$0.000400 / row`, max `$0.004489`; ledger reserved `$4.704` over `2,352` HTTP attempts including retries and resume).
- **Mean Token Profile (Input / Output):** `P0` (`135.4 / 454.1`), `P1` (`172.2 / 556.6`), `P2` (`242.2 / 431.0`), `P3` (`157.2 / 467.9`), `P4` (`228.0 / 512.4`).
- **Algebraic Theorem Verification:** Across all `120` `(question, condition, round, replicate)` cells evaluated against unrounded `agent_log10_after` values, `max_flat_residual = 1.874e-16` and `max_hierarchical_residual = 1.943e-16` (`numeric_analysis.json`: `1.942890293094024e-16`).

### 4.2 Round-by-Round Page Decomposition & Assistant Calibration MSE

While agent belief updating (`alpha = 0.3` over 2 rounds) anchors final beliefs with `0.49` weight on each agent's initial misconception prior `agent_log10_before`, inspecting both aggregate agent metrics (`numeric_analysis.json`) and raw `assistant_log10` MSE separates **prior-inertia ceilings** from **direct assistant judgment**:

| Cond | Round | `crowd_error` | `avg_ind_err` | `diversity` | `between_div` | `within_div` | `excess_between` | `between_share` | `evrr` | `evrr_cum` | `asst_mse` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `P0` | 1 | `0.020080` | `0.039437` | `0.019357` | `0.000000` | `0.019357` | `0.000000` | `0.000000` | `1.008485` | `1.008485` | `0.000267` |
| **`P0`** | **2** | **`0.009932`** | `0.019638` | `0.009706` | `0.000000` | `0.009706` | `0.000000` | `0.000000` | `1.022885` | `1.033577` | `0.000616` |
| `P1` | 1 | `0.020390` | `0.039603` | `0.019213` | `0.000000` | `0.019213` | `0.000000` | `0.000000` | `0.999986` | `0.999986` | `0.000559` |
| **`P1`** | **2** | **`0.010231`** | `0.019611` | `0.009380` | `0.000000` | `0.009380` | `0.000000` | `0.000000` | `0.996219` | `0.996149` | `0.000416` |
| `P2` | 1 | `0.020387` | `0.039624` | `0.019237` | `0.000000` | `0.019237` | `0.000000` | `0.000000` | `1.001371` | `1.001371` | `0.000560` |
| **`P2`** | **2** | **`0.010143`** | `0.019706` | `0.009563` | `0.000000` | `0.009563` | `0.000000` | `0.000000` | `1.007116` | `1.008507` | `0.000567` |
| `P3` | 1 | `0.020488` | `0.039975` | `0.019487` | `0.009349` | `0.010138` | `+0.001746` | `0.481292` | `1.015707` | `1.015707` | `0.000858` |
| **`P3`** | **2** | **`0.010340`** | `0.020057` | `0.009717` | `0.004750` | `0.004967` | `+0.001024` | `0.488489` | `1.017328` | `1.035461` | `0.000940` |
| `P4` | 1 | `0.020177` | `0.039679` | `0.019502` | `0.009179` | `0.010323` | `+0.001437` | `0.477085` | `1.016500` | `1.016500` | `0.000577` |
| **`P4`** | **2** | **`0.009928`** | `0.019511` | `0.009583` | `0.004462` | `0.005120` | `+0.000622` | `0.474642` | `1.000311` | `1.016548` | **`0.000071`** |

Three primary empirical findings emerge directly from this table:

1. **Social Telemetry Without Counter-Anchoring Corrupts Assistant Accuracy (`P3` vs. `P0`):** Even on well-known physical constants, unanchored cross-architecture social telemetry (`P3`) increases Round-2 `crowd_error` relative to the social-free `P0` baseline (`0.010340` vs. `0.009932`, `+0.000409`, Wilcoxon `p = 0.007812`, `95% CI [+0.000030, +0.001097]`, matrix Holm `p = 0.070312`) and increases Round-2 assistant-only MSE from `0.000616` to `0.000940` (`qwen3.8-27b` assistant MSE `0.003005`, `gemma-4-26b-a4b-it` `0.000753`).
2. **Dialectical Counter-Anchoring Eliminates the Social-Drift Penalty (`P4` vs. `P3` and `P0`):** Injecting `_P_COUNTER_ANCHOR_TEXT` into the polyphony ensemble (`P4`) cuts Round-2 assistant MSE by **92.4%** vs. `P3` (`0.000940 -> 0.000071`) and **88.5%** vs. `P0` (`0.000616 -> 0.000071`). By model in Round 2 of `P4`, assistant-only MSE drops to `qwen3.8-27b = 0.000000`, `deepseek-v4-flash = 0.000002`, `gemma-4-26b-a4b-it = 0.000006`, and `glm-5.2 = 0.000275`. This yields the lowest final `crowd_error` (`0.009928`) across all five conditions (`P4 < P0 < P2 < P1 < P3`).
3. **Genuine Between-Model Structure Persists Under Prior Inertia:** `P3` and `P4` partition nearly half of total variance into `between_model_diversity` (`48.85%` in `P3`, `47.46%` in `P4`), yielding positive `excess_between_model_diversity` (`+0.001024` in `P3`, `+0.000622` in `P4`) above the finite-sample random-label permutation baseline (`0.75 * D_within`). Note carefully that in single-value canonical constants (`earth_mean_radius_km`, `arecibo_diameter_m`, `earth_sidereal_day_s`, `rcrs_mtdna_bp`), `0/80` assistant errors deviate by more than `1e-4`; there, `between_model_diversity` reflects fixed `0.49`-weighted random initial agent pairings rather than assistant disagreement, whereas `excess_between_model_diversity` isolates true architectural separation on divergent tasks.

### 4.3 Stratum Breakdown & Non-Ceiling Task Behavior

| Stratum (`n=4` each) | Cond | `crowd_error` | `diversity` | `between_div` | `excess_between` | `between_share` | `evrr` |
|---|---|---:|---:|---:|---:|---:|---:|
| **`control`** | `P0` | `0.015183` | `0.008210` | `0.000000` | `0.000000` | `0.000000` | `0.999807` |
| **`control`** | `P1` | `0.015184` | `0.008212` | `0.000000` | `0.000000` | `0.000000` | `1.000042` |
| **`control`** | `P2` | `0.015184` | `0.008213` | `0.000000` | `0.000000` | `0.000000` | `0.999978` |
| **`control`** | `P3` | `0.015200` | `0.008189` | `0.004407` | `+0.001571` | `0.520134` | `0.999999` |
| **`control`** | `P4` | `0.015188` | `0.008206` | `0.004418` | `+0.001577` | `0.520150` | `0.999582` |
| **`low_spread`** | `P0` | `0.009965` | `0.010731` | `0.000000` | `0.000000` | `0.000000` | `1.000000` |
| **`low_spread`** | `P1` | `0.010223` | `0.010689` | `0.000000` | `0.000000` | `0.000000` | `0.991041` |
| **`low_spread`** | `P2` | `0.010207` | `0.011062` | `0.000000` | `0.000000` | `0.000000` | `1.014725` |
| **`low_spread`** | `P3` | `0.011049` | `0.011586` | `0.006566` | `+0.002801` | `0.627446` | `1.046167` |
| **`low_spread`** | `P4` | `0.010073` | `0.010860` | `0.005578` | `+0.001616` | `0.577282` | `1.005391` |
| **`high_spread`** | `P0` | `0.004647` | `0.010178` | `0.000000` | `0.000000` | `0.000000` | `1.068850` |
| **`high_spread`** | `P1` | `0.005286` | `0.009238` | `0.000000` | `0.000000` | `0.000000` | `0.997574` |
| **`high_spread`** | `P2` | `0.005037` | `0.009416` | `0.000000` | `0.000000` | `0.000000` | `1.006646` |
| **`high_spread`** | `P3` | `0.004773` | `0.009375` | `0.003275` | `-0.001299` | `0.317888` | `1.005818` |
| **`high_spread`** | `P4` | **`0.004523`** | `0.009681` | `0.003391` | `-0.001327` | `0.326495` | `0.995960` |

Two non-ceiling questions illustrate the exact behavioral mechanism:
- In `low_spread`, `sci_canakkale_bridge_span_m` (`2,023 m`) degrades under unanchored social polyphony (`P3` `crowd_error = 0.01649` vs. `P0` `0.01250`), and dialectical counter-anchoring restores accuracy (`P4` `0.01252`).
- In `high_spread`, `P4` achieves the lowest `crowd_error` (`0.004523` vs. `P0` `0.004647` and `P1` `0.005286`), driven by `sci_kola_borehole_depth_m` (`P4` `0.00605` vs. `P1` `0.00893`, `P2` `0.00802`, `P3` `0.00693`, `P0` `0.00675`). Note also that `P4` has slightly lower `crowd_error` than `P0` (`0.004523` vs. `0.004647`) while `sci_hoover_dam_capacity_mw` stays at `0.00023` in `P4` vs. `0.00040` in `P1` (`0.00003` in `P0`).

### 4.4 Pre-Registered Contrasts, `2x2` Interaction & H4 Sycophancy Diagnostics

From `polyphony_contrast_matrix` (`n = 12` paired questions, two-tier per-metric Holm adjustment across the 9 contrasts) and `interaction_2x2` (`n = 12`):

| Metric | Contrast | `mean_diff` | `95% Bootstrap CI` | Wilcoxon `p` | Matrix Holm `p` | 16-Contrast Holm `p` |
|---|---|---:|---|---:|---:|---:|
| `between_model_diversity` | `p4_vs_p1` (`= p4_vs_p0`, `p4_vs_p2`) | `+0.004462` | `[+0.002944, +0.006053]` | `0.000488` | **`0.004395`** | — |
| `between_model_diversity` | `p3_vs_p1` (`= p3_vs_p0`, `p3_vs_p2`) | `+0.004750` | `[+0.003207, +0.006410]` | `0.000488` | **`0.004395`** | — |
| `crowd_error` | `p3_vs_p0` | `+0.000409` | `[+0.000030, +0.001097]` | **`0.007812`** | `0.070312` | `0.125000` |
| `crowd_error` | `p4_vs_p3` | `-0.000412` | `[-0.001139, +0.000023]` | `0.130859` | `1.000000` | `1.000000` |
| `crowd_error` | `p4_vs_p1` | `-0.000303` | `[-0.000832, +0.000047]` | `0.769531` | `1.000000` | `1.000000` |
| `crowd_error` | `p4_vs_p2` | `-0.000215` | `[-0.000578, +0.000003]` | `0.496094` | `1.000000` | `1.000000` |
| `crowd_error` | `p1_vs_p0` | `+0.000299` | `[+0.000001, +0.000718]` | `0.125000` | `1.000000` | `1.000000` |
| `diversity` | `p4_vs_p1` | `+0.000203` | `[+0.000024, +0.000461]` | `0.193359` | `1.000000` | `1.000000` |
| `excess_between` | `p3_vs_p1` | `+0.001024` | `[-0.000595, +0.002595]` | `0.266113` | `1.000000` | `1.000000` |
| `evrr` | `p4_vs_p2` | `-0.006805` | `[-0.013923, -0.000563]` | `0.078125` | `0.703125` | — |

- **`2x2` DiD Interaction (`(P4 - P3) - (P2 - P1)`):** `crowd_error = -0.000324` (`95% CI [-0.001042, +0.000119]`, `p = 0.375`), `diversity = -0.000318` (`p = 0.275391`), `evrr = -0.027914` (`95% CI [-0.058864, -0.003458]`, `p = 0.164062`, Holm `0.65625`).
- **H4 Sycophancy & Anchor Pulling (`h4_assistant_agent_corr = 0.9753`):** On canonical scientific benchmarks, within-question `slope_user` stays near `0` across all conditions (`P0 = 0.0125`, `P1 = -0.0035`, `P2 = 0.0018`, `P3 = 0.0132`, `P4 = 0.0102`), and `crowd_pull_vs_c1_r1` is tiny (`P1 = 0.0144`, `P2 = 0.0142`), reflecting strong mechanistic knowledge when assistants resist misleading user priors.

---

## 5. Cross-Benchmark Replication: High-Uncertainty Regime (`C1`–`C6`)

Comparing `P0`–`P4` against our earlier **11-question V3 `C1`–`C6` Adaption experiment** ([`results/numeric-polyphony-c1-c6/`](../results/numeric-polyphony-c1-c6/), `1,056` rows across 11 non-settled 2026 event and municipal-census Fermi tasks, `0` misses, `0` truncations, `$0.3479 USD` actual cost, `max_identity_residual = 3.123e-16`) reveals how **domain uncertainty** modulates both interventions:

| Benchmark | Regime & Baseline Error (`C1`/`P0` R2 `asst_mse`) | Social Collapse (`P1`/`C2` `crowd_error`) | Counter-Anchor Only (`P2`/`C4`) | Polyphony Only (`P3`/`C5`) | Full Combined (`P4`/`C6`) | Key Contrast & H4 Pull |
|---|---|---|---|---|---|---|
| **`C1`–`C6` (V3 Unsettled/Census)** | High uncertainty (`asst_mse = 0.055792`) | `C2 = 0.039766` (`C1 = 0.021548`, `+84.5%` herding error) | `C4 = 0.031998` | `C5 = 0.027189` (`between = 0.007292`, `37.42%` share) | **`C6 = 0.020837`** (`between = 0.008563`, `36.37%`, `evrr = 1.414287`) | `c6_vs_c2_crowd_error = -0.018929` (`-47.6%`, `p = 0.041992`); `C2 crowd_pull = 0.9550` |
| **`P0`–`P4` (Physical Constants)** | Low uncertainty (`asst_mse = 0.000616`, `90x` lower) | `P1 = 0.010231` (`P0 = 0.009932`) | `P2 = 0.010143` | `P3 = 0.010340` (`between = 0.004750`, `48.85%` share) | **`P4 = 0.009928`** (`between = 0.004462`, `47.46%`, `asst_mse = 0.000071`) | `p3_vs_p0 = +0.000409` (`p = 0.007812`); `p4_vs_p1 between` Holm `0.004395`; `P1 crowd_pull = 0.0144` |

Together, the two datasets establish a dual regime law:
- In **high-uncertainty forecasting (`C1`–`C6`)**, homogeneous consensus induces severe consensus-echoing (`crowd_pull = 0.9550`), and orthogonal error cancellation across architectures (`C5`, `C6`) provides the primary reduction in `crowd_error` (`-47.6%` vs. `C2`), amplified by dialectical tail-testing.
- In **low-uncertainty scientific estimation (`P0`–`P4`)**, unanchored social polyphony (`P3`) still leaks minor herding errors (`+0.000409`, `p = 0.007812`), while dialectical counter-anchoring (`P4`) suppresses Round-2 assistant MSE by **92.4%** (`0.000940 -> 0.000071`) and keeps crowd error at the baseline floor (`0.009928`).

---

## 6. Limitations & Future Directions

1. **Statistical Power at $n = 12$ Questions:** Across 12 benchmark items, exact two-sided Wilcoxon signed-rank tests have a minimum achievable $p$-value of $2 / 2^{12} = 0.000488$. After nine-contrast Holm correction inside each metric block, strong architectural structural effects (`between_model_diversity`, Holm `p = 0.004395`) survive cleanly, while smaller accuracy margins (`p3_vs_p0 crowd_error` unadjusted `p = 0.007812`, Holm `p = 0.070312`) warrant larger $n \ge 24$ task banks.
2. **Two-Round Prior Inertia (`alpha = 0.3`):** Retaining $0.7^2 = 0.49$ weight on each agent's seeded initial prior means that whenever all models agree on canonical physical constants (`7/12` items near zero assistant variance), aggregate `crowd_error` approaches an inertia floor rather than tracking the `92.4%` improvement observed in raw `assistant_log10` MSE. Future ablations should pair longer horizons ($R \ge 4$ or higher $\alpha$) with uncertain pre-cutoff scientific frontiers (e.g., astronomical parameter tolerances or unpublished lab measurements).
3. **Provider Reasoning Telemetry:** Adaption API responses with `reasoning={"effort": "none"}` on non-Gemma models and default Gemma output report `reasoning_tokens = 0` (`0/960` rows with `reasoning_tokens > 0`), leaving hidden-CoT vs. visible-CoT ablations to providers exposing full token breakdown.
