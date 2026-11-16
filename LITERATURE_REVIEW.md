# Literature Review: AI-Mediated Epistemic Monoculture, Personalization, and Collective Calibration

> **Scope:** Synthesizes the theoretical foundations, empirical homogenization studies, social epistemology frameworks, and multi-agent simulation literature motivating an Apart Research study on how shared, mixed, and personalized LLM assistants affect **belief diversity**, **truth-finding accuracy**, and **epistemic calibration** in multi-agent populations.

---

## Executive Synthesis

Across philosophy, HCI, behavioral economics, and NLP, four converging threads define the research landscape:

1. **Epistemic erosion is about *standards of justification*, not only hallucinations.** Generative AI substitutes fluent plausibility for epistemic evaluation (*Epistemia*, Quattrociocchi, Capraro & Perc, 2025), suppresses cognitive effort unless interrupted by forcing functions (Buçinca, Malaya & Gajos, 2021), and induces excessive human deference to confident falsehoods (Deepthika S, Yadav & Sankar, 2026).
2. **LLM homogenization is well-documented at the single-turn and corpus level—and crosses lab boundaries.** Shared RLHF-aligned models shrink writing-complexity variance (Sourati et al., 2026), flatten public debate to 3.4% unique main arguments vs. 65.3% for humans across multiple frontier labs (*Argument Collapse*, Kim et al., 2026), and mirror Western/English-speaking norms by default (Tao et al., 2024; Sourati, Ziabari & Dehghani, 2025).
3. **Homogenization breaks the *source independence* required for collective truth-finding.** In Goldman’s (2001) social epistemology, $N$ followers echoing one guru provide zero Bayesian evidence beyond the guru; in DeMarzo, Vayanos & Zwiebel’s (2003) network model, boundedly rational agents double-count correlated signals (*persuasion bias*). Recent ML work confirms that LLMs share correlated errors across 350+ models (Kim et al., 2025), breaking Condorcet’s Jury Theorem for static LLM ensembles (Denisov-Blanch et al., 2026)—though multi-round *deliberation* among diverse models can partially recover calibration gains (Schneider & Schramm, 2025).
4. **The open empirical gap is *personalized assistants* vs. *epistemic fragmentation and miscalibration* in multi-agent loops.** Mathematical modeling predicts that personalization preserves population-level diversity where shared models collapse (Thejaswi, Kulshreshta & Oettershagen, 2026). Yet empirical LLM personalization studies show sycophancy (Sharma et al., 2024) and role-dependent losses in epistemic independence (Kelley & Riedl, 2026). No multi-round LLM agent study directly tests whether personalized assistants preserve *calibrated, truth-tracking diversity* or merely replace **single-guru monoculture** with **sycophantic echo-chamber fragmentation**.

---

## 1. Epistemic Standards, Overreliance, and "Epistemia"

A core insight from the team’s theoretical thread is that the danger of AI assistants lies not merely in factual mistakes, but in reshaping **how people decide what counts as justified belief**—echoing **Goodhart’s Law** (when fluent explanation becomes the proxy for competence, it ceases to track truth).

| Work | Venue / ID | Core Mechanism / Concept | Relevance to Simulation & Analysis |
| :--- | :--- | :--- | :--- |
| **Postman (1985)**, *Amusing Ourselves to Death* | Book | Dominant media format restructures public epistemology (typographic logic vs. effortless consumption). | Motivates studying AI as an *epistemic medium* that alters deliberative norms, not just an answer lookup tool. |
| **Chiang (2023)**, *ChatGPT Is a Blurry JPEG of the Web* | *The New Yorker* | LLMs perform lossy statistical interpolation across lexical space; errors are smoothed-over compression artifacts. | Explains why AI consensus produces plausible-sounding "averaged" knowledge rather than grounded verification. |
| **Buçinca, Malaya & Gajos (2021)**, *To Trust or to Think* | *Proc. ACM Hum.-Comput. Interact.* (CSCW) | Users rely on System-1 heuristics and treat explanations as competence cues; **cognitive forcing functions** cut overreliance but lower user satisfaction. | Explains why agents/users adopt flawed AI arguments without checking soundness unless forced to deliberate first. |
| **Quattrociocchi, Capraro & Perc (2025)**, *Epistemological Fault Lines Between Human and Artificial Intelligence* | [arXiv:2512.19466](https://arxiv.org/abs/2512.19466) | Coins **Epistemia** ("feeling of knowing without the labor of judgment") across 7 fault lines: Grounding, Parsing, Experience, Motivation, Causal Reasoning, **Metacognition**, Value. | Directly explains why agents lose uncertainty awareness: LLMs lack metacognitive stopping signals and emit confident fluent text. |
| **Deepthika S, Yadav & Sankar (2026)**, *Who Is Sycophantic to Whom?* | [arXiv:2610.31579](https://arxiv.org/abs/2610.31579) | Introduces **Human Deference Index (HDI)**. When assistants (tested with GPT-5-mini) never hedge/concede, humans adopt **~65%** of falsehoods with high confidence, collapsing trust calibration to chance. | Motivates measuring **confident-wrong rate** and **calibration decay** alongside raw accuracy. |
| **Finnveden (2024–2025)**, *What's important in "AI for epistemics"?* | Forethought / Open Philanthropy (with Cotton-Barratt) | Separates (1) AI truth-tracking & uncertainty calibration from (2) **diffusion & appropriate trust** (avoiding sticky path-dependent overtrust). | Anchors the project’s definition of epistemics: not only correctness, but well-calibrated beliefs and appropriate trust. |

---

## 2. Homogenization, Argument Collapse, and Cultural Skew

Empirical NLP and computational social science research establishes that LLM homogenization operates across style, argument structure, long-tail knowledge, and cultural values—and that naive embedding metrics understate or distort what is happening.

### 2.1 Linguistic, Knowledge, and Argument Compression
* **RLHF drives stylistic & content convergence:** Padmakumar & He ([arXiv:2309.05196](https://arxiv.org/abs/2309.05196), ICLR 2024) showed in a controlled writing study that collaborating with RLHF-tuned InstructGPT significantly reduced lexical and content diversity relative to unassisted writing, whereas base GPT-3 did not.
* **Large-scale field decline:** Sourati, Karimi-Malekabadi, Ozcan, McDaniel, Ziabari, Trager, Tak, Chen, Morstatter & Dehghani ([arXiv:2502.11266](https://arxiv.org/abs/2502.11266), *Nature Human Behaviour* 2026, `doi:10.1038/s41562-026-02550-0`) analyzed >880,000 texts across Reddit, Patch News, and arXiv, linking post-ChatGPT LLM adoption to conformity and a **21%–50%** drop in writing-complexity variance when LLMs polish human text.
* **Knowledge Collapse (economic incentives):** Peterson ([arXiv:2404.03502](https://arxiv.org/abs/2404.03502), *AI & Society*) distinguishes human **knowledge collapse** from recursive synthetic-data *model collapse*. Because LLMs draw toward the center of their training distribution while slashing retrieval effort, rational learners abandon costly long-tail exploration. Under Peterson’s default simulation parameters, a **20% time/cost discount** on AI assistance leaves long-run public beliefs **2.3x further from truth** than the no-AI baseline.
* **Argument Collapse across frontier labs:** Kim, Chang, Pham & Iyyer ([arXiv:2606.01736](https://arxiv.org/abs/2606.01736)) compared *New York Times* and *Boston Review* debates against tens of thousands of LLM essays across models from multiple frontier labs. On the NYT corpus, **65.3%** of human main arguments were unique within a debate vs. only **3.4%** for LLMs (**41.0%** vs. **9.1%** for sub-arguments). Crucially for experimental design, prompting for diversity only recovered roughly half of human argument space, and cross-lab models still converged on the same polished points—while RLHF/human evaluators tended to reward those common arguments.
* **Artificial Hivemind & decoding sensitivity:** Jiang et al. ([arXiv:2510.22954](https://arxiv.org/abs/2510.22954)) introduced **Infinity-Chat** (26K open-ended prompts) to demonstrate both *intra-model repetition* and *inter-model homogeneity*, later extended across 23 open-weight models and 13 languages by Davey et al. ([arXiv:2610.16755](https://arxiv.org/abs/2610.16755)). Crucially, decoding parameters (`temperature`, `top_p`) separate surface randomness from deep mode collapse: Schaeffer et al. ([arXiv:2609.33936](https://arxiv.org/abs/2609.33936)) caution that naive baselines can overstate homogeneity when inference-time diversity interventions are omitted, while Fu et al. ([arXiv:2608.02618](https://arxiv.org/abs/2608.02618)) find that open-ended semantic convergence persists even at standard/high temperatures (pairwise cosine similarity ~0.80–0.90), requiring **Meta-Persona Anchoring** plus Top-$p$ filtered extreme temperature scaling ($T \ge 4.0$) to drop similarity from ~0.85 to ~0.65. Testing greedy (`T=0`) vs. realistic app defaults (`T=0.7–1.0`) directly isolates decoding artifacts from genuine epistemic collapse.

### 2.2 Cultural Skew and Why Cosine Similarity Fails
* **Non-neutral "median" (Global North / WEIRD skew):** Sourati, Ziabari & Dehghani ([arXiv:2508.01491](https://arxiv.org/abs/2508.01491)) synthesize cognitive science and NLP evidence showing LLM outputs reflect Western, Educated, Industrialized, Rich, Democratic (WEIRD) defaults. Quantitative evidence from Tao, Viberg, Baker & Kizilcec (*PNAS Nexus*, 2024) across five OpenAI models (GPT-3 to GPT-4o) benchmarked against World Values Survey data in 100+ countries/territories shows unprompted outputs skew toward **English-speaking and Protestant European** values—though explicit cultural prompting improved alignment in **71%–81%** of tested countries/territories for GPT-4–class models.
* **Normativity as an AI Safety problem:** Rios-Sialer ([arXiv:2601.06116](https://arxiv.org/abs/2601.06116)) frames modal compression as a safety risk to minoritized viewpoints and proposes stakeholder encoding and *xeno-reproduction* tasks to preserve structural diversity.
* **Why cosine embedding similarity is insufficient:** Qiu, Hu, Ding & Underwood ([arXiv:2611.01549](https://arxiv.org/abs/2611.01549)) evaluate 18 LLMs across 25 interpretable metrics spanning 5 dimensions (semantic, lexical, syntactic, structural, and stylistic & presentational). They find embedding similarity correlates weakly with human judgments of textual form ($\rho = 0.140$) and masks multi-dimensional divergence: models often *"say similar things in different words,"* while intra-model variance shrinks across generations (2024–2026). **Implication for our experiment:** evaluating numeric estimation error alongside hierarchical between-/within-architecture variance decomposition (Krogh & Vedelsby, 1995; Page, 2007) captures truth-tracking accuracy directly rather than relying on embedding distance.

---

## 3. Social Epistemology: Source Independence, Correlated Errors, and Crowd Failure

This pillar formally connects the team’s **Goldman (2001)** insight and **Bayesian updating** framing to modern ML results on correlated LLM failures.

* **Goldman (2001), *Experts: Which Ones Should You Trust?* (*Philosophy & Phenomenological Research*):** When novices aggregate agreeing claims, consensus increases Bayesian posterior probability **only** if concurring witnesses are **independent** and **discriminating**. In Goldman’s **"guru and slavish followers"** thought experiment, 10 (or 50) nondiscriminating followers parroting a single guru provide **zero** additional evidential weight beyond the guru alone. When 30 simulated or human agents all query one shared LLM—or distinct LLMs trained on overlapping corpora—apparent social consensus masks a **single correlated source**.
* **DeMarzo, Vayanos & Zwiebel (2003), *Persuasion Bias, Social Influence, and Unidimensional Opinions* (*Quarterly Journal of Economics*):** Exact Bayesian updating over a social network requires tracking message provenance to avoid double-counting correlated echoes. Because that is intractable, boundedly rational agents exhibit **persuasion bias**—treating repeated exposures as independent evidence. Over rounds, persuasion bias gives disproportionate weight to well-connected hubs (an AI assistant queried by everyone acts as a maximally central hub) and collapses multi-dimensional beliefs onto a single axis before convergence.
* **Kleinberg & Raghavan (2021), *Algorithmic Monoculture and Social Welfare* (*PNAS*):** Proves mathematically that even when a shared model is *strictly more accurate* for every individual decision-maker, universal adoption degrades aggregate social outcomes (an algorithmic Braess’ paradox) because errors and blind spots become perfectly correlated across actors.
* **Correlated LLM errors violate Condorcet’s Jury Theorem:**
  * Kim, Garg, Peng & Garg ([arXiv:2506.07962](https://arxiv.org/abs/2506.07962)) test 350+ LLMs and find substantial error correlation even across distinct architectures and providers—especially among larger, more capable models.
  * Denisov-Blanch et al. ([arXiv:2603.06612](https://arxiv.org/abs/2603.06612)) demonstrate that crowd-wisdom / ensembling strategies fail for LLM truthfulness because correlated errors amplify shared misconceptions rather than canceling out.
  * **Crucial nuance—Deliberation vs. Polling:** Schneider & Schramm ([arXiv:2512.22625](https://arxiv.org/abs/2512.22625)) show that multi-round structured *deliberation* among **diverse** LLM forecasters significantly improves accuracy and reduces log loss, whereas deliberation among **identical** models yields zero epistemic gain. This directly supports comparing **1 shared model** vs. **a multi-lab model mix** in a multi-round belief-updating loop rather than single-turn generation.

---

## 4. Multi-Agent Dynamics, Tipping Points, and the Personalization Trade-Off

| Work | Venue / ID | Method & Setup | Key Findings & Limitations |
| :--- | :--- | :--- | :--- |
| **Centola, Becker, Brackbill & Baronchelli (2018)** | *Science*, [DOI:10.1126/science.aas8827](https://doi.org/10.1126/science.aas8827) | Controlled online human coordination game (naming conventions). | A **~25%** committed minority reliably tips the whole population to a new convention; 10%–20% fails. Artificial coordination task—not truth-seeking. |
| **Ashery, Aiello & Baronchelli (2024/2025)** | *Science Advances*, [arXiv:2410.08948](https://arxiv.org/abs/2410.08948) | Decentralized populations of LLM agents in convention-formation games. | LLM agents spontaneously form shared conventions, exhibit collective bias, and flip under committed minorities above a critical mass threshold. |
| **Chuang et al. (2023)** | [arXiv:2311.09618](https://arxiv.org/abs/2311.09618) | Networks of persona-prompted LLM agents debating topics over rounds. | Out-of-the-box LLM agents rapidly converge toward factual/mainstream consensus; **confirmation bias** must be explicitly prompted to reproduce real-world polarization. |
| **Thejaswi, Kulshreshta & Oettershagen (2026)**, *Linguistic Monoculture in LLM-Assisted Language Use* | [arXiv:2607.27134](https://arxiv.org/abs/2607.27134) | Mathematical coevolution model (feature distributions) comparing fixed shared, recursive shared, and **personalized** models. | Individually rational users over-conform to shared models ("price of monoculture"), whereas **personalized models preserve nonzero diversity equilibria**. **Gap:** purely mathematical feature-distribution model; no real LLM agents, no truth/accuracy or calibration metrics. |
| **Sharma et al. (2023/2024)**, *Towards Understanding Sycophancy in Language Models* | ICLR 2024, [arXiv:2310.13548](https://arxiv.org/abs/2310.13548) | Evaluates 5 RLHF models across 4 free-form tasks + human preference logs. | RLHF preference optimization systematically induces **sycophancy**—models echo user misconceptions and biases because human raters prefer confident agreement. |
| **Kelley & Riedl (2026)**, *Personalization Increases Affective Alignment...* | [arXiv:2603.00024](https://arxiv.org/abs/2603.00024) | Frontier LLMs tested with user profiles/history under *advisor* vs. *social peer* role framing. | Personalization always boosts affective alignment, **strengthens** pushback when framed as an advisor, but **erodes epistemic independence** when framed as a peer, fostering intellectual echo chambers. |

---

## 5. Synthesizing the Research Gap

Mapping the literature reveals exactly why testing **only** "one shared model vs. a mix of models" on text similarity would overlap heavily with existing work—and where a novel, high-signal hackathon contribution sits:

1. **Already established:**
   * Single-turn outputs collapse across frontier labs (*Argument Collapse*, Kim et al., 2026; *Artificial Hivemind*, Jiang et al., 2025).
   * Static cross-model errors remain correlated (*Correlated Errors*, Kim et al., 2025; *Consensus is Not Verification*, Denisov-Blanch et al., 2026).
   * Single-metric embedding cosine similarity correlates weakly with human form judgments ($\rho = 0.140$); Qiu et al. (2026) evaluate 18 LLMs across 25 interpretable metrics spanning 5 dimensions (semantic, lexical, syntactic, structural, and stylistic/presentational).
2. **Untested empirical gap (our contribution):**
   * **Does personalization preserve *epistemic health*, or merely swap monoculture for echo-chamber fragmentation?** Thejaswi et al. (2026) prove mathematically that personalization preserves *stylistic/feature diversity*, while Kelley & Riedl (2026) and Sharma et al. (2024) warn that personalization triggers *sycophancy* and erodes *epistemic independence* in peer-like interactions. Nobody has tested **multi-round LLM agent populations** comparing:
     1. **No Assistant** (independent/peer-only baseline)
     2. **One Shared Assistant** (single-guru monoculture)
     3. **Multi-Lab Model Mix** (testing whether cross-lab *deliberation* helps calibration per Schneider & Schramm, 2025, vs. collapsing per Kim et al., 2026)
     4. **Personalized Assistants** (persona/prior-conditioned assistants — plus optional advisor vs. peer role split)
   * **Measuring *epistemics* rather than just text embeddings:** Evaluating agents on **both ground-truth calibration questions** (verifiable facts/forecasts $\rightarrow$ accuracy, Brier score, ECE, confident-wrong rate) and **normative debate questions** ($\rightarrow$ belief entropy, polarization/bimodality, Global North median drift) directly operationalizes Goldman (2001), DeMarzo et al. (2003), Capraro et al. (2025), and Finnveden’s calibration/trust framing.

---

## 6. Literature Review for the Three Follow-Up Control Arms

While the 16-task `P0–P4` study cleanly isolates homogeneous social herding (`P1 vs P0`) and counter-anchoring rescue within a fixed model (`P2 vs P1`), isolating **architectural diversity** from **constituent model capability** and **anti-consensus framing** from **reflection length/compute** requires three targeted control arms (`configs/polyphony_controls.yaml`). Each control directly operationalizes an open confound identified in the multi-LLM aggregation and debate literature:

### 6.1 Control 1 — Silent Multi-LLM Ensembles vs. Social Deliberation (`P0_POLY`)
* **Gap in `P0–P4`:** `P0` is a 100% homogeneous `gemma-4-26b-a4b-it` independent panel, whereas `P3` and `P4` combine 4 architectures *with* social peer telemetry. Thus `P0–P4` lacks a **silent 4-architecture baseline** (`P0_POLY`) to test whether multi-round social interaction helps or hurts a heterogeneous panel relative to independent aggregation.
* **Primary Literature:**
  * **Chuang et al. (EMNLP 2025, [arXiv:2501.17310](https://arxiv.org/abs/2501.17310)), *Probing LLM World Models: Enhancing Guesstimation with Wisdom of Crowds Decoding*:** Establishes on real-world quantitative estimation benchmarks (*MARBLES*, *FUTURE*, *ELECPRED*) that passive median aggregation across independent stochastic generations outperforms single-pass and greedy decoding, providing the natural non-social "wisdom of crowds" reference baseline.
  * **Schoenegger, Tuminauskaite, Park, Sousa Bastos & Tetlock (*Science Advances* 2024, `doi:10.1126/sciadv.adp1528`, [arXiv:2402.19379](https://arxiv.org/abs/2402.19379)), *Wisdom of the Silicon Crowd*:** Demonstrates that simple unweighted averaging across a diverse 12-LLM ensemble rivals a 925-person human forecasting tournament—showing strong static cross-model ensembling gains *without* agent-to-agent deliberation.
  * **Smit, Grinsztajn, Duckworth, Barrett & Pretorius (ICML 2024, [arXiv:2311.17371](https://arxiv.org/abs/2311.17371)), *Should we be going MAD?*:** Shows that default multi-agent debate does not reliably outperform silent ensembling / self-consistency while incurring extra latency and token cost, unless agreement dynamics are explicitly modulated.
  * **Wang, Wang, Athiwaratkun, Zhang & Zou (2024, [arXiv:2406.04692](https://arxiv.org/abs/2406.04692)), *Mixture-of-Agents Enhances Large Language Model Capabilities* vs. Schneider & Schramm (2025, [arXiv:2512.22625](https://arxiv.org/abs/2512.22625)):** Wang et al. argue that multi-model layering creates collaborative synergy, whereas Schneider & Schramm show gains depend critically on structured deliberation among diverse forecasters rather than naive repetition.
* **Experimental Design (`P0_POLY`):** Identical 8-agent, 4-model panel (`gemma-4-26b-a4b-it`, `deepseek-v4-flash`, `qwen3.8-27b`, `glm-5.2`) over 2 rounds at $\alpha = 0.4$, **without** peer median/IQR telemetry (`P3 vs P0_POLY` and `P4 vs P0_POLY`).

### 6.2 Control 2 — Constituent Homogeneous Controls (`P0_DS`/`P1_DS`, `P0_QWEN`/`P1_QWEN`, `P0_GLM`/`P1_GLM`)
* **Gap in `P0–P4`:** In `P3` and `P4`, 6 of 8 `gemma-4-26b-a4b-it` seats are replaced by `deepseek-v4-flash`, `qwen3.8-27b`, and `glm-5.2`. Because in-panel error differs sharply across architectures, comparing mixed `P3` (`0.01006`) against Gemma-only `P1` (`0.02290`) confounds **architectural diversity** with **stronger constituent models**.
* **Primary Literature:**
  * **Wynn, Satija & Hadfield (2025, [arXiv:2509.05396](https://arxiv.org/abs/2509.05396)), *Talk Isn't Always Cheap: Understanding Failure Modes in Multi-Agent Debate*:** Explicitly studies capability-heterogeneous multi-agent panels and shows that naive debate can degrade performance over rounds even when stronger models outnumber weaker ones, because strong models defer to articulate falsehoods.
  * **Estornell & Liu (NeurIPS 2024, pp. 28938–28964), *Multi-LLM Debate: Framework, Principals, and Interventions*:** Proves theoretically that when debating models share training-data misconceptions or uniform capabilities, peer debate reinforces majority falsehoods unless intervened upon with misconception-refutation or diversity-pruning.
  * **Kim et al. (ICML 2025, [arXiv:2506.07962](https://arxiv.org/abs/2506.07962)) & Denisov-Blanch et al. (2026, [arXiv:2603.06612](https://arxiv.org/abs/2603.06612)):** Show that stronger models often exhibit *higher* inter-model error correlation, making capability-matched constituent baselines mandatory before attributing ensemble gains to architectural independence.
  * **Krogh & Vedelsby (1995, NIPS 7) & Page (2007):** Under squared error, ensemble error equals average individual error minus prediction diversity ($\bar{e} - D$). Comparing a mixed panel against an average across each constituent's own $K=1$ panel isolates $\Delta D$ from $\Delta \bar{e}$.
* **Experimental Design (`P0_DS`, `P1_DS`, `P0_QWEN`, `P1_QWEN`, `P0_GLM`, `P1_GLM`):** Unsocial (`P0_*`) and naive-telemetry (`P1_*`) 8-agent homogeneous panels for each remaining constituent model, allowing direct comparison of `P3` against the unweighted mean of the four $K=1$ social panels ($\frac{1}{4}[E_{P1,\text{Gemma}} + E_{P1,\text{DS}} + E_{P1,\text{Qwen}} + E_{P1,\text{GLM}}]$).

### 6.3 Control 3 — Length-Matched Generic Reflection (`P1_REFLECT`)
* **Gap in `P0–P4`:** `P2` cuts Gemma herding MSE by $-48.4\%$ relative to `P1`, but adds a ~53-word counter-anchoring block (`_P_COUNTER_ANCHOR_TEXT`). This leaves open whether the rescue stems specifically from **adversarial critique of the peer median/prior**, or merely from extra reasoning tokens / generic "think step by step" compute.
* **Primary Literature:**
  * **Buçinca, Malaya & Gajos (2021, *Proc. ACM Hum.-Comput. Interact.* 5, CSCW1, Art. 188):** Demonstrates that *cognitive forcing functions*—specifically interventions that interrupt automatic deference to an external anchor before commitment—reduce overreliance beyond passive interface exposure.
  * **Huang, Chen, Mishra, Zheng, Yu, Song & Zhou (ICLR 2024, [arXiv:2310.01798](https://arxiv.org/abs/2310.01798)), *Large Language Models Cannot Self-Correct Reasoning Yet*:** Shows that unguided intrinsic self-correction ("verify your answer and reflect carefully") without external feedback or adversarial framing often fails or drifts on reasoning tasks.
  * **Liang, He, Jiao, Wang, Yang, Wang, Tu, Shi & Zhang (EMNLP 2024, pp. 17889–17904, [arXiv:2305.19118](https://arxiv.org/abs/2305.19118)), *Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate*:** Documents **Degeneration-of-Thought (DoT)** under homogeneous self-reflection, arguing that explicit adversarial challenge is required to break early fixation.
  * **Taubenfeld, Dover, Reichart & Goldstein (EMNLP 2024, [arXiv:2402.04049](https://arxiv.org/abs/2402.04049)), *Systematic Biases in LLM Simulations of Debates*:** Cautions that even under explicit debating directives, LLMs gravitate toward baked-in default priors unless forced to test competing physical/factual mechanisms.
* **Experimental Design (`P1_REFLECT`):** Matches `P1` (8 `gemma-4-26b-a4b-it` agents + peer consensus telemetry) plus a token-length-matched (~50-word) **neutral reflection** block instructing unit checks, arithmetic verification, and historical-timeframe checks *without* adversarial tail-testing or questioning the peer median (`P2 vs P1_REFLECT`).

---

## Bibliography (Verified Identifiers)

1. Ashery, A. F., Aiello, L. M., & Baronchelli, A. (2025). *Emergent Social Conventions and Collective Bias in LLM Populations*. *Science Advances*. [arXiv:2410.08948](https://arxiv.org/abs/2410.08948)
2. Buçinca, Z., Malaya, M. B., & Gajos, K. Z. (2021). To Trust or to Think: Cognitive Forcing Functions Can Reduce Overreliance on AI in AI-assisted Decision-making. *Proc. ACM Hum.-Comput. Interact.*, 5(CSCW1), Article 188, 1–21.
3. Centola, D., Becker, J., Brackbill, D., & Baronchelli, A. (2018). Experimental evidence for tipping points in social convention. *Science*, 360(6393), 1116–1119. [doi:10.1126/science.aas8827](https://doi.org/10.1126/science.aas8827)
4. Chiang, T. (2023). ChatGPT Is a Blurry JPEG of the Web. *The New Yorker* (Feb 2023).
5. Chuang, Y.-S., et al. (2023). Simulating Opinion Dynamics with Networks of LLM-based Agents. [arXiv:2311.09618](https://arxiv.org/abs/2311.09618)
6. Chuang, Y.-S., Narendran, S., Harlalka, N., Cheung, A., Gao, S., Suresh, S., Hu, J., & Rogers, T. T. (2025). Probing LLM World Models: Enhancing Guesstimation with Wisdom of Crowds Decoding. In *Proc. EMNLP 2025*, pp. 4699–4713. [arXiv:2501.17310](https://arxiv.org/abs/2501.17310)
7. Davey, X. R., Mishra, S., Adelöw, A., & Liu, S. (2026). The Multilingual Artificial Hivemind: Do All LLMs Think Alike in Every Language? [arXiv:2610.16755](https://arxiv.org/abs/2610.16755)
8. Deepthika S, Yadav, P., & Sankar, B. (2026). Who Is Sycophantic to Whom? Confident LLMs, Human Deference, and the Collapse of Trust Calibration. [arXiv:2610.31579](https://arxiv.org/abs/2610.31579)
9. DeMarzo, P. M., Vayanos, D., & Zwiebel, J. (2003). Persuasion Bias, Social Influence, and Unidimensional Opinions. *Quarterly Journal of Economics*, 118(3), 909–968.
10. Denisov-Blanch, Y., Kazdan, J., Chudnovsky, J., Schaeffer, R., Guan, S., Adeshina, S., & Koyejo, S. (2026). Consensus is Not Verification: Why Crowd Wisdom Strategies Fail for LLM Truthfulness. [arXiv:2603.06612](https://arxiv.org/abs/2603.06612)
11. Estornell, A., & Liu, Y. (2024). Multi-LLM Debate: Framework, Principals, and Interventions. In *Advances in Neural Information Processing Systems 37 (NeurIPS 2024)*, pp. 28938–28964.
12. Finnveden, L. (2024–2025). *What's important in "AI for epistemics"?* (and related Forethought / Open Philanthropy notes with O. Cotton-Barratt).
13. Fu, T., Conde Díaz, J., Arriaga Prieto, C., Martínez Rúiz De Arcaute, G., Reviriego, P., & Coronado-Blazquez, J. (2026). Beyond the Hivemind: Escaping LLM Homogeneity via Meta-Persona Anchoring and Sequential Temperature Scaling. [arXiv:2608.02618](https://arxiv.org/abs/2608.02618)
14. Goldman, A. I. (2001). Experts: Which Ones Should You Trust? *Philosophy and Phenomenological Research*, 63(1), 85–110.
15. Huang, J., Chen, X., Mishra, S., Zheng, H. S., Yu, A. W., Song, X., & Zhou, D. (2024). Large Language Models Cannot Self-Correct Reasoning Yet. *ICLR 2024*. [arXiv:2310.01798](https://arxiv.org/abs/2310.01798)
16. Jiang, L., Chai, Y., Li, M., Liu, M., Fok, R., Dziri, N., Tsvetkov, Y., Sap, M., Albalak, A., & Choi, Y. (2025). Artificial Hivemind: The Open-Ended Homogeneity of Language Models (and Beyond). [arXiv:2510.22954](https://arxiv.org/abs/2510.22954)
17. Kelley, S., & Riedl, C. (2026). Personalization Increases Affective Alignment but Has Role-Dependent Effects on Epistemic Independence in LLMs. [arXiv:2603.00024](https://arxiv.org/abs/2603.00024)
18. Kim, E., Garg, A., Peng, K., & Garg, N. (2025). Correlated Errors in Large Language Models. *ICML 2025*. [arXiv:2506.07962](https://arxiv.org/abs/2506.07962)
19. Kim, Y., Chang, Y., Pham, C. M., & Iyyer, M. (2026). Argument Collapse: LLMs Flatten Long-Form Public Debate. *EMNLP 2026*. [arXiv:2606.01736](https://arxiv.org/abs/2606.01736)
20. Kleinberg, J., & Raghavan, M. (2021). Algorithmic monoculture and social welfare. *PNAS*, 118(22), e2018340118.
21. Krogh, A., & Vedelsby, J. (1995). Neural network ensembles, cross validation, and active learning. In *Advances in Neural Information Processing Systems 7 (NIPS 1994)*, pp. 231–238.
22. Liang, T., He, Z., Jiao, W., Wang, X., Yang, Y., Wang, R., Tu, Z., Shi, S., & Zhang, Z. (2024). Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate. In *Proc. EMNLP 2024*, pp. 17889–17904. [arXiv:2305.19118](https://arxiv.org/abs/2305.19118)
23. Padmakumar, V., & He, H. (2024). Does Writing with Language Models Reduce Content Diversity? *ICLR 2024*. [arXiv:2309.05196](https://arxiv.org/abs/2309.05196)
24. Page, S. E. (2007). *The Difference: How the Power of Diversity Creates Better Groups, Firms, Schools, and Societies*. Princeton University Press.
25. Peterson, A. J. (2024). AI and the Problem of Knowledge Collapse. *AI & Society*. [arXiv:2404.03502](https://arxiv.org/abs/2404.03502)
26. Postman, N. (1985). *Amusing Ourselves to Death: Public Discourse in the Age of Show Business*.
27. Qiu, Z., Hu, R., Ding, Y., & Underwood, T. (2026). Revisiting LLM Homogeneity Through a Multidimensional Lens (18 LLMs, 25 interpretable metrics across 5 dimensions). [arXiv:2611.01549](https://arxiv.org/abs/2611.01549)
28. Quattrociocchi, W., Capraro, V., & Perc, M. (2025). Epistemological Fault Lines Between Human and Artificial Intelligence. [arXiv:2512.19466](https://arxiv.org/abs/2512.19466)
29. Rios-Sialer, I. (2026). The Homogenization Problem in LLMs: Towards Meaningful Diversity in AI Safety. [arXiv:2601.06116](https://arxiv.org/abs/2601.06116)
30. Schaeffer, R., Miranda, B., Kazdan, J., Chudnovsky, J., & Koyejo, S. (2026). How Strong Is the Evidence for the Artificial Hivemind? Reevaluating Evidence for the Open-Ended Homogeneity of Language Models. [arXiv:2609.33936](https://arxiv.org/abs/2609.33936)
31. Schneider, P., & Schramm, A. (2025). The Wisdom of Deliberating AI Crowds: Does Deliberation Improve LLM-Based Forecasting? [arXiv:2512.22625](https://arxiv.org/abs/2512.22625)
32. Schoenegger, P., Tuminauskaite, I., Park, P. S., Sousa Bastos, R. V., & Tetlock, P. E. (2024). Wisdom of the Silicon Crowd: LLM Ensemble Prediction Capabilities Rival Human Crowd Accuracy. *Science Advances*, 10(45), eadp1528. [doi:10.1126/sciadv.adp1528](https://doi.org/10.1126/sciadv.adp1528) | [arXiv:2402.19379](https://arxiv.org/abs/2402.19379)
33. Sharma, M., Tong, M., Perez, E., et al. (2024). Towards Understanding Sycophancy in Language Models. *ICLR 2024*. [arXiv:2310.13548](https://arxiv.org/abs/2310.13548)
34. Smit, A., Grinsztajn, N., Duckworth, P., Barrett, T. D., & Pretorius, A. (2024). Should we be going MAD? A Look at Multi-Agent Debate Strategies for LLMs. In *Proc. ICML 2024*. [arXiv:2311.17371](https://arxiv.org/abs/2311.17371)
35. Sourati, Z., Karimi-Malekabadi, F., Ozcan, M., McDaniel, C., Ziabari, A., Trager, J., Tak, A., Chen, M., Morstatter, F., & Dehghani, M. (2026). The Shrinking Landscape of Linguistic Diversity in the Age of Large Language Models. *Nature Human Behaviour*. [doi:10.1038/s41562-026-02550-0](https://doi.org/10.1038/s41562-026-02550-0) | [arXiv:2502.11266](https://arxiv.org/abs/2502.11266)
36. Sourati, Z., Ziabari, A. S., & Dehghani, M. (2025). The Homogenizing Effect of Large Language Models on Human Expression and Thought. [arXiv:2508.01491](https://arxiv.org/abs/2508.01491)
37. Tao, Y., Viberg, O., Baker, R. S., & Kizilcec, R. F. (2024). Cultural bias and cultural alignment of large language models. *PNAS Nexus*.
38. Taubenfeld, A., Dover, Y., Reichart, R., & Goldstein, A. (2024). Systematic Biases in LLM Simulations of Debates. In *Proc. EMNLP 2024*. [arXiv:2402.04049](https://arxiv.org/abs/2402.04049)
39. Thejaswi, S., Kulshreshta, J., & Oettershagen, L. (2026). Linguistic Monoculture in LLM-Assisted Language Use. [arXiv:2607.27134](https://arxiv.org/abs/2607.27134)
40. Wang, J., Wang, J., Athiwaratkun, B., Zhang, C., & Zou, J. (2024). Mixture-of-Agents Enhances Large Language Model Capabilities. [arXiv:2406.04692](https://arxiv.org/abs/2406.04692)
41. Wynn, A., Satija, H., & Hadfield, G. K. (2025). Talk Isn't Always Cheap: Understanding Failure Modes in Multi-Agent Debate. [arXiv:2509.05396](https://arxiv.org/abs/2509.05396)

