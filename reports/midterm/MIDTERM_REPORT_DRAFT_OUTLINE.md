# SPAR Midterm Report Draft Outline

## Project
**Working title:** Replicating the Realization Effect in LLM Decision-Making  
**Contributors:** [Add names]  
**Emails:** [Add emails in matching order]

---

## Abstract (45-90 words) — Draft
This project tests whether large language models reproduce the realization effect in gambling decisions. We adapted Flepp, Meier, and Franck’s casino framework into vignette prompts that manipulate whether prior outcomes are paper (within-visit) or realized (between-visits). Across multiple models, temperatures, and prompt framings, we measured wager size and estimated condition effects using OLS with robust errors. Preliminary pooled results show substantial condition effects, but many signs diverge from the human-field predictions, motivating further data cleaning and model-level analysis.

---

## Introduction (90-175 words) — Draft
The realization effect predicts that people take more risk after paper outcomes than after realized outcomes because mental accounts remain open during ongoing play but close after cash-out. This question matters for AI safety and behavioral modeling because LLMs are increasingly used for simulation, preference elicitation, and policy prototyping; if model behavior departs from human decision patterns, downstream conclusions may be misleading. We replicate Flepp et al. (2021) with LLM agents by presenting casino scenarios that vary prior gains/losses and account status (paper vs. realized). We test directional hypotheses on subsequent risk-taking and examine whether effects depend on outcome magnitude. Our contribution is an empirical multi-model benchmark of realization-effect predictions under controlled prompt framing, with transparent code and reproducible analysis.

---

## Related Work (65-130 words) — Draft
Our design is anchored in the realization-effect literature: Imas (2016) distinguishes paper from realized losses, and Merkle et al. (2020) extends predictions to gains. Flepp et al. (2021) provides field evidence from casino data, including stronger risk-taking responses for larger paper outcomes and reduced risk-taking after large realized losses. We use this framework as the primary benchmark and test whether LLMs recover the same directional patterns. This connects behavioral-economics findings to current questions about when LLM outputs can serve as credible proxies for human choices in risky, sequential settings.

---

## Methods (110-200 words) — Draft
We implemented an 11-condition vignette experiment (`conditions.csv`) with two outcome types: paper outcomes (`paper_even`, three paper losses, two paper gains) and realized outcomes (`realized_small_loss` baseline, two larger realized losses plus extreme loss, and realized gain). Each trial asks the model for two integers: next-session wager (1-1000 CHF) and slot-risk preference (1-5). Prompts are generated in `src/realization_effect/runner.py` with two active framings in current data (`absolute`, `balance`; `qualitative` is implemented for future runs).

Current `results/results.csv` includes the cleaned canonical dataset. Analysis in `scripts/analyze_realization_results.py` estimates OLS regressions for `log_wager` and `risk_profile` with condition dummies and model/temperature/prompt-version fixed effects, using HC3 robust standard errors. Hypothesis tests mirror H1a-H4 from Flepp et al. with one-sided directional tests where applicable.

---

## Results (155-265 words) — Draft
### 1) Data and sample
- Valid trials in pooled analysis: **12,138**.
- Regression split used by script: **paper N = 6,065**, **realized N = 6,073**.
- Current file contains both canonical and legacy condition labels, so harmonization is a priority before final estimates.

### 2) Preliminary pooled coefficient pattern (`log_wager`)
- **Paper baseline:** `paper_even`.
- `paper_loss_small = -0.1833***`, `paper_loss_medium = -0.5085***`, `paper_loss_large = -0.3584***`.
- `paper_gain_small = -0.5024***`, `paper_gain_large = +0.1231***`.

- **Realized baseline:** `realized_small_loss`.
- `realized_medium_loss = +0.3948***`, `realized_large_loss = +0.4653***`, `realized_extreme_loss = +0.4175***`.
- `realized_gain = +0.0786***`.

### 3) Hypothesis status (preliminary)
- H1a, H2a, H3a: not supported in pooled run (many effects opposite predicted sign).
- H2b: supported (`gain_large > gain_small`).
- H4: rejected (realized gain differs from baseline).

### 4) Suggested visuals
- Figure 1: mean wager by condition (paper and realized panels).
- Figure 2: coefficient plot with 95% CIs versus each baseline.
- Table 1: Table-2-style regression coefficients and hypothesis verdicts.

---

## Discussion (110-175 words) — Draft
Preliminary results suggest that LLM behavior does not straightforwardly reproduce the human realization-effect pattern. Instead, models respond systematically to prior outcomes, but often in directions that differ from the mental-accounting mechanism proposed in the literature.

In particular, all paper-loss coefficients are negative relative to the paper-even baseline, indicating reduced wagering following paper losses. More strikingly, realized losses are associated with increased wagering relative to the small-loss baseline, directly contradicting the prediction that closing a mental account should reduce subsequent risk-taking.

One interpretation is that models react more to prompt semantics or numeric anchors than to the intended distinction between paper and realized outcomes. However, an alternative explanation is that model behavior reflects a different form of outcome sensitivity, potentially consistent with reference-dependent or prospect-theory-like patterns. The nonlinear response to gains—where small gains reduce wagering while large gains increase it—further suggests that magnitude effects may play a more central role than account closure.

At the same time, the current pooled estimates may be confounded by mixed condition labels and uneven trial counts across model–temperature cells. The strongest robust signal so far is a magnitude effect within paper gains (\texttt{paper\_gain\_large > paper\_gain\_small}), but this alone does not imply full replication of the original framework.

Immediate next steps for the final report are: (1) harmonize legacy condition names into the 11-condition schema, (2) rerun primary regressions on the cleaned sample, (3) report per-model estimates and prompt-framing interactions, and (4) add analysis for the \texttt{risk\_profile} outcome. These checks will clarify whether deviations reflect genuine model behavior or data-integration artifacts.

---

## Conclusion (45-90 words) — Draft
This project builds a reproducible LLM replication of the casino realization-effect design and provides initial quantitative evidence across models and prompt framings. Early pooled findings show strong condition sensitivity but limited alignment with core human-field predictions. The midterm milestone establishes the pipeline and reveals where methodological tightening is needed; the final phase will focus on cleaned-condition analyses, model heterogeneity, and stronger inference about when LLMs can emulate human risk-taking dynamics.

---

## References (starter list)
1. Flepp, R., Meier, P., & Franck, E. (2021). *The effect of paper outcomes versus realized outcomes on subsequent risk-taking: Field evidence from casino gambling*. Organizational Behavior and Human Decision Processes, 165, 45-55.
2. Imas, A. (2016). *The realization effect: Risk-taking after realized versus paper losses*. American Economic Review.
3. Merkle, C., Muller-Dethard, G., & Weber, M. (2020). [Add full citation used in your lit review].
