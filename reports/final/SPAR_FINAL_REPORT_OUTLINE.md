# SPAR Final Report Outline

Working title: **Representation Without Control: Testing the Realization Effect in Language Models**

Alternative titles:
- **When a Model Knows an Outcome Is Realized, Does It Act Differently?**
- **The Realization Effect in LLMs: Behavioral Replication, Activation Readout, and a Null Steering Result**
- **Represented but Not Causal: A Case Study in LLM Realization Framing**

Template source: `the-kairos-project/spar-report-template`, adapted from the template sections `Abstract`, `Introduction`, `Methodology`, `Preliminary Results`, `Challenges & Open Questions`, and `Next Steps`. For the final report, this outline converts the midterm-style structure into a paper-style structure: `Abstract`, `Introduction`, `Related Work`, `Methods`, `Results`, `Discussion`, `Limitations`, `Future Work`, and `Conclusion`.

## Central Claim

The original behavioral hypothesis was that LLMs would reproduce a realization effect: risk-taking would differ depending on whether prior gains or losses were merely paper outcomes or had been realized. Our final result is narrower and more mechanistic. Gemma appears to linearly represent the realized-versus-open distinction in its residual stream, including on held-out readout prompts, but steering along that direction does not robustly control downstream risk behavior in our task.

The most defensible claim:

> We find evidence that realization status is represented in Gemma activations and generalizes to held-out readout prompts, but not that this representation is a simple causal lever for risk-taking behavior under our assay.

## Abstract

Draft:

Large language models are increasingly used as behavioral simulators, but it remains unclear when their choices reflect human-like cognitive mechanisms rather than prompt-sensitive surface patterns. We study the realization effect, a behavioral-economics finding in which risk-taking differs after paper versus realized gains and losses. We first adapt casino and investment-style realization scenarios into LLM prompts that elicit a wager and a risk preference. Initial behavioral results suggested condition sensitivity, but follow-up tests showed that the effect was not robust across models and prompt variants. We then analyzed Gemma residual-stream activations and found that a train-only realization direction separates realized/closed outcomes from hypothetical or paper/open outcomes on held-out prompts, including newly generated DeepSeek-authored prompts. However, activation steering along a layer-18 realization direction did not reliably shift risk choices, despite perturbing generation and format compliance. These results suggest a representation-behavior dissociation: models may encode semantically meaningful state variables without those variables being sufficient to causally drive the measured behavior.

## 1. Introduction

Purpose:
- Motivate the realization effect as a test case for whether LLMs reproduce human behavioral regularities.
- Connect this to AI safety and alignment: LLMs are often used for simulation, forecasting, user modeling, policy prototyping, and behavioral experiments.
- Introduce the interpretability question: if a model behaves as if a factor matters, can we find that factor internally, and does intervening on it change behavior?

Key framing:
- The project began as a behavioral replication attempt.
- It became a test of whether a readable internal representation actually controls behavior.
- This matters because mechanistic interpretability often finds linearly decodable features, but decodability alone does not imply causal control.

Research questions:
1. Do LLMs show behavioral sensitivity to whether prior outcomes are paper/open versus realized/closed?
2. Is realization status linearly represented in model activations, including on held-out prompts?
3. Does steering along a realization direction causally change downstream risk choices?

Suggested final paragraph:

Our contribution is a negative but informative case study. We do not find a stable behavioral realization effect across the models and interventions tested. We do find evidence for a Gemma realization representation, including a train-only held-out readout check, but steering that representation does not robustly move risk-taking. This makes the project useful less as a replication of human risk behavior and more as a cautionary example: semantic representations can be readable without being simple behavioral control knobs.

## 2. Related Work

### 2.1 The realization effect

Include:
- Imas (2016): realized versus paper losses and risk-taking.
- Merkle et al. (2020): extension to gains.
- Flepp, Meier, and Franck (2021): casino field setting used as the primary inspiration for our prompt design.

Main idea to explain:
- Paper outcomes occur while the account is still open.
- Realized outcomes occur after the account is closed or cashed out.
- Human risk-taking may differ because closing an account changes how prior gains/losses are mentally integrated.

### 2.2 LLMs as behavioral subjects

Discuss:
- Why people use LLMs as proxies for human judgment.
- Why this is risky: LLM outputs may be shaped by prompt wording, learned stereotypes, instruction-following priors, or numeric anchors rather than cognitive mechanisms.
- Our project tests whether a behavioral-economics pattern survives controlled prompting and mechanistic follow-up.

### 2.3 Linear representations and causal interventions

Discuss:
- Linear probes and activation directions can reveal represented information.
- But represented information may be epiphenomenal or only one input among many.
- Steering provides a stronger causal test than readout alone.

Useful framing:
- Behavioral correlation -> activation readout -> activation steering.
- Each step asks for stronger evidence than the previous one.

## 3. Methods

### 3.1 Behavioral task

Describe the task:
- Models read short vignettes about prior outcomes.
- Conditions vary whether outcomes are paper/open or realized/closed.
- Domains include casino gambling and finance/investment scenarios in the activation-vector pass.
- The model answers with two integers:
  - wager or investment amount, constrained to 1-1000
  - risk preference, constrained to 1-5

Primary outcomes:
- `parsed_amount`
- `risk_profile`
- exactly-two-integer compliance

Files to reference:
- `configs/realization_effect/conditions.csv`
- `results/results.csv`
- `experiments/activation_analysis/prompts/activation_vectors/realization_vector_v1.csv`

### 3.2 Behavioral analysis

Original cleaned behavioral dataset:
- `results/results.csv`
- 46,750 canonical rows after reconciling the absolute and balance prompt blocks.
- 45,865 valid wager rows and 45,808 valid risk-profile rows.
- OLS regressions with condition dummies plus model, temperature, and prompt-version fixed effects.
- HC3 robust standard errors.

Current cleaned behavioral read:
- For `risk_profile`, paper outcomes show more consistently positive movement than wagers.
- For realized outcomes, model responses diverge from the original human prediction: realized losses increase risk-taking rather than decreasing it.
- Overall: models are condition-sensitive, but not in the clean human-realization-effect pattern.

### 3.3 Activation dataset and realization vector

Describe:
- We generated paired prompts contrasting `paper_open` and `realized_closed` framings.
- We logged Gemma residual-stream activations.
- We built a mean-difference realization direction:
  - positive direction: `realized_closed - paper_open`
  - main layer used for steering: layer 18
- Steering artifact caveat: the layer-18 direction used in steering was computed from all complete paired prompts in the original activation run, including direction-training, direction-validation, and behavior-evaluation splits.
- Stricter readout check: a second layer-18 direction was rebuilt from `direction_train` only and evaluated on original `direction_val`, original `behavior_eval`, and a newly generated DeepSeek held-out set.
- DeepSeek held-out set:
  - 40 matched paper/realized pairs.
  - 28 `heldout_readout` pairs and 12 `heldout_behavior_eval` pairs.
  - Gain/loss contrasts only; neutral cells were omitted after preliminary generation made neutral wording ambiguous.
  - Local overlap audit found no reused original prompts; one internal similarity flag reflected shared behavior-question wording, not an exact duplicate.

Files to reference:
- `results/final/activation_vectors/realization_vector_v1_layer18/`
- `results/final/activation_vectors/realization_vector_v1_layer18_direction_train_only/`
- `experiments/activation_analysis/prompts/activation_vectors/realization_vector_heldout_v1.csv`
- `results/audits/heldout_prompt_overlap.csv`
- `src/activation_analysis/vector_analysis.py`
- `src/activation_analysis/residual_streams.py`

### 3.4 Steering intervention

Describe:
- We added a residual-stream steering hook for local Gemma.
- Direction is normalized before injection.
- Steering is applied at layer 18.
- Position mode used in full pilots: `last`.
- Tested scales: `-50`, `+50`, `+75`, `+100`, `+150`.

Implementation files:
- `src/activation_analysis/steering.py`
- `scripts/steer_realization_direction.py`
- `docs/steering_architecture.md`

Output location:
- Raw steering runs: `results/test/activation_vectors/steering_runs/`
- Report package: `results/final/report_realization_v1/03_steering_intervention/`

Important caveat:
- Qwen API behavior runs can test behavioral replication, but OpenRouter-style hosted APIs cannot support activation steering because hidden states and hooks are not exposed.

## 4. Results

### 4.1 Behavioral replication is not robust

Report:
- Initial behavioral runs showed condition sensitivity.
- After cleaning and replication, the pattern did not match the original human realization-effect predictions in a stable way.
- The strongest broad finding is not "LLMs reproduce the realization effect"; it is "LLMs respond systematically to outcome framing and magnitude, but not according to the intended mental-accounting mechanism."

Use numbers from `reports/current_findings.md`:
- Canonical rows: 46,750.
- Valid wager rows: 45,865.
- Valid risk-profile rows: 45,808.
- Paper outcomes, `risk_profile` coefficients relative to baseline:
  - `paper_loss_large`: +0.0764, significant.
  - `paper_loss_medium`: +0.0287, significant.
  - `paper_loss_small`: -0.0120, not significant.
  - `paper_gain_small`: +0.1004, significant.
  - `paper_gain_large`: +0.2470, significant.
- Realized outcomes, `risk_profile`:
  - `realized_extreme_loss`: +0.1323, significant.
  - `realized_large_loss`: +0.1434, significant.
  - `realized_medium_loss`: +0.0063, not significant.
  - `realized_gain`: +0.2834, significant.

Interpretation:
- These effects show outcome sensitivity.
- They do not validate the original realization-effect mechanism.

Suggested figure:
- Condition coefficient plot for wager and risk profile from cleaned behavioral analysis.

### 4.2 Gemma contains a readable realization representation

Report:
- The activation analysis supports a realization readout: `paper_open` and `realized_closed` prompts separate along a mean-difference vector.
- This is the positive result of the project.
- It justifies asking the causal question: if we push the model along that direction, does risk behavior move?
- The original visualization uses the all-pairs readout artifact and should be described as descriptive rather than the strict held-out test.
- The train-only direction gives a genuine held-out readout check:
  - original `direction_val`: 756 pairs, mean projection delta +413.43, correct direction 91.1%.
  - original `behavior_eval`: 324 pairs, mean projection delta +137.19, correct direction 80.6%.
  - DeepSeek `heldout_readout`: 28 pairs, mean projection delta +443.62, correct direction 92.9%.
  - DeepSeek `heldout_behavior_eval`: 12 pairs, mean projection delta +123.08, correct direction 75.0%.
- Interpretation: the representation claim is now stronger than “readable on the training artifact,” but the held-out behavior-eval subset is still small and should not be overclaimed.

Suggested figure:
- Projection distributions for `paper_open` versus `realized_closed` prompts along the layer-18 realization direction.
- Optional: layer-wise separation plot if available.

### 4.3 Steering the realization direction does not robustly shift risk behavior

Matched steering deltas, all valid matched rows:

| Scale | Matched rows | Mean wager delta | Median wager delta | Mean risk delta | Median risk delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| -50 | 483 | +12.795 | 0 | +0.068 | 0 |
| +50 | 475 | +18.326 | 0 | +0.013 | 0 |
| +75 | 480 | +13.017 | 0 | +0.010 | 0 |
| +100 | 482 | +14.521 | 0 | -0.039 | 0 |
| +150 | 475 | +9.133 | 0 | -0.046 | 0 |

Interpretation:
- Positive steering produces small mean wager movements, but medians remain zero.
- Risk changes are essentially null and fluctuate around zero.
- The negative `-50` sign-symmetry run does not reverse the positive-scale pattern.
- Higher scale worsens or does not improve exactly-two-integer compliance.
- The evidence does not support the claim that the layer-18 realization vector is a reliable causal control direction for risk preference.

Suggested figure:
- Dose-response plot for `-50`, `+50`, `+75`, `+100`, `+150` showing mean and median deltas for wager and risk.
- Compliance plot by scale.

### 4.4 Compliance and prompt-source effects

Report:
- Exactly-two-integer compliance is imperfect across all steering runs.
- Sonnet-derived prompts have especially high noncompliance.
- At `+100`, noncompliance:
  - casino: 96/324
  - finance: 103/324
  - GPT-5.4-generated prompts: 46/216
  - Grok-fast-generated prompts: 47/216
  - Sonnet-generated prompts: 106/216

Interpretation:
- Some apparent effects may be driven by parseability and prompt-source artifacts.
- Exactly-two-integer subset analysis is therefore important.

## 5. Discussion

Main interpretation:

This project began with a behavioral hypothesis and ended with a representation-behavior dissociation. The model appears to encode whether an outcome is realized or merely open/paper, but that encoded distinction does not by itself determine risk behavior in our task.

Claims we can make:
- LLMs are sensitive to outcome framing and magnitude.
- Gemma has a linearly readable realization representation that generalizes to held-out readout prompts.
- Direct steering of this representation, as implemented here, does not robustly reproduce the behavioral realization effect.

Claims we should not make:
- LLMs robustly reproduce the human realization effect.
- The Gemma realization vector is a causal risk-taking circuit.
- The behavioral effect is explained by the activation direction.

Why this matters:
- It is a warning against treating linear decodability as explanation.
- It gives a concrete example where a semantic feature is readable but not sufficient for behavioral control.
- It suggests that LLM risk-choice behavior may depend more on task format, instruction following, numeric priors, or other features than on the realization-status representation alone.

## 6. Limitations

Include:
- Steering has only been fully tested at Gemma layer 18 with a limited set of scales.
- We have not completed a layer sweep or position-mode sweep.
- Qwen behavioral tests do not provide activation-level causal evidence.
- Strict compliance is imperfect, especially for Sonnet-derived prompts.
- The risk task may be too indirect for the realization representation to causally control the answer.
- The activation vector may encode realization semantics without capturing the broader decision policy.
- The new DeepSeek held-out set is useful but small and single-source.
- The DeepSeek `heldout_behavior_eval` rows currently test projection separation, not newly generated downstream Gemma wager/risk responses.

## 7. Future Work

Highest-value next steps:
1. Rerun steering from the stricter train-only direction and compare it to the all-pairs steering artifact.
2. Run a layer sweep around layers 14, 16, 18, 20, and 22.
3. Test `position_mode=all` on a small smoke set to see whether prompt-wide steering has stronger causal effects.
4. Expand the held-out prompt set with additional source models and larger behavior-evaluation cells.
5. Generate downstream Gemma wager/risk responses for the DeepSeek `heldout_behavior_eval` prompts.
6. Rerun the direct realization-classification positive control with corrected prompt construction.
7. Build a local Qwen activation pipeline if compute allows, because hosted API behavior tests cannot support steering.
8. Separate prompt-generation-source effects from core realization effects.

## 8. Conclusion

Draft:

We do not find strong evidence that LLMs robustly reproduce the human realization effect in risk-taking. We do find that Gemma represents realization status in its activations, including on held-out readout prompts, but activation steering along this direction does not reliably change downstream wager or risk choices. The project therefore supports a more cautious conclusion: semantic representations in LLMs can be linearly readable without being simple causal levers for behavior. This negative result is useful for both behavioral evaluation and mechanistic interpretability, because it clarifies the evidential gap between observing a behavior, decoding an internal feature, and demonstrating causal control.

## Figures and Tables To Prepare

1. Behavioral coefficient plot:
   - wager and risk-profile coefficients from cleaned canonical behavioral dataset.
2. Activation projection plot:
   - `paper_open` vs `realized_closed` projection distributions along the Gemma layer-18 realization direction.
3. Steering dose-response plot:
   - scales `+50`, `+75`, `+100`, `+150`; mean and median deltas for wager and risk.
4. Steering compliance plot:
   - strict two-integer compliance by scale and prompt source.
5. Summary table:
   - behavioral replication, activation readout, steering intervention, interpretation.

## Minimum Viable Final Report

If time is tight, prioritize:
1. Abstract.
2. Introduction with the revised claim.
3. Methods covering behavioral assay, activation vector, and steering.
4. Results with three subsections:
   - behavioral replication is unstable,
   - realization is readable in Gemma activations,
   - steering does not robustly move risk behavior.
5. Discussion framing this as representation without control.
6. Limitations and future work.

## Bibliography Starter List

- Flepp, R., Meier, P., & Franck, E. (2021). The effect of paper outcomes versus realized outcomes on subsequent risk-taking: Field evidence from casino gambling.
- Imas, A. (2016). The realization effect: Risk-taking after realized versus paper losses.
- Merkle, C., Muller-Dethard, G., & Weber, M. (2020). Realization utility and gains/losses. [Verify exact title/citation before final.]
- Olah et al. / Transformer Circuits references for mechanistic interpretability background. [Add exact citation if used.]
- Turner et al. or related activation-addition/steering work. [Add exact citation if used.]
