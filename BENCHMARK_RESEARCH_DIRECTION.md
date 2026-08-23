# Research direction: the representation–steerability correspondence benchmark

**Working title:** Representation–Steerability Correspondence Benchmark
(`RSC-Bench`). `D2S-Bench` was an earlier working name.

**Status:** research proposal plus an implemented multi-construct control-plane
prototype. This document does not claim that the end-to-end benchmark has been
run.

## Executive assessment

This is promising, but the broad claim “benchmark the relationship between
representation and steerability” is no longer distinctive enough. Recent work
has already reported a relationship between representation separability and
linear steerability during training, and newer studies directly examine
detection–intervention dissociations and the sensitivity of steering trends to
normalization and operating point.

The sharper contribution is a benchmark for the *predictive validity of a
representation profile as a causal control signal*:

> Given measurable properties of how a construct is represented internally,
> can we predict how reliably that construct can be causally controlled across
> contexts, tasks, layers, models, and model scales?

This turns the project from another activation-steering study into a
measurement benchmark about the correspondence between representation
geometry and causal control. Held-out linear decodability remains the anchor
feature, but it is no longer the whole representation profile.

The benchmark should estimate which representation properties predict
independent-task state transfer, how well the mapping transfers across models
and contexts, and how often the two properties dissociate.

The project is worth pursuing if it preserves four commitments:

1. decodability is measured on held-out prompts;
2. the direction is frozen before downstream steering is evaluated;
3. the steering task is meaningfully independent from the probe task; and
4. null results, shuffled controls, collateral effects, and uncertainty are
   part of the benchmark rather than filtered out.

A benchmark consisting only of many prompt pairs and a steering leaderboard
would be much less novel and would not test the central scientific claim. A
50–100-concept expansion may eventually be useful for prediction, but it is a
later breadth module. The first release should prioritize construct validity,
independent tasks, and calibrated interventions over an uncurated concept
count.

## The scientific object

For each construct and model, the benchmark records a representation profile
and a steerability profile. The profiles are evaluated at registered layers,
activation sites, contexts, and downstream tasks rather than being silently
collapsed into one favorable number.

### Representation profile (`R`)

The core profile can include:

| Feature | Question | Status in v1 |
| --- | --- | --- |
| Held-out decodability (`D`) | Is the state linearly recoverable on new prompts? | Primary |
| Direction stability | Does the direction persist across resamples and prompt families? | Secondary |
| Cross-context consistency | Does the direction generalize beyond the inducing prompt family? | Secondary |
| Layer localization | Where and how broadly is the signal present? | Secondary |
| Separability/calibration | How cleanly are the states separated beyond accuracy alone? | Secondary |
| Intrinsic dimensionality | Is the state a narrow direction or a broader subspace? | Exploratory |

The last feature is deliberately exploratory. Intrinsic-dimensionality estimates
are expensive and sensitive to sample size, and should not be added to the
primary score without a measurement review.

### Steerability profile (`S`)

The control profile can include:

| Feature | Question | Status in v1 |
| --- | --- | --- |
| Directed behavioral effect | Does intervention transfer the state to an independent task? | Primary |
| Dose-response monotonicity | Does the effect track calibrated intervention strength? | Secondary |
| Minimal intervention norm | How much normalized intervention is needed? | Secondary |
| Specificity | Does the target move without generic response bias or collateral effects? | Secondary |
| Cross-task generalization | Does the effect transfer to a second independent task? | Secondary |

`B`, prompt-only behavioral sensitivity, remains a construct-validity and
quality-control measure. In a small four-construct pilot, the primary analysis
should remain the simple `D → S` relationship. A multivariate prediction of
`S` from the full `R` profile requires a larger, carefully curated concept bank
or repeated model/task cells; it should not be overfit to four constructs.

The benchmark is not trying to establish that a model has a human-like
psychological mechanism. It tests whether reproducibly measured properties of
an internal state are useful predictors of a specified causal intervention.

## Formal benchmark estimands

Let `v` be the direction estimated from the direction-training split, with its
sign fixed by the construct specification. Let `h_i` be the activation for a
held-out prompt and let `s_train` be the pre-registered projection scale from
training calibration prompts.

The primary decodability measure is a held-out standardized projection
margin, schematically:

```text
m_i = dot(h_i, v) / s_train
D = mean(m_positive - m_negative) on held-out paired prompts
```

Pairwise classification accuracy, probe calibration, direction stability, and
layer profiles are secondary diagnostics. They should not replace the
continuous margin as the primary readout because a binary probe can hide
effect-size and calibration differences.

For an independent behavioral outcome `Y`, let `+d`, `0`, and `-d` denote
positive, zero, and negative calibrated intervention doses. The primary
steerability estimand is a signed, standardized directed contrast:

```text
S_mean = expected_sign * [mean(Y | +d) - mean(Y | -d)]
         / [2 * SD(Y | zero dose)]
```

The zero-dose condition is a baseline and manipulation check; it is not
optional merely because the positive and negative doses differ. A dose-response
or policy-slope estimate is secondary. A policy-gain intervention is
exploratory method development, not an assumption of the benchmark.

The cross-construct analysis estimates the predictive relationship, for
example with a hierarchical errors-in-variables model or a bootstrap-aware
equivalent:

```text
S_mean(model, construct, task, layer)
    = alpha + beta * D(model, construct, layer)
      + model effects + construct effects + task effects + error
```

The target scientific quantity is `beta` and the out-of-sample predictive
performance of `D`, not a potentially unstable ratio such as `S / D`.
Reporting should include the scatter, uncertainty, incremental prediction over
baseline controls, and the prevalence of dissociation patterns.

When the benchmark has a sufficiently broad reference set, a secondary
descriptive summary can standardize a representation score and a steerability
score and define a Representation–Steerability Gap:

```text
RSG(c) = z(R_score(c)) - z(S_score(c))
```

Positive `RSG` means that a construct is more accessible than controllable
relative to the reference distribution. The RSG is useful for a two-dimensional
map and for naming dissociations, but it is not the primary estimand: its value
depends on the choice of component metrics, reference set, and normalization.
For v1, report the underlying profile and the `D → S` relationship first.

## Claim levels and required design scale

The benchmark should distinguish three levels of claim:

| Stage | Scientific purpose | Permitted claim |
| --- | --- | --- |
| Two constructs | Engineering and protocol validation | The measurement pipeline works and produces interpretable cells |
| Four constructs across tasks and models | Descriptive benchmark pilot | Decodability–steerability patterns and dissociation frequencies are observed |
| Larger construct × task × model matrix | Predictive correspondence analysis | Representation-profile features predict control out of sample |

The two-construct vertical slice is necessary but cannot support a persuasive
multivariable theory of steerability. The four-construct stage can support a
descriptive pilot, not a heavily parameterized profile predictor. A larger,
hand-curated matrix is required before claiming that stability, accessibility,
or dimensionality predicts causal control beyond decodability alone.

## The four benchmark outcomes

The benchmark should make the following quadrants visible rather than
compressing them into one score:

| Held-out decodability | Independent-task steering | Interpretation |
| --- | --- | --- |
| High | High | A readable direction also functions as a useful causal handle |
| High | Low or null | The core “encoded but not actionable” dissociation |
| Low | High | The mean-difference readout is incomplete or the intervention acts through another feature |
| Low | Low | No evidence for this construct-direction/task cell under the protocol |

The third quadrant is important. A null probe does not prove that the model
lacks the construct, and a successful intervention does not prove that the
chosen probe direction was the model's endogenous representation. The
benchmark evaluates the validity of a method pipeline, not the metaphysical
absence or presence of a concept.

## Construct suite

The current four-construct proposal is a good starting point because it spans
different kinds of behavior without treating behavioral economics as the
whole project.

| Construct | Family | State to transfer | Independent task candidate |
| --- | --- | --- | --- |
| Realization/account closure | Decision | Paper/open account versus realized/closed account | Risk, wager, or related decision |
| Evidence diagnosticity | Epistemic | Evidence perceived as reliable/diagnostic versus weak/unreliable | Confidence or belief revision |
| Source reliability/authority | Social | Deference to a source versus independent verification | Follow the source versus check the evidence |
| Persistence/continuation | Agentic | Continue pursuing a goal versus abandon/reallocate effort | Continue, quit, revise, or reallocate |

These are candidate operationalizations, not conclusions. The evidence
construct needs especially careful wording: the intended transferred state is
*perceived evidence quality or diagnosticity*. It is not automatically a
direction for “updating responsiveness.” Those are different hypotheses and
would require different contrasts and outcomes.

For the final confirmatory benchmark, each construct should have at least one
primary independent task and, if feasible, a second task with a different
response format. One task per construct is acceptable for an engineering
pilot, but it makes construct and task effects difficult to separate. A
stronger benchmark crosses tasks and constructs wherever the scientific
meaning permits it.

## Benchmark axes beyond the first vertical slice

The full correspondence benchmark should eventually vary four axes:

1. **Construct:** the theory-backed state being measured.
2. **Context:** prompt family, topic, and surface realization.
3. **Model:** family, instruction-tuning status, size, and—where available—
   training checkpoint.
4. **Intervention:** layer, position, timing, normalized dose, and steering
   method.

This makes several scientifically useful questions possible:

- Does a representation profile transfer across contexts even when a steering
  effect does not?
- Does representation quality saturate before causal steerability emerges as
  models scale or train?
- Are some behaviors controlled through a different or broader subspace than
  the one that best detects them?

The scale/checkpoint question is a strong extension, but it is not a premise
of the first experiment. A result such as “representation quality saturates
early while controllability emerges later” would be a high-value finding only
after intervention norm, layer choice, and operating point are normalized.

Likewise, a 50–100-concept bank should be treated as a later breadth module.
It should be hand-curated into families with independent tasks and quality
gates, not produced as a large collection of loosely defined labels. The
current four constructs provide theory and behavioral depth; the later bank
would provide statistical breadth for learning a representation-to-control
mapping.

The archived specification-gaming work could later supply a difficult agentic
module—for example, distinguishing “follow the user's intended objective” from
“optimize the evaluator's proxy.” That would test whether correspondence
changes for abstract, behaviorally consequential policies. It should remain a
future extension until the simpler persistence and continuation construct is
validated; reintroducing it now would expand the scientific scope before the
measurement pipeline is stable.

## Inclusion criteria for constructs

A candidate should enter the benchmark only if it satisfies all of the
following:

1. **Directional definition:** two states and an expected behavioral direction
   can be written without relying on a vague personality label.
2. **Theory-backed contrast:** the prompt manipulation has a reason to induce
   the state beyond changing a keyword or response style.
3. **Independent task:** a new downstream task measures the construct without
   repeating the probe's wording, answer format, or surface decision.
4. **Leakage audit:** lexical, format, length, option-order, and template
   effects can be measured and controlled.
5. **Reliable parsing:** the output can be scored with a frozen parser and an
   explicit ambiguity policy.
6. **Falsifiability:** both decodability and steering can plausibly be null.

The benchmark should reject constructs whose only evidence is that a judge can
recognize the desired style after being told what to look for.

Before a construct-specific adapter or generic benchmark code is written, its
specification must state:

- positive and negative state definitions;
- expected behavioral direction and outcome valence;
- elicitation contrast and prompt families;
- independent downstream task;
- nuisance variables and plausible lexical shortcuts;
- invalid, ambiguous, and noncompliant responses;
- controls and predicted nulls.

This is particularly important because evidence diagnosticity and source
reliability can collapse into a shared evidence-quality contrast, while
persistence can measure downstream continuation dynamics rather than a common
upstream state. The realization construct also needs outcome-valence checks;
its direction may reverse or disappear for different decision outcomes.

## Experimental contract for each benchmark cell

1. **Register the construct.** Freeze the states, expected sign, prompt
   families, downstream task, parsing rules, invalidity criteria, and controls.
2. **Generate paired prompts.** Keep pair metadata explicit and make the rows
   reviewable before model execution.
3. **Freeze partitions.** Use direction-training, direction-validation,
   direction-heldout, behavior-evaluation, and steering-evaluation splits.
   Hold out prompt families when testing generalization, not only paraphrases.
4. **Measure baseline sensitivity.** Establish whether the prompt contrast
   produces the registered behavioral change and audit format compliance.
5. **Construct the direction.** Estimate the direction from the training split
   only. Candidate layer and intervention choices may use validation only.
6. **Measure decodability.** Evaluate the frozen direction on held-out
   prompts, using continuous projections as the primary outcome.
7. **Calibrate the intervention.** Express doses in training-calibration
   projection units, register timing and positions, and record residual-norm
   changes as safety diagnostics.
8. **Run independent-task steering.** Include positive, zero, negative,
   shuffled/random, and—where feasible—wrong-layer or unrelated-direction
   controls.
9. **Check the manipulation.** Test downstream activation persistence, output
   accessibility, compliance, and unrelated behavior. Projection at the
   injection layer alone is not a sufficient manipulation check.
10. **Aggregate without selection.** Keep failed constructs, layers, signs,
    doses, and outcomes in the report. Fit the crossed model × construct ×
    task analysis with uncertainty from both readout and steering.

## Calibration and outcome scales

Dose calibration must not mechanically couple decodability and intervention
strength. If the projection standard deviation is computed over a mixture of
positive and negative conditions, more strongly separated constructs may
receive systematically different physical interventions. The analysis
specification must therefore freeze one of:

- variance estimated from neutral calibration prompts; or
- within-condition, within-cell centered activation variance.

Every result should retain both the normalized dose and the unstandardized
intervention magnitude.

The steering-effect denominator must be outcome-specific:

- binary outcome: probability-scale marginal effect;
- bounded score: effect divided by a fixed registered scale or range;
- continuous outcome: baseline or externally defined reference standard
  deviation.

The unstandardized effect must be reported beside every standardized result.
This avoids unstable effect sizes when a binary baseline is near zero or one
or when a continuous outcome has negligible baseline variation.

## What would count as a finding?

The strongest version of the proposed result would be:

1. many theory-relevant states are recoverable above chance on held-out prompts;
2. held-out decodability has weak predictive value for independent-task
   steering after controlling for model, layer, task, prompt compliance, and
   intervention magnitude;
3. high-decoding/low-steering cells are common and survive format, lexical,
   random-direction, and collateral-behavior controls; and
4. a smaller set of positive controls demonstrates that the intervention
   machinery can steer when a suitable causal handle exists.

That result would support the careful claim that **linear decodability is not a
control certificate**. It would not support the stronger claim that decoded
states are causally irrelevant or that the model lacks the underlying trait.

Other outcomes remain scientifically useful:

- If `D` strongly predicts `S`, the benchmark would revise the central
  hypothesis and show when linear readouts are useful control signals.
- If steering occurs with low `D`, the mean-difference probe is not a sufficient
  account of the intervention and the project should investigate alternate
  directions or nonlinear representations.
- If both are weak, the construct, prompts, model, or task may not express the
  intended state under the chosen setup.

## Competing predictors of successful control

The benchmark should not test only whether `D` predicts `S`. Plausible
competitors include:

- output accessibility at the candidate layer;
- layer localization and downstream persistence;
- direction stability across resamples and contexts;
- cross-context and cross-task consistency;
- normalized intervention cost and operating point.

These are scientifically relevant predictors, not merely implementation
details. A direction can be readable at one layer but inaccessible to the
output, or can change the state briefly without surviving generation. The
benchmark should register which predictors are primary, secondary, or
exploratory and use out-of-sample or leave-one-construct-out evaluation once
the matrix is large enough.

Recent work on [Predicting Where Steering Vectors
Succeed](https://arxiv.org/abs/2604.15557) argues that a layer-wise linear
accessibility profile can predict steering success, while [Prompt-Activation
Duality](https://arxiv.org/abs/2605.10664) highlights stateful generation and
KV-cache contamination as failure modes. These results make output
accessibility, intervention timing, and downstream persistence mandatory
measurements for the eventual correspondence benchmark.

## Why this could be new

Existing work establishes important pieces of the landscape. Activation
addition demonstrates inference-time behavioral steering; the linear
representation literature connects linear readouts and intervention geometry;
and controllability benchmarks evaluate whether models follow behavioral
steering instructions. The project should not claim to be the first work to
compare decoding and steering in any broad sense.

The related-work bar is now higher for three specific reasons:

- [How Does Controllability Emerge in Language Models During
  Pretraining?](https://arxiv.org/abs/2508.01892) reports that increasing
  representation separability correlates with the emergence of linear
  steerability across training checkpoints and concepts.
- [Perfect Detection, Failed Control: The Geometry of Knowing vs.
  Steering](https://arxiv.org/abs/2606.24952) directly studies the angle between
  detection and intervention directions and argues that a static geometric
  signature need not predict steerability.
- [Encoded but Not Actionable](https://arxiv.org/abs/2608.17843) studies a
  decode–generate–steer gap in a geometric-reasoning domain, while [When Is a
  Steerable Concept Representation Real?](https://arxiv.org/abs/2608.08159)
  emphasizes normalization and operating-point confounds.

Work such as [Refusal Lives Downstream of Persona in Chat
Models](https://arxiv.org/abs/2606.26161) also suggests that a behavior may be
controlled by interacting or downstream gates rather than one isolated
direction. This supports measuring layer localization, context transfer, and
specificity instead of treating one mean-difference vector as the complete
representation.

The defensible contribution is therefore:

> A calibrated, cross-construct benchmark of the correspondence between a
> representation profile and a causal-control profile, using held-out readout,
> independent behavioral tasks, context/task transfer, and uncertainty-aware
> analysis across models and layers.

This is stronger than asking whether a model can follow a steering instruction,
but it is not equivalent to claiming that no one has measured a
decodability–steerability relationship. The contribution must be the
measurement design and the generalization question: which representation
properties predict control, under what normalization, and when do those
properties fail to transfer? The phrase “decodable but not steerable” alone is
not a sufficient novelty claim.

## Benchmark report card

The public artifact should be a report card rather than a single leaderboard
number. Each model × construct × task × layer cell should expose:

- prompt-only behavioral sensitivity;
- held-out continuous decodability and secondary classification metrics;
- direction stability across resamples and prompt families;
- cross-context consistency and layer-localization profiles;
- intrinsic-dimensionality estimates only where sample size supports them;
- steering effect at positive, zero, and negative calibrated doses;
- dose-response monotonicity and normalized intervention cost;
- dose-response or policy-slope estimates where applicable;
- downstream activation persistence and output-accessibility checks;
- unrelated behavior and collateral effects;
- compliance, parsing exclusions, sample counts, and confidence intervals;
- direction-construction split, layer/site/position settings, intervention
  timing, and random/shuffled controls.

At the aggregate level, report:

- the decodability–steerability scatter and uncertainty;
- the hierarchical slope or incremental predictive `R²` for `D → S`;
- the representation and steerability profiles for each cell;
- the descriptive Representation–Steerability Gap once a reference
  distribution is frozen;
- the distribution of the four dissociation quadrants;
- model, construct, task, and layer heterogeneity;
- sensitivity to alternate valid parsers and prompt-family exclusions.

Avoid an overall score that rewards a large effect with poor specificity or a
highly decodable prompt artifact with no independent behavioral transfer. A
full-profile predictor should be fit only after the concept bank is large
enough to support it; the four-construct pilot should not be used to train a
high-dimensional correspondence model.

## Staged research plan

### Phase 0: stabilize the repository state

- Consolidate the archive reorganization, canonical documents, package changes,
  and Makefile adjustments into one reviewable checkpoint.
- Record the known test, dependency, model, and data limitations.
- Do not build new abstractions on top of an ambiguous repository boundary.

### Phase 1: verify and harden the active activation pipeline

- Verify the new `ActivationVectorRecord` and `iter_activation_vectors()`
  implementation against existing activation-store manifests, shards, and
  indices.
- Keep the active path free of the absent `sae.dataset` dependency rather than
  restoring an obsolete SAE-training package for one iterator.
- Keep tests that genuinely belong to the archived SAE-training pipeline out of
  the active test suite.
- Add iterator, filtering, region, and memory-map regression tests.
- The current clean Python 3.11 editable install passes `make check`; the
  optional PyTorch-dependent interpreter tests are skipped when that extra is
  not installed, and benchmark raw-output paths are now covered by `.gitignore`.

### Phase 2: implement the minimum benchmark core

- Freeze the primary estimands, controls, and quadrant definitions.
- The initial `construct_spec`, `run_config`, and `analysis_spec` schemas,
  canonical prompt inventory, split validation, and shared run manifest are
  now implemented under `src/construct_benchmark/`.
- Use one combined prompt inventory and one activation logging pass for two or
  four constructs, then namespace all direction and outcome artifacts by
  `construct_id`.
- Simulate plausible prompt, readout, and steering effect sizes to determine
  how many constructs, task templates, models, and repetitions are needed.
- Decide whether two downstream tasks per construct are required for the
  confirmatory release.
- Give each schema a version, canonical serialization, and content hash.
- Use a generic prompt-row schema with IDs, pair roles, task, family, domain,
  split, expected direction, prompt text, and validated metadata.
- Keep construct-specific attributes in metadata or adapters rather than
  extending one global CSV schema indefinitely.

### Phase 3: complete the readout vertical slice

- Construct the direction from `direction_train` only and use validation only
  for registered layer or position choices.
- Implement held-out paired projection margins and training-only
  standardization.
- Add bootstrap intervals grouped by pair and prompt family.
- Add secondary accuracy, AUC, layer-stability, and context-transfer measures.
- Record model revision, layer, site, region, roles, signs, prompt hashes,
  normalization, and configuration hashes in direction artifacts.
- Use realization and evidence diagnosticity as the first two constructs.

### Phase 4: implement steering as a controlled experiment

- Register prefill-only, generation-only, every-step, and fixed-window timing
  explicitly.
- Freeze neutral or within-condition calibration variance so dose does not
  mechanically couple decodability and physical intervention strength.
- Include positive, negative, zero, shuffled, and multiple random-direction
  controls, with fixed seeds and randomized condition order.
- Measure post-intervention projections at downstream layers, compliance,
  collateral behavior, and output accessibility.

### Phase 5: add behavioral task adapters

Use a small code interface rather than making every construct fully
declarative:

```python
render(example, condition) -> prompt
parse(output) -> parsed_response
score(parsed_response) -> outcome
collateral(parsed_response) -> control_metrics
```

Implement the realization/account-closure adapter first, followed by evidence
diagnosticity. Prefer structured or constrained outputs for primary outcomes;
use free-text judging only as a secondary measure where necessary.

### Phase 6: build the correspondence analysis

Create one tidy cell-level dataset:

```text
model × construct × task × layer
```

Record readout and steering estimates with uncertainty, intervention norm,
compliance, collateral effects, context-transfer score, layer stability,
output accessibility, and manipulation-check strength. Use hierarchical
bootstrap or errors-in-variables analysis. Once enough constructs exist, use
leave-one-construct-out evaluation for profile-based predictors.

### Phase 7: preregistered pilot and benchmark expansion

- Run the complete two-construct slice on one open-weight model for
  engineering and protocol validation.
- Add a second model family before drawing general conclusions.
- Add authority/source reliability and persistence only after construct review.
- Add a second task template per construct where resources permit.
- Release complete manifests, aggregate summaries, and negative results.

### Phase 8: breadth and scale module

- Build a hand-curated concept bank only after the four-construct protocol is
  stable. A 50–100-concept bank is a possible target, not a prerequisite for
  the vertical slice.
- Add concept families such as style, epistemic, social, safety, and agentic
  behavior only when each has a directional contrast, independent task, and
  frozen parser.
- Compare model families, sizes, and—if accessible—training checkpoints under
  normalized intervention strength and registered operating points.
- Test whether the representation profile predicts the steerability profile
  out of sample, rather than fitting and evaluating the correspondence on the
  same concepts.

### Phase 9: exploratory methods

- Investigate conditional, multiplicative, Jacobian-derived, or
  responsiveness-trained interventions.
- Keep those methods separate from the v1 additive benchmark so a more powerful
  intervention cannot retroactively redefine the primary result.

## First runnable milestone

The minimum credible milestone is:

- realization and evidence diagnosticity;
- one open-weight model for engineering;
- one preregistered layer plus a small validation-only candidate set;
- at least two prompt families per construct;
- one independent behavioral task per construct;
- deterministic fake-model and fake-activation fixtures;
- complete artifact manifests;
- positive, negative, zero, shuffled, and random controls; and
- no prompt-family or item leakage.

This milestone validates the machinery and produces a reviewable pilot. It is
not yet the confirmatory correspondence benchmark. The next scientific step
is a second model family, followed by additional constructs and tasks only
after the first result's provenance and controls are sound.

## Main risks and safeguards

| Risk | Safeguard |
| --- | --- |
| The probe detects format or lexical cues | Family-held-out prompts, shuffled-label controls, length/format matching, and residualized diagnostics |
| A steering dose creates an off-manifold artifact or couples dose to decodability | Neutral or within-cell calibration variance, residual-norm monitoring, zero/negative/random controls, and collateral outcomes |
| The same task is measured twice | Independent downstream task and distinct output format |
| One construct is one task | Two task templates per construct for the confirmatory benchmark, or explicitly label the result a pilot |
| Layer selection leaks outcome information | Register candidates and use validation only; do not select by steering success |
| Parser or judge choices create the effect | Frozen parsers, structured outputs, ambiguity rules, and manual audit samples |
| A null steering result is overinterpreted | Describe it as failure of this direction/intervention/task cell, not proof of no causal representation |
| Too few constructs support a noisy cross-construct claim | Use a precision simulation and report the aggregate relationship with uncertainty |
| A high-dimensional representation profile is overfit | Treat profile prediction as a later out-of-sample module; keep v1 focused on pre-registered primary features |
| Scale or architecture trends are artifacts of intervention units | Normalize dose, residual impact, layer selection, and operating point before comparing models |
| The novelty claim duplicates recent decode–steer audits | Position the contribution around calibrated correspondence, independent behavioral transfer, and cross-context prediction |

## Decisions for review

The next review should answer these questions before implementation begins:

1. Is the central estimand the hierarchical `D → S` predictive relationship,
   rather than a benchmark leaderboard?
2. Should the confirmatory release require two independent downstream task
   templates per construct?
3. Is the primary steering outcome the signed mean state-transfer contrast,
   with policy-slope change secondary?
4. Are the four candidate constructs sufficiently directional and distinct,
   especially evidence diagnosticity versus updating responsiveness?
5. What model/layer coverage gives useful heterogeneity without turning the
   project into an unpowered sweep?
6. Which representation-profile features should be primary in the eventual
   correspondence model, and which should remain exploratory?
7. Should a 50–100-concept breadth module be part of the first paper, or a
   later release after the four-construct protocol is validated?
8. Does the related-work positioning remain distinct from adjacent
   decodability/steering-gap audits and pretraining-emergence studies?

If these decisions are accepted, this benchmark is a credible next direction:
it turns the original realization-effect observation into a calibrated test of
which internal representation properties, if any, make a state controllable.
