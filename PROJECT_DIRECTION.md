# Current project direction

**Status:** active scientific direction, version 1.0. This document is the
canonical statement of scope and claims for the current project.

## The project in one sentence

We are building a benchmark to test which measurable properties of a
theory-relevant internal representation predict causal steerability of an
independent behavioral task.

## Central question

For a language model and a clearly defined behavioral construct:

1. Does a controlled prompt contrast produce a measurable behavioral
   difference?
2. Is the corresponding internal state linearly decodable from activations on
   held-out prompts?
3. Does steering the frozen direction change an independent downstream
   behavior in the predicted direction?

The central hypothesis is:

> Linear decodability is common across behavioral constructs, but decodability
> alone is an incomplete and often poor predictor of causal steerability.

The scientific unit is therefore not “does the model have a behavior?” It is
the relationship between three separate estimands: behavioral sensitivity,
held-out readout, and causal state transfer.

## What this project is and is not

This project is:

- a cross-construct representation–steerability correspondence benchmark;
- grounded in explicit theory-relevant contrasts;
- based on train-only linear directions and independent downstream tasks;
- designed to retain null decodability and null steering results;
- centered initially on additive state-transfer interventions.

This project is not:

- a claim that probe accuracy demonstrates a psychological mechanism;
- a claim that every direction is a behavioral-policy or gain-control vector;
- a replication project for the original human realization-effect pattern;
- the specification-gaming project previously considered for coding agents;
- an implementation-complete benchmark yet;
- a reason to launch large model runs before the schemas and controls are
  frozen;
- a claim that the decodability–steerability relationship is novel by itself.

The specification-gaming direction and the original realization behavioral
pipeline are preserved as historical material, not as the current scope.

The detailed benchmark proposal is in
[`BENCHMARK_RESEARCH_DIRECTION.md`](BENCHMARK_RESEARCH_DIRECTION.md), with a
review-ready brief in
[`BENCHMARK_REVIEW_HANDOFF.md`](BENCHMARK_REVIEW_HANDOFF.md).

## Construct families

The initial construct set is deliberately diverse. Each construct must have a
directional state contrast and a separate downstream task.

| Construct | Family | State to transfer | Independent downstream behavior |
|---|---|---|---|
| Realization/account closure | Decision | Open/paper versus closed/realized account | Risk choice, wager, or related decision |
| Evidence diagnosticity | Epistemic | Evidence perceived as reliable/diagnostic versus weak/unreliable | Confidence or belief revision |
| Source reliability and authority | Social | Deference to a source versus independent verification | Follow the source versus check the evidence |
| Persistence/continuation | Agentic | Continue pursuing a goal versus abandon/reallocate effort | Continue, quit, revise, or reallocate effort |

These labels are working constructs, not conclusions. Before inclusion, each
construct needs an operational definition, a directional outcome, a leakage-
safe prompt design, and a parsing rule that can be frozen in advance.

The first engineering vertical slice should use realization as the anchor and
evidence diagnosticity as the first non-economic construct. Authority/source
reliability and persistence follow only after the first two pass engineering
and measurement checks.

## Common experiment

Every construct follows the same sequence:

1. Define two or more theory-relevant states and the expected behavioral
   direction.
2. Generate matched paired probe prompts with explicit metadata.
3. Freeze train, validation, held-out, and downstream-task splits.
4. Measure the prompt-only behavioral contrast.
5. Build a linear direction using the training split only.
6. Evaluate continuous projections on held-out prompts.
7. Calibrate additive steering doses using training-set projection variation.
8. Steer an independent downstream behavioral task.
9. Run manipulation, compliance, and collateral-behavior checks.
10. Aggregate construct-level effects with uncertainty.

Probe prompts and downstream tasks must not be the same task in different
wording. Otherwise the experiment cannot distinguish construct transfer from
prompt or label matching.

## Primary and secondary outcomes

Primary decodability outcome:

- continuous standardized projection margin on held-out paired prompts.

Pairwise classification accuracy is secondary and diagnostic.

Primary causal outcome:

- directed mean state-transfer effect on the independent behavioral task under
  positive, zero, and negative calibrated additive steering.

Secondary outcomes:

- policy-slope change across task difficulty or evidence strength;
- cross-task transfer;
- downstream persistence of the activation shift;
- output-accessibility or same-task positive controls;
- unrelated behavior, refusal, verbosity, and compliance.

Policy-gain or multiplicative steering is a later exploratory method-development
question. An ordinary additive state direction must not be described as a
policy-gain direction.

## Cross-construct claim

The final comparison will treat model, construct, task, context, and layer as
crossed factors where the design permits. The first analysis will estimate
whether held-out readout strength predicts steering strength across constructs
while propagating uncertainty from both direction construction and behavioral
intervention. A later breadth module may test whether a larger representation
profile—stability, context consistency, localization, and dimensionality—adds
predictive value beyond decodability alone.

A weak relationship is an informative result. We will not select constructs,
layers, signs, doses, or outcomes after inspecting confirmatory results.

## Current implementation status

The repository is currently in **scientific protocol development and
repository preparation**:

- the original realization implementation is archived;
- activation prompt generation and activation primitives remain active;
- the generic `construct_benchmark` package does not yet exist;
- construct, run, and analysis schemas do not yet exist;
- projection-margin, neutral/within-cell dose calibration, state-transfer
  adapters, and manipulation-check orchestration are not yet implemented;
- the active vector iterator now lives in `activation_analysis.activation_store`
  and passes the active Python 3.11 `make check` suite; end-to-end validation
  against a real activation run remains pending;
- no new construct dataset or large benchmark run has been launched.

## Next implementation gates

1. Consolidate the archive and documentation changes into one reviewable
   repository checkpoint.
2. Add a manifest-backed activation-run smoke fixture when a representative
   local run is available, while preserving the passing clean-install checks.
3. Define and validate the three versioned schemas and canonical manifests.
4. Generalize the active paired-prompt generator beyond realization metadata.
5. Implement continuous held-out projection margins and neutral/within-cell
   dose calibration.
6. Build the realization/evidence-diagnosticity vertical slice with
   deterministic fixtures and outcome-specific adapters.
7. Add explicit intervention timing, output-accessibility, downstream-
   persistence, and manipulation checks.
8. Run the precision simulation before expanding the construct count or fitting
   full representation-profile predictors.

Until these gates are passed, documentation should describe the benchmark as
planned rather than implemented.
