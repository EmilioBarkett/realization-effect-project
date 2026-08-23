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

## Frozen construct bank and execution waves

The candidate bank is deliberately balanced across decision, epistemic, social,
and agentic families. Membership and wave order are frozen before confirmatory
results are inspected. Each construct must still pass its own operational,
leakage, parsing, and independent-task gates; a failed construct remains in the
record with an exclusion reason rather than being silently replaced. The
versioned registry is
[`configs/construct_benchmark/construct_registry_v1.json`](configs/construct_benchmark/construct_registry_v1.json).

| Construct ID | Family | State to transfer | Independent downstream behavior |
|---|---|---|---|
| `realization_account_closure` | Decision | Open/pending versus closed/settled account | Risk allocation in a new domain |
| `reference_frame` | Decision | Above-reference versus below-reference outcome | Unrelated sure-versus-risky choice |
| `ambiguity_orientation` | Decision | Accept underspecified probabilities versus prefer resolved probabilities | Known-versus-unknown lottery allocation |
| `temporal_orientation` | Decision | Immediate-consequence versus long-term-consequence focus | Smaller-sooner versus larger-later allocation |
| `evidence_diagnosticity` | Epistemic | Highly diagnostic versus weakly diagnostic evidence | Posterior update magnitude |
| `prior_weighting` | Epistemic | Prior/base-rate-sensitive versus case-evidence-sensitive reasoning | Structured Bayesian probability judgment |
| `causal_interpretation` | Epistemic | Causal versus correlational representation | Intervention-versus-observation prediction |
| `epistemic_uncertainty` | Epistemic | Resolved/certain versus unresolved/uncertain state | Seek more information versus commit now |
| `source_reliability` | Social | Reliable-source versus unreliable-source weighting | Testimony weighting in a new factual domain |
| `authority_deference` | Social | Deference to legitimate authority versus independent verification | Follow advice versus conflicting direct evidence |
| `consensus_conformity` | Social | Follow group consensus versus independent judgment | Factual choice with controlled peer responses |
| `reciprocity_obligation` | Social | Reciprocal obligation versus no obligation | Return/help allocation in a new interaction |
| `persistence_continuation` | Agentic | Continue versus abandon/reallocate after a setback | Resource reallocation after a setback |
| `exploration_exploitation` | Agentic | Explore alternatives versus exploit a known option | Structured search or bandit choice |
| `plan_replanning` | Agentic | Preserve the current plan versus adaptively revise means | Maintain or revise after changed constraints |
| `goal_shielding` | Agentic | Shield the focal goal versus attend to competing goals | Continue focal task versus switch to a distractor |

Execution is balanced one construct per family per wave:

| Wave | Decision | Epistemic | Social | Agentic |
|---|---|---|---|---|
| 1 — anchor | `realization_account_closure` | `evidence_diagnosticity` | `source_reliability` | `persistence_continuation` |
| 2 — weighting/control | `reference_frame` | `prior_weighting` | `authority_deference` | `exploration_exploitation` |
| 3 — uncertainty/adaptation | `ambiguity_orientation` | `causal_interpretation` | `consensus_conformity` | `plan_replanning` |
| 4 — horizon/goal management | `temporal_orientation` | `epistemic_uncertainty` | `reciprocity_obligation` | `goal_shielding` |

Wave 1 is the immediate implementation target. Its four construct specs and
generation plans are defined; the other twelve entries are registry-planned
and intentionally do not yet have generated prompt inventories. The distinction
between source reliability and authority deference, and between diagnosticity
and updating responsiveness, is substantive rather than cosmetic.

The maturity claims are staged: two constructs validate the engineering
vertical slice, four constructs support a descriptive pilot, and the full
16-construct bank is the later matrix needed for out-of-sample
representation-profile prediction.

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
benchmark infrastructure implementation**:

- the original realization implementation is archived;
- activation prompt generation and activation primitives remain active;
- the initial `construct_benchmark` control-plane package now exists;
- construct, run, analysis, prompt, split, and provenance schemas are now
  implemented;
- the versioned 16-construct registry exists, with four specified Wave 1
  construct definitions and four reviewable generation plans;
- the benchmark-facing generation adapter emits canonical `PromptRecord`
  rows, supports deterministic mocks and no-API dry runs, and refuses to treat
  partial inventories as complete;
- the prompt-overlap audit reports construct, split, family, role, template,
  response-format, and probe/downstream independence metadata;
- fixture-tested train-only directions, continuous held-out projection margins,
  neutral/within-condition calibration, strict Wave 1 output parsing, directed
  state-transfer scoring, deterministic steering controls, and timing-aware
  injection are implemented;
- readout, steering-plan, environment-check, remote execution, and scoring CLIs
  now separate frozen scientific decisions from GPU/model execution;
- prompt-only behavior composition, real-model validation, downstream
  manipulation checks, uncertainty orchestration, and correspondence analysis
  are not yet implemented end to end;
- the active vector iterator now lives in `activation_analysis.activation_store`
  and passes the active Python 3.11 `make check` suite; end-to-end validation
  against a real activation run remains pending;
- a shared activation plan can batch the four Wave 1 constructs without
  pooling directions and can later expand to all registry entries;
- no API-generated benchmark dataset, model download, or large benchmark run
  has been launched; the current generation artifact is a no-API dry-run
  summary only.

## Next implementation gates

1. Review the four Wave 1 generation plans, expand them with the approved
   request function only after the no-API dry run is accepted, and connect the
   resulting inventories to the canonical combined activation manifest.
2. Add a manifest-backed activation-run smoke fixture when a representative
   local run is available, while preserving the passing clean-install checks.
3. Validate continuous held-out projection margins and neutral/within-condition
   dose calibration on a representative manifest-backed activation run.
4. Complete the four-construct Wave 1 measurement slice, beginning with the
   realization/evidence engineering adapters and preserving source-reliability
   and persistence namespaces.
5. Add explicit intervention timing, output-accessibility, downstream-
   persistence, and manipulation checks.
6. Run the precision simulation before fitting representation-profile
   predictors or advancing to Waves 2–4.

Until these gates are passed, documentation should describe the benchmark as
an implemented control plane and fixture-tested measurement core without a
validated end-to-end experimental run.
