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

All 16 construct specifications and paired-vector generation plans now exist
and are marked `specified` in the registry. Wave 2–4 calibration-aware
downstream generation plans and review/full inventory workflow are also
implemented. The existing Wave 2–4 composed inventories have been audited
and are retained as engineering artifacts. The audit found probe-wrapper,
downstream-episode, direct-cue, and task-independence blockers in the frozen
inputs, so no Wave 2–4 inventory is currently released for confirmatory model
execution. The distinction between source reliability and authority
deference, and between diagnosticity and updating responsiveness, is
substantive rather than cosmetic.

The vector-only generation scope is frozen at 100 train pairs, 40 validation
pairs, and 40 held-out pairs per construct (2,880 pairs / 5,760 records across
the 16-construct bank). The completed API-generated vector/probe inventory is
tracked at
`results/benchmark/vector_prompts_v2_luna/full_final_all16/combined.csv` with
its final inventory manifest. It is explicitly non-confirmatory and
`scope_partial`: it contains only vector/probe prompts, not the independent
behavior, calibration, or steering-task inventory. Generation is orchestrated
with four workers and uses `anthropic/claude-sonnet-4.6` only for the vector
inventory. The independent Wave 2–4 downstream engineering source inventory is retained at
`results/benchmark/downstream_prompts_v1_waves2_4_full_luna_b20_o30k_v1/` with
384 records; it is a non-confirmatory Luna engineering artifact, not a
real-model result. The audit and repair record is
`agents/WAVES2_4_PROMPT_AUDIT.md`; repaired confirmatory inputs must receive a
new versioned release after fresh generation and audit.

The next expansion is packaged as three separate four-construct execution
plans, one for each of Waves 2–4, with shared activation logging and
construct-scoped analysis within each wave. Repaired, audited prompt-input
releases are now present under
`results/benchmark/prompt_inventories/wave[2-4]_four_construct_confirmatory_v1/`.
This releases prompt inputs only; the Wave 1 measurement gate and precision
simulation are still required before confirmatory model execution or empirical
claims.

The current Wave 1 repaired model input is
`results/benchmark/prompt_inventories/wave1_repaired_v2_full_openai_luna_normalized/combined.csv`.
Its manifest records 1,824 frozen engineering rows: 1,440 vector/probe rows
and 384 independent behavior, steering, and calibration rows. It remains
non-confirmatory; no real-model downstream result is implied. The older
1,650-row composition remains historical engineering provenance and must not
be mixed with repaired v2.

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
7. Calibrate additive steering doses using neutral or within-condition
   variation, then begin with prefill-only doses `[-1, -0.5, 0, 0.5, 1]`.
8. Steer an independent downstream behavioral task.
9. Run manipulation, compliance, and collateral-behavior checks.
10. Aggregate construct-level effects with uncertainty.

Probe prompts and downstream tasks must not be the same task in different
wording. Otherwise the experiment cannot distinguish construct transfer from
prompt or label matching.

Before interpreting additive steering as a control result, the causal pathway
extension uses a matched episode: positive and negative induction contexts are
each followed by the same downstream task, and one condition's residual state
is interchanged into the other at a tokenizer-verified induction/task boundary
during prefill only. This C1 residual-interchange test asks whether the state
is contextually causally sufficient for a matched continuation. It does not
establish necessity, a unique circuit, or a behavioral policy variable.

The causal-method sequence is:

```text
B behavioral validity → R representation profile → C1 residual interchange
→ C2 temporal tracing → C3 component/path tracing → C4 ablation → S steering
```

C2–C4 are follow-up methods. C1 is implemented as a separate
`MatchedEpisodeResidualPatcher` and must not be replaced by arbitrary
cross-prompt activation transplantation.

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
- the versioned 16-construct registry exists, with all 16 construct
  specifications and paired-vector generation plans marked `specified`; Wave
  2–4 downstream generation plans and review/full inventory workflow are
  implemented, while all generated downstream inventories remain
  non-confirmatory engineering artifacts gated from confirmatory execution;
- the benchmark-facing generation adapter emits canonical `PromptRecord`
  rows, supports deterministic mocks and no-API dry runs, and refuses to treat
  partial inventories as complete; named `review`/`full` generation modes and
  deterministic model-side `test`/`full` inventory selection are implemented;
- the prompt-overlap audit reports construct, split, family, role, template,
  response-format, and probe/downstream independence metadata;
- fixture-tested train-only directions, continuous held-out projection margins,
  neutral/within-condition calibration, strict Wave 1 output parsing, directed
  state-transfer scoring, deterministic steering controls, validation-only
  candidate-layer selection, bootstrap intervals, timing-aware injection,
  injection pre/post arithmetic, downstream-layer projection tracking,
  expected-versus-observed shift scoring, and resumable trace manifests are
  implemented;
- behavior, steering, and calibration prompt families are kept separate and
  task-category schedules are pre-registered in the generation plans;
- the vector-only all-registry orchestrator supports a four-worker review/full
  workflow, per-construct outputs, combined manifests, and `--resume`; the
  paired vector scope is 100/40/40 per construct;
- `scripts/audit_vector_pairs.py` is the QA audit entry point for structural,
  nuisance, and leakage review of generated vector pairs;
- a deterministic fake vertical slice exercises the control plane without an
  API, model weights, or a GPU;
- readout, steering-plan, environment-check, remote execution, and scoring CLIs
  now separate frozen scientific decisions from GPU/model execution;
- real-model prompt-only behavior composition, output-accessibility and
  collateral checks, real-model validation, real-run uncertainty orchestration,
  and correspondence analysis are not yet validated end to end; the
  manifest-backed prompt-only runner/scorer, fail-closed tokenizer preflight,
  and scalar injection/downstream manipulation tracking and scoring are
  implemented as model-side artifacts;
- the active vector iterator now lives in `activation_analysis.activation_store`
  and passes the active Python 3.11 `make check` suite; end-to-end validation
  against a real activation run remains pending;
- a shared activation plan can batch the four Wave 1 constructs without
  pooling directions and can later expand to all registry entries;
- the API-generated all-16 vector/probe inventory and a real-model realization
  decode pilot are available as engineering/reference artifacts; neither is a
  completed generalized benchmark or a steering result;
- matched-episode C1 residual interchange, bidirectional donor swaps,
  same-condition controls, tokenizer-verified boundary localization, and a
  fail-closed causal-output manifest validator are implemented and
  fixture-tested; this is causal infrastructure, not a real-model causal
  result;
- no model download or large generalized benchmark run has been launched.

## Next implementation gates

1. Review and audit the existing all-16 vector/probe inventory and its final
   manifest with `scripts/audit_vector_pairs.py`; regenerate only as an
   explicitly versioned replacement.
2. Add a manifest-backed activation-run smoke fixture when a representative
   benchmark run is available, while preserving the passing clean-install
   checks.
3. Validate continuous held-out projection margins and neutral/within-condition
   dose calibration on a representative manifest-backed activation run.
4. Run the local fake vertical slice and validate the prompt-only baseline,
   tokenization preflight, and zero-dose gate before starting a GPU.
5. Complete the one-construct realization measurement slice first, then the
   realization/evidence engineering pair while preserving source-reliability
   and persistence namespaces.
6. Validate the implemented intervention traces and downstream-persistence
   summaries on a representative model run; run the C1 matched-episode
   residual-interchange diagnosis on a small registered subset; then add
   output-accessibility, collateral-behavior, and prompt-only baseline checks.
7. Run the precision simulation before fitting representation-profile
   predictors or treating Waves 2–4 as confirmatory model work.

Until these gates are passed, documentation should describe the benchmark as
an implemented control plane and fixture-tested measurement core without a
validated end-to-end experimental run.
