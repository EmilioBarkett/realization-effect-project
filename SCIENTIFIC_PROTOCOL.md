# Current scientific protocol

**Status:** advanced protocol draft. This is the current experimental contract,
not evidence that the generalized benchmark has been implemented.

## 1. Research question and hypothesis

For each theory-relevant behavioral construct, we will separate:

1. prompt-condition behavioral sensitivity;
2. linear decodability of the associated internal state; and
3. causal steerability of an independent downstream behavior.

Primary hypothesis:

> Linear decodability is common across constructs, but decodability alone is an
> incomplete and often poor predictor of the magnitude and specificity of
> causal state transfer.

The first intervention is additive state transfer. We are not initially
claiming to decode a behavioral policy parameter or gain-control variable.
The longer-term benchmark may measure a representation profile—including
direction stability, context consistency, localization, and dimensionality—
but those additional features are not to be fit as a high-dimensional
predictor in the small engineering or Wave 1 pilot.

## 2. Scope and pilot constructs

The candidate bank is frozen in the versioned
[`construct_registry_v1.json`](configs/construct_benchmark/construct_registry_v1.json)
and is balanced across four families:

| Family | Wave 1 | Wave 2 | Wave 3 | Wave 4 |
|---|---|---|---|---|
| Decision | `realization_account_closure` | `reference_frame` | `ambiguity_orientation` | `temporal_orientation` |
| Epistemic | `evidence_diagnosticity` | `prior_weighting` | `causal_interpretation` | `epistemic_uncertainty` |
| Social | `source_reliability` | `authority_deference` | `consensus_conformity` | `reciprocity_obligation` |
| Agentic | `persistence_continuation` | `exploration_exploitation` | `plan_replanning` | `goal_shielding` |

The directional contrasts and independent tasks are specified in the registry's
construct files and are summarized in the project direction. The distinctions
must remain explicit: evidence diagnosticity is not updating responsiveness;
source reliability is not authority status; persistence is not replanning; and
epistemic uncertainty is not ambiguity orientation.

All 16 construct definitions and paired-vector generation plans now exist and
are marked `specified` in the registry. Wave 2–4 calibration-aware downstream
generation plans and the review/full inventory workflow are implemented. The
existing Wave 2–4 composed inventories have been audited and remain
non-confirmatory engineering artifacts: they contain prompt-wrapper,
downstream-episode, direct-cue, and task-independence blockers documented in
`agents/WAVES2_4_PROMPT_AUDIT.md`. Confirmatory model execution remains gated
on versioned prompt repairs, fresh audit, the Wave 1 measurement gates, and a
precision simulation. The frozen
vector-generation scope is 100 train, 40 validation, and 40 held-out pairs per
construct: 2,880 pairs and 5,760 records across the 16-construct bank. Vector
generation is Sonnet 4.6 only, with a four-worker orchestrator; downstream
prompt generation uses the separately pinned Luna review/full workflow. There
is no cross-construct pooling of directions. The completed Wave 2–4 downstream
engineering source prompt inventory is retained at
`results/benchmark/downstream_prompts_v1_waves2_4_full_luna_b20_o30k_v1/`;
it contains 384 non-confirmatory engineering records and is not a real-model
experiment result. No Wave 2–4 prompt inventory is currently released for
confirmatory execution.

The current Wave 1 repaired model input is
`results/benchmark/prompt_inventories/wave1_repaired_v2_full_openai_luna_normalized/combined.csv`.
Its manifest records 1,824 frozen engineering rows: 1,440 vector/probe rows
and 384 independent behavior, steering, and calibration rows. These are
frozen prompt-preparation artifacts, not real-model empirical results. The
older 1,650-row composition remains historical provenance and must not be
mixed with repaired v2.

The later Wave 1 release-v3 inventory and model-side pilots are likewise
engineering artifacts. Mistral and Qwen residual/readout/steering outputs and
the Qwen C1 diagnostic are stored on the RunPod persistent volume with
manifest-backed summaries; they do not release a confirmatory claim. The
model-side behavioral/accessibility preflight in
`agents/MODEL_BEHAVIOR_ACCESSIBILITY_PREFLIGHT.md` is now the required gate
before any future large execution, and must pass separately for every model
and construct on a frozen 8--16-item subset.

For the eventual expansion, Waves 2–4 are organized as three separate
four-construct runs rather than one twelve-construct execution. Each run
shares one activation pass across its wave and fans out by construct. The
execution package is prepared and its test selections are deterministic, but
full confirmatory mode remains gated on the Wave 1 measurement release and the
precision simulation; explicit prompt-input release is complete.

These are operational hypotheses. A construct enters a confirmatory analysis
only after its state definition, directional outcome, parsing rules, and
leakage controls are frozen.

Specification gaming in coding agents is outside the current protocol and is
preserved only as historical project material.

## 3. Design for one construct

Each construct requires four distinct components:

1. **Probe contrast:** matched paired prompts intended to induce two defined
   construct states.
2. **Readout:** a linear direction estimated from training prompts only.
3. **Independent behavioral task:** a new task whose output has a registered
   directional interpretation.
4. **Steering:** a frozen direction applied at a pre-specified intervention
   point with calibrated positive, zero, and negative doses.

The probe task and downstream task must be meaningfully independent. Prompt
overlap, lexical shortcuts, task identity, and response-format effects must be
audited.

## 4. Prompt and split policy

Every prompt row must have a stable ID, construct ID, pair ID, pair role, task or
prompt family, source, expected direction, and split assignment.

Required partitions:

- `direction_train`: used to estimate the main direction;
- `direction_validation`: used only for pre-registered engineering choices;
- `direction_heldout`: used for confirmatory readout;
- `behavior_eval`: independent downstream-task prompts;
- `steering_eval`: matched independent prompts used for causal intervention.
- `calibration`: neutral or within-cell prompts used only to set intervention
  scale.

No validation, held-out, or downstream behavioral prompt may enter direction
construction. Entire prompt families or task templates should be held out when
testing generalization, not only paraphrases.

Behavior prompts are generated and stored separately from probe prompts. The
behavior, steering, and calibration roles each use distinct prompt families,
and canonical validation rejects normalized prompt-text reuse across records
or across roles within a construct. A prompt-only behavior baseline is a
separate estimand from post-steering scoring; the probe-to-downstream episode
runner still needs real-model validation.

Generation plans pre-register categorical schedules before generation. Cells
balance their relevant task factors rather than asking a model to choose the
dataset composition. Generation transport retries are set to two where
configured, but each retry repeats the identical failed request; retries never
regenerate content or alter a prompt after a content-based failure.

The legacy generator in `activation_analysis` remains realization-focused and
is not the interface for new multi-construct data. The benchmark-facing
`construct_benchmark.generation` adapter now takes semantics from a construct
specification and generation plan, emits canonical prompt records, supports
deterministic mocks and no-API dry runs, and is used by the vector-only
orchestrator. The completed API-generated vector/probe inventory is tracked at
`results/benchmark/vector_prompts_v2_luna/full_final_all16/combined.csv` with a
final manifest. It is non-confirmatory and scope-partial: behavior,
calibration, and steering-task prompts remain outside that artifact.

Wave 1 generation plans also freeze task composition: paired probe context is
presented before the independent downstream task; only an induced internal
state may carry over; probe surface text, condition labels, and entities do
not carry over; and behavior and steering content pools remain separate. The
C1 matched-episode runner now enforces the shared downstream-task and fixed
boundary contract for causal diagnosis. Its real-model execution remains
unvalidated, so no external pilot may be treated as an end-to-end experiment.

## 5. Causal pathway profile

The first causal method is matched-episode residual interchange. A positive
and a negative induction context are each followed by the same downstream
task. The runner captures the residual state at the last complete token of the
induction prefix and replaces the receiver's state at that same semantic
boundary on the prompt-prefill forward pass only. It then allows the model to
complete the identical downstream task without further intervention.

For each registered layer, the causal diagnosis records positive-to-negative
and negative-to-positive swaps, plus same-condition donor controls. The
primary causal outcome is the pre-registered parsed downstream behavior; a
fixed teacher-forced logit contrast may be added as a secondary outcome. The
runner's tokenizer-offset check fails closed when the boundary cannot be
located exactly.

This estimates contextual causal sufficiency of a state at a defined point.
It does not establish necessity, a unique circuit, linear reproducibility,
cross-domain generalization, or a behavioral policy/gain variable. Arbitrary
cross-prompt residual transplantation is not a valid substitute because it
confounds scenario identity, entities, syntax, difficulty, and downstream
task text.

Temporal tracing, component/path patching, and ablation are later C2–C4
methods. The implementation and run contract are in
`agents/CAUSAL_PATHWAY_ARCHITECTURE.md`; the C1 output must have a complete
adjacent manifest before it can be scored as causal evidence.

## 6. Direction and decodability

The primary direction estimator is a train-only matched mean difference:

```text
direction = mean(positive_train) - mean(negative_train)
```

The exact sign is fixed by the construct specification before held-out results
are inspected. Candidate layers, activation site, token/region mode, and
position mode are registered in the run configuration. When multiple layers
are registered, the default rule selects the layer with the largest validation
mean standardized margin on `direction_validation`; held-out prompts are not
used for this choice. A fixed single-layer run remains available as an
explicit diagnostic.

The primary decodability estimand is the continuous standardized held-out
projection margin. Pairwise accuracy, calibration curves, direction stability,
and layer profiles are secondary diagnostics.

Projection at the injection layer is not a sufficient manipulation check. The
analysis should also test downstream-layer persistence, output accessibility,
and a same-task positive control where appropriate.

The steering runner now records scalar manipulation artifacts for every
condition and registered tracking layer. The injection-layer record contains
the pre-injection projection, post-injection projection, observed shift,
expected shift from the requested dose and frozen calibration scale, and their
difference. Later-layer records contain projections onto independently
constructed train-only directions when available. A same-vector later-layer
projection is retained only under an explicit diagnostic role and is not a
downstream construct-state readout.

Scoring is permitted for confirmatory analysis only when the adjacent steering
output manifest is marked complete, its expected condition-by-layer identities
are all present, and its provenance and raw-output hash agree with the JSONL.
The scorer has an explicit `--allow-incomplete-diagnostic` override for
engineering inspection; outputs produced under that override are not
confirmatory evidence.

## 7. Steering intervention

Steering doses are expressed in units of a frozen training calibration scale.
The calibration variance must come either from neutral calibration prompts or
from within-condition, within-cell centered activation variance. A variance
computed over a positive/negative mixture is not acceptable if it mechanically
couples construct separation to physical intervention strength. Residual-norm
ratios and unstandardized intervention magnitudes are recorded as safety
diagnostics alongside the normalized dose.

The intervention timing must use one of the canonical registered values:

- `prefill_only`;
- `generation_only`;
- `every_step`; or
- `fixed_window`.

The first intervention battery uses prefill-only injection and five registered
doses:

```text
-1, -0.5, 0, +0.5, +1
```

Continuous or generation-time injection is a secondary timing comparison. The
minimum control battery includes a zero dose, a negative dose, a shuffled-label
direction, and three reproducible random directions orthogonal to the target
where the hidden size permits it. Wrong-layer or unrelated-direction controls
should be included where feasible.

## 8. Behavioral estimands

### Primary: directed mean state transfer

For each construct, estimate whether steering changes the independent task
outcome in the pre-registered state-consistent direction. Report the mean
effect, uncertainty, compliance, and collateral outcomes. Use an outcome
adapter for standardization: binary outcomes use probability-scale marginal
effects; bounded scores use a fixed registered range or scale; continuous
outcomes use a baseline or externally defined reference standard deviation.
Always report the unstandardized effect beside the standardized result.

### Secondary: policy-slope change

If the downstream task contains a graded cue such as evidence strength,
difficulty, or payoff, estimate whether steering changes the response slope.
This is stronger evidence about policy responsiveness, but additive state
steering does not guarantee such an effect.

### Exploratory: policy-gain intervention

Conditional, multiplicative, Jacobian-derived, or responsiveness-trained
interventions are exploratory method development. They are not assumed to be
available from an ordinary mean-difference direction.

## 9. Controls and exclusion rules

Required controls include:

- shuffled labels or shuffled directions;
- zero and negative steering doses;
- held-out prompt families;
- format and compliance controls;
- unrelated behavioral outcomes;
- refusal, verbosity, and generic response-bias measures;
- downstream persistence or output-accessibility checks;
- manual review and judge-disagreement reporting where parsing is ambiguous.

Do not flip a direction sign, select a layer, remove a construct, or choose a
steering scale after inspecting confirmatory outcomes. Failed decodability and
failed steering remain results.

## 10. Cross-construct analysis

The inferential structure is crossed rather than nested:

```text
model × construct × task × prompt
```

The cross-construct analysis will estimate whether held-out decodability and
pre-registered competing representation features predict steering strength
while propagating uncertainty from direction construction and behavioral
measurement. Candidate competing features include output accessibility,
downstream persistence, context consistency, layer stability, and normalized
intervention cost. Model and task effects must not be treated as noise-free
observations of a construct. Full-profile prediction is a later out-of-sample
analysis, not a four-construct pilot claim.

A descriptive Representation–Steerability Gap may be plotted after a reference
distribution and normalization are frozen. It is not a primary estimand, since
the result depends on the component metrics and standardization. The 16-entry
bank is the current staged breadth plan; a 50–100-concept bank and
model-scale/checkpoint comparison are later extensions, not requirements for
Wave 1.

The final construct count will be selected using a precision simulation, not an
automatic target such as eight to twelve constructs.

## 11. Current maturity and implementation gates

The current status is protocol development and benchmark infrastructure
implementation:

- the realization behavioral pipeline is archived;
- the activation paired-prompt generator and core activation primitives are
  retained;
- the initial benchmark control plane exists with versioned construct, run,
  analysis, prompt, split, and provenance schemas;
- the 16-construct registry, all 16 specified construct definitions and paired
  generation plans, generic canonical-record generation adapter, and
  generalized leakage-audit metadata are implemented as prompt-preparation
  artifacts; Wave 2–4 downstream plans and review/full inventory generation
  are implemented, and the existing composed Wave 2–4 inventories are audited
  engineering artifacts with release blockers rather than confirmatory inputs;
- train-only direction estimation, projection-margin measurement,
  neutral/within-condition calibration, strict Wave 1 parsing, directed
  state-transfer scoring, control-direction generation, and timing-aware
  injection are implemented and fixture-tested;
- scalar injection pre/post tracking, registered downstream-layer projections,
  expected-versus-observed manipulation scoring, downstream persistence ratios,
  and manifest-backed resumable steering output are implemented and
  fixture-tested; this is instrumentation, not a real-model result;
- matched-episode C1 residual interchange, bidirectional donor swaps,
  same-condition controls, tokenizer-verified boundary localization, and a
  fail-closed causal-output manifest validator are implemented and
  fixture-tested; this is causal infrastructure, not a real-model causal
  result;
- prompt-role/family separation, pre-registered category schedules, validation
  layer selection, and pair/item bootstrap interval primitives are implemented
  and fixture-tested;
- the outcome-independent 8--16-item model-side behavioral/accessibility
  preflight selection and validator are implemented; real-model preflight
  release remains pending until each model/construct pair passes;
- a deterministic `scripts/run_fake_benchmark.py` exercises the vertical slice
  without APIs, model weights, or a GPU; its outputs are explicitly
  non-empirical;
- real-model confirmatory execution, real-run uncertainty reporting, all-16
  downstream parsers and behavior execution, and correspondence analysis
  remain unvalidated end to end; Wave 1 engineering pilots exposed model- and
  construct-specific parser/accessibility blockers, and the preflight must
  resolve those blockers before scaling;
- the tracked activation iterator now lives in
  `activation_analysis.activation_store` and passes the clean Python 3.11
  `make check` suite; real-run validation remains pending;
- a four-worker vector-only orchestrator can review or prepare the frozen
  100/40/40 paired-vector inventory per construct, then fan out construct-
  specific artifacts; this does not execute downstream behavior tasks;
- the completed API-generated all-16 vector/probe inventory and a realization
  real-model decode pilot are available as engineering/reference artifacts;
  neither is a completed generalized benchmark or a steering result;
- no evidence-diagnosticity behavior/steering run or large generalized
  benchmark run has been collected.

Before the first real measurement:

1. inspect and audit the completed all-16 vector/probe inventory and freeze its
   manifest and audit outputs as the current engineering artifact;
2. verify `ActivationVectorRecord` and `iter_activation_vectors()` against
   existing activation-store manifests;
3. keep the active activation tests green and add
   iterator/filtering/region/memory-map regression tests;
4. run the local fake vertical slice and validate the manifest-backed
   prompt-only baseline, tokenizer preflight, and zero-dose variation gate;
5. freeze and run the model-side behavioral/accessibility preflight on 8--16
   real items per model/construct pair; hold any pair that fails;
6. validate the implemented projection, calibration, parsing, steering trace,
   downstream-persistence, and C1 residual-interchange adapters on a
   representative model run, then add output-accessibility and collateral
   checks.

Then implement in this order:

1. run the implemented train-only readout on the frozen vector inventories for
   the Wave 1 measurement gate;
2. validate held-out readout, neutral/within-condition calibration, and
   outcome-specific effect adapters on a representative model;
3. execute the implemented timing, parsing, injection-trace, downstream-
   persistence, and C1 residual-interchange paths, then add output-
   accessibility and collateral checks;
4. precision simulation and expansion decision;
5. second model family before general conclusions;
6. Waves 2–4 only after the Wave 1 measurement and construct gates pass and
   the precision simulation supports expansion.

## 12. Claims this protocol does not support by itself

The following claims require additional evidence and must not be implied by a
successful probe:

- the model has a human-like psychological mechanism;
- the direction is a general behavioral policy variable;
- the direction will steer unrelated tasks;
- a mean state shift implies a policy-slope or gain change;
- a null steering result proves the representation is not causally relevant.
