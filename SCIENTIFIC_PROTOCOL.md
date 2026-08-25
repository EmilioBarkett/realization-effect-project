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

Wave 1 is the immediate implementation target. Its four construct definitions
and generation plans are specified, while Waves 2–4 remain planned registry
entries. Two constructs are the engineering validation slice, four constructs
form the descriptive Wave 1 pilot, and the full 16-construct matrix is reserved
for later out-of-sample representation-profile prediction.

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

Generation plans pre-register categorical schedules before generation. Wave 1
cells balance their relevant task factors (for example gain/loss/neutral,
supporting/contradicting evidence, or setback severity) rather than asking a
model to choose the dataset composition. Automatic retries are disabled in the
reviewed Wave 1 plans: a poor pilot is revised at the plan level and regenerated
only after review.

The legacy generator in `activation_analysis` remains realization-focused and
is not the interface for new multi-construct data. The benchmark-facing
`construct_benchmark.generation` adapter now takes semantics from a construct
specification and generation plan, emits canonical prompt records, supports
deterministic mocks and no-API dry runs, and is the path for Wave 1 generation.

Wave 1 generation plans also freeze task composition: paired probe context is
presented before the independent downstream task; only an induced internal
state may carry over; probe surface text, condition labels, and entities do
not carry over; and behavior and steering content pools remain separate. The
activation-state orchestration needed to enforce that sequence is not yet
implemented, so no external pilot may be treated as an end-to-end experiment.

## 5. Direction and decodability

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

## 6. Steering intervention

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

## 7. Behavioral estimands

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

## 8. Controls and exclusion rules

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

## 9. Cross-construct analysis

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

## 10. Current maturity and implementation gates

The current status is protocol development and benchmark infrastructure
implementation:

- the realization behavioral pipeline is archived;
- the activation paired-prompt generator and core activation primitives are
  retained;
- the initial benchmark control plane exists with versioned construct, run,
  analysis, prompt, split, and provenance schemas;
- the 16-construct registry, four specified Wave 1 construct definitions,
  four generation plans, generic canonical-record generation adapter, and
  generalized leakage-audit metadata are implemented;
- train-only direction estimation, projection-margin measurement,
  neutral/within-condition calibration, strict Wave 1 parsing, directed
  state-transfer scoring, control-direction generation, and timing-aware
  injection are implemented and fixture-tested;
- prompt-role/family separation, pre-registered category schedules, validation
  layer selection, and pair/item bootstrap interval primitives are implemented
  and fixture-tested;
- a deterministic `scripts/run_fake_benchmark.py` exercises the vertical slice
  without APIs, model weights, or a GPU; its outputs are explicitly
  non-empirical;
- real-model execution, prompt-only behavior composition, real-run uncertainty
  reporting, downstream manipulation checks, and correspondence analysis
  remain unvalidated or unimplemented end to end;
- the tracked activation iterator now lives in
  `activation_analysis.activation_store` and passes the clean Python 3.11
  `make check` suite; real-run validation remains pending;
- a single run plan can batch the four Wave 1 inventories through one activation
  pass and then fan out construct-specific analyses;
- no API-generated prompt dataset, evidence-diagnosticity data, or large
  benchmark run has been collected; only no-API dry-run and fake-fixture
  artifacts exist.

Before the first real measurement:

1. run and review the deterministic fake vertical slice;
2. review the four Wave 1 generation plans and, after explicit approval,
   connect generated inventories to the canonical combined inventory;
3. verify `ActivationVectorRecord` and `iter_activation_vectors()` against
   existing activation-store manifests;
4. keep the active activation tests green and add
   iterator/filtering/region/memory-map regression tests;
5. validate the implemented projection, calibration, parsing, and steering
   adapters, then add manipulation checks and downstream persistence.

Then implement in this order:

1. run the implemented train-only readout on the frozen Wave 1 inventories;
2. validate held-out readout, neutral/within-condition calibration, and
   outcome-specific effect adapters on a representative model;
3. execute the implemented timing and parsing paths, then add
   output-accessibility and downstream-persistence checks;
4. precision simulation and expansion decision;
5. second model family before general conclusions;
6. Waves 2–4 only after the Wave 1 measurement and construct gates pass.

## 11. Claims this protocol does not support by itself

The following claims require additional evidence and must not be implied by a
successful probe:

- the model has a human-like psychological mechanism;
- the direction is a general behavioral policy variable;
- the direction will steer unrelated tasks;
- a mean state shift implies a policy-slope or gain change;
- a null steering result proves the representation is not causally relevant.
