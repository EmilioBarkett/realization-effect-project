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
predictor in the four-construct pilot.

## 2. Scope and pilot constructs

The initial construct families are:

| Construct | Transferred state | Probe contrast | Independent task |
|---|---|---|---|
| Realization/account closure | Open/paper versus closed/realized account | Matched realization framing | Risk or wager choice |
| Evidence diagnosticity | Perceived reliable/diagnostic versus weak/unreliable evidence | Matched evidence-quality framing | Confidence or belief revision |
| Source reliability/authority | Deference versus independent verification | Matched source/evidence framing | Follow source versus check evidence |
| Persistence/continuation | Continue versus abandon/reallocate effort | Matched goal-progress framing | Continue, quit, revise, or reallocate |

These are operational hypotheses. A construct enters the confirmatory set only
after its state definition, directional outcome, parsing rules, and leakage
controls are frozen.

The first vertical slice is realization plus evidence diagnosticity. Authority
and persistence are subsequent pilot candidates, not guaranteed confirmatory
constructs.

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

No validation, held-out, or downstream behavioral prompt may enter direction
construction. Entire prompt families or task templates should be held out when
testing generalization, not only paraphrases.

The active paired-prompt generator currently produces realization-focused
outputs. It will be generalized before new construct data are generated.

## 5. Direction and decodability

The primary direction estimator is a train-only matched mean difference:

```text
direction = mean(positive_train) - mean(negative_train)
```

The exact sign is fixed by the construct specification before held-out results
are inspected. Candidate layers, activation site, token/region mode, and
position mode are registered in the run configuration.

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

The intervention timing must be explicit:

- prompt-prefill only;
- generation-only;
- every decoding step; or
- a fixed position or window.

The minimum intervention battery includes positive, zero, and negative doses,
plus a shuffled-label or random-direction control. Wrong-layer or unrelated
direction controls should be included where feasible.

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
the result depends on the component metrics and standardization. A broader
50–100-concept bank and model-scale/checkpoint comparison are later extensions,
not requirements for the first vertical slice.

The final construct count will be selected using a precision simulation, not an
automatic target such as eight to twelve constructs.

## 10. Current maturity and implementation gates

The current status is protocol development and repository preparation:

- the realization behavioral pipeline is archived;
- the activation paired-prompt generator and core activation primitives are
  retained;
- the generic benchmark package does not exist;
- the three configuration schemas do not exist;
- projection-margin, neutral/within-cell calibration, state-transfer adapters,
  and manipulation-check orchestration do not exist;
- the tracked activation iterator now lives in
  `activation_analysis.activation_store` and passes the clean Python 3.11
  `make check` suite; real-run validation remains pending;
- no evidence-diagnosticity data have been collected.

Before schema implementation:

1. stabilize the repository boundary and record the completed clean-install
   check plus the remaining optional-dependency limitation;
2. verify `ActivationVectorRecord` and `iter_activation_vectors()` against
   existing activation-store manifests;
3. make the active activation tests collect in a clean environment and add
   iterator/filtering/region/memory-map regression tests;
4. add an explicit `.gitignore` rule for benchmark raw artifacts;
5. define the three versioned artifacts:
   `construct_spec`, `run_config`, and `analysis_spec`.

Then implement in this order:

1. generic prompt/split schema and leakage audit;
2. train-only realization/evidence-diagnosticity readout vertical slice;
3. continuous held-out readout, neutral/within-cell dose calibration, and
   outcome-specific effect adapters;
4. explicit intervention timing, independent-task parsing, output-accessibility
   and downstream-persistence checks;
5. precision simulation and expansion decision;
6. second model family before general conclusions.

## 11. Claims this protocol does not support by itself

The following claims require additional evidence and must not be implied by a
successful probe:

- the model has a human-like psychological mechanism;
- the direction is a general behavioral policy variable;
- the direction will steer unrelated tasks;
- a mean state shift implies a policy-slope or gain change;
- a null steering result proves the representation is not causally relevant.
