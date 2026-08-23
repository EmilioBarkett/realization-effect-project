# Scientific Protocol: Cross-Task Causal Transfer of Behavioral Representations

**Status:** Draft v0.2 — proposal for review before implementation or
confirmatory data collection.

## Executive decision

Proceed with a two-construct state-transfer vertical slice first, followed by a
four-construct pilot. Do not describe the four-construct study as a
confirmatory benchmark until this scientific contract is frozen and a
power/precision simulation has been completed.

The project should not claim to be the first work to show that probing does not
establish causal use. That point is established by earlier probing and
intervention work, and recent papers now directly study when steering vectors
succeed or fail.

The defensible initial contribution is narrower:

> A preregistered, theory-backed benchmark of whether a construct-state
> direction learned from one task transfers causally to a predicted behavioral
> shift in an independent task.

The first implementation will use additive state-transfer steering, for which a
directed mean behavioral change is the primary outcome. A change in policy slope
will be a pre-specified secondary outcome. A later policy-gain phase may explore
conditional, multiplicative, Jacobian-derived, or independently induced
high-versus-low responsiveness directions; that phase is method development,
not an automatic consequence of a factorial contrast.

## 1. Research question and claims

### Primary question

When a language model contains a linearly decodable, theory-relevant construct
state, does steering that state produce a predicted behavioral shift in a new
task and domain?

### Primary hypothesis

Across constructs and models, held-out linear decodability alone will have weak
predictive value for cross-task state transfer. Policy-slope changes may occur
because the transformer is nonlinear, but the current state direction gives no
special reason to expect them and they are not the primary claim.

This is a falsifiable hypothesis, not an assumption. If decodability predicts
transfer strongly, that is still useful: it would identify a boundary condition
under which the existing realization-effect conclusion does not generalize.

### Claims we will not make

- A probe score is not evidence that the model uses the representation.
- A steering effect on one output format is not evidence of a general trait.
- A null correlation with four constructs is not evidence for a population-level
  law.
- A failed linear intervention does not prove that no nonlinear intervention
  could work.
- A construct will not be included or removed based on observed decodability or
  steering success.

## 2. Novelty positioning

The general separation between “information is decodable” and “information is
causally used” is established. The proposal must therefore earn novelty from
the experimental unit and the strength of the transfer test.

### Relevant prior directions

- [Amnesic Probing](https://arxiv.org/abs/2006.00995) showed that conventional
  probing performance need not track task importance and advocated causal
  interventions on representations.
- [Predicting Where Steering Vectors Succeed](https://arxiv.org/abs/2604.15557)
  introduced a Linear Accessibility Profile and reports strong prediction of
  steering effectiveness for controlled binary concept families, largely using
  same-concept next-token tasks and output accessibility.
- [Curveball Steering](https://arxiv.org/abs/2603.09313) studies geometric
  distortion and proposes a nonlinear steering method; it is primarily a
  method for improving intervention reliability, not a behavioral-policy
  transfer benchmark.
- [Encoded but Not Actionable](https://arxiv.org/abs/2608.17843) explicitly
  audits decoding, generation, activation influence, and steerability in a
  structured geometric reasoning setting.

### Proposed contribution

The contribution should be framed as the combination of four elements:

1. **Theory-backed construct states:** constructs are defined by a specific
   state to be transferred, not only by a binary prompt label or an asserted
   behavioral policy parameter.
2. **Cross-task causal transfer:** the state direction is learned from probe prompts
   and tested on new task templates, domains, and outcome formats.
3. **Directed behavioral outcomes:** the primary causal outcome is a
   pre-specified state-consistent mean shift, with policy-slope change retained
   as stronger secondary evidence.
4. **Competing-predictor analysis:** decodability is compared with output
   alignment, stability, geometric distortion, and perturbation sensitivity.

The novelty is not “we discovered that decodability and steering can diverge.”
It is “we measure the conditions under which a behaviorally defined
representation transfers from readout to causal policy control.”

## 3. Common ontology: theory-relevant construct states

The original four-construct proposal mixed external states, evidence polarity,
information sources, and actions. That makes a cross-construct correlation
hard to interpret.

The revised common object for the initial study is a **theory-relevant
construct state**.

For construct `c`, define:

- `z_c`: the state being transferred, such as account closure or perceived
  evidence diagnosticity;
- `Y_c`: the structured downstream behavioral outcome;
- `m_c`: the pre-specified directional mean outcome expected under `z_c`;
- `β_c`: an optional policy-slope parameter used as a secondary outcome;
- `u`: a dimensionless steering dose;
- `m_c(u)` and `β_c(u)`: the mean outcome and policy coefficient under dose `u`.

The direction is not interpreted as “the model has the word or concept,” and it
is not initially interpreted as a decoded policy-gain parameter. It is a
candidate internal state that should produce a directional outcome in a new
task.

The primary causal quantity is:

```text
M_c = change in m_c per unit normalized steering dose
```

The secondary quantity is:

```text
G_c = change in β_c per unit normalized steering dose
```

The stronger policy-gain interpretation is reserved for a later method phase
that explicitly constructs directions for responsiveness.

## 4. Pilot constructs and operational definitions

The names below are provisional. The exact task templates and coefficients must
be finalized before prompt generation.

| Construct | Transferred state | Probe contrast | Independent behavioral test | Primary state-transfer outcome |
|---|---|---|---|---|
| Realization effect | Account-closure state | Paper/open versus realized/closed account, crossed with valence and amount | New gamble choices across account status and amounts | Directional risk shift under a standardized downstream gamble |
| Evidence diagnosticity | Perceived reliability/diagnosticity state | Reliable versus unreliable evidence, balanced across polarity and topic | Confidence or magnitude of belief revision on new ambiguous cases | Directional confidence/revision shift under standardized evidence |
| Authority deference | Source-reliability state | Source expertise crossed with evidence strength and conflict | Follow-source versus follow-evidence choice on new domains | Directional source-following shift under controlled conflict |
| Persistence | Continuation/abandonment state | Setback and continuation contexts crossed with value and sunk cost | Continue/quit/allocate effort decisions on new tasks | Directional continuation shift at reference value/cost |

### Construct correction

“Evidence supports versus contradicts” is not sufficient and should not be
called an updating direction. The first epistemic state should instead be
**perceived evidence diagnosticity/reliability**, induced through reliability
contrasts balanced across evidence polarity and topic. Updating responsiveness
can be measured as a secondary slope outcome, but it is not what the initial
direction claims to represent.

Likewise:

- authority must separate warranted reliance from blind deference;
- persistence must separate rational quitting from abandonment;
- realization must separate account status from amount and valence.

If a construct cannot be represented by a clear state definition with a
directional, factorial behavioral task, it should not enter the confirmatory
set. A stronger policy-gain interpretation requires a separate direction
estimator and should be labeled exploratory method development.

## 5. Causal structure of one construct

Each construct should follow this logic:

```text
theory-backed probe contrast
          |
          v
construct state / activation direction
          |
          +--> held-out linear readout
          |
          +--> injection into a new task
                         |
                         v
              directed mean behavioral shift
```

The probe and downstream tasks are related by the same theoretical construct
state, but they must differ in surface form and preferably in domain, template
source, and output format. A policy-slope change is a stronger secondary test,
not the default interpretation of an additive state direction.

## 6. Three versioned scientific artifacts

Do not combine theory, model settings, and statistical decisions into one JSON
file. Use three separate artifacts.

### 6.1 `construct_spec`

Stable scientific definition:

```text
construct_id and version
family
theory and motivating literature
state_definition
probe contrast definition
factorial nuisance factors
independent task definition
primary directional outcome and secondary slope outcome
expected activation sign
inclusion criteria
invalidity criteria
```

This file should not change across models or runs.

### 6.2 `run_config`

Model- and run-specific settings:

```text
model ID and revision
activation site
candidate layers
token/region mode
position mode
direction estimator
normalization rule
steering dose grid
random seeds
generation settings
hardware/runtime settings
```

### 6.3 `analysis_spec`

Frozen analysis decisions:

```text
primary decodability estimand
primary state-transfer estimand
secondary diagnostics
outcomes and exclusions
uncertainty method
cross-construct statistical model
equivalence bounds
multiple-comparison policy
```

The three artifacts should be hashed and recorded in every run manifest.

## 7. Prompt and data design

Every prompt receives stable metadata:

```text
prompt_id
pair_id
construct_id
condition or factorial cell
state target
domain
source_template
task_id
split
seed
prompt_hash
```

### 7.1 Probe set

The probe set contains paired or factorial prompts from which the direction is
constructed. It should contain multiple domains and templates so the state
direction cannot be a single-topic or lexical feature.

### 7.2 Behavioral set

The behavioral set contains new task templates with structured outputs. It must
include the cue required to estimate the primary state-consistent outcome, but
it must not reuse the probe
wording, examples, labels, or answer format.

### 7.3 Split policy

Use the following logical splits:

```text
direction_train       construct the direction
direction_validation  select pre-registered technical settings
probe_heldout         final decodability estimate
behavior_baseline     unsteered independent-task baseline
behavior_steering     independent-task intervention evaluation
calibration           estimate activation scale from training data only
```

Splits should be grouped by prompt family, domain, or source template rather
than randomly splitting near-identical paraphrases. The confirmatory behavioral
set must not be used to choose the direction, layer, sign, position mode, or
steering scale.

### 7.4 Factorial controls

Each construct should use enough factorial variation to separate the intended
state-consistent outcome from nuisance effects.

#### Realization

Cross account status with outcome valence and amount. Report the pre-specified
directional risk shift rather than a raw realized-versus-paper mean difference.

#### Evidence updating

Cross prior odds, evidence polarity, and likelihood-ratio magnitude. Estimate
revision as a function of the known likelihood ratio, including symmetric
positive and negative evidence.

#### Authority deference

Cross source expertise, evidence quality, and agreement/conflict. The source
recommendation should be content-matched across source conditions.

#### Persistence

Cross setback severity, expected continuation value, and sunk cost. Include
cases where continuing and quitting are each rational under different future
values.

## 8. Direction and readout contract

### 8.1 Primary direction estimator

The primary estimator should remain close to the current implementation:

1. mean-pool the registered activation region for each prompt;
2. compute paired differences `h_positive - h_negative` on the training split;
3. average the paired differences into `d_train`;
4. normalize only when used for steering, retaining the raw norm in metadata.

This makes the main readout axis identical to the main steering axis.

The existing implementation is in
[`src/activation_analysis/vector_analysis.py`](../../../src/activation_analysis/vector_analysis.py).
Its `build_pair_directions` function is the starting point, but the new
pipeline must make the positive/negative roles construct-configurable.

### 8.2 Primary decodability estimand

The single primary decodability score is:

```text
D_c = standardized held-out paired projection margin
```

For each held-out pair, project the positive-minus-negative activation
difference onto the frozen unit direction. Standardize the mean signed margin
using a scale estimated from the direction-training split only. `D_c` is a
continuous held-out separation score with uncertainty clustered by pair and
prompt family.

The exact standardization rule must be frozen in `analysis_spec`; it must not
use the confirmatory held-out distribution. The score should retain the sign of
the expected state contrast.

Secondary readout diagnostics may include:

- paired projection accuracy;
- AUC of individual prompt projections;
- cross-domain and cross-template accuracy;
- bootstrap cosine stability of `d_train`;
- a trained logistic probe, clearly labeled as a different estimator;
- model-native output accessibility such as a logit-lens score where the task
  has an unambiguous output-token mapping.

The trained probe must not replace `D_c` as the primary score because it may
find a different direction than the one used for steering.

### 8.3 Layer selection

For the realization reproduction pilot, use the existing target layer as a
fixed replication setting. For the confirmatory benchmark, define a candidate
layer set in `run_config`.

Layer selection must use only `direction_train` and
`direction_validation`, with a rule frozen in `analysis_spec`. The selected
layer must be carried unchanged into `probe_heldout` and `behavior_steering`.

If the goal is to test decodability as a predictor, report the full layerwise
profile as a secondary analysis rather than silently selecting the layer with
the largest steering effect.

### 8.4 Direction stability

On the training split, bootstrap pairs and recompute the direction. Record:

```text
mean cosine(d_bootstrap, d_train)
lower-tail cosine stability
between-domain cosine stability
between-template cosine stability
```

This is a competing predictor of transfer, not part of the primary decodability
score.

## 9. Steering contract

### 9.1 Dimensionless dose calibration

Raw scales such as `+100` are not comparable across constructs, layers, or
models. For each target layer, estimate a calibration statistic from training
prompts only:

```text
s_train = SD(<h, d_unit>) over the registered calibration prompts
```

Let `d_unit = d_train / ||d_train||`. Define the injected perturbation as:

```text
Δh = u × s_train × d_unit
```

where `u` is a dimensionless dose. The confirmatory dose grid must be frozen
after a calibration-only procedure, before reading held-out behavioral results.
The raw perturbation norm, projection displacement, and ratio to residual-stream
RMS must all be recorded. Residual-stream RMS is a safety diagnostic, not the
primary cross-construct dose unit.

### 9.2 Position mode

Position mode and intervention timing are part of the intervention, not
implementation details. The run contract must distinguish:

- prompt-prefill intervention;
- generation-only intervention;
- every-step intervention during decoding;
- a registered fixed-position intervention.

The pilot may use the existing `last` behavior as its primary replication
condition, but the manifest must state exactly which positions and decoding
steps are modified. `all` or a registered decision-token region can be a
secondary robustness condition.

The current hooks support both modes in
[`src/activation_analysis/steering.py`](../../../src/activation_analysis/steering.py).
The new run manifest must record the mode explicitly.

### 9.3 Required intervention conditions

Use:

```text
negative dose(s)
zero dose baseline
positive dose(s)
label-shuffled direction
multiple norm- and variance-matched random directions
```

Use at least 32 random directions in the confirmatory control, or justify a
different number with a simulation. A single random vector is not a sufficient
null distribution.

### 9.4 Collateral and compliance outcomes

For each dose, record:

- parseable behavioral outcomes;
- output length and format compliance;
- refusal or truncation rates;
- unrelated-prompt logit/output changes where feasible;
- unrelated behavioral outcomes that could reveal a generic response bias;
- task success independent of the target construct state.

A steering effect that only changes formatting, verbosity, refusal, or output
length is not counted as successful policy steering.

### 9.5 Downstream manipulation checks

Projection change at the injection layer is not a useful manipulation check: it
is mechanically implied by adding the vector. The useful checks are whether the
shift persists into later layers, changes a registered output-accessibility
measure, or succeeds on a same-task positive-control classification. If these
checks fail, an independent-task null is interpreted as an ineffective
intervention rather than evidence against state transfer.

## 10. Behavioral estimands

### 10.1 Primary state-transfer outcome

Fit a pre-specified outcome model for each independent task. The primary
outcome is the directed mean shift predicted by the transferred construct
state. It must be defined before steering data are collected and should use
neutral or ambiguous downstream prompts when that makes the state implication
clearer.

Examples:

```text
realization: risk_outcome ~ dose + amount + valence + controls
evidence:    confidence ~ dose + prior_odds + evidence_polarity + controls
authority:   follow_choice ~ dose + evidence_quality + conflict + controls
persistence: continue_choice ~ dose + future_value + setback + sunk_cost + controls
```

The exact link function may differ by outcome type. The primary state-transfer
estimand is:

```text
m_c(u) = mean behavioral outcome under normalized dose u
M_c = slope of m_c(u) with respect to u
```

The sign of `M_c` must be pre-specified from the construct-state theory. A mean
shift counts only if it is directional, dose-consistent, and distinguishable
from matched random, shuffled, and unrelated-outcome controls.

### 10.2 Secondary policy-slope outcome

Policy-slope change remains a pre-specified secondary outcome. For tasks with a
theoretical cue `x_c`, estimate:

```text
β_c(u) = policy coefficient under normalized dose u
G_c = slope of β_c(u) with respect to u
```

Operationally, the downstream model includes a `dose × theoretical_cue`
interaction. The coefficient of that interaction is `G_c`.

An observed `G_c` may arise from nonlinear downstream computation even when the
intervention is additive. However, the current state direction does not
specifically target gain control. Therefore `G_c` is stronger evidence than
`M_c`, but it is not the primary claim of the initial vertical slice.

Other secondary outcomes include dose-response curvature, sign reversal, and
compliance-adjusted effects.

### 10.3 Cross-task state-transfer estimand

Each construct should have at least two independent downstream task templates
in the confirmatory benchmark. Define:

```text
T_c = mean standardized M_c across independent tasks
```

The primary transfer result is based on `T_c`, while task-specific estimates and
secondary `G_c` estimates are reported separately. A direction that only
affects the task family used to construct it is not considered broadly
transferable.

## 11. Cross-construct statistical analysis

### 11.1 Unit of inference

The construct is the main unit of scientific generalization. Prompts are
repeated observations nested within task; models are crossed with constructs
and tasks. The design is therefore:

```text
construct
  └── task
        └── prompt

model × construct × task
```

Prompts are not independent evidence for a construct-level correlation.

The four-construct pilot is for feasibility and variance estimation. It cannot
establish the confirmatory cross-construct hypothesis.

### 11.2 Confirmatory sample size

Do not use “8–12 constructs” as an automatic target. Before confirmatory data
collection, simulate the full hierarchical design and choose the number of
constructs, tasks, models, and prompts needed to achieve a pre-specified
precision or power target.

The confirmatory construct list must be fixed using theory and feasibility
criteria before activation readout or steering outcomes are inspected. If a
construct fails a pre-specified parseability or factorial-validity criterion,
retain it as a failed benchmark cell; use only a previously registered reserve
construct as a replacement.

### 11.3 Primary prediction model

The primary model is a cross-classified model over construct, task, model, and
pre-registered layer/setting:

```text
M[c,m,k] = γ0
         + γD * standardized(D[c,m])
         + γA * standardized(A_lin[c,m])
         + γR * standardized(Stability[c,m])
         + γG * standardized(Geometry[c,m])
         + model_effect[m]
         + construct_effect[c]
         + task_within_construct_effect[k:c]
         + model_construct_effect[m:c]
         + model_task_effect[m:k:c]
         + error[c,m,k]
```

`A_lin`, geometry, and perturbation measures are secondary predictors and may
be omitted when unavailable for a task. The decodability-only model is the
primary test of the original claim; the augmented model asks what explains
transfer when decodability is insufficient.

Uncertainty must be propagated through direction construction, state-transfer
estimation, and steering. Measurement error in both `D` and `M` must be
represented, rather than treating either as known. Use a cross-classified
bootstrap clustered at construct and model, or a Bayesian model with an
explicitly reported prior and posterior sensitivity analysis.

### 11.4 What counts as “poor prediction”

Failure to reach statistical significance is not enough. Before confirmatory
data collection:

1. run a simulation using pilot-derived variance estimates and independently
   chosen plausible effect sizes;
2. freeze an equivalence bound for the standardized `γD` slope predicting `M`;
3. freeze a practical out-of-construct prediction threshold.

Draft defaults for simulation, not final values, are:

```text
poor standardized slope: |γD| < 0.25
poor incremental cross-validated R²: ΔR² < 0.05
```

The final bounds must be chosen before the confirmatory benchmark. A result
inside the equivalence bound supports “decodability alone is practically weak
as a predictor”; a result outside it supports a relationship; everything else
is inconclusive.

## 12. Interpretation matrix

The benchmark should distinguish at least these outcomes:

| Readout | Independent behavior | Steering | Interpretation |
|---|---|---|---|
| Low | Any | No effect | No evidence of a generalizable linear state direction |
| Low | Any | Mean shift | Possible task-local, unstable, or steerable-without-generalizable-readout effect; investigate controls |
| High | No baseline state-consistent effect | No effect | Decodable probe state may not transfer to the independent task |
| High | Baseline state-consistent effect | No effect | State is readable and behaviorally relevant, but not causally accessible under this intervention |
| High | Baseline state-consistent effect | Mean shift only | Evidence for state transfer, subject to generic response-bias controls |
| High | Baseline state-consistent effect | Policy-slope change | Stronger evidence that the state affects downstream cue sensitivity |
| High | High same-task effect | Low independent-task effect | Evidence for task-local actionability rather than general state transfer |

## 13. Inclusion, exclusion, and stopping rules

### Include a construct only if, before confirmatory results:

- its theory specifies a concrete construct state;
- its probe contrast can be factorially separated from nuisance cues;
- its independent task has a structured, directional outcome;
- its parsing rules are validated on a pre-run sample;
- its probe and behavior prompts can be separated by template/domain;
- its primary dose, intervention timing, and layer/position rules are specified;
- any secondary policy-slope outcome is clearly separated from the state claim.

### Retain as a failure, not silently drop, if:

- decodability is at chance;
- the behavioral baseline is null;
- steering fails;
- the state-consistent outcome is unstable;
- compliance is poor under intervention.

### Do not stop or expand based on:

- seeing a promising probe score;
- seeing a successful steering scale;
- seeing a null result;
- selecting only constructs that “work.”

## 14. Implementation gates

Before any large run:

1. Add explicit ignores for `results/benchmark/raw/` and other large benchmark
   outputs. The current `.gitignore` does not yet protect the proposed
   `results/benchmark/.../raw/` path.
2. Implement the three-artifact schema and validation without API/model calls.
3. Implement construct-state outcome adapters; do not assume arbitrary tasks
   are safely “config only.”
4. Implement train-only projection-SD calibration and manifest recording.
5. Add downstream-layer, output-accessibility, and same-task manipulation
   checks.
6. Add multiple random-direction controls and overlap/leakage tests.
7. Re-run the realization vertical slice using the generic path.
8. Run a small evidence-diagnosticity vertical slice.
9. Run the power/precision simulation before labeling the four-construct set
   confirmatory.

## 15. Staged study plan

### Phase 0: Protocol freeze

- choose evidence diagnosticity/reliability as the first epistemic construct;
- finalize the four `construct_spec` files;
- finalize state definitions, directional mean outcomes, and factorial cells;
- finalize `analysis_spec`, equivalence bounds, and stopping rules.

### Phase 1: Engineering validation

- implement schema validators and deterministic fixtures;
- add `.gitignore` protection;
- adapt the current realization direction/readout/steering code;
- verify that the same vector is used for primary readout and steering.

### Phase 2: Two-construct vertical slice

- run realization and evidence diagnosticity end to end;
- verify cross-template prompt separation;
- verify state-transfer mean estimation, secondary policy-slope estimation, and
  calibrated doses;
- fix engineering problems before adding social or agentic tasks.

### Phase 3: Four-construct pilot

- run all four constructs under the same run contract;
- estimate variance, compliance, direction stability, and task transfer;
- do not make the confirmatory population-level claim.

### Phase 4: Confirmatory benchmark

- select the final construct/task set using pre-registered feasibility rules;
- run the simulated sample size and freeze the analysis;
- collect the full benchmark across the selected models and tasks;
- report both decodability-only and augmented predictor models.

## 16. Expected contribution under different outcomes

The project should be publishable only if it produces a useful answer beyond a
generic null result.

- If decodability is weakly related to state-transfer mean effects, the
  contribution is a controlled cross-task actionability gap with
  behavioral-science ground truth.
- If decodability predicts state transfer, the contribution is a boundary-
  condition result showing that prediction holds for independent construct
  states, not only synthetic or same-task concepts.
- If direction stability or output alignment outperforms decodability, the
  contribution is a practical diagnostic for deciding when a decoded behavior
  is worth intervening on.
- If steering changes policy slopes but not mean outcomes, the result is an
  exploratory indication of interaction-specific control; it is not the
  primary success criterion for the initial state-transfer design.
- If effects are task-local, the contribution is a measured transfer gap and a
  warning against interpreting same-task steering as a general behavioral
  trait.

## 17. Required records for every run

Save a manifest containing:

```text
construct_spec path and hash
run_config path and hash
analysis_spec path and hash
git commit
model ID and revision
prompt and split manifest hashes
random seeds
activation site, layer, region, and position mode
direction estimator and norm
calibration statistic and dose mapping
steering scales and controls
sample counts and exclusions
parser/compliance rates
primary and secondary metrics
command line and timestamps
software environment summary
```

Large raw generations, model weights, and activation tensors remain local and
ignored. Configs, small prompt manifests, audit outputs, and curated summaries
should be tracked.

## 18. Open decisions before implementation

These are the remaining scientific decisions, not coding details:

1. Confirm evidence diagnosticity/reliability as the first epistemic construct.
2. Choose exact independent task templates for all four pilot constructs.
3. Choose the primary directional mean outcome for persistence.
4. Decide the initial model set and fixed pilot layer.
5. Choose the calibration-only projection-SD dose grid procedure.
6. Define prompt-prefill versus generation-step intervention timing.
7. Run the precision simulation and freeze the equivalence thresholds.

Until these decisions are resolved, the project should remain in protocol and
schema design rather than moving to large-scale prompt generation or model
inference.
