# Proposed Project Architecture

## Status and scope

This document describes the proposed architecture for the next phase of the
project. It is a target design, not a claim that all components have already
been implemented.

The archived realization-effect pipeline remains the reference case. The new
system should generalize its experimental logic across multiple behavioral
constructs while preserving a strict separation between:

1. condition-sensitive behavior;
2. linear decodability of an internal representation; and
3. causal steerability of downstream behavior.

The initial state-transfer hypothesis is:

> Linear decodability is common across behavioral traits, but decodability
> alone is a poor predictor of whether a theory-relevant construct state
> transfers causally to an independent behavioral task.

The realization effect is therefore Case Study 1, not the full scope of the
project. The first implementation transfers construct states using additive
steering and treats directed mean behavioral change as primary. Policy-slope
change is a stronger secondary outcome; policy-gain directions are deferred to
an exploratory method-development phase.

## 1. Scientific design

The first deliberately diverse pilot should contain four constructs:

| Construct | Family | Transferred state | Independent behavioral outcome |
|---|---|---|---|
| Realization effect | Decision | Account closure | Wager or risk choice |
| Evidence diagnosticity | Epistemic | Perceived evidence reliability | Confidence or belief revision |
| Authority deference | Social | Source reliability | Follow source versus follow evidence |
| Persistence | Agentic | Continuation/abandonment state | Continue, quit, or reallocate effort |

These are provisional operationalizations. Each must be reviewed for a clear
ground truth, a directional behavioral measure, and a task that is independent
of the prompts used to construct the activation direction.

For each construct, the experiment has four linked but distinct pieces:

1. **State contrast:** paired conditions intended to create one
   theory-relevant construct state versus another.
2. **Readout:** a linear direction built on the training split and evaluated on
   held-out prompts.
3. **Behavioral evaluation:** a fresh task measuring a pre-specified
   state-consistent directional outcome through a choice, estimate, action, or
   other structured output.
4. **Steering:** an intervention along the frozen state direction, with a
   baseline, positive and negative calibrated doses, downstream manipulation
   checks, and compliance controls.

The probe task must not be the downstream task. Otherwise, apparent steering
could be prompt or label matching rather than control of the intended trait.

## 2. System overview

The target data flow is:

```text
construct config
      |
      v
prompt generation and schema validation
      |
      +--> paired probe prompts ------------------+
      |                                           |
      v                                           v
split manifest                              activation logging
      |                                           |
      |                                           v
      |                                  train-only direction
      |                                           |
      |                         +-----------------+------------------+
      |                         |                                    |
      v                         v                                    v
independent behavior      held-out readout                  steering runs
      |                         |                                    |
      +-------------------------+----------------+-------------------+
                                                v
                                  parsing, quality checks, metrics
                                                |
                                                v
                                   cross-construct analysis
```

The pipeline should be config-driven. The same orchestration code should run
all constructs; construct-specific differences should live in versioned
configs, prompt templates, task definitions, and parsing rules.

## 3. Proposed repository additions

The active activation-analysis primitives should remain usable while a small
generic layer is introduced. The original behavioral implementation is
preserved under `archive/realization_effect/` rather than remaining in the
active import path.

```text
src/
├── activation_analysis/      # active activation/vector/prompt primitives
└── construct_benchmark/      # proposed generic benchmark orchestration
    ├── config.py             # schema, validation, and config loading
    ├── schemas.py            # prompt, split, result, and manifest schemas
    ├── prompts.py            # construct-agnostic prompt generation
    ├── splits.py             # leakage-safe split construction and audits
    ├── readout.py            # train-only direction construction/evaluation
    ├── behavior.py           # independent behavioral task execution
    ├── steering.py           # intervention orchestration and scale handling
    ├── parsing.py            # structured output parsing and quality flags
    ├── metrics.py            # per-construct metrics and uncertainty
    ├── manifests.py          # provenance and reproducibility metadata
    └── orchestration.py      # end-to-end run coordination

configs/
├── activation_analysis/      # existing configs
└── constructs/               # construct_spec files and registered adapters

archive/
└── realization_effect/       # original behavioral pipeline and adapters

experiments/
└── constructs/<construct_id>/
    ├── probe/                # reviewable probe prompt specifications
    └── behavior/             # reviewable independent-task specifications

results/
└── benchmark/<construct_id>/<model>/<run_id>/
    ├── manifest.json         # tracked or selectively retained provenance
    ├── summaries/            # small curated summaries
    └── raw/                  # must be explicitly ignored before benchmark runs

tests/
└── fixtures/benchmark/       # small deterministic prompt/output fixtures
```

Large raw outputs remain local and ignored. Small configs, prompt manifests,
audits, and curated summary tables should be tracked when they are part of the
reproducible experiment record.

The generic layer should call or wrap the existing activation primitives rather
than duplicate residual-stream logging, direction construction, or steering
logic.

## 4. Three-part configuration contract

The repository currently uses JSON configs. The new benchmark should use three
separate versioned JSON artifacts rather than one combined construct/run file.

### `construct_spec`

Stable scientific definition:

```text
construct_id and version
family
theory and motivating literature
state_definition
probe contrast and factorial controls
independent task definition
primary directional outcome
secondary policy-slope outcome
expected activation sign
inclusion and invalidity criteria
```

### `run_config`

Model- and run-specific settings:

```text
model ID and revision
activation site and candidate layers
token/region mode
position mode and intervention timing
direction estimator
projection-SD dose calibration
steering dose grid
random seeds and generation settings
```

### `analysis_spec`

Frozen analysis decisions:

```text
continuous held-out projection-margin estimand
primary state-transfer mean estimand
secondary policy-slope estimand
manipulation checks and controls
outcomes and exclusions
uncertainty and measurement-error method
cross-classified model × construct × task model
equivalence bounds and power-simulation inputs
multiple-comparison policy
```

The three artifacts must be hashed and recorded in every run manifest. A
construct-state direction must not be described as a decoded policy-gain
direction unless it was constructed specifically for responsiveness.

## 5. Prompt and split design

Every prompt should have stable metadata:

```text
prompt_id
pair_id
construct_id
condition
family
domain
source_template
split
seed
prompt_hash
```

Splits should be created by prompt family, template, or domain where possible,
not by randomly splitting near-identical paraphrases. The direction-building
split must be frozen before held-out evaluation.

The probe and downstream behavior sets should differ in at least one meaningful
dimension, such as topic, surface form, source model, or task format. A prompt
overlap audit should be a required pipeline step.

Before expensive model runs, validate:

- condition balance and pair completeness;
- expected fields and unique IDs;
- no train/held-out overlap;
- no direct condition labels in the prompt text unless intentional;
- sufficient domain and template diversity;
- no obvious lexical shortcut that identifies the label.

## 6. Execution stages

### Stage A: Validate the construct config

Fail early on missing fields, invalid scales, malformed parsing rules, missing
templates, duplicate IDs, or incompatible model settings. This stage must not
make API calls or load model weights.

### Stage B: Generate and freeze prompts

Generate paired probe prompts and independent behavioral prompts. Save a
manifest containing prompt hashes, source metadata, and split assignments.
Human-review a small sample before running a full generation.

### Stage C: Run the behavioral baseline

Run the independent task without steering. Confirm that the task produces a
directional, parseable outcome and that the intended construct is not merely a
formatting artifact.

### Stage D: Log activations

Use the existing activation-analysis primitives with the model, layer, token
mode, and activation site specified by the config. Store raw tensors locally;
retain only the outputs needed for review and reproduction.

### Stage E: Build and evaluate the direction

Construct the direction from the training split only. Evaluate on validation
and held-out readout prompts. The held-out behavioral prompts must not be used
to construct or tune the direction.

### Stage F: Run steering

Use the frozen train-only direction on fresh behavioral prompts. Include scale
zero, positive scales, negative scales, and a random-direction control. Record
both behavioral outcomes and formatting/compliance outcomes.

### Stage G: Parse and score

Convert outputs into typed outcomes while retaining raw text locally and
quality flags in summaries. Invalid or noncompliant responses must be reported,
not silently discarded.

### Stage H: Aggregate and compare constructs

Produce one summary per construct and a cross-construct table containing:

- continuous held-out projection margin;
- projection separation and generalization;
- primary state-transfer mean effect;
- secondary policy-slope effect;
- steering dose-response;
- sign-symmetry and random-direction controls;
- downstream-layer and output-accessibility manipulation checks;
- compliance and exclusion rates;
- uncertainty intervals.

The main cross-construct analysis should quantify whether readout strength
predicts state-transfer strength. It should not treat a single layer, prompt
family, or post-selected scale as the whole result.

## 7. Metrics

### Decodability

Use a continuous held-out standardized projection margin as the primary
decodability metric. Report paired accuracy, AUC, projection separation,
direction stability, and cross-domain generalization as secondary diagnostics.
Report training metrics only as diagnostics.

### Behavioral relevance

Measure whether the transferred state produces the pre-specified directional
mean outcome after controlling for prompt structure, model, task, and other
pre-specified covariates. Include unrelated outcomes to detect generic response
bias. Within-pair changes are especially important because they reduce
between-prompt confounds.

### Steerability

Estimate the state-transfer mean effect as a function of calibrated dose. Check
whether:

- the effect is distinguishable from zero;
- the dose-response is stable;
- negative scales reverse the direction;
- the effect survives compliance filtering;
- a random direction produces a comparable shift;
- the activation shift persists into later layers or output accessibility;
- a same-task positive control responds as expected.

Policy-slope change is a stronger secondary outcome. It should not be elevated
to the primary claim unless a responsiveness-specific direction estimator is
developed and validated.

### Cross-construct relationship

For each construct, define a pre-specified continuous decodability summary and
state-transfer summary. Compare them with uncertainty at the construct level,
with tasks within construct and models crossed with both. Four constructs are a
pilot; the confirmatory number of constructs must be selected by simulation,
not by an automatic 8–12 target.

## 8. Controls and failure criteria

Required controls include:

- shuffled condition labels;
- random or unrelated directions;
- negative steering scales;
- scale-zero matched baselines;
- downstream-layer and output-accessibility manipulation checks;
- same-task positive-control classification;
- neutral or ambiguous prompts with directional state predictions;
- lexical and prompt-length baselines where appropriate;
- prompt-overlap audits;
- output-format and compliance checks;
- independent behavioral tasks rather than direct label classification.

Do not claim steerability when:

- the direction does not generalize beyond the training split;
- the steering effect is only a formatting/compliance change;
- the behavioral task cannot be parsed reliably;
- the result depends on a post-selected layer, scale, prompt subset, or sign;
- the control direction produces the same effect.

A failure to decode is also a meaningful result. It should be retained in the
benchmark rather than used as a reason to keep searching until a direction is
found.

## 9. Reproducibility and artifact policy

Every run should save a manifest containing at least:

```text
run_id
construct_id
config_path and config_hash
git_commit
model_id and model_revision
prompt_manifest_hash
split_manifest_hash
random_seeds
activation settings
steering settings
command line
timestamps
software environment summary
```

Configs, small prompt manifests, audit outputs, and curated summaries should be
tracked. Model weights, API keys, raw generations, large activation tensors,
and exploratory checkpoints should remain ignored according to `.gitignore`.

Tests must use deterministic fixtures and must not require API calls or local
model weights.

## 10. Staged implementation plan

### Milestone 0: Scientific contract

- Finalize the first four construct operationalizations.
- Choose evidence diagnosticity/reliability as the first epistemic state.
- Specify state-consistent directional outcomes and secondary slope outcomes.
- Freeze primary hypotheses, controls, intervention timing, and initial
  model/layer settings.
- Run the precision simulation before setting the confirmatory construct count.

### Milestone 1: Schema and validation

- Implement separate `construct_spec`, `run_config`, and `analysis_spec`
  schemas.
- Add config and prompt-manifest validators.
- Add deterministic fixtures and tests for invalid configs, duplicate IDs,
  split leakage, and malformed outputs.

### Milestone 2: Generic realization vertical slice

- Wrap the existing realization pipeline behind the generic config interface.
- Reproduce its direction-building, held-out readout, and steering stages.
- Confirm that the new summaries preserve the existing analysis definitions.

### Milestone 3: First non-economic vertical slice

- Implement evidence diagnosticity/reliability.
- Run probe, continuous readout, independent state-transfer behavior, and
  downstream manipulation checks end to end.
- Fix schema, parsing, or quality-control problems before adding more
  constructs.

### Milestone 4: Four-construct pilot

- Run the realization, epistemic, social, and agentic cases.
- Review construct-level summaries and failure modes.
- Do not expand the benchmark until all four use the same protocol.

### Milestone 5: Confirmatory benchmark

- Select the confirmatory construct count using the pre-run precision simulation.
- Add a pre-specified layer/model robustness evaluation.
- Freeze the analysis code and run the final benchmark.

### Milestone 6: Paper and release

- Produce the cross-construct decodability-versus-steerability analysis.
- Report null results, controls, exclusions, and limitations.
- Track configs, manifests, curated summaries, and enough prompt data for
  independent review without committing private or oversized artifacts.

## 11. Non-goals

- Do not collapse all constructs into one universal behavioral direction.
- Do not treat a high probe score as causal evidence.
- Do not reuse the probe task as the behavioral endpoint.
- Do not optimize layers, signs, scales, or prompt subsets on held-out behavior.
- Do not begin with a large multi-model sweep before one complete vertical slice
  passes the quality gates.
