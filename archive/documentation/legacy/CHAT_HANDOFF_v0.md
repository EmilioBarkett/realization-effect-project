# Handoff: Generalized Decodability–Steerability Project

## Context

This repository is being developed into a NeurIPS-style benchmark testing
whether internal linear decodability predicts causal behavioral steerability in
language models.

The existing realization-effect experiment is the anchor case study. The new
project should generalize beyond behavioral economics and test a deliberately
diverse set of constructs.

The initial state-transfer hypothesis is:

> Linear decodability is common across behavioral traits, but decodability
> alone is a poor predictor of whether a theory-relevant construct state
> transfers causally to an independent behavioral task.

Read these files first:

1. `AGENTS.md` — repository and research instructions.
2. `SCIENTIFIC_PROTOCOL.md` — current v0.2 scientific contract.
3. `PROJECT_ARCHITECTURE.md` — proposed system design and implementation plan.
4. `readme.md` — existing pipeline, commands, and reference results.

## Current repository state

- Repository: `realization-effect-project`
- Origin: `https://github.com/EmilioBarkett/realization-effect-project.git`
- Current branch: `main`
- Active package: `src/activation_analysis/`
- Archived realization package: `archive/realization_effect/src/realization_effect/`
- Active command-line scripts: `scripts/`
- Archived realization-specific scripts: `archive/realization_effect/scripts/`
- Active configs: `configs/activation_analysis/`
- Existing prompt artifacts: `experiments/`
- Existing tests: `tests/`
- Existing paper/report source: `reports/`
- Full behavioral results, model weights, raw generations, and large activation
  outputs are local-only or ignored and should not be assumed to exist.

The root-level planning files `AGENTS.md`, `PROJECT_ARCHITECTURE.md`, and this
file are currently local working-tree additions unless they have been committed
by the user.

## Agreed project direction

The first pilot should include four construct-state families:

1. realization effect — decision/economic anchor;
2. evidence diagnosticity/reliability — epistemic;
3. authority deference versus independent verification — social;
4. persistence versus abandonment — agentic.

Each construct needs:

- a controlled contrast used to construct a theory-relevant state direction;
- a separate independent behavioral task;
- train-only direction construction;
- held-out readout evaluation;
- a pre-specified directional mean state-transfer outcome;
- policy-slope change as a secondary outcome;
- calibrated steering doses, timing, and manipulation checks;
- structured parsing and quality checks.

The first epistemic direction should represent perceived evidence
diagnosticity/reliability. It should not be described as an updating-
responsiveness direction. Updating slope is a secondary behavioral outcome.

## Proposed engineering direction

Build a config-driven generic layer instead of duplicating the realization
scripts. The proposed package is `src/construct_benchmark/`, with components for
config validation, prompt/split management, readout, state-transfer behavior,
steering, parsing, metrics, manipulation checks, manifests, and orchestration.

Proposed construct configs live under:

```text
configs/constructs/<construct_id>/construct_spec.json
configs/runs/<run_id>.json
configs/analysis/<analysis_id>.json
```

The three artifacts should separate stable state definitions from model/run
settings and frozen analysis decisions. The state direction must not be called
a policy-gain direction unless it was explicitly constructed for responsiveness.

The generic pipeline should be:

```text
validate config
  → generate and freeze prompts
  → create leakage-safe splits
  → run behavioral baseline
  → log activations
  → build train-only direction
  → evaluate held-out readout
  → calibrate and run independent state-transfer task
  → run downstream manipulation checks
  → parse and quality-check outputs
  → compute state-transfer and secondary slope metrics
  → fit cross-classified model × construct × task summaries
```

The activation-analysis paired-prompt generator remains active. It is the
starting point for replacing the realization-specific prompt builder with
construct-agnostic paired contrasts. Activation logging consumes frozen prompt
CSVs or structured probe configs and no longer imports the archived behavioral
package.

## Immediate next tasks

Work in this order:

1. Review and finalize the state definition and directional mean outcome for
   each pilot construct.
2. Implement the three-artifact schema and validator.
3. Add continuous projection-margin readout and projection-SD dose calibration.
4. Add prompt/split-manifest schemas plus overlap/leakage tests.
5. Build a generic realization state-transfer vertical slice using the existing
   code.
6. Add downstream-layer, output-accessibility, and same-task manipulation
   checks.
7. Run the local test suite with deterministic fixtures; do not require APIs or
   model weights in tests.
8. Implement evidence diagnosticity end to end.
9. Run the precision simulation before deciding the confirmatory construct
   count.

## Non-negotiable research constraints

- Never use held-out behavioral prompts to construct or tune the direction.
- Never equate probe accuracy with causal control.
- Keep probe prompts and downstream behavioral tasks meaningfully independent.
- Treat the initial direction as a construct-state direction, not a decoded
  policy-gain direction.
- Make directed mean state transfer primary; treat policy-slope change as
  secondary unless a responsiveness-specific estimator is developed.
- Include shuffled-label, random-direction, zero-scale, negative-scale, and
  compliance controls where applicable.
- Use projection standard deviation from training calibration prompts as the
  primary dose unit; record residual-norm ratios as safety diagnostics.
- Distinguish prompt-prefill, generation-only, every-step, and fixed-position
  intervention timing.
- Check downstream-layer persistence and output accessibility; injection-layer
  projection change alone is not a manipulation check.
- Do not flip signs or select layers/scales after inspecting held-out behavior.
- Retain failed decodability and failed steerability cases as results.
- Treat models as crossed with constructs and tasks in the statistical model.
- Do not select an automatic 8–12-construct target; use a pre-run simulation.
- Record config hashes, prompt/split hashes, Git commit, model revision, seeds,
  activation settings, steering settings, and exclusions in every run manifest.
- Keep API keys, model weights, raw generations, and large tensors out of Git.

## Safe operating instructions for the next chat

Before editing:

```bash
git status --short --branch
rg --files
```

Do not launch API calls, download models, or start a large experiment until the
user explicitly requests it and the config/protocol has been reviewed. Prefer
small implementation steps, deterministic tests, and explicit diffs. Do not
push or publish changes unless the user asks.

The next useful deliverable is the config schema plus validator, not a new
large experiment.
