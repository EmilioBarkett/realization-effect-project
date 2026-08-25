# AGENTS.md

## Read first

The current scientific authority is
[`PROJECT_DIRECTION.md`](PROJECT_DIRECTION.md). Read it before making design
or implementation decisions. Then consult:

1. [`BENCHMARK_RESEARCH_DIRECTION.md`](BENCHMARK_RESEARCH_DIRECTION.md) for
   the detailed benchmark proposal and novelty boundary;
2. [`SCIENTIFIC_PROTOCOL.md`](SCIENTIFIC_PROTOCOL.md) for experimental rules;
3. [`PROJECT_ARCHITECTURE.md`](PROJECT_ARCHITECTURE.md) for implementation
   boundaries and milestones;
4. [`readme.md`](readme.md) for the human-facing repository overview.

Archived documents are historical snapshots. They do not override the root
documents above.

## Mission

This repository is developing a representation–steerability correspondence
benchmark for the relationship between:

1. prompt-condition behavioral sensitivity;
2. linear decodability of a theory-relevant internal state; and
3. causal steerability of an independent downstream behavior.

The central hypothesis is that decodability is common but is an incomplete and
often poor predictor of steerability. The longer-term benchmark may add
representation-profile features such as direction stability, context
consistency, localization, and dimensionality, but v1 must not overfit a
high-dimensional predictor to the small engineering or Wave 1 pilot. Do not
collapse representation and control into one outcome.

The active candidate bank is the versioned 16-entry registry, balanced across
decision, epistemic, social, and agentic families. Wave 1 is
`realization_account_closure`, `evidence_diagnosticity`, `source_reliability`,
and `persistence_continuation`; Waves 2–4 are planned, not yet specified for
generation. Specification gaming in coding agents is not the current project
direction.

## Maturity and scope

The project is currently in scientific protocol development and benchmark
infrastructure implementation. The shared multi-construct control plane,
16-construct registry, Wave 1 construct specifications, generic generation
adapter, generation plans, no-API dry-run path, deterministic fake vertical
slice, deterministic prompt run-mode selection, and environment-independent
measurement core are implemented; real-model end-to-end validation is not.

Do not describe the following as completed:

- real-run projection-margin results;
- real-run neutral/within-condition calibration results;
- a validated local or RunPod steering run;
- downstream manipulation-check orchestration;
- the evidence-diagnosticity experiment.

Fixture-tested code now covers train-only directions, held-out standardized
projection margins, neutral/within-condition calibration, strict Wave 1 output
parsing, directed state-transfer scoring, deterministic shuffled/random
controls, timing-aware injection, validation-only layer selection,
independent prompt-role/family checks, pre-registered category schedules,
bootstrap interval primitives, readout/steering planning CLIs, named
prompt-generation review/full modes, pair-preserving model-side test/full
selection, and an optional activation wall-clock guard. These are implemented
code, not empirical results. PyTorch, Transformers, a concrete
model revision, model weights, and a representative activation run remain
absent from the base development environment.

The original realization behavioral pipeline is archived under
[`archive/realization_effect/`](archive/realization_effect/). The active
activation prompt generator is retained because it will be generalized to
construct-specific paired prompts.

## Repository map

- `src/activation_analysis/`: active prompt generation, residual logging,
  activation storage, vector primitives, and steering primitives.
- `src/construct_benchmark/`: shared construct/run/analysis schemas,
  canonical prompt records, split validation, registry validation, generic
  generation, and multi-construct run plans.
- `scripts/`: active activation-analysis entrypoints.
- `scripts/generate_construct_prompts.py`: benchmark-facing generic generation
  CLI with dry-run support.
- `scripts/select_benchmark_run_mode.py`: deterministic test/full inventory
  selection without external calls.
- `scripts/run_fake_benchmark.py`: deterministic no-API vertical-slice runner.
- `scripts/validate_construct_registry.py`: registry/spec agreement check.
- `configs/activation_analysis/`: active prompt-generation and probe configs.
- `experiments/activation_analysis/`: reviewable prompt CSVs.
- `tests/`: active tests for activation and prompt-generation behavior.
- `archive/realization_effect/`: archived realization behavioral code,
  configs, notebooks, adapters, and tests.
- `archive/sae/`: archived SAE-training and feature-analysis tests; not active
  test scope.
- `archive/documentation/`: archived planning documents.
- `agents/`: maintainer handoffs and local/GPU execution notes; these do not
  override the root scientific documents.
- `configs/construct_benchmark/`: versioned construct definitions, run
  configurations, analysis specifications, the 16-entry registry, and Wave 1
  generation plans.
- `reports/` and `results/`: reference artifacts and local/ignored outputs.

The active vector path now imports its iterator from
`activation_analysis.activation_store`. The old SAE-training and
feature-analysis tests are archived. The active environment is now a Python
3.11 editable install, and `make check` passes; the two PyTorch-dependent
interpreter smoke tests are skipped when the optional `interp` extra is not
installed.

The multi-construct control plane uses one combined prompt inventory and one
shared activation pass, then fans out into construct-scoped directions,
readouts, calibration, behavior, and steering outputs. The benchmark-facing
generator creates canonical rows before activation logging; never pool
directions across `construct_id` values.

## Research invariants

For every construct:

- define a directional state contrast before data collection;
- freeze train, validation, held-out, and downstream-task splits;
- construct the main direction from the training split only;
- keep probe prompts and downstream behavior tasks independent;
- compose Wave 1 as probe first, independent downstream task second; allow only
  the induced state—not probe text, labels, or entities—to carry over;
- use continuous held-out projection margin as the primary readout;
- treat pair accuracy as secondary;
- use directed mean state transfer as the primary steering outcome;
- treat policy-slope change as secondary;
- calibrate steering dose using frozen neutral or within-condition/within-cell
  training variance, not a positive/negative mixture that couples separation to
  intervention magnitude;
- treat `review` prompt inventories and `test` model runs as engineering
  artifacts only; use the complete `full` inventory and `full` run mode for
  confirmatory analysis;
- include zero, negative, shuffled/random, compliance, and collateral controls;
- do not select signs, layers, scales, or subsets after seeing held-out results;
- report uncertainty and retain null results;
- distinguish injection-layer projection from downstream manipulation checks;
- record model, layer, activation site, token/position mode, splits, scales,
  timing, sample counts, and exclusions in every manifest.

Do not call a direction a policy-gain direction unless it was explicitly
constructed to represent responsiveness or gain control.

## Safe workflow

Before editing:

```bash
git status --short --branch
rg --files
```

Use small source-level changes and deterministic fixtures. Do not launch API
calls, download model weights, or run a large experiment during ordinary
development. Those actions require explicit user intent and a reviewed
configuration.

Before creating outputs, inspect `.gitignore`. In particular,
`results/benchmark/<run_id>/raw/` is already covered by `.gitignore` before
benchmark runs begin. The shared run root is intentionally multi-construct;
construct-specific outputs live below its `constructs/<construct_id>/`
namespace.

Run the narrowest relevant check first, then `make check` when the project
environment is available. The current development baseline uses the Python
3.11 `venv`; if it is absent in a fresh checkout, report that limitation
instead of claiming tests passed.

Do not delete or regenerate existing local data. Preserve the archived source
and reference artifacts. Do not commit or push unless explicitly requested.

## Handoff requirements

Every handoff should state:

- changed files;
- active versus archived paths;
- commands and checks run;
- tests that could not run and why;
- assumptions about local data, model weights, credentials, and tooling; and
- whether the work is protocol, repository preparation, or implemented
  benchmark code.
