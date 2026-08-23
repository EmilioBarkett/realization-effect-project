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
high-dimensional predictor to the four-construct pilot. Do not collapse
representation and control into one outcome.

The active construct families are realization/account closure,
evidence diagnosticity, source reliability/authority, and
persistence/continuation. Specification gaming in coding agents is not the
current project direction.

## Maturity and scope

The project is currently in scientific protocol development and benchmark
infrastructure implementation. The shared multi-construct control plane is
implemented; the end-to-end measurement layer is not.

Do not describe the following as completed:

- continuous projection-margin analysis;
- neutral/within-cell dose calibration;
- generic state-transfer metrics;
- downstream manipulation-check orchestration;
- the evidence-diagnosticity experiment.

The original realization behavioral pipeline is archived under
[`archive/realization_effect/`](archive/realization_effect/). The active
activation prompt generator is retained because it will be generalized to
construct-specific paired prompts.

## Repository map

- `src/activation_analysis/`: active prompt generation, residual logging,
  activation storage, vector primitives, and steering primitives.
- `src/construct_benchmark/`: shared construct/run/analysis schemas,
  canonical prompt records, split validation, and multi-construct run plans.
- `scripts/`: active activation-analysis entrypoints.
- `configs/activation_analysis/`: active prompt-generation and probe configs.
- `experiments/activation_analysis/`: reviewable prompt CSVs.
- `tests/`: active tests for activation and prompt-generation behavior.
- `archive/realization_effect/`: archived realization behavioral code,
  configs, notebooks, adapters, and tests.
- `archive/sae/`: archived SAE-training and feature-analysis tests; not active
  test scope.
- `archive/documentation/`: archived planning documents.
- `configs/construct_benchmark/`: versioned construct definitions, run
  configurations, and analysis specifications.
- `reports/` and `results/`: reference artifacts and local/ignored outputs.

The active vector path now imports its iterator from
`activation_analysis.activation_store`. The old SAE-training and
feature-analysis tests are archived. The active environment is now a Python
3.11 editable install, and `make check` passes; the two PyTorch-dependent
interpreter smoke tests are skipped when the optional `interp` extra is not
installed.

The multi-construct control plane uses one combined prompt inventory and one
shared activation pass, then fans out into construct-scoped directions,
readouts, calibration, behavior, and steering outputs. Never pool directions
across `construct_id` values.

## Research invariants

For every construct:

- define a directional state contrast before data collection;
- freeze train, validation, held-out, and downstream-task splits;
- construct the main direction from the training split only;
- keep probe prompts and downstream behavior tasks independent;
- use continuous held-out projection margin as the primary readout;
- treat pair accuracy as secondary;
- use directed mean state transfer as the primary steering outcome;
- treat policy-slope change as secondary;
- calibrate steering dose using frozen neutral or within-condition/within-cell
  training variance, not a positive/negative mixture that couples separation to
  intervention magnitude;
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
`results/benchmark/<construct>/<model>/<run>/raw/` is already covered by
`.gitignore` before benchmark runs begin.

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
