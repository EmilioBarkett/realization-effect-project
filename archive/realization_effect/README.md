# Archived realization-effect implementation

This directory contains the original realization-effect behavioral pipeline
and its realization-specific activation/evaluation adapters. It is preserved
for reproducibility and historical reference, but it is not part of the
active generalized benchmark pipeline.

## What was archived

- `src/realization_effect/`: behavioral prompt construction, API orchestration,
  parsing, reconciliation, partitioning, and analysis.
- `configs/`: the original realization condition table and model catalogue.
- `notebooks/`: exploratory realization-effect analyses.
- `scripts/`: behavioral collection, behavioral analysis, realization-specific
  activation behavior evaluation, report construction, and steering adapters.
- `tests/`: tests that exercise the archived behavioral package or its
  realization-specific adapters.
- `run_dual_experiment.sh`: the original multi-run launcher.

The generated behavioral results, model weights, raw generations, and large
activation outputs were not moved or deleted. They remain in their existing
ignored or reference locations unless separately documented.

## Active replacement boundary

The active repository retains:

- `src/activation_analysis/`, including the paired-prompt generator,
  residual-stream logger, activation store, vector analysis, and steering
  primitives;
- `configs/activation_analysis/` and
  `experiments/activation_analysis/`;
- generic activation scripts such as prompt-overlap auditing, residual
  logging, direction construction, readout evaluation, and run validation.

Activation logging now consumes a frozen prompt CSV or structured probe config
explicitly. It no longer imports the archived realization package or creates
realization prompts inline.

## Historical execution

The archived scripts assume the archived source layout. If an old result must
be reproduced, run the script from the repository root with both source roots
available, for example:

```bash
PYTHONPATH=archive/realization_effect/src:src \
  ./venv/bin/python archive/realization_effect/scripts/analyze_realization_results.py \
  results/results.csv
```

Collection commands also require explicit paths under
`archive/realization_effect/configs/realization_effect/`. Preserved source is
not the same as verified reproducibility; the archived tests are excluded from
the active test suite and require the archived environment/dependencies.

Do not use the archived code as the template for the new construct benchmark;
adapt the active activation primitives and the forthcoming construct-specific
schemas instead.
