# Repository Layout

The project is split into two active code areas:

- `src/realization_effect/` contains the behavioral experiment: prompt
  construction, OpenRouter collection, parsing, reconciliation, dashboarding,
  and statistical analysis.
- `src/interpretability/` contains residual-stream logging and later SAE-facing
  utilities.

Preferred command-line entrypoints live in `scripts/`. The root-level Python
compatibility wrappers have been removed so each command has a single obvious
home.

Regression tests live in `tests/` and cover parser behavior plus analysis
guardrails that protect baseline-dependent regressions.
Ignored non-canonical CSV archives live under `tests/fixtures/noncanonical/`
so they stay out of the active `results/` workflow.

Static inputs live under `configs/realization_effect/`. Active generated
outputs live in `results/`: `results/results.csv` is canonical, and
`results/sample_results.csv` is a small review sample. Generated grouped
outputs and resumable blocks are rebuildable and ignored by git.
Local model weights live in `models/` and are intentionally gitignored.
