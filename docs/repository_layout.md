# Repository Layout

The project is split into three active code areas:

- `src/realization_effect/` contains the behavioral experiment: prompt
  construction, OpenRouter collection, parsing, reconciliation, dashboarding,
  and statistical analysis.
- `src/activation_analysis/` contains residual-stream logging, emotion probes,
  OpenRouter prompt-generation helpers, and activation-run validation.
- `src/sae/` contains archived/supporting SAE utilities. The current
  interpretability path is activation-vector analysis rather than SAE training.

Preferred command-line entrypoints live in `scripts/`. The root-level Python
compatibility wrappers have been removed so each command has a single obvious
home.

Regression tests live in `tests/` and cover parser behavior plus analysis
guardrails that protect baseline-dependent regressions.
Ignored non-canonical CSV archives live under `tests/fixtures/noncanonical/`
so they stay out of the active `results/` workflow.

Static inputs live under `configs/`: realization-effect conditions stay in
`configs/realization_effect/`, while emotion/realization activation-vector
generation plans live in `configs/activation_analysis/`. Archived SAE dataset
selections live under `configs/sae/archive/`. Reviewable experiment material
that is not package code lives under `experiments/`, currently
`experiments/activation_analysis/`.

Active generated outputs live in `results/`: `results/results.csv` is
canonical, and `results/sample_results.csv` is a small review sample.
Generated grouped outputs and resumable blocks are rebuildable and ignored by
git.
Archived SAE outputs live under ignored `results/legacy/`.
Local model weights live in `models/` and are intentionally gitignored.
