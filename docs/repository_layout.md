# Repository Layout

The project is split into two active code areas:

- `src/realization_effect/` contains the behavioral experiment: prompt
  construction, OpenRouter collection, parsing, reconciliation, dashboarding,
  and statistical analysis.
- `src/emotion_activation/` contains residual-stream logging, emotion probes,
  vector extraction, and later steering utilities.

Preferred command-line entrypoints live in `scripts/`. The root-level Python
compatibility wrappers have been removed so each command has a single obvious
home.

Regression tests live in `tests/` and cover parser behavior plus analysis
guardrails that protect baseline-dependent regressions.
Ignored non-canonical CSV archives live under `tests/fixtures/noncanonical/`
so they stay out of the active `results/` workflow.

Static inputs live under `configs/`: realization-effect conditions stay in
`configs/realization_effect/`, while emotion-vector contrast definitions live
in `configs/emotion_activation/`. Reviewable experiment material that is not
package code lives under `experiments/`, currently
`experiments/emotion_activation/`.

Active generated outputs live in `results/`: `results/results.csv` is
canonical, and `results/sample_results.csv` is a small review sample.
Generated grouped outputs and resumable blocks are rebuildable and ignored by
git.
Local model weights live in `models/` and are intentionally gitignored.
