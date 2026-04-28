# Repository Layout

The project is split into two active code areas:

- `src/realization_effect/` contains the behavioral experiment: prompt
  construction, OpenRouter collection, parsing, reconciliation, dashboarding,
  and statistical analysis.
- `src/interpretability/` contains residual-stream logging and later SAE-facing
  utilities.

Preferred command-line entrypoints live in `scripts/`. The root-level Python
files are compatibility wrappers so older commands and notebooks can still
import names such as `run_experiment.build_prompt`.

Static inputs live under `configs/realization_effect/`. Generated outputs still
live in `results/` for compatibility with the existing analysis and dashboard
flow. Local model weights live in `models/` and are intentionally gitignored.
