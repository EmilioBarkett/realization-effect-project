# Results Layout

This folder has three kinds of outputs:

- `results.csv` and `blocks/` are the canonical behavioral realization-effect
  results used by the analysis scripts.
- `final/` is reserved for future reference artifacts from the larger activation
  extraction and SAE training runs.
- `test/` contains disposable smoke-test artifacts used to check formatting,
  storage, and pipeline behavior.

Keep new local smoke runs under `results/test/`. Put the current SAE training
dataset and current reference SAE checkpoints under `results/final/`.
