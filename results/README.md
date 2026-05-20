# Results Layout

This folder has four kinds of outputs:

- `results.csv` and `blocks/` are the canonical behavioral realization-effect
  results used by the analysis scripts. `results.csv` is local-only and ignored
  because it is large; regenerate or copy it into place before running analyses
  that need the full behavioral table.
- `final/` is for reference artifacts from larger activation extraction and SAE
  training runs. These runs may still be exploratory, but they should be
  reproducible and tied to checked-in configs.
- `test/` contains disposable smoke-test artifacts used to check formatting,
  storage, and pipeline behavior.
- `audits/` contains small audit outputs that are useful for checking prompt
  overlap and other data-integrity questions.

Keep new local smoke runs under `results/test/`. Put small, curated reference
artifacts under `results/final/`; keep large activation tensors, raw steering
generations, and exploratory checkpoints ignored unless they are explicitly
needed for publication or review.

## Current Final Artifacts

- `final/activation_vectors/realization_vector_v1_layer18_direction_train_only/`
  is the current reference activation-vector artifact. It contains the
  train-only layer-18 realization direction and held-out readout summaries.
- `final/residual_streams/` contains README stubs in git. Full local activation
  tensors are intentionally ignored because they are large and reproducible from
  the checked-in prompt/config files plus local model weights.
- `final/sae/` contains README stubs in git. Earlier SAE outputs are archived
  locally under ignored `results/legacy/` and are not the active report path.
- `test/activation_vectors/` contains ignored local smoke and steering runs.
  The final report's train-only steering run lives there locally, while
  report-ready text, figures, and selected train-only readout artifacts are
  tracked separately.
