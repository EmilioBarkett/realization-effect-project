# Results layout

The current project has not yet produced a generalized benchmark dataset. Most
tracked results in this directory are realization-anchor or activation-pipeline
reference artifacts.

This folder has four kinds of outputs:

- `results.csv` and `blocks/` are historical realization-effect behavioral
  results used by the archived analysis scripts. `results.csv` is local-only
  and ignored because it is large.
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

Before any generalized benchmark run, add an explicit ignore rule for
`results/benchmark/<construct>/<model>/<run>/raw/` and record the run manifest
and split/config hashes.

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
