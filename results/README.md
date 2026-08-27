# Results layout

The current project has produced a complete API-generated vector/probe
inventory, but not a complete generalized benchmark dataset. Behavior,
calibration, and steering-task prompts remain to be generated separately. Most
other tracked results in this directory are realization-anchor or
activation-pipeline reference artifacts.

This folder has four kinds of outputs:

- `results.csv` and `blocks/` are historical realization-effect behavioral
  results used by the archived analysis scripts. `results.csv` is local-only
  and ignored because it is large.
- `final/` is for reference artifacts from larger activation extraction and SAE
  training runs. These runs may still be exploratory, but they should be
  reproducible and tied to checked-in configs.
- `benchmark/vector_prompts_v2_luna/full_final_all16/` contains the complete
  16-construct vector/probe inventory and its final manifest. It is
  non-confirmatory and scope-partial by design.
- `test/` contains disposable smoke-test artifacts used to check formatting,
  storage, and pipeline behavior.
- `audits/` contains small audit outputs that are useful for checking prompt
  overlap and other data-integrity questions.

Keep new local smoke runs under `results/test/`. Put small, curated reference
artifacts under `results/final/`; keep large activation tensors, raw steering
generations, and exploratory checkpoints ignored unless they are explicitly
needed for publication or review.

Generalized benchmark runs use one shared multi-construct root at
`results/benchmark/<run_id>/`. Its `raw/` directory is ignored by Git, while
`constructs/<construct_id>/` keeps directions, readouts, calibration, and
steering outputs namespaced by construct. The preparation command snapshots
the frozen configs and run plan; finalization writes `checksums.sha256` and
can sync the complete run to an S3-compatible archive.

## Current Final Artifacts

- `final/activation_vectors/realization_vector_v1_layer18_direction_train_only/`
  is the current reference activation-vector artifact. It contains the
  train-only layer-18 realization direction and real-model engineering decode
  summaries; it is not evidence of generalized benchmark steerability.
- `final/residual_streams/` contains README stubs in git. Full local activation
  tensors are intentionally ignored because they are large and reproducible from
  the checked-in prompt/config files plus local model weights.
- `final/sae/` contains README stubs in git. Earlier SAE outputs are archived
  locally under ignored `results/legacy/` and are not the active report path.
- `test/activation_vectors/` contains ignored local smoke and steering runs.
  The final report's train-only steering run lives there locally, while
  report-ready text, figures, and selected train-only readout artifacts are
  tracked separately.
