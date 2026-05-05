# SAE Architecture

The SAE layer is intentionally separate from residual-stream logging. It now
has an executable local PyTorch training scaffold, but the data collection and
research settings are still expected to be chosen before a serious run.

`src/emotion_activation/` owns prompt construction, forward passes, activation
capture, and run validation. `src/sae/` starts from completed activation run
directories and turns those tensors into vector datasets for contrast analysis,
local SAE training, and future feature interpretation.

## Current Boundary

Activation runs provide:

- `.npy` tensors shaped `[batch, selected_tokens, d_model]`
- paired `.jsonl` metadata with prompt IDs, token IDs, token positions, token
  regions, layer, token mode, and activation site
- a `manifest.json` with model and extraction settings

The SAE package consumes those run folders through `sae.dataset`.

## Proposed Package Shape

```text
src/sae/
├── __init__.py
├── config.py       # Dataset config objects for activation-run selection
├── dataset.py      # Iterator from activation shards to token-level vectors
├── metrics.py      # Planned metric names for training/evaluation
├── model.py        # Small local sparse autoencoder module
├── features.py     # Placeholder feature-analysis interfaces
└── training.py     # Local training loop over activation vectors
```

The first stable API is:

```python
from sae.dataset import iter_activation_vectors

for record in iter_activation_vectors(
    "results/test/residual_streams/test_gemma3_4b_regions_smoke",
    layers={12},
    token_regions={"scenario", "decision_question"},
):
    vector = record.vector
    metadata = record.metadata
```

This lets us build simple contrast vectors before choosing a full SAE training
implementation.

## Why This Helps

SAEs need broad activation samples, but this project also needs emotion-specific
comparisons. The activation logger now keeps broad non-padding activations and
adds token-region metadata. The SAE dataset layer can therefore train or analyze
over all vectors while still slicing later by:

- loss/gain condition
- emotion/control contrast role
- token region
- layer
- activation site

The immediate next step for a real SAE is still to collect enough validated
activation runs and inspect whether the dataset contains the right vector counts
by layer and token region. The training scaffold exists so that once those runs
are available, the project can move directly from a dataset config to a saved
checkpoint.

## Training Scaffold

The first backend is deliberately small and local:

- a single-layer encoder/decoder SAE in `sae.model`
- `relu` or `topk` feature activations
- MSE reconstruction loss plus optional L1 penalty
- checkpoint and JSON manifest output under ignored `results/final/sae/`
  for current reference runs or `results/test/sae/` for smoke runs

Example command after filling `configs/sae/templates/initial_dataset_template.json`:

```bash
./venv/bin/python scripts/train_sae.py \
  --dataset-config configs/sae/templates/initial_dataset_template.json \
  --training-config configs/sae/templates/initial_training_template.json
```

This is meant to validate the full post-inference path before committing to a
larger SAE library or a more expensive training run.

Implementation decisions are tracked in `docs/sae_decisions.md`.
