# Activation Analysis Experiments

This directory holds reviewable prompt CSVs used before residual-stream logging.

These files are the reviewable prompt inputs for the activation foundation. The
current CSVs are realization-anchor prompts; the next task is to generalize
their paired metadata and split structure to evidence diagnosticity and later
constructs. Older emotion-probe and SAE prompt sets are reference material.

The checked-in realization CSVs predate the canonical multi-construct prompt
inventory and therefore preserve their original generation labels for
reproducibility. They are not benchmark manifests and must not be passed to
the construct-benchmark validator as-is. New combined inventories and newly
generated files use `direction_validation` and `direction_heldout`.

## Current Path

- Generation plan:
  `configs/activation_analysis/realization_vector_generation_v1.json`
- Generated paired prompt CSV:
  `experiments/activation_analysis/prompts/activation_vectors/realization_vector_v1.csv`
- Residual activation outputs:
  `results/final/residual_streams/`
- Vector-analysis outputs:
  `results/final/activation_vectors/`

## Pilot Generation

```bash
export OPENROUTER_API_KEY=your_key_here

./venv/bin/python scripts/generate_activation_prompts.py \
  --pilot-all-cells \
  --output experiments/activation_analysis/prompts/activation_vectors/realization_vector_v1_pilot.csv
```

## Parallel Full Generation

Use one output file per model when running multiple terminals:

```bash
./venv/bin/python scripts/generate_activation_prompts.py \
  --models gpt54 \
  --output-template experiments/activation_analysis/prompts/activation_vectors/realization_vector_v1__{model}.csv \
  --resume
```

Repeat the same command with `sonnet` and `grok_fast`. Then merge the
completed model CSVs:

```bash
./venv/bin/python scripts/generate_activation_prompts.py \
  --merge-inputs \
    experiments/activation_analysis/prompts/activation_vectors/realization_vector_v1__gpt54.csv \
    experiments/activation_analysis/prompts/activation_vectors/realization_vector_v1__sonnet.csv \
    experiments/activation_analysis/prompts/activation_vectors/realization_vector_v1__grok_fast.csv \
  --output experiments/activation_analysis/prompts/activation_vectors/realization_vector_v1.csv
```

## Residual Logging

```bash
./venv/bin/python scripts/log_residual_streams.py \
  --model-id models/gemma-3-4b-pt \
  --prompt-csv experiments/activation_analysis/prompts/activation_vectors/realization_vector_v1.csv \
  --layers 18 \
  --token-mode nonpad \
  --include-token-regions scenario \
  --storage-dtype float32 \
  --local-files-only
```

## Vector Build/Eval

The vector build/evaluation scripts are retained as activation primitives and
now use the tracked iterator in `activation_analysis.activation_store`. A
clean Python 3.11 editable installation and active test run now pass `make
check`; end-to-end validation against a real activation run remains pending.

```bash
./venv/bin/python scripts/build_activation_vectors.py \
  --activation-run results/final/residual_streams/realization_vector_v1_layer18_regions_float32 \
  --layers 18 \
  --output-dir results/final/activation_vectors/realization_vector_v1_layer18

./venv/bin/python scripts/evaluate_activation_vectors.py \
  --activation-run results/final/residual_streams/realization_vector_v1_layer18_regions_float32 \
  --direction results/final/activation_vectors/realization_vector_v1_layer18/mean_direction.npy \
  --layers 18 \
  --output-dir results/final/activation_vectors/realization_vector_v1_layer18/evaluation
```
