# Activation Analysis Experiments

This directory holds reviewable prompt CSVs used before residual-stream logging.

The active direction is an Anthropic-style activation-vector experiment for
realization framing and risk-taking behavior. Older emotion-probe and SAE prompt
sets are preserved as archive/reference material.

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

./venv/bin/python scripts/run_realization_experiment.py \
  --prompt-version generation \
  --generation-plan configs/activation_analysis/realization_vector_generation_v1.json \
  --generation-pilot-all-cells \
  --generation-output experiments/activation_analysis/prompts/activation_vectors/realization_vector_v1_pilot.csv
```

Preferred direct entrypoint:

```bash
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
