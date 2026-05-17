# Cloud Open-Model Activation Logging

This note describes the activation logging path for a larger open model after
the behavior-only run shows enough signal to justify the cost.

## When To Run

Do not start with activation logging. First run behavior generation on the same
larger model and confirm:

- high parse validity,
- meaningful paper/open versus realized/closed wager or risk deltas,
- no prompt-format failures in the smoke run.

Then log residual activations for the same prompt CSV and model.

## Inputs

- Prompt CSV:
  `experiments/activation_analysis/prompts/activation_vectors/realization_vector_v1.csv`
- Prompt text column:
  `prompt_text`
- Prompt IDs:
  `prompt_id`

For a full model-specific activation vector, include at least:

- `direction_train`
- `direction_val`
- `behavior_eval`

The current logger reads the full CSV unless `--limit` is provided.

## Output Layout

Use a model-specific directory under:

```text
results/final/residual_streams/
```

Example:

```text
results/final/residual_streams/qwen3_32b_instruct_realization_v1_layerXX_regions_float16/
├── manifest.json
├── prompts.jsonl
└── activations/
```

## Cloud Smoke Test

Run a tiny activation smoke test first:

```bash
./venv/bin/python scripts/log_residual_streams.py \
  --model-id Qwen/Qwen3-32B-Instruct \
  --prompt-csv experiments/activation_analysis/prompts/activation_vectors/realization_vector_v1.csv \
  --layers 32 \
  --activation-site resid_post \
  --token-mode nonpad \
  --token-region-strategy auto \
  --include-token-regions scenario \
  --storage-dtype float16 \
  --batch-size 1 \
  --limit 4 \
  --max-length 512 \
  --device-map auto \
  --dtype bf16 \
  --trust-remote-code \
  --attn-implementation flash_attention_2 \
  --run-name qwen3_32b_instruct_activation_smoke \
  --output-dir results/test/residual_streams/qwen3_32b_instruct_activation_smoke \
  --overwrite
```

If FlashAttention is unavailable, omit:

```bash
--attn-implementation flash_attention_2
```

Validate immediately:

```bash
./venv/bin/python scripts/validate_activation_run.py \
  results/test/residual_streams/qwen3_32b_instruct_activation_smoke
```

## Full Activation Run

After the smoke test validates:

```bash
./venv/bin/python scripts/log_residual_streams.py \
  --model-id Qwen/Qwen3-32B-Instruct \
  --prompt-csv experiments/activation_analysis/prompts/activation_vectors/realization_vector_v1.csv \
  --layers 32 \
  --activation-site resid_post \
  --token-mode nonpad \
  --token-region-strategy auto \
  --include-token-regions scenario \
  --storage-dtype float16 \
  --batch-size 1 \
  --max-length 512 \
  --device-map auto \
  --dtype bf16 \
  --trust-remote-code \
  --run-name qwen3_32b_instruct_realization_v1_layer32_regions_float16 \
  --output-dir results/final/residual_streams/qwen3_32b_instruct_realization_v1_layer32_regions_float16
```

## After Logging

Validate, build the model-specific realization direction, and evaluate
projections:

```bash
./venv/bin/python scripts/validate_activation_run.py \
  results/final/residual_streams/qwen3_32b_instruct_realization_v1_layer32_regions_float16

./venv/bin/python scripts/build_activation_vectors.py \
  --activation-run results/final/residual_streams/qwen3_32b_instruct_realization_v1_layer32_regions_float16 \
  --layers 32 \
  --output-dir results/final/activation_vectors/qwen3_32b_instruct_realization_v1_layer32

./venv/bin/python scripts/evaluate_activation_vectors.py \
  --activation-run results/final/residual_streams/qwen3_32b_instruct_realization_v1_layer32_regions_float16 \
  --direction results/final/activation_vectors/qwen3_32b_instruct_realization_v1_layer32/mean_direction.npy \
  --layers 32 \
  --output-dir results/final/activation_vectors/qwen3_32b_instruct_realization_v1_layer32/evaluation
```

Important: do not reuse Gemma's activation vector for Qwen. Each model needs its
own direction in its own residual-stream coordinate space.

