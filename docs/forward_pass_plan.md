# Forward-Pass Plan

This project has two halves:

1. The behavioral realization-effect experiment, which produces
   `results/results.csv`.
2. The forward-pass interpretability workflow, which will log residual stream
   activations for the same prompt set and prepare them for later SAE analysis.

The second half is intentionally still in progress. This note defines the
expected process before the implementation gets larger.

## Inputs

The default input prompt set comes from
`configs/realization_effect/conditions.csv` plus one prompt version:

- `absolute`
- `balance`
- `qualitative`

The extraction script can also accept a custom prompt CSV through
`--prompt-csv`. In that mode the prompt text column defaults to `prompt_text`,
and prompt IDs are read from `--id-column`, `prompt_id`, or generated as
`prompt_00000`, `prompt_00001`, and so on.

## Command

The main entrypoint is:

```bash
./venv/bin/python scripts/log_residual_streams.py \
  --model-id models/gemma-3-4b-pt \
  --layers 12,18 \
  --prompt-version absolute \
  --batch-size 1 \
  --limit 2 \
  --local-files-only \
  --output-dir results/residual_streams/gemma3_4b_smoke
```

The script should support a tiny smoke run before any full extraction. The
smoke run should verify that tokenizer loading, model loading, hook placement,
activation writing, and manifest writing all work for the selected local model.

## Outputs

Each run writes a self-contained directory:

```text
results/residual_streams/<run_name>/
├── prompts.jsonl
├── manifest.json
└── activations/
    └── layer_XX/
        ├── batch_000000.npy
        └── batch_000000.jsonl
```

`prompts.jsonl` stores the exact prompt text and condition metadata.

`manifest.json` records the model, tokenizer, selected layers, batch size,
maximum token length, local-files-only mode, dtype, device, total prompt count,
and activation shard list.

Each `.npy` activation shard is expected to have shape
`[batch, sequence_length, d_model]` and be saved as float32. The paired `.jsonl`
file stores prompt IDs, token IDs, token positions, and prompt metadata for each
row in the tensor.

## Guardrails

- Layer numbers are 1-based transformer block indices.
- Requested layers must be present in the model.
- `prompts` and `prompt_ids` must have the same length.
- The workflow should fail loudly if it cannot resolve transformer blocks.
- Full extraction outputs stay gitignored under `results/residual_streams/`.
- Small tests should cover parsing, prompt metadata loading, manifest shape, and
  shard-writing structure without requiring a real model download.

## Next Implementation Work

Before larger SAE logic, tighten the forward-pass adapter around:

- model architecture support and clearer errors,
- optional extraction of final token positions only,
- deterministic run naming,
- metadata joins back to `results/results.csv`,
- lightweight smoke fixtures for one tiny local or mocked model path.
