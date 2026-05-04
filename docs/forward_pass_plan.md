# Forward-Pass Plan

This project has two halves:

1. The behavioral realization-effect experiment, which produces
   `results/results.csv`.
2. The emotion-activation workflow, which will log residual stream activations,
   estimate emotion vectors, and prepare them for later steering and SAE
   analysis.

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
  --activation-site resid_post \
  --token-mode nonpad \
  --token-region-strategy auto \
  --include-token-regions scenario,decision_question \
  --storage-dtype float16 \
  --batch-size 1 \
  --limit 2 \
  --local-files-only \
  --run-name gemma3_4b_smoke
```

The script should support a tiny smoke run before any full extraction. The
smoke run should verify that tokenizer loading, model loading, hook placement,
activation writing, and manifest writing all work for the selected local model.
If `--output-dir` is omitted, the script writes to
`results/test/residual_streams/<deterministic-run-name>`. The automatic run name is
derived from model, prompt source, selected layers, token mode, and a prompt
fingerprint. Pass `--overwrite` to reuse a non-empty output directory.

## Outputs

Each run writes a self-contained directory:

```text
results/test/residual_streams/<run_name>/
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
resolved transformer block path, activation site, token mode,
behavioral-results source, and activation shard list.

Each `.npy` activation shard is expected to have shape
`[batch, sequence_length, d_model]`. New runs default to float16 storage to
reduce disk use; use `--storage-dtype float32` if full-precision local storage
is needed. The paired `.jsonl` file stores prompt IDs, token IDs, token
positions, token regions, and prompt metadata for each row in the tensor.

The paired metadata can include condition-level behavioral summaries from
`results/results.csv`. This join is enabled by default and can be disabled with
`--no-results-join`.

With `--token-region-strategy auto`, each saved token also receives a region
label in the paired `.jsonl` file. Realization-effect prompts are labeled with
regions such as `scenario`, `decision_question`, and `response_instruction`.
Emotion-probe prompts are labeled with regions such as `wrapper`, `scenario`,
and `processing_instruction`. These labels do not filter the activations; they
make later vector extraction and SAE dataset construction easier to slice.

## Guardrails

- Layer numbers are 1-based transformer block indices.
- Requested layers must be present in the model.
- Hook placement can be forced with `--block-path`, for example
  `--block-path model.layers`.
- `--activation-site resid_post` names the current hook contract: residual
  stream activations after each selected transformer block.
- `--activation-site block_output` is retained as an explicit alias for the
  current hook while the project is still pre-SAE.
- `prompts` and `prompt_ids` must have the same length.
- The workflow should fail loudly if it cannot resolve transformer blocks.
- `--token-mode all` saves every padded sequence position.
- `--token-mode nonpad` saves all non-padding prompt tokens and pads within the
  activation shard only as needed for rectangular tensors.
- `--token-mode final` saves only the final non-padding token for each prompt.
- `--token-region-strategy auto` should be used for research runs so broad
  non-padding activations remain available while region-specific analyses can
  still exclude boilerplate later.
- `--include-token-regions` can be used for SAE-focused extraction runs to
  write only selected region labels, such as `scenario,decision_question`,
  instead of storing answer-format or wrapper tokens.
- `--storage-dtype float16` is the default for new runs. Compute-sensitive
  downstream code should cast loaded vectors to float32 when aggregating or
  training.
- Smoke extraction outputs stay gitignored under `results/test/residual_streams/`.
  Current reference activation datasets should be moved or written under
  `results/final/residual_streams/`.
- Small tests should cover parsing, prompt metadata loading, manifest shape, and
  shard-writing structure without requiring a real model download.
- `scripts/validate_activation_run.py <run_dir>` should pass before using an
  activation run for vector extraction or SAE prep.

## Next Implementation Work

Before larger SAE logic, tighten the forward-pass adapter around:

- choosing the precise activation site for the first serious SAE run,
- model-family-specific block paths for the first target model,
- metadata joins from prompt-level activations to row-level behavioral outputs,
- optional answer-span extraction once generated responses enter the pipeline.
