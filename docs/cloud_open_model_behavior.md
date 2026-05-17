# Cloud Open-Model Behavior Runs

This note describes the clean process for rerunning the behavior-evaluation
prompts on a larger open model such as Qwen or Llama on cloud compute.

## Goal

Run the same `behavior_eval` prompt rows through a larger open-source model
without changing the prompt set, then compare paper/open versus realized/closed
wager and risk outputs.

These runs are exploratory behavior probes, not canonical final artifacts.
Qwen 32B and Qwen3.5 397B did not produce a strong matched-pair behavior
effect, so the active next step is local activation steering rather than more
model scaling. Use this document only when running an additional behavior probe
or validating a new prompt format.

## Inputs

- Prompt CSV:
  `experiments/activation_analysis/prompts/activation_vectors/realization_vector_v1.csv`
- Prompt split:
  `behavior_eval`
- Prompt count:
  `648` rows, `324` matched pairs

Before any cloud run, validate the prompt set:

```bash
./venv/bin/python scripts/validate_behavior_prompts.py --fail-on-issues
```

The validation checks template use, answer instructions, pair structure,
metadata consistency, duplicate prompt IDs, and obvious experiment-label leaks.

## Output Layout

Each exploratory model/run should write to its own ignored test directory:

```text
results/test/activation_vectors/behavior_runs/<run_name>/
├── manifest.json
├── behavior_eval.csv
├── behavior_eval_summary.json
└── behavior_pair_deltas.csv
```

Do not overwrite another model's behavior file. The run directory is the unit of
comparison. Promote only stable reference artifacts into `results/final/`.

## Recommended Cloud Command

For an OpenRouter instruction model with reasoning disabled at the API level:

```bash
./venv/bin/python scripts/run_behavior_vector_eval.py \
  --backend openrouter \
  --model-id qwen/qwen3.5-397b-a17b \
  --prompt-format chat \
  --openrouter-reasoning-effort none \
  --run-name qwen3_5_397b_a17b_behavior_probe \
  --output-dir results/test/activation_vectors/behavior_runs \
  --max-new-tokens 32 \
  --min-new-tokens 1
```

For local or self-hosted Hugging Face models, use `--backend transformers` and
the relevant `--device-map`, `--dtype`, and `--attn-implementation` settings.

For a first cloud smoke test:

```bash
./venv/bin/python scripts/run_behavior_vector_eval.py \
  --model-id Qwen/Qwen3-32B-Instruct \
  --prompt-format chat \
  --device-map auto \
  --dtype bf16 \
  --trust-remote-code \
  --limit 12 \
  --run-name qwen3_32b_instruct_behavior_smoke \
  --output-dir results/test/activation_vectors/behavior_runs
```

Then inspect validity before running the full pass:

```bash
./venv/bin/python scripts/analyze_behavior_vector_eval.py \
  --input results/test/activation_vectors/behavior_runs/qwen3_32b_instruct_behavior_smoke/behavior_eval.csv \
  --summary-output results/test/activation_vectors/behavior_runs/qwen3_32b_instruct_behavior_smoke/behavior_eval_summary.json \
  --pair-output results/test/activation_vectors/behavior_runs/qwen3_32b_instruct_behavior_smoke/behavior_pair_deltas.csv
```

## Analysis

After a full exploratory run:

```bash
./venv/bin/python scripts/reparse_behavior_vector_eval.py \
  --input results/test/activation_vectors/behavior_runs/qwen3_32b_instruct_behavior_v1/behavior_eval.csv

./venv/bin/python scripts/analyze_behavior_vector_eval.py \
  --input results/test/activation_vectors/behavior_runs/qwen3_32b_instruct_behavior_v1/behavior_eval.csv \
  --summary-output results/test/activation_vectors/behavior_runs/qwen3_32b_instruct_behavior_v1/behavior_eval_summary.json \
  --pair-output results/test/activation_vectors/behavior_runs/qwen3_32b_instruct_behavior_v1/behavior_pair_deltas.csv
```

For behavior-only cloud runs, projection fields are allowed to be blank. Once
we later log residual activations for the same larger model, we can compute
model-specific activation projections and rerun the analysis with projection
metadata.

## Decision Rule

Use any new cloud behavior result as a diagnostic, not as the primary rescue
path:

- Stronger paper/realized wager or risk deltas: proceed to residual logging for
  the same model.
- Weak or noisy behavior again: tighten behavior prompts further before logging
  expensive activations.

Given the current Qwen 32B and Qwen3.5 397B results, the preferred next test is
activation steering with the existing Gemma realization direction.
