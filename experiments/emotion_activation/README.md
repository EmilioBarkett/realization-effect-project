# Emotion Probes

This directory holds reviewable prompt material for emotion-vector experiments.

The design follows the Anthropic emotion-vector paper at a smaller scale:
emotion concepts are evoked through short scenarios, paired with matched
controls, and then tested through activation differences. The current workflow
separates general emotion discovery from casino-domain evaluation.

## Files

- `configs/emotion_activation/emotions_general_v2.json` defines the general
  emotion discovery prompts: 8 emotions x 3 variants x positive/control.
- `experiments/emotion_activation/prompts/archive/general_emotion_contrasts_v2.csv`
  is the exported general discovery table.
- `configs/emotion_activation/emotions_initial.json` defines the earlier
  casino-domain contrast pairs for evaluation and sanity checks.
- `experiments/emotion_activation/prompts/archive/initial_emotion_contrasts.csv`
  is the exported casino-domain table.
- `configs/emotion_activation/final_inference_prompt_generation_v1.json`
  defines the OpenRouter generation plan for the larger final inference prompt
  set, balanced across three source LLMs.
- `scripts/export_emotion_probes.py` regenerates the prompt CSV from the config.
- `scripts/run_realization_experiment.py --prompt-version generation` calls
  OpenRouter and writes the generated final-inference prompt CSV.

## Current Emotion Set

- `regret`
- `frustration`
- `desperation`
- `temptation`
- `anxiety`
- `caution`
- `relief`
- `calm`

## Run

Export prompts:

```bash
./venv/bin/python scripts/export_emotion_probes.py \
  --config configs/emotion_activation/emotions_general_v2.json \
  --output experiments/emotion_activation/prompts/archive/general_emotion_contrasts_v2.csv
```

Run a small activation extraction:

```bash
./venv/bin/python scripts/log_residual_streams.py \
  --model-id models/gemma-3-4b-pt \
  --emotion-config configs/emotion_activation/emotions_general_v2.json \
  --layers 12,18 \
  --activation-site resid_post \
  --token-mode nonpad \
  --token-region-strategy auto \
  --include-token-regions scenario \
  --storage-dtype float32 \
  --local-files-only \
  --run-name general_emotion_v2_smoke
```

Validate the output before using it for vector extraction:

```bash
./venv/bin/python scripts/validate_activation_run.py \
  results/test/residual_streams/general_emotion_v2_smoke
```

You can also use the exported CSV directly:

```bash
./venv/bin/python scripts/log_residual_streams.py \
  --model-id models/gemma-3-4b-pt \
  --prompt-csv experiments/emotion_activation/prompts/archive/general_emotion_contrasts_v2.csv \
  --layers 12,18 \
  --token-mode nonpad \
  --include-token-regions scenario \
  --storage-dtype float32 \
  --local-files-only
```

## Generate Final Inference Prompts

The final prompt generator is config-driven and writes a reviewable CSV with
metadata for source LLM, domain, emotion, risk orientation, casino context,
control type, and contrast role.

```bash
export OPENROUTER_API_KEY=your_key_here

./venv/bin/python scripts/run_realization_experiment.py \
  --prompt-version generation
```

This mode does not write behavioral results to `results/results.csv`. It writes
the generated prompt CSV to
`experiments/emotion_activation/prompts/final/final_inference_prompts_v1.csv`
by default.

Run a small pilot first with one source model alias and a few jobs:

```bash
./venv/bin/python scripts/run_realization_experiment.py \
  --prompt-version generation \
  --generation-pilot-all-cells \
  --generation-output experiments/emotion_activation/prompts/final/final_inference_prompts_v1_pilot.csv
```

In generation mode, `--models` uses aliases from the generation plan, currently
`codex`, `sonnet`, and `flash`. Use `--generation-resume` to append missing
batches to an existing CSV after an interruption.
