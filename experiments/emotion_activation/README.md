# Emotion Probes

This directory holds reviewable prompt material for emotion-vector experiments.

The design follows the Anthropic emotion-vector paper at a smaller scale:
emotion concepts are evoked through short scenarios, paired with matched
controls, and then tested through activation differences. The current workflow
separates general emotion discovery from casino-domain evaluation.

## Files

- `configs/emotion_activation/emotions_general_v2.json` defines the general
  emotion discovery prompts: 8 emotions x 3 variants x positive/control.
- `experiments/emotion_activation/prompts/general_emotion_contrasts_v2.csv` is
  the exported general discovery table.
- `configs/emotion_activation/emotions_initial.json` defines the earlier
  casino-domain contrast pairs for evaluation and sanity checks.
- `experiments/emotion_activation/prompts/initial_emotion_contrasts.csv` is the
  exported casino-domain table.
- `scripts/export_emotion_probes.py` regenerates the prompt CSV from the config.

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
  --output experiments/emotion_activation/prompts/general_emotion_contrasts_v2.csv
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
  --prompt-csv experiments/emotion_activation/prompts/general_emotion_contrasts_v2.csv \
  --layers 12,18 \
  --token-mode nonpad \
  --include-token-regions scenario \
  --storage-dtype float32 \
  --local-files-only
```
