# Emotion Probes

This directory holds reviewable prompt material for emotion-vector experiments.

The initial design follows the Anthropic emotion-vector paper at a smaller
scale: each target emotion has a positive scenario and a matched control
scenario. Positive prompts are written to evoke the emotion mostly implicitly,
while control prompts preserve the broad situation without emphasizing the
target emotion.

## Files

- `configs/emotion_activation/emotions_initial.json` defines the reviewed emotion
  concepts and contrast pairs.
- `experiments/emotion_activation/prompts/initial_emotion_contrasts.csv` is the exported prompt
  table consumed by the forward-pass logger.
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
./venv/bin/python scripts/export_emotion_probes.py
```

Run a small activation extraction:

```bash
./venv/bin/python scripts/log_residual_streams.py \
  --model-id models/gemma-3-4b-pt \
  --emotion-config configs/emotion_activation/emotions_initial.json \
  --layers 12,18 \
  --activation-site resid_post \
  --token-mode final \
  --token-region-strategy auto \
  --storage-dtype float16 \
  --local-files-only \
  --run-name emotion_probe_smoke
```

Validate the output before using it for vector extraction:

```bash
./venv/bin/python scripts/validate_activation_run.py \
  results/residual_streams/emotion_probe_smoke
```

You can also use the exported CSV directly:

```bash
./venv/bin/python scripts/log_residual_streams.py \
  --model-id models/gemma-3-4b-pt \
  --prompt-csv experiments/emotion_activation/prompts/initial_emotion_contrasts.csv \
  --layers 12,18 \
  --token-mode final \
  --local-files-only
```
