# SAE Configs

This folder stores small, reviewable JSON configs for selecting activation runs
and training local SAEs. It should not contain trained model weights or large
generated artifacts.

## Layout

- `templates/` contains starter dataset/training configs to copy when creating
  a new run.
- `test/` contains disposable smoke-test and first-SAE configs.
- `final/` contains current reference-run configs that point at the larger
  activation datasets under `results/final/residual_streams/`.
- `legacy/` contains old checked-in SAE configs moved out of the current path.

Generated SAE outputs should go under ignored `results/final/sae/` for current
reference runs or `results/test/sae/` for disposable smoke runs.

Dataset configs may include `prompt_metadata_filters` to train on only selected
prompt families from an activation run. This lets us generate emotion, risk, and
realization prompts together, then train one SAE with realization held out and a
comparison SAE with realization included.

## Current Runs

- `final/general_emotion_risk_v1_layer18.json` and
  `final/general_emotion_risk_v1_training.json` train the current full
  general emotion/risk/realization SAE.
- `final/general_emotion_only_v1_layer18.json` and
  `final/general_emotion_only_v1_training.json` train the emotion-only
  comparison SAE on `emotion_positive` plus `emotion_control` prompt families.

## Example

```bash
./venv/bin/python scripts/train_sae.py \
  --dataset-config configs/sae/final/general_emotion_risk_v1_layer18.json \
  --training-config configs/sae/final/general_emotion_risk_v1_training.json
```
