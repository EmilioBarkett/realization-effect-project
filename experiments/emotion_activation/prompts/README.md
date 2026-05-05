# Prompt CSV Layout

This folder contains reviewable CSV prompt sets used before residual-stream
inference.

- `final/` is for current final-inference prompt sets generated through
  `scripts/run_realization_experiment.py --prompt-version generation`.
- `test/` is for smoke runs and first-SAE prompt mixes used to validate the
  activation and training pipeline.
- `archive/` is for earlier hand-authored/exported contrast sets that remain
  useful references but are not the default final-inference input. Generated
  pilots and failed/partial API generations live under `archive/generated/`.

## Current Canonical Prompt Files

- `final/final_inference_prompts_v1.csv` is the current casino-heavy generated
  prompt set used for the first full SAE/inference pass. It is useful as a
  reference artifact and as evidence that casino-context features dominate.
- Future non-casino emotion/risk SAE training prompt sets should get a new
  explicit version name rather than overwriting `final_inference_prompts_v1.csv`.
  The first planned name for this direction is `general_emotion_risk_v1.csv`,
  generated from
  `configs/emotion_activation/general_emotion_risk_generation_v1.json`.

Generated activation tensors and SAE checkpoints do not belong here; they live
under ignored `results/test/` or `results/final/`.
