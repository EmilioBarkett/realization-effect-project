# SAE Artifact Inventory

This document labels the current SAE-related artifacts so the repo stays
organized while the research direction changes.

## Current Reference

- `experiments/emotion_activation/prompts/final/general_emotion_risk_v1.csv`
  - Status: current generated prompt set for non-casino emotion, risk,
    neutral, and realization-style prompts.
  - Size: 3,060 prompts, balanced across `gpt54`, `sonnet`, and `qwen32`.
  - Why keep: reproducible prompt source for the current activation run.
- `results/final/residual_streams/general_emotion_risk_v1_layer18_regions_float32/`
  - Status: current validated activation run.
  - Contents: layer-18 `resid_post` scenario-token activations stored as fp32.
  - Git status: ignored, because it is a large generated artifact.

## Current SAEs

- `configs/sae/final/general_emotion_risk_v1_layer18.json`
- `configs/sae/final/general_emotion_risk_v1_training.json`
  - Status: checked-in configs for the full general emotion/risk/realization
    SAE.
  - Output: `results/final/sae/general_emotion_risk_v1_layer18_normalized/`
  - First read: strongest feature associations are risk-oriented.
- `configs/sae/final/general_emotion_only_v1_layer18.json`
- `configs/sae/final/general_emotion_only_v1_training.json`
  - Status: checked-in configs for the emotion-only comparison SAE.
  - Output: `results/final/sae/general_emotion_only_v1_layer18_normalized/`
  - First read: stronger broad emotion-vs-control feature than the full SAE,
    with individual emotion features still weaker.

The SAE output directories contain checkpoints, normalization stats, manifests,
and feature-inspection CSVs. They are ignored by git under `results/final/**`.

## Prompt Generation

- `configs/emotion_activation/general_emotion_risk_generation_v1.json`
  - Status: current generation plan for the general prompt set.
  - Notes: includes retry, validation, forbidden-term, and model-balance
    settings for OpenRouter prompt generation.
- `configs/emotion_activation/final_inference_prompt_generation_v1.json`
  - Status: older generated-prompt plan retained for provenance.

## Legacy

- `configs/sae/legacy/20260505_previous_sae/`
  - Old SAE dataset/training configs moved out of the current final/test paths.
- `results/legacy/20260505_previous_sae/`
  - Old SAE checkpoints and old activation inputs moved out of current final/test
    paths.
  - Git status: ignored with `results/legacy/**`.
- `experiments/emotion_activation/prompts/archive/`
  - Earlier hand-authored prompt sets and generated pilots.

## Disposable

- `results/test/`
  - Smoke activations and smoke SAE checkpoints.
  - Safe to delete when disk space matters, as long as checked-in configs and
    scripts are kept.
