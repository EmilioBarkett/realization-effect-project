# SAE Artifact Inventory

This document labels the current SAE-related artifacts so the repo stays
organized while the research direction changes.

## Current Reference

- `experiments/activation_analysis/prompts/final/general_emotion_risk_v1.csv`
  - Status: current generated prompt set for non-casino emotion, risk,
    neutral, and realization-style prompts.
  - Size: 3,060 prompts, balanced across `gpt54`, `sonnet`, and `qwen32`.
  - Why keep: reproducible prompt source for the current activation run.
- `results/final/residual_streams/general_emotion_risk_v1_layer18_regions_float32/`
  - Status: current validated activation run.
  - Contents: layer-18 `resid_post` scenario-token activations stored as fp32.
  - Git status: ignored, because it is a large generated artifact.

## Current SAEs

- `configs/sae/archive/20260506_sae_first_pass/final/general_emotion_risk_v1_layer18.json`
- `configs/sae/archive/20260506_sae_first_pass/final/general_emotion_risk_v1_training.json`
  - Status: checked-in configs for the full general emotion/risk/realization
    SAE.
  - Output: `results/legacy/20260506_sae_first_pass/general_emotion_risk_v1_layer18_normalized/`
  - First read: strongest feature associations are risk-oriented.
- `configs/sae/archive/20260506_sae_first_pass/final/general_emotion_only_v1_layer18.json`
- `configs/sae/archive/20260506_sae_first_pass/final/general_emotion_only_v1_training.json`
  - Status: checked-in configs for the emotion-only comparison SAE.
  - Output: `results/legacy/20260506_sae_first_pass/general_emotion_only_v1_layer18_normalized/`
  - First read: stronger broad emotion-vs-control feature than the full SAE,
    with individual emotion features still weaker.

The SAE output directories contain checkpoints, normalization stats, manifests,
and feature-inspection CSVs. They are ignored by git under `results/final/**`.

## External General SAE Baseline

- `configs/sae/archive/20260506_sae_first_pass/external/gemma_scope_2_4b_pt_layer17_resid_post_16k_l0_small.json`
  - Status: checked-in metadata config for the layer-aligned public Gemma Scope
    SAE baseline.
  - Source: `google/gemma-scope-2-4b-pt`
  - Source path: `resid_post_all/layer_17_width_16k_l0_small`
  - Target model: `google/gemma-3-4b-pt`
  - Hook point: `model.layers.17.output`
  - Project logger alignment: layer `18`, because project logging hooks
    `blocks[layer - 1]`.
- `configs/sae/archive/20260506_sae_first_pass/external/gemma_scope_2_4b_pt_layer18_resid_post_16k_l0_small.json`
  - Status: checked-in metadata config for the adjacent public Gemma Scope SAE
    baseline.
  - Source: `google/gemma-scope-2-4b-pt`
  - Source path: `resid_post_all/layer_18_width_16k_l0_small`
  - Target model: `google/gemma-3-4b-pt`
  - Hook point: `model.layers.18.output`
  - Project logger alignment: layer `19`, because project logging hooks
    `blocks[layer - 1]`.
- `external/archive/20260506_sae_first_pass/saes/gemma_scope_2_4b_pt/layer_18_resid_post_16k_l0_small/`
  - Status: local imported external SAE folder.
  - Tracked files: README/manifest/provider config.
  - Ignored file: `params.safetensors`, because it is a large pretrained
    weight artifact.
- `external/archive/20260506_sae_first_pass/saes/gemma_scope_2_4b_pt/layer_17_resid_post_16k_l0_small/`
  - Status: local imported external SAE folder for the current comparison.
  - Tracked files: manifest/provider config.
  - Ignored file: `params.safetensors`, because it is a large pretrained
    weight artifact.

This external SAE gives us a broad pretrained dictionary to compare against the
project-trained, curated-distribution SAEs, but it should be run on matching
layer-aligned activations. For the current local layer-18 activation data, use
the Gemma Scope layer-17 baseline.

## Prompt Generation

- `configs/activation_analysis/general_emotion_risk_generation_v1.json`
  - Status: current generation plan for the general prompt set.
  - Notes: includes retry, validation, forbidden-term, and model-balance
    settings for OpenRouter prompt generation.
- `configs/activation_analysis/final_inference_prompt_generation_v1.json`
  - Status: older generated-prompt plan retained for provenance.

## Legacy

- `configs/sae/legacy/20260505_previous_sae/`
  - Old SAE dataset/training configs moved out of the current final/test paths.
- `results/legacy/20260505_previous_sae/`
  - Old SAE checkpoints and old activation inputs moved out of current final/test
    paths.
  - Git status: ignored with `results/legacy/**`.
- `experiments/activation_analysis/prompts/archive/`
  - Earlier hand-authored prompt sets and generated pilots.

## Disposable

- `results/test/`
  - Smoke activations and smoke SAE checkpoints.
  - Safe to delete when disk space matters, as long as checked-in configs and
    scripts are kept.
