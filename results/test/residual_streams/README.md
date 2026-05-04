# Test Activation Runs

This folder contains disposable local smoke-test residual-stream runs.

These runs are useful for checking model loading, hook placement, tensor
formatting, token-region labels, storage dtype behavior, and rough disk growth.
They should not be treated as the canonical SAE training dataset.

Notable runs:

- `sae_layers_12_18_format_smoke` tested `float16` storage and showed overflow.
- `sae_layers_12_18_format_smoke_float32` confirmed `float32` storage is finite.
- `sae_layers_12_18_100prompt_smoke_float32` tested larger local storage growth.
- `first_sae_prompt_mix_v1_layers_12_18_float32` is the pre-fix mixed prompt run
  with incorrect realization prompt region labels.
- `first_sae_prompt_mix_v1_layers_12_18_float32_regions_fixed` is the corrected
  81-prompt mixed activation run used for the normalized SAE smoke test.
