# Test SAE Runs

This folder contains disposable SAE training smoke checkpoints.

These runs verify that checkpoints and manifests can be written and loaded, but
they are not intended for interpretation.

Current test artifact:

- `first_sae_prompt_mix_v1_layer18_smoke`: trained on raw, unnormalized
  activations before mean-centering/global-norm scaling was added.
- `first_sae_prompt_mix_v1_layer18_normalized_smoke`: trained on the corrected
  81-prompt activation run with mean-centering/global-norm scaling.
