# Final Results

This folder is reserved for future reference generated outputs.

- `residual_streams/` will hold activation runs intended for SAE training or
  evaluation.
- `sae/` will hold reference SAE checkpoints and normalization stats.

The current small SAE and activation artifacts are still smoke/test outputs and
live under `results/test/`.

When we run the larger inference pass, its dataset config should point into this
folder. The current smoke dataset config is:

- `configs/sae/first_sae_prompt_mix_v1_layer18.json`

The current normalized SAE training config is:

- `configs/sae/first_sae_prompt_mix_v1_training_smoke.json`
