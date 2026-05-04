# Final Activation Runs

This folder contains activation runs intended to feed SAE training or evaluation
configs.

Current larger local SAE-training run:

- `first_sae_prompt_mix_v1_repeated_5022_layer18_regions_float32`

Properties:

- 5,022 prompts from `experiments/emotion_activation/prompts/first_sae_prompt_mix_v1_repeated_5022.csv`
- layer `18`
- activation site `resid_post`
- token mode `nonpad`
- stored token regions `scenario,decision_question`
- storage dtype `float32`
- 218,674 activation vectors

The matching SAE dataset config is:

- `configs/sae/first_sae_prompt_mix_v1_repeated_5022_layer18.json`

Note: this run repeats the current 81-prompt mix 62 times. It is larger for SAE
pipeline scale, but it does not add semantic prompt diversity.
