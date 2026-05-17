# Results Layout

This folder has three kinds of outputs:

- `results.csv` and `blocks/` are the canonical behavioral realization-effect
  results used by the analysis scripts.
- `final/` is for reference artifacts from larger activation extraction and SAE
  training runs. These runs may still be exploratory, but they should be
  reproducible and tied to checked-in configs.
- `test/` contains disposable smoke-test artifacts used to check formatting,
  storage, and pipeline behavior.

Keep new local smoke runs under `results/test/`. Put the current SAE training
dataset and current reference SAE checkpoints under `results/final/`.

## Current SAE Artifacts

- `final/residual_streams/final_inference_prompts_v1_layer18_regions_float32/`
  is the corrected full activation run for the first generated prompt set.
- `final/sae/final_inference_prompts_v1_layer18_normalized/` is the SAE trained
  on that corrected activation run. Its feature inspection shows strong
  casino-context features, so treat it as an exploratory reference, not the
  final research SAE.
- `final/residual_streams/first_sae_prompt_mix_v1_repeated_5022_layer18_regions_float32/`
  and `final/sae/first_sae_prompt_mix_v1_repeated_5022_layer18_normalized/`
  are earlier larger local runs kept for comparison.
