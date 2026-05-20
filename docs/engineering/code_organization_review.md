# Code Organization Review

This is a short engineering review of where the repo has started to sprawl and
what should be split next.

## Completed Cleanup

1. Split prompt generation out of `src/realization_effect/runner.py`.
   - OpenRouter prompt-generation routing now lives in
     `src/activation_analysis/generation_cli.py`.

2. Move feature inspection logic from `scripts/inspect_sae_features.py` into
   `src/sae/inspection.py`.
   - `scripts/inspect_sae_features.py` is now a thin CLI wrapper.

3. Separate external/pretrained SAE support from project-trained SAE support.
   - Gemma Scope config parsing and JumpReLU loading now live in
     `src/sae/external.py`.

4. Archive the SAE-first research pass.
   - SAE configs, docs, local outputs, and Gemma Scope imports are preserved
     under dated archive/legacy folders.

5. Split the behavioral experiment runner into focused modules.
   - Prompt construction now lives in `src/realization_effect/prompts.py`.
   - Response parsing now lives in `src/realization_effect/parsing.py`.
   - OpenRouter API helpers now live in `src/realization_effect/api.py`.
   - Block CSV merge/resume logic now lives in `src/realization_effect/blocks.py`.
   - Experiment execution now lives in `src/realization_effect/orchestration.py`.
   - `src/realization_effect/runner.py` remains the CLI entrypoint and compatibility
     import surface.

## Archived Imported External SAE

- Source: `google/gemma-scope-2-4b-pt`
- Aligned path: `resid_post_all/layer_17_width_16k_l0_small`
- Aligned local metadata: `configs/sae/archive/20260506_sae_first_pass/external/gemma_scope_2_4b_pt_layer17_resid_post_16k_l0_small.json`
- Adjacent path: `resid_post_all/layer_18_width_16k_l0_small`
- Local weights: ignored `params.safetensors` under `external/archive/20260506_sae_first_pass/saes/`

Layer alignment note: project residual logging uses user-facing 1-based layer
numbers and hooks `blocks[layer - 1]`. The imported Gemma Scope layer-17 SAE
targets `model.layers.17.output`, so it aligns with the existing project
logger layer-18 activation run.
