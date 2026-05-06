# External SAEs

This folder is for pretrained third-party SAE artifacts used for comparison
against project-trained SAEs.

Tracked files should be small metadata files such as `README.md`, manifests, and
provider configs. Large weight files, including `*.safetensors`, are ignored by
the repo-level `.gitignore`.

## Current Import

- `gemma_scope_2_4b_pt/layer_17_resid_post_16k_l0_small/`
  - Source: `google/gemma-scope-2-4b-pt`
  - Hugging Face path:
    `resid_post_all/layer_17_width_16k_l0_small/`
  - Target model: `google/gemma-3-4b-pt`
  - Hook/site: `model.layers.17.output`.
  - Project alignment: our residual logger uses user-facing 1-based layer
    numbers and hooks `blocks[layer - 1]`, so this external hook aligns with
    the existing project logger layer `18` activation run.
  - Width: `16384`
  - Architecture: `jump_relu`
  - Expected local weight file:
    `resid_post_all/layer_17_width_16k_l0_small/params.safetensors`

- `gemma_scope_2_4b_pt/layer_18_resid_post_16k_l0_small/`
  - Source: `google/gemma-scope-2-4b-pt`
  - Hugging Face path:
    `resid_post_all/layer_18_width_16k_l0_small/`
  - Target model: `google/gemma-3-4b-pt`
  - Hook/site: `model.layers.18.output`.
  - Project alignment: our residual logger uses user-facing 1-based layer
    numbers and hooks `blocks[layer - 1]`, so this external hook aligns with
    project logger layer `19`, not the existing project layer-18 activation
    runs.
  - Width: `16384`
  - Architecture: `jump_relu`
  - Expected local weight file:
    `resid_post_all/layer_18_width_16k_l0_small/params.safetensors`

The imported weights are local-only. Re-download with:

```bash
./venv/bin/hf download google/gemma-scope-2-4b-pt \
  resid_post_all/layer_17_width_16k_l0_small/config.json \
  resid_post_all/layer_17_width_16k_l0_small/params.safetensors \
  --local-dir external/saes/gemma_scope_2_4b_pt/layer_17_resid_post_16k_l0_small

./venv/bin/hf download google/gemma-scope-2-4b-pt \
  resid_post_all/layer_18_width_16k_l0_small/config.json \
  resid_post_all/layer_18_width_16k_l0_small/params.safetensors \
  --local-dir external/saes/gemma_scope_2_4b_pt/layer_18_resid_post_16k_l0_small
```
