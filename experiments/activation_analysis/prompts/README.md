# Prompt CSV Layout

This folder contains generated or hand-authored prompt CSVs used for activation
logging. Prompt files are reviewable inputs, not hidden runtime state.

- `activation_vectors/` is the active home for paired prompt CSVs. The current
  files use realization as the anchor construct; future files will cover the
  broader construct set.
- `archive/20260506_sae_first_pass/final/` contains earlier generated prompt
  sets from the SAE-first pass.
- `archive/20260506_sae_first_pass/test/` contains SAE smoke-run prompt sets.
- `archive/` also contains earlier hand-authored/exported emotion contrast
  sets.

Generated activation tensors do not belong here; they live under
`results/test/residual_streams/` or `results/final/residual_streams/`.
