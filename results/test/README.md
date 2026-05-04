# Test Results

This folder contains disposable generated outputs.

- `residual_streams/` holds activation extraction smoke runs.
- `sae/` holds SAE training smoke runs.

These artifacts are useful for checking that the pipeline works, but they should
not be used as the current reference training data unless a config explicitly
points to them.

The current 81-prompt activation run and normalized test SAE are deliberately in
this folder because they are pipeline validation artifacts, not final research
outputs.
