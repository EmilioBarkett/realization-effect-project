# Final Residual Stream Runs

This folder contains reference residual activation runs used by downstream
activation-vector analysis.

Current active target for the next run:

- prompts:
  `experiments/activation_analysis/prompts/activation_vectors/realization_vector_v1.csv`
- expected output:
  `results/final/residual_streams/realization_vector_v1_layer18_regions_float32/`

Cloud/open-model runs should use model-specific directory names and include the
model, prompt set, layer, token regions, and storage dtype, for example:

```text
qwen3_32b_instruct_realization_v1_layer32_regions_float16/
```

Every run must validate with:

```bash
./venv/bin/python scripts/validate_activation_run.py <run_dir>
```

Archived SAE-first activation runs remain local/ignored or live under
`results/legacy/`.
