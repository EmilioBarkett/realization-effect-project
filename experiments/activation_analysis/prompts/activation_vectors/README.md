# Activation-Vector Prompt CSVs

This folder contains generated synthetic prompt pairs used by the
activation-vector pipeline. The current files are realization-anchor data, not
the final multi-construct benchmark.

Current planned output:

- `realization_vector_v1.csv`
- `realization_vector_heldout_v1.csv`

These CSVs are the prompts that get fed into residual-stream logging. The
OpenRouter generation plans that create them live under
`configs/activation_analysis/`.

The checked-in realization-anchor CSVs are preserved historical generation
artifacts and retain their original split labels. They are not canonical
multi-construct benchmark inventories; new inventories must use the canonical
`direction_train`, `direction_validation`, and `direction_heldout` names.

Held-out prompts should be generated with:

```bash
./venv/bin/python scripts/generate_heldout_activation_prompts.py --dry-run
./venv/bin/python scripts/generate_heldout_activation_prompts.py
```

The held-out generator currently uses DeepSeek-authored prompts and runs
`scripts/audit_prompt_overlap.py` locally after generation to flag any
near-duplicate prompts without uploading the original prompt set to an LLM.
