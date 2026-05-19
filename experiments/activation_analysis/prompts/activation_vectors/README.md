# Activation-Vector Prompt CSVs

This folder is for generated synthetic prompts used by the Anthropic-style
activation-vector pipeline.

Current planned output:

- `realization_vector_v1.csv`
- `realization_vector_heldout_v1.csv`

These CSVs are the prompts that get fed into residual-stream logging. The
OpenRouter generation plans that create them live under `configs/activation_analysis/`.

Held-out prompts should be generated with:

```bash
./venv/bin/python scripts/generate_heldout_activation_prompts.py --dry-run
./venv/bin/python scripts/generate_heldout_activation_prompts.py
```

The held-out generator currently uses DeepSeek-authored prompts and runs
`scripts/audit_prompt_overlap.py` locally after generation to flag any
near-duplicate prompts without uploading the original prompt set to an LLM.
