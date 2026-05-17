# Emotion/Activation Generation Configs

This folder contains prompt-generation plans and older contrast definitions for
residual-stream interpretability work.

Current activation-vector plan:

- `realization_vector_generation_v1.json`
  - mode: `paired_contrast`
  - source models: `gpt54`, `sonnet`, `grok_fast`
    (`x-ai/grok-4-fast`)
  - default output:
    `experiments/activation_analysis/prompts/activation_vectors/realization_vector_v1.csv`
  - purpose: generate matched paper/open vs realized/closed synthetic prompts
    for direction building, validation, and behavioral evaluation.

Archived SAE-oriented generation plans are kept here for provenance, but the
active research direction is activation-vector analysis.
