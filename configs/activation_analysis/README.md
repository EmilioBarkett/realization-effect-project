# Activation-analysis generation configs

This folder contains prompt-generation plans and older contrast definitions for
residual-stream interpretability work.

The activation-analysis prompt generator is intentionally kept active. Its
current plans are realization-focused, but the generator will be adapted to
emit construct-specific paired contrasts for the broader benchmark.

Current checked-in plan:

- `realization_vector_generation_v1.json`
  - mode: `paired_contrast`
  - source models: `gpt54`, `sonnet`, `grok_fast`
    (`x-ai/grok-4-fast`)
  - default output:
    `experiments/activation_analysis/prompts/activation_vectors/realization_vector_v1.csv`
  - purpose: generate matched paper/open vs realized/closed synthetic prompts
    for direction building, validation, and behavioral evaluation.
- `realization_vector_heldout_generation_v1.json`
  - mode: `paired_contrast`
  - source model: `deepseek_v32`
  - default output:
    `experiments/activation_analysis/prompts/activation_vectors/realization_vector_heldout_v1.csv`
  - purpose: generate unused DeepSeek-authored held-out prompts for readout and
    behavior-evaluation checks after a train-only vector is frozen.
  - held-out cells use gain/loss contrasts only; neutral cells are omitted to
    avoid ambiguous no-change versus gain/loss prompt wording.

These plans are the realization anchor for the broader decodability–steerability
benchmark. They are not the generalized benchmark itself. New construct plans
should preserve the paired-prompt, explicit-metadata, train/held-out split
contract rather than adding another realization-specific generator.
