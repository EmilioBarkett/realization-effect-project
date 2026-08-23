# Current project architecture

**Status:** target architecture for the next implementation phase. The
generalized benchmark described here does not yet exist in code.

## 1. Architectural principle

The repository has one active reusable foundation and one planned benchmark
layer:

```text
active activation primitives + active paired-prompt generator
                              ↓
                    planned construct benchmark
                              ↓
             cross-construct decodability/steerability analysis
```

The old realization behavioral pipeline is a reference implementation and is
archived. It is not the template for new behavioral collection.

## 2. Current versus planned repository

### Active today

- `src/activation_analysis/openrouter_prompt_generation.py`: paired-prompt
  generation foundation;
- `src/activation_analysis/log_residuals.py`: prompt-CSV/config-driven
  activation logging;
- `src/activation_analysis/activation_store.py`: activation-run loading and
  validation;
- `src/activation_analysis/residual_streams.py`: model hooks and residual
  extraction;
- `src/activation_analysis/vector_analysis.py`: vector analysis using the
  tracked activation-store iterator; clean-install verification passes under
  Python 3.11, while real-run validation remains pending;
- `src/activation_analysis/steering.py`: reusable intervention primitive;
- `configs/activation_analysis/` and `experiments/activation_analysis/`;
- active prompt-generation, logging, evaluation, validation, and audit scripts.

The active vector path no longer depends on the absent `sae` package. The
legacy SAE-training tests are archived under `archive/sae/`; the active
iterator and tests pass the clean Python 3.11 install check.

### Planned package

Start with the minimum benchmark core rather than creating a dozen empty
modules:

```text
src/construct_benchmark/
├── schemas.py         Prompt, split, result, and manifest schemas
├── config.py          Versioned config loading and validation
├── manifests.py       Canonical serialization, hashes, and provenance
├── splits.py          Leakage-safe split construction and audits
├── readout.py         Train-only directions and held-out projections
└── calibration.py     Neutral/within-cell dose and outcome-scale adapters
```

Add behavioral adapters, steering orchestration, parsing, metrics, and
correspondence analysis only when the two-construct vertical slice requires
them. The later profile layer can then add:

```text
behavior.py  steering.py  parsing.py  metrics.py
profiles.py  correspondence.py
```

This package is a design target, not an existing module.

## 3. Configuration boundary

Each experiment should use three versioned artifacts:

### `construct_spec`

Stable scientific meaning: construct identity, theory, state definitions,
paired probe contrast, independent task, expected direction, primary outcome,
parsing rules, nuisance variables, lexical-shortcut risks, controls, predicted
nulls, and invalidity criteria.

### `run_config`

Model and intervention settings: model revision, activation site, candidate
layers, token/region mode, position mode, direction estimator, calibration
source, steering doses, timing, generation settings, and seeds.
The calibration source must be neutral or within-condition/within-cell
centered variance.

### `analysis_spec`

Frozen analysis decisions: primary decodability and state-transfer estimands,
secondary slope and collateral outcomes, inclusion rules, uncertainty model,
multiple-comparison policy, and stopping rules.

These schemas are not implemented yet.

## 4. Planned data flow

```text
construct_spec
      ↓
paired prompts + split manifest
      ├───────────────┐
      ↓               ↓
activation logging   independent behavioral baseline
      ↓               ↓
train-only direction downstream task definition
      ├───────────────┤
      ↓               ↓
held-out readout     calibrated steering runs
      └───────────────┘
              ↓
parsing, manipulation checks, quality flags
              ↓
construct summaries and crossed model × construct × task analysis
```

The prompt generator creates reviewable rows first. Activation logging and
behavioral execution consume those frozen rows rather than silently generating
or changing prompts during a run.

The profile and correspondence modules are a later benchmark layer. The first
vertical slice should implement the primary held-out projection and directed
state-transfer estimands before adding stability, dimensionality, normalized
intervention-cost, or out-of-sample profile prediction.

## 5. Planned execution stages

1. Validate versioned construct, run, and analysis configs.
2. Generate paired prompts and freeze canonical hashes.
3. Create and audit leakage-safe splits.
4. Run the prompt-only behavioral baseline.
5. Log activations at registered layers, sites, and positions.
6. Construct the direction from training prompts only.
7. Evaluate continuous standardized held-out projection margins.
8. Calibrate doses from neutral or within-cell training variance and record
   unstandardized magnitudes.
9. Run independent-task steering with explicit timing and positive, zero,
   negative, shuffled, and random controls.
10. Check downstream persistence, output accessibility, compliance, and
    collateral behavior.
11. Parse outputs through construct-specific adapters and compute outcome-
    appropriate estimands.
12. Fit the crossed model × construct × task summary; add profile prediction
    only after the matrix is large enough for out-of-sample evaluation.

## 6. Artifact policy

Track when small and reviewable:

- construct, run, and analysis configs;
- prompt and split manifests;
- overlap and leakage audits;
- run manifests and curated summaries;
- deterministic fixtures and tests.

Keep ignored:

- API keys and environment files;
- model weights;
- raw model generations;
- residual tensors and large activation runs;
- benchmark raw outputs under
  `results/benchmark/<construct>/<model>/<run>/raw/`.

The benchmark raw path must be added to `.gitignore` before the first new run.

## 7. Implementation gates

The next work is not a large experiment. It is:

1. consolidate the current archive and documentation changes into one
   reviewable repository checkpoint;
2. verify `ActivationVectorRecord` and `iter_activation_vectors()` against
   activation-store manifests and keep the active path free of `sae.dataset`;
3. add a representative manifest-backed activation-run smoke fixture and keep
   the iterator/filtering/region/memory-map regression tests green;
4. add the three versioned config schemas and validators;
5. generalize paired-prompt metadata beyond realization fields;
6. implement held-out projection margins and neutral/within-cell calibration;
7. implement the realization/evidence-diagnosticity vertical slice with
   construct-specific task adapters;
8. add timing, manipulation, output-accessibility, and downstream-persistence
   checks;
9. run the precision simulation before expanding the construct count.

Until these gates are passed, this document describes a target architecture,
not a completed system.
