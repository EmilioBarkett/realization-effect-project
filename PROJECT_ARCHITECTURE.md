# Current project architecture

**Status:** the shared multi-construct control plane is implemented; readout,
calibration, and behavioral/steering adapters remain the next implementation
phase.

## 1. Architectural principle

The repository has one active reusable foundation and one active benchmark
control-plane layer:

```text
active activation primitives + active paired-prompt generator
                              ↓
          construct specs + shared multi-construct run plan
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
- `src/construct_benchmark/`: versioned construct/run/analysis schemas,
  canonical prompt records, split validation, and shared-activation run plans;
- `scripts/validate_construct_run.py`: validates a combined inventory and
  emits a construct-namespaced execution manifest;
- `configs/construct_benchmark/`: initial realization and evidence-
  diagnosticity specs, two-construct smoke config, and analysis spec;
- `configs/activation_analysis/` and `experiments/activation_analysis/`;
- active prompt-generation, logging, evaluation, validation, and audit scripts.

The active vector path no longer depends on the absent `sae` package. The
legacy SAE-training tests are archived under `archive/sae/`; the active
iterator and tests pass the clean Python 3.11 install check.

### Remaining benchmark package

Start with the minimum benchmark core rather than creating a dozen empty
modules:

```text
src/construct_benchmark/
├── schemas.py         Implemented construct/run/analysis schemas
├── config.py          Implemented versioned config loading and validation
├── prompts.py         Implemented canonical prompt inventory format
├── manifests.py       Implemented run plans, hashes, and provenance
├── splits.py          Implemented split coverage helpers
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

The first five modules are the current control plane. Readout, calibration,
behavior, steering, parsing, and correspondence modules remain deliberately
unimplemented until the two-construct smoke path is reviewed.

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

The schemas are implemented as JSON-friendly dataclasses with explicit
validation and content-hash support in the run manifest. The canonical prompt
inventory requires `construct_id` on every row, so a combined run cannot
silently lose construct identity.

## 4. Shared and construct-scoped data flow

```text
construct specs (2–4 or more)
      ↓
one combined prompt inventory + split validation
      ↓
one shared activation logging pass
      ├──────────────────────────────────────────────┐
      ↓                                              ↓
construct A namespace                             construct B namespace
train-only direction                               train-only direction
held-out readout                                   held-out readout
calibration + steering                            calibration + steering
independent behavior task                          independent behavior task
      └──────────────────────────────────────────────┘
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
2. Generate one combined prompt inventory and freeze canonical hashes.
3. Create and audit leakage-safe splits for every construct namespace.
4. Run the prompt-only behavioral baseline per construct.
5. Log activations once at registered layers, sites, and positions.
6. Construct one direction per construct from training prompts only.
7. Evaluate continuous standardized held-out projection margins per construct.
8. Calibrate doses from neutral or within-cell training variance per construct and record
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

The benchmark raw path is already covered by `.gitignore` before the first new
run.

## 7. Implementation gates

The next work is not a large experiment. It is:

1. validate the new plan against a representative manifest-backed activation
   fixture when one is available;
2. connect prompt generation adapters to the canonical combined inventory;
3. implement held-out projection margins and neutral/within-cell calibration;
4. implement the realization/evidence-diagnosticity vertical slice with
   construct-specific task adapters;
5. add timing, manipulation, output-accessibility, and downstream-persistence
   checks;
6. run the precision simulation before expanding the construct count.

The shared control plane is implemented, but the benchmark is not yet an
end-to-end experiment runner.
