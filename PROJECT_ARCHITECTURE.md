# Current project architecture

**Status:** the shared multi-construct control plane, 16-construct registry,
Wave 1 synthetic-prompt path, and environment-independent measurement core are
implemented. Real-model execution and manipulation-check validation remain.

## 1. Architectural principle

The repository has one active reusable foundation and one active benchmark
control-plane layer:

```text
active activation primitives + active paired-prompt generator
                              ↓
          construct registry/specs + generation plans + shared run plan
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
  canonical prompt records, split validation, registry validation, generic
  generation, and shared-activation run plans;
- `scripts/validate_construct_run.py`: validates a combined inventory and
  emits a construct-namespaced execution manifest;
- `scripts/validate_construct_registry.py`: validates the frozen 16-entry
  registry against loaded construct specifications;
- `scripts/generate_construct_prompts.py`: generic Wave 1 generation CLI with
  dry-run and mock-friendly execution paths;
- `configs/construct_benchmark/`: the 16-entry registry, four specified Wave 1
  construct specs, four generation plans, smoke config, and analysis spec;
- `configs/activation_analysis/` and `experiments/activation_analysis/`;
- active prompt-generation, logging, evaluation, validation, and audit scripts.

The active vector path no longer depends on the absent `sae` package. The
legacy SAE-training tests are archived under `archive/sae/`; the active
iterator and tests pass the clean Python 3.11 install check.

### Measurement package

The registry and generation layer now sit alongside the existing control-plane
modules. The measurement layer should still be added incrementally rather than
creating a dozen empty modules:

```text
src/construct_benchmark/
├── schemas.py         Implemented construct/run/analysis schemas
├── config.py          Implemented versioned config loading and validation
├── prompts.py         Implemented canonical prompt inventory format
├── manifests.py       Implemented run plans, hashes, and provenance
├── splits.py          Implemented split coverage helpers
├── registry.py        Implemented 16-construct registry validation
├── generation.py      Implemented generic canonical prompt generation
├── readout.py         Implemented train-only directions and held-out projections
├── calibration.py     Implemented neutral/within-condition projection scales
├── behavior.py        Implemented strict Wave 1 parsing and primary effect metric
└── steering.py        Implemented condition plans and control directions
```

The remaining measurement layer should add manipulation checks, bootstrap
uncertainty, prompt-only behavior composition, and correspondence analysis as
the two-construct vertical slice requires them. The later profile layer can add:

```text
profiles.py  correspondence.py
```

The registry, generation, prompt-validation, readout, calibration, behavior,
and steering-plan modules are the current Wave 1 control path. Their numerical
logic is fixture-tested, but the model-side runner has not been validated on a
representative activation run or GPU environment.

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
16-entry registry → specified Wave 1 construct specs → generation plans
      ↓
one combined prompt inventory + split/leakage validation
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

The benchmark-facing generator creates reviewable canonical rows first.
Activation logging and behavioral execution consume those frozen rows rather
than silently generating or changing prompts during a run. The legacy
realization-only generator remains an archived-compatible activation-analysis
utility and is not the multi-construct generation interface.

The profile and correspondence modules are a later benchmark layer. The first
vertical slice should implement the primary held-out projection and directed
state-transfer estimands before adding stability, dimensionality, normalized
intervention-cost, or out-of-sample profile prediction.

## 5. Planned execution stages

1. Validate the registry, versioned construct, generation, run, and analysis configs.
2. Review and, after explicit approval, generate one combined Wave 1 prompt
   inventory and freeze canonical hashes.
3. Audit leakage-safe splits for every construct namespace.
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

1. review the no-API Wave 1 dry-run summary and generation plans;
2. validate the combined generated inventory against a representative
   manifest-backed activation fixture when one is available;
3. validate held-out projection margins and neutral/within-condition calibration
   on a representative activation run;
4. validate the Wave 1 task adapters, beginning with the realization/evidence
   engineering slice while retaining source-reliability and persistence cells;
5. add timing, manipulation, output-accessibility, and downstream-persistence
   checks;
6. run the precision simulation before advancing to Waves 2–4 or fitting
   representation-profile predictors.

The shared control plane is implemented, but the benchmark is not yet an
end-to-end experiment runner.
