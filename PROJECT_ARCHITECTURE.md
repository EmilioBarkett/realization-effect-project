# Current project architecture

**Status:** the shared multi-construct control plane, 16-construct registry,
all-16 construct specifications and paired-vector plans, calibration-aware
downstream prompt-generation workflow, and environment-independent measurement
core are implemented. Scalar injection/downstream trace instrumentation,
prompt-only baseline execution, fail-closed tokenizer preflight, scoring, and
the C1 matched-episode residual-interchange runner are also implemented;
real-model validation, all-16 downstream behavior parsing, and
output-accessibility/collateral checks remain.

## 1. Architectural principle

The repository has one active reusable foundation and one active benchmark
control-plane layer:

```text
active activation primitives + active paired-prompt generator
                              ↓
          construct registry/specs + vector generation plans + shared run plan
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
  extraction with no-truncation token preflight; the logger also exposes an
  opt-in all-layer, residual-only trace for later mechanistic analyses;
- `src/activation_analysis/causal_patching.py`: matched-episode residual
  interchange at a tokenizer-verified induction/task boundary, with
  bidirectional and same-condition controls;
- `src/activation_analysis/tokenization.py`: shared prompt formatting,
  no-truncation length inspection, and fail-closed padded encoding;
- `src/activation_analysis/vector_analysis.py`: vector analysis using the
  tracked activation-store iterator; clean-install verification passes under
  Python 3.11, while real-run validation remains pending;
- `src/activation_analysis/steering.py`: reusable intervention primitive;
- `src/construct_benchmark/`: versioned construct/run/analysis schemas,
  canonical prompt records, split validation, registry validation, generic
  generation, shared-activation run plans, and portable storage/archive
  helpers, prompt-only behavior baselines, and variation gates;
- `scripts/validate_construct_run.py`: validates a combined inventory and
  emits a construct-namespaced execution manifest;
- `scripts/validate_construct_registry.py`: validates the frozen 16-entry
  registry against loaded construct specifications;
- `scripts/prepare_benchmark_run.py` and
  `scripts/finalize_benchmark_run.py`: workspace preparation, checksums, and
  optional S3-compatible archival;
- `scripts/generate_construct_prompts.py`: generic plan-level generation CLI
  with dry-run and mock-friendly execution paths;
- `scripts/generate_all_vector_prompts.py`: four-worker, vector-only
  review/full orchestrator for the frozen paired inventory;
- `scripts/audit_vector_pairs.py`: structural pair/leakage audit entrypoint;
- `scripts/run_fake_benchmark.py`: deterministic no-API vertical-slice smoke
  runner;
- `scripts/run_prompt_only_behavior.py` and
  `scripts/score_prompt_only_behavior.py`: independent-task baseline execution,
  manifest validation, strict parsing, and prompt-only variation gating;
- `scripts/preflight_tokenizer.py`: model-tokenizer length preflight that
  refuses silent truncation;
- `scripts/run_residual_interchange.py` and
  `scripts/score_residual_interchange.py`: manifest-backed C1 causal
  interchange execution, fail-closed validation, and summary;
- `agents/STEERING_MANIPULATION_CHECKS.md`: model-side trace and output
  contract for injection and downstream manipulation checks;
- `configs/construct_benchmark/`: the 16-entry registry, all 16 specified
  construct specs and paired generation plans, smoke config, and analysis spec;
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
├── behavior_baseline.py Implemented manifest-backed prompt-only baseline
├── behavioral_variation.py Implemented prompt-only and zero-dose gates
├── steering.py        Implemented condition plans and control directions
├── manipulation.py    Implemented scalar injection and downstream checks
├── uncertainty.py      Implemented pair/item bootstrap interval primitives
├── fake.py             Implemented deterministic no-model measurement fixtures
└── storage.py          Implemented workspace, checksum, and archive helpers
```

The remaining measurement layer should add real-run uncertainty orchestration,
output-accessibility/collateral checks, and correspondence analysis as the
two-construct vertical slice requires them. Prompt-only composition is now
implemented as a separate baseline stage; it still needs real-model
validation. C1 residual interchange is the first causal-pathway method; C2
temporal tracing, C3 component/path patching, and C4 ablation remain later
extensions and must not be inferred from the C1 output.
The later profile layer can add:

```text
profiles.py  correspondence.py
```

The registry, generation, prompt-validation, readout, calibration, behavior,
prompt-only baseline, steering-plan, uncertainty, manipulation, and
fake-fixture modules are the current control path. Completed vector/probe and
downstream prompt inventories are engineering artifacts, not empirical
benchmark results; their model-side runner and downstream behavior parsers
have not been validated on a representative benchmark activation run or GPU
environment. A separate realization real-model decode pilot is retained as a
reference artifact, not as generalized benchmark or steering evidence.

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
16-entry registry → specified candidate construct specs → vector generation plans
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

The benchmark-facing generator creates reviewable canonical rows first. The
vector orchestrator is Sonnet 4.6 only, uses four workers, and freezes 100
train, 40 validation, and 40 held-out pairs per construct (2,880 pairs and
5,760 records in the full 16-construct inventory). The completed inventory is
tracked at
`results/benchmark/vector_prompts_v2_luna/full_final_all16/combined.csv` and
its final manifest. It is vector/probe-only, explicitly non-confirmatory, and
does not include downstream single prompts, behavior prompts, or calibration
prompts. The completed Wave 2–4 downstream engineering inventory is retained at
`results/benchmark/downstream_prompts_v1_waves2_4_full_luna_b20_o30k_v1/`;
it is a 384-record, non-confirmatory Luna engineering artifact. The audit
found prompt-wrapper, downstream-episode, direct-cue, and task-independence
blockers in those inputs; the findings are recorded in
`agents/WAVES2_4_PROMPT_AUDIT.md`. Repaired, audited prompt-input releases are
present under
`results/benchmark/prompt_inventories/wave[2-4]_four_construct_confirmatory_v1/`,
but they do not release model execution or empirical results. Neither the
source inventory nor the audit is a real-model benchmark result.

The current Wave 1 downstream inventory is retained at
`results/benchmark/downstream_prompts_v1_wave1_full_luna_current_b50_o60k_v3/`.
The current repaired Wave 1 model input is
`results/benchmark/prompt_inventories/wave1_repaired_v2_full_openai_luna_normalized/combined.csv`.
Its manifest records 1,824 frozen engineering rows: 1,440 vector/probe rows
and 384 independent behavior, steering, and calibration rows. It remains
non-confirmatory until model-side validation is complete. The older 1,650-row
composition remains historical engineering provenance and must not be mixed
with the repaired v2 inventory.

The Waves 2–4 campaign follows the same boundary in three wave-scoped
packages: one four-construct shared activation run for each wave, with
construct-specific directions, calibration, behavior, and steering fan-out.
The no-API composer, prompt-release command, and execution validator are
implemented. They allow the 72-record engineering smoke selections and
recognize the released prompt inputs, but refuse full mode while the Wave 1
measurement and precision prerequisites remain pending.
Activation logging and behavioral execution consume those frozen rows rather
than silently generating or changing prompts during a run. The legacy
realization-only generator remains an archived-compatible activation-analysis
utility and is not the multi-construct generation interface.

The profile and correspondence modules are a later benchmark layer. The first
vertical slice should implement the primary held-out projection and directed
state-transfer estimands before adding stability, dimensionality, normalized
intervention-cost, or out-of-sample profile prediction.

### Causal pathway layer

The causal layer is intentionally separated from additive steering:

```text
matched positive/negative episodes
        ├── common downstream task
        ├── source residual capture at boundary
        └── bidirectional receiver patch at registered layer
                    ↓
       baseline versus patched downstream output
```

The first implementation is `C1` matched residual interchange. It patches the
last complete induction token during prefill only, then lets the identical
downstream task run without further intervention. The unit of analysis is a
donor/receiver/layer observation, with same-condition controls and an adjacent
manifest. This tests contextual causal sufficiency; it does not establish
necessity, a unique circuit, or a general policy parameter. The full method
specification and later-method boundary are in
`agents/CAUSAL_PATHWAY_ARCHITECTURE.md`.

## 4.1 Staged run-mode architecture

Prompt generation is deliberately completed before model-side execution. Each
generation plan has two named modes: `review` emits one pair per paired cell
for human inspection, while `full` emits the complete frozen vector inventory.
Both can be expanded with a no-API dry run. A review inventory is never
silently promoted to confirmatory data.

Transport retries are limited to two identical repeats of a failed request;
they never regenerate content or alter a prompt after a content-based failure.

The model-side run configuration has corresponding `test` and `full` modes.
`test` deterministically derives a pair-preserving subset from the full
inventory, retains every required split, records `confirmatory=false`, selects
at most two pairs per paired split and two independent-task items per single
split, and uses a 60-minute engineering budget. `full` selects every prompt, records
`confirmatory=true`, and has no time cap. The selector writes a separate
selection manifest containing source and selected inventory hashes, IDs, split
counts, and the intended budget.

The first model-side slice therefore has this artifact flow:

```text
all-16 vector review → human prompt audit → full frozen inventory
                                      ↓
                       test subset + selection manifest
                                      ↓
                 one-hour activation/readout/steering smoke run
                                      ↓
                          inspect artifacts and decide
                                      ↓
                        full inventory + full model run
```

The test run is an engineering gate, not an early estimate of the scientific
effect. Its subset is large enough to preserve paired direction splits,
neutral calibration, and independent behavior/steering roles, but its output
cannot support confirmatory claims. The implementation entrypoint is
`scripts/select_benchmark_run_mode.py`; activation logging records the
optional wall-clock budget and whether the selected subset completed.

The independent prompt-only baseline is a separate stage from zero-dose
steering. `behavior_eval` prompts are executed and scored by
`scripts/run_prompt_only_behavior.py` and
`scripts/score_prompt_only_behavior.py`; `steering_eval` prompts are used for
the target-direction zero-dose variation gate. The outputs have distinct
manifests and must not be pooled. Both paths use the shared no-truncation
tokenizer contract from `activation_analysis.tokenization`.

## 5. Planned execution stages

1. Validate the registry, versioned construct, generation, run, and analysis configs.
2. Review and audit the existing all-16 vector/probe inventory, its final
   manifest, and canonical hashes; regenerate only as an explicitly versioned
   replacement.
3. Audit leakage-safe splits for every construct namespace.
4. Run the manifest-backed prompt-only behavioral baseline per construct where
   its construct-specific parser and task execution are implemented; audit
   compliance and outcome variation before interpreting steering effects.
5. Log activations once at all registered candidate layers, sites, and positions.
6. Construct one direction per construct from training prompts only and select
   the layer using validation prompts only.
7. Evaluate continuous standardized held-out projection margins per construct.
8. Calibrate doses from neutral or within-cell training variance per construct and record
   unstandardized magnitudes.
9. Run independent-task prefill-only steering with `[-1, -0.5, 0, 0.5, 1]`,
   shuffled, and three random controls.
10. Check the recorded downstream persistence and injection arithmetic, then
   add output accessibility, compliance, and collateral behavior checks.
11. Run C1 matched-episode residual interchange on the registered causal
    diagnosis subset, validate its complete manifest, and compare the
    bidirectional swaps with same-condition controls.
12. Parse outputs through construct-specific adapters, compute outcome-
    appropriate estimands, and bootstrap complete pairs/items.
13. Fit the crossed model × construct × task summary; add profile prediction
    only after the matrix is large enough for out-of-sample evaluation and the
    precision simulation supports expansion beyond Wave 1.

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
  `results/benchmark/<run_id>/raw/`.

The shared benchmark raw path is already covered by `.gitignore` before the
first new run. Construct-specific outputs remain namespaced below
`results/benchmark/<run_id>/constructs/<construct_id>/`.

## 7. Automated data lifecycle

The repository now separates execution storage from durable archival. The
same run layout works on a local machine and on a RunPod network volume:

```text
results/benchmark/<run_id>/
├── raw/                    ignored model generations and worker outputs
├── prompts/               frozen combined prompt inventory
├── activations/           shared activation run
├── constructs/<id>/       direction/readout/calibration/steering outputs
├── config_snapshot/       exact run, analysis, and construct configs
├── run_plan.json           shared execution graph and hashes
├── storage_manifest.json   resolved paths and archive policy
├── checksums.sha256       artifact integrity record
└── run_status.json         mutable lifecycle status
```

`scripts/prepare_benchmark_run.py` validates the selected constructs and
configs, creates this layout, snapshots the frozen inputs, and writes the
shared run plan. It performs no API calls and does not load model weights.

`scripts/finalize_benchmark_run.py` writes the status and deterministic SHA-256
inventory, verifies the inventory, and automatically syncs to the configured
S3-compatible archive when `RSC_BENCH_ARCHIVE_URI` is present and the run
configuration has `storage.sync_on_finalize=true`. The archive destination is
always `<archive-prefix>/<run_id>`; credentials and optional endpoint settings
are read from the process environment and never persisted in manifests.

The local/RunPod workspace is a staging copy. The S3-compatible archive is the
durable private master. GitHub contains code and small reviewable metadata;
Hugging Face and Zenodo remain separate, later curation/release targets. This
automation does not claim that the real-model end-to-end runner is validated;
it wraps the existing model-side stages with reproducible storage and handoff
semantics.

All new model-side NumPy artifacts use FP16 on disk by default: activation
shards, train-derived direction arrays, pair-difference arrays, and shuffled or
random control directions. When loaded, readout and calibration routines
promote them to FP32/FP64 for numerical work. JSONL/CSV metadata and scalar
manipulation checks remain text representations rather than typed FP16 files.

## 8. Implementation gates

Before the next large model-side experiment, the required release sequence is:

1. run and review the no-API all-16 vector-prompt pilot, then audit paired
   rows with `scripts/audit_vector_pairs.py`;
2. after human approval, validate the combined full vector inventory against a representative
   manifest-backed activation fixture when one is available;
3. validate held-out projection margins and neutral/within-condition calibration
   on a representative activation run;
4. run the local fake vertical slice and validate the prompt-only baseline
   manifest, parser, and variation gate before using a GPU;
5. validate the implemented Wave 1 task adapters, beginning with the
   realization/evidence engineering slice while retaining source-reliability
   and persistence cells; all-16 downstream parsers remain future work;
6. validate timing, manipulation, and downstream-persistence traces on a
   representative model run, then add output-accessibility and collateral
   checks;
7. run the precision simulation before advancing to Waves 2–4 or fitting
   representation-profile predictors.

The shared control plane and local model-side entrypoints are implemented, but
the benchmark is not yet a real-model end-to-end validated experiment runner.
