# AGENTS.md

## Read first

The current scientific authority is
[`PROJECT_DIRECTION.md`](PROJECT_DIRECTION.md). Read it before making design
or implementation decisions. Then consult:

1. [`BENCHMARK_RESEARCH_DIRECTION.md`](BENCHMARK_RESEARCH_DIRECTION.md) for
   the detailed benchmark proposal and novelty boundary;
2. [`SCIENTIFIC_PROTOCOL.md`](SCIENTIFIC_PROTOCOL.md) for experimental rules;
3. [`PROJECT_ARCHITECTURE.md`](PROJECT_ARCHITECTURE.md) for implementation
   boundaries and milestones;
4. [`readme.md`](readme.md) for the human-facing repository overview.

Archived documents are historical snapshots. They do not override the root
documents above.

## Mission

This repository is developing a representation–steerability correspondence
benchmark for the relationship between:

1. prompt-condition behavioral sensitivity;
2. linear decodability of a theory-relevant internal state; and
3. causal steerability of an independent downstream behavior.

The central hypothesis is that decodability is common but is an incomplete and
often poor predictor of steerability. The longer-term benchmark may add
representation-profile features such as direction stability, context
consistency, localization, and dimensionality, but v1 must not overfit a
high-dimensional predictor to the small engineering or Wave 1 pilot. Do not
collapse representation and control into one outcome.

The active candidate bank is the versioned 16-entry registry, balanced across
decision, epistemic, social, and agentic families. Wave 1 is
`realization_account_closure`, `evidence_diagnosticity`, `source_reliability`,
and `persistence_continuation`. Wave 2–4 construct specifications,
calibration-aware generation plans, and frozen confirmatory prompt-input
inventories are implemented. The released prompt inputs are not empirical
results; model-side confirmatory execution remains gated on the Wave 1
measurement and precision-simulation prerequisites. Specification gaming in
coding agents is not the current project direction.

The versioned
`configs/construct_benchmark/constructs/persistence_continuation_v2.json`
and its generation plan are a candidate redesign, not an active registry
replacement. The existing persistence v1 specification and artifacts remain
frozen for provenance; do not mix v1 and v2 prompts or manifests in one run.
The cross-family audit and release conditions are recorded in
`agents/WAVE1_PROMPT_FAMILY_AUDIT.md`.

## Maturity and scope

The project is currently in scientific protocol development and benchmark
infrastructure implementation. The shared multi-construct control plane,
16-construct registry, Wave 1 construct specifications, generic generation
adapter, generation plans, API-generated vector/probe inventory, released
Wave 2–4 prompt-input packages, deterministic fake vertical slice, deterministic
prompt run-mode selection, manifest-backed prompt-only behavior baseline,
fail-closed tokenizer preflight, and environment-independent measurement core
are implemented. A realization real-model decode pilot is available as an
engineering/reference artifact, and the C1 matched-episode residual
interchange runner is fixture-tested;
generalized real-model end-to-end validation is not.

Do not describe the following as completed:

- generalized benchmark real-run projection-margin or calibration results;
- real-run neutral/within-condition calibration results;
- a validated local or RunPod steering run;
- real-model validation of the downstream manipulation checks and prompt-only
  baseline orchestration;
- the evidence-diagnosticity experiment.

Fixture-tested code now covers train-only directions, held-out standardized
projection margins, neutral/within-condition calibration, strict Wave 1 output
parsing, directed state-transfer scoring, deterministic shuffled/random
controls, timing-aware injection, validation-only layer selection,
independent prompt-role/family checks, pre-registered category schedules,
bootstrap interval primitives, readout/steering planning CLIs, named
prompt-generation review/full modes, pair-preserving model-side test/full
selection, and an optional activation wall-clock guard. These are implemented
code, not empirical results. PyTorch, Transformers, a concrete
Model weights and raw activation tensors remain absent from the base
development environment. The tracked realization decode pilot and all-16
vector/probe inventory are not substitutes for a validated benchmark run.

Active model-side configurations store activation shards, train-derived
directions, pair differences, and control directions as NumPy FP16 artifacts
to reduce persistent-volume use. Readout and calibration code promotes loaded
arrays to FP32/FP64 for arithmetic; this storage choice must not be reported
as a lower-precision statistical analysis. Model inference may use BF16 on
Ampere/Hopper GPUs independently of the on-disk storage dtype.

`src/construct_benchmark/behavioral_variation.py` and
`scripts/audit_behavioral_variation.py` provide the fail-closed model-side
zero-dose variation gate. A completed steering manifest, sufficient valid
target injection-layer rows, multiple distinct outcomes, and positive sample
standard deviation are required before a full run is released. The dedicated
`wave1_four_construct_variation_gate_v1.json` test config selects enough
single-task items for this check; the older smoke config remains an interface
smoke test only.

The original realization behavioral pipeline is archived under
[`archive/realization_effect/`](archive/realization_effect/). The active
activation prompt generator is retained because it will be generalized to
construct-specific paired prompts.

## Repository map

- `src/activation_analysis/`: active prompt generation, residual logging,
  activation storage, vector primitives, steering primitives, matched-episode
  residual interchange, and shared no-truncation tokenization.
- `src/construct_benchmark/`: shared construct/run/analysis schemas,
  canonical prompt records, split validation, registry validation, generic
  generation, multi-construct run plans, and model-independent manipulation
  scoring, prompt-only baseline execution, and behavioral variation gates.
- `scripts/`: active activation-analysis entrypoints.
- `scripts/generate_construct_prompts.py`: benchmark-facing generic generation
  CLI with dry-run support.
- `scripts/generate_downstream_prompts.py`: calibration-aware, Luna-pinned
  review/full generator for independent behavior, steering, and calibration
  prompts.
- `scripts/compose_wave_execution_inventory.py`: no-API composer for one
  four-construct vector-plus-downstream inventory per wave.
- `scripts/release_wave_prompt_inventories.py`: hash-verified, non-destructive
  promotion of frozen wave inventories into confirmatory prompt-input releases.
- `scripts/validate_confirmatory_execution.py`: fail-closed campaign validator
  separating test readiness from confirmatory release readiness.
- `scripts/select_benchmark_run_mode.py`: deterministic test/full inventory
  selection without external calls.
- `scripts/run_fake_benchmark.py`: deterministic no-API vertical-slice runner.
- `scripts/run_residual_interchange.py`: model-side C1 matched-episode
  residual-interchange runner with a resumable output manifest.
- `scripts/score_residual_interchange.py`: fail-closed validation and summary
  for causal-interchange outputs.
- `scripts/run_prompt_only_behavior.py` and
  `scripts/score_prompt_only_behavior.py`: independent downstream-task
  baseline execution, manifest validation, strict parsing, and prompt-only
  variation gating.
- `scripts/preflight_tokenizer.py`: exact no-truncation tokenizer preflight.
- `scripts/validate_construct_registry.py`: registry/spec agreement check.
- `configs/activation_analysis/`: active prompt-generation and probe configs.
- `experiments/activation_analysis/`: reviewable prompt CSVs.
- `tests/`: active tests for activation and prompt-generation behavior.
- `archive/realization_effect/`: archived realization behavioral code,
  configs, notebooks, adapters, and tests.
- `archive/sae/`: archived SAE-training and feature-analysis tests; not active
  test scope.
- `archive/documentation/`: archived planning documents.
- `agents/`: maintainer handoffs and local/GPU execution notes; these do not
  override the root scientific documents.
- `agents/NEXT_RUN.md`: single operative handoff for the upcoming RunPod B300
  Wave 1 campaign; its local controller must use `RUNPOD_2_API_KEY` without
  falling back to the earlier RunPod account.
- `agents/PRE_RUN_GATES.md`: local prompt-count rationale and the checklist
  that must pass before starting a GPU.
- `configs/construct_benchmark/`: versioned construct definitions, run
  configurations, analysis specifications, the 16-entry registry, and Wave 1
  generation plans.
- `reports/` and `results/`: reference artifacts and local/ignored outputs.

The active vector path now imports its iterator from
`activation_analysis.activation_store`. The old SAE-training and
feature-analysis tests are archived. The active environment is now a Python
3.11 editable install, and `make check` passes; the two PyTorch-dependent
interpreter smoke tests are skipped when the optional `interp` extra is not
installed.

The multi-construct control plane uses one combined prompt inventory and one
shared activation pass, then fans out into construct-scoped directions,
readouts, calibration, behavior, and steering outputs. The benchmark-facing
generator creates canonical rows before activation logging; never pool
directions across `construct_id` values.

The steering runner records scalar pre/post injection projections at the
injection layer and scalar projections at registered later layers. Its output
is one row per condition × tracking layer plus an output manifest; later-layer
rows are labelled as independently constructed train-only readouts or as
same-vector diagnostics. The scorer keeps behavior scoring on the injection
layer and reports expected-versus-observed shifts and downstream persistence
separately. These are instrumentation and fixture-tested analysis paths, not
real-model results.

The first causal-pathway method is documented in
`agents/CAUSAL_PATHWAY_ARCHITECTURE.md`. It patches a source residual state
from a matched positive/negative induction episode into the other episode at
the fixed probe-to-task boundary during prefill only. It requires one shared
downstream task, records bidirectional swaps and same-condition controls, and
does not claim necessity, a unique circuit, or a complete mechanistic
explanation. Component/path patching, temporal tracing, and ablation remain
later methods.

The completed Wave 2–4 downstream engineering source inventory is retained at
`results/benchmark/downstream_prompts_v1_waves2_4_full_luna_b20_o30k_v1/`.
It contains 384 non-confirmatory engineering prompts across 12 constructs and
is pinned to the Luna quality gate; it is not a real-model experiment result.
The exact composed inputs have now been released as confirmatory prompt
inputs under
`results/benchmark/prompt_inventories/wave[2-4]_four_construct_confirmatory_v1/`.
This release freezes prompt inputs only; it does not release model execution or
empirical results.

The current Wave 1 repaired model input is
`results/benchmark/prompt_inventories/wave1_repaired_v2_full_openai_luna_normalized/combined.csv`.
Its manifest records 1,824 frozen engineering rows: 1,440 vector/probe rows
and 384 independent behavior, steering, and calibration rows. It remains an
engineering artifact (`confirmatory=false`), not a model-side result. The
older 1,650-row composition remains historical engineering provenance and
must not be mixed with repaired v2.

The Waves 2–4 confirmatory execution package is prepared as three balanced
four-construct run configurations under
`configs/construct_benchmark/run_configs/wave[2-4]_four_construct_confirmatory_v1.json`.
Their campaign index is
`configs/construct_benchmark/confirmatory_campaigns/waves2_4_confirmatory_v1.json`.
The released wave inventories and 72-record test selections are under
`results/benchmark/prompt_inventories/`. Test-mode validation passes, while
full-mode validation remains intentionally blocked by the Wave 1 measurement
gate and precision simulation; the prompt-input release gate is satisfied.

## Research invariants

For every construct:

- define a directional state contrast before data collection;
- freeze train, validation, held-out, and downstream-task splits;
- construct the main direction from the training split only;
- keep probe prompts and downstream behavior tasks independent;
- compose Wave 1 as probe first, independent downstream task second; allow only
  the induced state—not probe text, labels, or entities—to carry over;
- use continuous held-out projection margin as the primary readout;
- treat pair accuracy as secondary;
- use directed mean state transfer as the primary steering outcome;
- treat policy-slope change as secondary;
- calibrate steering dose using frozen neutral or within-condition/within-cell
  training variance, not a positive/negative mixture that couples separation to
  intervention magnitude;
- treat `review` prompt inventories and `test` model runs as engineering
  artifacts only; use the complete `full` inventory and `full` run mode for
  confirmatory analysis;
- include zero, negative, shuffled/random, compliance, and collateral controls;
- do not select signs, layers, scales, or subsets after seeing held-out results;
- report uncertainty and retain null results;
- distinguish injection-layer projection from downstream manipulation checks;
- treat matched-episode residual interchange as contextual causal sufficiency,
  not as proof of necessity, a unique circuit, or a general policy variable;
- require identical downstream task text and a tokenizer-verified fixed
  boundary for C1 patching; do not transplant residuals across unmatched
  prompts;
- record model, layer, activation site, token/position mode, splits, scales,
  timing, sample counts, and exclusions in every manifest.

Do not call a direction a policy-gain direction unless it was explicitly
constructed to represent responsiveness or gain control.

## Safe workflow

Before editing:

```bash
git status --short --branch
rg --files
```

Use small source-level changes and deterministic fixtures. Do not launch API
calls, download model weights, or run a large experiment during ordinary
development. Those actions require explicit user intent and a reviewed
configuration.

Before creating outputs, inspect `.gitignore`. In particular,
`results/benchmark/<run_id>/raw/` is already covered by `.gitignore` before
benchmark runs begin. The shared run root is intentionally multi-construct;
construct-specific outputs live below its `constructs/<construct_id>/`
namespace.

Run the narrowest relevant check first, then `make check` when the project
environment is available. The current development baseline uses the Python
3.11 `venv`; if it is absent in a fresh checkout, report that limitation
instead of claiming tests passed.

Do not delete or regenerate existing local data. Preserve the archived source
and reference artifacts. Do not commit or push unless explicitly requested.

## Handoff requirements

Every handoff should state:

- changed files;
- active versus archived paths;
- commands and checks run;
- tests that could not run and why;
- assumptions about local data, model weights, credentials, and tooling; and
- whether the work is protocol, repository preparation, or implemented
  benchmark code.
