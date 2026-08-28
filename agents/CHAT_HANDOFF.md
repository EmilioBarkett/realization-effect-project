# Historical chat handoff: representation–steerability correspondence project

> **Superseded for execution.** This file preserves an earlier implementation
> snapshot. Use [`NEXT_RUN.md`](NEXT_RUN.md) for the current B300 campaign,
> [`RUNPOD_EXECUTION.md`](RUNPOD_EXECUTION.md) for detailed commands, and the
> root authority documents for scientific decisions. Do not infer current
> blockers or next actions from the historical lists below.

## One-line status

This repository is in scientific protocol development and benchmark
infrastructure implementation. The shared multi-construct control plane and
scalar steering-manipulation instrumentation are implemented; the numerical
measurement core is fixture-tested. The complete all-16 vector/probe inventory
and a realization real-model decode pilot are available as engineering/reference
artifacts; no generalized real-model end-to-end benchmark run exists.

## Read in this order

1. [`PROJECT_DIRECTION.md`](../PROJECT_DIRECTION.md) — canonical scientific scope;
2. [`BENCHMARK_RESEARCH_DIRECTION.md`](../BENCHMARK_RESEARCH_DIRECTION.md) —
   detailed representation–steerability correspondence proposal;
3. [`BENCHMARK_REVIEW_HANDOFF.md`](../BENCHMARK_REVIEW_HANDOFF.md) — brief for an
   independent reviewer;
4. [`SCIENTIFIC_PROTOCOL.md`](../SCIENTIFIC_PROTOCOL.md) — current experimental
   contract;
5. [`PROJECT_ARCHITECTURE.md`](../PROJECT_ARCHITECTURE.md) — engineering target;
6. [`AGENTS.md`](../AGENTS.md) — operating rules;
7. [`CODEX_NEXT_STEPS.md`](CODEX_NEXT_STEPS.md) — selected construct bank and
   synthetic-prompt implementation handoff;
8. [`readme.md`](../readme.md) — human-facing overview.

## Current scientific direction

Test which measurable properties of internal representations predict causal
steerability across diverse theory-relevant constructs. Held-out linear
decodability is the primary representation feature, but the longer-term
benchmark also tracks direction stability, cross-context consistency, layer
localization, intervention cost, specificity, and cross-task transfer. The
frozen candidate bank has 16 constructs in four families, recorded in the
versioned registry; Wave 1 is:

- realization/account closure;
- evidence diagnosticity;
- source reliability;
- persistence/continuation.

Waves 2–4 are planned one-per-family expansions. Two constructs validate the
engineering slice, Wave 1 supports a descriptive four-construct pilot, and the
full bank is reserved for later out-of-sample profile prediction. Use a
train-only direction, independent downstream behavior, continuous held-out
projection as the primary readout, and directed mean state transfer as the
primary steering outcome. Policy-slope change is secondary.

Specification gaming in coding agents is not active scope.

## Repository boundary

Active:

- `src/activation_analysis/`;
- `src/construct_benchmark/` — shared schemas, canonical prompt records, split
  validation, registry validation, generic generation, uncertainty primitives,
  fake fixtures, and construct-fan-out run plans;
- `configs/activation_analysis/`;
- `configs/construct_benchmark/`;
- `experiments/activation_analysis/`;
- active activation-generation, logging, vector, evaluation, validation, and
  audit scripts.

Archived:

- `archive/realization_effect/` — original behavioral pipeline and
  realization-specific adapters;
- `archive/documentation/legacy/` — superseded planning documents.

## Known blockers

- Projection-margin analysis, neutral/within-condition calibration, strict
  Wave 1 parsing, primary state-transfer scoring, deterministic steering
  controls, and timing-aware injection are implemented and fixture-tested.
- Injection pre/post traces, independently labelled downstream projections,
  expected-versus-observed scoring, downstream persistence ratios, and
  manifest-backed resumability are implemented and fixture-tested. Prompt-only
  behavior composition, real-run uncertainty, output-accessibility/collateral
  checks, and real-model validation remain.
- The 16-entry registry, four specified Wave 1 construct specs, four
  generation plans, generic canonical-record generation adapter, and
  generalized overlap audit are now implemented.
- The initial `src/construct_benchmark/` package now validates construct,
  run, analysis, prompt, and split schemas and emits a shared-activation,
  construct-fan-out run plan.
- the active vector iterator now lives in
  `src/activation_analysis/activation_store.py`; the legacy SAE tests are
  archived and `make check` passes under Python 3.11;
- the two residual-stream tests that require the optional PyTorch extra are
  skipped in the base environment;
- the API-generated all-16 vector/probe inventory is available, but it excludes
  behavior, calibration, and steering-task prompts and is explicitly
  non-confirmatory; the realization decode pilot is not an end-to-end steering
  run;
- a representative generalized activation/steering smoke test is still needed;
  no-API and fake-fixture artifacts remain explicitly non-empirical.

## Historical next actions (do not use as the current runbook)

1. Review the existing all-16 vector/probe inventory, final manifest, and
   structural audit; obtain explicit approval before any new external
   generation.
2. Use the realization decode pilot as a reference while adding a
   representative activation-run fixture and verifying the new
   `ActivationVectorRecord` and `iter_activation_vectors()` implementation
   against its manifest and shards.
3. Connect the approved canonical Wave 1 inventory to activation logging while
   keeping construct namespaces intact.
4. Run the implemented readout and calibration CLIs against that activation
   fixture and verify the frozen hashes and margins.
5. Execute the timing-aware steering runner for one construct and one model,
   using candidate-layer validation, five prefill-only doses, and all controls,
   before expanding to all Wave 1 cells.
6. Validate the steering trace and persistence artifacts on one real model
   run, then add prompt-only behavior composition, output accessibility,
   collateral checks, and real-run uncertainty reporting.
7. Run the precision simulation before advancing to Waves 2–4 or fitting full
   representation-profile predictors.

Do not launch APIs, download weights, or begin a large run until the protocol,
schemas, split manifest, and artifact policy are reviewed.
