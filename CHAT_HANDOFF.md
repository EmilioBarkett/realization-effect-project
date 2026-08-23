# Handoff: current representation–steerability correspondence project

## One-line status

This repository is in scientific protocol development and benchmark
infrastructure implementation. The shared multi-construct control plane is
implemented; no new construct data have been collected and the end-to-end
measurement layer is not complete.

## Read in this order

1. [`PROJECT_DIRECTION.md`](PROJECT_DIRECTION.md) — canonical scientific scope;
2. [`BENCHMARK_RESEARCH_DIRECTION.md`](BENCHMARK_RESEARCH_DIRECTION.md) —
   detailed representation–steerability correspondence proposal;
3. [`BENCHMARK_REVIEW_HANDOFF.md`](BENCHMARK_REVIEW_HANDOFF.md) — brief for an
   independent reviewer;
4. [`SCIENTIFIC_PROTOCOL.md`](SCIENTIFIC_PROTOCOL.md) — current experimental
   contract;
5. [`PROJECT_ARCHITECTURE.md`](PROJECT_ARCHITECTURE.md) — engineering target;
6. [`AGENTS.md`](AGENTS.md) — operating rules;
7. [`readme.md`](readme.md) — human-facing overview.

## Current scientific direction

Test which measurable properties of internal representations predict causal
steerability across diverse theory-relevant constructs. Held-out linear
decodability is the primary representation feature, but the longer-term
benchmark also tracks direction stability, cross-context consistency, layer
localization, intervention cost, specificity, and cross-task transfer. The
working constructs are:

- realization/account closure;
- evidence diagnosticity;
- source reliability/authority;
- persistence/continuation.

The first vertical slice is realization plus evidence diagnosticity. Use a
train-only direction, independent downstream behavior, continuous held-out
projection as the primary readout, and directed mean state transfer as the
primary steering outcome. Policy-slope change is secondary. A 50–100-concept
correspondence bank and scale/checkpoint comparisons are later extensions, not
requirements for the first engineering slice.

Specification gaming in coding agents is not active scope.

## Repository boundary

Active:

- `src/activation_analysis/`;
- `src/construct_benchmark/` — shared schemas, canonical prompt records, split
  validation, and construct-fan-out run plans;
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

- Projection-margin analysis, neutral/within-cell dose calibration,
  outcome-specific state-transfer adapters, and manipulation checks do not
  exist.
- The initial `src/construct_benchmark/` package now validates construct,
  run, analysis, prompt, and split schemas and emits a shared-activation,
  construct-fan-out run plan.
- the active vector iterator now lives in
  `src/activation_analysis/activation_store.py`; the legacy SAE tests are
  archived and `make check` passes under Python 3.11;
- the two residual-stream tests that require the optional PyTorch extra are
  skipped in the base environment;
- no representative local activation run is currently available for an
  end-to-end manifest-backed smoke test.

## Next actions

1. Add a representative activation-run fixture and verify the new
   `ActivationVectorRecord` and `iter_activation_vectors()` implementation
   against its manifest and shards.
2. Connect the active prompt generator to the canonical combined inventory and
   keep the construct namespace intact through activation logging.
3. Keep the active iterator, filtering, region, and memory-map regression
   tests green as the generic benchmark package is introduced.
4. Build the realization/evidence-diagnosticity readout vertical slice.
5. Add neutral/within-cell calibration, explicit intervention timing,
   outcome-specific parsing, output accessibility, and manipulation checks.
6. Run the precision simulation before expanding the construct count or fitting
   full representation-profile predictors.

Do not launch APIs, download weights, or begin a large run until the protocol,
schemas, split manifest, and artifact policy are reviewed.
