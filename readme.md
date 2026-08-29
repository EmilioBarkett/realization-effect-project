# Representation–Steerability Correspondence

This repository is developing a cross-construct benchmark for testing which
properties of internal representations predict causal steerability in language
models.

## Current status

The project is currently in **scientific protocol development and benchmark
infrastructure implementation**. The shared multi-construct control plane is
implemented. API-generated inventories, Mistral/Qwen Wave 1 model-side pilots,
and a realization decode pilot are available as engineering/reference
artifacts; they are non-confirmatory and include explicit release failures.
The environment-independent numerical measurement core and the deterministic
8--16-item model-side behavioral/accessibility preflight are implemented, but
confirmatory end-to-end measurement remains gated.

The original realization-effect behavioral pipeline is archived, while the
activation-analysis prompt generator and activation primitives remain active.

Maintainer-only GPU execution notes and continuation handoffs live under
[`agents/`](agents/); they are not part of the scientific claims of the
project.

## The research question

For a clearly defined behavioral construct, we will ask:

1. Does a controlled prompt contrast produce a behavioral difference?
2. Is the corresponding internal state linearly decodable on held-out prompts?
3. Does steering the frozen direction change an independent downstream
   behavior in the predicted direction?

Our central hypothesis is:

> Linear decodability is common across behavioral constructs, but decodability
> alone is an incomplete and often poor predictor of causal steerability.

Behavioral sensitivity, representation quality, readout, and steering are
separate estimands. A probe that separates two prompt conditions is not
automatically a causal control signal. The detailed proposal treats this as a
representation–steerability correspondence problem rather than a steering
leaderboard.

## Frozen construct bank and staged execution

The benchmark is deliberately broader than behavioral economics. The selected
bank contains 16 theory-relevant constructs balanced across four families; the
versioned registry is
[`configs/construct_benchmark/construct_registry_v1.json`](configs/construct_benchmark/construct_registry_v1.json).

| Wave | Decision | Epistemic | Social | Agentic |
|---|---|---|---|---|
| 1 — anchor | `realization_account_closure` | `evidence_diagnosticity` | `source_reliability` | `persistence_continuation` |
| 2 — weighting/control | `reference_frame` | `prior_weighting` | `authority_deference` | `exploration_exploitation` |
| 3 — uncertainty/adaptation | `ambiguity_orientation` | `causal_interpretation` | `consensus_conformity` | `plan_replanning` |
| 4 — horizon/goal management | `temporal_orientation` | `epistemic_uncertainty` | `reciprocity_obligation` | `goal_shielding` |

All 16 construct definitions and paired-vector generation plans now exist and
are marked `specified` in the registry. Wave 2–4 calibration-aware downstream
generation plans and the review/full inventory workflow are also implemented.
The original Wave 2–4 source inventories remain engineering artifacts after
the audit identified prompt-wrapper, downstream-episode, direct-cue, and
task-independence blockers. Repaired, audited prompt-input releases now exist,
but confirmatory model execution remains gated on Wave 1 measurement and a
precision simulation.
The vector-only inventory is frozen at 100 train, 40 validation, and 40
held-out pairs per construct: 2,880 pairs and 5,760 records total. The current
complete vector/probe inventory is
`results/benchmark/vector_prompts_v2_luna/full_final_all16/combined.csv` with
its final manifest. It is explicitly non-confirmatory and does not include the
independent behavior, calibration, or steering-task prompts. Generation uses
Sonnet 4.6 only and a four-worker orchestrator for vector prompts. The
completed Wave 2–4 downstream engineering source inventory is retained at
`results/benchmark/downstream_prompts_v1_waves2_4_full_luna_b20_o30k_v1/` with
384 non-confirmatory Luna-generated records. Repaired, audited prompt-input
releases now exist under
`results/benchmark/prompt_inventories/wave[2-4]_four_construct_confirmatory_v1/`.
Those releases freeze prompt inputs only; confirmatory model execution remains
gated on Wave 1 measurement and precision prerequisites. The repair history is
in `agents/WAVES2_4_PROMPT_AUDIT.md`.
Source reliability is distinct
from authority deference; persistence is distinct from replanning; and
evidence diagnosticity is not automatically updating responsiveness.

The current Wave 1 repaired model input is also complete as an engineering
artifact at
`results/benchmark/prompt_inventories/wave1_repaired_v2_full_openai_luna_normalized/combined.csv`.
Its manifest records 1,824 frozen rows: 1,440 vector/probe rows and 384
independent behavior, steering, and calibration rows. It is ready for the
staged RunPod smoke test, but it is not a real-model empirical result and
remains `confirmatory=false`. The older 1,650-row composition remains
historical engineering provenance and must not be mixed with repaired v2.

The Waves 2–4 execution package is staged as one four-construct run per wave.
The run configurations live under `configs/construct_benchmark/run_configs/`,
with the gated campaign index at
`configs/construct_benchmark/confirmatory_campaigns/waves2_4_confirmatory_v1.json`.
These are execution plans, not confirmatory-ready prompt packages: the
release validator must continue to refuse full mode until the prompt repairs,
fresh audit, Wave 1 measurement, and precision prerequisites are satisfied.

Specification gaming in coding agents is not the current project direction;
that proposal is preserved as historical material outside the active scope.

## Common experimental design

Each construct will use the same protocol:

```text
theory-relevant paired prompts
        ↓
frozen train/validation/held-out splits
        ↓
train-only linear direction
        ↓
continuous held-out projection readout
        ↓
matched-episode residual causal diagnosis
        ↓
independent downstream behavioral task
        ↓
calibrated additive steering
        ↓
state-transfer, compliance, and collateral-behavior analysis
```

The primary readout is continuous standardized projection margin on held-out
prompts. The primary causal outcome is directed mean behavioral state transfer
under five calibrated doses (`-1`, `-0.5`, `0`, `+0.5`, `+1`). The first
intervention uses prefill-only injection; pair accuracy and policy-slope
changes are secondary outcomes. Policy-gain steering is future exploratory
method development, not a current claim.

Probe prompts and downstream tasks must be meaningfully independent. Splits,
prompt overlap, parsing rules, model settings, layer settings, intervention
timing, and exclusions must be recorded before confirmatory runs.

## What is currently implemented

Active and reusable:

- [`src/activation_analysis/`](src/activation_analysis/) — prompt generation,
  residual logging, activation storage, vector primitives, and steering;
- [`configs/activation_analysis/`](configs/activation_analysis/) — current
  activation and paired-prompt generation plans;
- [`experiments/activation_analysis/`](experiments/activation_analysis/) —
  reviewable prompt CSVs;
- active scripts for prompt generation, residual logging, vector construction,
  evaluation, validation, and overlap auditing.

Implemented control plane:

- `src/construct_benchmark/` — construct, run, analysis, prompt, split, and
  provenance schemas;
- the versioned 16-construct registry with all 16 specified candidate construct
  definitions;
- the generic benchmark-facing prompt generator, all 16 paired-vector
  generation plans, and the four-worker vector-only review/full orchestrator,
  including no-API dry-run support;
- the structural pair/leakage QA entrypoint at
  `scripts/audit_vector_pairs.py`;
- explicit `review`/`full` prompt-generation modes and deterministic
  pair-preserving `test`/`full` model-run selection;
- a separate manifest-backed prompt-only behavior baseline and a target
  zero-dose behavioral-variation gate;
- a shared tokenizer preflight that records exact no-truncation lengths and
  refuses to run when a frozen prompt would be truncated;
- separate behavior, steering, and calibration prompt-family validation plus
  pre-registered categorical schedules for balanced task factors;
- the frozen 8--16-item model-side behavioral/accessibility preflight selector
  and manifest/checksum validator in
  `scripts/prepare_model_behavior_accessibility_preflight.py` and
  `scripts/validate_model_behavior_accessibility_preflight.py`;
- canonical combined prompt inventories with global IDs and construct-scoped
  pair validation;
- shared-activation/construct-fan-out run manifests;
- generalized overlap auditing for construct, split, family, role, template,
  response-format, and probe/downstream independence metadata;
- a four-construct Wave 1 smoke configuration path without pooling directions;
- a separate C1 matched-episode residual-interchange path with a fixed,
  tokenizer-verified induction/task boundary and fail-closed output manifest;

Implemented and fixture-tested, but not yet validated on a real model:

- train-only directions and continuous held-out projection margins;
- neutral/within-condition dose calibration;
- strict Wave 1 parsing and directed state-transfer scoring;
- deterministic shuffled/random controls and timing-aware residual injection;
- validation-only candidate-layer selection and pair/item bootstrap intervals;
- scalar injection pre/post traces, independently labelled downstream-layer
  projections, expected-vs-observed manipulation scoring, persistence ratios,
  and resumable steering-output manifests;
- bidirectional C1 residual interchange, same-condition donor controls, and
  truncated-output validation for matched causal episodes;
- a deterministic no-API fake vertical slice at
  `scripts/run_fake_benchmark.py`;
- local/RunPod readout, steering-plan, execution, and scoring entrypoints.

Still not validated end to end:

- confirmatory real-model behavior composition and real-run uncertainty
  orchestration;
- a passing model-side behavioral/accessibility preflight for every Wave 1
  model/construct pair;
- all-16 downstream parsers and behavior execution;
- real-model validation of the steering traces and the Wave 1 experiments;
- real-model validation of C1 causal interchange, including downstream parsed
  behavior and its registered controls;
- a validated generalized real-model benchmark run; generated downstream
  prompt inventories are available as engineering artifacts, but downstream
  behavior execution and real-model validation remain outstanding.

The all-16 API-generated vector/probe inventory is an engineering artifact, not
an empirical benchmark result. The realization decode pilot is likewise a
real-model engineering/reference result; it does not validate steering,
downstream behavior, or the generalized benchmark.

The active vector path now uses the tracked iterator in
`activation_analysis.activation_store`; the obsolete SAE-only tests are
archived. Under Python 3.11, the clean editable install and active suite now
pass `make check`; the optional PyTorch-dependent interpreter tests are
skipped when that extra is not installed.

## Repository layout

```text
src/activation_analysis/           Active activation-analysis primitives
scripts/                          Active activation entrypoints
configs/activation_analysis/      Active prompt-generation configs
experiments/activation_analysis/  Reviewable prompt datasets
tests/                            Active tests
archive/realization_effect/       Original behavioral pipeline and adapters
archive/sae/                      Archived optional SAE-training tests
archive/documentation/             Superseded planning documents
reports/                           Paper and historical reference artifacts
results/                            Curated summaries and local/ignored outputs
configs/construct_benchmark/        Multi-construct specs and run configs
src/construct_benchmark/            Shared schemas, prompt validation, run plans
results/benchmark/<run_id>/         Portable run workspace and raw artifacts
scripts/run_fake_benchmark.py       No-API deterministic vertical-slice smoke test
scripts/run_residual_interchange.py C1 matched-episode causal diagnosis runner
scripts/score_residual_interchange.py C1 manifest validator and summary
scripts/select_benchmark_run_mode.py  Frozen test/full prompt selection
scripts/generate_all_vector_prompts.py  Review/full vector-only prompt orchestrator
scripts/audit_vector_pairs.py        Structural vector-pair QA audit
scripts/run_prompt_only_behavior.py  Independent behavior baseline runner
scripts/score_prompt_only_behavior.py Baseline parser and variation gate
scripts/preflight_tokenizer.py       Fail-closed tokenizer length check
```

The original realization-effect paper and its implementation remain useful as
the anchor case study. The code is preserved in
[`archive/realization_effect/`](archive/realization_effect/), and the earlier
planning documents are indexed in
[`archive/documentation/`](archive/documentation/).

## Documentation map

- [`PROJECT_DIRECTION.md`](PROJECT_DIRECTION.md) — canonical scientific scope,
  claims, constructs, status, and next gates;
- [`BENCHMARK_RESEARCH_DIRECTION.md`](BENCHMARK_RESEARCH_DIRECTION.md) — the
  detailed representation–steerability correspondence benchmark proposal;
- [`BENCHMARK_REVIEW_HANDOFF.md`](BENCHMARK_REVIEW_HANDOFF.md) — focused brief
  to send to another chat or reviewer;
- [`SCIENTIFIC_PROTOCOL.md`](SCIENTIFIC_PROTOCOL.md) — current experimental
  protocol;
- [`PROJECT_ARCHITECTURE.md`](PROJECT_ARCHITECTURE.md) — current engineering
  architecture and implementation roadmap;
- [`agents/NEXT_RUN.md`](agents/NEXT_RUN.md) — operative RunPod B300 Wave 1
  handoff and credential boundary for the next model-side campaign;
- [`agents/VECTOR_PROMPT_GENERATION_HANDOFF.md`](agents/VECTOR_PROMPT_GENERATION_HANDOFF.md)
  — vector-only review/full generation contract and handoff;
- [`agents/STEERING_MANIPULATION_CHECKS.md`](agents/STEERING_MANIPULATION_CHECKS.md)
  — scalar injection-trace, downstream-persistence, and resumable-output
  contract;
- [`agents/PRE_RUN_GATES.md`](agents/PRE_RUN_GATES.md) — prompt-count
  interpretation, active interpretability components, and local gates before
  GPU execution;
- [`agents/CAUSAL_PATHWAY_ARCHITECTURE.md`](agents/CAUSAL_PATHWAY_ARCHITECTURE.md)
  — matched-episode C1 causal method and later C2–C4 boundary;
- [`AGENTS.md`](AGENTS.md) — instructions and invariants for coding agents;
- [`agents/`](agents/) — maintainer handoffs and GPU execution notes.

## Development

```bash
python -m venv venv
./venv/bin/python -m pip install -e ".[dev]"
make check
```

Do not make API calls, download model weights, or launch a large experiment as
part of ordinary tests. Raw generations, model weights, and large activation
tensors must remain outside Git. The shared benchmark raw path
`results/benchmark/<run_id>/raw/` is already ignored before benchmark runs
begin. Use the preparation and finalization commands to snapshot, checksum,
and optionally archive a run; they do not require RunPod credentials until an
actual archive sync is requested.

The staged execution workflow is: inspect and audit the existing complete
all-16 vector/probe inventory, derive a `test` subset with
`scripts/select_benchmark_run_mode.py`, run the one-hour non-confirmatory
RunPod smoke test, inspect its artifacts, and only then select `full` for the
complete model run. Test outputs must not be pooled with full-run outputs.

The vector review command remains available for a future versioned regeneration
and can be run without an API:

```bash
./venv/bin/python scripts/generate_all_vector_prompts.py \
  --registry configs/construct_benchmark/construct_registry_v1.json \
  --waves all --mode review --workers 4 --dry-run
```

The current full vector/probe inventory is already frozen at
`results/benchmark/vector_prompts_v2_luna/full_final_all16/`. If a new prompt
version is needed after review, write it to a new explicitly versioned output
directory with resumability; do not overwrite the current artifact. The
historical v1 command was:

```bash
./venv/bin/python scripts/generate_all_vector_prompts.py \
  --registry configs/construct_benchmark/construct_registry_v1.json \
  --waves all --mode full --workers 4 \
  --output-dir results/benchmark/vector_prompts_v1/prompts \
  --resume
```

The QA entrypoint for a newly generated inventory is:

```bash
./venv/bin/python scripts/audit_vector_pairs.py \
  --input <versioned-output-dir>/combined.csv \
  --summary-output <versioned-output-dir>/vector_pair_audit.json \
  --flags-output <versioned-output-dir>/vector_pair_flags.csv \
  --fail-on-severe
```

Generation transport retries repeat an identical failed request at most twice;
they never regenerate content or alter a prompt after a content-based failure.

```bash
./venv/bin/python scripts/prepare_benchmark_run.py \
  --construct-spec configs/construct_benchmark/constructs/realization_account_closure_v1.json \
  --construct-spec configs/construct_benchmark/constructs/evidence_diagnosticity_v1.json \
  --run-config configs/construct_benchmark/run_configs/two_construct_smoke_v1.json \
  --analysis-spec configs/construct_benchmark/analysis_specs/rsc_benchmark_core_v1.json

./venv/bin/python scripts/finalize_benchmark_run.py \
  --run-root results/benchmark/two_construct_smoke_v1
```

Set `RSC_BENCH_WORKSPACE_ROOT` on RunPod to the checked-out project directory
and `RSC_BENCH_ARCHIVE_URI` to a credential-free `s3://bucket/prefix` before
finalization. The configured AWS CLI credentials and optional
`RSC_BENCH_S3_ENDPOINT_URL` remain in the environment, never in Git or run
manifests. The durable archive is separate from the eventual curated public
release on Hugging Face or Zenodo.

New model-side NumPy outputs are stored as FP16 to reduce persistent-volume
usage. Readout and calibration calculations promote loaded arrays to FP32/FP64,
so this is a storage optimization rather than a claim that the statistical
analysis itself is performed in FP16.

## Historical reference

The original paper is available at
[`reports/Realization_Effect_in_Language_Models____Ciaran____Emilio.pdf`](reports/Realization_Effect_in_Language_Models____Ciaran____Emilio.pdf).
Its main lesson motivates the new benchmark: a model can show behavioral
sensitivity and contain a linearly decodable signal without that signal being
a reliable causal handle on downstream behavior.
