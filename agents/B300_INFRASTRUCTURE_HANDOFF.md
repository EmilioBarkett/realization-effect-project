# B300 four-wave infrastructure handoff

## Purpose

This is a focused implementation handoff for the active infrastructure worker.
Continue from the current worktree; do not restart the architecture or create a
second orchestration framework. The scientific authority remains in the root
documents, and [`NEXT_RUN.md`](NEXT_RUN.md) remains the operative runtime
runbook.

This handoff explicitly authorizes the worker to launch paid compute after the
local implementation and preflight gates pass. It should complete the missing
machinery, provision one B300 using `RUNPOD_2_API_KEY`, download the pinned
model weights onto persistent storage, and execute as much of the four-wave
campaign as the registered scientific gates and compute budget permit. Do not
commit or push unless the user separately requests it; stage the exact current
worktree and record its content hash when the remote run cannot use a Git
commit.

## Existing machinery to reuse

Preparation and control:

- `scripts/stage_benchmark_bundle.py`
- `scripts/prepare_benchmark_run.py`
- `scripts/select_benchmark_run_mode.py`
- `scripts/preflight_benchmark_run.py`
- `scripts/preflight_tokenizer.py`
- `scripts/validate_confirmatory_execution.py`

Parallel execution and recovery:

- `scripts/shard_benchmark_inventory.py`
- `scripts/run_parallel_benchmark.py`
- `scripts/monitor_parallel_benchmark.py`
- `scripts/compose_benchmark_shards.py`
- `scripts/benchmark_concurrency.py`
- `src/construct_benchmark/sharding.py`
- `src/construct_benchmark/parallel_executor.py`
- `src/construct_benchmark/distributed_contracts.py`
- `src/construct_benchmark/concurrency_benchmark.py`
- `src/construct_benchmark/pod_lifecycle.py`

Scientific stages:

- Behavior: `run_prompt_only_behavior.py`, `score_prompt_only_behavior.py`,
  and `audit_behavioral_variation.py`.
- Representation: `log_residual_streams.py`, `build_activation_vectors.py`,
  and `analyze_construct_readout.py`.
- Steering: `plan_construct_steering.py`, `run_construct_steering.py`, and
  `score_construct_steering.py`.
- Causal diagnosis: `compose_causal_interchange_inventory.py`,
  `run_residual_interchange.py`, and `score_residual_interchange.py`.
- Campaign output: `score_benchmark_campaign.py` and
  `finalize_benchmark_run.py`.
- RunPod shutdown: `stop_benchmark_pod.py`.

Use thin adapters around these entrypoints. Do not reimplement working
scientific stages inside the generic parallel executor.

## Default hardware topology

- Provision exactly one RunPod pod with exactly one NVIDIA B300.
- Use only `RUNPOD_2_API_KEY` in the local controller, with no fallback to
  `RUNPOD_API_KEY` and no key copied into the pod.
- Run up to four simultaneous model processes on the one B300, normally one
  construct-pure process per construct in the active wave.
- Every process owns separate shards, outputs, checkpoints, logs, and
  manifests. No shared-file appends.
- Validate one model load first, then launch three processes, then the fourth
  only if measured and projected VRAM retain the configured safety margin.
- Mistral 24B should target four replicas. Qwen3.8-27B must use the measured
  one-to-three-to-four rollout and may remain at three if four is unsafe.
- Base concurrency decisions only on infrastructure measurements, never on
  scientific effects.

## Default campaign plan

Run one wave at a time on the same B300:

1. Wave 1: realization/account closure, evidence diagnosticity, source
   reliability, and persistence/continuation.
2. Wave 2: reference frame, prior weighting, authority deference, and
   exploration/exploitation.
3. Wave 3: ambiguity orientation, causal interpretation, consensus conformity,
   and plan/replanning.
4. Wave 4: temporal orientation, epistemic uncertainty,
   reciprocity/obligation, and goal shielding.

Resolve Waves 2–4 through
`configs/construct_benchmark/confirmatory_campaigns/waves2_4_confirmatory_v1.json`
rather than duplicating inventory paths.

The active Wave 1 inputs are:

- `results/benchmark/prompt_inventories/wave1_repaired_v2_full_openai_luna_normalized/combined.csv`
- `configs/construct_benchmark/run_configs/wave1_four_construct_repaired_v2.json`
- `configs/construct_benchmark/analysis_specs/rsc_benchmark_core_v1.json`

Do not mix the repaired inventory with the historical 1,650-row composition.

## Default per-wave sequence

1. Runtime and tokenizer preflight.
2. Prompt-only behavioral baseline.
3. Residual logging.
4. Train-only direction construction.
5. Validation-only layer selection.
6. Held-out representation scoring.
7. Neutral or within-cell calibration.
8. Frozen steering-plan construction.
9. Zero-dose behavioral-variation gate.
10. Complete steering and manipulation scoring.
11. C1 matched-episode residual interchange and scoring.
12. Construct-safe wave composition and B/R/C/S summary.

Unload the wave's processes after durable finalization, retain cached weights,
and begin the next wave with clean execution state and distinct namespaces.

## Remaining integration work

Concentrate on these gaps rather than expanding the architecture:

1. Make fake execution explicitly opt-in; missing production adapters must
   fail closed.
2. Make the active RunPod preflight require `RUNPOD_2_API_KEY` only.
3. Complete the local single-B300 provision/readiness/status/stop controller.
4. Complete the real construct-shard-to-stage adapter using the existing CLIs.
5. Implement the VRAM-aware one-to-three-to-four process rollout.
6. Implement wave advancement and resumable four-wave campaign state.
7. Add a versioned Wave 1 full-coverage engineering configuration with
   `confirmatory=false`; do not mutate the frozen source config or inventory.
8. Keep Waves 2–4 confirmatory execution blocked until the registered Wave 1
   measurement, variation, manipulation, prompt-release, and precision gates
   pass. Non-confirmatory engineering selections may run when clearly labelled.
9. Emit an all-16 correspondence-analysis input with B, R, C, and S kept
   separate.

## Verification boundary

The current baseline is:

- `make check`: 283 passed, 3 skipped;
- focused P0 suite: 50 passed;
- the actual 1,824-row Wave 1 inventory completes fake parallel execution;
- the deterministic Wave 1 engineering selection contains 336 records;
- bundle staging, sharding, monitoring, recovery, and fake composition work.

Reuse the current tests. Add or modify only focused tests needed by changed
behavior; do not create a redundant test matrix. Run narrow checks and then
`make check`. Also run a fake four-wave campaign covering all 16 constructs
before starting paid compute.

## Authorized live execution

After the local checks pass, continue without waiting for another prompt:

1. Verify `RUNPOD_2_API_KEY` without printing it and query the intended RunPod
   account and exact B300 availability.
2. Enforce a total campaign ceiling of USD 200 with USD 25 held in reserve;
   never commit more than USD 175 without new user authorization. Record the
   provider's actual hourly price rather than assuming USD 7.50.
3. Provision exactly one B300 with persistent storage mounted at `/workspace`.
4. Stage the exact current worktree, frozen inventories, manifests, and configs
   without relying on an older `origin/main` checkout. Record hashes.
5. Install the required B300-compatible runtime, cache the pinned model weights
   on persistent storage, and run the exact tokenizer, CUDA, hook, storage, and
   one-model VRAM preflights.
6. Perform the measured one-to-three-to-four replica rollout. Use three if the
   fourth would violate the configured memory margin.
7. Run the Wave 1 engineering gate. If it passes, continue immediately into
   the truthful full-coverage Wave 1 execution.
8. Evaluate the registered Wave 1 measurement, variation, manipulation, and
   precision gates. Advance through Waves 2–4 when eligible. If confirmatory
   release remains blocked but the protocol permits a non-confirmatory
   engineering execution, keep it in a separate namespace and label it
   truthfully; never bypass a hard scientific or provenance failure.
9. Produce the eligible all-16 B/R/C/S outputs, validate manifests and hashes,
   preserve resumable state, finalize/archive as configured, and stop the B300
   immediately after a durable terminal report.

The worker must actively monitor the paid run rather than exit after launch.
It should recover compatible interrupted workers, reduce concurrency after an
OOM when safe, and stop the pod on success, terminal failure, budget cutoff, or
an unrecoverable idle condition. It must leave a precise continuation report
if the budget expires before the campaign completes.

## Handoff requirements

Report changed files, reused machinery, exact checks and test counts, fake and
live campaign results, pod ID and exact B300 SKU/count, model/revision, wave and
stage progress, measured VRAM/throughput/storage/spend, output paths, manifest
status, remaining work, and every external call made. Confirm that no secret
value was printed or persisted and state truthfully whether anything was
committed or pushed.
