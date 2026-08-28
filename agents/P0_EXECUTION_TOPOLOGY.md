# P0 execution topology

This note records the implemented, model-independent execution boundary for
the B300 campaign. It is an operational supplement to the root scientific
documents; it does not release a confirmatory model run or change any
construct contract.

## Frozen inputs and shard ownership

`shard_benchmark_inventory.py` validates a complete frozen inventory before
writing any shard. It records the parent hash, exact request and observation
IDs, construct, split counts, expected observations, version family, and
sharding seed. Pair, factor-cell, and unit components are kept together and
constructs are never pooled into one physical shard.

The supported layouts are:

| worker slots | physical shards | schedule |
| --- | --- | --- |
| 3 | 4 for a four-construct inventory | one construct-pure shard per construct; one slot receives two shards |
| 4 | 4 | one construct-pure shard per slot |
| 5 | 5 | the deterministically selected largest construct is split into two disjoint shards |

For the real GPU path, worker_count is a count of model processes on one
pod, not a pod count. The B300 campaign has exactly one RunPod pod with one
NVIDIA B300 GPU. Its staged replica policy is 1 -> 3 -> 4; the fourth
replica is admitted only after measured loaded-model VRAM and runtime checks.
The four model processes each own a construct-pure shard and its output,
checkpoint, manifest, and log. CPU-only readout/scoring may use a separate
parallelism setting after activation artifacts are durable.

The executor retains the physical shard manifests and creates a separate
immutable worker bundle for each active slot. Thus the three-slot schedule is
inspectable even though one worker processes two sequential construct-pure
shards. Each worker has its own output JSONL, checkpoint manifest, and log.

## Recovery and lifecycle

`run_parallel_benchmark.py` launches shell-free worker subprocesses with an
optional positive stagger between launches. Worker manifests persist PID,
stage, heartbeat, completed request and observation IDs, retry count, and
terminal reason. A valid output prefix is resumed; a worker is restarted only
within the configured retry bound. The budget governor refuses launches that
would exceed the configured ceiling or reserve, and the idle watchdog
terminates a worker after the configured no-progress interval.

The terminal report is written before an optional shutdown hook runs. The hook
is disabled by default, receives only fixed non-secret tokens, discards child
stdout/stderr, and cannot change the scientific success/failure status. The
local RunPod controller owns provisioning and shutdown. Its credential is read
only in the local controller process and is never exported inside the pod or
included in the pod-side command:

```bash
./venv/bin/python scripts/runpod_b300_controller.py create \
  --spec configs/construct_benchmark/runpod_b300_v1.json \
  --state /workspace/realization-effect-project/results/runpod/b300_controller.json
```

The pod-side worker command contains no RunPod credential:

```bash
./venv/bin/python scripts/run_parallel_benchmark.py \
  --inventory /workspace/inventory.csv \
  --run-config /workspace/run_config.json \
  --adapter gpu --worker-count 4 \
  --output /workspace/results/parallel_b300 \
  --stage residual_logging
```

`stop_benchmark_pod.py` POSTs to the documented RunPod v1 stop endpoint and
reads only `RUNPOD_2_API_KEY`; it never falls back to the older
`RUNPOD_API_KEY`. Use `--dry-run` to validate the pod ID and endpoint without
making a request.

## Local P0 checks

The no-GPU path is deterministic and does not contact an API:

```bash
./venv/bin/python scripts/run_parallel_benchmark.py \
  --inventory results/benchmark/prompt_inventories/wave1_repaired_v2_full_openai_luna_normalized/combined.csv \
  --run-config configs/construct_benchmark/run_configs/wave1_four_construct_repaired_v2.json \
  --worker-count 5 --fake-model --output /tmp/rsc_fake_parallel
make check
```

The concurrency CLI defaults to the registered 100-request workload size;
an explicitly labelled smoke fixture may override it with
`--expected-request-count`. Throughput selection uses valid requests per
dollar and never reads scientific effect sizes.

`compose_benchmark_shards.py` validates worker output and manifests before
writing a new immutable composition. It rejects duplicate, missing, unknown,
malformed, incomplete, mixed-version, mixed-runtime, and test-to-confirmatory
inputs. `score_benchmark_campaign.py` keeps B, R, C, and S summaries,
exclusions, uncertainty, and existing expansion gates in separate namespaces.

## Scope boundary

The active implementation is under `src/construct_benchmark/`,
`scripts/`, and the active `tests/` files named by the P0 handoff. The
historical realization behavioral pipeline and SAE material remain under
`archive/` and are not imported by this topology. A fake slice or a frozen
prompt inventory is an infrastructure artifact, not a model-side empirical
result. The protected Wave 1 causal run remains a separate active artifact and
must be scored or stopped by its supervising task before any B300 launch.
