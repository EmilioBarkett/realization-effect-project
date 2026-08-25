# Construct-benchmark configurations

This directory contains the versioned construct, registry, run, and
analysis artifacts for the representation–steerability benchmark:

- `construct_registry_v1.json`: the 16-construct candidate bank and frozen
  four-wave schedule;
- `constructs/`: one scientific definition per construct;
- `run_configs/`: shared model, activation, and steering settings plus the
list of constructs in a joint run, including the portable workspace and
S3-compatible archive policy;
- `analysis_specs/`: frozen estimands, controls, exclusions, and uncertainty
  rules.

Only the four Wave 1 specifications are currently scientific specifications;
later registry entries are planned and must not be silently substituted or
dropped.

Each Wave 1 generation plan freezes the probe-to-downstream composition:
probes are paired and precede the independent behavior task, only the induced
state may carry over, probe surface text does not carry over, and behavior and
steering pools are separate. Behavior, steering, and calibration roles also use
distinct prompt families, and category schedules are balanced before
generation. Steering execution is now plan-driven; prompt-only
probe-to-downstream behavior composition still remains to be implemented and
validated on a representative model.

The run configuration is intentionally multi-construct. A Wave 1 run loads
one combined prompt inventory and performs one activation logging pass. The
analysis then fans out by `construct_id` into independent direction, readout,
calibration, behavior, and steering artifacts. The same format can later list
all registry entries; directions are never pooled.

Validate the existing two-construct engineering smoke plan with:

```bash
./venv/bin/python scripts/validate_construct_run.py \
  --construct-spec configs/construct_benchmark/constructs/realization_account_closure_v1.json \
  --construct-spec configs/construct_benchmark/constructs/evidence_diagnosticity_v1.json \
  --run-config configs/construct_benchmark/run_configs/two_construct_smoke_v1.json \
  --analysis-spec configs/construct_benchmark/analysis_specs/rsc_benchmark_core_v1.json
```

The command validates configuration and prints the shared/per-construct
execution layout. It does not call an API, load a model, or launch an
experiment.

Before using a GPU, run the deterministic local vertical slice:

```bash
./venv/bin/python scripts/run_fake_benchmark.py \
  --construct-spec configs/construct_benchmark/constructs/realization_account_closure_v1.json \
  --run-config configs/construct_benchmark/run_configs/wave1_realization_single_construct_smoke_v1.json \
  --analysis-spec configs/construct_benchmark/analysis_specs/rsc_benchmark_core_v1.json \
  --output-dir /tmp/rsc_fake_run
```

This creates a validated fake inventory, selects a candidate layer on fake
validation prompts, exercises neutral calibration, five-dose prefill-only
steering, controls, and bootstrap intervals. It makes no external calls and
does not produce empirical results.

The actual four-construct Wave 1 fan-out uses
`run_configs/wave1_four_construct_smoke_v1.json` with the four Wave 1
specifications. It has the same shared activation settings and is intended for
manifest validation before any model execution; no prompt inventory is created
by the run-config validator itself.

For an execution workspace that snapshots configs and archives finalized data,
use `scripts/prepare_benchmark_run.py` followed by
`scripts/finalize_benchmark_run.py`. The run configs refer to environment
variable names, not credentials: set `RSC_BENCH_WORKSPACE_ROOT` and, when a
durable archive is desired, `RSC_BENCH_ARCHIVE_URI` in the worker environment.

## Staged generation and model-side run modes

Prompt generation and model execution are separate stages. The generation
plans expose two named modes:

- `review`: one item per model and cell, explicitly incomplete and intended
  for prompt inspection;
- `full`: all registered counts and models, the only mode that can produce the
  frozen prompt inventory for a confirmatory run.

Both modes can be expanded with `--dry-run`, which makes no API calls. For
example, inspect the realization plan before spending credits:

```bash
./venv/bin/python scripts/generate_construct_prompts.py \
  --plan configs/construct_benchmark/generation_plans/wave1_realization_account_closure_v1.json \
  --mode review --dry-run \
  --summary-output /tmp/realization_prompt_review_summary.json
```

After the strategy is approved, generate the complete inventory explicitly:

```bash
./venv/bin/python scripts/generate_construct_prompts.py \
  --plan configs/construct_benchmark/generation_plans/wave1_realization_account_closure_v1.json \
  --mode full \
  --output /workspace/realization_prompts_full.csv \
  --summary-output /workspace/realization_prompts_full_summary.json
```

The full inventory is the source of truth. Before the first GPU run, derive a
deterministic test inventory from it. The test mode preserves complete probe
pairs and every required split, selects at most two pairs per paired split
and two independent-task items per single split, and records a 60-minute
engineering budget. It is non-confirmatory and must not be interpreted as a
benchmark result:

```bash
./venv/bin/python scripts/select_benchmark_run_mode.py \
  --run-config configs/construct_benchmark/run_configs/wave1_realization_single_construct_smoke_v1.json \
  --construct-spec configs/construct_benchmark/constructs/realization_account_closure_v1.json \
  --prompts /workspace/realization_prompts_full.csv \
  --mode test \
  --output /workspace/realization_prompts_test.csv \
  --manifest-output /workspace/realization_test_selection.json
```

Use that selected inventory for activation logging, readout, calibration, and
steering. The activation logger accepts `--run-mode test
--max-runtime-minutes 60` and records whether the selected subset completed.
Readout analysis refuses an incomplete activation run unless
`--allow-incomplete-run` is supplied for explicitly diagnostic inspection.
Only after reviewing the test artifacts should the same selector be run with
`--mode full`; the full inventory and the `full` run configuration are the
confirmatory path.

Validate the current registry/spec boundary with:

```bash
./venv/bin/python scripts/validate_construct_registry.py \
  --registry configs/construct_benchmark/construct_registry_v1.json
```
