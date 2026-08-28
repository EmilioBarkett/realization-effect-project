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

All 16 construct specifications and paired generation plans are now versioned
in the registry. Wave 1 is the immediate real-model measurement gate. The
existing Wave 2–4 inventories are preserved engineering artifacts and have
not been released as confirmatory prompt inputs because the audit found
prompt-wrapper, downstream-episode, direct-cue, and task-independence
blockers. Model-side execution remains gated from confirmatory use until the
prompt repairs, Wave 1, and precision prerequisites pass.

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

The current full downstream prompt inventories are retained under
`results/benchmark/`: the Wave 1 inventory contains 210 records and the
Waves 2–4 engineering source inventory contains 384 records. The exact
composed Wave 2–4 inputs are not currently released for confirmatory use; see
`agents/WAVES2_4_PROMPT_AUDIT.md` for the blockers and repair requirements.

The current Wave 1 full inventory is
`results/benchmark/downstream_prompts_v1_wave1_full_luna_current_b50_o60k_v3/`.
Its 210 records were generated with four Luna workers using batch size 50.
The composed four-construct Wave 1 model input is
`results/benchmark/prompt_inventories/wave1_four_construct_full_luna_current_b50_v1/combined.csv`.

## Waves 2–4 confirmatory execution package

The execution package uses one shared activation pass and construct-scoped
fan-out for each balanced four-construct wave. The no-API composer creates the
wave inputs from the frozen all-16 vector inventory and the audited downstream
inventory:

```bash
./venv/bin/python scripts/compose_wave_execution_inventory.py --waves 2 3 4
```

The resulting engineering inputs are under
`results/benchmark/prompt_inventories/wave2_four_construct_full_luna_v1/`,
`wave3_four_construct_full_luna_v1/`, and
`wave4_four_construct_full_luna_v1/`. These inputs remain non-confirmatory
until the blockers in `agents/WAVES2_4_PROMPT_AUDIT.md` are repaired. The
campaign validator checks the test path immediately and refuses full mode
until the remaining release blockers in
`configs/construct_benchmark/confirmatory_campaigns/waves2_4_confirmatory_v1.json`
are satisfied:

```bash
./venv/bin/python scripts/validate_confirmatory_execution.py --mode test
./venv/bin/python scripts/validate_confirmatory_execution.py --mode full
```

The current test path is an engineering path for RunPod; it does not make the
historical inventories confirmatory. The full path remains correctly blocked
because prompt repairs, Wave 1 measurement, and precision-simulation evidence
are still pending.

To reproduce the non-destructive prompt promotion for a new release version,
use the release command with an explicit authority and scope statement:

```bash
./venv/bin/python scripts/release_wave_prompt_inventories.py \
  --waves 2 3 4 \
  --released-by "repository owner" \
  --release-statement "Release frozen prompt inputs only; model execution remains gated."
```

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

For the first model-side Wave 1 variation gate, use
`run_configs/wave1_four_construct_variation_gate_v1.json`. It is a separate,
non-confirmatory test config that preserves complete probe pairs while
selecting eight pairs per direction split and twelve independent-task items
per construct. This is the minimum useful test size for checking zero-dose
behavioral variation; the older smoke config intentionally remains smaller.

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
