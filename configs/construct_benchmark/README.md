# Construct-benchmark configurations

This directory contains the versioned construct, registry, run, and
analysis artifacts for the representation–steerability benchmark:

- `construct_registry_v1.json`: the 16-construct candidate bank and frozen
  four-wave schedule;
- `constructs/`: one scientific definition per construct;
- `run_configs/`: shared model, activation, and steering settings plus the
list of constructs in a joint run;
- `analysis_specs/`: frozen estimands, controls, exclusions, and uncertainty
  rules.

Only the four Wave 1 specifications are currently scientific specifications;
later registry entries are planned and must not be silently substituted or
dropped.

Each Wave 1 generation plan freezes the probe-to-downstream composition:
probes are paired and precede the independent behavior task, only the induced
state may carry over, probe surface text does not carry over, and behavior and
steering pools are separate. Steering execution is now plan-driven; prompt-only
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

The actual four-construct Wave 1 fan-out uses
`run_configs/wave1_four_construct_smoke_v1.json` with the four Wave 1
specifications. It has the same shared activation settings and is intended for
manifest validation before any model execution; no prompt inventory is created
by the run-config validator itself.

For the explicitly incomplete one-item-per-cell pilot, use
`--count-per-model-per-cell 1 --allow-partial` with
`scripts/generate_construct_prompts.py`. A dry run marks this inventory
incomplete, reports deterministic content-domain assignments and token
estimates, and calculates dollar cost only when both token-price flags are
provided.

Validate the current registry/spec boundary with:

```bash
./venv/bin/python scripts/validate_construct_registry.py \
  --registry configs/construct_benchmark/construct_registry_v1.json
```
