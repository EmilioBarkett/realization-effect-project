# Construct-benchmark configurations

This directory contains the first versioned control-plane artifacts for the
representation–steerability benchmark:

- `constructs/`: one scientific definition per construct;
- `run_configs/`: shared model, activation, and steering settings plus the
  list of constructs in a joint run;
- `analysis_specs/`: frozen estimands, controls, exclusions, and uncertainty
  rules.

The run configuration is intentionally multi-construct. A two-construct run
loads one combined prompt inventory and performs one activation logging pass.
The analysis then fans out by `construct_id` into independent direction,
readout, calibration, behavior, and steering artifacts. The same format can
list four constructs; directions are never pooled.

Validate the initial two-construct plan with:

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
