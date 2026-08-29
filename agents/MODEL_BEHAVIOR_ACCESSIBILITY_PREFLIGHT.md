# Model-side behavioral/accessibility preflight

This is the release gate for the next real-model execution. It is deliberately
smaller than the benchmark and is not confirmatory evidence.

## Required scope

Run the gate separately for every model and every Wave 1 construct. The frozen
selection contains 16 items per construct when the inventory has at least 16;
it may contain 8--15 only when the registered inventory has that many. No
selection may use generated outputs, parser success, or any other outcome.

Each model/construct pair must have:

- an independent `behavior_eval` baseline;
- an independent `collateral_eval` task; and
- `steering_eval` accessibility under the target, shuffled, and random
  directions.

The target direction must include doses `-1`, `0`, and `+1`; controls must
include zero dose. Steering is prefill-only and must show nonzero target
injection with the correct sign at the injection layer and a nonzero dose
response. All numeric response parsers use the shared tokenizer-aware
constraint (only registered values can be emitted), while the strict parser
still rejects any malformed or extra-text output. Every output is checked
against its adjacent complete manifest and checksum.

## Release criteria

The behavior subset must have 100% valid primary outcomes, no invalid or
unscorable items, at least three distinct directed outcomes, sample SD at least
2.0, and no more than 80% of valid outcomes at either observed floor or
ceiling. The collateral subset must have at least 95% valid task outcomes and
at least 75% factual correctness. Every required steering group must have at
least 95% valid primary outcomes, with all selected items represented; the
target injection must have the registered sign and nonzero response at both
nonzero doses.

A pass releases only that model/construct pair for a later larger run. It does
not establish decodability, causal interchange, steerability, or confirmatory
status. Any failure holds the large execution and is recorded with its exact
counts, thresholds, paths, and checksums.

## Commands

First freeze one selection per model from the frozen Wave 1 inventory:

```bash
PYTHONPATH=src python scripts/prepare_model_behavior_accessibility_preflight.py \
  --prompt-inventory results/benchmark/prompt_inventories/wave1_preflight_v4_luna/combined.csv \
  --construct-spec configs/construct_benchmark/constructs/realization_account_closure_v3.json \
  --construct-spec configs/construct_benchmark/constructs/evidence_diagnosticity_v4.json \
  --construct-spec configs/construct_benchmark/constructs/source_reliability_v3.json \
  --construct-spec configs/construct_benchmark/constructs/persistence_continuation_v3.json \
  --model-id MODEL_ID --revision MODEL_REVISION --tokenizer-id MODEL_ID \
  --output PREFLIGHT_SELECTION.json
```

Then run the model-side behavior, collateral, and steering outputs using that
selection and validate them together:

```bash
PYTHONPATH=src python scripts/validate_model_behavior_accessibility_preflight.py \
  --selection-manifest PREFLIGHT_SELECTION.json \
  --construct-spec configs/construct_benchmark/constructs/realization_account_closure_v3.json \
  --construct-spec configs/construct_benchmark/constructs/evidence_diagnosticity_v4.json \
  --construct-spec configs/construct_benchmark/constructs/source_reliability_v3.json \
  --construct-spec configs/construct_benchmark/constructs/persistence_continuation_v3.json \
  --behavior-output BEHAVIOR.jsonl \
  --collateral-output COLLATERAL.jsonl \
  --steering-output realization_account_closure=REALIZATION_STEERING.jsonl \
  --steering-output evidence_diagnosticity=EVIDENCE_STEERING.jsonl \
  --steering-output source_reliability=SOURCE_STEERING.jsonl \
  --steering-output persistence_continuation=PERSISTENCE_STEERING.jsonl \
  --gate-config configs/construct_benchmark/gates/model_behavior_accessibility_preflight_v2_diagnostic_repair.json \
  --output PREFLIGHT_REPORT.json
```

The validator performs no inference and returns a nonzero exit status for any
failure. Existing full runs may be used as inputs only after their selected
items are frozen; they do not retroactively become confirmatory results. The
v3 inventory remains preserved; `wave1_preflight_v4_luna` is the new
non-confirmatory release used by the next small preflight.

## Diagnostic-bundle repair overlay

The stopped Wave 1 diagnostic is recorded in
`agents/WAVE1_DIAGNOSTIC_BUNDLE_REVIEW.md`. Its repair contract is frozen in
`configs/construct_benchmark/gates/model_behavior_accessibility_preflight_v2_diagnostic_repair.json`.
That overlay preserves the v3 inventory, requires a new independent prompt
inventory for repaired items, keeps exact parsing, and requires the
no-thinking/text-only model adapter where the model supports it. It is the
next local release contract; it does not authorize a large run until the
8--16-item real-model preflight passes for every model/construct pair.
