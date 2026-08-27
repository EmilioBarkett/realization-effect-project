# Steering traces and manipulation checks

This document records the implementation contract for the model-side steering
instrumentation. It is an engineering handoff, not an empirical result.

## What the runner records

`scripts/run_construct_steering.py` writes an output manifest beside the raw
JSONL file and emits one JSONL row for every:

```text
prompt × frozen steering condition × registered tracking layer
```

Every row carries the prompt and construct IDs, condition/direction kind, dose,
physical scale, model ID and revision, injection layer, tracking layer, layer
site, position mode, timing, dtype/device, direction IDs and hashes, and the
output text. The embedded trace contains scalar observations for all hooked
prefill/generation forwards; no residual tensor is retained in the trace.

At the injection layer the prefill row records:

- projection immediately before the residual addition;
- projection immediately after the addition;
- observed shift (`post - pre`);
- expected shift from the requested dose and frozen calibration scale;
- expected-minus-observed difference;
- whether injection was active on that forward and the absolute token position.

At each later registered layer the prefill row records a projection onto that
layer's independently constructed `direction_train` direction when available.
The row is labelled `downstream_construct_state`. If an older readout artifact
does not contain a later-layer direction, the plan may retain a
`same_vector_persistence_diagnostic`, but that value is not a downstream
construct-state readout.

## Frozen layer policy

The readout command saves candidate directions for every analyzed layer under
`candidate_directions/layer_<N>/`. The steering plan uses the selected layer
for injection and all registered layers at or after it for tracking. Thus a
selected layer 10 tracks later layers 20 and 30; a selected layer 20 tracks
layer 30. A future layer 40 is supported only when it is registered and its
own train-only direction artifact exists. The plan never invents a layer-40
direction.

## Scoring

`scripts/score_construct_steering.py` keeps behavior scoring on the unique
injection-layer row per condition. It writes:

- `parsed_generations.csv` for behavior outcomes;
  - `manipulation_checks.csv` for immediate and downstream scalar records;
  - `summary.json` with expected-versus-observed shift summaries, dose-response
    slopes, control-direction summaries, and downstream persistence ratios.

Persistence is computed against the same prompt's target zero-dose projection
at the same tracking layer and direction. The downstream shift is divided by
that layer's frozen training-calibration scale, and the observed injection
shift is divided by the injection calibration scale before taking their ratio.
This makes the reported persistence ratio comparable across layer-specific
direction spaces. The uncalibrated transfer ratio is retained as a raw diagnostic
only. Missing calibration, a zero denominator, or a missing zero-dose baseline
is reported as missing, not replaced with zero.

## Resumption and provenance

The output manifest is `<raw-output>.manifest.json`. `--resume` requires the
same manifest, steering-plan hash, prompt-inventory hash, model configuration,
layer/site/timing/position settings, generation arguments, and dtype/device
settings. It records the expected condition and row identities, completion
status, row count, and final raw-output hash. Existing rows are keyed by
`(condition_id, tracking_layer)`; complete rows are skipped and incomplete
conditions are safely filled without duplicating existing rows. The scorer
refuses missing, incomplete, truncated, duplicated, or provenance-incompatible
outputs by default; `--allow-incomplete-diagnostic` is an explicit
non-confirmatory override. Incompatible or legacy outputs are rejected.

The deterministic fake vertical slice exercises the numerical summary path
without Torch, Transformers, model weights, APIs, or a GPU. Successful fake or
smoke output is still not evidence of real-model steerability.
