# Realization Steering Architecture

The steering path tests whether the Gemma realization direction is causally
usable during behavior generation. It starts with Gemma because the current
direction vector lives in Gemma residual-stream coordinates. Qwen steering must
wait until we have a Qwen residual activation run and a Qwen-specific direction.

## Current Gemma Flow

1. Build or load a realization direction from matched prompts:
   `realized_closed - paper_open`.
2. Load local Gemma with `transformers`.
3. Register a forward hook on the selected residual layer, currently layer 18.
4. During generation, add `scale * unit_direction` to the residual stream.
5. Log wager/risk outputs for each behavior prompt under several scales.
6. Summarize same-prompt deltas from the unsteered baseline.

The first steering target is:

```text
model: models/gemma-3-4b-pt
direction: results/final/activation_vectors/realization_vector_v1_layer18/mean_direction.npy
layer: 18
position_mode: last
scales: -150, -75, 0, 75, 150
```

Positive scale points toward `realized_closed`; negative scale points toward
`paper_open`.

## Files

```text
src/activation_analysis/steering.py
  ResidualSteeringGenerator
  SteeringConfig
  generation-time residual hook and direction validation

scripts/steer_realization_direction.py
  Gemma steering runner over behavior_eval prompts
  writes manifest.json, steering_eval.csv, steering_summary.json

results/test/activation_vectors/steering_runs/<run_name>/
  manifest.json
  steering_eval.csv
  steering_summary.json
```

The steering runner intentionally writes to `results/test` by default. Promote a
run to `results/final` only after the smoke path is stable and the run is
selected as a canonical report artifact.

## Hook Semantics

Layer numbers are 1-based, matching the residual logger and activation-vector
artifact names.

`position_mode=last` adds the vector only to the final token position seen by
each forward pass. In generation this steers the active causal position while
leaving most prompt-token activations untouched. `position_mode=all` is available
as a stronger diagnostic intervention.

Directions are normalized by default, so steering scale means additive magnitude
in residual activation units along the realization axis. Use
`--no-normalize-direction` only for raw-vector ablations.

## Smoke Command

```bash
./venv/bin/python scripts/steer_realization_direction.py \
  --model-id models/gemma-3-4b-pt \
  --direction results/final/activation_vectors/realization_vector_v1_layer18/mean_direction.npy \
  --layer 18 \
  --position-mode last \
  --scale -150 \
  --scale 0 \
  --scale 150 \
  --limit 12 \
  --run-name gemma_realization_steering_smoke
```

## Qwen Port

The same hook architecture can steer Qwen only when Qwen is self-hosted through
`transformers`. OpenRouter behavior APIs cannot support residual-stream
interventions because they do not expose hidden states or generation hooks.

Qwen port sequence:

1. Log Qwen residual activations for the same realization prompt CSV.
2. Build a Qwen realization direction.
3. Pick Qwen layer(s) by held-out projection separation.
4. Re-run `scripts/steer_realization_direction.py` with Qwen model weights,
   Qwen direction path, and Qwen layer choice.
