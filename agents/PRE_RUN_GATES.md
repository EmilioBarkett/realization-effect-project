# Local pre-RunPod gates

For the upcoming B300 campaign, apply this checklist within the operative
sequence in [`NEXT_RUN.md`](NEXT_RUN.md). That handoff freezes the hardware and
controller credential; this file defines the reusable scientific and local
engineering gates.

This document describes the work that can be completed and checked locally
before a GPU is started. It is operational guidance, not a source of new
scientific claims. The canonical scientific rules remain in
`PROJECT_DIRECTION.md`, `SCIENTIFIC_PROTOCOL.md`, and
`PROJECT_ARCHITECTURE.md`.

## What the prompt count means

There is no scientific requirement that every wave contain exactly 5,000
unique source prompts. That number is a conservative planning target that can
be useful for a large benchmark, but it must not be treated as a protocol
invariant or a substitute for a precision analysis.

Three different quantities must be kept separate:

1. **Source inventory rows.** These are the frozen prompts written to the
   combined CSV. They include paired probe prompts and separate behavior,
   steering, and calibration prompts.
2. **Independent information-bearing items.** These are the distinct
   downstream task items that determine behavioral variation and statistical
   precision. They matter more for a steering effect than a large number of
   repeated model calls on the same item.
3. **Model evaluations.** A single steering item is evaluated at multiple
   target doses and control directions. Under the current battery, one item
   expands to five target doses, four nonzero shuffled-vector doses, and four
   doses for each of three random vectors: 21 condition cells before counting
   tracking layers or retries. Thus thousands of evaluations can come from a
   much smaller source inventory.

The current frozen engineering packages illustrate why a fixed 5,000-row
rule would be misleading:

| Package | Frozen rows | Vector/probe rows | Downstream rows |
| --- | ---: | ---: | ---: |
| Wave 1 repaired v2 | 1,824 | 1,440 | 384 |
| Wave 2 repaired v2 | 1,584 | 1,440 | 144 |
| Wave 3 repaired v2 | 1,584 | 1,440 | 144 |
| Wave 4 repaired v2 | 1,536 | 1,440 | 96 |

Wave 1 deliberately has 16 independent behavior, steering, and calibration
items for each of the first three constructs and 80 of each for
`persistence_continuation`. If the prompt-only and zero-dose gates show that
the smaller task cells are too constant or too noisy, the correct response is
to expand those independent items using a preregistered precision rule—not to
add arbitrary probe rows until a round number is reached. Bootstrap resampling
quantifies uncertainty; it does not create new information.

The local decision rule is therefore: run the precision simulation using the
observed Wave 1 variance and a scientifically justified minimum effect, then
choose the number of independent downstream items and constructs required for
the desired interval width or power. Keep the 100/40/40 paired-probe scope
separate from that decision because probe sample size controls direction and
readout stability, not downstream behavioral precision by itself.

## What is implemented locally

The repository already contains the representation-measurement and steering
primitives needed for the experiment:

- `src/activation_analysis/residual_streams.py` loads a Hugging Face model,
  resolves transformer blocks, captures residual activations at registered
  layers/sites, records token regions, and supports an opt-in all-layer
  residual-only trace for later localization;
- `src/activation_analysis/log_residuals.py` runs the activation pass and
  writes shard/index files plus a manifest;
- `src/activation_analysis/activation_store.py` memory-maps activation shards,
  validates their structure, and yields prompt/token-level activation records;
- `src/activation_analysis/vector_analysis.py` and
  `scripts/analyze_construct_readout.py` construct train-only directions,
  select layers on validation prompts, and score held-out projection margins;
- `src/construct_benchmark/calibration.py` defines neutral or
  within-condition dose calibration;
- `src/activation_analysis/steering.py` installs residual hooks, supports
  prefill-only additive injection, records pre/post injection arithmetic, and
  tracks later-layer projections;
- `src/activation_analysis/causal_patching.py` implements C1 matched-episode
  residual interchange: a positive/negative induction pair shares one exact
  downstream task, and source state is patched at a tokenizer-verified
  induction/task boundary during prefill only;
- `scripts/plan_construct_steering.py`,
  `scripts/run_construct_steering.py`, and
  `scripts/score_construct_steering.py` plan, execute, validate, and score the
  steering battery with target, shuffled, and random controls;
- `src/construct_benchmark/behavior.py` provides strict construct-specific
  output parsing and directed outcome orientation;
- `scripts/run_prompt_only_behavior.py` and
  `scripts/score_prompt_only_behavior.py` provide the independent prompt-only
  baseline and its manifest-backed variation gate;
- `src/activation_analysis/tokenization.py` and
  `scripts/preflight_tokenizer.py` perform no-truncation token preflight so a
  frozen prompt cannot be silently changed by `max_length` truncation.

This is activation-space interpretability, matched residual causal diagnosis,
and causal residual steering. It is not a complete mechanistic-
interpretability system: SAE feature analysis is archived, C2 temporal
tracing, C3 component/path patching, and C4 ablation are later methods, and
real-model generalized activation/causal/steering validation,
output-accessibility, collateral checks, and correspondence analysis still
need to be run or completed.

The optional `--all-layers` logger mode is deliberately narrower than full
mechanistic interpretability. It preserves every block's residual stream but
does not capture attention heads, MLP outputs, QKV tensors, or sparse features.
It is a future-proofing trace and should be stored separately from the primary
benchmark activation run.

The first causal extension is a separate C1 run, not an interpretation of the
all-layer trace alone. Build a matched-episode JSONL inventory with one
positive induction prompt, one negative induction prompt, and one identical
downstream task per request. Run the fake integration test first, then use the
model-side entrypoint on a small registered subset:

```bash
./venv/bin/python scripts/run_residual_interchange.py \
  --model-id <model-id> \
  --tokenizer-id <model-id> \
  --requests /workspace/causal_interchange_test.jsonl \
  --output /workspace/results/benchmark/<run_id>/raw/residual_interchange.jsonl \
  --layers 10,20,30 --max-length 512 --max-new-tokens 16 \
  --min-new-tokens 1 --dtype bf16 --device-map auto
```

The adjacent manifest must be complete before scoring:

```bash
./venv/bin/python scripts/score_residual_interchange.py \
  --raw-output /workspace/results/benchmark/<run_id>/raw/residual_interchange.jsonl \
  --summary-output /workspace/results/benchmark/<run_id>/causal_interchange_summary.json
```

Use `--resume` only with the same request inventory, model metadata, layers,
timing, and output path. A partial output is an engineering artifact and is
rejected by the summary command unless an explicit diagnostic override is
used. The C1 result tests contextual causal sufficiency; it is not evidence
for a unique circuit or a policy-gain parameter.

## Local checklist before starting a GPU

1. Keep the frozen prompt inventory, construct specs, run config, and analysis
   spec versioned and hashable. Do not mix the older 1,650-row Wave 1
   composition with repaired v2.
2. Run the no-API fake vertical slice and the full local test/lint/compile
   checks.
3. Derive the deterministic `test` selection from the frozen full inventory;
   do not hand-select a convenient prefix.
4. Run tokenizer preflight with the exact model tokenizer and prompt format.
   Any over-limit prompt is a hard failure requiring a versioned inventory or
   configuration decision.
5. Run the prompt-only baseline on `behavior_eval` items as a separate output
   from the target zero-dose `steering_eval` variation gate. Do not pool their
   rows.
6. Confirm that the selected model, revision, candidate layers, activation
   site, token mode, dose calibration, timing, controls, and output paths are
   all recorded in manifests.
7. Start with one model and the realization construct on the GPU. Expand to
   the four Wave 1 constructs only after the model-side hooks, output parsing,
   manifests, and manipulation checks pass.

The RunPod step supplies model/tokenizer files and GPU execution only. It does
not decide prompt counts, construct membership, layer selection, dose scales,
or analysis rules.
