# Vector prompt-generation handoff

**Status:** the Sonnet 4.6 review pilot and the complete all-16 vector/probe
inventory are available. The current canonical inventory is the v2 artifact
under `results/benchmark/vector_prompts_v2_luna/full_final_all16/`; this
handoff supersedes the earlier v1 credit-exhaustion snapshot.

## Scope and scientific status

The versioned registry contains all 16 construct entries, and all 16 have
paired-vector construct specifications and generation plans marked
`specified`. Waves 2–4 are preparatory candidate artifacts. They are not
completed experiments and remain gated from confirmatory model execution by
the Wave 1 measurement gates and the precision simulation.

The vector-only scope is frozen at 100 `direction_train` pairs, 40
`direction_validation` pairs, and 40 `direction_heldout` pairs per construct.
That is 180 pairs / 360 records per construct, or 2,880 pairs / 5,760 records
across all 16 constructs. Downstream single prompts are intentionally excluded
from this inventory; all-16 downstream parsers and behavior execution are not
implemented end to end.

## Generation contract

- Model: `anthropic/claude-sonnet-4.6` only, under the alias `sonnet`.
- Orchestration: four workers, one construct-scoped output per construct, then
  a combined manifest/inventory.
- Sequence: no-API review pilot → human prompt-pair audit → full inventory.
- Review mode emits one pair per paired cell for inspection. Full mode emits
  the frozen 100/40/40 paired inventory.
- `--resume` reuses only outputs that pass the expected-count and hash checks.
- Transport retries are at most two repeats of the identical failed request.
  They never regenerate content or alter a prompt after a content-based
  failure.

API-generated artifacts exist under
`results/benchmark/vector_prompts_v2_luna/`. The canonical full inventory
contains 2,880 pairs / 5,760 records: 180 pairs / 360 records for each of all
16 constructs, with the required 100/40/40 train/validation/held-out split.
Its `final_inventory_manifest.json` records `confirmatory: false` and
`scope_partial: true`: behavior, calibration, and steering-task prompts are
outside this vector/probe-only artifact.

OpenRouter connectivity and authentication were verified. Generation stopped
on 2026-08-25 when the credits endpoint reported 1,450 total credits and
1,449.41279901 used (about 0.59 remaining). The API returned HTTP 402 with
`in_flight_budget_exhausted` even after the requested wait and a sequential
retry. No RunPod or activation run was attempted.

## Completed review pilot

```bash
./venv/bin/python scripts/generate_all_vector_prompts.py \
  --registry configs/construct_benchmark/construct_registry_v1.json \
  --waves all --mode review --workers 4 \
  --output-dir results/benchmark/vector_prompts_v1/review_v2 --resume
```

The review run completed for all 16 constructs. It caught and corrected a
reference-frame labeling error before the full run. The current review
manifest reports 48 valid pairs / 96 records. Earlier files named
`*_superseded*.csv` are retained review history and are not canonical inputs.

## Earlier v1 generation command

```bash
./venv/bin/python scripts/generate_all_vector_prompts.py \
  --registry configs/construct_benchmark/construct_registry_v1.json \
  --waves all --mode full --workers 1 \
  --output-dir results/benchmark/vector_prompts_v1/prompts \
  --resume
```

This command is retained as historical reproduction context for the superseded
v1 run. Do not treat its partial output as the current input. The current v2
full inventory and manifest are already present at the path above; any new
generation must use an explicitly versioned output directory.

## QA audit

```bash
./venv/bin/python scripts/audit_vector_pairs.py \
  --input results/benchmark/vector_prompts_v1/prompts/combined.csv \
  --summary-output results/benchmark/vector_prompts_v1/prompts/vector_pair_audit.json \
  --flags-output results/benchmark/vector_prompts_v1/prompts/vector_pair_flags.csv \
  --fail-on-severe
```

The audit entrypoint is structural and lexical QA for paired rows, split
counts, leakage, and nuisance matching. Human review remains required for
semantic construct validity and for disposition of non-hard flags. A
14-construct audit was stopped after more than 13 minutes because the current
cross-pair near-duplicate search scales poorly; optimize or scope that phase
before treating full-inventory audit latency as acceptable. Generation-time
schema/count/split validation did pass for every written construct.

## Checks completed

- Review: 16/16 constructs, 48 pairs / 96 records.
- Full: 16/16 constructs, 2,880 pairs / 5,760 records in the v2 inventory.
- Every completed full construct: 100 train, 40 validation, 40 held-out pairs.
- `make check`: lint and compilation clean; re-run it in the receiving
  checkout and report the current test count because the suite evolves with
  model-side instrumentation.
- No benchmark activation logging, steering, or end-to-end empirical result was
  produced by this generation run. A separate realization real-model decode
  pilot exists as a reference artifact and is not a steering result.

## Gates and handoff assumptions

The review/full vector artifacts are repository preparation, not measurement
results. Before confirmatory claims, retain the frozen manifests, review/audit
outputs, model and prompt hashes, split counts, and transport logs; then run
the Wave 1 measurement and precision gates. Do not treat a successful vector
inventory as evidence of decodability, behavioral sensitivity, or
steerability. Keep independent downstream tasks separate from probe text,
labels, and entities; only an induced model state may carry over in a later
model-side episode.
