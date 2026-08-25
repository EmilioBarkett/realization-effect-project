# Local or RunPod execution boundary

The scientific configuration, prompt inventory, readout formulas, calibration,
control directions, condition order, and hashes are frozen by repository code.
RunPod supplies only the GPU runtime, model/tokenizer files, and raw execution.

## What the two credentials do

There are two separate services in this workflow:

- `OPENROUTER_API_KEY` is only for the reviewed synthetic-prompt generation
  stage. It is not used to load the open-weight model, log activations, or run
  steering. Prompt generation can be done on the local machine, so this key
  does not need to be copied to RunPod.
- A RunPod account is enough to create a pod through the RunPod dashboard. A
  `RUNPOD_API_KEY` is only required if we later automate pod provisioning from
  a script; the current repository does not require one for a manually
  launched pod.

The first one-construct smoke configuration is currently set to
`mistralai/Mistral-Small-24B-Instruct-2501` with an unpinned revision. The
multi-construct template configurations still contain placeholders until a
multi-construct model run is reviewed.

## Environment

Use Python 3.11 and install the optional interpretability dependencies:

```bash
python3.11 -m venv venv
./venv/bin/pip install -e '.[interp]'
./venv/bin/python scripts/check_interpretability_environment.py \
  --run-config configs/construct_benchmark/run_configs/wave1_four_construct_smoke_v1.json
```

Run the side-effect-free preflight before installing weights or making an API
request. From the local checkout, configuration-only validation is:

```bash
./venv/bin/python scripts/preflight_benchmark_run.py \
  --run-config configs/construct_benchmark/run_configs/wave1_realization_single_construct_smoke_v1.json \
  --construct-spec configs/construct_benchmark/constructs/realization_account_closure_v1.json \
  --analysis-spec configs/construct_benchmark/analysis_specs/rsc_benchmark_core_v1.json
```

On the RunPod pod, use the stricter runtime check after setting the model and
persistent workspace:

```bash
./venv/bin/python scripts/preflight_benchmark_run.py \
  --run-config configs/construct_benchmark/run_configs/wave1_realization_single_construct_smoke_v1.json \
  --construct-spec configs/construct_benchmark/constructs/realization_account_closure_v1.json \
  --analysis-spec configs/construct_benchmark/analysis_specs/rsc_benchmark_core_v1.json \
  --require-model --require-gpu --require-persistent-workspace
```

Add `--require-openrouter` only on the machine that will make the prompt
generation request. Add `--require-archive` only once a durable archive URI
and its sync tooling have been configured. The preflight never contacts any
of these services and never prints secret values.

Before the check can pass, replace `REPLACE_WITH_LOCAL_MODEL` in a reviewed
run configuration with a pinned Hugging Face model/tokenizer ID or mounted
local directory. Record a revision whenever the source supports one.

Useful RunPod environment variables are:

- `HF_HOME=/workspace/huggingface` for a persistent model cache;
- `HF_TOKEN` only when the selected model requires authenticated access.
- `RSC_BENCH_WORKSPACE_ROOT=/workspace/realization-effect-project` for the
  persistent project/run workspace;
- `RSC_BENCH_ARCHIVE_URI=s3://<bucket>/<prefix>` for the durable private
  archive; this is optional for local-only dry runs;
- `RSC_BENCH_S3_ENDPOINT_URL` when the S3-compatible provider requires a
  custom endpoint.

`OPENROUTER_API_KEY` is used for synthetic prompt generation, not for local
residual logging or steering. Never copy keys into tracked configuration.

For the first run, keep the OpenRouter key local and generate/review the pilot
inventory there. The RunPod pod only needs model-side dependencies and the
frozen prompt inventory. If the prompt generation is deliberately moved to the
pod, set `OPENROUTER_API_KEY` in the pod's private environment rather than in a
tracked file.

## First-run configuration sequence

1. Choose and freeze the exact open-weight model ID, tokenizer ID, revision,
   prompt format, and any architecture-specific block path. Put those values
   in a private execution copy of the one-construct run config; do not replace
   the placeholder in the repository-wide template until the choice has been
   reviewed.
2. Run the no-API dry run and fake vertical slice locally. Use generation mode
   `review` to inspect one item per model and cell, then use generation mode
   `full` to create the frozen inventory only after the strategy is approved.
3. Create a RunPod pod through the dashboard with a CUDA PyTorch image and a
   persistent volume/network volume mounted at `/workspace`. The volume should
   hold the checkout, Hugging Face cache, and run artifacts; ephemeral pod
   storage is not the durable copy.
4. On the pod, clone or copy the repository and install
   `./venv/bin/pip install -e '.[interp]'`. Set `HF_HOME` and
   `RSC_BENCH_WORKSPACE_ROOT` to directories on the persistent volume. Set
   `HF_TOKEN` only if the selected model requires it.
5. Copy the reviewed prompt inventory and the exact private run-config copy to
   the pod. Run the strict preflight, then log activations once for the shared
   inventory. Start with one construct and one model.
6. Construct the train-only readout, select the candidate layer on validation,
   freeze the steering/control plan, run the five prefill-only doses, and
   score the independent task. Inspect the output-accessibility and compliance
   checks before expanding to four constructs.
7. Configure `RSC_BENCH_ARCHIVE_URI` and the private AWS/S3-compatible
   credentials before finalization if the run must survive pod deletion. Run
   `finalize_benchmark_run.py --require-archive` and retain the receipt.

The first real GPU run should not combine prompt generation, model download,
activation logging, and steering in one unreviewed command. Each stage should
leave a frozen, hashable artifact for the next stage.

The AWS CLI credentials for the selected S3-compatible provider must also be
available in the worker environment when archive sync is enabled. The
repository never reads them into Python configuration or writes them to a
manifest.

## Named run modes

The generation plans and model-side run configurations now make the staged
workflow explicit:

1. `review` generation is a small, partial OpenRouter inventory for human
   prompt inspection. It is not the data used for a confirmatory run.
2. `full` generation creates the complete frozen inventory. Its output hash
   is the source artifact for all later model-side stages.
3. `test` model execution selects a deterministic, pair-preserving subset of
   the full inventory. It keeps every required split, is capped at two pairs
   per paired split and two items per single split, and carries a 60-minute
   engineering budget. Its `confirmatory` flag is false.
4. `full` model execution uses every record in the frozen inventory and has no
   runtime cap. It is the only mode that can support confirmatory claims.

Materialize the test subset on the persistent workspace before loading model
weights:

```bash
./venv/bin/python scripts/select_benchmark_run_mode.py \
  --run-config configs/construct_benchmark/run_configs/wave1_realization_single_construct_smoke_v1.json \
  --construct-spec configs/construct_benchmark/constructs/realization_account_closure_v1.json \
  --prompts /workspace/realization_prompts_full.csv \
  --mode test \
  --output /workspace/realization_prompts_test.csv \
  --manifest-output /workspace/realization_test_selection.json
```

Use the resulting CSV for activation logging and pass the selection budget to
the model-side logger:

```bash
./venv/bin/python scripts/log_residual_streams.py \
  --model-id mistralai/Mistral-Small-24B-Instruct-2501 \
  --tokenizer-id mistralai/Mistral-Small-24B-Instruct-2501 \
  --prompt-csv /workspace/realization_prompts_test.csv \
  --output-dir /workspace/results/benchmark/realization_test/activations \
  --layers 12,18,24 --batch-size 1 --max-length 512 \
  --run-mode test --max-runtime-minutes 60
```

Review the activation manifest, readout, calibration, and steering outputs
from this test run. If the runtime and model-interface checks pass, repeat the
selection command with `--mode full` and use the resulting full inventory for
the complete run. If the one-hour guard stops partway through the selected
subset, readout analysis refuses it by default; `--allow-incomplete-run` is
reserved for explicitly labelled diagnostic inspection. Do not combine the
test and full outputs in one analysis directory.

## Automatic workspace and archive lifecycle

From the checked-out repository on the network volume, prepare the run before
loading model weights:

```bash
export RSC_BENCH_WORKSPACE_ROOT=/workspace/realization-effect-project
export HF_HOME=/workspace/huggingface

./venv/bin/python scripts/prepare_benchmark_run.py \
  --construct-spec configs/construct_benchmark/constructs/realization_account_closure_v1.json \
  --construct-spec configs/construct_benchmark/constructs/evidence_diagnosticity_v1.json \
  --construct-spec configs/construct_benchmark/constructs/source_reliability_v1.json \
  --construct-spec configs/construct_benchmark/constructs/persistence_continuation_v1.json \
  --run-config configs/construct_benchmark/run_configs/wave1_four_construct_smoke_v1.json \
  --analysis-spec configs/construct_benchmark/analysis_specs/rsc_benchmark_core_v1.json \
  --prompts results/benchmark/wave1_four_construct_smoke_v1/prompts/combined.csv
```

The command creates one shared run root, snapshots all configs, and writes
`run_plan.json` and `storage_manifest.json`. If the prompt inventory has not
yet been generated, omit `--prompts`; generation should write the frozen
inventory to the prepared run's `prompts/` directory, after which rerun the
preparation command with `--resume --prompts ...` so its hash is frozen into
the run plan before activation logging.

After the reviewed model-side stages complete, finalize once:

```bash
./venv/bin/python scripts/finalize_benchmark_run.py \
  --run-root results/benchmark/wave1_four_construct_smoke_v1 \
  --require-archive
```

With `RSC_BENCH_ARCHIVE_URI` set, finalization writes checksums, verifies them,
syncs the complete run to `<archive-prefix>/<run_id>`, and publishes a small
archive receipt. Without an archive URI, use the same command without
`--require-archive` for a local-only finalized run. Use `--dry-run` to inspect
the planned archive command without writing or uploading anything.

## Model-side sequence

1. Upload or check out the frozen prompt inventory, construct/run/analysis
   configs, and their hashes.
2. Run `scripts/log_residual_streams.py` once over the combined inventory,
   retaining both `scenario` and `task` token regions.
3. For each construct, run `scripts/analyze_construct_readout.py` with all
   registered candidate layers, for example
   `--layers 12,18,24 --layer-selection validation_max_margin`. It writes the
   train-only direction, pair differences, neutral calibration, validation-only
   layer selection, held-out margins, bootstrap interval, and provenance summary.
4. Run `scripts/plan_construct_steering.py`. It freezes target, shuffled-label,
   and three orthogonal random controls plus randomized condition order. The
   reviewed Wave 1 run config supplies doses `[-1, -0.5, 0, 0.5, 1]` and
   `prefill_only` timing.
5. Run `scripts/run_construct_steering.py` against that frozen plan. Use
   `--resume` only with the same plan and prompt inventory.
6. Run `scripts/score_construct_steering.py` to parse outputs, compute the
   directed target-direction contrast standardized by zero-dose variation, and
   report an item-level bootstrap interval.

Begin with one construct and one model, while recording the three candidate
layers, five doses, and all registered controls. Review a small prompt batch
before scaling generation. The fake local path should pass first:

```bash
./venv/bin/python scripts/run_fake_benchmark.py \
  --construct-spec configs/construct_benchmark/constructs/realization_account_closure_v1.json \
  --run-config configs/construct_benchmark/run_configs/wave1_realization_single_construct_smoke_v1.json \
  --analysis-spec configs/construct_benchmark/analysis_specs/rsc_benchmark_core_v1.json \
  --output-dir /tmp/rsc_fake_run
```

Do not treat a successful fake or GPU smoke run as the Wave 1 experiment.
Output-accessibility, downstream-persistence, collateral-behavior, real-run
uncertainty validation, and prompt-only behavior composition still require
completion and review.

## Artifact safety

Keep model weights, activation shards, and raw generations outside Git. Raw
benchmark outputs belong under:

```text
results/benchmark/<run_id>/raw/
```

The shared run root also contains construct-namespaced derived outputs under
`constructs/<construct_id>/`. Copy only small reviewed manifests, hashes,
summaries, and audit outputs into Git; the durable archive receives the full
run. Multiple workers must write separate run IDs or disjoint files, with one
finalization/sync step, because concurrent writes to a shared network-volume
file can corrupt artifacts.
