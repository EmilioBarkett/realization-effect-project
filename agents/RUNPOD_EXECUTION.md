# Local or RunPod execution boundary

The scientific configuration, prompt inventory, readout formulas, calibration,
control directions, condition order, and hashes are frozen by repository code.
RunPod supplies only the GPU runtime, model/tokenizer files, and raw execution.

For the immediately upcoming B300 campaign, read
[`NEXT_RUN.md`](NEXT_RUN.md) first. It is the operative run handoff; this file
is the reusable command and artifact reference.

## Credential boundaries

There are three credential domains in this workflow:

- `OPENAI_API_KEY` is for the active reviewed/full synthetic-prompt generation
  stage through the OpenAI Responses API using Luna. It is not used to load
  the open-weight model, log activations, or run steering. Prompt generation
  can be done on the local machine, so this key does not need to be copied to
  RunPod.
- `OPENROUTER_API_KEY` is retained only for the legacy activation-analysis
  generator and historical reproduction. It is not part of the active
  construct-benchmark workflow.
- `RUNPOD_2_API_KEY` is the required local-controller credential for automated
  provisioning of the next B300 campaign. Check only for its presence and do
  not print it. Do not fall back to `RUNPOD_API_KEY`, which identifies the
  earlier RunPod account and resources.
- Neither RunPod key is required inside a provisioned pod. Pod-side execution
  receives only the frozen repository inputs plus any separately required
  model/archive credentials.

The one-construct smoke and four-construct Wave 1 configurations are pinned
to `mistralai/Mistral-Small-24B-Instruct-2501` with revision
`9527884be6e5616bdd54de542f9ae13384489724`. A model-side run still requires
the pod environment and architecture-specific hook validation.

The parallel Qwen replication configuration is
`configs/construct_benchmark/run_configs/wave1_four_construct_qwen38_27b_repaired_v2.json`.
It pins `Qwen/Qwen3.8-27B` to revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, uses candidate layers 16, 32, and
48 for its 64-layer language stack, and is processor-first. The shared loader
uses `AutoProcessor`/`AutoModelForMultimodalLM` fallbacks for this model; do
not infer that the Mistral block path or layer numbers transfer unchanged.

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

Add `--require-openai` only on the machine that will make the active prompt
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

The active run configurations use `storage_dtype=float16`. This controls
persistent NumPy activation and direction artifacts only; use `--dtype bf16`
for normal Ampere/Hopper inference unless a reviewed preflight shows another
choice is required.

`OPENAI_API_KEY` is used for active synthetic prompt generation, not for local
residual logging or steering. Never copy keys into tracked configuration.

For the first run, keep the OpenAI key local and generate/review the pilot
inventory there. The RunPod pod only needs model-side dependencies and the
frozen prompt inventory. If prompt generation is deliberately moved to the
pod, set `OPENAI_API_KEY` in the pod's private environment rather than in a
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

1. `review` generation is a small, partial OpenAI/Luna inventory for human
   prompt inspection. It is not the data used for a confirmatory run.
2. `full` generation creates the complete frozen inventory. Its output hash
   is the source artifact for all later model-side stages.
3. `test` model execution selects a deterministic, pair-preserving subset of
   the full inventory. The legacy smoke configs select two items per single
   split, which is enough to exercise interfaces but not enough to estimate
   zero-dose behavioral variation. For the Wave 1 release gate, use
   `wave1_four_construct_variation_gate_v1.json`, which selects eight pairs
   per paired split and twelve items per single split within the same
   60-minute engineering budget. Its `confirmatory` flag is false.
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
  --layers 10,20,30 --batch-size 1 --max-length 512 \
  --run-mode test --max-runtime-minutes 60
```

Review the activation manifest, readout, calibration, and steering outputs
from this test run. If the runtime and model-interface checks pass, repeat the
selection command with `--mode full` and use the resulting full inventory for
the complete run. If the one-hour guard stops partway through the selected
subset, readout analysis refuses it by default; `--allow-incomplete-run` is
reserved for explicitly labelled diagnostic inspection. Do not combine the
test and full outputs in one analysis directory.

For the prepared Waves 2–4 campaign, use one run configuration and one
composed inventory per wave. Validate all three test paths before starting a
pod:

```bash
./venv/bin/python scripts/validate_confirmatory_execution.py --mode test
```

The command selects 72 prompts per wave while preserving the four construct
namespaces and every required split. Do not run the full configurations yet:
the prompt-input release is complete, but the same validator will refuse them
until the Wave 1 measurement gate and precision simulation are recorded in the
campaign manifest.

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
  --prompts results/benchmark/prompt_inventories/wave1_four_construct_full_luna_current_b50_v1/combined.csv \
  --run-mode full
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

For the optional interpretability extension, replace `--layers 10,20,30` with
`--all-layers`. This records `resid_post` at every transformer block and marks
the run as `instrumentation.mode=residual_all_layers`. It is useful for later
layer-localization, representation drift, and causal-tracing analyses, but it
is explicitly a residual-only trace: it does not capture attention-head,
MLP, QKV, or sparse-feature internals and must not replace the primary
benchmark extraction. Estimate storage first and consider `--token-mode final`
or a narrow token-region filter for this extension.

The corresponding command shape is:

```bash
./venv/bin/python scripts/log_residual_streams.py \
  --model-id mistralai/Mistral-Small-24B-Instruct-2501 \
  --tokenizer-id mistralai/Mistral-Small-24B-Instruct-2501 \
  --revision 9527884be6e5616bdd54de542f9ae13384489724 \
  --prompt-csv /workspace/realization_prompts_test.csv \
  --output-dir /workspace/results/benchmark/realization_test/residual_trace \
  --all-layers --batch-size 1 --max-length 512 \
  --token-mode final --run-mode test --max-runtime-minutes 60
```

Do not interpret this residual trace as a complete mechanistic explanation.
After the B/R gates pass, run the first causal method as a separate C1
matched-episode diagnosis. Its JSONL input must contain a positive induction,
a negative induction, and one identical downstream task for every request. The
runner locates the last complete induction token and patches only that
boundary during prefill:

```bash
./venv/bin/python scripts/run_residual_interchange.py \
  --model-id mistralai/Mistral-Small-24B-Instruct-2501 \
  --tokenizer-id mistralai/Mistral-Small-24B-Instruct-2501 \
  --revision 9527884be6e5616bdd54de542f9ae13384489724 \
  --requests /workspace/causal_interchange_test.jsonl \
  --output /workspace/results/benchmark/realization_test/raw/residual_interchange.jsonl \
  --layers 10,20,30 --max-length 512 --max-new-tokens 16 \
  --min-new-tokens 1 --dtype bf16 --device-map auto
```

The command writes an adjacent manifest and can resume only with the same
frozen inventory and settings. Validate and summarize it with:

```bash
./venv/bin/python scripts/score_residual_interchange.py \
  --raw-output /workspace/results/benchmark/realization_test/raw/residual_interchange.jsonl \
  --summary-output /workspace/results/benchmark/realization_test/causal_interchange_summary.json
```

The scorer refuses incomplete output by default. C1 tests contextual causal
sufficiency at the registered boundary; component-level tracing, temporal
tracing, and ablation remain later analyses.

3. For each construct, run `scripts/analyze_construct_readout.py` with all
   registered candidate layers, for example
   `--layers 10,20,30 --layer-selection validation_max_margin`. It writes the
   train-only direction for every candidate layer, candidate-layer artifacts,
   neutral calibration, validation-only layer selection, held-out margins,
   bootstrap interval, and provenance summary.
4. Run `scripts/plan_construct_steering.py`. It freezes target, shuffled-label,
   and three orthogonal random controls plus randomized condition order. The
   reviewed Wave 1 run config supplies doses `[-1, -0.5, 0, 0.5, 1]` and
   `prefill_only` timing.
5. Run `scripts/run_construct_steering.py` against that frozen plan. Use
   `--resume` only with the same plan and prompt inventory.
6. Run `scripts/score_construct_steering.py` to parse outputs, compute the
   directed target-direction contrast standardized by zero-dose variation, and
   report an item-level bootstrap interval. The same command writes
   `manipulation_checks.csv` and includes expected-versus-observed injection
   shifts plus calibration-standardized downstream persistence summaries. It
   requires the adjacent completed output manifest and rejects truncated or
   provenance-incompatible JSONL by default; use
   `--allow-incomplete-diagnostic` only for explicitly non-confirmatory
   inspection.

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
The scalar downstream-persistence and injection-manipulation checks are now
implemented, but still require real-model validation. Output accessibility,
collateral behavior, real-run uncertainty validation, and prompt-only behavior
composition still require completion and review.

After each completed target steering test, run the fail-closed behavioral
variation gate before preparing a full inventory. It requires the adjacent
completed steering manifest, parses only target-direction injection-layer
zero-dose rows, excludes registered neutral realization controls, and rejects
constant or invalid outcomes:

```bash
./venv/bin/python scripts/audit_behavioral_variation.py \
  --raw-generations /workspace/results/benchmark/<run_id>/raw/constructs/<construct_id>/steering_generations.jsonl \
  --construct-spec configs/construct_benchmark/constructs/<construct_id>_v1.json \
  --output /workspace/results/benchmark/<run_id>/constructs/<construct_id>/behavioral_variation_gate.json
```

Run it separately for realization, evidence diagnosticity, source reliability,
and persistence. A nonzero exit blocks the full run; do not replace a zero
standard-deviation denominator with an epsilon or change sampling after seeing
the test result.

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
