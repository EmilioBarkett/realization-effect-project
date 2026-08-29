# Next run: Wave 1 on RunPod B300

## Status and authority

This is the single operative handoff for the next model-side run. It describes
what Codex should execute after reopening the repository. Scientific decisions
remain governed by [`PROJECT_DIRECTION.md`](../PROJECT_DIRECTION.md) and
[`SCIENTIFIC_PROTOCOL.md`](../SCIENTIFIC_PROTOCOL.md); detailed command shapes
and artifact rules remain in [`RUNPOD_EXECUTION.md`](RUNPOD_EXECUTION.md).

The next paid run is only the non-confirmatory model-side Wave 1 preflight. It
loads each registered model once and runs the frozen 16-item behavior,
collateral, and tiny steering checks for every Wave 1 construct. It must not
run representation logging, C1, or a full inventory. If any model/construct
pair fails, stop the pod promptly, preserve the outputs, and revise locally.
Passing the preflight releases only that pair for a separately reviewed
affected-cell rerun; it does not authorize automatic full execution.

The selected inventory manifest is currently `confirmatory=false`. Executing
all of its rows does not silently convert it into confirmatory evidence. Unless
a separate reviewed release step changes that status, Codex must use a
versioned execution copy whose full-coverage mode remains
`confirmatory=false` and describe the results as engineering/exploratory.

## Frozen infrastructure choices

- Cloud provider: RunPod.
- First worker hardware: exactly one NVIDIA B300 GPU.
- Provisioning credential: `RUNPOD_2_API_KEY` in the local Codex process
  environment.
- `RUNPOD_2_API_KEY` is the required and preferred key for every new B300 pod
  in this campaign. Do not silently fall back to `RUNPOD_API_KEY`; that key
  belongs to the earlier RunPod account and existing resources.
- The RunPod API key is used only by the local controller to create, inspect,
  and stop pods. It must not be copied into the pod, written to a command,
  configuration, manifest, log, or Git-tracked file, or printed during a
  presence check.
- Persistent workspace mount: `/workspace`.
- Project workspace: `/workspace/realization-effect-project` via
  `RSC_BENCH_WORKSPACE_ROOT`.
- Hugging Face cache: `/workspace/huggingface` via `HF_HOME`.
- Model inference dtype: BF16 unless the B300 runtime preflight establishes a
  reviewed incompatibility. Persistent NumPy activations and directions remain
  FP16 as specified by the run configuration.

The first pod must report an NVIDIA B300 before model download or experiment
execution begins. Record the RunPod pod ID, exact GPU name/count, image, CUDA,
PyTorch, Transformers, storage mount, repository commit, and model revision in
the private run metadata. Never record credential values.

## Frozen Wave 1 inputs

- Prompt inventory:
  `results/benchmark/prompt_inventories/wave1_preflight_v4_luna/combined.csv`
- Prompt release manifest:
  `results/benchmark/prompt_inventories/wave1_preflight_v4_luna/inventory_manifest.json`
- Model-side gate:
  `configs/construct_benchmark/gates/model_behavior_accessibility_preflight_v2_diagnostic_repair.json`
- Model run configurations:
  `configs/construct_benchmark/run_configs/wave1_four_construct_mistral_supplemental_v3.json`
  and `configs/construct_benchmark/run_configs/wave1_four_construct_qwen38_27b_supplemental_v3.json`
- Analysis specification:
  `configs/construct_benchmark/analysis_specs/rsc_benchmark_core_v1.json`
- Constructs: realization/account closure, evidence diagnosticity, source
  reliability, and persistence/continuation.
- Models: Mistral Small 24B and Qwen3.8 27B at the exact revisions pinned in
  their run configurations. Do not mix model outputs, directions, activations,
  or manifests.

The v3 release remains preserved as historical engineering provenance. The
preflight inventory is a separate frozen v4 release with 2,064 rows; it is
non-confirmatory and replaces only the evidence-diagnosticity downstream rows
with the Luna-audited trade-off design. Hash the inventory, gate, specs, run
configuration, and selection manifest before model execution.

## Codex execution sequence

1. **Recover exact state.** Read the root authority documents and this file,
   inspect Git status, record the exact commit, and preserve all existing
   outputs. Do not delete, overwrite, or silently regenerate prior runs.
2. **Verify the controller credential safely.** Check only that
   `RUNPOD_2_API_KEY` is non-empty. Refuse B300 provisioning if it is absent.
   Do not print it and do not substitute `RUNPOD_API_KEY`.
3. **Finish local gates before billing GPU time.** Run the narrow validation
   commands, the fake vertical slice, the complete local test suite, v4 prompt
   audit, v2 gate validation, and the deterministic per-model selections.
   Resolve failures before provisioning unless they require the exact model
   tokenizer or runtime.
4. **Provision one B300.** Use `RUNPOD_2_API_KEY`, attach persistent storage at
   `/workspace`, and select a recent CUDA/PyTorch image capable of recognizing
   B300. Do not assume an older Hopper image supports the device.
5. **Validate the live runtime.** Confirm the exact GPU, free storage, CUDA,
   PyTorch CUDA availability, model loader, pinned tokenizer/revision,
   no-truncation tokenizer preflight, residual hook path, and a writable
   persistent run directory before starting the campaign.
6. **Freeze v2 selections.** Run
   `scripts/prepare_model_behavior_accessibility_preflight.py` once per model
   using the v4 inventory, all four registered specs, and the v2 gate. Derive a
   construct-pure steering preflight plan for each construct with
   `scripts/prepare_model_steering_preflight.py`; do not edit the full plan.
7. **Run only the model-side preflight.** For each model, run prompt-only
   `behavior_eval` and `collateral_eval` from the frozen selection, then the
   derived steering plan for each construct. Use chat formatting, explicit
   no-thinking mode where supported, max 8 new tokens, min 1, and the shared
   constrained numeric response channel. Do not log activations, construct
   directions, residual interchange, or C1.
8. **Validate immediately.** Run
   `scripts/validate_model_behavior_accessibility_preflight.py` with the v2
   gate. Require 100% valid behavior, three outcomes, SD >= 2, no dominant
   floor/ceiling, collateral >=95% valid and >=75% correct, steering >=95%
   valid, correct injection sign, and nonzero dose response. Any failure holds
   all larger work.
9. **Finalize and stop compute.** Validate completeness and hashes, sync to the
   configured durable archive if available, retain receipts, and terminate
   idle pods. A failed stage must leave resumable artifacts and a truthful
   status rather than an apparently complete result.

## After the preflight

Only after the relevant model/construct pairs pass may the project rerun the
affected Wave 1 behavior, steering, and collateral cells using the valid probe
activations and directions. C1 may resume only after behavioral validity is
restored. Waves 2--4 receive the same 16-item model-side preflight before any
larger execution. A preflight pass never retroactively upgrades earlier
engineering outputs to confirmatory evidence.

## Scaling beyond the first B300

Do not launch several B300s before the first device has demonstrated model
compatibility, measured throughput, stable storage writes, and a resumable
output contract. After that gate, additional B300 workers may run genuinely
disjoint model/wave jobs or disjoint frozen shards. Each worker must have its
own run ID and raw-output path; workers must never append to the same file.

Adding workers does not authorize changing the scientific design. Estimate
remaining wall time, storage, and spend from the measured first-worker rate
before each scale-out step, and keep enough budget to rerun failed shards and
finalize artifacts.

## Fail-closed conditions

Stop the affected stage, preserve its artifacts, diagnose it, and resume only
with compatible provenance if any of these occur:

- `RUNPOD_2_API_KEY` is absent or cannot access the intended RunPod account;
- the provisioned device is not a B300;
- the repository commit, inventory hash, config hash, or model revision differs
  from the recorded run plan;
- tokenizer preflight finds truncation or the fixed task boundary is invalid;
- the model hook, parser, manifest, manipulation check, or zero-dose variation
  gate fails;
- persistent storage is missing, nearly full, or receiving incomplete writes;
- a resume would mix test/full, construct, model, prompt-version, or worker
  outputs.

## Reporting contract

Every progress report and final handoff must state the repository commit,
RunPod pod ID, exact B300 count, model/revision, completed and pending stages,
record counts, run mode, output paths, manifest status, storage use, measured
spend, and whether the claim is engineering-only or empirically eligible. It
must identify the credential only as `RUNPOD_2_API_KEY`; never include the
credential value.
