# Next run: Wave 1 on RunPod B300

## Status and authority

This is the single operative handoff for the next model-side run. It describes
what Codex should execute after reopening the repository. Scientific decisions
remain governed by [`PROJECT_DIRECTION.md`](../PROJECT_DIRECTION.md) and
[`SCIENTIFIC_PROTOCOL.md`](../SCIENTIFIC_PROTOCOL.md); detailed command shapes
and artifact rules remain in [`RUNPOD_EXECUTION.md`](RUNPOD_EXECUTION.md).

The next run is an end-to-end, full-inventory Wave 1 engineering campaign with
a short engineering gate at the beginning. The gate is a subset of the same
campaign, not a separate scientific pilot and not the final result. If it
passes, Codex should continue into the complete frozen inventory without
changing prompts, layers, signs, doses, outcomes, or analysis rules in response
to the gate results.

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
  `results/benchmark/prompt_inventories/wave1_repaired_v2_full_openai_luna_normalized/combined.csv`
- Primary run configuration:
  `configs/construct_benchmark/run_configs/wave1_four_construct_repaired_v2.json`
- Analysis specification:
  `configs/construct_benchmark/analysis_specs/rsc_benchmark_core_v1.json`
- Constructs: realization/account closure, evidence diagnosticity, source
  reliability, and persistence/continuation.
- Primary model: `mistralai/Mistral-Small-24B-Instruct-2501` at the exact
  revision pinned in the run configuration.
- Qwen is a separate replication run using
  `wave1_four_construct_qwen38_27b_repaired_v2.json`. Do not mix its layers,
  activations, outputs, or manifests with the primary Mistral run.

The repaired Wave 1 inventory contains 1,824 frozen engineering rows. Do not
mix it with the older 1,650-row composition or any review-generation output.
Hash the selected inventory and configuration snapshot before model execution.

## Codex execution sequence

1. **Recover exact state.** Read the root authority documents and this file,
   inspect Git status, record the exact commit, and preserve all existing
   outputs. Do not delete, overwrite, or silently regenerate prior runs.
2. **Verify the controller credential safely.** Check only that
   `RUNPOD_2_API_KEY` is non-empty. Refuse B300 provisioning if it is absent.
   Do not print it and do not substitute `RUNPOD_API_KEY`.
3. **Finish local gates before billing GPU time.** Run the narrow validation
   commands, the fake vertical slice, the complete local test suite, prompt
   inventory/config cross-validation, and the deterministic test selection.
   Resolve failures before provisioning unless they require the exact B300
   tokenizer or runtime.
4. **Provision one B300.** Use `RUNPOD_2_API_KEY`, attach persistent storage at
   `/workspace`, and select a recent CUDA/PyTorch image capable of recognizing
   B300. Do not assume an older Hopper image supports the device.
5. **Validate the live runtime.** Confirm the exact GPU, free storage, CUDA,
   PyTorch CUDA availability, model loader, pinned tokenizer/revision,
   no-truncation tokenizer preflight, residual hook path, and a writable
   persistent run directory before starting the campaign.
6. **Run the engineering gate.** Execute the deterministic pair-preserving
   `test` selection using the registered one-hour guard. Validate manifests,
   output parsing, non-constant zero-dose behavior, direction construction,
   candidate-layer readout, injection arithmetic, and storage/runtime
   estimates. Test artifacts remain explicitly non-confirmatory.
7. **Continue to full-inventory Wave 1 if the gate passes.** Use every frozen
   row. Before execution, ensure the execution config truthfully preserves the
   inventory's non-confirmatory status; create a versioned execution copy
   rather than mutating the frozen source config. Run the prompt-only
   behavioral baseline, shared activation logging, train-only direction
   construction, validation-only layer selection, held-out readout, frozen
   steering plan and controls, steering execution, manipulation checks, and
   scoring. Preserve construct namespaces throughout.
8. **Run causal diagnosis as a separate output.** Perform the registered C1
   matched-episode residual interchange after the B/R prerequisites pass.
   Treat it as contextual causal sufficiency, not proof of necessity or a
   unique circuit. Component/path patching and ablation are later extensions.
9. **Finalize and stop compute.** Validate completeness and hashes, sync to the
   configured durable archive if available, retain receipts, and terminate
   idle pods. A failed stage must leave resumable artifacts and a truthful
   status rather than an apparently complete result.

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
