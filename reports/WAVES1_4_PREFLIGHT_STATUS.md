# Waves 1–4 A100 engineering preflight status

Status: **engineering preflight blocked; paid compute stopped; no model-side
results produced**.

This is an engineering preflight and repository-preparation record only. It is
not a confirmatory experiment, does not establish a scientific result, and does
not release the Waves 1–4 benchmark for production.

## Current verified state

- The registered canonical input audit is ready for **8/8 entries**: four waves
  × Qwen/Mistral.
- Each model's audit contains **3,200 direction-train rows** and **1,600
  direction pairs**. These are frozen/reviewable inputs, not model-side
  measurements or empirical results.
- Both authorized Qwen A100 attempts ended fail-closed with **zero model-side
  behavior/readout (B/R) artifacts**:
  - The first attempt lacked a source-controlled launcher.
  - The replacement reached a provider endpoint, but no source-controlled
    launcher or runner heartbeat appeared before the deadline.
- Both Qwen pods are `EXITED`; the provider reports `active_count=0`.
- Mistral was not created.
- No downstream, full, confirmatory, B300, or C1 run occurred.
- No local model weights or raw activations are available, and no empirical
  readout or steering results exist.
- Offline hardening patches are implemented and tested. They are repository
  preparation, not evidence of a successful model-side run.

The active registered input audit is recorded at
`results/benchmark/a100_waves1_4_steering_preparation_20260830_0fb462d/raw/source_input_audit/registered_input_audit.json`.
The historical inventory audit retained for provenance is
`results/benchmark/prompt_inventories/wave1_preflight_v4_luna_v2/inventory_audit.json`;
the frozen prompt-input area is under `results/benchmark/prompt_inventories/`.
The active fail-closed preparation validator and related offline checks are
under `scripts/` and `src/construct_benchmark/`. The validator does not load a
model or execute B/R.

The newly added `scripts/run_train_only_br_preflight.py` is only a fail-closed
planned-manifest validator. Its manifest records
`semantic_runner=not_executed`; it is repository preparation and is not
evidence of B/R execution.

## Release preparation

- The stale release `8ca1c81f3db84ba855a13b10050a38cedbb8c902` was not used.
- The reviewed parser/diagnostic repair was committed and pushed on `main` as
  `d440b35eaeec33b6280056e9ea0a8be591e4775f`.
- Local `HEAD` and `origin/main` matched exactly at that SHA before pod
  creation, and the remote runner recorded the same checkout SHA.
- The preparation commit contains only:
  - `src/construct_benchmark/behavior.py`
  - `src/construct_benchmark/model_preflight.py`
  - `scripts/_ssh_preflight_runner.py`
  - `tests/test_ssh_preflight_runner.py`
- Focused tests: `36 passed`.
- Full check: `make check` — `415 passed, 5 skipped`; Ruff and compilation
  passed; `git diff --check` passed; canonical preflight index validation was
  ready with 8/8 entries.
- Existing unrelated/user-owned worktree changes remain uncommitted and were
  not staged or changed by this run.

## RunPod provenance and contract

- Credential source: `RUNPOD_2_API_KEY` only; the legacy key and config
  fallbacks were ignored. The token was nonempty; recorded fingerprint prefix:
  `af06638e2360e2ff`.
- Intended account: `user_2uiuAdrQLeUatCEZ38R4sX8XsTb`; provider identity
  matched.
- Registered/local SSH public-key fingerprint matched:
  `SHA256:d0Sb+40P8mYLGpSckLADFmLI+0HtD5YoDEu1nGviO8Y`.
- Provider identity reads returned HTTP 200. No API credential was placed in
  the pod environment.
- Pod contract: one non-interruptible secure NVIDIA A100-SXM4-80GB, public
  SSH, official default image
  `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, 160GB container disk,
  no network volume, and no location selector.

## Authorized Qwen attempts

- The first authorized Qwen A100 attempt failed closed because it lacked a
  source-controlled launcher.
- The replacement authorized Qwen A100 attempt reached a provider endpoint,
  but no source-controlled launcher or runner heartbeat appeared before the
  deadline and it also failed closed.
- Both attempts produced zero model-side behavior/readout (B/R) artifacts.
- These failures yielded no empirical readout, steering, downstream behavior,
  calibration, or causal evidence.

## Shutdown and remaining gate

- Both authorized Qwen pods are `EXITED`; provider `active_count=0`.
- Mistral was not created. No downstream/full/confirmatory/B300/C1 run
  occurred.
- No local model weights or raw activations exist, and no empirical readout or
  steering results exist.

The next run must be a fresh authorized A100 attempt using a reviewed,
source-controlled model-side train-only B/R command, with the preparation
validator used to freeze and verify the audit identity. Its launcher and
heartbeat must be verified first, followed by independent bundle/staging
verification, and only then the Mistral attempt.

The Wave 1 measurement and precision-simulation gates still block
confirmatory/full release. Passing the canonical input audit or offline
repository checks does not remove those gates.

## Scope classification

- Repository work: implemented parser/diagnostic repair, tested and pushed.
- Run work: two authorized Qwen-only A100 engineering attempts, both fail-closed
  with zero model-side B/R artifacts, followed by verified shutdown.
- Scientific status: no model-side confirmatory, representation, steerability,
  downstream, calibration, B300, or C1 result; no causal claim or Waves 1–4
  readiness claim.
- Active paths used: `src/construct_benchmark/`, `scripts/`, `tests/`,
  `configs/construct_benchmark/`, and the frozen `results/benchmark/` input
  packages. Archived paths were not used as active inputs.
