# Local or RunPod execution boundary

The scientific configuration, prompt inventory, readout formulas, calibration,
control directions, condition order, and hashes are frozen by repository code.
RunPod supplies only the GPU runtime, model/tokenizer files, and raw execution.

## Environment

Use Python 3.11 and install the optional interpretability dependencies:

```bash
python3.11 -m venv venv
./venv/bin/pip install -e '.[interp]'
./venv/bin/python scripts/check_interpretability_environment.py \
  --run-config configs/construct_benchmark/run_configs/wave1_four_construct_smoke_v1.json
```

Before the check can pass, replace `REPLACE_WITH_LOCAL_MODEL` in a reviewed
run configuration with a pinned Hugging Face model/tokenizer ID or mounted
local directory. Record a revision whenever the source supports one.

Useful RunPod environment variables are:

- `HF_HOME=/workspace/huggingface` for a persistent model cache;
- `HF_TOKEN` only when the selected model requires authenticated access.

`OPENROUTER_API_KEY` is used for synthetic prompt generation, not for local
residual logging or steering. Never copy keys into tracked configuration.

## Model-side sequence

1. Upload or check out the frozen prompt inventory, construct/run/analysis
   configs, and their hashes.
2. Run `scripts/log_residual_streams.py` once over the combined inventory,
   retaining both `scenario` and `task` token regions.
3. For each construct, run `scripts/analyze_construct_readout.py`. It writes the
   train-only direction, pair differences, calibration, held-out margins, and
   provenance summary.
4. Run `scripts/plan_construct_steering.py`. It freezes target, shuffled-label,
   and two orthogonal random controls plus randomized condition order.
5. Run `scripts/run_construct_steering.py` against that frozen plan. Use
   `--resume` only with the same plan and prompt inventory.
6. Run `scripts/score_construct_steering.py` to parse outputs and compute the
   directed target-direction contrast standardized by zero-dose variation.

Begin with one construct, one registered layer, and a few reviewed items. Do
not treat a successful smoke run as the Wave 1 experiment. Output-accessibility,
downstream-persistence, collateral-behavior, uncertainty, and prompt-only
behavior composition still require completion and validation.

## Artifact safety

Keep model weights, activation shards, and raw generations outside Git. Raw
benchmark outputs belong under:

```text
results/benchmark/<construct>/<model>/<run>/raw/
```

Copy back small reviewed manifests, hashes, summaries, and audit outputs only.
