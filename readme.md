# Realization Effect Replication — LLM Study

Replication of Flepp, Meier & Franck (2021) *"The effect of paper outcomes
versus realized outcomes on subsequent risk-taking: Field evidence from casino
gambling"* (OBHDP 165, 45–55), substituting language models for human
subjects.

The project now has two connected goals:

1. Measure whether LLMs reproduce realization-effect gambling behavior.
2. Use residual-stream logging and activation-vector methods to test whether
   realization framing is internally represented and whether that representation
   predicts or can steer risk-taking behavior.

The active interpretability path is no longer SAE-first. The earlier SAE work is
archived as supporting infrastructure, while the current experiment follows an
Anthropic-style paired activation-vector approach: build a direction from
matched `paper_open` versus `realized_closed` prompts, evaluate projection on
held-out prompts, then test whether the same direction relates to generated
wager/risk behavior.

## What this study tests

The **realization effect** predicts that risk-taking differs depending on whether prior outcomes are *paper* (still in play, mental account open) or *realized* (cashed out, mental account closed):

| Condition type | Paper prediction | Realized prediction |
|---|---|---|
| Loss | ↑ risk-taking (loss-chasing) | ↓ risk-taking (if large) |
| Gain | ↑ risk-taking (house money effect) | no change |

## Experimental design

Each trial presents an LLM with a casino vignette and asks for two responses: (1) how much to wager in the next slot machine session (1–1000 CHF total session wager), and (2) a slot machine risk preference (1–5 scale). The manipulation is the prior outcome history embedded in the vignette.

### Conditions

Conditions map to the quintile structure from Table 2 of the paper:

**Paper outcomes (within-visit — mental account open):**
| Condition | Amount (CHF) | Paper quintile |
|---|---|---|
| `paper_loss_large` | −350 | Q1 ≤ −310 |
| `paper_loss_medium` | −200 | Q2 −309 to −97 |
| `paper_loss_small` | −60 | Q3 −96 to 0 (small-loss sub-case) |
| `paper_even` | 0 | Q3 −96 to 0 (**baseline**) |
| `paper_gain_small` | +40 | Q4 1 to 80 |
| `paper_gain_large` | +150 | Q5 ≥ 81 |

**Realized outcomes (between-visits — mental account closed):**
| Condition | Amount (CHF) | Paper quintile |
|---|---|---|
| `realized_extreme_loss` | −3500 | Q1 ≤ −2,791 |
| `realized_large_loss` | −1800 | Q2 −2,790 to −788 |
| `realized_medium_loss` | −400 | Q3 −787 to −63 (not sig. in paper) |
| `realized_small_loss` | −30 | Q4 −62 to 0 (**baseline**) |
| `realized_gain` | +200 | Q5 ≥ 1 (not sig. in paper) |

### Key hypotheses being tested (from Flepp et al. 2021)

- **H1a/b**: Paper losses increase risk-taking; larger losses increase it more.
- **H2a/b**: Paper gains increase risk-taking; larger gains increase it more.
- **H3a/b**: Realized losses decrease risk-taking; only large realized losses are significant.
- **H4**: Realized gains do not alter risk-taking.

### Prompt framing

Prompt versions are implemented in `src/realization_effect/runner.py`:

- `absolute` (default): States win/loss amounts directly ("you have won/lost X CHF").
- `balance`: Frames outcomes as card balance relative to starting point.
- `qualitative`: Describes outcomes in relative terms only ("a modest amount", "a substantial amount") with no CHF figures — tests whether the effect holds without numeric anchoring.

The key distinction across all versions: **paper** scenarios specify the player is still in the casino (balance on card); **realized** scenarios specify the player cashed out and has returned for a new visit.

## Repository structure

```
realization-effect-project/
├── src/
│   ├── realization_effect/      # Prompting, running, parsing, analysis, reconciliation
│   ├── activation_analysis/     # Residual streams, prompt generation, vector analysis
│   └── sae/                     # Archived/supporting SAE utilities
├── scripts/                     # Preferred command-line entrypoints
├── tests/                       # Regression tests for parsing and analysis checks
│   └── fixtures/noncanonical/   # Local ignored archive of non-canonical CSVs
├── configs/
│   ├── realization_effect/      # Conditions and model catalogues
│   ├── activation_analysis/     # Activation-vector and archived prompt definitions
│   └── sae/                     # Archived SAE configs
├── experiments/
│   └── activation_analysis/      # Reviewable prompt CSVs for activation work
├── notebooks/realization_effect/ # Ordered exploratory notebooks
├── reports/                     # Current findings, midterm material, source papers
├── results/                     # Active canonical CSVs plus resumable block CSVs
└── models/                      # Local model weights, gitignored
```

The exploratory notebooks are numbered in the order they are most useful to
read:

- `01_experiment_design.ipynb`
- `02_results_merge_and_cleaning.ipynb`
- `03_multi_model_pilot.ipynb`
- `04_large_sample_7000_rows.ipynb`
- `05_large_sample_8000_rows.ipynb`
- `06_gpt54mini_haiku_comparison.ipynb`
- `07_kimi_grok_comparison.ipynb`

The current cleaned-results summary is in
`reports/current_findings.md`.

The current interpretability direction is the activation-vector realization/risk
experiment documented in `docs/activation_vector_realization_plan.md` and
`experiments/activation_analysis/README.md`. Earlier emotion-probe materials are
kept in `docs/emotion_probe_design.md` for reference.
The earlier SAE-first pass is archived under `docs/archive/20260506_sae_first_pass/`
and `configs/sae/archive/20260506_sae_first_pass/`.

Current activation-vector status:

- Prompt dataset: `experiments/activation_analysis/prompts/activation_vectors/realization_vector_v1.csv`
  contains 3,672 synthetic prompts across realization, control, and behavior
  evaluation cells.
- Local activation run: `results/final/residual_streams/realization_vector_v1_layer18_regions_float32`
  logs Gemma-3-4B layer 18 residual-stream activations for scenario tokens.
- Realization direction: `results/final/activation_vectors/realization_vector_v1_layer18/mean_direction.npy`
  is the mean paired direction from `realized_closed - paper_open`.
- Projection evaluation: `results/final/activation_vectors/realization_vector_v1_layer18/evaluation/prompt_projections.csv`
  shows the direction separates matched realization prompts strongly.
- Behavior evaluation: `results/final/activation_vectors/realization_vector_v1_layer18/behavior_eval.csv`
  stores free-generated wager/risk outputs from local Gemma on 648 behavior
  prompts, with paired summaries in
  `results/final/activation_vectors/realization_vector_v1_layer18/evaluation/behavior_eval_summary.json`.

The local Gemma-3-4B behavior link is currently weak: the realization direction
separates prompt representations, but generated wager/risk changes are small.
The next planned test is to repeat behavior generation and, if promising,
residual logging on a larger open model using cloud compute.

## Workflow

Examples below use `./venv/bin/python`, which works from a fresh shell in this
checkout. If you have activated the virtual environment, `python` is equivalent.

Common checks are available through `make`:

```bash
make test
make compile
make audit
make analyze
```

### 1. Inspect prompts before running

```bash
# Export every prompt text across all conditions and prompt versions
./venv/bin/python scripts/export_prompts.py --output prompts.csv

# One version only
./venv/bin/python scripts/export_prompts.py --version qualitative --output prompts_qualitative.csv
```

### 2. Run the experiment

```bash
export OPENROUTER_API_KEY=your_key_here

# Single model, 100 trials per condition
./venv/bin/python scripts/run_realization_experiment.py \
  --models openai/gpt-4o \
  --n-trials 100 \
  --prompt-version absolute

# Grid over multiple models and temperatures
./venv/bin/python scripts/run_realization_experiment.py \
  --models openai/gpt-4o anthropic/claude-3-5-sonnet \
  --temperatures 0.5 1.0 \
  --n-trials 100 \
  --shuffle
```

Runs are resumable: interrupted experiments can be restarted with the same command and already-completed trials are skipped.

The run script has one canonical write process:

1. Each model / temperature / prompt-version block writes to its own resumable CSV in `results/blocks/`.
2. After all selected blocks finish, the script performs one final reconciliation into `results/results.csv`.
3. `results/results_grouped.csv` is refreshed as the grouped companion file.

Avoid sidecar outputs for ordinary runs. Keep the default `--output results/results.csv`
unless you are intentionally creating a separate scratch dataset.

Current active data files:

- `results/results.csv` is the canonical analysis dataset.
- `results/sample_results.csv` is a small schema/sample extract for quick review.
- `results/results_grouped.csv` is generated by reconciliation for dashboard/analysis compatibility and is not tracked.
- `results/blocks/*.csv` are resumable per-block raw outputs used to rebuild the canonical CSV.
- Non-canonical historical CSVs are archived locally under `tests/fixtures/noncanonical/`.

If parser logic changes, audit or repair existing parsed columns without
re-querying models:

```bash
# Dry-run audit
./venv/bin/python scripts/reparse_realization_results.py results/results.csv results/blocks

# Rewrite parsed_wager, log_wager, risk_profile, and validity metadata
./venv/bin/python scripts/reparse_realization_results.py --write results/results.csv results/blocks
```

If you need to rebuild the canonical CSV from existing blocks without querying
models, run:

```bash
./venv/bin/python scripts/reconcile_realization_results.py
```

### 3. Generate synthetic prompts for activation logging

Use the activation prompt generator to create reviewable synthetic prompt CSVs
through OpenRouter. This does not write behavioral results to
`results/results.csv`; it writes prompt CSVs under
`experiments/activation_analysis/prompts/activation_vectors/`.

Run a small pilot before spending credits on the full prompt set:

```bash
export OPENROUTER_API_KEY=your_key_here

./venv/bin/python scripts/generate_activation_prompts.py \
  --pilot-all-cells \
  --output experiments/activation_analysis/prompts/activation_vectors/realization_vector_v1_pilot.csv
```

For full generation, run each source model into its own CSV. This lets separate
terminals work at the same time without racing on one shared file:

```bash
./venv/bin/python scripts/generate_activation_prompts.py \
  --models gpt54 \
  --output-template experiments/activation_analysis/prompts/activation_vectors/realization_vector_v1__{model}.csv \
  --resume

./venv/bin/python scripts/generate_activation_prompts.py \
  --models sonnet \
  --output-template experiments/activation_analysis/prompts/activation_vectors/realization_vector_v1__{model}.csv \
  --resume

./venv/bin/python scripts/generate_activation_prompts.py \
  --models grok_fast \
  --output-template experiments/activation_analysis/prompts/activation_vectors/realization_vector_v1__{model}.csv \
  --resume
```

After the model-specific files finish, merge them:

```bash
./venv/bin/python scripts/generate_activation_prompts.py \
  --merge-inputs \
    experiments/activation_analysis/prompts/activation_vectors/realization_vector_v1__gpt54.csv \
    experiments/activation_analysis/prompts/activation_vectors/realization_vector_v1__sonnet.csv \
    experiments/activation_analysis/prompts/activation_vectors/realization_vector_v1__grok_fast.csv \
  --output experiments/activation_analysis/prompts/activation_vectors/realization_vector_v1.csv
```

The active paired-contrast generation plan is
`configs/activation_analysis/realization_vector_generation_v1.json`. Its default
full output is
`experiments/activation_analysis/prompts/activation_vectors/realization_vector_v1.csv`.

### Fresh clone to analysis

```bash
python -m venv venv
./venv/bin/python -m pip install -e ".[dev]"
make test
make audit
make analyze
```

This path uses the tracked canonical dataset at `results/results.csv`.
`make audit` checks that the parsed wager/risk columns still match the current
parser without re-querying any model.

### 4. Log residual streams for activation-vector analysis

The `src/activation_analysis` package contains a Hugging Face forward-pass
adapter adapted from the metageniuses extraction code. It registers forward
hooks on selected transformer blocks and writes residual stream tensors plus
prompt metadata for later activation-vector analysis.

See `docs/forward_pass_plan.md` for the intended extraction contract.

Example smoke run against local Gemma files:

```bash
./venv/bin/python scripts/log_residual_streams.py \
  --model-id models/gemma-3-4b-pt \
  --layers 12,18 \
  --prompt-version absolute \
  --activation-site resid_post \
  --token-mode nonpad \
  --token-region-strategy auto \
  --include-token-regions scenario,decision_question \
  --storage-dtype float16 \
  --batch-size 1 \
  --limit 2 \
  --local-files-only \
  --run-name gemma3_4b_smoke
```

Outputs:

- `prompts.jsonl` — prompt text and condition metadata.
- `manifest.json` — model, layer, run, and shard metadata.
- `activations/layer_XX/batch_*.npy` — float16 tensors by default, shaped
  `[batch, sequence_length, d_model]`.
- `activations/layer_XX/batch_*.jsonl` — prompt IDs and token IDs aligned to
  each batch tensor.

Validate a completed run before using it downstream:

```bash
./venv/bin/python scripts/validate_activation_run.py \
  results/test/residual_streams/gemma3_4b_smoke
```

Inspect a validated run as a vector dataset. This only counts the vectors
available for later analysis:

```bash
./venv/bin/python scripts/inspect_activation_dataset.py \
  results/test/residual_streams/gemma3_4b_smoke \
  --layers 12 \
  --token-regions scenario,decision_question
```

Useful extraction options:

- `--block-path model.layers` forces hook placement when automatic model
  architecture detection is not enough.
- `--activation-site resid_post` records the current hook contract: residual
  stream output after a selected transformer block. `block_output` is an alias
  for the same current hook.
- `--token-mode all|nonpad|final` chooses whether to save every padded token,
  all non-padding tokens, or only the final non-padding token.
- `--token-region-strategy auto` labels saved tokens as regions such as
  `scenario`, `decision_question`, `response_instruction`, or
  `processing_instruction` without filtering out the broader activation data.
- `--include-token-regions scenario,decision_question` filters written activation
  shards to only those region labels, reducing storage for SAE-focused runs.
- `--storage-dtype float16|float32` controls saved tensor precision. The
  default is `float16` to reduce local storage; downstream analysis can cast
  vectors back to float32 when averaging or training.
- `--results-csv results/results.csv` attaches condition-level behavioral
  summaries to prompt metadata; use `--no-results-join` to skip this.
- If `--output-dir` is omitted, the script creates a deterministic run
  directory under `results/test/residual_streams/` unless `--output-dir` is provided.
Use `results/test/residual_streams/` for disposable smoke runs and
`results/final/residual_streams/` for current reference activation datasets.

Activation-vector analysis outputs should go under
`results/final/activation_vectors/`.

Build and evaluate a first realization direction:

```bash
./venv/bin/python scripts/build_activation_vectors.py \
  --activation-run results/final/residual_streams/realization_vector_v1_layer18_regions_float32 \
  --layers 18 \
  --output-dir results/final/activation_vectors/realization_vector_v1_layer18

./venv/bin/python scripts/evaluate_activation_vectors.py \
  --activation-run results/final/residual_streams/realization_vector_v1_layer18_regions_float32 \
  --direction results/final/activation_vectors/realization_vector_v1_layer18/mean_direction.npy \
  --layers 18 \
  --output-dir results/final/activation_vectors/realization_vector_v1_layer18/evaluation
```

Run and summarize free-generated behavior outputs for the behavior-evaluation
prompt split:

```bash
./venv/bin/python scripts/run_behavior_vector_eval.py \
  --model-id models/gemma-3-4b-pt \
  --local-files-only \
  --max-new-tokens 32 \
  --min-new-tokens 4 \
  --output results/final/activation_vectors/realization_vector_v1_layer18/behavior_eval.csv

./venv/bin/python scripts/reparse_behavior_vector_eval.py

./venv/bin/python scripts/analyze_behavior_vector_eval.py
```

This behavior pass uses free numeric generation. It does not force the model to
choose from discrete wager/risk buckets.

Current local Gemma summary:

- `648` behavior rows generated.
- `523` valid wager/allocation rows after reparsing.
- `518` valid risk rows after reparsing.
- `226` complete valid paper/realized pairs.
- Mean realized-minus-paper wager delta: `+13.96`.
- Mean realized-minus-paper risk delta: `-0.03`.
- Mean realized-minus-paper projection delta: `+149.14`.

Interpretation: the representation result is promising, but local Gemma-3-4B is
not yet strong behavioral evidence. Larger cloud-run open models are the next
check before making a causal claim.

### 5. Archived SAE-first pass

The earlier SAE-first branch of the interpretability work is preserved but is no
longer the active research path:

- configs: `configs/sae/archive/20260506_sae_first_pass/`
- docs: `docs/archive/20260506_sae_first_pass/`
- local outputs: `results/legacy/20260506_sae_first_pass/`
- imported Gemma Scope weights: `external/archive/20260506_sae_first_pass/saes/`

The reusable `src/sae/` package and inspection scripts remain in the repo in case
we later use SAEs as a supporting analysis rather than the main method.

### 6. Analyse results

```bash
# Full analysis, pooled across all models
./venv/bin/python scripts/analyze_realization_results.py results/results.csv

# Separate analysis per model
./venv/bin/python scripts/analyze_realization_results.py results/results.csv --per-model

# Filter to one model or one prompt version
./venv/bin/python scripts/analyze_realization_results.py results/results.csv --model openai/gpt-4o
./venv/bin/python scripts/analyze_realization_results.py results/results.csv --prompt-version qualitative
```

The analysis script outputs OLS regression tables (condition dummies + model/temperature/prompt_version fixed effects, HC3 robust SEs) for both `log(wager)` and `risk_profile`, and structured hypothesis verdicts for H1a–H4 — mirroring Table 2 of Flepp et al. (2021).

### 7. Monitor Block Progress (Live Dashboard)

```bash
# Launch local dashboard (default: http://127.0.0.1:8765)
./venv/bin/python scripts/block_dashboard.py

# Example for temp sweep phases (n=25 target per condition)
./venv/bin/python scripts/block_dashboard.py --target-trials 25 --refresh-seconds 5
```

If you use a virtual environment, launch with that interpreter (for example `./venv/bin/python scripts/block_dashboard.py`) so the Analysis tab can run with the same installed dependencies.

The dashboard reads `results/blocks/*.csv` and shows:
- per-block model/temperature/prompt version
- per-condition min/max run counts
- remaining runs to target
- active/idle status based on recent file updates

You can override values in the URL directly:
- `?target=25`
- `&refresh=5`
- `&active_window=90`

### 8. Run Analysis From Dashboard

Open the Analysis tab at:
- `http://127.0.0.1:8765/analyze`

From that page you can:
- select any `results/**/*.csv` dataset
- optionally filter by `model` and `prompt_version`
- toggle `--per-model`
- choose robust SE type (`HC0`–`HC3`)

Click **Run Analysis** to execute the analysis wrapper on demand and view output inline.

## Reference

Flepp, R., Meier, P., & Franck, E. (2021). The effect of paper outcomes versus realized outcomes on subsequent risk-taking: Field evidence from casino gambling. *Organizational Behavior and Human Decision Processes*, 165, 45–55. https://doi.org/10.1016/j.obhdp.2021.04.003
