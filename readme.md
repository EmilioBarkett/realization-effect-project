# Representation Without Control: Testing the Realization Effect in Language Models

**Ciarán Walsh and Emilio Barkett** — Columbia University

Code and data for the paper *Representation Without Control: Testing the Realization Effect in Language Models* (SPAR Spring 2026 / preprint).

Paper: [`reports/Realization_Effect_in_Language_Models____Ciaran____Emilio.pdf`](reports/Realization_Effect_in_Language_Models____Ciaran____Emilio.pdf)

## Overview

The **realization effect** is a behavioral-economics finding in which risk-taking depends on whether prior gains or losses remain *paper* (open mental account) or have been *realized* (closed account). We use it as a test case for a broader question about LLM behavioral research: does condition-sensitive behavior imply genuine internal implementation of the underlying construct?

We evaluate LLM behavior at three levels:

1. **Behavioral** — Do prompt-only responses show the pattern of condition sensitivity that human subjects exhibit?
2. **Readout** — Is realization status linearly decodable from residual-stream activations, including on held-out prompts?
3. **Steering** — Does adding a decoded realization direction during inference causally shift downstream risk choices?

**Core finding:** Models are condition-sensitive, but the directional pattern does not reproduce the human realization-effect predictions. Gemma's residual stream contains a linearly decodable realization-status signal at layer 18 that generalizes to held-out prompts. Steering along that direction does not, however, reliably shift downstream risk choices — a null result that holds across positive scales and in a negative sign-symmetry run. Behavioral sensitivity, latent readout, and causal control are three distinct properties that do not automatically co-occur.

## Results summary

| Test | Outcome | Evidence |
|---|---|---|
| Prompt-only behavioral replication | Weak support | Condition-sensitive, but directional pattern does not match human realization-effect predictions |
| Activation readout (layer-18 direction) | Supported | Train-only direction separates realized/closed from paper/open on original and DeepSeek held-out splits |
| Projection strength vs. behavior | Not supported | Within-pair projection deltas do not predict wager or risk deltas after controlling for prompt structure |
| Risk-behavior steering intervention | Not supported | Small mean shifts, zero medians, no sign-symmetric reversal |
| Positive-control classification steering | Weak diagnostic | Directional rate shift occurs; accuracy near chance with strong PAPER prediction bias |

The prompt-only dataset covers **54,450 rows** across **25 models** (53,547 valid wagers; 49,351 valid risk-profile responses). The activation analysis uses **Gemma 3 4B**, a train-only layer-18 direction built from **756 matched pairs**.

## Repository structure

```
realization-effect-project/
├── src/
│   ├── realization_effect/      # Prompting, running, parsing, analysis, reconciliation
│   ├── activation_analysis/     # Residual streams, prompt generation, vector analysis
│   └── sae/                     # Archived SAE utilities
├── scripts/                     # Command-line entrypoints for all three stages
├── tests/                       # Regression tests for parsing and analysis
├── configs/
│   ├── realization_effect/      # Conditions (11) and model catalogue
│   ├── activation_analysis/     # Activation-vector prompt generation configs
│   └── sae/                     # Archived SAE configs
├── experiments/
│   └── activation_analysis/     # Reviewable prompt CSVs for activation work
├── notebooks/realization_effect/ # Exploratory notebooks (01–07)
├── reports/
│   ├── Realization_Effect_in_Language_Models____Ciaran____Emilio.pdf  # preprint
│   ├── final/                   # SPAR report source (report.tex, sparreport.cls, figures/)
│   └── papers/                  # Source papers (Flepp et al., Imas, etc.)
├── results/
│   ├── sample_results.csv       # Schema sample for the behavioral dataset (tracked)
│   └── final/                   # Tracked activation vectors and evaluation summaries
└── docs/                        # Architecture notes and planning documents
```

**Key tracked data artifacts:**

- [`results/sample_results.csv`](results/sample_results.csv) — schema sample for the behavioral dataset
- [`results/final/activation_vectors/realization_vector_v1_layer18_direction_train_only/`](results/final/activation_vectors/realization_vector_v1_layer18_direction_train_only/) — train-only layer-18 mean direction (`mean_direction.npy`) and held-out evaluation summaries
- [`experiments/activation_analysis/prompts/activation_vectors/realization_vector_v1.csv`](experiments/activation_analysis/prompts/activation_vectors/realization_vector_v1.csv) — 3,672 synthetic prompts used for activation logging
- [`experiments/activation_analysis/prompts/activation_vectors/realization_vector_heldout_v1.csv`](experiments/activation_analysis/prompts/activation_vectors/realization_vector_heldout_v1.csv) — DeepSeek-authored held-out prompt set (40 pairs)

The full behavioral results CSV (`results/results.csv`) and all local activation-run outputs are gitignored and local-only.

## Stage 1: Behavioral prompting

Each trial presents an LLM with a casino vignette and elicits two responses: a next-session wager (1–1000 CHF) and a slot-machine risk preference (1–5). Eleven conditions map to the quintile structure from Table 2 of Flepp et al. (2021).

**Paper conditions** (within-visit, mental account open):

| Condition | CHF | Paper quintile |
|---|---|---|
| `paper_loss_large` | −350 | Q1 ≤ −310 |
| `paper_loss_medium` | −200 | Q2 −309 to −97 |
| `paper_loss_small` | −60 | Q3 small-loss sub-case |
| `paper_even` | 0 | Q3 **baseline** |
| `paper_gain_small` | +40 | Q4 1–80 |
| `paper_gain_large` | +150 | Q5 ≥ 81 |

**Realized conditions** (between-visits, mental account closed):

| Condition | CHF | Paper quintile |
|---|---|---|
| `realized_extreme_loss` | −3500 | Q1 ≤ −2,791 |
| `realized_large_loss` | −1800 | Q2 −2,790 to −788 |
| `realized_medium_loss` | −400 | Q3 −787 to −63 |
| `realized_small_loss` | −30 | Q4 **baseline** |
| `realized_gain` | +200 | Q5 ≥ 1 |

Two prompt versions were used in the main analysis: `absolute` (CHF amounts stated explicitly) and `balance` (card balance framing). A `qualitative` version uses relative descriptors with no CHF figures.

Main regressions use condition indicators with model, temperature, and prompt-version fixed effects, and HC3 robust standard errors. Paper conditions are compared against a `paper_even` baseline; realized conditions against a `realized_small_loss` baseline.

### Running the behavioral experiment

```bash
export OPENROUTER_API_KEY=your_key_here

./venv/bin/python scripts/run_realization_experiment.py \
  --models openai/gpt-4o anthropic/claude-3-5-sonnet \
  --temperatures 0.5 1.0 \
  --n-trials 100 \
  --prompt-version absolute \
  --shuffle
```

Runs are resumable. Each model/temperature/prompt-version block writes to `results/blocks/`; a final step reconciles them into `results/results.csv`.

### Analysing behavioral results

```bash
# Pooled across all models
./venv/bin/python scripts/analyze_realization_results.py results/results.csv

# Per-model
./venv/bin/python scripts/analyze_realization_results.py results/results.csv --per-model

# Filter to one model or prompt version
./venv/bin/python scripts/analyze_realization_results.py results/results.csv --model openai/gpt-4o
./venv/bin/python scripts/analyze_realization_results.py results/results.csv --prompt-version qualitative
```

## Stage 2: Activation readout

Paired prompts contrast `paper_open` and `realized_closed` framings across several domains (finance, reimbursement, budget, compensation, academic, project-outcome, casino). A mean-difference realization direction is computed at layer 18 of local Gemma 3 4B:

```
v_realization = μ(realized_closed) − μ(paper_open)
```

The **train-only direction** is built from the `direction_train` split (756 matched pairs) and is the direction used for all generalization tests and steering interventions reported as main results. The all-pairs direction appears only in descriptive plots (labeled accordingly).

### Logging residual streams

```bash
./venv/bin/python scripts/log_residual_streams.py \
  --model-id models/gemma-3-4b-pt \
  --layers 18 \
  --activation-site resid_post \
  --token-mode nonpad \
  --include-token-regions scenario,decision_question \
  --storage-dtype float16 \
  --local-files-only \
  --run-name gemma3_4b_layer18
```

### Building and evaluating the realization direction

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

Held-out readout results (Table 1 of the paper) are in:
`results/final/activation_vectors/realization_vector_v1_layer18_direction_train_only/summary_tables/`

## Stage 3: Activation steering

A forward hook adds a scaled, normalized realization direction at layer 18 during Gemma generation. Positive scales push toward `realized_closed`; negative scales toward `paper_open`. All main steering results use the train-only direction.

```bash
./venv/bin/python scripts/steer_realization_direction.py \
  --model-id models/gemma-3-4b-pt \
  --direction results/final/activation_vectors/realization_vector_v1_layer18_direction_train_only/mean_direction.npy \
  --scales -50 0 50 75 100 150 \
  --layer 18 \
  --local-files-only \
  --run-name gemma3_4b_steering_v1
```

The steering architecture is documented in [`docs/steering_architecture.md`](docs/steering_architecture.md).

## Reproducing paper figures and tables

```bash
./venv/bin/python scripts/build_report_figures.py
./venv/bin/python scripts/build_behavioral_report_tables.py
./venv/bin/python scripts/summarize_steering_report_tables.py
```

## Setup

```bash
python -m venv venv
./venv/bin/python -m pip install -e ".[dev]"
make test
```

Common checks:

```bash
make test      # regression tests
make compile   # type-check
make lint      # ruff
make audit     # verify parsed columns match current parser (no API calls)
make analyze   # run analysis on results/results.csv
```

`make test` and `make audit` run against `results/sample_results.csv` (tracked) and do not require the full local dataset.

## Exploratory notebooks

[`notebooks/realization_effect/`](notebooks/realization_effect/), numbered in reading order:

- `01_experiment_design.ipynb` — condition structure and prompt design
- `02_results_merge_and_cleaning.ipynb` — data cleaning and merging
- `03_multi_model_pilot.ipynb` — early multi-model results
- `04_large_sample_7000_rows.ipynb` — scale-up analysis
- `05_large_sample_8000_rows.ipynb` — further scale-up
- `06_gpt54mini_haiku_comparison.ipynb` — GPT-4.1-mini vs Haiku
- `07_kimi_grok_comparison.ipynb` — Kimi vs Grok

## Citation

```bibtex
@article{walsh2026representation,
  title   = {Representation Without Control: Testing the Realization Effect in Language Models},
  author  = {Walsh, Ciar{\'a}n and Barkett, Emilio},
  year    = {2026},
  note    = {Supervised Program for Alignment Research, Spring 2026}
}
```

## References

- Flepp, R., Meier, P., & Franck, E. (2021). The effect of paper outcomes versus realized outcomes on subsequent risk-taking. *OBHDP*, 165, 45–55.
- Imas, A. (2016). The realization effect: Risk-taking after realized versus paper losses. *AER*, 106(8), 2086–2109.
- Merkle, C., Müller-Dethard, J., & Weber, M. (2021). Closing a mental account: The realization effect for gains and losses. *Experimental Economics*, 24(1), 303–329.
- Turner et al. (2023). Steering language models with activation engineering. *arXiv:2308.10248*.
- Zou et al. (2023). Representation engineering: A top-down approach to AI transparency. *arXiv:2310.01405*.
- Park, K., Choe, Y. J., & Veitch, V. (2024). The linear representation hypothesis and the geometry of large language models. *ICML 2024*.
