# Realization Effect Replication — LLM Study

Replication of Flepp, Meier & Franck (2021) *"The effect of paper outcomes versus realized outcomes on subsequent risk-taking: Field evidence from casino gambling"* (OBHDP 165, 45–55), substituting language models for human subjects.

## What this study tests

The **realization effect** predicts that risk-taking differs depending on whether prior outcomes are *paper* (still in play, mental account open) or *realized* (cashed out, mental account closed):

| Condition type | Paper prediction | Realized prediction |
|---|---|---|
| Loss | ↑ risk-taking (loss-chasing) | ↓ risk-taking (if large) |
| Gain | ↑ risk-taking (house money effect) | no change |

## Experimental design

Each trial presents an LLM with a casino vignette and asks for two responses: (1) how much to wager in the next slot machine session (1–1000 CHF total session wager), and (2) a slot machine risk preference (1–5 scale). The manipulation is the prior outcome history embedded in the vignette.

### Conditions (`conditions.csv`)

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

Two prompt versions are implemented in `run_experiment.py`:

- `absolute` (default): States win/loss amounts directly ("you have won/lost X CHF").
- `balance`: Frames outcomes as card balance relative to starting point.
- `qualitative`: Describes outcomes in relative terms only ("a modest amount", "a substantial amount") with no CHF figures — tests whether the effect holds without numeric anchoring.

The key distinction across all versions: **paper** scenarios specify the player is still in the casino (balance on card); **realized** scenarios specify the player cashed out and has returned for a new visit.

## Repository structure

```
realization-effect-project/
├── conditions.csv          # 11 experimental conditions (mapped to paper's quintiles)
├── run_experiment.py       # Data collection: calls models via OpenRouter API
├── analyze_results.py      # Statistical analysis: OLS regressions + hypothesis tests
├── generate_prompts.py     # Utility: export all prompt texts to CSV for inspection
├── reorganize_csv.py       # Utility: reorder/split result CSVs by column
├── llm_eval_list_v2.csv    # Catalogue of LLMs included in the study
├── notebooks/              # Jupyter notebooks for exploratory analysis
│   ├── Experiment.ipynb
│   ├── Multiple models test.ipynb
│   ├── 5.4mini+haiku.ipynb
│   ├── Kimi+Grok.ipynb
│   ├── 7000 row .ipynb
│   ├── Merged CSV.ipynb
│   └── Trial 4 - 8000 rows.ipynb
└── results/
    ├── results.csv           # Canonical merged dataset used for analysis
    ├── results_grouped.csv   # Grouped companion refreshed during reconcile
    ├── blocks/               # Canonical per-block CSVs for all prompt versions
    ├── balance/              # Sidecar staging outputs for concurrent balance runs
    └── legacy/               # Legacy/old-prompt rows and partition reports
```

## Workflow

### 1. Inspect prompts before running

```bash
# Export every prompt text across all conditions and prompt versions
python generate_prompts.py --output prompts.csv

# One version only
python generate_prompts.py --version qualitative --output prompts_qualitative.csv
```

### 2. Run the experiment

```bash
export OPENROUTER_API_KEY=your_key_here

# Single model, 100 trials per condition
python run_experiment.py \
  --models openai/gpt-4o \
  --n-trials 100 \
  --prompt-version absolute

# Grid over multiple models and temperatures
python run_experiment.py \
  --models openai/gpt-4o anthropic/claude-3-5-sonnet \
  --temperatures 0.5 1.0 \
  --n-trials 100 \
  --shuffle
```

Runs are resumable: interrupted experiments can be restarted with the same command and already-completed trials are skipped.

When you run sidecar jobs (for example writing to `results/balance/results.csv`), reconcile them into canonical outputs:

```bash
./venv/bin/python reconcile_results.py
```

This command:
- copies sidecar block files (default: `results/balance/blocks`) into `results/blocks`
- refreshes canonical `results/results.csv` and `results/results_grouped.csv`
- partitions legacy prompt-structure rows (including early non-canonical runs such as initial 4.1-style prompts) into `results/legacy/results_legacy.csv`

If you only want partitioning without copying sidecar blocks first:

```bash
./venv/bin/python partition_results.py
```

### 3. Analyse results

```bash
# Full analysis, pooled across all models
python analyze_results.py results/results.csv

# Separate analysis per model
python analyze_results.py results/results.csv --per-model

# Filter to one model or one prompt version
python analyze_results.py results/results.csv --model openai/gpt-4o
python analyze_results.py results/results.csv --prompt-version qualitative
```

The analysis script outputs OLS regression tables (condition dummies + model/temperature/prompt_version fixed effects, HC3 robust SEs) for both `log(wager)` and `risk_profile`, and structured hypothesis verdicts for H1a–H4 — mirroring Table 2 of Flepp et al. (2021).

### 4. Monitor Block Progress (Live Dashboard)

```bash
# Launch local dashboard (default: http://127.0.0.1:8765)
python block_dashboard.py

# Example for temp sweep phases (n=25 target per condition)
python block_dashboard.py --target-trials 25 --refresh-seconds 5
```

If you use a virtual environment, launch with that interpreter (for example `./venv/bin/python block_dashboard.py`) so the Analysis tab can run with the same installed dependencies.

The dashboard reads `results/blocks/*.csv` and shows:
- per-block model/temperature/prompt version
- per-condition min/max run counts
- remaining runs to target
- active/idle status based on recent file updates

You can override values in the URL directly:
- `?target=25`
- `&refresh=5`
- `&active_window=90`

### 5. Run Analysis From Dashboard

Open the Analysis tab at:
- `http://127.0.0.1:8765/analyze`

From that page you can:
- select any `results/**/*.csv` dataset
- optionally filter by `model` and `prompt_version`
- toggle `--per-model`
- choose robust SE type (`HC0`–`HC3`)

Click **Run Analysis** to execute `analyze_results.py` on demand and view output inline.

## Reference

Flepp, R., Meier, P., & Franck, E. (2021). The effect of paper outcomes versus realized outcomes on subsequent risk-taking: Field evidence from casino gambling. *Organizational Behavior and Human Decision Processes*, 165, 45–55. https://doi.org/10.1016/j.obhdp.2021.04.003
