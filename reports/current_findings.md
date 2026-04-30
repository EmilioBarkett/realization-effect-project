# Current Findings

This is the current read of the cleaned canonical dataset in
`results/results.csv`.

## Dataset

- Canonical valid trials: 32,251.
- Active analysis file: `results/results.csv`.
- Small review sample: `results/sample_results.csv`.
- Generated grouped companion: `results/results_grouped.csv`.
- Non-canonical historical data is archived locally under
  `tests/fixtures/noncanonical/` and is ignored by git.

## Analysis Snapshot

The analysis uses OLS regressions with condition dummies plus model,
temperature, and prompt-version fixed effects. Standard errors use HC3.

For `log_wager`, paper outcomes are mixed relative to the original human
casino pattern:

- `paper_loss_large`: +0.1208, significant.
- `paper_loss_medium`: -0.0039, not significant.
- `paper_loss_small`: -0.5604, significant.
- `paper_gain_small`: -0.7050, significant.
- `paper_gain_large`: -0.0637, significant.

For `risk_profile`, paper outcomes are more consistently positive:

- `paper_loss_large`: +0.0198, not significant.
- `paper_loss_medium`: +0.0308, significant.
- `paper_loss_small`: +0.0568, significant.
- `paper_gain_small`: +0.1203, significant.
- `paper_gain_large`: +0.2436, significant.

For realized outcomes, the LLM responses do not match the original paper's
large-loss prediction. Realized losses increase `log_wager` instead of
decreasing it:

- `realized_extreme_loss`: +0.4803, significant.
- `realized_large_loss`: +0.4255, significant.
- `realized_medium_loss`: +0.1817, significant.
- `realized_gain`: -0.2202, significant.

The `risk_profile` results also move upward for realized outcomes:

- `realized_extreme_loss`: +0.1877, significant.
- `realized_large_loss`: +0.1758, significant.
- `realized_medium_loss`: +0.0377, significant.
- `realized_gain`: +0.1399, significant.

## Interpretation

The cleaned results suggest that language models partially reproduce the
paper-outcome risk shift on the 1-5 risk-profile measure, but not cleanly on
wager size. For realized outcomes, the model behavior diverges more sharply
from the human result: realized losses increase risk-taking, and realized gains
also differ from baseline.

The earlier parser issue that confused numbered answer lines with actual
answers has been repaired, and the active CSVs have been audited with the new
parser.
