# Current Findings

This is the current read of the cleaned canonical dataset in
`results/results.csv`.

## Dataset

- Canonical rows: 54,450.
- Valid wager rows: 53,547.
- Valid risk-profile rows: 49,351.
- Prompt versions: `absolute` and `balance`.
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

- `paper_loss_large`: +0.1530, significant.
- `paper_loss_medium`: -0.0121, not significant.
- `paper_loss_small`: -0.5804, significant.
- `paper_gain_small`: -0.7065, significant.
- `paper_gain_large`: -0.0269, significant.

For `risk_profile`, paper outcomes are mostly positive:

- `paper_loss_large`: +0.0666, significant.
- `paper_loss_medium`: +0.0223, not significant.
- `paper_loss_small`: -0.0116, not significant.
- `paper_gain_small`: +0.1114, significant.
- `paper_gain_large`: +0.2527, significant.

For realized outcomes, the LLM responses do not match the original paper's
large-loss prediction. Realized losses increase `log_wager` instead of
decreasing it:

- `realized_extreme_loss`: +0.6463, significant.
- `realized_large_loss`: +0.6271, significant.
- `realized_medium_loss`: +0.3721, significant.
- `realized_gain`: +0.0965, significant.

The `risk_profile` results also move upward for realized outcomes:

- `realized_extreme_loss`: +0.1358, significant.
- `realized_large_loss`: +0.1501, significant.
- `realized_medium_loss`: +0.0204, not significant.
- `realized_gain`: +0.2923, significant.

## Interpretation

The cleaned results suggest that language models show some paper-outcome risk
movement on the 1-5 risk-profile measure, but the pattern is not clean and does
not carry over to wager size. For realized outcomes, the model behavior
diverges more sharply from the human result: realized losses increase
risk-taking, and realized gains also differ from baseline.

The earlier parser issue that confused numbered answer lines with actual
answers has been repaired, and the active CSVs have been audited with the new
parser.
