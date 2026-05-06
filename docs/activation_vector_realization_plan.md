# Activation-Vector Realization Plan

The current interpretability direction is to adapt the Anthropic-style
activation-vector method to realization framing and risk-taking behavior.

## Core Question

Can we identify an internal realization/open-vs-closed direction, show that it is
distinct from risk and emotion, and test whether moving along that direction
changes risk-taking behavior?

## Domain Set

Use domains where "paper/open" versus "realized/closed" is natural and
real-world meaningful:

- `finance_investments`: unrealized portfolio gain/loss vs sold position.
- `gambling_casino`: chips/card balance still in play vs cashed out/paid.
- `budget_spending`: planned/available budget change vs money actually spent or
  received.
- `refund_reimbursement`: approved/pending refund vs refund deposited.
- `compensation_bonus`: expected bonus/commission vs paid bonus/commission.
- `academic_grades`: provisional score/grade estimate vs final posted grade.
- `project_outcomes`: tentative project result vs officially closed outcome.

Finance/investments and gambling should be the main behavioral domains. The
others are controls that help separate realization/finality from casino words,
money words, and explicit risk-taking.

## Splits

- `direction_train`: paired prompts used to compute realization directions.
- `direction_val`: held-out paired prompts with different wording/domains.
- `confound_control`: prompts isolating tense, finality, money vocabulary,
  outcome valence, and source-model style.
- `behavior_eval`: gambling/risk-choice prompts used to test prediction and
  steering effects.

## File Layout

- Generation plans/prompts-to-generate-prompts:
  `configs/activation_analysis/realization_vector_generation_v1.json`
- Generated synthetic prompt CSVs:
  `experiments/activation_analysis/prompts/activation_vectors/`
- Residual activation logs from those prompts:
  `results/final/residual_streams/`
- Activation-vector analysis outputs:
  `results/final/activation_vectors/`
- Archived SAE-first materials:
  `docs/archive/20260506_sae_first_pass/`,
  `configs/sae/archive/20260506_sae_first_pass/`, and
  `results/legacy/20260506_sae_first_pass/`

## Required Metadata

Each generated prompt row should include:

- `prompt_id`
- `split`
- `pair_id`
- `domain`
- `realization_frame`: `paper_open`, `realized_closed`, or `control`
- `outcome_valence`: `gain`, `loss`, or `neutral`
- `amount_bucket`
- `risk_context`: `none`, `risk_seeking`, `risk_avoidant`, or `mixed`
- `emotion`
- `confound_axis`
- `source_llm`
- `prompt_text`

## First Analysis Path

1. Log residual activations for `direction_train` and `direction_val`.
2. Compute realization vectors from paired contrasts.
3. Project out neutral/confound directions where needed.
4. Validate on held-out domains.
5. Run behavioral prompts and test whether vector projection predicts wager and
   risk preference.
6. Steer with the realization direction and measure whether behavior changes.

SAEs are archived for now and should be treated as optional supporting analysis,
not the main method.
