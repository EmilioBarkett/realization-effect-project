# Wave 1 prompt-family audit

Date: 2026-08-27

Status: engineering audit only. No API calls, model-weight downloads, or RunPod
jobs were started for this audit. The existing RunPod pod remains stopped.

This audit applies the same checks to all four Wave 1 families:

- probe-pair isolation and construct validity;
- downstream task independence;
- duplicates, leakage, balance, ordering, and response-collapse risk;
- zero-dose variation and provenance required before a full steering run.

The existing v1 artifacts are preserved. Findings below are blockers or
pre-registration requirements, not empirical benchmark results.

## Executive summary

| Construct | Probe | Downstream task | Zero-dose risk | RunPod status |
| --- | --- | --- | --- | --- |
| realization/account closure | Structurally clean, but heavily lexicalized as administrative finality | Generic risk choice; primary rating is underidentified | Only 8 non-neutral steering items; 1--5 midpoint collapse is plausible | Do not treat v1 as confirmatory |
| evidence diagnosticity | Strong numeric LR contrast, with small split imbalance and fixed ordering | Near-transfer: still asks for arithmetic belief updating | Nominal schedule varies, but actual model variation is unverified | Use only with explicit downstream-independence caveat or redesign |
| source reliability | Exact five-report contrast is well controlled | Confounds source history with current evidence, authority, and testimony | Full schedule has 32 items; test selection of 2 is too small | Add entity/evidence controls before full run |
| persistence/continuation | v1 probe is structurally controlled | v1 task measures default incumbent allocation, not renewal after failure | Observed zero-dose outcome is constant (90/10); v1 scorer correctly refuses | Use the versioned v2 redesign below |

The common operational conclusion is: do not restart a full RunPod run from the
current v1 inventories. First run a bounded model-side test with enough
steering items to estimate zero-dose variation, audit the completed manifest,
and require the variation gate to pass before releasing a full run.

## 1. Realization/account closure

Audited paths:

- `configs/construct_benchmark/constructs/realization_account_closure_v1.json`
- `configs/construct_benchmark/generation_plans/wave1_realization_account_closure_v1.json`
- `results/benchmark/vector_prompts_v2_luna/full_final_all16/combined.csv`
- `results/benchmark/downstream_prompts_v1_wave1_full_luna_current_b50_o60k_v3/realization_account_closure.csv`

Inventory:

- 180 paired probe items (100/40/40 pairs across train/validation/held-out);
- 36 downstream items (12 behavior, 12 steering, 12 calibration).

What passed:

- no cross-split exact overlap;
- no within-pair numeric or entity changes;
- no sentence-count changes;
- no exact role-pool duplicates.

What remains risky:

- Most pairs directly contrast words such as `provisional/unresolved` with
  `finalized/resolved`, `settled`, or `complete`. A direction can therefore
  decode lexical or administrative finality rather than the intended
  realization/account-closure state.
- The downstream task is an indirect generic risk choice. The primary
  `risk_preference` rating is only 1--5, while the actual allocation is
  secondary, and the parser does not require the two fields to agree.
- Sure/risky option order and output-field order are fixed. Unequal
  probabilities also change expected value, so risk preference is not isolated.
- Only eight of twelve steering items are non-neutral for the directed primary
  outcome; the neutral cells do not contribute to its zero-dose denominator.

Required before a confirmatory claim:

1. Add a finite status-clause lexicon, status-template-family metadata, and
   held-out paraphrase families.
2. Add an administrative-finality/lexical control.
3. Make risky allocation the primary outcome, or define a keyed and
   consistency-checked joint response.
4. Counterbalance option and output order and either equalize expected value or
   register payoff asymmetry as a nuisance factor.
5. Increase replicated non-neutral steering cells and apply the behavioral
   variation gate.

The existing v1 realization run can remain a labeled engineering/reference
artifact; these changes should be versioned rather than silently applied to its
frozen hashes.

## 2. Evidence diagnosticity

Audited paths:

- `configs/construct_benchmark/constructs/evidence_diagnosticity_v1.json`
- `configs/construct_benchmark/generation_plans/wave1_evidence_diagnosticity_v1.json`
- `results/benchmark/vector_prompts_v2_luna/full_final_all16/combined.csv`
- `results/benchmark/downstream_prompts_v1_wave1_full_luna_current_b50_o60k_v3/evidence_diagnosticity.csv`

Inventory:

- 180 paired probe contrasts (100/40/40);
- 18 behavior, 18 steering, and 18 calibration items.

What passed:

- every probe pair changes the numerical frequency evidence while preserving
  the scenario core and evidence polarity;
- high and low likelihood-ratio thresholds are satisfied;
- no exact or normalized duplicate prompts were found;
- the downstream schedule is balanced across priors, evidence valence, and
  diagnosticity.

What remains risky:

- The probe and downstream task both ask the model to compute a
  hypothesis-versus-alternative likelihood ratio and posterior. New wording
  and entities do not make this a genuinely independent behavior task; it is
  better described as near-transfer or a positive control.
- Focal-hypothesis order and frequency order are fixed. Numeric token patterns
  can be exploited without representing diagnosticity.
- Train and held-out supporting/contradicting counts are not exactly balanced.
- The declared posterior-calibration and confidence outcomes are not
  identified by the current single-integer parser/scorer.
- A prior-copying strategy makes absolute update zero for every item. The
  nominal item schedule has variation, but this does not establish model-side
  zero-dose variation.

Required before a confirmatory claim:

1. Add a dedicated evidence audit for denominators, likelihood ratios,
   thresholds, polarity, pair invariants, and split balance.
2. Counterbalance hypothesis/frequency order and register numeric-token nuisance
   factors.
3. Either create a genuinely independent downstream task or explicitly label
   the current task a near-transfer/positive-control analysis.
4. Add expected posterior/LR metadata and implement the declared calibration
   outcome, or remove the unimplemented secondary claims.
5. Run the completed zero-dose variation gate; never use neutral calibration
   items as the directed-outcome denominator.

## 3. Source reliability

Audited paths:

- `configs/construct_benchmark/constructs/source_reliability_v1.json`
- `configs/construct_benchmark/generation_plans/wave1_source_reliability_v1.json`
- `results/benchmark/vector_prompts_v2_luna/full_final_all16/combined.csv`
- `results/benchmark/downstream_prompts_v1_wave1_full_luna_current_b50_o60k_v3/source_reliability.csv`

Inventory:

- 180 paired probe contrasts (100/40/40);
- 32 behavior, 32 steering, and 32 calibration items.

What passed:

- paired probes use five historical reports with 4/1 versus 1/4 accuracy;
- the minority report position is balanced across the registered positions and
  across direction splits;
- parser arithmetic and downstream factorial schedules are explicit;
- no exact or normalized duplicate downstream prompts were found.

What remains risky:

- Probe text can expose explicit accuracy counts and current evidence, so a
  direction may encode report-count or claim-reliability markers rather than
  source reliability.
- The downstream task combines source history, current evidence quality,
  testimony polarity, authority status, and prior probability. Current
  evidence is usually aligned with testimony, so source weighting is not
  isolated from evidence weighting or generic credibility.
- Authority credentials are a second route to the same response.
- Historical entities/events are reused between probe and downstream pools,
  including contradictory identities in some roles. This is semantic leakage
  even where full text is not duplicated.
- The full steering schedule has 32 items, but the default test selection of
  two items is not a meaningful variation pilot.

Required before a confirmatory claim:

1. Add history-only/neutral-claim and matched-current-evidence controls.
2. Make source names, events, and report identities globally unique across
   probe and downstream pools; add an entity/event leakage checker.
3. Balance and register minority positions, testimony polarity, authority, and
   current evidence quality as separate nuisance factors.
4. Increase the model-side test selection to at least 8--12 steering items
   per construct, then require the completed variation gate.
5. Freeze one effective generation model and plan hash. The existing plan
   metadata names Sonnet while current downstream artifacts were generated
   through the Luna workflow; this is a provenance ambiguity, not evidence
   that the rows are invalid.

## 4. Persistence/continuation

The v1 audit found the only observed model-side failure in the prior Wave 1
gate:

- every persistence zero-dose target outcome parsed as `existing_program_allocation=90`;
- the scorer rejected the run because the zero-dose sample SD was zero.

This is a useful fail-closed result, not a null steerability result. The v1
downstream task did not include a fresh underperformance/renewal context,
quantitative return tradeoffs, or explicit switching costs. It measured default
incumbent allocation and was especially vulnerable to deterministic `90/10`
completion.

A versioned `persistence_continuation_v2` specification and generation plan
are being added without modifying the v1 files. The v2 task:

- describes a fresh goal-renewal decision after a recent shortfall;
- includes numerical expected-return differences and switching costs;
- uses a single integer from 0--100 for the established-goal allocation;
- counterbalances narrative option order;
- schedules return advantage, repairability, setback severity, switching cost,
  and option order explicitly;
- keeps behavior, steering, and calibration domains separate.

The v2 inventory is a candidate until its generated prompts pass review. It
must not replace the v1 registry or any frozen v1 manifest automatically.

## Common RunPod release gate

Before a full run for any family:

1. Freeze the exact spec, generation plan, prompt inventory, effective model,
   tokenizer revision, activation layer/site, timing, and run-config hashes.
2. Run the local deterministic/fake vertical slice.
3. Run a bounded model-side test using a completed output manifest. The test
   must include at least 8--12 steering items per construct; two items are
   insufficient. The repository supplies
   `configs/construct_benchmark/run_configs/wave1_four_construct_variation_gate_v1.json`
   for this purpose.
4. Run `scripts/audit_behavioral_variation.py` on each completed target
   output. It fails closed on missing manifests, incomplete outputs, invalid
   rows, constant outcomes, or non-positive zero-dose SD.
5. Inspect parser compliance, injection manipulation checks, downstream
   tracking, collateral controls, and prompt-only baseline before full release.
6. Use the full inventory only after the gate passes. Keep raw outputs and
   manifests on the persistent workspace and copy checksummed artifacts to
   the configured archive.

A failed variation gate blocks a full run. It does not justify adding noise,
changing sampling after seeing the outcome, or substituting an epsilon for a
zero denominator.

## Provenance and artifact policy

- Existing v1 configs and outputs are preserved.
- This audit is not an empirical result and must not be cited as one.
- Any redesign is versioned; do not edit a spec after collecting a run whose
  manifest hashes it.
- RunPod remains stopped until a reviewed candidate inventory and test plan
  are ready.
