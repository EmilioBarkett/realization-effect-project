# Wave 2–4 prompt audit

Date: 2026-08-27

Status: **audited, not released for confirmatory execution**

This document records the audit of the existing Luna prompt inventories for
Waves 2–4. The inventories are preserved as historical engineering artifacts.
They are not rewritten in place and must not be used as confirmatory inputs.

## Scope and evidence

The audit covered the combined full inventories at:

- `results/benchmark/prompt_inventories/wave2_four_construct_full_luna_v1/combined.csv`
- `results/benchmark/prompt_inventories/wave3_four_construct_full_luna_v1/combined.csv`
- `results/benchmark/prompt_inventories/wave4_four_construct_full_luna_v1/combined.csv`

It used the schema-aware downstream validator, pair/count/duplicate checks,
the registered probe-wrapper check, and the new construct-specific semantic
checks in `scripts/audit_wave_prompt_inventories.py`. Independent read-only
reviews were also obtained from subagents for each wave. No model weights,
RunPod process, or external model execution was used for this audit.

The mechanical inventories are otherwise well formed: Wave 2 has 1,584 rows,
Wave 3 has 1,584 rows, and Wave 4 has 1,536 rows. Their split counts and
construct namespaces are balanced, and the released inventories have no
global normalized text duplicates. Those checks do not establish scientific
independence.

## Wave 2

Constructs: `reference_frame`, `prior_weighting`, `authority_deference`, and
`exploration_exploitation`.

### Blockers

- `reference_frame` steering prompts contain an explicit comparison-point
  scenario before the nominal downstream sure/risky task. This makes the
  purported independent item visibly carry the state being tested.
- `authority_deference` repeats the probe's specialist-recommendation versus
  direct-measurement choice in the downstream task.
- `exploration_exploitation` repeats the probe's known-option versus new-option
  choice in the downstream task.
- `prior_weighting` uses another prior-plus-case posterior judgment. This is a
  near-transfer task rather than a distal behavioral transfer unless the
  interpretation is explicitly changed and preregistered.
- The released vector inventory has 80 probe-wrapper violations for
  `exploration_exploitation`; these prompts are not complete instances of the
  registered wrapper.

### Engineering consistency issue

The released vector inventory is attributed to Luna, while the current v1
plans/specification path has historical Sonnet provenance. A future repaired
release must regenerate or explicitly point to the exact source snapshot; it
must not mix these identities in a manifest.

## Wave 3

Constructs: `ambiguity_orientation`, `causal_interpretation`,
`consensus_conformity`, and `plan_replanning`.

### Blockers

- `consensus_conformity` downstream records include the probe-only suffix
  `Continue processing the scenario.`. The suffix appears in behavior,
  steering, and calibration records and violates the downstream episode
  boundary.
- The existing downstream scorer did not expose named outcome aliases and
  orientation adapters for the Wave 3 task IDs. Those adapters are now added
  to the active measurement core, but the old inventory remains invalid until
  its prompts are repaired.
- `ambiguity_orientation` and `causal_interpretation` are close to the probe
  task surface. They should be treated as near-transfer diagnostics unless a
  distal downstream task is designed and frozen.
- The released vector inventory contains 360 probe-wrapper violations,
  concentrated in the existing generated probe rows.

## Wave 4

Constructs: `temporal_orientation`, `epistemic_uncertainty`,
`reciprocity_obligation`, and `goal_shielding`.

### Blockers

- The scorer previously lacked named outcome aliases and orientation adapters
  for all four Wave 4 task IDs. These adapters are now implemented in the
  active measurement core; the existing inventory is not retroactively
  released.
- `temporal_orientation` steering items explicitly say to prioritize a
  near-term or longer-term consequence. That makes the target state visible in
  the downstream prompt.
- `goal_shielding` steering items explicitly state which task receives
  attentional priority. This similarly confounds a zero-dose comparison.
- Downstream records carry the probe-only suffix in the temporal and
  reciprocity families. They must be regenerated with a strict downstream
  episode boundary.
- Reciprocity review found five probe pairs requiring entity/semantic
  adjudication before release; these are not safe to resolve by an automatic
  lexical rewrite.
- Some reciprocity and goal-shielding records contain extra response formats
  or an instruction after the registered response contract. The new strict
  validator rejects these rather than guessing which format is authoritative.

## Common execution blockers

Across the waves, the historical activation inventory uses a raw-text/512-token
configuration while the steering path has used chat formatting/1,024 tokens.
Before any confirmatory run, one model manifest must freeze the prompt format,
tokenizer behavior, maximum length, truncation policy, activation site, hook
path, and probe→downstream episode boundary. Truncation must fail closed.

The current benchmark also requires the Wave 1 measurement gate and precision
simulation before a full multi-wave campaign. A mechanically complete prompt
inventory is not a completed scientific result.

## Release decision

The Waves 2–4 v1 inventories remain **engineering artifacts**. The active
repository now fails closed on the identified prompt-composition and
same-surface issues, and the scorer has the named adapters needed for future
releases. No Wave 2–4 inventory should be labeled confirmatory until a
versioned repair, fresh review/full generation, semantic adjudication where
needed, and the standard release audit all pass.

The repaired Wave 1 v2 review inventory is tracked separately as a
non-confirmatory review artifact. It passed the strict downstream text gate;
the next step is to generate and audit its complete v2 inventory before model
execution.
