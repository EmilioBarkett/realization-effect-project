# Codex next steps: construct bank and synthetic prompt generation

> **Historical implementation handoff.** The prompt-generation work described
> here has been superseded by frozen prompt inventories. The operative model-
> side handoff is [`NEXT_RUN.md`](NEXT_RUN.md). Retain this file for provenance;
> do not use it as the next-run checklist.

## Purpose and authority

This document gives the next Codex session a concrete implementation sequence
for the construct bank and synthetic-prompt phase agreed with the user.

`PROJECT_DIRECTION.md` remains the canonical scientific authority. The
16-construct bank below is synchronized into the canonical direction and
protocol documents. Retain the current claim limits: two constructs validate
engineering, four constructs support a descriptive pilot, and a larger matrix
is required for out-of-sample representation-profile prediction.

## Current repository state

At the time of this handoff:

- the current measurement worktree may contain user-owned implementation
  changes; re-check Git state before editing;
- the multi-construct control plane is implemented under
  `src/construct_benchmark/`;
- construct, run, analysis, prompt, split, and provenance validation exists;
- train-only direction construction and construct isolation are enforced;
- `make check` is the required Python 3.11 baseline; report the current test
  count rather than relying on this historical handoff;
- the complete API-generated all-16 vector/probe inventory is available under
  `results/benchmark/vector_prompts_v2_luna/full_final_all16/`; it excludes
  downstream behavior, calibration, and steering-task prompts;
- no API call or model download was made during the planning review;
- projection margins, calibration, parsing, primary state-transfer scoring,
  control-direction generation, timing-aware injection, and local/RunPod
  planning/execution entrypoints are fixture-tested;
- independent prompt-role/family validation, pre-registered category schedules,
  validation-only layer selection, bootstrap intervals, and the no-API fake
  vertical slice are now implemented and fixture-tested;
- scalar injection pre/post traces, independently labelled downstream-layer
  projections, expected-versus-observed scoring, persistence ratios, and
  manifest-backed resumable steering output are implemented and fixture-tested;
- a realization real-model decode pilot exists as an engineering/reference
  artifact, but no representative real-model activation or steering run has
  validated the generalized measurement code. Prompt-only behavior
  composition, output-accessibility, collateral checks, and real-run
  uncertainty remain incomplete.

Always re-check these claims:

```bash
git status --short --branch
rg --files
make check
```

## Implementation completed from this handoff

The repository-side implementation sequence described below has now been
completed for Wave 1:

- the canonical documents and handoffs recognize the frozen 16-construct bank
  and four-wave schedule;
- the versioned registry contains all 16 IDs, families, waves, statuses, and
  future specification paths;
- all four Wave 1 construct specifications and generation plans validate;
- the generic adapter emits canonical `PromptRecord` rows, preserves
  provenance including preassigned content domains, rejects malformed
  structured responses, and marks model-subset, count-override, and limited
  runs incomplete;
- the registry enforces the declared four-family-by-four-wave topology;
- Wave 1 plans freeze probe-to-downstream task composition and expose
  one-item-per-cell pilot counts;
- the overlap audit reports construct, split, family, role, template,
  response-format, and probe/downstream independence metadata;
- the no-API Wave 1 dry run expands to 96 requests and 240 expected rows with
  token estimates;
- `make check` remains the final local gate; the exact count must be recorded
  from the current run.

The all-16 vector/probe inventory has since been generated and frozen as an
engineering artifact. The remaining external step is a reviewed model-side
smoke run for one construct, followed by the Wave 1 measurement gates; no
generalized benchmark result should be inferred from the inventory alone.

## Selected 16-construct bank

The bank is balanced across decision, epistemic, social, and agentic behavior.
Membership should be frozen before results are inspected. A construct that
fails its preregistered measurement gate should remain in the record with an
exclusion reason rather than being silently replaced.

| Construct ID | Family | Directional state contrast | Independent task |
| --- | --- | --- | --- |
| `realization_account_closure` | Decision | open/pending vs. closed/settled | risk allocation in a new domain |
| `reference_frame` | Decision | above-reference vs. below-reference outcome | unrelated sure-versus-risky choice |
| `ambiguity_orientation` | Decision | accept underspecified probabilities vs. prefer resolved probabilities | known-versus-unknown lottery allocation |
| `temporal_orientation` | Decision | immediate-consequence vs. long-term-consequence focus | smaller-sooner versus larger-later allocation |
| `evidence_diagnosticity` | Epistemic | highly diagnostic vs. weakly diagnostic evidence | posterior update magnitude |
| `prior_weighting` | Epistemic | prior/base-rate-sensitive vs. case-evidence-sensitive | structured Bayesian probability judgment |
| `causal_interpretation` | Epistemic | causal vs. correlational representation | intervention-versus-observation prediction |
| `epistemic_uncertainty` | Epistemic | resolved/certain vs. unresolved/uncertain | seek more information versus commit now |
| `source_reliability` | Social | reliable-source vs. unreliable-source weighting | weight testimony in a new factual domain |
| `authority_deference` | Social | defer to legitimate authority vs. independently verify | follow advice versus conflicting direct evidence |
| `consensus_conformity` | Social | follow group consensus vs. independent judgment | factual choice with controlled peer responses |
| `reciprocity_obligation` | Social | reciprocal obligation vs. no obligation | return/help allocation in a new interaction |
| `persistence_continuation` | Agentic | continue vs. abandon/reallocate | resource allocation after a setback |
| `exploration_exploitation` | Agentic | explore alternatives vs. exploit a known option | structured bandit or search choice |
| `plan_replanning` | Agentic | preserve the current plan vs. adaptively revise means | maintain or revise after changed constraints |
| `goal_shielding` | Agentic | shield the focal goal vs. attend to competing goals | continue focal task versus switch to distractor |

Do not collapse these distinctions:

- evidence diagnosticity concerns the information; source reliability concerns
  its origin;
- source reliability concerns track record; authority deference concerns
  status or legitimacy;
- persistence concerns retaining the goal; replanning concerns changing the
  means;
- exploration concerns sampling alternatives; goal shielding concerns
  resisting distraction;
- epistemic uncertainty concerns whether a belief is resolved; ambiguity
  orientation concerns acting with unknown probabilities;
- reference framing concerns position relative to a reference point; account
  closure concerns whether an outcome is finalized.

## Frozen execution waves

Run four constructs per wave, one from each family. Do not reorder waves after
observing readout or steering results.

### Wave 1: anchor wave

- `realization_account_closure`
- `evidence_diagnosticity`
- `source_reliability`
- `persistence_continuation`

### Wave 2: weighting and control

- `reference_frame`
- `prior_weighting`
- `authority_deference`
- `exploration_exploitation`

### Wave 3: uncertainty and adaptation

- `ambiguity_orientation`
- `causal_interpretation`
- `consensus_conformity`
- `plan_replanning`

### Wave 4: horizon and goal management

- `temporal_orientation`
- `epistemic_uncertainty`
- `reciprocity_obligation`
- `goal_shielding`

## Legacy generator boundary

The retained legacy generator in
`src/activation_analysis/openrouter_prompt_generation.py` is still
realization-specific. It hardcodes the `paper_open` and `realized_closed`
response fields and describes realization, emotion, risk, casinos, and
gambling in its instructions. It does not generate evidence-diagnosticity
prompts and does not directly emit the complete canonical prompt record.

The current generator is missing these canonical output fields:

- `construct_id`;
- `condition_id`;
- `prompt_role`;
- `task_id`;
- `parser_id`;
- `metadata_json`.

Do not launch new multi-construct generation through that legacy interface.
Use `src/construct_benchmark/generation.py` and
`scripts/generate_construct_prompts.py` for benchmark-facing generation. The
current plans also disable automatic retries and require category assignments
to match their frozen schedules.
Existing realization prompts may be adapted and reused if they pass the new
schema and leakage audits; do not automatically regenerate them.

## Historical implementation plan

The sequence below records the earlier implementation plan and is retained as
handoff history. The current scientific gates are in the root protocol and
architecture documents; the fake local vertical slice and one-construct
RunPod configuration now precede any external run.

### 1. Synchronize scientific scope

Update `PROJECT_DIRECTION.md`, `SCIENTIFIC_PROTOCOL.md`,
`PROJECT_ARCHITECTURE.md`, `readme.md`, `AGENTS.md`, and the handoff documents
to distinguish:

- the frozen 16-construct candidate bank;
- Wave 1 as the immediate implementation target;
- two constructs as an engineering result;
- four constructs as a descriptive pilot;
- the larger bank as the later predictive matrix.

Splitting `source_reliability` from `authority_deference` is intentional and
must be reflected consistently.

### 2. Add a construct registry

Add one versioned registry recording all 16 IDs, families, waves, statuses,
and specification paths. Validate unique IDs, exactly one family and wave per
construct, and agreement between registry entries and loaded construct specs.

Do not create all 14 missing specifications mechanically. Define the registry,
then write and review the two missing Wave 1 specifications:

- `source_reliability_v1.json`;
- `persistence_continuation_v1.json`.

Review the existing realization and evidence-diagnosticity specs against the
same requirements rather than assuming they are final.

### 3. Implement a generic generation adapter

Prefer a benchmark-facing module such as
`src/construct_benchmark/generation.py` and a CLI such as
`scripts/generate_construct_prompts.py`. The active adapter uses the OpenAI
Responses transport with `OPENAI_API_KEY`; legacy transport utilities may be
retained only for explicit historical reproduction. Construct semantics must
come from the selected specification and generation plan.

Requirements:

- build paired response fields dynamically from the condition IDs;
- emit `PromptRecord` objects directly;
- generate globally unique prompt IDs and construct-scoped pair IDs;
- retain source model, seed, temperature, batch, and plan hashes;
- preassign a content domain per generated item, require the model to return
  that domain, and retain it in canonical metadata;
- keep probe, behavior, steering, and calibration templates distinct;
- never infer a direction sign from model output;
- support injected/mock request functions for deterministic tests;
- support a no-API dry run that reports expected rows and requests;
- refuse to present a partial inventory as complete.

### 4. Add Wave 1 generation plans

Each Wave 1 construct needs a reviewable generation plan defining:

- condition IDs and pair roles from the construct specification;
- prompt families, domains, and counts;
- source models, temperatures, seeds, and retry limits;
- train, validation, and held-out family separation;
- separate content pools for probe and downstream tasks;
- neutral or within-cell calibration items;
- `behavior_eval` and `steering_eval` items;
- response format and parser ID;
- forbidden lexical shortcuts.

The task-composition contract is frozen for Wave 1: paired probe context comes
first, downstream prompts come only from the independent behavior-task
template, only the induced state may carry over, no probe surface text carries
over, and behavior and steering content pools remain separate. The actual
orchestration that carries an activation state between these prompts is still
unimplemented and must not be inferred from generated text.

Required canonical splits are:

```text
direction_train
direction_validation
direction_heldout
behavior_eval
steering_eval
calibration
```

### 5. Add deterministic tests

Tests must demonstrate that:

- all generated rows validate through `validate_prompt_records()`;
- output survives CSV and JSONL round trips without metadata type changes;
- condition IDs and signs come from the selected construct spec;
- prompt roles match splits;
- pair members remain within one construct and split;
- global IDs do not collide across parallel jobs;
- missing Wave 1 splits fail validation;
- source provenance is retained;
- plan expansion is deterministic for a fixed seed;
- mocked responses with wrong fields, missing pairs, duplicate IDs, or invalid
  text are rejected;
- a legacy adapter cannot bypass the canonical validator.

### 6. Generalize leakage auditing

The audit must report overlap by construct, split, prompt family, and prompt
role, including exact duplicates, normalized lexical overlap, and template or
response-format overlap. The critical audit is not only train versus held-out:
probe content must also be independent from behavior and steering content.

### 7. Run a no-API dry run

Before external generation, expand the Wave 1 job matrix locally and save only
a small reviewable summary under `results/test/`. Review request count, prompt
count, condition balance, split coverage, source-model balance, deterministic
content-domain assignment, estimated input/output tokens, and—when explicit
prices are supplied—estimated dollar cost. A model-subset or per-cell-count
dry run must be marked incomplete.

### 8. Run a tiny API pilot only after review

After the dry run and tests pass, request explicit user confirmation before
the first external generation. Use
`--count-per-model-per-cell 1 --allow-partial` to generate one pair or item
per Wave 1 cell for each selected source model, then:

1. validate each construct inventory independently;
2. inspect every pilot prompt manually;
3. run construct/split/role leakage audits;
4. merge the four inventories;
5. validate the combined inventory with `scripts/validate_construct_run.py`;
6. freeze its hash and provenance;
7. review quality before approving larger generation.

Do not proceed directly from a successful API response to a full dataset.

## Parallel execution rules

Prompt generation may run in parallel by construct, with one isolated output
and seed namespace per job:

```text
four construct generation jobs
             ↓
four independent validations and audits
             ↓
one canonical combined inventory
             ↓
one shared activation pass per model
             ↓
four construct-scoped directions and analyses
```

Never pool directions across `construct_id` values. Parallel jobs must not
append concurrently to the same CSV. Merge only validated, completed
inventories through the canonical prompt utilities.

## Environment and artifact safety

No credential is required for schema work, dry runs, tests, prompt validation,
or local fixture generation.

Active external generation requires `OPENAI_API_KEY`. Check only whether it is
present; never print or persist its value. The legacy activation-analysis path
may still use `OPENROUTER_API_KEY` when explicitly selected.

Do not add `.env` files, raw generations, model outputs, or credentials to Git.
Raw outputs under `results/benchmark/<run_id>/raw/` are ignored. Keep smoke
artifacts under `results/test/` and track only small reviewed configs,
manifests, audits, and curated summaries. The prepared run root is shared
across constructs; construct-specific artifacts belong below
`constructs/<construct_id>/`.

## Definition of ready for synthetic generation

The project is ready for a one-item-per-cell external pilot only when:

- the canonical documents recognize the bank and Wave 1 scope;
- all four Wave 1 construct specs validate;
- all four Wave 1 generation plans are reviewed;
- the generic adapter emits canonical `PromptRecord` rows;
- the no-API dry run reports the expected balanced matrix;
- deterministic generator and validator tests pass;
- overlap auditing covers probe/downstream independence;
- `make check` passes;
- output paths and ignore rules are confirmed;
- the user explicitly approves the external API pilot.

Full Wave 1 generation requires manual acceptance of the pilot prompts and a
combined pilot inventory that passes validation and leakage audits.

## Handoff expectations

At the end of the next session, report:

- changed files;
- protocol versus implemented code changes;
- exact checks and test counts;
- whether any API call occurred and how many requests were made;
- generated output paths and whether they are ignored or tracked;
- credentials, model, or tooling assumptions;
- unresolved construct-validity or prompt-quality questions;
- whether readiness applies to a dry run, a tiny pilot, or full generation.
