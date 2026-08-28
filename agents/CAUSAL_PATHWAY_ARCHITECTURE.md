# Causal pathway architecture

**Status:** implemented C1 infrastructure; real-model validation is still
pending.

This document specifies the first causal-pathway layer of the benchmark. It
is deliberately narrower than a complete mechanistic-interpretability claim.
The causal unit is a matched episode, not an arbitrary transplantation of an
activation from one prompt into another.

## B/R/C/S placement

The benchmark separates four kinds of evidence:

| Profile | Question | Current implementation |
|---|---|---|
| B — behavioral validity | Does the prompt condition change the registered task outcome? | Prompt-only baseline and outcome adapters |
| R — representation profile | Is the state linearly readable, stable, localized, and context-consistent? | Train-only directions, held-out margins, optional all-layer residual trace |
| C — causal pathway profile | Can a state at a defined point causally change a matched continuation? | C1 matched residual interchange runner |
| S — steerability | Does an intervention transfer the state to an independent behavior? | Additive steering runner; separate estimand |

Behavioral validity gates interpretation. Representation and causal evidence
must not be collapsed into one score, and successful patching must not be
described as proof of a unique mechanism.

## C1: matched-episode residual interchange

Each request contains:

1. a positive induction prompt;
2. a negative induction prompt;
3. one identical downstream task prompt appended to both conditions; and
4. a fixed boundary separator and boundary rule.

The runner composes:

```text
positive_induction + separator + downstream_task
negative_induction + separator + downstream_task
```

It locates the last complete token inside each induction prefix using tokenizer
offsets. The source state is captured at that token for registered residual
layers. The downstream text, entities, response format, and task identity are
therefore held fixed across the two receiver episodes.

For each requested layer, the runner records:

- a positive-to-negative swap;
- a negative-to-positive swap; and
- by default, positive-to-positive and negative-to-negative same-condition
  donor controls.

All swaps happen on the first prompt-prefill forward call at the receiver's
own boundary position. No patch is applied during answer generation. Hooks
are removed in `finally` blocks, and only scalar norms, bookkeeping, and
generated text are written to the JSONL artifact.

The implementation is in
[`src/activation_analysis/causal_patching.py`](../src/activation_analysis/causal_patching.py)
and the model-side entrypoint is
[`scripts/run_residual_interchange.py`](../scripts/run_residual_interchange.py).
The output has an adjacent manifest. The validator and summary entrypoint
refuse incomplete output by default:
[`scripts/score_residual_interchange.py`](../scripts/score_residual_interchange.py).

### What C1 identifies

C1 tests contextual causal sufficiency at a specified residual state and
location: replacing that state can alter a common continuation. It does not,
by itself, establish:

- necessity of the state;
- a unique or localized circuit;
- that the state is linearly reproducible across contexts;
- cross-domain generalization;
- a policy-gain or slope variable; or
- a complete mechanistic explanation.

The primary causal outcome is the pre-registered parsed downstream behavior.
A teacher-forced logit contrast over fixed, pre-registered valid alternatives
may be a secondary outcome. Token choices must not be selected after seeing
the patch results.

## Controls and generalization ladder

The minimum C1 control battery is:

- no-patch positive and negative baselines;
- bidirectional cross-condition swaps;
- same-condition donor swaps;
- shuffled donor assignments;
- random-example donors where the design includes more than one episode;
- norm-matched random perturbations;
- neighboring-construct donors when model capacity permits; and
- output/norm checks showing that a change is not just a malformed generation.

The initial runner implements natural-state replacement and an explicitly
labelled donor-minus-recipient variant. The broader control battery is a
manifest-level requirement for a confirmatory causal campaign; controls that
need multiple episodes or neighboring constructs belong in the campaign
composer rather than being silently invented inside a single pair.

Generalization should be staged:

1. same-pair interchange;
2. cross-pair donor/receiver interchange;
3. cross-domain held-out interchange.

Any level beyond same-pair interchange must freeze donor pools, matching
rules, and exclusions before the relevant data are inspected.

## Later causal methods

The recommended order is:

1. **C1 residual interchange:** matched episode and boundary layer/position;
2. **C2 temporal tracing:** patch successive token positions and compare the
   causal window with all-layer decodability;
3. **C3 component decomposition:** split residual changes into attention
   outputs, MLP outputs, heads, or selected paths after candidate layers are
   identified;
4. **C4 targeted ablation:** projection removal or component ablation with
   matched means and random/neighbor controls; and
5. later SAE, gradient, or full-circuit analyses if C1–C4 justify the cost.

Path patching before candidate layers are localized is premature. Ablation is
useful for necessity but can move the model off-manifold, so it should not be
treated as a replacement for natural-state interchange.

## Run order

The model-side sequence is:

```text
B behavioral validity
  -> R all-layer residual representation profile
  -> C1 matched residual interchange
  -> C2 temporal/layer follow-up
  -> C3 focused component/path tracing
  -> C4 ablation
  -> S additive steering
```

The first implementation milestone is C1 on a small Wave 1 test inventory,
after the prompt-only baseline and representation measurement gates pass. It
must be treated as an engineering/causal-diagnosis artifact until the same
model, layer, prompt, and output controls have been validated on a complete
registered run.

## Example request record

```json
{
  "request_id": "realization__episode_0001",
  "construct_id": "realization_account_closure",
  "positive_induction_prompt": "The account is closed and the outcome is final.",
  "negative_induction_prompt": "The account remains open and the outcome can change.",
  "downstream_prompt": "Choose one action and return only its name.",
  "boundary_separator": "\n\n",
  "prompt_format": "completion",
  "boundary_mode": "last_induction_token",
  "intervention_timing": "prefill_only"
}
```

The input inventory is intentionally separate from vector/probe and ordinary
steering prompt inventories. It must be audited for induction balance,
downstream independence, identical task text, and absence of post hoc
boundary changes before a model run.
