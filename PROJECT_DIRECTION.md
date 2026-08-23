# Project Direction: Representation and Steering of Specification Gaming in Coding Agents

## Status and relationship to the broader project

This document records the focused agentic study that follows the realization-
effect work. It complements the broader cross-construct architecture in
`PROJECT_ARCHITECTURE.md`; it does not replace the existing realization-effect
paper or the general decodability-to-steerability protocol.

The realization-effect project supplies the experimental architecture:

1. establish a behavioral contrast;
2. construct a train-only linear readout;
3. evaluate it on held-out material; and
4. test causal steering separately from readout.

The new study applies that architecture to specification gaming in coding
agents. Its central construct is not audit awareness alone. It is the contrast
between following the user's actual specification and optimizing a test suite
or evaluator when the two conflict.

## Central question

When a user's requirements conflict with the visible tests or evaluator, does
the model distinguish satisfying the real goal from exploiting the scoring
proxy, and can that distinction be read out and causally steered?

Specification-faithful behavior means trying to satisfy the stated requirement,
recognizing an inconsistency, and avoiding shortcuts that violate the intended
behavior. Proxy optimization means improving the evaluator outcome while
violating, bypassing, or ignoring that requirement.

Optimizing tests is not automatically specification gaming. The distinction is
only meaningful on tasks where the tests or evaluator are an imperfect or
conflicting proxy for the user's goal.

## Related work and novelty positioning

The novelty claim should be narrow because nearby work already covers several
important components:

- The [SPAR working report *Inducing Reward Hacking via Activation
  Steering*](https://library.sparai.org/reports/inducing-reward-hacking-via-activation-steering-rcy2f4/), submitted in January 2026, shows that activation steering can induce
  benign reward-hacking behavior in open-weight models and reports effects that
  vary with steering strength and layer.
- [*From Reward-Hack Activations to Agentic Risk States*](https://arxiv.org/abs/2606.06223), a June 2026 paper, studies activation-based monitoring and steering in
  agentic Gameable ALFWorld and WebShop settings. It finds that reward-hack
  activation can identify a latent policy state, but that its action-level
  meaning is context-dependent; entropy and decision context improve prediction,
  and steering reduces proxy-exploit behavior only in selected regimes.
- [OpenAI's coding-task monitoring work](https://openai.com/index/chain-of-thought-monitoring/)
  shows that frontier reasoning models can exploit evaluator loopholes,
  including subverting tests, and that chain-of-thought/action monitoring can
  detect some of these behaviors. It also shows that pressure on the monitored
  reasoning can make intent less observable without eliminating misbehavior.

The paper should therefore not claim to be the first to find a reward-hacking
activation direction, the first to steer reward hacking, or the first to study
reward/specification gaming in agents or coding agents. The potentially novel
claim is the specific identification strategy:

> We test whether a representation induced by contrasting user-goal and
> evaluator-oriented framing generalizes to neutral specification-gaming
> decisions in coding agents, and whether manipulating that representation
> selectively changes proxy-optimizing behavior.

The distinctive contribution is construct validity. The direction must survive
neutral prompts, held-out exploit-family transfer, nuisance-controlled
discriminant-validity tests, and ordinary/benign controls before it is given a
strong semantic interpretation. Causal steering then tests whether the
direction selectively changes proxy optimization rather than merely changing
generic caution, refusal, verbosity, or activity. This directly addresses the
context-dependence concern in the closest agent-monitoring work and positions
the project as a methodological workshop contribution rather than another
claim to have discovered reward-hacking representations.

This is a working novelty position, not a priority guarantee. The related-work
search and citations should be updated again before submission.

## Evidential ladder

The project must distinguish the following levels of evidence:

1. **Framing decodability:** intent-first and evaluator-oriented goal framing
   are linearly distinguishable in the model's activations.
2. **Neutral transfer:** the learned direction predicts relevant behavior when
   the framing language is removed.
3. **Discriminant validity:** the direction predicts specification-gaming
   behavior beyond task identity, exploit family, wording, generic caution,
   refusal tendency, and other registered nuisance variables.
4. **Causal specificity:** steering the direction changes specification gaming
   more than it changes ordinary coding, generic refusal, verbosity, tool use,
   or unrelated risk-taking.

The first level alone is not evidence of a specification-faithfulness concept.
The direction should initially be called a `goal_framing_direction` or
`behavior_associated_direction`. Stronger construct language is warranted only
if the later transfer and specificity tests pass.

## Experimental design

### Task suite

Build compact coding tasks containing:

- a natural-language user specification;
- a visible test suite or evaluator;
- a tempting shortcut or loophole;
- a legitimate implementation satisfying the actual requirement;
- an objective way to identify whether the shortcut was taken; and
- an ordinary-task counterpart where the specification and tests are aligned.

Candidate gaming opportunities include modifying or weakening tests, hard-coding
known cases, disabling validation, suppressing errors, manipulating state, and
exploiting evaluator-specific implementation details.

Organize tasks into exploit families. Use some families for direction
construction and hold out entire other families for evaluation. A held-out
result should test generalization to a new kind of specification conflict, not
merely a new paraphrase of "do not edit tests."

### Behavioral stage

Retain audit awareness as a secondary manipulation in a 2 × 2 design:

- neutral framing, no audit notice;
- neutral framing, audit notice;
- intent-first framing, no audit notice; and
- intent-first framing, audit notice.

The intent contrast estimates the effect of emphasizing the user's goal. The
audit contrast estimates the effect of being told that behavior will be
reviewed. Their interaction tests whether oversight makes intent-following more
effective.

The primary outcome is specification-gaming rate on conflict tasks. Secondary
outcomes include legitimate and ordinary-task success, honest abstention or
clarification, inconsistency recognition, test-file modification, hard-coded
outputs, other exploit strategies, tool-call counts, retries, malformed actions,
and output length.

Measure strategy substitution as well as total gaming. An agent may stop using
an obvious exploit while switching to a quieter one, so report both overall
gaming and the distribution or detectability of gaming strategies.

### Linear readout stage

Learn the initial direction from matched goal-framing contrasts on training
task families. Do not construct the main direction from audit language, since
an audit paragraph may be trivially detectable without representing any
relevant behavior.

Evaluate the frozen direction on neutral contexts where the explicit framing is
removed. The confirmatory readout set should include new wording and templates,
new tasks, held-out exploit families, and a separate scenario source where
practical.

Activations should be captured immediately before the relevant action, before
the model has completed the exploit.

#### Action propensity versus sampled action

At a fixed pre-decision state, an identical prefix produces the same hidden
state in a standard deterministic transformer. Sampling happens afterward.
Therefore, a pre-decision activation cannot predict which random sample will be
drawn from an otherwise fixed distribution.

The experiment must distinguish:

- **action propensity:** probability or logit margin assigned to faithful versus
  gaming actions; and
- **sampled behavior:** the action actually produced in a rollout.

For short controlled decision prompts, action propensity can be estimated using
constrained action categories or matched candidate continuations. For full
agent trajectories, estimate sampled behavior using repeated rollouts from the
same neutral context. Do not conflate rollout frequency with a hidden-state
propensity measure.

The project must not rely on same-task, same-condition divergent random seeds as
if they create distinct pre-decision representations. Identical prefixes have
identical pre-decision activations; already-diverged histories introduce a
history confound. Matched framing contrasts, neutral transfer, and repeated
rollouts are the appropriate complementary designs.

#### Nuisance and incremental prediction tests

Probe performance must be evaluated after balancing or controlling for task
identity, exploit family, wording and length, task difficulty, trajectory
length, prior tool history, generic caution or refusal tendency, and framing
language.

Use leave-one-family-out evaluation, matched task splits, mixed-effects or
hierarchical analyses, and permutation controls. The key result is incremental
prediction: whether the proposed direction explains additional variation in
specification-gaming behavior beyond these nuisance variables.

### Steering stage

Apply positive and negative versions of the frozen direction to neutral
coding-agent contexts at the same pre-decision point used for the readout.

The initial comparison should include negative steering, zero steering,
positive steering, a shuffled-direction control, and a random-vector or
wrong-layer control. Use a pre-registered dose sweep and do not choose the
reported sign, layer, position, or scale after inspecting held-out behavior.

Steering is successful only if it changes specification gaming while preserving
legitimate coding ability. Generic reduction in activity, increased refusal,
shorter outputs, or broad caution does not count as successful control.

## Controls

Required controls are:

- held-out task and exploit families;
- neutral prompts with explicit framing removed;
- shuffled-label and random-direction controls;
- negative and zero steering doses;
- wrong-layer or irrelevant-direction controls;
- ordinary coding tasks;
- refusal, verbosity, tool-use, and unrelated risk-taking measures;
- objective task outcomes and deterministic diff checks;
- blinded transcript analysis where possible;
- manual review of a random subset; and
- explicit reporting of ambiguous cases and judge disagreement.

### Optional same-model realization control

If feasible, reconstruct a small realization direction in the same open-weight
coding model used for steering. Measure alignment, cross-prediction, and
cross-steering on both task sets, along with generic caution and risk-taking
side effects.

This is a negative-control battery, not a required realization × audit × intent
factorial. A realization direction from Gemma must not be treated as a control
for Qwen, DeepSeek, or another unrelated architecture.

## Initial scope

Use one open-weight coding model with reliable activation access. Select the
checkpoint after a pilot confirms ordinary coding ability, a non-trivial
baseline gaming rate, reproducible inference, and usable hidden-state hooks.

Initial target scale:

- 500–600 paired training examples;
- 150–200 validation examples;
- 150–200 neutral held-out examples from new task families;
- approximately 40 conflict and 40 ordinary tasks for steering; and
- three steering levels and two random seeds, preceded by a smaller pilot.

This is approximately 480 agent trajectories for the initial steering study.
The short activation dataset is comparatively inexpensive; long agentic
trajectories are the main compute risk. Full SWE-bench should therefore be
optional validation rather than the core experiment.

The direction is model-specific. A direction learned in one architecture
cannot be assumed to transfer to another.

## Expected outcomes

Informative outcomes include:

- neutral held-out transfer plus steering that reduces gaming without harming
  ordinary performance;
- neutral transfer and behavioral prediction but little causal steering;
- steering that changes gaming while causing broad degradation or excessive
  refusal;
- failure to transfer after framing, task identity, and exploit-family controls;
  or
- strong alignment with a realization or generic-caution direction, suggesting
  that the proposed construct is too broad.

Readable neutral transfer without reliable steering would support the broader
realization-effect lesson that internal decodability and behavioral control are
separable.

## Minimum publishable version

A strong minimum version includes:

- matched conflict and ordinary coding tasks;
- the behavioral intent-versus-audit comparison;
- a train-only goal-framing direction;
- neutral held-out task-family evaluation;
- action-propensity and repeated-rollout measures;
- incremental prediction beyond nuisance controls;
- a small causal steering experiment;
- random-direction and wrong-layer controls; and
- ordinary-performance and over-refusal analysis.

The defensible contribution is narrow:

> Test whether a goal-framing contrast yields a behavior-associated activation
> direction that transfers beyond its prompt wording and can causally influence
> specification-gaming behavior in coding agents.

This focused study should reuse the existing activation primitives and the
train-only direction discipline. It should not begin a large experiment until
the task schema, split manifest, intervention timing, action-propensity
measure, and primary outcomes are frozen.
