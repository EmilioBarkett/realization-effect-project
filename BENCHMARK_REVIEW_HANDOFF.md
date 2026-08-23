# Review handoff: proposed representation–steerability correspondence benchmark

Please review the proposal in
[`BENCHMARK_RESEARCH_DIRECTION.md`](BENCHMARK_RESEARCH_DIRECTION.md) as an
independent scientific reviewer.

## Proposal in brief

The broad “representation predicts steerability” framing is not unique enough
by itself. We want to sharpen it into a benchmark of whether a measurable
representation profile predicts a causal-control profile across theory-backed
behavioral constructs, contexts, layers, and models. Held-out linear
decodability is the primary representation feature, not the entire profile.

For each construct, the pipeline is:

```text
paired theory-relevant prompts
  → frozen train/validation/held-out splits
  → train-only linear direction
  → held-out continuous decodability
  → calibrated additive intervention
  → independent downstream behavioral task
  → uncertainty-aware cross-construct analysis
```

The main hypothesis is that single-direction decodability is an incomplete
control certificate: some representation properties may predict transfer, but
clean decoding alone will often fail to predict independent-task state
transfer. The important result is the calibrated correspondence and its
dissociations, not a collection of successful steering examples.

## Proposed measurements

- `B`: prompt-only behavioral sensitivity, used as a construct-validity check;
- `D`: held-out standardized projection margin from a direction built on the
  training split only;
- `S`: signed mean state transfer on an independent task under positive, zero,
  and negative calibrated doses;
- representation profile: direction stability, cross-context consistency,
  layer localization, and—later—intrinsic dimensionality;
- steerability profile: dose monotonicity, normalized intervention cost,
  specificity, and cross-task generalization.

The aggregate analysis would estimate a hierarchical relationship such as
`S = alpha + beta * D + model effects + construct effects + task effects`.
For a later broad concept bank, it could test whether the full representation
profile predicts the full steerability profile out of sample. The report should
show high/low representation × high/low steerability quadrants and may include
a descriptive standardized Representation–Steerability Gap, but should not
collapse the benchmark into one score.

## Candidate constructs

1. Realization/account closure — decision/economic anchor.
2. Evidence diagnosticity — epistemic; explicitly evidence reliability or
   diagnosticity, not automatically updating responsiveness.
3. Source reliability/authority — social deference versus independent
   verification.
4. Persistence/continuation — agentic continue versus abandon/reallocate.

The first vertical slice is realization plus evidence diagnosticity. A final
confirmatory benchmark should ideally include two independent downstream task
templates per construct; one task per construct is acceptable only for the
engineering pilot because it confounds construct and task effects.

## What I want the reviewer to challenge

1. Is this genuinely novel enough for a serious ML/interpretability venue,
   given recent work on pretraining-time steerability emergence,
   detection–intervention gaps, and measurement confounds?
2. Is the proposed contribution the predictive validity of a representation
   profile for causal control, rather than merely another benchmark of steering
   outcomes or another `D → S` correlation?
3. Which representation-profile features are theoretically meaningful and
   feasible enough for the first release?
4. Are the independent tasks sufficiently independent to support causal
   transfer claims?
5. Is the evidence-diagnosticity construct specified at the right level, or
   are we accidentally claiming to measure update responsiveness?
6. Is two tasks per construct necessary for the confirmatory version?
7. Should a 50–100-concept breadth module be part of the first paper, or be
   staged after the four-construct protocol is validated?
8. Could the archived specification-gaming work become a later hard agentic
   module, and what evidence would justify adding it?
9. Are the primary estimands, controls, and uncertainty model well defined?
10. What result would falsify the central hypothesis, and how should a strong
   positive `D → S` relationship change the paper?
11. Which adjacent papers or benchmarks make the novelty claim too strong?
12. What is the smallest adequately powered vertical slice before expanding to
   four constructs and multiple model families?
13. Which implementation assumptions in the current repository are premature,
    especially given the recently repaired activation boundary, completed clean
    test verification, and absent generic benchmark package?

## Recommended implementation sequence

The proposed order is:

1. stabilize the repository/archive checkpoint;
2. harden the active activation boundary against a representative real-run
   manifest, building on the new `ActivationVectorRecord` and
   `iter_activation_vectors()` implementation rather than restoring an
   obsolete SAE package;
3. implement only the minimum schemas, manifests, splits, readout, and
   calibration core;
4. complete the two-construct readout vertical slice;
5. add explicit steering timing, neutral/within-cell dose calibration, and
   outcome-specific effect adapters;
6. add downstream persistence, output accessibility, and behavioral task
   adapters; and
7. fit correspondence/profile models only after the construct/task/model
   matrix is large enough for out-of-sample evaluation.

The first runnable milestone should be realization plus evidence diagnosticity,
one open-weight model, at least two prompt families per construct, one
independent task per construct, deterministic activation fixtures, complete
manifests, and positive/negative/zero/shuffled/random controls. That milestone
validates the machinery; it is not yet the final predictive benchmark.

## Review standard

Please return:

- a verdict: pursue, revise, or abandon;
- the strongest novelty claim that is defensible;
- the most serious construct-validity or statistical threat;
- a recommended minimum experiment;
- any changes needed to the current protocol or architecture before coding.

The repository is still in protocol development. No generalized benchmark
has been implemented, and the existing realization behavioral pipeline is
archived as a reference case rather than the active experiment.
