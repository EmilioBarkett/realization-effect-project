# Project-deepening notes: from behavior to causal control

**Status:** future research note. This is not part of the frozen v1 protocol
and does not imply that the listed causal-pathway analyses are implemented or
empirically validated. The root authority documents continue to define the
active project direction.

## Organizing framework

The benchmark can eventually be deepened from a comparison of decodability
and steering into a staged account of how a behavior becomes causally
controllable:

```text
Behavioral validity (B)
        ↓
Representation profile (R)
geometry · stability · localization · context
        ↓
Causal pathway profile (C)
patching · tracing · ablation
        ↓
Steerability profile (S)
intervention · transfer · specificity
```

The four stages answer different questions and should remain separate rather
than being collapsed into one score.

## B — Behavioral validity

Establish that each construct produces a reliable, interpretable behavioral
contrast before making claims about internal representations. Relevant checks
include prompt-only sensitivity, parser compliance, outcome variation,
replication across task variants, nuisance-factor balance, and discriminant
validity against neighboring constructs.

## R — Representation profile

Characterize more than linear decodability. Candidate properties include:

- geometry: margin, dimensionality, linearity, and overlap with other
  construct directions;
- stability: agreement across prompt samples, seeds, models, and direction
  estimators;
- localization: where and when the state becomes readable across layers and
  token positions;
- context: whether the representation survives new domains, phrasings, and
  independent downstream tasks.

These features should be frozen before confirmatory steering results are used
to predict steerability. The small Wave 1 pilot should not be used to fit a
high-dimensional predictor.

## C — Causal pathway profile

Test how the represented state participates in computation, rather than
assuming that a decodable direction lies on the causal route to behavior.
Possible later methods include:

- activation patching between matched conditions;
- causal tracing across layers, positions, and task stages;
- targeted ablation or projection removal;
- mediation-style tests of whether downstream-state changes account for the
  behavioral effect;
- necessity and sufficiency comparisons using matched random and neighboring-
  construct controls.

This layer should distinguish a readable correlate from a representation that
is actually used by the model.

## S — Steerability profile

Measure causal control as a profile rather than a binary success:

- intervention: dose response, sign consistency, calibration, and intervention
  cost;
- transfer: persistence across later layers, contexts, and independent tasks;
- specificity: target effect relative to shuffled/random directions,
  neighboring constructs, compliance, verbosity, refusal, and collateral
  behavior.

The primary v1 estimands remain held-out projection margin and directed mean
state transfer. The broader profile is a later extension.

## Possible research questions

1. Which representation properties predict successful intervention after
   behavioral validity is established?
2. Does causal-pathway involvement explain steerability beyond decodability?
3. Are stable and context-general directions more transferable but less
   specific?
4. Do highly localized representations steer strongly at one layer but decay
   quickly downstream?
5. Can a construct be behaviorally valid and decodable yet fail both causal
   pathway and steering tests?

## Suggested staged expansion

1. Complete the existing Wave 1 behavioral, readout, calibration, and steering
   gates.
2. Add a small preregistered representation profile using a few low-dimensional
   features.
3. Pilot patching, tracing, and ablation on one construct with positive,
   negative, random, and neighboring-construct controls.
4. Add intervention transfer and specificity measures using the already
   registered downstream tasks.
5. Expand the B → R → C → S analysis only after the precision simulation and
   model-side infrastructure support it.
