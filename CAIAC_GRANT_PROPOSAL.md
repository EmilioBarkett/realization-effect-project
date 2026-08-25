# CAIAC grant application draft

## Suggested funding request

`$100`

## Tell us about your proposal

### Decodability is not control: a benchmark for steering behavioral states in language models

Language models can contain information about a behavioral state without that
state being causally controllable. A model may respond differently when a
scenario is framed in two ways, making the difference easy to decode from its
activations, while adding the corresponding activation direction fails to
change an independent decision. This detection–control gap matters for any
interpretability method that treats a readable internal representation as a
reliable handle for intervention.

I will build and run a small, open-source benchmark testing this gap across
four deliberately different construct families:

- decision: account closure or realization;
- epistemic: evidence diagnosticity;
- social: source reliability and independent verification;
- agentic: persistence versus abandonment after a setback.

For each construct, I will generate separate paired probe prompts and
independent downstream behavior prompts. The probe prompts will be split into
training, validation, and held-out sets. A linear direction will be estimated
from training activations only, evaluated using continuous held-out projection
margins, and then applied to a separate behavioral task. Steering will begin
with neutral-variance calibration, prefill-only injection, five registered
doses (`-1`, `-0.5`, `0`, `+0.5`, `+1`), shuffled-label controls, random-direction
controls, and bootstrap confidence intervals.

The first phase is intentionally modest: validate the complete pipeline on one
open-weight model and one construct, then expand to the four-construct Wave 1
pilot only if the prompts and controls pass review. The requested $100 will be
used for temporary GPU time and small-scale storage. Prompt generation will be
done in a reviewed pilot before any larger run, so the project does not spend
compute scaling a flawed prompt design.

The deliverable will be a reproducible codebase, frozen prompt and analysis
schemas, and a transparent report of positive and null results. If
decodability reliably predicts independent-task steering, the benchmark will
identify conditions under which a readout is also a useful control signal. If
decodability is common but steerability is weak or inconsistent, that
dissociation is the central result: readable representations should not be
treated as causal control certificates.

## How is this related to AI safety?

AI safety increasingly depends on both monitoring internal model states and
intervening on them. These are different capabilities. A monitor may detect a
state without providing a reliable way to change it, while an apparent
steering effect may actually be a generic response bias, prompt artifact, or
off-manifold disturbance. Confusing detection with control could lead to
overconfidence in methods intended to modify model behavior.

This project provides a controlled way to measure that distinction. It freezes
the readout before intervention, uses independent downstream tasks, includes
negative, shuffled, random, and collateral-behavior controls, and reports null
results rather than selecting only successful interventions. The initial
constructs are low-risk behavioral states; the methodological result is meant
to generalize to future work on safety-relevant states such as refusal,
deception-related behavior, or specification-following, without requiring the
pilot itself to study harmful capabilities.

The project is therefore a small infrastructure and measurement contribution:
it tests when activation-based interpretability supports causal behavioral
control, and when it only supports observation.

## Suggested supporting documents

- [`PROJECT_DIRECTION.md`](PROJECT_DIRECTION.md)
- [`SCIENTIFIC_PROTOCOL.md`](SCIENTIFIC_PROTOCOL.md)
- [`BENCHMARK_REVIEW_HANDOFF.md`](BENCHMARK_REVIEW_HANDOFF.md)
- [`readme.md`](readme.md)
