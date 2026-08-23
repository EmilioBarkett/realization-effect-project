# Representation–Steerability Correspondence

This repository is developing a cross-construct benchmark for testing which
properties of internal representations predict causal steerability in language
models.

## Current status

The project is currently in **scientific protocol development and benchmark
infrastructure implementation**. The shared multi-construct control plane is
implemented, but no new benchmark dataset has been collected and the
end-to-end measurement layer is still under construction.

The repository cleanup is mostly complete: the original realization-effect
behavioral pipeline is archived, while the activation-analysis prompt
generator and activation primitives remain active.

## The research question

For a clearly defined behavioral construct, we will ask:

1. Does a controlled prompt contrast produce a behavioral difference?
2. Is the corresponding internal state linearly decodable on held-out prompts?
3. Does steering the frozen direction change an independent downstream
   behavior in the predicted direction?

Our central hypothesis is:

> Linear decodability is common across behavioral constructs, but decodability
> alone is an incomplete and often poor predictor of causal steerability.

Behavioral sensitivity, representation quality, readout, and steering are
separate estimands. A probe that separates two prompt conditions is not
automatically a causal control signal. The detailed proposal treats this as a
representation–steerability correspondence problem rather than a steering
leaderboard.

## Initial construct set

The first benchmark is deliberately broader than behavioral economics:

| Construct | Family | State contrast | Independent task |
|---|---|---|---|
| Realization/account closure | Decision | Paper/open versus realized/closed account | Risk or wager choice |
| Evidence diagnosticity | Epistemic | Reliable/diagnostic versus weak/unreliable evidence | Confidence or belief revision |
| Source reliability/authority | Social | Deference versus independent verification | Follow source versus check evidence |
| Persistence/continuation | Agentic | Continue versus abandon/reallocate effort | Continue, quit, revise, or reallocate |

The first engineering vertical slice is planned around realization and evidence
diagnosticity. The remaining constructs require the same measurement review
before inclusion in a confirmatory benchmark.

Specification gaming in coding agents is not the current project direction;
that proposal is preserved as historical material outside the active scope.

## Common experimental design

Each construct will use the same protocol:

```text
theory-relevant paired prompts
        ↓
frozen train/validation/held-out splits
        ↓
train-only linear direction
        ↓
continuous held-out projection readout
        ↓
independent downstream behavioral task
        ↓
calibrated additive steering
        ↓
state-transfer, compliance, and collateral-behavior analysis
```

The primary readout is continuous standardized projection margin on held-out
prompts. The primary causal outcome is directed mean behavioral state transfer
under positive, zero, and negative calibrated doses. Pair accuracy and
policy-slope changes are secondary outcomes. Policy-gain steering is future
exploratory method development, not a current claim.

Probe prompts and downstream tasks must be meaningfully independent. Splits,
prompt overlap, parsing rules, model settings, layer settings, intervention
timing, and exclusions must be recorded before confirmatory runs.

## What is currently implemented

Active and reusable:

- [`src/activation_analysis/`](src/activation_analysis/) — prompt generation,
  residual logging, activation storage, vector primitives, and steering;
- [`configs/activation_analysis/`](configs/activation_analysis/) — current
  activation and paired-prompt generation plans;
- [`experiments/activation_analysis/`](experiments/activation_analysis/) —
  reviewable prompt CSVs;
- active scripts for prompt generation, residual logging, vector construction,
  evaluation, validation, and overlap auditing.

Implemented control plane:

- `src/construct_benchmark/` — construct, run, analysis, prompt, split, and
  provenance schemas;
- canonical combined prompt inventories with global IDs and construct-scoped
  pair validation;
- shared-activation/construct-fan-out run manifests;
- two validated construct definitions and a two-construct smoke config;

Not implemented yet:

- continuous projection-margin analysis;
- neutral/within-cell dose calibration;
- generic state-transfer and manipulation-check adapters;
- the evidence-diagnosticity experiment.

The active vector path now uses the tracked iterator in
`activation_analysis.activation_store`; the obsolete SAE-only tests are
archived. Under Python 3.11, the clean editable install and active suite now
pass `make check`; the optional PyTorch-dependent interpreter tests are
skipped when that extra is not installed.

## Repository layout

```text
src/activation_analysis/           Active activation-analysis primitives
scripts/                          Active activation entrypoints
configs/activation_analysis/      Active prompt-generation configs
experiments/activation_analysis/  Reviewable prompt datasets
tests/                            Active tests
archive/realization_effect/       Original behavioral pipeline and adapters
archive/sae/                      Archived optional SAE-training tests
archive/documentation/             Superseded planning documents
reports/                           Paper and historical reference artifacts
results/                            Curated summaries and local/ignored outputs
configs/construct_benchmark/        Multi-construct specs and run configs
src/construct_benchmark/            Shared schemas, prompt validation, run plans
```

The original realization-effect paper and its implementation remain useful as
the anchor case study. The code is preserved in
[`archive/realization_effect/`](archive/realization_effect/), and the earlier
planning documents are indexed in
[`archive/documentation/`](archive/documentation/).

## Documentation map

- [`PROJECT_DIRECTION.md`](PROJECT_DIRECTION.md) — canonical scientific scope,
  claims, constructs, status, and next gates;
- [`BENCHMARK_RESEARCH_DIRECTION.md`](BENCHMARK_RESEARCH_DIRECTION.md) — the
  detailed representation–steerability correspondence benchmark proposal;
- [`BENCHMARK_REVIEW_HANDOFF.md`](BENCHMARK_REVIEW_HANDOFF.md) — focused brief
  to send to another chat or reviewer;
- [`SCIENTIFIC_PROTOCOL.md`](SCIENTIFIC_PROTOCOL.md) — current experimental
  protocol;
- [`PROJECT_ARCHITECTURE.md`](PROJECT_ARCHITECTURE.md) — current engineering
  architecture and implementation roadmap;
- [`AGENTS.md`](AGENTS.md) — instructions and invariants for coding agents;
- [`CHAT_HANDOFF.md`](CHAT_HANDOFF.md) — concise continuation brief.

## Development

```bash
python -m venv venv
./venv/bin/python -m pip install -e ".[dev]"
make check
```

Do not make API calls, download model weights, or launch a large experiment as
part of ordinary tests. Raw generations, model weights, and large activation
tensors must remain outside Git. The benchmark raw path
`results/benchmark/<construct>/<model>/<run>/raw/` is already ignored before
benchmark runs begin.

## Historical reference

The original paper is available at
[`reports/Realization_Effect_in_Language_Models____Ciaran____Emilio.pdf`](reports/Realization_Effect_in_Language_Models____Ciaran____Emilio.pdf).
Its main lesson motivates the new benchmark: a model can show behavioral
sensitivity and contain a linearly decodable signal without that signal being
a reliable causal handle on downstream behavior.
