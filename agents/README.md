# Maintainer and agent notes

This directory contains operational handoffs for maintainers and coding
agents. These notes do not define the scientific direction; the authoritative
documents remain at the repository root:

1. [`PROJECT_DIRECTION.md`](../PROJECT_DIRECTION.md) — scientific scope and
   claim limits;
2. [`BENCHMARK_RESEARCH_DIRECTION.md`](../BENCHMARK_RESEARCH_DIRECTION.md) —
   research proposal and novelty boundary;
3. [`SCIENTIFIC_PROTOCOL.md`](../SCIENTIFIC_PROTOCOL.md) — experimental
   contract;
4. [`PROJECT_ARCHITECTURE.md`](../PROJECT_ARCHITECTURE.md) — implementation
   boundaries;
5. [`readme.md`](../readme.md) — human-facing overview.

The files here are operational aids and do not establish that an end-to-end
benchmark run has been completed.

Current execution documents:

- [`NEXT_RUN.md`](NEXT_RUN.md) — single operative handoff for the upcoming
  RunPod B300 Wave 1 campaign, including the `RUNPOD_2_API_KEY` boundary;
- [`B300_INFRASTRUCTURE_HANDOFF.md`](B300_INFRASTRUCTURE_HANDOFF.md) — focused
  implementation handoff for the one-B300, four-wave controller and thin
  real-stage adapter;
- [`RUNPOD_EXECUTION.md`](RUNPOD_EXECUTION.md) — reusable local/GPU commands,
  execution boundaries, and artifact-safety rules;
- [`PRE_RUN_GATES.md`](PRE_RUN_GATES.md) — local checks required before GPU
  execution.

Scientific and instrumentation references:

- [`STEERING_MANIPULATION_CHECKS.md`](STEERING_MANIPULATION_CHECKS.md) —
  injection-trace, calibrated downstream-persistence, and completed-output
  manifest contract.
- [`PROJECT_DEEPENING_NOTES.md`](PROJECT_DEEPENING_NOTES.md) — future B → R →
  C → S research framework spanning behavioral validity, representation,
  causal pathways, and steerability.

Historical handoffs retained for provenance:

- [`CHAT_HANDOFF.md`](CHAT_HANDOFF.md) — superseded implementation snapshot;
- [`CODEX_NEXT_STEPS.md`](CODEX_NEXT_STEPS.md) — construct-bank and prompt-
  generation implementation history;
- [`VECTOR_PROMPT_GENERATION_HANDOFF.md`](VECTOR_PROMPT_GENERATION_HANDOFF.md)
  — completed vector-generation provenance.
