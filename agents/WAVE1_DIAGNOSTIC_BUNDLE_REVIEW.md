# Wave 1 diagnostic bundle review

This is a non-confirmatory local review of the stopped Qwen C1 diagnostic and
the completed Wave 1 engineering outputs. It does not promote any engineering
result to confirmatory evidence.

## Bundle and execution state

- Base snapshot: `88e29f27ee027942c082994e866c42384149b98e`
- Local bundle: `results/runpod/wave1_diagnostic_bundle_v1/`
- Remote bundle: `/workspace/realization-effect-project/results/benchmark/wave1_diagnostic_bundle_v1/`
- Remote raw-data location record: `metadata/artifact_locations.json` in the bundle
- B300: `5y2l23j8wh77wc`, volume `pwq7nznkzn`, stopped with API desired status `EXITED`
- C1: 391/5,120 requests and 4,692/61,440 observations, `complete=false`,
  resumable output retained on the persistent volume
- C1 archive SHA-256: `c51071d027c7f923597638409fa5b4772b0c82823e481ca16946027c6cb636af`
- The archive matched between the remote volume and the laptop; its internal
  134-file checksum set passed. No raw-data file was transferred.

## Findings that remain blockers

| Model | Construct | Finding |
| --- | --- | --- |
| Qwen | evidence diagnosticity | 32/32 valid, but only two outcomes, SD 0.8839, and 0.96875 dominant share |
| Qwen | evidence diagnosticity | 62 steering records were missing or unscorable in the manipulation checks |
| Mistral | realization, persistence | 0 valid primary baseline rows; extra text violated the strict parser |
| Mistral | evidence, source | 11/32 and 24/32 baseline rows invalid due to extra text |
| Mistral | all four constructs | steering accessibility failed; overall primary-valid rates were 0, .5, .125, and .25 |
| Mistral | all four constructs | collateral accessibility/correctness was below the registered .95 validity gate |

The stopped C1 summary is diagnostic only. Its 391 completed requests and
partial output changes cannot be used as a complete causal-patching result.
No Wave 2--4 model execution was started.

## Local revisions made from the bundle

### Prompt contract

The frozen Wave 1 v3 inventory is preserved. The new
`model_behavior_accessibility_preflight_v2_diagnostic_repair.json` requires a
new, independently generated preflight inventory rather than editing v3 in
place. Evidence-diagnosticity repair must increase heterogeneous, balanced
variation in information contrast, cost, stakes, time pressure, and option
order, while preserving the intended high-information-versus-low-information
contrast. Every construct must continue to use downstream prompts independent
of its probe prompts.

### Parser contract

The strict registered parser is retained. Extra text, reasoning text, and
out-of-range values remain invalid; the system must not salvage a plausible
integer from a verbose answer. Mistral must be rerun through the explicit
no-thinking/text-only generation contract where supported, with chat format and
short output limits. Parser failures and empty outputs remain separate
accessibility diagnostics in the bundle samples.

### Preflight contract

The v2 repair gate freezes 8--16 outcome-independent items per model,
construct, and stage. It requires 100% valid behavior outputs, at least three
distinct directed outcomes, SD at least 2.0, collateral validity at least .95
and correctness at least .75, and at least .95 validity for every registered
steering/control group. Steering remains prefill-only with target doses -1, 0,
and +1 plus zero-dose shuffled/random controls. Any model/construct failure
blocks a larger execution.

The active implementation is still the v1 selector/validator code path; the v2
file is the reviewed repair contract to use when new selections are frozen.
The bundle itself is not a preflight pass and does not authorize a new GPU run.

## Next permitted action

Create and review the revised prompt inventory locally, freeze new selections,
then run only the 8--16-item real-model preflight per model and construct. A
short-lived inexpensive analysis pod may later be attached to the same volume
if complete raw-data analysis is genuinely required. A B300 must not be used
for CPU-side analysis.
