# SAE Decisions To Make

This document tracks decisions to make before implementing SAE training. The
repo currently has activation logging, activation-run validation, and an SAE
dataset boundary, but no trained SAE.

## Backend

Decision needed: use a small local PyTorch SAE, an external library, or both.

Current bias: keep the first version readable and local unless an external
library clearly improves reliability.

## Training Data

Decision needed: which activation runs form the training distribution.

Candidate mix:

- realization-effect prompts
- emotion contrast prompts
- neutral/control prompts

Avoid training only on explicit emotion prompts, because that could learn prompt
style rather than general emotional or decision-relevant features.

## Layers

Decision needed: train one SAE per layer or share one SAE across layers.

Current bias: start with one SAE per layer. Begin with one middle-late layer
such as 18 after enough validated activations exist.

## Token Regions

Decision needed: which token regions enter training and which are reserved for
analysis.

Current bias: train on broad non-padding activations but prefer
`scenario` and `decision_question` for the first analysis views. Keep
`response_instruction` available as a comparison group, not the main signal.

## Storage Precision

Decision needed: whether large activation runs should store float32 or float16.

Current state: residual runs default to float16 storage. Downstream vector
averaging, SAE training, and metric computation should cast to float32 when
numerical precision matters.

## Evaluation

Minimum useful metrics before trusting an SAE:

- reconstruction MSE
- average active features per vector
- fraction of active features
- feature usage distribution
- top activating prompt/token examples
- condition or emotion separation checks

## Validation Split

Decision needed: how to avoid circularity.

Current bias: separate activation runs for training, feature discovery, and
behavior/steering validation, even if each split is small at first.
