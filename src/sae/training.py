from __future__ import annotations


class SAETrainingNotImplementedError(NotImplementedError):
    """Raised until the project chooses a concrete SAE training backend."""


def train_sae(*_args, **_kwargs):
    """Placeholder for the future SAE training entrypoint.

    The immediate architecture work is focused on a reliable activation dataset
    boundary. Once the first vector experiments are stable, this function can
    wrap a concrete SAE implementation without changing the logging pipeline.
    """

    raise SAETrainingNotImplementedError(
        "SAE training is not implemented yet. Build datasets with src/sae/dataset.py first."
    )

