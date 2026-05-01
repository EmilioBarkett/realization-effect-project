from __future__ import annotations

import pandas as pd
import pytest

from realization_effect.analysis import _build_design_matrix


def test_build_design_matrix_requires_reference_condition() -> None:
    df = pd.DataFrame(
        {
            "condition": ["paper_gain_small", "paper_gain_large"],
            "model": ["model-a", "model-a"],
            "temperature": ["1", "1"],
            "prompt_version": ["absolute", "absolute"],
            "log_wager": [5.0, 5.2],
        }
    )

    with pytest.raises(ValueError, match="Reference condition 'paper_even'"):
        _build_design_matrix(
            df,
            conditions=["paper_gain_small", "paper_gain_large"],
            reference="paper_even",
        )


def test_build_design_matrix_uses_non_reference_condition_dummies() -> None:
    df = pd.DataFrame(
        {
            "condition": ["paper_even", "paper_gain_small"],
            "model": ["model-a", "model-a"],
            "temperature": ["1", "1"],
            "prompt_version": ["absolute", "absolute"],
            "log_wager": [5.0, 5.2],
            "risk_profile": [3, 4],
        }
    )

    X, y = _build_design_matrix(
        df,
        conditions=["paper_gain_small"],
        reference="paper_even",
    )

    assert "const" in X.columns
    assert "paper_gain_small" in X.columns
    assert X["paper_gain_small"].tolist() == [0.0, 1.0]
    assert y["log_wager"].tolist() == [5.0, 5.2]
    assert y["risk_profile"].tolist() == [3, 4]
