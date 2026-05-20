#!/usr/bin/env python3
"""Build behavioral report tables from the local realization results CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import pandas as pd

from realization_effect.analysis import (
    PAPER_CONDITIONS,
    PAPER_REFERENCE,
    REALIZED_CONDITIONS,
    REALIZED_REFERENCE,
    _build_design_matrix,
    run_ols,
)


DEFAULT_RESULTS = Path("results/results.csv")
DEFAULT_GEMMA_BEHAVIOR = Path("results/final/activation_vectors/realization_vector_v1_layer18/behavior_eval.csv")
DEFAULT_OUTPUT_DIR = Path("results/final/report_realization_v1/01_behavioral_replication")


def _truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _clean_regular_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["valid_bool"] = _truthy(df["valid"])
    df["valid_risk_bool"] = _truthy(df["valid_risk_profile"])
    df["parsed_wager"] = pd.to_numeric(df["parsed_wager"], errors="coerce")
    df["log_wager"] = pd.to_numeric(df["log_wager"], errors="coerce")
    df["risk_profile"] = pd.to_numeric(df["risk_profile"], errors="coerce")
    df["temperature"] = df["temperature"].astype(str)
    return df


def _coefficient_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    valid_df = df[df["valid_bool"]].dropna(subset=["log_wager"]).copy()
    rows: list[dict[str, Any]] = []
    specs = [
        ("paper", PAPER_CONDITIONS, PAPER_REFERENCE),
        ("realized", REALIZED_CONDITIONS, REALIZED_REFERENCE),
    ]
    for outcome_type, conditions, reference in specs:
        sub = valid_df[valid_df["outcome_type"] == outcome_type].copy()
        x, y_df = _build_design_matrix(sub, conditions, reference)
        for dependent_variable in ["log_wager", "risk_profile"]:
            y = y_df[dependent_variable].dropna()
            x_aligned = x.loc[y.index]
            result = run_ols(x_aligned, y, "HC3")
            for condition in conditions:
                if condition not in result.params:
                    continue
                rows.append(
                    {
                        "dataset_id": "regular_prompting_realization_effect",
                        "outcome_type": outcome_type,
                        "dependent_variable": dependent_variable,
                        "reference_condition": reference,
                        "condition": condition,
                        "n_obs": int(result.nobs),
                        "coef": float(result.params[condition]),
                        "std_error": float(result.bse[condition]),
                        "p_value_two_sided": float(result.pvalues[condition]),
                        "significant_05": bool(result.pvalues[condition] < 0.05),
                    }
                )
    return rows


def _summary_regular(df: pd.DataFrame, input_path: Path) -> dict[str, Any]:
    return {
        "dataset_id": "regular_prompting_realization_effect",
        "section": "01_behavioral_replication",
        "input_path": str(input_path),
        "description": "Original prompt-only realization-effect behavioral replication dataset, reconciled from canonical absolute and balance prompt blocks.",
        "rows_total": len(df),
        "valid_wager_rows": int(df["valid_bool"].sum()),
        "valid_risk_rows": int(df["valid_risk_bool"].sum()),
        "both_valid_rows": int((df["valid_bool"] & df["valid_risk_bool"]).sum()),
        "model_count": df["model"].nunique(),
        "condition_count": df["condition"].nunique(),
        "prompt_source_count": "",
        "notes": "This is the regular realization-effect-by-prompting result and belongs in Section 1. It includes both absolute and balance prompt versions.",
    }


def _summary_gemma(path: Path) -> dict[str, Any]:
    df = pd.read_csv(path)
    valid_amount = _truthy(df["valid_amount"])
    valid_risk = _truthy(df["valid_risk_profile"])
    return {
        "dataset_id": "gemma_activation_prompt_behavior_eval",
        "section": "01_behavioral_replication",
        "input_path": str(path),
        "description": "Prompt-only Gemma behavior run on the paired paper_open vs realized_closed activation-vector prompt set.",
        "rows_total": len(df),
        "valid_wager_rows": int(valid_amount.sum()),
        "valid_risk_rows": int(valid_risk.sum()),
        "both_valid_rows": int((valid_amount & valid_risk).sum()),
        "model_count": df["model_id"].nunique(),
        "condition_count": df["pair_role"].nunique(),
        "prompt_source_count": df["source_llm"].nunique(),
        "notes": "This is still behavioral prompting, but uses the prompt set later used for activation readout and steering.",
    }


def _robustness_rows_regular(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group_type, group_cols in [
        ("model_prompt_version", ["model", "prompt_version"]),
        ("model_temperature_prompt_version", ["model", "temperature", "prompt_version"]),
    ]:
        for key, sub in df.groupby(group_cols, dropna=False):
            if not isinstance(key, tuple):
                key = (key,)
            valid_wager = sub[sub["valid_bool"]]
            valid_risk = sub[sub["valid_risk_bool"]]
            rows.append(
                {
                    "dataset_id": "regular_prompting_realization_effect",
                    "group_type": group_type,
                    "group": " | ".join(str(part) for part in key),
                    "rows": len(sub),
                    "valid_wager_rows": int(sub["valid_bool"].sum()),
                    "valid_risk_rows": int(sub["valid_risk_bool"].sum()),
                    "mean_wager": float(valid_wager["parsed_wager"].mean()) if len(valid_wager) else "",
                    "mean_risk_profile": float(valid_risk["risk_profile"].mean()) if len(valid_risk) else "",
                    "condition_count": sub["condition"].nunique(),
                }
            )
    return rows


def _robustness_rows_gemma(path: Path) -> list[dict[str, Any]]:
    df = pd.read_csv(path)
    df["valid_amount_bool"] = _truthy(df["valid_amount"])
    df["valid_risk_bool"] = _truthy(df["valid_risk_profile"])
    df["parsed_amount"] = pd.to_numeric(df["parsed_amount"], errors="coerce")
    df["risk_profile"] = pd.to_numeric(df["risk_profile"], errors="coerce")
    rows: list[dict[str, Any]] = []
    for source_llm, sub in df.groupby("source_llm", dropna=False):
        valid_wager = sub[sub["valid_amount_bool"]]
        valid_risk = sub[sub["valid_risk_bool"]]
        rows.append(
            {
                "dataset_id": "gemma_activation_prompt_behavior_eval",
                "group_type": "prompt_source_llm",
                "group": source_llm,
                "rows": len(sub),
                "valid_wager_rows": int(sub["valid_amount_bool"].sum()),
                "valid_risk_rows": int(sub["valid_risk_bool"].sum()),
                "mean_wager": float(valid_wager["parsed_amount"].mean()) if len(valid_wager) else "",
                "mean_risk_profile": float(valid_risk["risk_profile"].mean()) if len(valid_risk) else "",
                "condition_count": sub["pair_role"].nunique(),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build behavioral report tables.")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--gemma-behavior", type=Path, default=DEFAULT_GEMMA_BEHAVIOR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    regular = _clean_regular_results(args.results)
    coefficient_rows = _coefficient_rows(regular)
    summary_rows = [_summary_regular(regular, args.results), _summary_gemma(args.gemma_behavior)]
    robustness_rows = _robustness_rows_regular(regular) + _robustness_rows_gemma(args.gemma_behavior)

    _write_csv(
        args.output_dir / "condition_coefficients.csv",
        coefficient_rows,
        [
            "dataset_id",
            "outcome_type",
            "dependent_variable",
            "reference_condition",
            "condition",
            "n_obs",
            "coef",
            "std_error",
            "p_value_two_sided",
            "significant_05",
        ],
    )
    _write_csv(
        args.output_dir / "cleaned_behavior_summary.csv",
        summary_rows,
        [
            "dataset_id",
            "section",
            "input_path",
            "description",
            "rows_total",
            "valid_wager_rows",
            "valid_risk_rows",
            "both_valid_rows",
            "model_count",
            "condition_count",
            "prompt_source_count",
            "notes",
        ],
    )
    _write_csv(
        args.output_dir / "model_robustness_summary.csv",
        robustness_rows,
        [
            "dataset_id",
            "group_type",
            "group",
            "rows",
            "valid_wager_rows",
            "valid_risk_rows",
            "mean_wager",
            "mean_risk_profile",
            "condition_count",
        ],
    )
    print(f"wrote behavioral report tables to {args.output_dir}")


if __name__ == "__main__":
    main()
