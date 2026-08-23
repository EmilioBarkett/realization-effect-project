#!/usr/bin/env python3
"""Analyze whether realization-direction projections predict behavior."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf


DEFAULT_BEHAVIOR = Path("results/final/activation_vectors/realization_vector_v1_layer18/behavior_eval.csv")
DEFAULT_OUTPUT_DIR = Path("results/final/report_realization_v1/02_activation_readout")
DEFAULT_FIGURE = Path("results/final/report_realization_v1/figures/projection_behavior_link.pdf")
DEFAULT_REPORT_FIGURE = Path("reports/final/figures/projection_behavior_link.pdf")


def _truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _zscore(series: pd.Series) -> pd.Series:
    return (series - series.mean()) / series.std(ddof=0)


def _ols_row(
    df: pd.DataFrame,
    *,
    analysis: str,
    outcome: str,
    formula: str,
    projection_term: str,
) -> dict[str, Any]:
    result = smf.ols(formula, data=df).fit(cov_type="HC3")
    return {
        "analysis": analysis,
        "outcome": outcome,
        "n": int(result.nobs),
        "formula": formula,
        "projection_term": projection_term,
        "projection_coef": float(result.params[projection_term]),
        "projection_se_hc3": float(result.bse[projection_term]),
        "projection_p_value": float(result.pvalues[projection_term]),
        "r_squared": float(result.rsquared),
    }


def _corr_row(
    df: pd.DataFrame,
    *,
    analysis: str,
    x: str,
    y: str,
) -> dict[str, Any]:
    sub = df[[x, y]].dropna()
    return {
        "analysis": analysis,
        "x": x,
        "y": y,
        "n": len(sub),
        "pearson_r": float(sub[x].corr(sub[y], method="pearson")) if len(sub) >= 2 else None,
        "spearman_r": float(sub[x].corr(sub[y], method="spearman")) if len(sub) >= 2 else None,
    }


def _load_behavior(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["projection"] = pd.to_numeric(df["projection"], errors="coerce")
    df["parsed_amount"] = pd.to_numeric(df["parsed_amount"], errors="coerce")
    df["risk_profile"] = pd.to_numeric(df["risk_profile"], errors="coerce")
    df["valid_amount_bool"] = _truthy(df["valid_amount"])
    df["valid_risk_bool"] = _truthy(df["valid_risk_profile"])
    df["both_valid_bool"] = df["valid_amount_bool"] & df["valid_risk_bool"]
    df["projection_z"] = _zscore(df["projection"])
    df["projection_centered"] = df["projection"] - df["projection"].mean()
    return df


def _pair_deltas(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pair_id, group in df.groupby("pair_id"):
        roles = {row["pair_role"]: row for row in group.to_dict("records")}
        if "paper_open" not in roles or "realized_closed" not in roles:
            continue
        paper = roles["paper_open"]
        realized = roles["realized_closed"]
        row = {
            "pair_id": pair_id,
            "domain": realized.get("domain") or paper.get("domain"),
            "outcome_valence": realized.get("outcome_valence") or paper.get("outcome_valence"),
            "amount_bucket": realized.get("amount_bucket") or paper.get("amount_bucket"),
            "source_llm": realized.get("source_llm") or paper.get("source_llm"),
            "paper_projection": paper["projection"],
            "realized_projection": realized["projection"],
            "projection_delta": realized["projection"] - paper["projection"],
            "paper_amount": paper["parsed_amount"],
            "realized_amount": realized["parsed_amount"],
            "amount_delta": realized["parsed_amount"] - paper["parsed_amount"],
            "paper_risk": paper["risk_profile"],
            "realized_risk": realized["risk_profile"],
            "risk_delta": realized["risk_profile"] - paper["risk_profile"],
            "valid_amount_pair": bool(paper["valid_amount_bool"] and realized["valid_amount_bool"]),
            "valid_risk_pair": bool(paper["valid_risk_bool"] and realized["valid_risk_bool"]),
            "both_valid_pair": bool(paper["both_valid_bool"] and realized["both_valid_bool"]),
        }
        rows.append(row)
    out = pd.DataFrame(rows)
    out["projection_delta_z"] = _zscore(out["projection_delta"])
    return out


def _write_tables(
    df: pd.DataFrame,
    pairs: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "projection_behavior_prompt_level.csv", index=False)
    pairs.to_csv(output_dir / "projection_behavior_pair_deltas.csv", index=False)

    prompt_amount = df[df["valid_amount_bool"]].copy()
    prompt_risk = df[df["valid_risk_bool"]].copy()
    pair_amount = pairs[pairs["valid_amount_pair"]].copy()
    pair_risk = pairs[pairs["valid_risk_pair"]].copy()

    regression_rows = [
        _ols_row(
            prompt_amount,
            analysis="prompt_level_no_controls",
            outcome="parsed_amount",
            formula="parsed_amount ~ projection_z",
            projection_term="projection_z",
        ),
        _ols_row(
            prompt_risk,
            analysis="prompt_level_no_controls",
            outcome="risk_profile",
            formula="risk_profile ~ projection_z",
            projection_term="projection_z",
        ),
        _ols_row(
            prompt_amount,
            analysis="prompt_level_with_prompt_controls",
            outcome="parsed_amount",
            formula=(
                "parsed_amount ~ projection_z + C(pair_role) + C(domain) + "
                "C(outcome_valence) + C(amount_bucket) + C(source_llm)"
            ),
            projection_term="projection_z",
        ),
        _ols_row(
            prompt_risk,
            analysis="prompt_level_with_prompt_controls",
            outcome="risk_profile",
            formula=(
                "risk_profile ~ projection_z + C(pair_role) + C(domain) + "
                "C(outcome_valence) + C(amount_bucket) + C(source_llm)"
            ),
            projection_term="projection_z",
        ),
        _ols_row(
            pair_amount,
            analysis="pair_delta_no_controls",
            outcome="amount_delta",
            formula="amount_delta ~ projection_delta_z",
            projection_term="projection_delta_z",
        ),
        _ols_row(
            pair_risk,
            analysis="pair_delta_no_controls",
            outcome="risk_delta",
            formula="risk_delta ~ projection_delta_z",
            projection_term="projection_delta_z",
        ),
        _ols_row(
            pair_amount,
            analysis="pair_delta_with_prompt_controls",
            outcome="amount_delta",
            formula="amount_delta ~ projection_delta_z + C(domain) + C(outcome_valence) + C(amount_bucket) + C(source_llm)",
            projection_term="projection_delta_z",
        ),
        _ols_row(
            pair_risk,
            analysis="pair_delta_with_prompt_controls",
            outcome="risk_delta",
            formula="risk_delta ~ projection_delta_z + C(domain) + C(outcome_valence) + C(amount_bucket) + C(source_llm)",
            projection_term="projection_delta_z",
        ),
    ]
    regression_df = pd.DataFrame(regression_rows)
    regression_df.to_csv(output_dir / "projection_behavior_regressions.csv", index=False)

    corr_rows = [
        _corr_row(prompt_amount, analysis="prompt_level_valid_amount", x="projection_z", y="parsed_amount"),
        _corr_row(prompt_risk, analysis="prompt_level_valid_risk", x="projection_z", y="risk_profile"),
        _corr_row(pair_amount, analysis="pair_delta_valid_amount", x="projection_delta_z", y="amount_delta"),
        _corr_row(pair_risk, analysis="pair_delta_valid_risk", x="projection_delta_z", y="risk_delta"),
    ]
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(output_dir / "projection_behavior_correlations.csv", index=False)

    summary = {
        "input_rows": len(df),
        "valid_amount_rows": int(df["valid_amount_bool"].sum()),
        "valid_risk_rows": int(df["valid_risk_bool"].sum()),
        "both_valid_rows": int(df["both_valid_bool"].sum()),
        "pair_rows": len(pairs),
        "valid_amount_pairs": int(pairs["valid_amount_pair"].sum()),
        "valid_risk_pairs": int(pairs["valid_risk_pair"].sum()),
        "both_valid_pairs": int(pairs["both_valid_pair"].sum()),
        "projection_by_role": (
            df.groupby("pair_role")["projection"]
            .agg(["count", "mean", "std"])
            .reset_index()
            .to_dict(orient="records")
        ),
        "regressions": regression_rows,
        "correlations": corr_rows,
    }
    (output_dir / "projection_behavior_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _plot(df: pd.DataFrame, pairs: pd.DataFrame, figure_path: Path, report_figure_path: Path | None) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    if report_figure_path is not None:
        report_figure_path.parent.mkdir(parents=True, exist_ok=True)

    sns.set_theme(context="paper", style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.0))

    prompt_amount = df[df["valid_amount_bool"]].copy()
    prompt_risk = df[df["valid_risk_bool"]].copy()
    pair_amount = pairs[pairs["valid_amount_pair"]].copy()
    pair_risk = pairs[pairs["valid_risk_pair"]].copy()

    panels = [
        (axes[0, 0], prompt_amount, "projection_z", "parsed_amount", "Prompt level: projection vs wager", "Projection z-score", "Wager"),
        (axes[0, 1], prompt_risk, "projection_z", "risk_profile", "Prompt level: projection vs risk", "Projection z-score", "Risk profile"),
        (axes[1, 0], pair_amount, "projection_delta_z", "amount_delta", "Pair deltas: projection vs wager", "Projection-delta z-score", "Realized - paper wager"),
        (axes[1, 1], pair_risk, "projection_delta_z", "risk_delta", "Pair deltas: projection vs risk", "Projection-delta z-score", "Realized - paper risk"),
    ]
    for ax, sub, x, y, title, xlabel, ylabel in panels:
        sns.regplot(
            data=sub,
            x=x,
            y=y,
            ax=ax,
            scatter_kws={"s": 12, "alpha": 0.35, "color": "#2F6F9F"},
            line_kws={"color": "#B44E3A", "linewidth": 1.4},
            ci=95,
        )
        ax.axhline(0, color="#333333", linewidth=0.7, alpha=0.6)
        ax.axvline(0, color="#333333", linewidth=0.7, alpha=0.6)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    fig.suptitle("Projection-behavior links are weak", y=0.995, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(figure_path)
    if report_figure_path is not None:
        report_figure_path.write_bytes(figure_path.read_bytes())
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze projection strength versus behavior outcomes.")
    parser.add_argument("--behavior-eval", type=Path, default=DEFAULT_BEHAVIOR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--report-figure", type=Path, default=DEFAULT_REPORT_FIGURE)
    args = parser.parse_args()

    df = _load_behavior(args.behavior_eval)
    pairs = _pair_deltas(df)
    summary = _write_tables(df, pairs, args.output_dir)
    _plot(df, pairs, args.figure, args.report_figure)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
