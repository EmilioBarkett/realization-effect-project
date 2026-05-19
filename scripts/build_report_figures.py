#!/usr/bin/env python3
"""Build report-ready figures for the realization-effect final report."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


REPORT_ROOT = Path("results/final/report_realization_v1")
FIGURE_DIR = REPORT_ROOT / "figures"
REPORT_FIGURE_DIR = Path("reports/final/figures")


def _style() -> None:
    sns.set_theme(
        context="paper",
        style="whitegrid",
        rc={
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "font.family": "DejaVu Sans",
        },
    )


def _condition_label(condition: str) -> str:
    return condition.replace("_", " ")


def behavioral_coefficients() -> None:
    df = pd.read_csv(REPORT_ROOT / "01_behavioral_replication" / "condition_coefficients.csv")
    df["lower"] = df["coef"] - 1.96 * df["std_error"]
    df["upper"] = df["coef"] + 1.96 * df["std_error"]
    df["condition_label"] = df["condition"].map(_condition_label)
    order = [
        "paper_loss_large",
        "paper_loss_medium",
        "paper_loss_small",
        "paper_gain_small",
        "paper_gain_large",
        "realized_extreme_loss",
        "realized_large_loss",
        "realized_medium_loss",
        "realized_gain",
    ]
    label_order = [_condition_label(value) for value in order]
    df["condition_label"] = pd.Categorical(df["condition_label"], categories=label_order, ordered=True)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.4), sharex=False)
    for ax, dv, title in zip(axes, ["log_wager", "risk_profile"], ["Log wager", "Risk profile"]):
        sub = df[df["dependent_variable"] == dv].sort_values("condition_label")
        colors = sub["outcome_type"].map({"paper": "#2F6F9F", "realized": "#B44E3A"}).tolist()
        y = range(len(sub))
        ax.axvline(0, color="#333333", linewidth=0.8)
        ax.hlines(y, sub["lower"], sub["upper"], color="#555555", linewidth=1.1)
        ax.scatter(sub["coef"], y, c=colors, s=34, zorder=3)
        ax.set_yticks(list(y))
        ax.set_yticklabels(sub["condition_label"])
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel("OLS coefficient vs. condition baseline")
        ax.set_ylabel("")
    fig.suptitle("Prompt-only realization-effect coefficients", y=0.995, fontsize=12)
    fig.text(0.5, 0.02, "Error bars show +/- 1.96 HC3 robust standard errors.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(FIGURE_DIR / "behavioral_coefficients.pdf")
    plt.close(fig)


def activation_projection() -> None:
    df = pd.read_csv(REPORT_ROOT / "02_activation_readout" / "layer18_projection_by_prompt.csv")
    df = df[df["split"].isin(["direction_val", "behavior_eval"])].copy()
    df["projection_centered"] = df["projection"] - df.groupby("split")["projection"].transform("mean")
    df["pair_role_label"] = df["pair_role"].map(
        {"paper_open": "Paper/open", "realized_closed": "Realized/closed"}
    )
    df["split_label"] = df["split"].map(
        {"direction_val": "Direction-validation prompts", "behavior_eval": "Behavior prompts"}
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
    palette = {"Paper/open": "#2F6F9F", "Realized/closed": "#B44E3A"}
    for ax, split_label in zip(axes, ["Direction-validation prompts", "Behavior prompts"]):
        sub = df[df["split_label"] == split_label]
        sns.violinplot(
            data=sub,
            x="pair_role_label",
            y="projection_centered",
            hue="pair_role_label",
            palette=palette,
            inner="quartile",
            cut=0,
            linewidth=0.7,
            ax=ax,
            legend=False,
        )
        means = sub.groupby("pair_role_label")["projection_centered"].mean()
        for i, label in enumerate(["Paper/open", "Realized/closed"]):
            ax.scatter(i, means[label], color="black", s=18, zorder=4)
        ax.axhline(0, color="#333333", linewidth=0.7, alpha=0.7)
        ax.set_title(split_label)
        ax.set_xlabel("")
        ax.set_ylabel("Projection on realization direction, centered" if ax is axes[0] else "")
    fig.suptitle("Gemma layer-18 realization readout", y=0.99, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGURE_DIR / "activation_projection_layer18.pdf")
    plt.close(fig)


def steering_dose_response() -> None:
    df = pd.read_csv(REPORT_ROOT / "03_steering_intervention" / "steering_matched_deltas.csv")
    df = df[df["subset"] == "strict_valid_matched"].copy()
    df = df.sort_values("scale")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    panels = [
        ("mean_amount_delta", "median_amount_delta", "Wager delta", "CHF"),
        ("mean_risk_delta", "median_risk_delta", "Risk-profile delta", "1-5 risk scale"),
    ]
    for ax, (mean_col, median_col, title, ylabel) in zip(axes, panels):
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.plot(df["scale"], df[mean_col], marker="o", color="#2F6F9F", label="Mean matched delta")
        ax.scatter(df["scale"], df[median_col], marker="s", color="#B44E3A", label="Median matched delta", zorder=3)
        ax.set_title(title)
        ax.set_xlabel("Steering scale")
        ax.set_ylabel(ylabel)
        ax.set_xticks(df["scale"])
        ax.legend(frameon=False)
    fig.suptitle("Layer-18 realization steering does not produce a sign-symmetric risk shift", y=0.995, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGURE_DIR / "steering_dose_response.pdf")
    plt.close(fig)


def compliance_by_scale() -> None:
    df = pd.read_csv(REPORT_ROOT / "03_steering_intervention" / "compliance_by_scale_source.csv")
    df = df[df["group_type"] == "source_llm"].copy()
    df["source"] = df["group"].map({"gpt54": "GPT-5.4", "grok_fast": "Grok fast", "sonnet": "Sonnet"})
    df["noncompliance_percent"] = df["noncompliance_rate"] * 100.0

    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    sns.barplot(
        data=df,
        x="scale",
        y="noncompliance_percent",
        hue="source",
        palette=["#2F6F9F", "#6A8F4E", "#B44E3A"],
        ax=ax,
    )
    ax.set_title("Exactly-two-integer failures by steering scale and prompt source")
    ax.set_xlabel("Steering scale")
    ax.set_ylabel("Noncompliant responses (%)")
    ax.legend(title="Prompt source", frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "compliance_by_scale.pdf")
    plt.close(fig)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    _style()
    behavioral_coefficients()
    activation_projection()
    steering_dose_response()
    compliance_by_scale()
    for name in [
        "behavioral_coefficients.pdf",
        "activation_projection_layer18.pdf",
        "steering_dose_response.pdf",
        "compliance_by_scale.pdf",
    ]:
        source = FIGURE_DIR / name
        target = REPORT_FIGURE_DIR / name
        target.write_bytes(source.read_bytes())
        print(FIGURE_DIR / name)


if __name__ == "__main__":
    main()
