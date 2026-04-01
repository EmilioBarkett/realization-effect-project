#!/usr/bin/env python3
"""Statistical analysis for the realization-effect LLM experiment.

Mirrors Flepp, Meier & Franck (2021) Table 2 as closely as the vignette
design allows.

Design correspondence
─────────────────────────────────────────────────────────────────────────────
Paper (field study)            →  LLM replication (vignette study)
─────────────────────────────────────────────────────────────────────────────
Within-person FE OLS           →  OLS with model + temperature +
                                  prompt_version as covariate fixed effects
Quintile dummies               →  Condition dummies (same reference categories)
HC-robust standard errors      →  HC3 robust standard errors
LogTotalWager (DV 1)           →  log(parsed_wager)
LogT-winCasino (DV 2)          →  risk_profile (1–5 slot machine risk choice)
Single joint regression with   →  Two separate regressions (each LLM trial
  both paper & realized DVs       tests one type only; in the field data both
                                  accumulate for the same session)
─────────────────────────────────────────────────────────────────────────────

Hypotheses (Flepp et al. 2021, Section 3.2):
  H1a  Paper losses  → ↑ log-wager vs paper_even baseline          (one-sided)
  H1b  Larger paper losses → ↑ log-wager more
         (loss_large > loss_medium > loss_small)
  H2a  Paper gains   → ↑ log-wager vs paper_even baseline          (one-sided)
  H2b  Larger paper gains  → ↑ log-wager more
         (gain_large > gain_small)
  H3a  Realized losses → ↓ log-wager vs realized_small_loss        (one-sided)
  H3b  Larger realized losses → ↓ log-wager more
         (extreme < large < medium, in the direction of less risk-taking)
  H4   Realized gains → no change vs realized_small_loss           (two-sided)

Usage:
  python analyze_results.py results/results.csv
  python analyze_results.py results/results.csv --model openai/gpt-4o
  python analyze_results.py results/results.csv --per-model
  python analyze_results.py results/results.csv --prompt-version qualitative
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as scipy_stats

# ── Condition ordering and reference categories ───────────────────────────────

# Reference category for paper outcomes (paper Q3: −96 to 0 in the paper).
PAPER_REFERENCE = "paper_even"

# Reference category for realized outcomes (realized Q4: −62 to 0 in the paper).
REALIZED_REFERENCE = "realized_small_loss"

# Paper conditions in ascending-loss order; reference last in block for display.
PAPER_CONDITIONS = [
    "paper_loss_large",   # Q1 ≤ −310
    "paper_loss_medium",  # Q2 −309 to −97
    "paper_loss_small",   # within Q3: −96 to 0 (new condition; not in paper)
    # paper_even omitted here — it is the reference
    "paper_gain_small",   # Q4 1–80
    "paper_gain_large",   # Q5 ≥ 81
]

# Realized conditions in ascending-loss order; reference last in block.
REALIZED_CONDITIONS = [
    "realized_extreme_loss",  # Q1 ≤ −2,791
    "realized_large_loss",    # Q2 −2,790 to −788
    "realized_medium_loss",   # Q3 −787 to −63 (not sig. in paper)
    # realized_small_loss omitted — it is the reference
    "realized_gain",          # Q5 ≥ 1 (not sig. in paper)
]

STARS = {0.01: "***", 0.05: "**", 0.1: "*"}


def significance_stars(p: float) -> str:
    for threshold, star in STARS.items():
        if p < threshold:
            return star
    return ""


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(
    csv_path: Path,
    model_filter: Optional[str] = None,
    prompt_version_filter: Optional[str] = None,
) -> pd.DataFrame:
    """Load and clean results CSV, returning only valid wager rows."""
    df = pd.read_csv(csv_path)

    required = {"condition", "outcome_type", "parsed_wager", "log_wager",
                "valid", "model", "temperature", "prompt_version"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Results CSV is missing columns: {missing}")

    # Keep only rows where the wager was successfully parsed.
    df = df[df["valid"].astype(str).str.lower() == "true"].copy()
    df["log_wager"] = pd.to_numeric(df["log_wager"], errors="coerce")
    df = df.dropna(subset=["log_wager"])

    if model_filter:
        df = df[df["model"] == model_filter]
    if prompt_version_filter:
        df = df[df["prompt_version"] == prompt_version_filter]

    if df.empty:
        raise ValueError("No valid rows remain after filtering.")

    # Coerce risk_profile if present.
    if "risk_profile" in df.columns:
        df["risk_profile"] = pd.to_numeric(df["risk_profile"], errors="coerce")

    df["temperature"] = df["temperature"].astype(str)
    return df


# ── Regression ────────────────────────────────────────────────────────────────

def _build_design_matrix(
    df: pd.DataFrame,
    conditions: List[str],
    reference: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (X, y_df) for the OLS regression.

    X includes:
      - dummy for each non-reference condition present in the data
      - model fixed effects (one-hot, first model dropped)
      - temperature fixed effects
      - prompt_version fixed effects
      - intercept

    y_df contains log_wager and (if available) risk_profile.
    """
    present = [c for c in conditions if c in df["condition"].unique()]
    cond_dummies = pd.get_dummies(df["condition"], prefix="", prefix_sep="")[present].astype(float)

    model_dummies = pd.get_dummies(df["model"], prefix="model", drop_first=True).astype(float)
    temp_dummies = pd.get_dummies(df["temperature"], prefix="temp", drop_first=True).astype(float)
    pv_dummies = pd.get_dummies(df["prompt_version"], prefix="pv", drop_first=True).astype(float)

    X = pd.concat([cond_dummies, model_dummies, temp_dummies, pv_dummies], axis=1)
    X = sm.add_constant(X, has_constant="add")

    y_df = df[["log_wager"]].copy()
    if "risk_profile" in df.columns:
        y_df["risk_profile"] = df["risk_profile"]

    return X, y_df


def run_ols(
    X: pd.DataFrame,
    y: pd.Series,
    cov_type: str = "HC3",
) -> sm.regression.linear_model.RegressionResultsWrapper:
    model = sm.OLS(y, X, missing="drop")
    return model.fit(cov_type=cov_type)


# ── Wald test for coefficient equality ───────────────────────────────────────

def wald_test_difference(
    result: sm.regression.linear_model.RegressionResultsWrapper,
    name_a: str,
    name_b: str,
) -> Tuple[float, float]:
    """Test H0: coef(name_a) == coef(name_b). Returns (F-statistic, two-sided p)."""
    params = result.params.index.tolist()
    if name_a not in params or name_b not in params:
        return np.nan, np.nan
    R = np.zeros((1, len(params)))
    R[0, params.index(name_a)] = 1.0
    R[0, params.index(name_b)] = -1.0
    wald = result.wald_test(R, use_f=True)
    f_stat = float(np.squeeze(wald.statistic))
    p_val = float(wald.pvalue)
    return f_stat, p_val


def one_sided_p(coef: float, two_sided_p: float, direction: str = "positive") -> float:
    """Convert two-sided p to one-sided, checking sign of coefficient."""
    if direction == "positive" and coef > 0:
        return two_sided_p / 2
    if direction == "negative" and coef < 0:
        return two_sided_p / 2
    return 1.0 - two_sided_p / 2


# ── Display ───────────────────────────────────────────────────────────────────

def print_regression_table(
    result: sm.regression.linear_model.RegressionResultsWrapper,
    conditions: List[str],
    reference: str,
    dv_label: str,
    n_obs: int,
) -> None:
    """Print a Table-2-style regression block for the condition dummies."""
    print(f"\n  Dependent variable: {dv_label}")
    print(f"  Reference category: {reference}")
    print(f"  N = {n_obs:,}  |  Within-R² ≈ {result.rsquared:.3f}")
    print(f"  {'Condition':<28}  {'Coef':>8}  {'SE':>8}  {'p (2-sided)':>12}  {'Stars':>5}")
    print("  " + "─" * 66)

    for cond in conditions:
        if cond not in result.params:
            print(f"  {cond:<28}  {'(not in data)':>8}")
            continue
        coef = result.params[cond]
        se = result.bse[cond]
        pval = result.pvalues[cond]
        stars = significance_stars(pval)
        print(f"  {cond:<28}  {coef:>8.4f}  {se:>8.4f}  {pval:>12.4f}  {stars:>5}")

    print(f"  {'[Baseline]':<28}  {0.0:>8.4f}  {'—':>8}  {'—':>12}")


def _coef_and_p(
    result: sm.regression.linear_model.RegressionResultsWrapper,
    name: str,
) -> Tuple[float, float]:
    if name not in result.params:
        return np.nan, np.nan
    return float(result.params[name]), float(result.pvalues[name])


# ── Hypothesis tests ──────────────────────────────────────────────────────────

def test_hypotheses(
    paper_results: Dict[str, sm.regression.linear_model.RegressionResultsWrapper],
    realized_results: Dict[str, sm.regression.linear_model.RegressionResultsWrapper],
) -> None:
    """Print structured accept/partial/reject verdicts for H1a–H4."""
    print("\n" + "=" * 70)
    print("HYPOTHESIS TESTS  (one-sided where direction is predicted)")
    print("=" * 70)

    dvs = list(paper_results.keys())

    for dv in dvs:
        pr = paper_results[dv]
        rr = realized_results[dv]
        print(f"\n── DV: {dv} ──")

        # H1a: paper losses ↑ risk-taking
        print("\n  H1a  Paper losses ↑ log-wager vs paper_even (one-sided)")
        for cond in ["paper_loss_small", "paper_loss_medium", "paper_loss_large"]:
            coef, p2 = _coef_and_p(pr, cond)
            if np.isnan(coef):
                continue
            p1 = one_sided_p(coef, p2, "positive")
            print(f"    {cond:<28}  coef={coef:+.4f}  p(one-sided)={p1:.4f}  {significance_stars(p1)}")

        # H1b: larger paper losses ↑ more
        print("\n  H1b  paper_loss_large > paper_loss_medium > paper_loss_small")
        for a, b in [("paper_loss_large", "paper_loss_medium"),
                     ("paper_loss_medium", "paper_loss_small")]:
            f, p = wald_test_difference(pr, a, b)
            ca = pr.params.get(a, np.nan)
            cb = pr.params.get(b, np.nan)
            diff = ca - cb
            if np.isnan(f):
                print(f"    {a} vs {b}: insufficient data")
                continue
            p1 = p / 2 if diff > 0 else 1.0 - p / 2
            print(f"    {a} − {b}:  Δcoef={diff:+.4f}  F={f:.2f}  p(one-sided)={p1:.4f}  {significance_stars(p1)}")

        # H2a: paper gains ↑ risk-taking
        print("\n  H2a  Paper gains ↑ log-wager vs paper_even (one-sided)")
        for cond in ["paper_gain_small", "paper_gain_large"]:
            coef, p2 = _coef_and_p(pr, cond)
            if np.isnan(coef):
                continue
            p1 = one_sided_p(coef, p2, "positive")
            print(f"    {cond:<28}  coef={coef:+.4f}  p(one-sided)={p1:.4f}  {significance_stars(p1)}")

        # H2b: larger paper gains ↑ more
        print("\n  H2b  paper_gain_large > paper_gain_small")
        f, p = wald_test_difference(pr, "paper_gain_large", "paper_gain_small")
        ca = pr.params.get("paper_gain_large", np.nan)
        cb = pr.params.get("paper_gain_small", np.nan)
        if not np.isnan(f):
            diff = ca - cb
            p1 = p / 2 if diff > 0 else 1.0 - p / 2
            print(f"    gain_large − gain_small:  Δcoef={diff:+.4f}  F={f:.2f}  p(one-sided)={p1:.4f}  {significance_stars(p1)}")

        # H3a: realized losses ↓ risk-taking
        print("\n  H3a  Realized losses ↓ log-wager vs realized_small_loss (one-sided)")
        for cond in ["realized_medium_loss", "realized_large_loss", "realized_extreme_loss"]:
            coef, p2 = _coef_and_p(rr, cond)
            if np.isnan(coef):
                continue
            p1 = one_sided_p(coef, p2, "negative")
            print(f"    {cond:<28}  coef={coef:+.4f}  p(one-sided)={p1:.4f}  {significance_stars(p1)}")

        # H3b: larger realized losses ↓ more
        print("\n  H3b  extreme < large < medium (more negative = more risk reduction)")
        for a, b in [("realized_extreme_loss", "realized_large_loss"),
                     ("realized_large_loss", "realized_medium_loss")]:
            f, p = wald_test_difference(rr, a, b)
            ca = rr.params.get(a, np.nan)
            cb = rr.params.get(b, np.nan)
            if np.isnan(f):
                print(f"    {a} vs {b}: insufficient data")
                continue
            diff = ca - cb  # expected to be negative (more extreme = lower coef)
            p1 = p / 2 if diff < 0 else 1.0 - p / 2
            print(f"    {a} − {b}:  Δcoef={diff:+.4f}  F={f:.2f}  p(one-sided)={p1:.4f}  {significance_stars(p1)}")

        # H4: realized gains = baseline (two-sided, expect no effect)
        print("\n  H4   Realized gains = realized_small_loss baseline (two-sided)")
        coef, p2 = _coef_and_p(rr, "realized_gain")
        if not np.isnan(coef):
            print(f"    realized_gain               coef={coef:+.4f}  p(two-sided)={p2:.4f}  {significance_stars(p2)}")
            verdict = "SUPPORTED (p > 0.10)" if p2 >= 0.10 else "REJECTED (significant effect found)"
            print(f"    → H4: {verdict}")

    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def analyse(
    df: pd.DataFrame,
    label: str = "",
    cov_type: str = "HC3",
) -> None:
    """Run the full paper/realized regression pair and hypothesis tests for one subset."""
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"SAMPLE: {label or 'all data'}  (N = {len(df):,} valid trials)")
    print(sep)

    paper_df = df[df["outcome_type"] == "paper"].copy()
    realized_df = df[df["outcome_type"] == "realized"].copy()

    if paper_df.empty or realized_df.empty:
        print("  ⚠  Insufficient data for one or both outcome types — skipping.")
        return

    # Build design matrices.
    Xp, yp_df = _build_design_matrix(paper_df, PAPER_CONDITIONS, PAPER_REFERENCE)
    Xr, yr_df = _build_design_matrix(realized_df, REALIZED_CONDITIONS, REALIZED_REFERENCE)

    # Determine which DVs are available.
    dvs_paper = ["log_wager"]
    dvs_realized = ["log_wager"]
    if "risk_profile" in yp_df.columns and yp_df["risk_profile"].notna().sum() > 10:
        dvs_paper.append("risk_profile")
    if "risk_profile" in yr_df.columns and yr_df["risk_profile"].notna().sum() > 10:
        dvs_realized.append("risk_profile")

    paper_results = {}
    realized_results = {}

    # ── Paper outcomes regression ────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("PAPER OUTCOMES  (within-visit; mental account open)")
    print("Analog to Flepp et al. 2021 Table 2, prior outcomes within the visit")
    print("─" * 70)
    for dv in dvs_paper:
        yp = yp_df[dv].dropna()
        Xp_aligned = Xp.loc[yp.index]
        res = run_ols(Xp_aligned, yp, cov_type)
        paper_results[dv] = res
        print_regression_table(res, PAPER_CONDITIONS, PAPER_REFERENCE, dv, len(yp))

    # ── Realized outcomes regression ─────────────────────────────────────────
    print("\n" + "─" * 70)
    print("REALIZED OUTCOMES  (between-visits; mental account closed)")
    print("Analog to Flepp et al. 2021 Table 2, prior outcomes between visits")
    print("─" * 70)
    for dv in dvs_realized:
        yr = yr_df[dv].dropna()
        Xr_aligned = Xr.loc[yr.index]
        res = run_ols(Xr_aligned, yr, cov_type)
        realized_results[dv] = res
        print_regression_table(res, REALIZED_CONDITIONS, REALIZED_REFERENCE, dv, len(yr))

    # ── Hypothesis tests ─────────────────────────────────────────────────────
    # Use log_wager for the primary test; risk_profile mirrors T-winCasino.
    primary_paper = {dv: paper_results[dv] for dv in paper_results}
    primary_realized = {dv: realized_results[dv] for dv in realized_results}
    test_hypotheses(primary_paper, primary_realized)

    print(f"Notes: HC3 heteroscedasticity-robust standard errors.")
    print(f"       * p<0.10  ** p<0.05  *** p<0.01")
    print(f"       Model, temperature, and prompt_version included as covariate FEs.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze realization-effect LLM experiment results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "results_csv",
        type=Path,
        help="Path to the results CSV produced by run_experiment.py.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Filter to a specific model (e.g. openai/gpt-4o).",
    )
    parser.add_argument(
        "--prompt-version",
        type=str,
        default=None,
        help="Filter to a specific prompt version (absolute, balance, qualitative).",
    )
    parser.add_argument(
        "--per-model",
        action="store_true",
        help="Run a separate analysis for each model in the data.",
    )
    parser.add_argument(
        "--cov-type",
        type=str,
        default="HC3",
        choices=["HC0", "HC1", "HC2", "HC3"],
        help="Heteroscedasticity-robust SE type (default: HC3, matching the paper).",
    )
    args = parser.parse_args()

    df = load_data(args.results_csv, args.model, args.prompt_version)

    if args.per_model:
        for model_name, model_df in df.groupby("model"):
            analyse(model_df, label=str(model_name), cov_type=args.cov_type)
    else:
        analyse(df, cov_type=args.cov_type)


if __name__ == "__main__":
    main()
