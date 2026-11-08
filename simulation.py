#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monte Carlo simulation for:
"Distributional Sensitivity of Slack-Based Fragility Measures in Weighted Bipartite Networks"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pareto, t, truncnorm


# ============================================================
# Configuration
# ============================================================

THETA_VALUES = np.linspace(0.1, 1.0, 10).round(1)

SCENARIOS = [
    (140, 60),
    (130, 70),
    (120, 80),
    (110, 90),
    (100, 100),
    (90, 110),
    (80, 120),
    (70, 130),
    (60, 140),
]

# Keep the submitted order EXACTLY for the legacy RNG stream.
LEGACY_DIST_NAMES = [
    "gaussian",
    "student_t",
    "power_law_1_2",
    "power_law_1_5",
    "power_law_2_0",
    "power_law_3_0",
    "power_law_5_0",
]

LOGNORMAL_DIST = "lognormal_baci"
DIST_NAMES = LEGACY_DIST_NAMES + [LOGNORMAL_DIST]
ALTERNATIVE_DISTS = [d for d in DIST_NAMES if d != "gaussian"]

TARGET_DEGREE = 20
DEFAULT_REPLICATIONS = 1000
DEFAULT_SEED = 42

# Independent RNG stream for the new distribution; does not disturb legacy RNG.
LOGNORMAL_SEED_OFFSET = 10_000_019

# Final BACI share-calibrated parameters.
LOGNORMAL_SIGMA_BACI = 1.45
LOGNORMAL_MU_BACI = float(np.log(100.0) - 0.5 * LOGNORMAL_SIGMA_BACI**2)

# Empirical BACI targets used in the calibration (metadata only).
BACI_MEAN_TARGET_HHI = 0.185733
BACI_MEAN_TARGET_MAX_SHARE = 0.322521


# ============================================================
# Slack computation
# ============================================================

def compute_quota_slack(
    edges: pd.DataFrame,
    theta: float = 0.70,
    source_col: str = "source",
    target_col: str = "target",
    weight_col: str = "weight",
    min_weight: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute edge-level and active-source-level quota-based Slack."""
    if not 0 < theta <= 1:
        raise ValueError("theta must lie in (0, 1].")

    required_cols = {source_col, target_col, weight_col}
    missing = required_cols - set(edges.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = edges[[source_col, target_col, weight_col]].copy()
    df[weight_col] = pd.to_numeric(df[weight_col], errors="coerce")
    df = df.dropna(subset=[source_col, target_col, weight_col])
    df = df[df[weight_col] > min_weight].copy()

    # Aggregate duplicates if a future robustness topology ever creates them.
    df = (
        df.groupby([source_col, target_col], as_index=False)[weight_col]
        .sum()
    )
    if df.empty:
        raise ValueError("No valid edges after cleaning/filtering.")

    target_totals = (
        df.groupby(target_col, as_index=False)[weight_col]
        .sum()
        .rename(columns={weight_col: "T_j"})
    )
    df = df.merge(target_totals, on=target_col, how="left")
    df["Q_j"] = theta * df["T_j"]
    df = df[(df["T_j"] > 0) & (df["Q_j"] > 0)].copy()

    df["Slack_ij"] = (df["T_j"] - df[weight_col]) / df["Q_j"]
    df["is_pivotal"] = df["Slack_ij"] < 1
    df["target_share_alpha_ij"] = df[weight_col] / df["T_j"]

    source_totals = (
        df.groupby(source_col, as_index=False)[weight_col]
        .sum()
        .rename(columns={weight_col: "s_i"})
    )
    df = df.merge(source_totals, on=source_col, how="left")
    df["beta_ij"] = df[weight_col] / df["s_i"]

    # D_i = sum_j beta_ij * alpha_ij, hence Slack_i^W = (1-D_i)/theta.
    df["weighted_slack_component"] = df["beta_ij"] * df["Slack_ij"]
    df["weighted_alpha_component"] = (
        df["beta_ij"] * df["target_share_alpha_ij"]
    )

    source_slack = (
        df.groupby(source_col, as_index=False)
        .agg(
            Slack_mean=("Slack_ij", "mean"),
            Slack_weighted=("weighted_slack_component", "sum"),
            dominance_D=("weighted_alpha_component", "sum"),
            TargetCount=(target_col, "nunique"),
            total_weight=("s_i", "first"),
            n_pivotal_edges=("is_pivotal", "sum"),
        )
        .rename(columns={source_col: "ID"})
    )

    source_slack["has_any_pivotal_edge"] = (
        source_slack["n_pivotal_edges"] > 0
    )

    return df, source_slack


# ============================================================
# Weight-generating regimes
# ============================================================

def sample_weights(
    dist_name: str,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw strictly positive edge weights for one regime."""
    if size == 0:
        return np.array([], dtype=float)

    if dist_name == "gaussian":
        # Proper N(100,20^2) left-truncation at zero (NOT abs transformation).
        mu, sigma = 100.0, 20.0
        lower = (0.0 - mu) / sigma
        return truncnorm.rvs(
            lower,
            np.inf,
            loc=mu,
            scale=sigma,
            size=size,
            random_state=rng,
        )

    if dist_name == "student_t":
        df, loc, scale = 3, 100.0, 20.0
        out = np.empty(size, dtype=float)
        filled = 0
        while filled < size:
            batch_size = max((size - filled) * 3, 1)
            batch = t.rvs(
                df=df,
                loc=loc,
                scale=scale,
                size=batch_size,
                random_state=rng,
            )
            batch = batch[batch > 0]
            take = min(batch.size, size - filled)
            if take:
                out[filled:filled + take] = batch[:take]
                filled += take
        return out

    pareto_alpha = {
        "power_law_1_2": 1.2,
        "power_law_1_5": 1.5,
        "power_law_2_0": 2.0,
        "power_law_3_0": 3.0,
        "power_law_5_0": 5.0,
    }
    if dist_name in pareto_alpha:
        return pareto.rvs(
            b=pareto_alpha[dist_name],
            loc=0,
            scale=50,
            size=size,
            random_state=rng,
        )

    if dist_name == LOGNORMAL_DIST:
        return rng.lognormal(
            mean=LOGNORMAL_MU_BACI,
            sigma=LOGNORMAL_SIGMA_BACI,
            size=size,
        )

    raise ValueError(f"Unknown distribution: {dist_name}")


# ============================================================
# Generic summary helpers
# ============================================================

def gini(x: np.ndarray) -> float:
    """Gini coefficient for finite nonnegative values."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    if np.all(x == 0):
        return 0.0
    x = np.sort(x)
    n = x.size
    cumulative = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n)


def bottom_share(x: np.ndarray, frac: float = 0.05) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    total = np.sum(x)
    if total == 0:
        return 0.0
    k = max(1, int(np.ceil(frac * x.size)))
    return float(np.sum(np.sort(x)[:k]) / total)


def safe_quantile(x: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.quantile(x, q)) if x.size else np.nan


def coefficient_of_variation(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    mean = np.mean(x)
    if mean == 0:
        return np.nan
    return float(np.std(x, ddof=0) / mean)


def topology_diagnostics(
    topology: pd.DataFrame,
    sources: np.ndarray,
    targets: np.ndarray,
) -> dict:
    """Realized degree diagnostics, including zero-degree nodes."""
    source_degree = (
        topology.groupby("source").size().reindex(sources, fill_value=0).to_numpy()
    )
    target_degree = (
        topology.groupby("target").size().reindex(targets, fill_value=0).to_numpy()
    )

    return {
        "n_active_sources": int(np.sum(source_degree > 0)),
        "n_zero_degree_sources": int(np.sum(source_degree == 0)),
        "active_source_fraction": float(np.mean(source_degree > 0)),
        "mean_source_degree_realized": float(np.mean(source_degree)),
        "sd_source_degree_realized": float(np.std(source_degree, ddof=0)),
        "max_source_degree_realized": int(np.max(source_degree)),
        "n_active_targets": int(np.sum(target_degree > 0)),
        "n_zero_degree_targets": int(np.sum(target_degree == 0)),
        "active_target_fraction": float(np.mean(target_degree > 0)),
        "mean_target_degree_realized": float(np.mean(target_degree)),
        "sd_target_degree_realized": float(np.std(target_degree, ddof=0)),
        "min_target_degree_realized": int(np.min(target_degree)),
        "max_target_degree_realized": int(np.max(target_degree)),
    }


def weight_concentration_diagnostics(
    edges_slack: pd.DataFrame,
    source_slack: pd.DataFrame,
) -> dict:
    """Reviewer-facing realized concentration measures."""
    weights = edges_slack["weight"].to_numpy(dtype=float)

    target_stats = (
        edges_slack.groupby("target")["target_share_alpha_ij"]
        .agg(
            target_HHI=lambda z: float(np.sum(np.square(z.to_numpy()))),
            target_max_share="max",
            target_degree="size",
        )
    )

    strengths = source_slack["total_weight"].to_numpy(dtype=float)
    total_strength = np.sum(strengths)

    return {
        # Global edge-weight concentration (simple reviewer diagnostic).
        "gini_edge_weight": gini(weights),
        "cv_edge_weight": coefficient_of_variation(weights),

        # Mechanism-aligned target concentration (preferred B4 measures).
        "mean_target_HHI": float(target_stats["target_HHI"].mean()),
        "median_target_HHI": float(target_stats["target_HHI"].median()),
        "mean_target_max_share": float(target_stats["target_max_share"].mean()),
        "median_target_max_share": float(target_stats["target_max_share"].median()),
        "max_target_max_share": float(target_stats["target_max_share"].max()),

        # Existing-measure comparison support (M8 / R4).
        "gini_source_strength": gini(strengths),
        "max_source_strength_share": (
            float(np.max(strengths) / total_strength)
            if total_strength > 0 else np.nan
        ),
    }


# ============================================================
# One scenario / theta simulation
# ============================================================

def run_simulation(
    n_sources: int,
    n_targets: int,
    p_edge: float,
    theta: float,
    replications: int,
    seed: int,
    dist_names: list[str],
) -> pd.DataFrame:
    """
    Run one (M,N,theta) experiment.

    The legacy RNG reproduces the submitted topology and seven submitted weight
    regimes. The new BACI-lognormal draws come from an independent RNG stream.
    """
    rng_legacy = np.random.default_rng(seed)
    rng_lognormal = np.random.default_rng(seed + LOGNORMAL_SEED_OFFSET)

    sources = np.array([f"Src_{i}" for i in range(n_sources)])
    targets = np.array([f"Trg_{j}" for j in range(n_targets)])

    def generate_bipartite_topology() -> pd.DataFrame:
        rows: list[tuple[str, str]] = []
        for source in sources:
            chosen = targets[rng_legacy.random(n_targets) < p_edge]
            rows.extend((source, target) for target in chosen)
        return pd.DataFrame(rows, columns=["source", "target"])

    # Preserve submitted distribution order for exact legacy RNG compatibility.
    ordered_dists = [d for d in LEGACY_DIST_NAMES if d in dist_names]
    if LOGNORMAL_DIST in dist_names:
        ordered_dists.append(LOGNORMAL_DIST)

    graph_summaries: list[dict] = []

    for replication in range(1, replications + 1):
        topology = generate_bipartite_topology()
        n_edges = len(topology)
        if n_edges == 0:
            continue

        topo = topology_diagnostics(topology, sources, targets)

        for dist in ordered_dists:
            rng_for_weights = (
                rng_lognormal if dist == LOGNORMAL_DIST else rng_legacy
            )

            edges = topology.copy()
            edges["weight"] = sample_weights(
                dist, n_edges, rng_for_weights
            ).astype(float)

            edges_slack, source_slack = compute_quota_slack(
                edges,
                theta=theta,
                source_col="source",
                target_col="target",
                weight_col="weight",
            )

            slack_ij = edges_slack["Slack_ij"].to_numpy(dtype=float)
            slack_mean = source_slack["Slack_mean"].to_numpy(dtype=float)
            slack_weighted = source_slack["Slack_weighted"].to_numpy(dtype=float)
            dominance = source_slack["dominance_D"].to_numpy(dtype=float)

            conc = weight_concentration_diagnostics(
                edges_slack=edges_slack,
                source_slack=source_slack,
            )

            identity_error = np.max(
                np.abs(slack_weighted - (1.0 - dominance) / theta)
            )

            graph_summaries.append(
                {
                    "dist": dist,
                    "rep": replication,
                    "n_edges": n_edges,

                    # Explicit denominator diagnostics for source-level metrics.
                    **topo,

                    # Edge-level Slack.
                    "frac_slack_ij_lt_one": float(np.mean(slack_ij < 1)),
                    "min_slack_ij": float(np.min(slack_ij)),
                    "q05_slack_ij": safe_quantile(slack_ij, 0.05),
                    "mean_slack_ij": float(np.mean(slack_ij)),
                    "gini_slack_ij": gini(np.maximum(0, slack_ij)),
                    "bottom5_share_slack_ij": bottom_share(
                        np.maximum(0, slack_ij)
                    ),

                    # Unweighted source-level Slack.
                    "frac_slack_mean_lt_one": float(np.mean(slack_mean < 1)),
                    "min_slack_mean": float(np.min(slack_mean)),
                    "q05_slack_mean": safe_quantile(slack_mean, 0.05),
                    "mean_slack_mean": float(np.mean(slack_mean)),
                    "gini_slack_mean": gini(np.maximum(0, slack_mean)),
                    "bottom5_share_slack_mean": bottom_share(
                        np.maximum(0, slack_mean)
                    ),

                    # Main weighted source-level Slack.
                    # Denominator = active sources only; zero-degree sources have
                    # undefined Slack and are reported separately above.
                    "frac_slack_weighted_lt_one": float(
                        np.mean(slack_weighted < 1)
                    ),
                    "min_slack_weighted": float(np.min(slack_weighted)),
                    "q05_slack_weighted": safe_quantile(slack_weighted, 0.05),
                    "median_slack_weighted": float(np.median(slack_weighted)),
                    "mean_slack_weighted": float(np.mean(slack_weighted)),
                    "gini_slack_weighted": gini(
                        np.maximum(0, slack_weighted)
                    ),
                    "bottom5_share_slack_weighted": bottom_share(
                        np.maximum(0, slack_weighted)
                    ),

                    # Clarifies aggregate criterion vs edge-level pivotality (M2).
                    "frac_active_sources_any_pivotal_edge": float(
                        source_slack["has_any_pivotal_edge"].mean()
                    ),
                    "mean_pivotal_edges_per_active_source": float(
                        source_slack["n_pivotal_edges"].mean()
                    ),

                    # Theta-invariant source dominance representation.
                    "max_source_dominance_D": float(np.max(dominance)),
                    "mean_source_dominance_D": float(np.mean(dominance)),
                    "max_slack_identity_error": float(identity_error),

                    # B4 + M8 realized concentration diagnostics.
                    **conc,
                }
            )

    return pd.DataFrame(graph_summaries)


# ============================================================
# Paired directional comparisons (M1)
# ============================================================

def paired_directional_comparison(
    graphs_df: pd.DataFrame,
    metric: str,
    alternative_dist: str,
    baseline_dist: str = "gaussian",
    direction: str = "lower",
) -> dict:
    """
    Descriptive paired comparison by replication ID.

    This intentionally does NOT label the result a conventional p-value.
    `directional_proportion` is the fraction of paired replications in which the
    alternative is more fragile than the Gaussian benchmark in the prespecified
    direction.
    """
    paired = graphs_df.pivot_table(
        index="rep",
        columns="dist",
        values=metric,
    )

    if baseline_dist not in paired.columns or alternative_dist not in paired.columns:
        raise ValueError(
            f"Missing {baseline_dist!r} or {alternative_dist!r} for {metric!r}."
        )

    paired = paired[[baseline_dist, alternative_dist]].dropna()
    diff = paired[alternative_dist] - paired[baseline_dist]

    if direction == "lower":
        more_fragile = diff < 0
    elif direction == "higher":
        more_fragile = diff > 0
    else:
        raise ValueError("direction must be 'lower' or 'higher'.")

    return {
        "metric": metric,
        "baseline_dist": baseline_dist,
        "alternative_dist": alternative_dist,
        "direction": direction,
        "n_replications": int(len(diff)),
        "baseline_mean": float(paired[baseline_dist].mean()),
        "alternative_mean": float(paired[alternative_dist].mean()),
        "mean_difference": float(diff.mean()),
        "median_difference": float(diff.median()),
        "q025_difference": safe_quantile(diff.to_numpy(), 0.025),
        "q975_difference": safe_quantile(diff.to_numpy(), 0.975),
        "directional_proportion": float(np.mean(more_fragile)),
        # Keep legacy-friendly alias, but no pseudo-p-value is produced.
        "share_alt_more_fragile": float(np.mean(more_fragile)),
        "tie_proportion": float(np.mean(diff == 0)),
    }


def run_directional_comparisons(
    graphs_df: pd.DataFrame,
    baseline_dist: str = "gaussian",
    alternative_dists: list[str] | None = None,
) -> pd.DataFrame:
    if alternative_dists is None:
        alternative_dists = ALTERNATIVE_DISTS

    lower_metrics = [
        "min_slack_ij",
        "q05_slack_ij",
        "min_slack_mean",
        "q05_slack_mean",
        "min_slack_weighted",
        "q05_slack_weighted",
    ]
    higher_metrics = [
        "frac_slack_ij_lt_one",
        "frac_slack_mean_lt_one",
        "frac_slack_weighted_lt_one",
        "frac_active_sources_any_pivotal_edge",
    ]

    rows: list[dict] = []
    for alternative in alternative_dists:
        for metric in lower_metrics:
            rows.append(
                paired_directional_comparison(
                    graphs_df,
                    metric,
                    alternative,
                    baseline_dist,
                    direction="lower",
                )
            )
        for metric in higher_metrics:
            rows.append(
                paired_directional_comparison(
                    graphs_df,
                    metric,
                    alternative,
                    baseline_dist,
                    direction="higher",
                )
            )
    return pd.DataFrame(rows)


# ============================================================
# Figures / compact paper-facing outputs
# ============================================================

def make_directional_frequency_surfaces(
    directional_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Revised Figure-3-style surfaces using directional proportions."""
    metric = "frac_slack_weighted_lt_one"
    alternatives = [
        LOGNORMAL_DIST,
        "power_law_3_0",
        "power_law_2_0",
        "power_law_1_5",
        "power_law_1_2",
    ]
    labels = {
        LOGNORMAL_DIST: "BACI-calibrated lognormal",
        "power_law_3_0": r"Pareto $\alpha=3.0$",
        "power_law_2_0": r"Pareto $\alpha=2.0$",
        "power_law_1_5": r"Pareto $\alpha=1.5$",
        "power_law_1_2": r"Pareto $\alpha=1.2$",
    }

    scenario_order = [f"({m},{n})" for m, n in sorted(SCENARIOS)]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    axes = axes.ravel()

    for ax, alternative in zip(axes, alternatives):
        subset = directional_df[
            (directional_df["metric"] == metric)
            & (directional_df["alternative_dist"] == alternative)
        ]
        matrix = subset.pivot_table(
            index="theta",
            columns="M_N_scenario",
            values="directional_proportion",
        )
        matrix = matrix.reindex(
            index=sorted(matrix.index),
            columns=scenario_order,
        )

        sns.heatmap(
            matrix,
            annot=True,
            fmt=".2f",
            cmap="viridis",
            vmin=0,
            vmax=1,
            linewidths=0.5,
            cbar=ax is axes[len(alternatives) - 1],
            cbar_kws={"label": "Paired directional proportion"},
            ax=ax,
        )
        ax.set_title(labels[alternative])
        ax.set_xlabel("(Sources, targets)")
        ax.set_ylabel(r"$\theta$")

    for ax in axes[len(alternatives):]:
        ax.remove()

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_theta_sensitivity_figures(
    mean_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """B3: balanced (100,100), full theta grid, all regimes."""
    balanced = mean_df[(mean_df["M"] == 100) & (mean_df["N"] == 100)].copy()
    if balanced.empty:
        return

    label_order = [
        "gaussian",
        "student_t",
        LOGNORMAL_DIST,
        "power_law_5_0",
        "power_law_3_0",
        "power_law_2_0",
        "power_law_1_5",
        "power_law_1_2",
    ]

    # Plot 1: mean minimum weighted Slack.
    fig, ax = plt.subplots(figsize=(9, 6))
    for dist in label_order:
        d = balanced[balanced["dist"] == dist].sort_values("theta")
        if d.empty:
            continue
        ax.plot(d["theta"], d["min_slack_weighted"], marker="o", label=dist)
    ax.axhline(1.0, linestyle="--", linewidth=1)
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel("Mean minimum weighted Slack")
    ax.set_title("Theta sensitivity: minimum weighted Slack\n(M,N)=(100,100)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        output_dir / "theta_sensitivity_min_slack_balanced.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Plot 2: mean fraction of active sources with aggregate weighted Slack < 1.
    fig, ax = plt.subplots(figsize=(9, 6))
    for dist in label_order:
        d = balanced[balanced["dist"] == dist].sort_values("theta")
        if d.empty:
            continue
        ax.plot(
            d["theta"],
            100.0 * d["frac_slack_weighted_lt_one"],
            marker="o",
            label=dist,
        )
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel("Mean pivotal active sources (%)")
    ax.set_title("Theta sensitivity: aggregate source pivotality\n(M,N)=(100,100)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        output_dir / "theta_sensitivity_pivotal_fraction_balanced.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_balanced_theta07_summary(
    graphs_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Candidate revised Table-2 input with realized concentration metrics."""
    d = graphs_df[
        (graphs_df["M"] == 100)
        & (graphs_df["N"] == 100)
        & np.isclose(graphs_df["theta"], 0.7)
    ].copy()
    if d.empty:
        return

    summary = (
        d.groupby("dist", as_index=False)
        .agg(
            mean_xmin=("min_slack_weighted", "mean"),
            sd_xmin=("min_slack_weighted", "std"),
            mean_q05_slack=("q05_slack_weighted", "mean"),
            mean_pivotal_fraction=("frac_slack_weighted_lt_one", "mean"),
            mean_gini_slack=("gini_slack_weighted", "mean"),
            mean_edge_weight_gini=("gini_edge_weight", "mean"),
            mean_target_HHI=("mean_target_HHI", "mean"),
            mean_target_max_share=("mean_target_max_share", "mean"),
            mean_any_pivotal_edge_fraction=(
                "frac_active_sources_any_pivotal_edge", "mean"
            ),
            mean_active_sources=("n_active_sources", "mean"),
        )
    )
    summary.to_csv(
        output_dir / "balanced_theta070_summary_REV1.csv",
        index=False,
    )


def save_theta07_across_scenarios_summary(
    graphs_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Compact input for the revised Table-3-style comparison."""
    d = graphs_df[np.isclose(graphs_df["theta"], 0.7)].copy()
    if d.empty:
        return

    summary = (
        d.groupby("dist", as_index=False)
        .agg(
            mean_xmin=("min_slack_weighted", "mean"),
            mean_q05_slack=("q05_slack_weighted", "mean"),
            mean_pivotal_fraction=("frac_slack_weighted_lt_one", "mean"),
            mean_target_HHI=("mean_target_HHI", "mean"),
            mean_target_max_share=("mean_target_max_share", "mean"),
            mean_edge_weight_gini=("gini_edge_weight", "mean"),
        )
    )
    summary.to_csv(
        output_dir / "theta070_across_scenarios_summary_REV1.csv",
        index=False,
    )


# ============================================================
# Full experiment
# ============================================================

def p_rule(n_sources: int) -> float:
    return float(np.clip(TARGET_DEGREE / n_sources, 0.0, 1.0))


def run_full_experiment(
    output_dir: Path,
    replications: int = DEFAULT_REPLICATIONS,
    seed: int = DEFAULT_SEED,
    make_figures: bool = True,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    all_graphs: list[pd.DataFrame] = []
    all_means: list[pd.DataFrame] = []
    all_directional: list[pd.DataFrame] = []

    for n_sources, n_targets in SCENARIOS:
        p_edge = p_rule(n_sources)
        scenario = f"({n_sources},{n_targets})"

        for theta in THETA_VALUES:
            print(
                f"Running M={n_sources}, N={n_targets}, "
                f"theta={theta:.1f}, R={replications}"
            )

            graphs = run_simulation(
                n_sources=n_sources,
                n_targets=n_targets,
                p_edge=p_edge,
                theta=float(theta),
                replications=replications,
                seed=seed,
                dist_names=DIST_NAMES,
            )

            graphs["M"] = n_sources
            graphs["N"] = n_targets
            graphs["theta"] = theta
            graphs["p_edge"] = p_edge
            graphs["target_degree_expected"] = TARGET_DEGREE
            graphs["source_degree_expected"] = n_targets * p_edge
            graphs["M_N_scenario"] = scenario
            all_graphs.append(graphs)

            means = graphs.groupby("dist").mean(numeric_only=True).reset_index()
            means["M"] = n_sources
            means["N"] = n_targets
            means["theta"] = theta
            means["p_edge"] = p_edge
            means["target_degree_expected"] = TARGET_DEGREE
            means["source_degree_expected"] = n_targets * p_edge
            means["M_N_scenario"] = scenario
            all_means.append(means)

            directional = run_directional_comparisons(
                graphs_df=graphs,
                baseline_dist="gaussian",
                alternative_dists=ALTERNATIVE_DISTS,
            )
            directional["M"] = n_sources
            directional["N"] = n_targets
            directional["theta"] = theta
            directional["p_edge"] = p_edge
            directional["target_degree_expected"] = TARGET_DEGREE
            directional["source_degree_expected"] = n_targets * p_edge
            directional["M_N_scenario"] = scenario
            all_directional.append(directional)

    full_graphs = pd.concat(all_graphs, ignore_index=True)
    full_means = pd.concat(all_means, ignore_index=True)
    full_directional = pd.concat(all_directional, ignore_index=True)

    full_graphs.to_csv(
        output_dir / "full_graphs_results_df.csv",
        index=False,
    )
    full_means.to_csv(
        output_dir / "full_mean_results.csv",
        index=False,
    )
    full_directional.to_csv(
        output_dir / "full_directional_results_df.csv",
        index=False,
    )

    # Compatibility filename: content is now directional/descriptive, not p-values.
    full_directional.to_csv(
        output_dir / "full_mc_results_df.csv",
        index=False,
    )

    save_balanced_theta07_summary(full_graphs, output_dir)
    save_theta07_across_scenarios_summary(full_graphs, output_dir)

    metadata = {
        "revision": "REV1",
        "replications": replications,
        "seed_legacy": seed,
        "seed_lognormal": seed + LOGNORMAL_SEED_OFFSET,
        "target_degree_expected": TARGET_DEGREE,
        "theta_values": THETA_VALUES.tolist(),
        "scenarios": SCENARIOS,
        "legacy_distributions": LEGACY_DIST_NAMES,
        "new_distribution": LOGNORMAL_DIST,
        "lognormal_mu_baci": LOGNORMAL_MU_BACI,
        "lognormal_sigma_baci": LOGNORMAL_SIGMA_BACI,
        "lognormal_arithmetic_mean": float(
            np.exp(LOGNORMAL_MU_BACI + 0.5 * LOGNORMAL_SIGMA_BACI**2)
        ),
        "baci_calibration_mean_target_HHI": BACI_MEAN_TARGET_HHI,
        "baci_calibration_mean_target_max_share": BACI_MEAN_TARGET_MAX_SHARE,
        "gaussian": {
            "mu": 100.0,
            "sigma": 20.0,
            "left_truncated_at_zero": True,
            "absolute_value_transform": False,
        },
        "zero_degree_source_rule": (
            "Slack_weighted is computed only for active sources; zero-degree "
            "sources are excluded from the Slack denominator and counted separately."
        ),
        "directional_comparison_rule": (
            "directional_proportion is the share of paired replications in which "
            "the alternative is more fragile than Gaussian; it is not a conventional p-value."
        ),
        "common_random_numbers_across_theta": True,
        "theta_rng_note": (
            "As in the submitted simulation, the same seed is reset for each theta, "
            "so each scenario uses common topology/weight realizations across theta values."
        ),
    }
    with open(output_dir / "simulation_metadata_REV1.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    if make_figures:
        make_directional_frequency_surfaces(
            full_directional,
            output_dir / "figure3_directional_proportion_surfaces.png",
        )
        make_theta_sensitivity_figures(full_means, output_dir)

    print(f"\nFinished REV1. Results written to: {output_dir.resolve()}")


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run REV1 Slack Monte Carlo simulation study."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results_REV1"),
        help="Output directory (default: results_REV1).",
    )
    parser.add_argument(
        "--replications",
        type=int,
        default=DEFAULT_REPLICATIONS,
        help="Replications per scenario/theta (final paper: 1000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Legacy simulation seed (submitted paper: 42).",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Skip PNG generation (useful for fast sanity/regression tests).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_full_experiment(
        output_dir=args.output_dir,
        replications=args.replications,
        seed=args.seed,
        make_figures=not args.skip_figures,
    )
