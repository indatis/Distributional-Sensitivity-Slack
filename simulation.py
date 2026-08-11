#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monte Carlo simulation for:
"Distributional Sensitivity of Slack-Based Fragility Measures
in Weighted Bipartite Networks"

This script reproduces the simulation design used in the paper:
- 9 source-target configurations
- theta = 0.1, ..., 1.0
- expected target degree = 20, so p = 20 / M
- R = 1000 replications by default
- paired topology across weight distributions within each replication
- Gaussian, Student-t, and Pareto weight-generating regimes
- directional paired Monte Carlo comparisons against the Gaussian benchmark

Outputs are written to ./results by default.

Tested with Python 3.12.12.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pareto, t, truncnorm


# ============================================================
# Configuration used in the paper
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

DIST_NAMES = [
    "gaussian",
    "student_t",
    "power_law_1_2",
    "power_law_1_5",
    "power_law_2_0",
    "power_law_3_0",
    "power_law_5_0",
]

ALTERNATIVE_DISTS = [
    "student_t",
    "power_law_1_2",
    "power_law_1_5",
    "power_law_2_0",
    "power_law_3_0",
    "power_law_5_0",
]

TARGET_DEGREE = 20
DEFAULT_REPLICATIONS = 1000
DEFAULT_SEED = 42


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
    """
    Compute edge-level and source-level quota-based Slack.

    For target j:
        T_j = sum_i w_ij
        Q_j = theta * T_j
        Slack_ij = (T_j - w_ij) / Q_j

    An edge is pivotal when Slack_ij < 1.

    The weighted source-level measure is:
        Slack_weighted(i) = sum_j beta_ij * Slack_ij,
    where beta_ij = w_ij / s_i and s_i is source i's total outgoing weight.
    """
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

    # Aggregate duplicate source-target links if present.
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

    source_slack = (
        df.assign(
            weighted_slack_component=lambda x: x["beta_ij"] * x["Slack_ij"]
        )
        .groupby(source_col, as_index=False)
        .agg(
            Slack_mean=("Slack_ij", "mean"),
            Slack_weighted=("weighted_slack_component", "sum"),
            TargetCount=(target_col, "nunique"),
            total_weight=("s_i", "first"),
            n_pivotal_edges=("is_pivotal", "sum"),
        )
        .rename(columns={source_col: "ID"})
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
    """
    Draw strictly positive edge weights.

    Gaussian:
        Normal(mu=100, sigma=20), truncated at zero.
    Student-t:
        df=3, loc=100, scale=20, retaining positive draws.
    Pareto:
        scale=50 and alpha in {1.2, 1.5, 2.0, 3.0, 5.0}.
    """
    if size == 0:
        return np.array([], dtype=float)

    if dist_name == "gaussian":
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

        # Rejection sampling gives the Student-t specification conditional on w > 0.
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

    raise ValueError(f"Unknown distribution: {dist_name}")


# ============================================================
# Summary statistics
# ============================================================

def gini(x: np.ndarray) -> float:
    """Gini coefficient for nonnegative values."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]

    if x.size == 0:
        return np.nan
    if np.all(x == 0):
        return 0.0

    x = np.sort(x)
    n = x.size
    cumulative = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n)


def bottom_share(x: np.ndarray, frac: float = 0.05) -> float:
    """Share of total mass contributed by the bottom `frac` of observations."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]

    if x.size == 0:
        return np.nan

    total = np.sum(x)
    if total == 0:
        return 0.0

    k = max(1, int(np.ceil(frac * x.size)))
    return float(np.sum(np.sort(x)[:k]) / total)


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
    Run one (M, N, theta) experiment.

    Within each replication the bipartite topology is generated once and reused
    across all weight-generating regimes. This is the paired design used in the
    paper, so distributional comparisons are not confounded by topology changes.
    """
    rng = np.random.default_rng(seed)

    sources = np.array([f"Src_{i}" for i in range(n_sources)])
    targets = np.array([f"Trg_{j}" for j in range(n_targets)])

    def generate_bipartite_topology() -> pd.DataFrame:
        rows: list[tuple[str, str]] = []
        for source in sources:
            chosen = targets[rng.random(n_targets) < p_edge]
            rows.extend((source, target) for target in chosen)
        return pd.DataFrame(rows, columns=["source", "target"])

    graph_summaries: list[dict] = []

    for replication in range(1, replications + 1):
        topology = generate_bipartite_topology()
        n_edges = len(topology)

        # With the configurations used in the paper this safeguard is
        # practically never triggered, but it keeps the function well-defined.
        if n_edges == 0:
            continue

        for dist in dist_names:
            edges = topology.copy()
            edges["weight"] = sample_weights(dist, n_edges, rng).astype(float)

            edges_slack, source_slack = compute_quota_slack(
                edges,
                theta=theta,
                source_col="source",
                target_col="target",
                weight_col="weight",
            )

            slack_ij = edges_slack["Slack_ij"].to_numpy()
            slack_mean = source_slack["Slack_mean"].to_numpy()
            slack_weighted = source_slack["Slack_weighted"].to_numpy()

            graph_summaries.append(
                {
                    "dist": dist,
                    "rep": replication,
                    "n_edges": n_edges,

                    "frac_slack_ij_lt_one": float(np.mean(slack_ij < 1)),
                    "min_slack_ij": float(np.min(slack_ij)),
                    "gini_slack_ij": gini(np.maximum(0, slack_ij)),
                    "bottom5_share_slack_ij": bottom_share(
                        np.maximum(0, slack_ij)
                    ),

                    "frac_slack_mean_lt_one": float(np.mean(slack_mean < 1)),
                    "min_slack_mean": float(np.min(slack_mean)),
                    "gini_slack_mean": gini(np.maximum(0, slack_mean)),
                    "bottom5_share_slack_mean": bottom_share(
                        np.maximum(0, slack_mean)
                    ),

                    "frac_slack_weighted_lt_one": float(
                        np.mean(slack_weighted < 1)
                    ),
                    "min_slack_weighted": float(np.min(slack_weighted)),
                    "gini_slack_weighted": gini(
                        np.maximum(0, slack_weighted)
                    ),
                    "bottom5_share_slack_weighted": bottom_share(
                        np.maximum(0, slack_weighted)
                    ),
                }
            )

    return pd.DataFrame(graph_summaries)


# ============================================================
# Paired directional Monte Carlo comparisons
# ============================================================

def paired_monte_carlo_test(
    graphs_df: pd.DataFrame,
    metric: str,
    alternative_dist: str,
    baseline_dist: str = "gaussian",
    direction: str = "lower",
) -> dict:
    """Paired directional Monte Carlo comparison by replication ID."""
    paired = graphs_df.pivot_table(
        index="rep",
        columns="dist",
        values=metric,
    )

    if (
        baseline_dist not in paired.columns
        or alternative_dist not in paired.columns
    ):
        raise ValueError(
            f"Missing {baseline_dist!r} or {alternative_dist!r} for {metric!r}."
        )

    paired = paired[[baseline_dist, alternative_dist]].dropna()
    diff = paired[alternative_dist] - paired[baseline_dist]

    if direction == "lower":
        share_alt_more_fragile = np.mean(diff < 0)
        mc_pvalue = (1 + np.sum(diff >= 0)) / (len(diff) + 1)
    elif direction == "higher":
        share_alt_more_fragile = np.mean(diff > 0)
        mc_pvalue = (1 + np.sum(diff <= 0)) / (len(diff) + 1)
    else:
        raise ValueError("direction must be 'lower' or 'higher'.")

    return {
        "metric": metric,
        "baseline_dist": baseline_dist,
        "alternative_dist": alternative_dist,
        "direction": direction,
        "n_replications": len(diff),
        "baseline_mean": paired[baseline_dist].mean(),
        "alternative_mean": paired[alternative_dist].mean(),
        "mean_difference": diff.mean(),
        "median_difference": diff.median(),
        "share_alt_more_fragile": share_alt_more_fragile,
        "mc_pvalue": mc_pvalue,
    }


def run_monte_carlo_tests_for_graphs(
    graphs_df: pd.DataFrame,
    baseline_dist: str = "gaussian",
    alternative_dists: list[str] | None = None,
) -> pd.DataFrame:
    """Run paired comparisons for the main Slack fragility statistics."""
    if alternative_dists is None:
        alternative_dists = ALTERNATIVE_DISTS

    lower_metrics = [
        "min_slack_ij",
        "min_slack_mean",
        "min_slack_weighted",
    ]
    higher_metrics = [
        "frac_slack_ij_lt_one",
        "frac_slack_mean_lt_one",
        "frac_slack_weighted_lt_one",
    ]

    rows: list[dict] = []
    for alternative in alternative_dists:
        for metric in lower_metrics:
            rows.append(
                paired_monte_carlo_test(
                    graphs_df,
                    metric,
                    alternative,
                    baseline_dist,
                    direction="lower",
                )
            )
        for metric in higher_metrics:
            rows.append(
                paired_monte_carlo_test(
                    graphs_df,
                    metric,
                    alternative,
                    baseline_dist,
                    direction="higher",
                )
            )

    return pd.DataFrame(rows)


# ============================================================
# Figure 3: paired Monte Carlo p-value surfaces
# ============================================================

def make_figure3_pvalue_surfaces(
    mc_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Create the four-panel p-value surface used for the pivotal-source result."""
    metric = "frac_slack_weighted_lt_one"
    alternatives = [
        "power_law_3_0",
        "power_law_2_0",
        "power_law_1_5",
        "power_law_1_2",
    ]
    labels = {
        "power_law_3_0": r"Pareto $\alpha=3.0$",
        "power_law_2_0": r"Pareto $\alpha=2.0$",
        "power_law_1_5": r"Pareto $\alpha=1.5$",
        "power_law_1_2": r"Pareto $\alpha=1.2$",
    }

    scenario_order = [
        f"({m},{n})"
        for m, n in sorted(SCENARIOS)
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    axes = axes.ravel()

    for ax, alternative in zip(axes, alternatives):
        subset = mc_df[
            (mc_df["metric"] == metric)
            & (mc_df["alternative_dist"] == alternative)
        ]

        matrix = subset.pivot_table(
            index="theta",
            columns="M_N_scenario",
            values="mc_pvalue",
        )
        matrix = matrix.reindex(
            index=sorted(matrix.index),
            columns=scenario_order,
        )

        sns.heatmap(
            matrix,
            annot=True,
            fmt=".3f",
            cmap="viridis_r",
            vmin=0,
            vmax=1,
            linewidths=0.5,
            cbar=ax is axes[-1],
            cbar_kws={"label": "Monte Carlo p-value"},
            ax=ax,
        )
        ax.set_title(labels[alternative])
        ax.set_xlabel("(Sources, targets)")
        ax.set_ylabel(r"$\theta$")

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Full experiment
# ============================================================

def p_rule(n_sources: int) -> float:
    """Edge probability chosen to keep expected target degree at 20."""
    return float(np.clip(TARGET_DEGREE / n_sources, 0.0, 1.0))


def run_full_experiment(
    output_dir: Path,
    replications: int = DEFAULT_REPLICATIONS,
    seed: int = DEFAULT_SEED,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    all_graphs: list[pd.DataFrame] = []
    all_means: list[pd.DataFrame] = []
    all_mc: list[pd.DataFrame] = []

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
            graphs["target_degree"] = TARGET_DEGREE
            graphs["M_N_scenario"] = scenario
            all_graphs.append(graphs)

            means = (
                graphs.groupby("dist")
                .mean(numeric_only=True)
                .reset_index()
            )
            means["M"] = n_sources
            means["N"] = n_targets
            means["theta"] = theta
            means["p_edge"] = p_edge
            means["target_degree"] = TARGET_DEGREE
            means["M_N_scenario"] = scenario
            all_means.append(means)

            mc = run_monte_carlo_tests_for_graphs(
                graphs_df=graphs,
                baseline_dist="gaussian",
                alternative_dists=ALTERNATIVE_DISTS,
            )
            mc["M"] = n_sources
            mc["N"] = n_targets
            mc["theta"] = theta
            mc["p_edge"] = p_edge
            mc["target_degree"] = TARGET_DEGREE
            mc["M_N_scenario"] = scenario
            all_mc.append(mc)

    full_graphs = pd.concat(all_graphs, ignore_index=True)
    full_means = pd.concat(all_means, ignore_index=True)
    full_mc = pd.concat(all_mc, ignore_index=True)

    full_graphs.to_csv(
        output_dir / "full_graphs_results_df.csv",
        index=False,
    )
    full_means.to_csv(
        output_dir / "full_mean_results.csv",
        index=False,
    )
    full_mc.to_csv(
        output_dir / "full_mc_results_df.csv",
        index=False,
    )

    make_figure3_pvalue_surfaces(
        full_mc,
        output_dir / "figure3_mc_pvalue_surfaces_pivotal_fraction.png",
    )

    print(f"\nFinished. Results written to: {output_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the Slack Monte Carlo simulation study."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Output directory (default: results).",
    )
    parser.add_argument(
        "--replications",
        type=int,
        default=DEFAULT_REPLICATIONS,
        help="Monte Carlo replications per scenario/theta (paper: 1000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed (paper: 42).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_full_experiment(
        output_dir=args.output_dir,
        replications=args.replications,
        seed=args.seed,
    )
