#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Topology robustness check for the Slack paper.

Purpose
-------
Targeted robustness exercise requested by the reviewers / Mario:
test whether the main Slack conclusions survive when the balanced
(M,N)=(100,100) network has a heterogeneous target-degree sequence
rather than the homogeneous Erdos-Renyi design.

Main specification
------------------
- M = 100 sources
- N = 100 targets
- theta = 0.7
- R = 1000 replications
- expected degree ≈ 20
- heterogeneous target degrees:
      Zipf / power-law degree sequence with gamma = 3.5
- source degree sequence:
      ER-like Binomial(N, 20/N), so source-degree heterogeneity remains
      comparable to the original Bernoulli design
- simple bipartite degree-constrained stub matching
      (no parallel source-target edges)
- weight regimes:
      Gaussian
      Pareto alpha = 1.5
      Lognormal (empirically calibrated; included because Mario asked
      for a separate "lognormal gamma 3.5 (100/100)" check)

The topology is generated once per replication and reused across the
three weight regimes.

Paper-facing outputs
--------------------
1. B5_table_original_request.csv / .tex
   Erdos-Renyi vs Zipf gamma=3.5, Gaussian and Pareto alpha=1.5 only.
   This matches Mario's original requested B-5 table.

2. B5_table_with_lognormal.csv / .tex
   Same comparison, adding the empirically calibrated Lognormal.

3. B5_topology_diagnostics.csv
   Minimal topology diagnostics showing that gamma=3.5 actually produces
   a more heterogeneous target-degree distribution than Erdos-Renyi.

4. B5_zipf_summary_long.csv
   Long-form summary for checking / response-letter use.

5. B5_replication_results.csv
   Replication-level raw results for reproducibility/audit.


Usage
-----
Typical:
    python b5_topology_robustness.py \
        --rev1-results-dir /path/to/results_REV1

Quick test:
    python b5_topology_robustness.py \
        --rev1-results-dir /path/to/results_REV1 \
        --replications 20 \
        --output-dir B5_test


Dependencies
------------
numpy, pandas, scipy
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pareto, truncnorm


# ============================================================
# 0. Configuration
# ============================================================

M = 100
N = 100
THETA = 0.70

DEFAULT_GAMMA = 3.5
DEFAULT_REPLICATIONS = 1000
DEFAULT_SEED = 42

TARGET_MEAN_DEGREE = 20.0
P_ER = TARGET_MEAN_DEGREE / M

# Same final lognormal calibration used in simulation_REV1.py
LOGNORMAL_SIGMA = 1.45
LOGNORMAL_MU = float(
    np.log(100.0) - 0.5 * LOGNORMAL_SIGMA**2
)

DIST_NAMES = [
    "gaussian",
    "power_law_1_5",
    "lognormal",
]

DIST_LABELS = {
    "gaussian": "Gaussian",
    "power_law_1_5": "Pareto alpha = 1.5",
    "lognormal": "Lognormal",
}

# Independent streams so adding/removing one weight regime does not
# alter the other regimes.
TOPOLOGY_SEED_OFFSET = 0
GAUSSIAN_SEED_OFFSET = 1_000_003
PARETO_SEED_OFFSET = 2_000_003
LOGNORMAL_SEED_OFFSET = 3_000_003


# ============================================================
# 1. Generic helpers
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

    return float(
        (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n
    )


def coefficient_of_variation(x: np.ndarray) -> float:
    """Population coefficient of variation."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if x.size == 0:
        return np.nan

    mean_x = np.mean(x)

    if mean_x == 0:
        return np.nan

    return float(np.std(x, ddof=0) / mean_x)


def adjust_integer_sum(
    values: np.ndarray,
    target_sum: int,
    lower: int,
    upper: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Adjust an integer sequence to an exact total while respecting bounds.

    Adjustments are randomized so there is no deterministic bias toward
    low-index nodes.
    """
    values = np.asarray(values, dtype=int).copy()

    if np.any(values < lower) or np.any(values > upper):
        raise ValueError("Initial sequence violates bounds.")

    diff = int(target_sum - values.sum())

    safety = 0

    while diff != 0:

        if diff > 0:
            candidates = np.flatnonzero(values < upper)

            if candidates.size == 0:
                raise RuntimeError(
                    "Cannot increase degree sequence to requested sum."
                )

            k = min(diff, candidates.size)

            chosen = rng.choice(
                candidates,
                size=k,
                replace=False,
            )

            values[chosen] += 1
            diff -= k

        else:
            candidates = np.flatnonzero(values > lower)

            if candidates.size == 0:
                raise RuntimeError(
                    "Cannot decrease degree sequence to requested sum."
                )

            k = min(-diff, candidates.size)

            chosen = rng.choice(
                candidates,
                size=k,
                replace=False,
            )

            values[chosen] -= 1
            diff += k

        safety += 1

        if safety > 10_000:
            raise RuntimeError(
                "Degree-sequence adjustment did not converge."
            )

    return values


# ============================================================
# 2. Heterogeneous degree-sequence construction
# ============================================================

def draw_source_degrees_er_like(
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Draw an ER-like source degree sequence.

    Under the original balanced Bernoulli topology,
        d_i ~ Binomial(N, p)
    with p = 20/M = 0.2 and M=N=100.

    We preserve this marginal source-side structure while changing
    the target side to a Zipf degree sequence.
    """
    return rng.binomial(
        n=N,
        p=P_ER,
        size=M,
    ).astype(int)


def draw_zipf_target_degrees(
    total_stubs: int,
    gamma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Draw target in-degrees from a Zipf(gamma), truncate to [1,M],
    rescale to the required total number of stubs, and enforce an
    exact stub count.

    The resulting mean target degree is:
        total_stubs / N
    which fluctuates around 20, just as in the original ER design.
    """
    if gamma <= 1:
        raise ValueError("gamma must be > 1.")

    raw = rng.zipf(
        a=gamma,
        size=N,
    ).astype(float)

    raw = np.clip(
        raw,
        1,
        M,
    )

    # Rescale the Zipf sequence to the required total number of stubs.
    scaled = (
        raw
        / raw.sum()
        * total_stubs
    )

    degrees = np.rint(
        scaled
    ).astype(int)

    degrees = np.clip(
        degrees,
        1,
        M,
    )

    degrees = adjust_integer_sum(
        values=degrees,
        target_sum=total_stubs,
        lower=1,
        upper=M,
        rng=rng,
    )

    return degrees


# ============================================================
# 3. Simple bipartite stub matching
# ============================================================

def realize_simple_bipartite_topology(
    source_degrees: np.ndarray,
    target_degrees: np.ndarray,
    rng: np.random.Generator,
    max_attempts: int = 100,
) -> pd.DataFrame:
    """
    Realize a simple bipartite graph with the requested degree sequences.

    Why this version?
    -----------------
    A literal configuration model can produce parallel source-target edges.
    The paper's networks use one edge per source-target pair, so here we use
    a simple degree-constrained stub-matching implementation:

    - source and target sequences contain the same number of stubs;
    - high-degree targets are processed first;
    - each target is connected to distinct sources;
    - source choices are weighted by remaining source stubs;
    - retries are used if a realization becomes infeasible.

    The result preserves both degree sequences exactly while avoiding
    duplicate source-target links.
    """
    source_degrees = np.asarray(
        source_degrees,
        dtype=int,
    )

    target_degrees = np.asarray(
        target_degrees,
        dtype=int,
    )

    if source_degrees.sum() != target_degrees.sum():
        raise ValueError(
            "Source and target degree sequences must have equal stub totals."
        )

    for _ in range(max_attempts):

        remaining_source = source_degrees.copy()

        edges: list[tuple[int, int]] = []

        # Largest targets first improves the probability of a feasible
        # simple realization; random jitter avoids deterministic ordering.
        target_order = np.lexsort(
            (
                rng.random(N),
                -target_degrees,
            )
        )

        success = True

        for target in target_order:

            degree = int(
                target_degrees[target]
            )

            eligible_sources = np.flatnonzero(
                remaining_source > 0
            )

            if eligible_sources.size < degree:
                success = False
                break

            probabilities = (
                remaining_source[
                    eligible_sources
                ].astype(float)
            )

            probabilities /= probabilities.sum()

            chosen_sources = rng.choice(
                eligible_sources,
                size=degree,
                replace=False,
                p=probabilities,
            )

            remaining_source[
                chosen_sources
            ] -= 1

            edges.extend(
                (
                    int(source),
                    int(target),
                )
                for source in chosen_sources
            )

        if (
            success
            and remaining_source.sum() == 0
        ):

            topology = pd.DataFrame(
                edges,
                columns=[
                    "source",
                    "target",
                ],
            )

            # Strong validation: no parallel edges.
            if topology.duplicated(
                subset=[
                    "source",
                    "target",
                ]
            ).any():
                raise RuntimeError(
                    "Unexpected duplicate source-target edge."
                )

            return topology

    raise RuntimeError(
        "Could not realize a simple bipartite topology after "
        f"{max_attempts} attempts."
    )


def generate_zipf_topology(
    gamma: float,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Generate one gamma-heterogeneous topology.

    Source degrees remain ER-like; target degrees follow the rescaled Zipf.
    """
    # Very rarely a source draw could create a pathological total.
    # Retry until the stub total is feasible for N targets with degree >=1.
    for _ in range(100):

        source_degrees = draw_source_degrees_er_like(
            rng
        )

        total_stubs = int(
            source_degrees.sum()
        )

        if not (
            N <= total_stubs <= M * N
        ):
            continue

        target_degrees = draw_zipf_target_degrees(
            total_stubs=total_stubs,
            gamma=gamma,
            rng=rng,
        )

        try:
            topology = realize_simple_bipartite_topology(
                source_degrees=source_degrees,
                target_degrees=target_degrees,
                rng=rng,
            )

            return (
                topology,
                source_degrees,
                target_degrees,
            )

        except RuntimeError:
            continue

    raise RuntimeError(
        "Failed to generate a valid Zipf topology."
    )


# ============================================================
# 4. Weight regimes
# ============================================================

def sample_weights(
    dist_name: str,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Same weight specifications used in simulation_REV1.py."""

    if dist_name == "gaussian":

        mu = 100.0
        sigma = 20.0

        lower = (
            0.0 - mu
        ) / sigma

        return truncnorm.rvs(
            lower,
            np.inf,
            loc=mu,
            scale=sigma,
            size=size,
            random_state=rng,
        )

    if dist_name == "power_law_1_5":

        return pareto.rvs(
            b=1.5,
            loc=0,
            scale=50,
            size=size,
            random_state=rng,
        )

    if dist_name == "lognormal":

        return rng.lognormal(
            mean=LOGNORMAL_MU,
            sigma=LOGNORMAL_SIGMA,
            size=size,
        )

    raise ValueError(
        f"Unknown distribution: {dist_name}"
    )


# ============================================================
# 5. Slack computation
# ============================================================

def compute_weighted_source_slack(
    topology: pd.DataFrame,
    weights: np.ndarray,
    theta: float,
) -> pd.DataFrame:
    """
    Compute the source-level weighted Slack used in the paper.
    """
    edges = topology.copy()

    edges["weight"] = np.asarray(
        weights,
        dtype=float,
    )

    # Target totals
    edges["T_j"] = (
        edges.groupby(
            "target"
        )["weight"]
        .transform("sum")
    )

    edges["alpha_ij"] = (
        edges["weight"]
        / edges["T_j"]
    )

    edges["Slack_ij"] = (
        edges["T_j"]
        - edges["weight"]
    ) / (
        theta
        * edges["T_j"]
    )

    edges["is_pivotal"] = (
        edges["Slack_ij"] < 1
    )

    # Source totals
    edges["s_i"] = (
        edges.groupby(
            "source"
        )["weight"]
        .transform("sum")
    )

    edges["beta_ij"] = (
        edges["weight"]
        / edges["s_i"]
    )

    edges[
        "weighted_slack_component"
    ] = (
        edges["beta_ij"]
        * edges["Slack_ij"]
    )

    source_slack = (
        edges.groupby(
            "source",
            as_index=False,
        )
        .agg(
            Slack_weighted=(
                "weighted_slack_component",
                "sum",
            ),
        )
    )

    return source_slack


# ============================================================
# 6. One replication summary
# ============================================================

def summarize_one_regime(
    topology: pd.DataFrame,
    dist_name: str,
    rng: np.random.Generator,
) -> dict:
    """Three paper-facing B-5 Slack statistics."""

    weights = sample_weights(
        dist_name=dist_name,
        size=len(topology),
        rng=rng,
    )

    source_slack = compute_weighted_source_slack(
        topology=topology,
        weights=weights,
        theta=THETA,
    )

    slack = source_slack[
        "Slack_weighted"
    ].to_numpy()

    return {
        "dist": dist_name,

        "min_slack_weighted":
            float(
                np.min(slack)
            ),

        "frac_pivotal_sources":
            float(
                np.mean(
                    slack < 1
                )
            ),

        "gini_slack_weighted":
            gini(slack),
    }


# ============================================================
# 7. Main Zipf robustness experiment
# ============================================================

def run_zipf_experiment(
    gamma: float,
    replications: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    rng_topology = np.random.default_rng(
        seed
        + TOPOLOGY_SEED_OFFSET
    )

    weight_rngs = {
        "gaussian":
            np.random.default_rng(
                seed
                + GAUSSIAN_SEED_OFFSET
            ),

        "power_law_1_5":
            np.random.default_rng(
                seed
                + PARETO_SEED_OFFSET
            ),

        "lognormal":
            np.random.default_rng(
                seed
                + LOGNORMAL_SEED_OFFSET
            ),
    }

    replication_rows: list[dict] = []
    topology_rows: list[dict] = []

    for rep in range(
        1,
        replications + 1,
    ):

        (
            topology,
            source_degrees,
            target_degrees,
        ) = generate_zipf_topology(
            gamma=gamma,
            rng=rng_topology,
        )

        # Minimal topology diagnostics needed to demonstrate that
        # gamma=3.5 is actually more heterogeneous than ER.
        topology_rows.append(
            {
                "rep": rep,

                "n_edges":
                    len(topology),

                "mean_source_degree":
                    float(
                        np.mean(
                            source_degrees
                        )
                    ),

                "sd_source_degree":
                    float(
                        np.std(
                            source_degrees,
                            ddof=0,
                        )
                    ),

                "cv_source_degree":
                    coefficient_of_variation(
                        source_degrees
                    ),

                "mean_target_degree":
                    float(
                        np.mean(
                            target_degrees
                        )
                    ),

                "sd_target_degree":
                    float(
                        np.std(
                            target_degrees,
                            ddof=0,
                        )
                    ),

                "cv_target_degree":
                    coefficient_of_variation(
                        target_degrees
                    ),

                "gini_target_degree":
                    gini(
                        target_degrees
                    ),

                "max_target_degree":
                    int(
                        np.max(
                            target_degrees
                        )
                    ),

                "min_target_degree":
                    int(
                        np.min(
                            target_degrees
                        )
                    ),
            }
        )

        # Same topology reused across all weight regimes.
        for dist_name in DIST_NAMES:

            row = summarize_one_regime(
                topology=topology,
                dist_name=dist_name,
                rng=weight_rngs[
                    dist_name
                ],
            )

            row.update(
                {
                    "rep": rep,
                    "topology":
                        f"Zipf gamma={gamma:g}",
                    "gamma":
                        gamma,
                    "M":
                        M,
                    "N":
                        N,
                    "theta":
                        THETA,
                    "n_edges":
                        len(topology),
                }
            )

            replication_rows.append(
                row
            )

        if (
            rep == 1
            or rep % 100 == 0
            or rep == replications
        ):
            print(
                f"Completed replication "
                f"{rep}/{replications}"
            )

    return (
        pd.DataFrame(
            replication_rows
        ),
        pd.DataFrame(
            topology_rows
        ),
    )


# ============================================================
# 8. Summarize Zipf results
# ============================================================

def summarize_zipf_results(
    replication_df: pd.DataFrame,
) -> pd.DataFrame:

    summary = (
        replication_df
        .groupby(
            [
                "topology",
                "dist",
            ],
            as_index=False,
        )
        .agg(
            mean_min_slack_weighted=(
                "min_slack_weighted",
                "mean",
            ),

            sd_min_slack_weighted=(
                "min_slack_weighted",
                "std",
            ),

            mean_frac_pivotal_sources=(
                "frac_pivotal_sources",
                "mean",
            ),

            mean_gini_slack_weighted=(
                "gini_slack_weighted",
                "mean",
            ),

            n_replications=(
                "rep",
                "nunique",
            ),
        )
    )

    summary[
        "mean_pivotal_sources_pct"
    ] = (
        100
        * summary[
            "mean_frac_pivotal_sources"
        ]
    )

    summary[
        "Distribution"
    ] = summary[
        "dist"
    ].map(
        DIST_LABELS
    )

    return summary


# ============================================================
# 9. Read original ER baseline from REV1
# ============================================================

def find_balanced_summary(
    rev1_results_dir: Path,
) -> Path | None:

    candidates = [
        rev1_results_dir
        / "balanced_theta070_summary_REV1.csv",

        rev1_results_dir
        / "paper_outputs_REV1"
        / "balanced_theta070_summary_REV1.csv",
    ]

    for candidate in candidates:

        if candidate.exists():
            return candidate

    return None


def read_er_baseline(
    rev1_results_dir: Path,
) -> pd.DataFrame | None:

    summary_path = find_balanced_summary(
        rev1_results_dir
    )

    if summary_path is None:

        print(
            "\nWARNING: Could not find "
            "balanced_theta070_summary_REV1.csv."
        )

        print(
            "Zipf results will be produced, "
            "but the ER row cannot be copied "
            "into the final comparison table."
        )

        return None

    print(
        "\nUsing ER baseline:"
    )

    print(
        summary_path
    )

    df = pd.read_csv(
        summary_path
    )

    needed = {
        "dist",
        "mean_xmin",
        "mean_pivotal_fraction",
        "mean_gini_slack",
    }

    missing = (
        needed
        - set(
            df.columns
        )
    )

    if missing:

        raise ValueError(
            "ER baseline summary is missing columns: "
            f"{sorted(missing)}"
        )

    mapping = {
        "gaussian":
            "gaussian",

        "power_law_1_5":
            "power_law_1_5",

        "lognormal_baci":
            "lognormal",
    }

    rows = []

    for old_name, new_name in mapping.items():

        match = df[
            df["dist"]
            == old_name
        ]

        if match.empty:
            continue

        row = match.iloc[0]

        rows.append(
            {
                "topology":
                    "Erdos-Renyi",

                "dist":
                    new_name,

                "Distribution":
                    DIST_LABELS[
                        new_name
                    ],

                "mean_min_slack_weighted":
                    float(
                        row[
                            "mean_xmin"
                        ]
                    ),

                "mean_pivotal_sources_pct":
                    100
                    * float(
                        row[
                            "mean_pivotal_fraction"
                        ]
                    ),

                "mean_gini_slack_weighted":
                    float(
                        row[
                            "mean_gini_slack"
                        ]
                    ),

                "n_replications":
                    DEFAULT_REPLICATIONS,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 10. Paper table builders
# ============================================================

def make_wide_paper_table(
    combined_long: pd.DataFrame,
    include_lognormal: bool,
) -> pd.DataFrame:

    dists = [
        "gaussian",
        "power_law_1_5",
    ]

    if include_lognormal:
        dists.append(
            "lognormal"
        )

    subset = combined_long[
        combined_long[
            "dist"
        ].isin(
            dists
        )
    ].copy()

    topology_order = [
        "Erdos-Renyi",
        "Zipf gamma=3.5",
    ]

    subset[
        "topology"
    ] = pd.Categorical(
        subset[
            "topology"
        ],
        categories=topology_order,
        ordered=True,
    )

    metrics = {
        "mean_min_slack_weighted":
            "x_min",

        "mean_pivotal_sources_pct":
            "pivotal_pct",

        "mean_gini_slack_weighted":
            "gini_slack",
    }

    pieces = []

    for dist in dists:

        d = subset[
            subset[
                "dist"
            ]
            == dist
        ][
            [
                "topology",
                *metrics.keys(),
            ]
        ].copy()

        prefix = (
            "Gaussian"
            if dist == "gaussian"
            else
            "Pareto1.5"
            if dist == "power_law_1_5"
            else
            "Lognormal"
        )

        d = d.rename(
            columns={
                metric:
                    f"{prefix}_{label}"
                for metric, label
                in metrics.items()
            }
        )

        pieces.append(
            d
        )

    if not pieces:
        return pd.DataFrame()

    out = pieces[0]

    for piece in pieces[1:]:

        out = out.merge(
            piece,
            on="topology",
            how="outer",
        )

    out = out.sort_values(
        "topology"
    ).reset_index(
        drop=True
    )

    out[
        "topology"
    ] = out[
        "topology"
    ].astype(str)

    return out


def dataframe_to_latex(
    df: pd.DataFrame,
    output_path: Path,
    caption: str,
) -> None:
    """Write a simple booktabs-compatible LaTeX table."""

    latex = df.to_latex(
        index=False,
        float_format=lambda x: f"{x:.4f}",
        escape=False,
        caption=caption,
        label=None,
    )

    output_path.write_text(
        latex,
        encoding="utf-8",
    )


# ============================================================
# 11. Topology diagnostics
# ============================================================

def summarize_topology_diagnostics(
    topology_df: pd.DataFrame,
    gamma: float,
) -> pd.DataFrame:

    zipf_row = {
        "topology":
            f"Zipf gamma={gamma:g}",

        "mean_source_degree":
            topology_df[
                "mean_source_degree"
            ].mean(),

        "sd_source_degree":
            topology_df[
                "sd_source_degree"
            ].mean(),

        "cv_source_degree":
            topology_df[
                "cv_source_degree"
            ].mean(),

        "mean_target_degree":
            topology_df[
                "mean_target_degree"
            ].mean(),

        "sd_target_degree":
            topology_df[
                "sd_target_degree"
            ].mean(),

        "cv_target_degree":
            topology_df[
                "cv_target_degree"
            ].mean(),

        "gini_target_degree":
            topology_df[
                "gini_target_degree"
            ].mean(),

        "mean_max_target_degree":
            topology_df[
                "max_target_degree"
            ].mean(),

        "mean_min_target_degree":
            topology_df[
                "min_target_degree"
            ].mean(),
    }

    # Theoretical ER comparator for the balanced Bernoulli design.
    er_degree_sd = math.sqrt(
        M
        * P_ER
        * (
            1
            - P_ER
        )
    )

    er_row = {
        "topology":
            "Erdos-Renyi (theoretical)",

        "mean_source_degree":
            TARGET_MEAN_DEGREE,

        "sd_source_degree":
            er_degree_sd,

        "cv_source_degree":
            er_degree_sd
            / TARGET_MEAN_DEGREE,

        "mean_target_degree":
            TARGET_MEAN_DEGREE,

        "sd_target_degree":
            er_degree_sd,

        "cv_target_degree":
            er_degree_sd
            / TARGET_MEAN_DEGREE,

        "gini_target_degree":
            np.nan,

        "mean_max_target_degree":
            np.nan,

        "mean_min_target_degree":
            np.nan,
    }

    return pd.DataFrame(
        [
            er_row,
            zipf_row,
        ]
    )


# ============================================================
# 12. CLI
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "B-5 topology robustness check "
            "for the Slack paper."
        )
    )

    parser.add_argument(
        "--rev1-results-dir",
        type=Path,
        default=Path("results_REV1"),
        help=(
            "Directory containing "
            "balanced_theta070_summary_REV1.csv "
            "from the main REV1 simulation."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "B5_topology_robustness"
        ),
    )

    parser.add_argument(
        "--replications",
        type=int,
        default=DEFAULT_REPLICATIONS,
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=DEFAULT_GAMMA,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    return parser.parse_args()


# ============================================================
# 13. Main
# ============================================================

def main() -> None:

    args = parse_args()

    if args.replications <= 0:
        raise ValueError(
            "replications must be positive."
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\n========================================"
    )

    print(
        "B-5 TOPOLOGY ROBUSTNESS CHECK"
    )

    print(
        "========================================"
    )

    print(
        f"M=N={M}, theta={THETA}, "
        f"gamma={args.gamma}, "
        f"R={args.replications}"
    )

    print(
        "Weight regimes: Gaussian, "
        "Pareto alpha=1.5, Lognormal"
    )

    # --------------------------------------------------------
    # Run Zipf topology experiment
    # --------------------------------------------------------

    (
        replication_df,
        topology_df,
    ) = run_zipf_experiment(
        gamma=args.gamma,
        replications=args.replications,
        seed=args.seed,
    )

    zipf_summary = summarize_zipf_results(
        replication_df
    )

    # --------------------------------------------------------
    # Save raw / long summaries
    # --------------------------------------------------------

    replication_file = (
        args.output_dir
        / "B5_replication_results.csv"
    )

    replication_df.to_csv(
        replication_file,
        index=False,
    )

    long_file = (
        args.output_dir
        / "B5_zipf_summary_long.csv"
    )

    zipf_summary.to_csv(
        long_file,
        index=False,
    )

    # --------------------------------------------------------
    # ER baseline
    # --------------------------------------------------------

    er_baseline = read_er_baseline(
        args.rev1_results_dir
    )

    if er_baseline is not None:

        combined_long = pd.concat(
            [
                er_baseline,
                zipf_summary[
                    [
                        "topology",
                        "dist",
                        "Distribution",
                        "mean_min_slack_weighted",
                        "mean_pivotal_sources_pct",
                        "mean_gini_slack_weighted",
                        "n_replications",
                    ]
                ],
            ],
            ignore_index=True,
        )

    else:

        combined_long = zipf_summary[
            [
                "topology",
                "dist",
                "Distribution",
                "mean_min_slack_weighted",
                "mean_pivotal_sources_pct",
                "mean_gini_slack_weighted",
                "n_replications",
            ]
        ].copy()

    combined_long_file = (
        args.output_dir
        / "B5_comparison_long.csv"
    )

    combined_long.to_csv(
        combined_long_file,
        index=False,
    )

    # --------------------------------------------------------
    # Paper-facing table matching Mario's ORIGINAL request:
    # Gaussian + Pareto 1.5
    # --------------------------------------------------------

    original_table = make_wide_paper_table(
        combined_long=combined_long,
        include_lognormal=False,
    )

    original_csv = (
        args.output_dir
        / "B5_table_original_request.csv"
    )

    original_table.to_csv(
        original_csv,
        index=False,
    )

    original_tex = (
        args.output_dir
        / "B5_table_original_request.tex"
    )

    dataframe_to_latex(
        df=original_table,
        output_path=original_tex,
        caption=(
            "Topology robustness at "
            "$(M,N)=(100,100)$ and "
            "$\\theta=0.7$: "
            "Erdos--Renyi versus "
            f"Zipf $\\gamma={args.gamma:g}$."
        ),
    )

    # --------------------------------------------------------
    # Extended table including Lognormal, because Mario later
    # explicitly asked for "lognormal gamma 3.5 (100/100)".
    # --------------------------------------------------------

    extended_table = make_wide_paper_table(
        combined_long=combined_long,
        include_lognormal=True,
    )

    extended_csv = (
        args.output_dir
        / "B5_table_with_lognormal.csv"
    )

    extended_table.to_csv(
        extended_csv,
        index=False,
    )

    extended_tex = (
        args.output_dir
        / "B5_table_with_lognormal.tex"
    )

    dataframe_to_latex(
        df=extended_table,
        output_path=extended_tex,
        caption=(
            "Topology robustness at "
            "$(M,N)=(100,100)$ and "
            "$\\theta=0.7$, including "
            "the empirically calibrated "
            "Lognormal regime."
        ),
    )

    # --------------------------------------------------------
    # Minimal topology diagnostics for reviewer response
    # --------------------------------------------------------

    topology_summary = summarize_topology_diagnostics(
        topology_df=topology_df,
        gamma=args.gamma,
    )

    topology_summary_file = (
        args.output_dir
        / "B5_topology_diagnostics.csv"
    )

    topology_summary.to_csv(
        topology_summary_file,
        index=False,
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {
        "M": M,
        "N": N,
        "theta": THETA,
        "gamma": args.gamma,
        "replications": args.replications,
        "seed": args.seed,
        "expected_degree_original_ER":
            TARGET_MEAN_DEGREE,
        "source_degree_model":
            "Binomial(N, 20/N)",
        "target_degree_model":
            (
                "Zipf(gamma), truncated to [1,M], "
                "rescaled to match source stub total"
            ),
        "topology_realization":
            (
                "simple degree-constrained stub matching; "
                "parallel source-target edges forbidden"
            ),
        "weight_regimes": {
            "Gaussian":
                "left-truncated N(100,20^2)",
            "Pareto alpha=1.5":
                "scale=50, alpha=1.5",
            "Lognormal": {
                "mu": LOGNORMAL_MU,
                "sigma": LOGNORMAL_SIGMA,
            },
        },
    }

    metadata_file = (
        args.output_dir
        / "B5_metadata.json"
    )

    metadata_file.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Console results
    # --------------------------------------------------------

    print(
        "\n========================================"
    )

    print(
        "ZIPF gamma ROBUSTNESS SUMMARY"
    )

    print(
        "========================================"
    )

    print(
        zipf_summary[
            [
                "Distribution",
                "mean_min_slack_weighted",
                "mean_pivotal_sources_pct",
                "mean_gini_slack_weighted",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\n========================================"
    )

    print(
        "TOPOLOGY DIAGNOSTICS"
    )

    print(
        "========================================"
    )

    print(
        topology_summary.to_string(
            index=False
        )
    )

    print(
        "\n========================================"
    )

    print(
        "PAPER TABLE WITH LOGNORMAL"
    )

    print(
        "========================================"
    )

    print(
        extended_table.to_string(
            index=False
        )
    )

    print(
        "\nSaved outputs to:"
    )

    print(
        args.output_dir.resolve()
    )

    print(
        "\nMain files to send Mario:"
    )

    print(
        "  B5_table_original_request.csv/.tex"
    )

    print(
        "  B5_table_with_lognormal.csv/.tex"
    )

    print(
        "  B5_topology_diagnostics.csv"
    )


if __name__ == "__main__":
    main()
