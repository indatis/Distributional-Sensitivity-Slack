# Distributional Sensitivity of Slack-Based Fragility Measures

Code accompanying the manuscript:

**Distributional Sensitivity of Slack-Based Fragility Measures in Weighted Bipartite Networks**

This repository contains code for the revised Monte Carlo experiments, simulation tables and figures, two robustness checks, and the empirical BACI 2023 analysis.

## Repository contents

- `simulation.py` — main paired Monte Carlo simulation, including the BACI-calibrated lognormal regime and descriptive directional comparisons.
- `paper_results.R` — produces manuscript-facing tables and figures from the Python simulation outputs.
- `topology_robustness.py` — B5 robustness check with heterogeneous target degrees (Zipf exponent `gamma = 3.5`).
- `sensitivity_check.py` — sensitivity check for replication counts and random seeds.
- `BACI_slack_empirical_analysis.ipynb` — Google Colab notebook for the empirical BACI analysis, tail diagnostics, and lognormal calibration.
- `requirements.txt` — Python dependencies.
- `R-packages.txt` — R packages required for `paper_results.R`.

## Simulation design

The default main simulation follows the revised manuscript:

- Nine source–target configurations: `(140,60), (130,70), ..., (60,140)`.
- Quota values: `theta = 0.1, 0.2, ..., 1.0`.
- Expected target degree: 20, with independent edge probability `p = 20/M`, where `M` is the number of sources.
- 1,000 Monte Carlo replications per configuration and quota value.
- Default random seed: 42.
- Gaussian benchmark: `N(100, 20^2)`, left-truncated at zero.
- Student-t: `df = 3`, `loc = 100`, `scale = 20`, retaining positive draws.
- Pareto: scale 50, with tail parameters `alpha = 1.2, 1.5, 2.0, 3.0, 5.0`.
- BACI-calibrated lognormal: `sigma = 1.45`, with `mu = log(100) - sigma^2/2` so that the arithmetic mean weight is 100.

Within each replication, all distributions share the same bipartite topology. The lognormal regime uses an independent random-number stream to avoid changing the legacy distributions' draws.

The simulation reports edge-level and source-level Slack, aggregate pivotality among active sources, concentration measures, and topology diagnostics. Paired **directional proportions** describe how often a distribution is more fragile than the Gaussian benchmark; they are **not conventional p-values**.

## Python setup

The code was developed with Python 3.12.12. Install the dependencies from the repository root:

```bash
pip install -r requirements.txt
```

## Run the main simulation

```bash
python simulation.py
```

By default, this writes results to `results_REV1/`. The full run uses 1,000 replications and can be time-consuming. For a quick test:

```bash
python simulation.py --replications 10 --skip-figures
```

To specify the full settings explicitly:

```bash
python simulation.py --output-dir results_REV1 --replications 1000 --seed 42
```

Key outputs include:

- `results_REV1/full_graphs_results_df.csv` — replication-level metrics.
- `results_REV1/full_mean_results.csv` — averages by distribution, topology, and quota.
- `results_REV1/full_directional_results_df.csv` — paired directional comparisons.
- `results_REV1/balanced_theta070_summary_REV1.csv` — balanced-scenario summary at `theta = 0.7`.
- `results_REV1/theta070_across_scenarios_summary_REV1.csv` — summary across the nine scenarios.
- `results_REV1/simulation_metadata_REV1.json` — simulation settings and calibration parameters.

The script also writes `full_mc_results_df.csv` for backward compatibility. **This file contains the same descriptive directional comparisons, not Monte Carlo p-values.** Figures are generated unless `--skip-figures` is used.

## Produce the paper tables and figures in R

Install the required R packages once:

```r
install.packages(c(
  "readr", "dplyr", "ggplot2", "stringr",
  "knitr", "patchwork", "ggrepel"
))
```

After running the main Python simulation, execute:

```bash
Rscript paper_results.R results_REV1
```

The R script writes files to `results_REV1/paper_outputs_REV1/`, including:

- Balanced-scenario summary and concentration diagnostics at `theta = 0.7`.
- Distributional-sensitivity figures (Figure 2).
- Paired directional-comparison table (Table 3).
- Directional-proportion heatmaps (Figure 3).
- Balanced-scenario quota-sensitivity and other robustness figures.

The generated `README_REV1_outputs.txt` in that directory describes the individual output files.

## B5: heterogeneous-topology robustness

This separate experiment uses `(M,N) = (100,100)` and `theta = 0.7`. It compares the original Erdős–Rényi baseline with a heterogeneous target-degree design constructed from a **rescaled, bounded Zipf degree sequence** (`gamma = 3.5`). It includes Gaussian, Pareto `alpha = 1.5`, and BACI-calibrated lognormal weights.

Run the main simulation first to produce the Erdős–Rényi baseline summary, then run:

```bash
python topology_robustness.py --rev1-results-dir results_REV1
```

The default output directory is `B5_topology_robustness/`. Main outputs include `B5_table_original_request.csv`, `B5_table_with_lognormal.csv`, `B5_topology_diagnostics.csv`, and replication-level results. Without the main simulation summary, the script still computes the heterogeneous-topology results but cannot include the Erdős–Rényi baseline row in its comparison table.

For a quick test:

```bash
python b5_topology_robustness.py --rev1-results-dir results_REV1 --replications 20 --output-dir B5_test
```

## R3-C5: replication and seed sensitivity

The separate stability check considers the balanced `(100,100)` configuration at `theta = 0.7`, with Gaussian, Pareto `alpha = 1.5`, and BACI-calibrated lognormal weights. It compares seeds 42 and 123 at `R = 250, 500, 1000`.

Keep this script beside `simulation.py`, then run:

```bash
python sensitivity_check.py
```

It creates `R3C5_seed_replication_sensitivity.csv` alongside the script. This is a targeted robustness check, **not** an exact rerun of the full eight-distribution random-number sequence used in the main simulation.

## Empirical BACI 2023 analysis

The cleaned notebook `BACI_slack_empirical_analysis.ipynb` covers the empirical component: construction of weighted trade networks by macro-sector, Slack and pivotality, sectoral tail diagnostics, summary tables, quota sensitivity, and calibration of the lognormal weight regime using observed target concentration.

The BACI data are available from **CEPII** but are **not included** in this repository. The notebook is configured for **Google Colab** and expects the user to mount Google Drive, set `DATA_DIR`, and provide the BACI input files. Its current settings use HS17, BACI release `V202601`, and year 2023; the principal trade file is named `BACI_HS17_Y2023_V202601.csv`. Country and product code tables may also be supplied as configured in the notebook.

Run the notebook cells in order. Its outputs are saved under `slack_empirical_outputs/` within `DATA_DIR`. The final lognormal calibration is carried out under the main simulation's expected target-degree topology (20), yielding the `sigma = 1.45` specification used by `simulation.py`.

## Reproducibility notes

For each fixed `(M,N)` configuration, the main simulation resets the same seed for each `theta`, aligning topology and weight realizations across quota values. Zero-degree sources are excluded from source-level Slack fractions and are reported separately. The robustness experiments have their own documented random-number settings.

The main simulation and R post-processing can run from the repository code alone after dependencies are installed. The B5 comparison additionally needs the main simulation's baseline summary; the empirical notebook requires independently obtained BACI input files. Generated result files do not need to be committed to run these workflows.
