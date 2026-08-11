# Distributional Sensitivity of Slack-Based Fragility Measures

Code accompanying the manuscript:

**Distributional Sensitivity of Slack-Based Fragility Measures in Weighted Bipartite Networks**

This repository contains the reproducible code for the **Monte Carlo simulation study** and the associated simulation tables and figures.

## Repository contents

- `simulation.py` — runs the paired Monte Carlo simulation and directional Monte Carlo comparisons.
- `paper_results.R` — reads the Python simulation outputs and produces the main simulation table/figure outputs used in the manuscript.
- `requirements.txt` — Python dependencies.
- `R-packages.txt` — R packages required for `paper_results.R`.


The exploratory notebook used during development is intentionally not required for reproduction of the reported results.

## Simulation design

The default configuration matches the manuscript:

- 9 bipartite source-target configurations:
  `(140,60), (130,70), ..., (60,140)`
- quota values: `theta = 0.1, 0.2, ..., 1.0`
- expected target degree: 20, implemented as `p = 20/M`
- 1000 Monte Carlo replications
- random seed: 42
- Gaussian benchmark: Normal with `mu=100`, `sigma=20`, truncated at zero
- Student-t: `df=3`, `loc=100`, `scale=20`, retaining positive draws
- Pareto: scale 50 and tail parameters `alpha = 1.2, 1.5, 2.0, 3.0, 5.0`

Within each replication, the same bipartite topology is reused across all weight-generating regimes. This is the paired design used to isolate the effect of the weight distribution.

## Python setup

The simulations were developed for Python 3.12.12.

Create and activate a virtual environment if desired, then install:

```bash
pip install -r requirements.txt
```

## Run the simulation

From the repository root:

```bash
python simulation.py
```

By default, outputs are saved in `results/`.

The full paper run uses 1000 replications and can take substantial time. For a quick test:

```bash
python simulation.py --replications 10
```

Optional arguments:

```bash
python simulation.py --output-dir results --replications 1000 --seed 42
```

Main outputs:

- `results/full_graphs_results_df.csv`
- `results/full_mean_results.csv`
- `results/full_mc_results_df.csv`
- `results/figure3_mc_pvalue_surfaces_pivotal_fraction.png`

## Produce the paper simulation outputs in R

Install the required R packages once:

```r
install.packages(c(
  "readr", "dplyr", "ggplot2",
  "stringr", "knitr", "patchwork"
))
```

Then run:

```bash
Rscript paper_results.R
```

or, if the simulation outputs are in another directory:

```bash
Rscript paper_results.R path/to/results
```

The script creates `results/paper_outputs/`, including:

- the balanced-scenario Monte Carlo summary (Table 2 in the manuscript),
- the distributional-sensitivity figure at `theta = 0.7` (Figure 2),
- the paired Monte Carlo summary at `theta = 0.7` (Table 3).

## Empirical BACI analysis

The empirical BACI trade-network analysis reported separately in the manuscript is **not included in this repository version**. The BACI data are publicly available from CEPII.

## Reproducibility note

The default seed is fixed at 42. For a fixed `(M,N)` configuration, the simulation restarts from the same seed for each quota value. Consequently, the underlying random sequences are aligned across quota values while the Slack threshold changes with `theta`.
