# ============================================================
# Revised paper outputs for the Slack Monte Carlo simulation
# ============================================================
# ============================================================

rm(list = ls())

# ============================================================
# 0. Packages
# ============================================================

required_packages <- c(
  "readr",
  "dplyr",
  "ggplot2",
  "stringr",
  "knitr",
  "patchwork",
  "ggrepel"
)

missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]

if (length(missing_packages) > 0) {
  stop(
    "Missing R packages: ",
    paste(missing_packages, collapse = ", "),
    "\nInstall them with:\n",
    "install.packages(c(",
    paste(sprintf('"%s"', missing_packages), collapse = ", "),
    "))"
  )
}

library(readr)
library(dplyr)
library(ggplot2)
library(stringr)
library(knitr)
library(patchwork)
library(ggrepel)


# ============================================================
# 1. Locate results directory
# ============================================================

args <- commandArgs(trailingOnly = TRUE)

if (length(args) >= 1) {
  
  base_dir <- args[[1]]
  
} else if (
  file.exists("full_mean_results.csv") &&
  file.exists("balanced_theta070_summary_REV1.csv")
) {
  
  base_dir <- "."
  
} else if (dir.exists("results_REV1")) {
  
  base_dir <- "results_REV1"
  
} else {
  
  base_dir <- "results"
}

out_dir <- file.path(base_dir, "paper_outputs_REV1")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

cat("\nUsing results directory:\n", normalizePath(base_dir, mustWork = FALSE), "\n")
cat("Writing paper outputs to:\n", normalizePath(out_dir, mustWork = FALSE), "\n\n")


# ============================================================
# 2. Input files
# ============================================================

means_path <- file.path(
  base_dir,
  "full_mean_results.csv"
)

directional_path <- file.path(
  base_dir,
  "full_directional_results_df.csv"
)

# Compatibility fallback: REV1 writes the same directional
# content to full_mc_results_df.csv, but it is NOT a p-value file.
if (!file.exists(directional_path)) {
  directional_path <- file.path(
    base_dir,
    "full_mc_results_df.csv"
  )
}

balanced_path <- file.path(
  base_dir,
  "balanced_theta070_summary_REV1.csv"
)

theta07_summary_path <- file.path(
  base_dir,
  "theta070_across_scenarios_summary_REV1.csv"
)

required_files <- c(
  means_path,
  directional_path,
  balanced_path
)

missing_files <- required_files[
  !file.exists(required_files)
]

if (length(missing_files) > 0) {
  stop(
    "Required REV1 simulation output files not found:\n",
    paste(missing_files, collapse = "\n"),
    "\n\nRun simulation_REV1.py first, or pass the correct results directory."
  )
}


# ============================================================
# 3. Read data
# ============================================================

means <- read_csv(
  means_path,
  show_col_types = FALSE
)

directional <- read_csv(
  directional_path,
  show_col_types = FALSE
)

balanced <- read_csv(
  balanced_path,
  show_col_types = FALSE
)

if (file.exists(theta07_summary_path)) {
  
  theta07_summary <- read_csv(
    theta07_summary_path,
    show_col_types = FALSE
  )
  
} else {
  
  theta07_summary <- means %>%
    filter(abs(theta - 0.7) < 1e-10) %>%
    group_by(dist) %>%
    summarise(
      mean_xmin =
        mean(min_slack_weighted, na.rm = TRUE),
      
      mean_q05_slack =
        mean(q05_slack_weighted, na.rm = TRUE),
      
      mean_pivotal_fraction =
        mean(frac_slack_weighted_lt_one, na.rm = TRUE),
      
      mean_target_HHI =
        mean(mean_target_HHI, na.rm = TRUE),
      
      mean_target_max_share =
        mean(mean_target_max_share, na.rm = TRUE),
      
      mean_edge_weight_gini =
        mean(gini_edge_weight, na.rm = TRUE),
      
      .groups = "drop"
    )
}


# ============================================================
# 4. Distribution labels and ordering
# ============================================================

dist_order <- c(
  "gaussian",
  "student_t",
  "power_law_5_0",
  "power_law_3_0",
  "power_law_2_0",
  "lognormal_baci",
  "power_law_1_5",
  "power_law_1_2"
)

dist_labels <- c(
  "gaussian"       = "Gaussian",
  "student_t"      = "Student-t",
  "lognormal_baci" = "Lognormal",
  "power_law_5_0"  = "Pareto alpha = 5.0",
  "power_law_3_0"  = "Pareto alpha = 3.0",
  "power_law_2_0"  = "Pareto alpha = 2.0",
  "power_law_1_5"  = "Pareto alpha = 1.5",
  "power_law_1_2"  = "Pareto alpha = 1.2"
)

axis_labels <- c(
  "gaussian"       = "Gaussian",
  "student_t"      = "Student-t",
  "power_law_5_0"  = "Pareto
alpha=5",
  "power_law_3_0"  = "Pareto
alpha=3",
  "power_law_2_0"  = "Pareto
alpha=2",
  "lognormal_baci" = "Lognormal",
  "power_law_1_5"  = "Pareto
alpha=1.5",
  "power_law_1_2"  = "Pareto
alpha=1.2"
)


scatter_labels <- c(
  "gaussian"       = "Gaussian",
  "student_t"      = "Student-t",
  "power_law_5_0"  = "Pareto alpha=5",
  "power_law_3_0"  = "Pareto alpha=3",
  "power_law_2_0"  = "Pareto alpha=2",
  "lognormal_baci" = "Lognormal",
  "power_law_1_5"  = "Pareto alpha=1.5",
  "power_law_1_2"  = "Pareto alpha=1.2"
)

regime_shapes <- c(
  "gaussian"       = 16,
  "student_t"      = 17,
  "power_law_5_0"  = 15,
  "power_law_3_0"  = 18,
  "power_law_2_0"  = 3,
  "lognormal_baci" = 8,
  "power_law_1_5"  = 1,
  "power_law_1_2"  = 2
)

means <- means %>%
  mutate(
    dist = factor(dist, levels = dist_order),
    M_N_scenario = str_replace_all(
      M_N_scenario,
      " ",
      ""
    )
  )

directional <- directional %>%
  mutate(
    alternative_dist = factor(
      alternative_dist,
      levels = dist_order
    ),
    M_N_scenario = str_replace_all(
      M_N_scenario,
      " ",
      ""
    )
  )

balanced <- balanced %>%
  mutate(
    dist = factor(dist, levels = dist_order),
    dist_label = unname(
      dist_labels[as.character(dist)]
    )
  )

theta07_summary <- theta07_summary %>%
  mutate(
    dist = factor(dist, levels = dist_order),
    dist_label = unname(
      dist_labels[as.character(dist)]
    )
  )


# ============================================================
# 5. Helper functions
# ============================================================

write_latex_table <- function(
    data,
    filename,
    caption,
    digits = 4
) {
  
  sink(
    file.path(
      out_dir,
      filename
    )
  )
  
  print(
    kable(
      data,
      format = "latex",
      booktabs = TRUE,
      digits = digits,
      caption = caption
    )
  )
  
  sink()
}


# ============================================================
# 6. TABLE 2 REV1
# Balanced scenario (100,100), theta=0.7
# ============================================================
# ============================================================

table2_extended <- balanced %>%
  arrange(dist) %>%
  transmute(
    Distribution =
      dist_label,
    
    `Mean x_min` =
      mean_xmin,
    
    `SD x_min` =
      sd_xmin,
    
    `Mean pivotal sources (%)` =
      100 * mean_pivotal_fraction,
    
    `Mean Gini(Slack^W)` =
      mean_gini_slack,
    
    `Mean target HHI` =
      mean_target_HHI,
    
    `Mean target max share` =
      mean_target_max_share
  )

write_csv(
  table2_extended,
  file.path(
    out_dir,
    "table2_balanced_theta07_REV1.csv"
  )
)

write_latex_table(
  table2_extended,
  "table2_balanced_theta07_REV1.tex",
  paste0(
    "Monte Carlo summary for the balanced scenario ",
    "$(M,N)=(100,100)$ and $\\theta=0.7$, including ",
    "realized target-level weight concentration."
  ),
  digits = 4
)


table2_compact <- balanced %>%
  arrange(dist) %>%
  transmute(
    Distribution =
      dist_label,
    
    `Mean x_min` =
      mean_xmin,
    
    `SD x_min` =
      sd_xmin,
    
    `Mean pivotal sources (%)` =
      100 * mean_pivotal_fraction,
    
    `Mean Gini(Slack^W)` =
      mean_gini_slack
  )

write_csv(
  table2_compact,
  file.path(
    out_dir,
    "table2_balanced_theta07_REV1_compact.csv"
  )
)

write_latex_table(
  table2_compact,
  "table2_balanced_theta07_REV1_compact.tex",
  paste0(
    "Monte Carlo summary for the balanced scenario ",
    "$(M,N)=(100,100)$ and $\\theta=0.7$."
  ),
  digits = 4
)


# ============================================================
# 7. B-4 CONCENTRATION / FRAGILITY TABLE
# ============================================================
# ============================================================

concentration_table <- balanced %>%
  arrange(dist) %>%
  transmute(
    Distribution =
      dist_label,
    
    `Edge-weight Gini` =
      mean_edge_weight_gini,
    
    `Mean target HHI` =
      mean_target_HHI,
    
    `Mean target max share` =
      mean_target_max_share,
    
    `Mean x_min` =
      mean_xmin,
    
    `Mean q05 Slack^W` =
      mean_q05_slack,
    
    `Pivotal sources (%)` =
      100 * mean_pivotal_fraction
  )

write_csv(
  concentration_table,
  file.path(
    out_dir,
    "table_concentration_fragility_balanced_theta07_REV1.csv"
  )
)

write_latex_table(
  concentration_table,
  "table_concentration_fragility_balanced_theta07_REV1.tex",
  paste0(
    "Realized weight concentration and Slack-based fragility ",
    "for $(M,N)=(100,100)$ and $\\theta=0.7$."
  ),
  digits = 4
)


# ============================================================
# 8. OPTIONAL q05 ROBUSTNESS TABLE
# ============================================================
#
# The minimum is an extreme order statistic.
# q05 checks whether conclusions extend beyond one extreme source.
# ============================================================

q05_table <- balanced %>%
  arrange(dist) %>%
  transmute(
    Distribution =
      dist_label,
    
    `Mean minimum Slack^W` =
      mean_xmin,
    
    `Mean 5th percentile Slack^W` =
      mean_q05_slack,
    
    `Pivotal sources (%)` =
      100 * mean_pivotal_fraction
  )

write_csv(
  q05_table,
  file.path(
    out_dir,
    "table_q05_robustness_balanced_theta07_REV1.csv"
  )
)

write_latex_table(
  q05_table,
  "table_q05_robustness_balanced_theta07_REV1.tex",
  paste0(
    "Minimum and lower-tail weighted Slack for ",
    "$(M,N)=(100,100)$ and $\\theta=0.7$."
  ),
  digits = 4
)


# ============================================================
# 9. OPTIONAL AGGREGATION DIAGNOSTIC
# ============================================================
# ============================================================

aggregation_table <- balanced %>%
  arrange(dist) %>%
  transmute(
    Distribution =
      dist_label,
    
    `Aggregate-pivotal sources (%)` =
      100 * mean_pivotal_fraction,
    
    `Sources with >=1 pivotal edge (%)` =
      100 * mean_any_pivotal_edge_fraction
  )

write_csv(
  aggregation_table,
  file.path(
    out_dir,
    "table_aggregation_diagnostic_balanced_theta07_REV1.csv"
  )
)

write_latex_table(
  aggregation_table,
  "table_aggregation_diagnostic_balanced_theta07_REV1.tex",
  paste0(
    "Aggregate source-level pivotality versus the presence ",
    "of at least one pivotal edge for $(M,N)=(100,100)$ ",
    "and $\\theta=0.7$."
  ),
  digits = 4
)


# ============================================================
# ============================================================
# 10. FIGURE 2 REV1
# Distributional sensitivity at theta=0.7
# ============================================================
#
# Metric-specific display ordering:
#
# Minimum weighted Slack:
#   Gaussian -> Student-t -> Pareto 5 -> Pareto 3 -> Pareto 2
#   -> Lognormal -> Pareto 1.5 -> Pareto 1.2
#
# Pivotal-source fraction:
#   Gaussian -> Student-t -> Pareto 5 -> Pareto 3 -> Pareto 2
#   -> Pareto 1.5 -> Lognormal -> Pareto 1.2
#
# The second ordering reflects the observed pivotality progression
# at theta=0.7. It is a display choice, not a claim that Lognormal
# belongs to the Pareto tail-index sequence.
#
# The script saves:
#   1. the combined two-panel figure;
#   2. the minimum-Slack panel separately;
#   3. the pivotal-fraction panel separately.
# ============================================================

m_vals <- seq(
  140,
  60,
  by = -10
)

n_vals <- seq(
  60,
  140,
  by = 10
)

all_scenarios <- paste0(
  "(",
  m_vals,
  ",",
  n_vals,
  ")"
)

theta_focus <- 0.7


# ------------------------------------------------------------
# Panel-specific display orders
# ------------------------------------------------------------

dist_order_min <- c(
  "gaussian",
  "student_t",
  "power_law_5_0",
  "power_law_3_0",
  "power_law_2_0",
  "lognormal_baci",
  "power_law_1_5",
  "power_law_1_2"
)

dist_order_piv <- c(
  "gaussian",
  "student_t",
  "power_law_5_0",
  "power_law_3_0",
  "power_law_2_0",
  "power_law_1_5",
  "lognormal_baci",
  "power_law_1_2"
)


plot_df <- means %>%
  filter(
    abs(theta - theta_focus) < 1e-10,
    as.character(dist) %in% dist_order,
    M_N_scenario %in% all_scenarios
  ) %>%
  mutate(
    dist_chr = as.character(dist),
    M_N_scenario = factor(
      M_N_scenario,
      levels = all_scenarios
    )
  )


create_transition_subplot <- function(
    metric,
    ylabel_text,
    order_levels,
    ylim = NULL,
    add_threshold = TRUE,
    title_text = paste0("theta = ", theta_focus),
    legend_position = "right",
    base_size = 11
) {
  
  local_df <- plot_df %>%
    mutate(
      dist_plot = factor(
        dist_chr,
        levels = order_levels
      )
    )
  
  p <- ggplot(
    local_df,
    aes(
      x = dist_plot,
      y = .data[[metric]],
      color = M_N_scenario,
      group = M_N_scenario
    )
  ) +
    geom_line(
      linewidth = 0.8
    ) +
    geom_point(
      size = 1.9
    ) +
    scale_x_discrete(
      labels = axis_labels
    ) +
    scale_color_viridis_d(
      option = "plasma",
      end = 0.9
    ) +
    labs(
      title = title_text,
      x = "Distributional regime",
      y = ylabel_text,
      color = "(M, N)"
    ) +
    theme_minimal(
      base_size = base_size
    ) +
    theme(
      panel.border = element_rect(
        color = "black",
        fill = NA,
        linewidth = 0.5
      ),
      legend.position = legend_position,
      axis.text.x = element_text(
        angle = 45,
        hjust = 1
      )
    )
  
  if (add_threshold) {
    p <- p +
      geom_hline(
        yintercept = 1,
        linetype = "dashed",
        color = "gray50"
      )
  }
  
  if (!is.null(ylim)) {
    p <- p +
      coord_cartesian(
        ylim = ylim
      )
  }
  
  p
}


# ------------------------------------------------------------
# Combined Figure 2
# ------------------------------------------------------------

panel_a <- create_transition_subplot(
  metric = "min_slack_weighted",
  ylabel_text = "Mean minimum weighted Slack",
  order_levels = dist_order_min,
  add_threshold = TRUE
)

panel_b <- create_transition_subplot(
  metric = "frac_slack_weighted_lt_one",
  ylabel_text = "Mean fraction of pivotal active sources",
  order_levels = dist_order_piv,
  ylim = c(0, 0.20),
  add_threshold = FALSE
)

figure2 <- (
  panel_a +
    panel_b +
    plot_layout(
      widths = c(1, 1)
    )
)

ggsave(
  filename = file.path(
    out_dir,
    "figure2_distributional_sensitivity_theta07_REV1.png"
  ),
  plot = figure2,
  width = 14,
  height = 5.5,
  dpi = 300,
  bg = "white"
)


# ------------------------------------------------------------
# Separate Figure 2A: minimum weighted Slack
# ------------------------------------------------------------

figure2_min_only <- create_transition_subplot(
  metric = "min_slack_weighted",
  ylabel_text = "Mean minimum weighted Slack",
  order_levels = dist_order_min,
  add_threshold = TRUE,
  title_text = paste0(
    "Distributional sensitivity of minimum weighted Slack, theta = ",
    theta_focus
  ),
  legend_position = "right",
  base_size = 12
)

ggsave(
  filename = file.path(
    out_dir,
    "figure2A_minimum_weighted_slack_theta07_REV1.png"
  ),
  plot = figure2_min_only,
  width = 9.5,
  height = 6.2,
  dpi = 300,
  bg = "white"
)


# ------------------------------------------------------------
# Separate Figure 2B: pivotal-source fraction
# ------------------------------------------------------------
# ------------------------------------------------------------

figure2_piv_only <- create_transition_subplot(
  metric = "frac_slack_weighted_lt_one",
  ylabel_text = "Mean fraction of pivotal active sources",
  order_levels = dist_order_piv,
  ylim = c(0, 0.20),
  add_threshold = FALSE,
  title_text = paste0(
    "Distributional sensitivity of aggregate source pivotality, theta = ",
    theta_focus
  ),
  legend_position = "right",
  base_size = 12
)

ggsave(
  filename = file.path(
    out_dir,
    "figure2B_pivotal_fraction_theta07_REV1.png"
  ),
  plot = figure2_piv_only,
  width = 9.5,
  height = 6.2,
  dpi = 300,
  bg = "white"
)


# 11. TABLE 3 REV1
# Paired directional comparison at theta=0.7
# ============================================================
#
# IMPORTANT:
# No p-values.
#
# directional_proportion =
# share of paired replications in which the alternative is
# more fragile than the Gaussian benchmark.
#
# For x_min:
#   more fragile = lower
#
# For pivotal fraction:
#   more fragile = higher
# ============================================================

table3_min <- directional %>%
  filter(
    abs(theta - 0.7) < 1e-10,
    metric == "min_slack_weighted"
  ) %>%
  group_by(
    alternative_dist
  ) %>%
  summarise(
    min_mean_difference =
      mean(
        mean_difference,
        na.rm = TRUE
      ),
    
    min_directional_proportion =
      mean(
        directional_proportion,
        na.rm = TRUE
      ),
    
    .groups = "drop"
  )


table3_piv <- directional %>%
  filter(
    abs(theta - 0.7) < 1e-10,
    metric == "frac_slack_weighted_lt_one"
  ) %>%
  group_by(
    alternative_dist
  ) %>%
  summarise(
    pivotal_mean_pct =
      100 * mean(
        alternative_mean,
        na.rm = TRUE
      ),
    
    pivotal_directional_proportion =
      mean(
        directional_proportion,
        na.rm = TRUE
      ),
    
    .groups = "drop"
  )


table3 <- left_join(
  table3_min,
  table3_piv,
  by = "alternative_dist"
) %>%
  mutate(
    alternative_dist_chr =
      as.character(
        alternative_dist
      ),
    
    Distribution =
      unname(
        dist_labels[
          alternative_dist_chr
        ]
      ),
    
    order =
      match(
        alternative_dist_chr,
        dist_order
      )
  ) %>%
  arrange(
    order
  ) %>%
  transmute(
    Distribution,
    
    `Mean difference in x_min` =
      min_mean_difference,
    
    `Directional proportion: lower x_min` =
      min_directional_proportion,
    
    `Mean pivotal sources (%)` =
      pivotal_mean_pct,
    
    `Directional proportion: higher pivotal fraction` =
      pivotal_directional_proportion
  )


write_csv(
  table3,
  file.path(
    out_dir,
    "table3_paired_directional_theta07_REV1.csv"
  )
)

write_latex_table(
  table3,
  "table3_paired_directional_theta07_REV1.tex",
  paste0(
    "Paired directional comparison against the Gaussian benchmark ",
    "at $\\theta=0.7$, averaged equally across the nine ",
    "$(M,N)$ configurations."
  ),
  digits = 4
)


# ============================================================
# 12. OPTIONAL q05 DIRECTIONAL ROBUSTNESS
# ============================================================

table3_q05 <- directional %>%
  filter(
    abs(theta - 0.7) < 1e-10,
    metric == "q05_slack_weighted"
  ) %>%
  group_by(
    alternative_dist
  ) %>%
  summarise(
    mean_q05_difference =
      mean(
        mean_difference,
        na.rm = TRUE
      ),
    
    mean_directional_proportion =
      mean(
        directional_proportion,
        na.rm = TRUE
      ),
    
    .groups = "drop"
  ) %>%
  mutate(
    alternative_dist_chr =
      as.character(
        alternative_dist
      ),
    
    Distribution =
      unname(
        dist_labels[
          alternative_dist_chr
        ]
      ),
    
    order =
      match(
        alternative_dist_chr,
        dist_order
      )
  ) %>%
  arrange(
    order
  ) %>%
  transmute(
    Distribution,
    `Mean difference in q05 Slack^W` =
      mean_q05_difference,
    `Directional proportion: lower q05` =
      mean_directional_proportion
  )


write_csv(
  table3_q05,
  file.path(
    out_dir,
    "table_q05_directional_theta07_REV1.csv"
  )
)


# ============================================================
# 13. FIGURE 3 REV1
# Directional-proportion surfaces
# ============================================================
#
# Main figure:
# Lognormal + Pareto 3, 2, 1.5, 1.2
#
# theta = 1 is omitted from the visual because it is a boundary
# case: with positive connected weights all active sources become
# pivotal and distributions tie.
# ============================================================

figure3_alternatives <- c(
  "lognormal_baci",
  "power_law_3_0",
  "power_law_2_0",
  "power_law_1_5",
  "power_law_1_2"
)

figure3_labels <- c(
  "lognormal_baci" = "Lognormal",
  "power_law_3_0"  = "Pareto alpha = 3.0",
  "power_law_2_0"  = "Pareto alpha = 2.0",
  "power_law_1_5"  = "Pareto alpha = 1.5",
  "power_law_1_2"  = "Pareto alpha = 1.2"
)

scenario_heatmap_order <- rev(
  all_scenarios
)

pretty_prop_label <- function(x) {
  dplyr::case_when(
    is.na(x)        ~ NA_character_,
    x == 0          ~ "0.00",
    x < 0.005       ~ formatC(x, format = "e", digits = 1),
    TRUE            ~ sprintf("%.2f", x)
  )
}

heatmap_df <- directional %>%
  filter(
    metric == "frac_slack_weighted_lt_one",
    as.character(alternative_dist) %in%
      figure3_alternatives,
    theta < 1.0
  ) %>%
  mutate(
    alternative_dist_chr =
      as.character(
        alternative_dist
      ),
    
    M_N_scenario =
      factor(
        M_N_scenario,
        levels = scenario_heatmap_order
      ),
    
    theta_factor =
      factor(
        sprintf(
          "%.1f",
          theta
        ),
        levels = sprintf(
          "%.1f",
          sort(
            unique(theta)
          )
        )
      ),
    
    label_text =
      pretty_prop_label(
        directional_proportion
      ),
    
    text_color =
      ifelse(
        directional_proportion >= 0.55,
        "black",
        "white"
      )
  )


make_heatmap_panel <- function(
    alternative,
    show_legend = TRUE
) {
  
  d <- heatmap_df %>%
    filter(
      alternative_dist_chr ==
        alternative
    )
  
  p <- ggplot(
    d,
    aes(
      x = M_N_scenario,
      y = theta_factor,
      fill = directional_proportion
    )
  ) +
    geom_tile() +
    geom_text(
      aes(
        label = label_text,
        color = text_color
      ),
      size = 3.1,
      fontface = "bold",
      show.legend = FALSE
    ) +
    scale_color_identity() +
    scale_fill_viridis_c(
      limits = c(0, 1),
      option = "viridis"
    ) +
    labs(
      title =
        figure3_labels[[alternative]],
      x = "(Sources, targets)",
      y = expression(theta),
      fill = "Paired directional
proportion"
    ) +
    theme_minimal(
      base_size = 10
    ) +
    theme(
      panel.grid = element_blank(),
      axis.text.x = element_text(
        angle = 90,
        hjust = 1,
        vjust = 0.5
      ),
      plot.title = element_text(
        hjust = 0.5,
        size = 11,
        face = "bold"
      ),
      legend.position = if (show_legend) "right" else "none"
    )
  
  p
}

# Combined figure
figure3_panels <- lapply(
  figure3_alternatives,
  function(x) make_heatmap_panel(x, show_legend = TRUE)
)

figure3 <- wrap_plots(
  figure3_panels,
  ncol = 3,
  guides = "collect"
) +
  plot_annotation(
    title = paste0(
      "Paired directional proportion for aggregate source pivotality
",
      "relative to the Gaussian benchmark"
    ),
    subtitle = paste0(
      "theta = 1 omitted from the display because it is a boundary case ",
      "in which all active positive-weight sources are pivotal."
    )
  )

ggsave(
  filename = file.path(
    out_dir,
    "figure3_directional_proportion_surfaces_REV1.png"
  ),
  plot = figure3,
  width = 17,
  height = 10.5,
  dpi = 300,
  bg = "white"
)

# Separate heatmaps as requested
for (alt in figure3_alternatives) {
  
  single_plot <- make_heatmap_panel(
    alt,
    show_legend = TRUE
  ) +
    labs(
      subtitle = "theta = 1 omitted (boundary case)"
    )
  
  safe_name <- gsub(
    "[^A-Za-z0-9]+",
    "_",
    alt
  )
  
  ggsave(
    filename = file.path(
      out_dir,
      paste0(
        "figure3_directional_",
        safe_name,
        "_REV1.png"
      )
    ),
    plot = single_plot,
    width = 7.8,
    height = 5.8,
    dpi = 300,
    bg = "white"
  )
}

# ============================================================
# 14. B-3 THETA SENSITIVITY FIGURE
# Balanced (100,100), all regimes
# ============================================================
# ============================================================

theta_df <- means %>%
  filter(
    M == 100,
    N == 100
  ) %>%
  mutate(
    dist_chr = as.character(dist),
    dist_label = unname(
      dist_labels[dist_chr]
    )
  )

# Sanity check: one curve per distributional regime
theta_counts <- theta_df %>%
  count(dist_chr)

if (nrow(theta_counts) != length(dist_order)) {
  warning(
    "Expected ", length(dist_order),
    " regimes in theta_df but found ",
    nrow(theta_counts), "."
  )
}


make_theta_min_plot <- function(data) {
  
  ggplot(
    data,
    aes(
      x = theta,
      y = min_slack_weighted,
      color = dist_chr,
      shape = dist_chr,
      group = dist_chr
    )
  ) +
    geom_hline(
      yintercept = 1,
      linetype = "dashed",
      color = "gray50"
    ) +
    geom_line(
      linewidth = 0.9,
      alpha = 0.9
    ) +
    geom_point(
      size = 2.6,
      stroke = 0.9
    ) +
    scale_color_discrete(
      breaks = dist_order,
      labels = dist_labels[dist_order]
    ) +
    scale_shape_manual(
      values = regime_shapes,
      breaks = dist_order,
      labels = dist_labels[dist_order]
    ) +
    labs(
      x = expression(theta),
      y = "Mean minimum weighted Slack",
      color = "Distribution",
      shape = "Distribution"
    ) +
    theme_minimal(
      base_size = 12
    ) +
    theme(
      panel.border = element_rect(
        color = "black",
        fill = NA,
        linewidth = 0.5
      ),
      legend.position = "bottom",
      legend.text = element_text(size = 10),
      legend.title = element_text(size = 10),
      legend.key.width = grid::unit(1.4, "cm")
    ) +
    guides(
      color = guide_legend(
        nrow = 2,
        byrow = TRUE
      ),
      shape = guide_legend(
        nrow = 2,
        byrow = TRUE
      )
    )
}


make_theta_piv_plot <- function(data) {
  
  ggplot(
    data,
    aes(
      x = theta,
      y = 100 * frac_slack_weighted_lt_one,
      color = dist_chr,
      shape = dist_chr,
      group = dist_chr
    )
  ) +
    geom_line(
      linewidth = 0.9,
      alpha = 0.9
    ) +
    geom_point(
      size = 2.6,
      stroke = 0.9
    ) +
    scale_color_discrete(
      breaks = dist_order,
      labels = dist_labels[dist_order]
    ) +
    scale_shape_manual(
      values = regime_shapes,
      breaks = dist_order,
      labels = dist_labels[dist_order]
    ) +
    labs(
      x = expression(theta),
      y = "Mean pivotal active sources (%)",
      color = "Distribution",
      shape = "Distribution"
    ) +
    theme_minimal(
      base_size = 12
    ) +
    theme(
      panel.border = element_rect(
        color = "black",
        fill = NA,
        linewidth = 0.5
      ),
      legend.position = "bottom",
      legend.text = element_text(size = 10),
      legend.title = element_text(size = 10),
      legend.key.width = grid::unit(1.4, "cm")
    ) +
    guides(
      color = guide_legend(
        nrow = 2,
        byrow = TRUE
      ),
      shape = guide_legend(
        nrow = 2,
        byrow = TRUE
      )
    )
}


# Full theta grid
theta_panel_min <- make_theta_min_plot(theta_df)
theta_panel_piv <- make_theta_piv_plot(theta_df)

theta_figure_full <- (
  theta_panel_min /
    theta_panel_piv
) +
  plot_annotation(
    title = paste0(
      "Theta sensitivity in the balanced scenario ",
      "(M,N)=(100,100)"
    ),
    subtitle = paste0(
      "Full simulated theta grid. theta = 1 is a boundary case. ",
      "Some curves nearly overlap because the corresponding regimes ",
      "produce very similar Slack values."
    )
  )

ggsave(
  filename = file.path(
    out_dir,
    "figure_theta_sensitivity_balanced_full_grid_REV1.png"
  ),
  plot = theta_figure_full,
  width = 12.5,
  height = 10.5,
  dpi = 300,
  bg = "white"
)


# Recommended substantive version: theta < 1
theta_df_nonboundary <- theta_df %>%
  filter(
    theta < 1.0
  )

theta_panel_min_nb <- make_theta_min_plot(
  theta_df_nonboundary
)

theta_panel_piv_nb <- make_theta_piv_plot(
  theta_df_nonboundary
)

theta_figure_nonboundary <- (
  theta_panel_min_nb /
    theta_panel_piv_nb
) +
  plot_annotation(
    title = paste0(
      "Theta sensitivity in the balanced scenario ",
      "(M,N)=(100,100)"
    ),
    subtitle = paste0(
      "Boundary case theta = 1 omitted from substantive comparison. ",
      "Distinct point shapes identify regimes whose curves overlap."
    )
  )

ggsave(
  filename = file.path(
    out_dir,
    "figure_theta_sensitivity_balanced_theta01_09_REV1.png"
  ),
  plot = theta_figure_nonboundary,
  width = 12.5,
  height = 10.5,
  dpi = 300,
  bg = "white"
)


# ============================================================
# 15. OPTIONAL B-4 FIGURE:
# concentration vs fragility
# ============================================================
# ============================================================

concentration_plot_df <- balanced %>%
  arrange(dist) %>%
  mutate(
    dist_chr = as.character(dist),
    plot_label = unname(
      scatter_labels[dist_chr]
    )
  )

# Strong sanity checks
if (nrow(concentration_plot_df) != length(dist_order)) {
  warning(
    "Expected exactly ", length(dist_order),
    " rows in balanced concentration data, but found ",
    nrow(concentration_plot_df), "."
  )
}

if (anyDuplicated(concentration_plot_df$dist_chr) > 0) {
  warning(
    "Duplicate distribution rows found in balanced concentration data."
  )
}


conc_panel_a <- ggplot(
  concentration_plot_df,
  aes(
    x = mean_target_HHI,
    y = mean_xmin,
    color = dist_chr,
    shape = dist_chr
  )
) +
  geom_hline(
    yintercept = 1,
    linetype = "dashed",
    color = "gray50"
  ) +
  geom_point(
    size = 3.4,
    stroke = 1
  ) +
  ggrepel::geom_text_repel(
    aes(
      label = plot_label
    ),
    size = 3.8,
    color = "black",
    max.overlaps = Inf,
    box.padding = 0.55,
    point.padding = 0.35,
    min.segment.length = 0,
    seed = 42,
    show.legend = FALSE
  ) +
  scale_color_discrete(
    breaks = dist_order,
    labels = dist_labels[dist_order]
  ) +
  scale_shape_manual(
    values = regime_shapes,
    breaks = dist_order,
    labels = dist_labels[dist_order]
  ) +
  labs(
    x = "Mean target HHI",
    y = "Mean minimum weighted Slack",
    color = "Distribution",
    shape = "Distribution"
  ) +
  theme_minimal(
    base_size = 12
  ) +
  theme(
    legend.position = "none"
  )


conc_panel_b <- ggplot(
  concentration_plot_df,
  aes(
    x = mean_target_HHI,
    y = 100 * mean_pivotal_fraction,
    color = dist_chr,
    shape = dist_chr
  )
) +
  geom_point(
    size = 3.4,
    stroke = 1
  ) +
  ggrepel::geom_text_repel(
    aes(
      label = plot_label
    ),
    size = 3.8,
    color = "black",
    max.overlaps = Inf,
    box.padding = 0.55,
    point.padding = 0.35,
    min.segment.length = 0,
    seed = 42,
    show.legend = FALSE
  ) +
  scale_color_discrete(
    breaks = dist_order,
    labels = dist_labels[dist_order]
  ) +
  scale_shape_manual(
    values = regime_shapes,
    breaks = dist_order,
    labels = dist_labels[dist_order]
  ) +
  labs(
    x = "Mean target HHI",
    y = "Mean pivotal active sources (%)",
    color = "Distribution",
    shape = "Distribution"
  ) +
  theme_minimal(
    base_size = 12
  ) +
  theme(
    legend.position = "none"
  )


concentration_figure <- (
  conc_panel_a +
    conc_panel_b
) +
  plot_annotation(
    title = paste0(
      "Realized target concentration and Slack-based fragility\n",
      "(M,N)=(100,100), theta=0.7"
    ),
    subtitle = paste0(
      "One point per distributional regime. Gaussian, Student-t, and ",
      "Pareto alpha=5 cluster closely because their realized concentration ",
      "is very similar."
    )
  )

ggsave(
  filename = file.path(
    out_dir,
    "figure_concentration_vs_fragility_balanced_theta07_REV1.png"
  ),
  plot = concentration_figure,
  width = 13.5,
  height = 6.5,
  dpi = 300,
  bg = "white"
)


# ============================================================
# 16. Across-scenario theta=0.7 summary
# ============================================================

theta07_table <- theta07_summary %>%
  arrange(dist) %>%
  transmute(
    Distribution =
      dist_label,
    
    `Mean x_min` =
      mean_xmin,
    
    `Mean q05 Slack^W` =
      mean_q05_slack,
    
    `Mean pivotal sources (%)` =
      100 * mean_pivotal_fraction,
    
    `Mean target HHI` =
      mean_target_HHI,
    
    `Mean target max share` =
      mean_target_max_share,
    
    `Mean edge-weight Gini` =
      mean_edge_weight_gini
  )


write_csv(
  theta07_table,
  file.path(
    out_dir,
    "table_theta07_across_scenarios_REV1.csv"
  )
)

write_latex_table(
  theta07_table,
  "table_theta07_across_scenarios_REV1.tex",
  paste0(
    "Distribution-level summaries at $\\theta=0.7$, ",
    "averaged equally across the nine $(M,N)$ configurations."
  ),
  digits = 4
)


# ============================================================
# 17. Plain-text output guide
# ============================================================

guide <- c(
  "paper_results_REV1.R output guide",
  "=================================",
  "",
  "MAIN PAPER CANDIDATES",
  "---------------------",
  "table2_balanced_theta07_REV1.*",
  "  Revised Table 2 with the empirically calibrated lognormal and target concentration.",
  "",
  "figure2_distributional_sensitivity_theta07_REV1.png",
  "  Revised combined Figure 2. The two panels use metric-specific display ordering.",
  "",
  "figure2A_minimum_weighted_slack_theta07_REV1.png",
  "  Separate minimum-Slack panel.",
  "",
  "figure2B_pivotal_fraction_theta07_REV1.png",
  "  Separate pivotal-fraction panel; Lognormal is placed between Pareto alpha=1.5 and alpha=1.2.",
  "",
  "table3_paired_directional_theta07_REV1.*",
  "  Revised Table 3. Uses directional proportions, NOT p-values.",
  "",
  "figure3_directional_proportion_surfaces_REV1.png",
  "  Revised Figure 3. Includes BACI lognormal and omits theta=1 boundary from display.",
  "",
  "figure_theta_sensitivity_balanced_theta01_09_REV1.png",
  "  Recommended substantive theta-sensitivity figure.",
  "",
  "OPTIONAL / APPENDIX / REVIEWER RESPONSE",
  "--------------------------------------",
  "table_concentration_fragility_balanced_theta07_REV1.*",
  "  B-4 realized concentration and fragility.",
  "",
  "table_q05_robustness_balanced_theta07_REV1.*",
  "  Checks that findings are not driven only by the absolute minimum.",
  "",
  "table_aggregation_diagnostic_balanced_theta07_REV1.*",
  "  Distinguishes aggregate Slack^W < 1 from having at least one pivotal edge.",
  "",
  "table_q05_directional_theta07_REV1.csv",
  "  Paired directional robustness for the 5th percentile of Slack^W.",
  "",
  "figure_concentration_vs_fragility_balanced_theta07_REV1.png",
  "  Optional visual link between realized target HHI and fragility.",
  "",
  "figure_theta_sensitivity_balanced_full_grid_REV1.png",
  "  Full theta grid including theta=1 boundary.",
  "",
  "NOTES",
  "-----",
  "1. theta=1 is a boundary case and should not be substantively interpreted in the same way as theta<1.",
  "2. The lognormal is an empirically calibrated benchmark, not one point on a universal Pareto tail-ordering axis.",
  "3. Directional proportion is descriptive: the share of paired replications in which the alternative is more fragile than Gaussian.",
  "4. q05 and aggregation diagnostics are generated so they are available for reviewer responses without forcing them into the main manuscript."
)

writeLines(
  guide,
  file.path(
    out_dir,
    "README_REV1_outputs.txt"
  )
)


# ============================================================
# 18. Console summary
# ============================================================

cat("\n============================================================\n")
cat("REV1 PAPER OUTPUTS COMPLETE\n")
cat("============================================================\n")
cat("Output directory:\n")
cat(normalizePath(out_dir, mustWork = FALSE), "\n\n")

cat("Key main-paper files:\n")
cat(" - table2_balanced_theta07_REV1.tex\n")
cat(" - figure2_distributional_sensitivity_theta07_REV1.png\n")
cat(" - figure2A_minimum_weighted_slack_theta07_REV1.png\n")
cat(" - figure2B_pivotal_fraction_theta07_REV1.png\n")
cat(" - table3_paired_directional_theta07_REV1.tex\n")
cat(" - figure3_directional_proportion_surfaces_REV1.png\n")
cat(" - figure_theta_sensitivity_balanced_theta01_09_REV1.png\n\n")

cat("Reviewer / appendix diagnostics are also generated.\n")
cat("See README_REV1_outputs.txt for the full list.\n")
