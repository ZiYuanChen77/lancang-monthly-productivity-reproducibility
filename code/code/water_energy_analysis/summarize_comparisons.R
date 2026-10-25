# Calculate marginal and simultaneous intervals from joint bootstrap differences.
script <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1L])
code_dir <- dirname(normalizePath(script, winslash = "/", mustWork = TRUE))
source(file.path(code_dir, "comparison_model.R"))
source(file.path(code_dir, "comparison_functions.R"))
opts <- comparison_options(commandArgs(trailingOnly = TRUE), list(
  "--estimates" = "outputs/water_energy/comparisons/comparison_estimates.csv",
  "--bootstrap" = "outputs/water_energy/comparisons/bootstrap_differences.csv",
  "--output" = "outputs/water_energy/comparisons"))
estimates <- read.csv(opts[["--estimates"]], check.names = FALSE)
bootstrap <- read.csv(opts[["--bootstrap"]], check.names = FALSE)
summary <- summarize_comparisons(estimates, bootstrap)
dir.create(opts[["--output"]], recursive = TRUE, showWarnings = FALSE)
write.csv(summary, file.path(opts[["--output"]], "comparison_summary.csv"), row.names = FALSE)
