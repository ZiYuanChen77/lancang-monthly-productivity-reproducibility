# Fit the water-energy GAMM and jointly bootstrap six comparisons.
script <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1L])
code_dir <- dirname(normalizePath(script, winslash = "/", mustWork = TRUE))
suppressPackageStartupMessages(library(mgcv))
source(file.path(code_dir, "comparison_model.R"))
source(file.path(code_dir, "comparison_functions.R"))

run_comparisons <- function() {
  opts <- comparison_options(commandArgs(trailingOnly = TRUE), list(
    "--input" = "data/water_energy/water_energy_analysis_ready_2022_2025.csv",
    "--output" = "outputs/water_energy/comparisons", "--replicates" = "2000", "--workers" = "1"))
  target <- as.integer(opts[["--replicates"]])
  workers <- as.integer(opts[["--workers"]])
  we_assert(is.finite(target) && target >= 2L && target <= 2000L, "Replicates must be 2 to 2000")
  we_assert(is.finite(workers) && workers >= 1L, "Workers must be positive")
  out <- opts[["--output"]]
  dir.create(out, recursive = TRUE, showWarnings = FALSE)
  d <- read_comparison_data(opts[["--input"]])
  model <- fit_comparison_model(d)
  defs <- comparison_definitions()
  estimates <- cbind(defs, estimate = unname(comparison_values(model, d)), n = nrow(d), sites = nlevels(d$point_id))
  write.csv(estimates, file.path(out, "comparison_estimates.csv"), row.names = FALSE)
  write.csv(comparison_positions(), file.path(out, "positions.csv"), row.names = FALSE)
  X <- comparison_lpmatrix(model, d)
  dx <- X[defs$end, , drop = FALSE] - X[defs$start, , drop = FALSE]
  alternate <- comparison_lpmatrix(model, d, q = .75, last_site = TRUE, last_time = TRUE)
  we_assert(max(abs(dx - (alternate[defs$end, ] - alternate[defs$start, ]))) < 1e-8,
            "Differences depend on reference covariates")
  draws <- we_bootstrap_draw_sequence(levels(d$point_id), n_draws = ceiling(target * 1.05))
  draw_groups <- split(draws, draws$attempt)
  attempt <- function(i) {
    tryCatch({
      b <- we_build_boot_data(d, draw_groups[[as.character(i)]])
      m <- fit_comparison_model(b, "boot_series_id")
      v <- comparison_values(m, b, "boot_series_id")
      we_assert(all(is.finite(v)), "Nonfinite differences")
      cbind(data.frame(attempt = i, success = TRUE, message = ""), as.data.frame(as.list(v)))
    }, error = function(e) {
      empty <- setNames(rep(NA_real_, nrow(defs)), defs$contrast)
      cbind(data.frame(attempt = i, success = FALSE, message = conditionMessage(e)), as.data.frame(as.list(empty)))
    })
  }
  cluster <- NULL
  if (workers > 1L) {
    cluster <- parallel::makePSOCKcluster(workers)
    on.exit(parallel::stopCluster(cluster), add = TRUE)
    parallel::clusterExport(cluster, "code_dir", envir = environment(run_comparisons))
    parallel::clusterEvalQ(cluster, {
      suppressPackageStartupMessages(library(mgcv))
      source(file.path(code_dir, "comparison_model.R"))
      source(file.path(code_dir, "comparison_functions.R"))
      NULL
    })
  }
  rows <- NULL
  next_id <- 1L
  while (is.null(rows) || sum(rows$success) < target) {
    success <- if (is.null(rows)) 0L else sum(rows$success)
    we_assert(next_id <= length(draw_groups), "Insufficient successful bootstrap fits")
    batch <- seq.int(next_id, min(length(draw_groups), next_id + min(workers, target - success) - 1L))
    new <- if (is.null(cluster)) lapply(batch, attempt) else parallel::parLapply(cluster, batch, attempt)
    rows <- rbind(rows, do.call(rbind, new))
    write.csv(rows, file.path(out, "bootstrap_attempts.csv"), row.names = FALSE)
    we_assert(!any(grepl("object .* not found|AR.start length|not strictly increasing|Unknown", rows$message)),
              "Structural bootstrap error")
    we_assert(nrow(rows) < 50L || mean(!rows$success) <= .05, "Bootstrap failure rate exceeds 5%")
    next_id <- max(batch) + 1L
    if (sum(rows$success) %% 50L == 0L) message(sum(rows$success), "/", target, " successful fits")
  }
  ok <- rows[rows$success, c("attempt", defs$contrast)]
  stopifnot(nrow(ok) == target)
  write.csv(ok, file.path(out, "bootstrap_differences.csv"), row.names = FALSE)
  summary <- summarize_comparisons(estimates, ok)
  write.csv(summary, file.path(out, "comparison_summary.csv"), row.names = FALSE)
  invisible(summary)
}
run_comparisons()
