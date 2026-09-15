source(file.path(Sys.getenv("WATER_ENERGY_CODE_DIR", unset = getwd()), "primary_common.R"))
suppressPackageStartupMessages(library(mgcv))

state_path <- we_path("14_LOGS", "WE_STAGE1_STATE.rds")
if (!file.exists(state_path)) {
  we_stop_check("STOP — PRIMARY MODEL checkS NOT COMPLETE", "BOOTSTRAP_ENTRY_check")
}
state <- readRDS(state_path)
if (!isTRUE(state[["bootstrap_structural_checks_pass"]])) {
  we_stop_check("STOP — PRIMARY MODEL checkS NOT PASSED", "BOOTSTRAP_ENTRY_check")
}
if (!identical(state[["gaussian_finite_check"]], "PASS_FINITE_VALUES")) {
  we_stop_check("Run validate_primary_residuals.R before bootstrap", "BOOTSTRAP_ENTRY_check")
}

master <- we_read_master()
primary <- we_primary_data(master)
rho_used <- state[["rho_used"]]
k_config <- state[["model1_k_final"]]
site_ids <- levels(primary[["point_id"]])
we_assert(length(site_ids) == 247L, "Bootstrap site count is not 247")

draw_path <- we_path("10_BOOTSTRAP", "WE_BOOTSTRAP_SITE_DRAW_SEQUENCE.csv")
if (file.exists(draw_path)) {
  draw_sequence <- read.csv(
    draw_path,
    colClasses = c(
      attempt = "integer", draw_id = "character", draw_position = "integer",
      original_point_id = "character", boot_series_id = "character"
    ),
    stringsAsFactors = FALSE
  )
  we_assert(nrow(draw_sequence) == WE_BOOT_DRAWS * length(site_ids), "Existing bootstrap draw sequence has wrong length")
  we_assert(identical(sort(unique(draw_sequence[["attempt"]])), seq_len(WE_BOOT_DRAWS)), "Existing draw attempts are incomplete")
  we_log("Loaded existing fixed bootstrap draw sequence for resumable run")
} else {
  we_log("Generating 2100 site bootstrap draws with seed", WE_SEED)
  draw_sequence <- we_bootstrap_draw_sequence(site_ids, WE_BOOT_DRAWS, WE_SEED)
  we_write_csv(draw_sequence, draw_path)
  we_log("Saved complete fixed bootstrap draw sequence rows=", nrow(draw_sequence))
}

checkpoint_path <- we_path("10_BOOTSTRAP", "WE_BOOTSTRAP_CHECKPOINT.rds")
attempt_columns <- c(
  "attempt", "draw_id", "success", "contrast", "convergence", "warning",
  "failure_reason", "failure_category", "n", "n_boot_series", "n_AR_sections",
  "elapsed_seconds"
)
if (file.exists(checkpoint_path)) {
  checkpoint <- readRDS(checkpoint_path)
  attempt_log <- checkpoint[["attempt_log"]]
  we_log("Resuming bootstrap checkpoint attempts=", nrow(attempt_log), "success=", sum(attempt_log[["success"]]))
} else {
  attempt_log <- as.data.frame(setNames(replicate(length(attempt_columns), logical(0), simplify = FALSE), attempt_columns))
  attempt_log[["attempt"]] <- integer()
  attempt_log[["draw_id"]] <- character()
  attempt_log[["success"]] <- logical()
  attempt_log[["contrast"]] <- numeric()
  attempt_log[["convergence"]] <- character()
  attempt_log[["warning"]] <- character()
  attempt_log[["failure_reason"]] <- character()
  attempt_log[["failure_category"]] <- character()
  attempt_log[["n"]] <- integer()
  attempt_log[["n_boot_series"]] <- integer()
  attempt_log[["n_AR_sections"]] <- integer()
  attempt_log[["elapsed_seconds"]] <- numeric()
}

flush_bootstrap <- function() {
  if (nrow(attempt_log)) attempt_log <<- attempt_log[order(attempt_log[["attempt"]]), , drop = FALSE]
  success_rows <- attempt_log[attempt_log[["success"]], , drop = FALSE]
  if (nrow(success_rows)) {
    success_output <- data.frame(
      successful_replicate = seq_len(nrow(success_rows)),
      attempt = success_rows[["attempt"]],
      draw_id = success_rows[["draw_id"]],
      contrast_estimate = success_rows[["contrast"]],
      rho_used = rho_used,
      unit = "g C m^-2 month^-1",
      stringsAsFactors = FALSE
    )
  } else {
    success_output <- data.frame(
      successful_replicate = integer(), attempt = integer(), draw_id = character(),
      contrast_estimate = numeric(), rho_used = numeric(), unit = character()
    )
  }
  failure_rows <- attempt_log[!attempt_log[["success"]], , drop = FALSE]
  we_write_csv(attempt_log, we_path("10_BOOTSTRAP", "WE_BOOTSTRAP_ATTEMPT_LOG.csv"))
  we_write_csv(success_output, we_path("10_BOOTSTRAP", "WE_MODEL1_PRIMARY_CONTRAST_BOOTSTRAP.csv"))
  we_write_csv(failure_rows, we_path("10_BOOTSTRAP", "WE_BOOTSTRAP_FAILURE_LOG.csv"))
  n_attempt <- nrow(attempt_log)
  n_success <- sum(attempt_log[["success"]])
  n_fail <- n_attempt - n_success
  progress <- data.frame(
    updated_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
    n_success = n_success,
    n_attempt = n_attempt,
    n_fail = n_fail,
    failure_rate = if (n_attempt) n_fail / n_attempt else 0,
    target_success = WE_BOOT_TARGET_SUCCESS,
    max_attempts = WE_BOOT_DRAWS,
    initial_batch_attempts = WE_BOOT_INITIAL_BATCH,
    fixed_seed = WE_SEED,
    fixed_rho = rho_used,
    stringsAsFactors = FALSE
  )
  we_write_csv(progress, we_path("10_BOOTSTRAP", "WE_BOOTSTRAP_PROGRESS.csv"))
  saveRDS(list(attempt_log = attempt_log, progress = progress), checkpoint_path)
  invisible(progress)
}

append_results <- function(new_rows) {
  new_rows <- new_rows[, attempt_columns, drop = FALSE]
  if (nrow(attempt_log)) {
    new_rows <- new_rows[!(new_rows[["attempt"]] %in% attempt_log[["attempt"]]), , drop = FALSE]
  }
  if (nrow(new_rows)) attempt_log <<- rbind(attempt_log, new_rows)
  flush_bootstrap()
}

completed_attempts <- if (nrow(attempt_log)) attempt_log[["attempt"]] else integer()
for (attempt in seq_len(WE_BOOT_INITIAL_BATCH)) {
  if (attempt %in% completed_attempts) next
  result <- we_run_boot_attempt(attempt, primary, draw_sequence, k_config, rho_used)
  append_results(result)
  we_log(sprintf(
    "Bootstrap initial_batch attempt %d/%d status=%s contrast=%s seconds=%.2f",
    attempt, WE_BOOT_INITIAL_BATCH,
    if (result[["success"]]) "SUCCESS" else paste0("FAIL:", result[["failure_category"]]),
    if (is.finite(result[["contrast"]])) format(result[["contrast"]], digits = 10) else "NA",
    result[["elapsed_seconds"]]
  ))
  if (!result[["success"]] && result[["failure_category"]] == "STRUCTURAL_ERROR") {
    we_stop_check("STOP — BOOTSTRAP STRUCTURAL ERROR", "BOOTSTRAP_INITIAL_BATCH", result[["failure_reason"]])
  }
}

initial_batch <- attempt_log[attempt_log[["attempt"]] <= WE_BOOT_INITIAL_BATCH, , drop = FALSE]
we_assert(nrow(initial_batch) == WE_BOOT_INITIAL_BATCH, "Bootstrap initial_batch does not contain exactly 50 attempts")
initial_batch_fail <- sum(!initial_batch[["success"]])
initial_batch_failure_rate <- initial_batch_fail / WE_BOOT_INITIAL_BATCH
initial_batch_structural_ok <- all(initial_batch[["n_boot_series"]][initial_batch[["success"]]] == 247L) &&
  all(is.finite(initial_batch[["contrast"]][initial_batch[["success"]]])) &&
  !any(initial_batch[["failure_category"]] == "STRUCTURAL_ERROR")
initial_batch_summary <- data.frame(
  attempts = WE_BOOT_INITIAL_BATCH,
  success = sum(initial_batch[["success"]]),
  fail = initial_batch_fail,
  failure_rate = initial_batch_failure_rate,
  all_successful_contrasts_finite = all(is.finite(initial_batch[["contrast"]][initial_batch[["success"]]])),
  all_successful_draws_have_247_boot_series = all(initial_batch[["n_boot_series"]][initial_batch[["success"]]] == 247L),
  structural_error_count = sum(initial_batch[["failure_category"]] == "STRUCTURAL_ERROR"),
  status = if (initial_batch_failure_rate <= WE_BOOT_MAX_FAILURE_RATE && initial_batch_structural_ok) "PASS_CONTINUE" else "STOP",
  first_50_count_toward_final = TRUE,
  stringsAsFactors = FALSE
)
we_write_csv(initial_batch_summary, we_path("10_BOOTSTRAP", "WE_BOOTSTRAP_COMPUTATIONAL_INITIAL_BATCH.csv"))
if (!initial_batch_structural_ok) {
  we_stop_check("STOP — BOOTSTRAP STRUCTURAL ERROR", "BOOTSTRAP_INITIAL_BATCH")
}
if (initial_batch_failure_rate > WE_BOOT_MAX_FAILURE_RATE) {
  we_stop_check(
    "STOP — BOOTSTRAP FAILURE RATE TOO HIGH",
    "BOOTSTRAP_INITIAL_BATCH",
    sprintf("failure_rate=%.6f", initial_batch_failure_rate)
  )
}
we_log(sprintf(
  "Bootstrap initial_batch PASS success=%d fail=%d failure_rate=%.4f",
  sum(initial_batch[["success"]]), initial_batch_fail, initial_batch_failure_rate
))

n_success <- sum(attempt_log[["success"]])
if (n_success < WE_BOOT_TARGET_SUCCESS) {
  detected <- parallel::detectCores(logical = FALSE)
  if (is.na(detected) || detected < 1L) detected <- 1L
  worker_count <- min(3L, as.integer(detected))
  we_log("Starting local PSOCK bootstrap workers=", worker_count, "after sequential initial_batch")
  cluster <- parallel::makePSOCKcluster(worker_count, outfile = "")
  parallel::clusterCall(cluster, function(path) {
    setwd(path)
    NULL
  }, WE_PROJECT_ROOT)
  parallel::clusterEvalQ(cluster, {
    source(file.path(Sys.getenv("WATER_ENERGY_CODE_DIR", unset = getwd()), "primary_common.R"))
    suppressPackageStartupMessages(library(mgcv))
    NULL
  })
  boot_primary_global <- primary
  boot_draw_sequence_global <- draw_sequence
  boot_k_config_global <- k_config
  boot_rho_global <- rho_used
  parallel::clusterExport(
    cluster,
    c("boot_primary_global", "boot_draw_sequence_global", "boot_k_config_global", "boot_rho_global"),
    envir = environment()
  )

  repeat {
    n_success <- sum(attempt_log[["success"]])
    if (n_success >= WE_BOOT_TARGET_SUCCESS) break
    next_attempt <- if (nrow(attempt_log)) max(attempt_log[["attempt"]]) + 1L else 1L
    if (next_attempt > WE_BOOT_DRAWS) break
    remaining_success <- WE_BOOT_TARGET_SUCCESS - n_success
    remaining_attempts <- WE_BOOT_DRAWS - next_attempt + 1L
    batch_n <- min(worker_count, remaining_success, remaining_attempts)
    attempts <- seq.int(next_attempt, length.out = batch_n)
    batch <- parallel::parLapply(
      cluster,
      attempts,
      function(attempt) {
        we_run_boot_attempt(
          attempt,
          boot_primary_global,
          boot_draw_sequence_global,
          boot_k_config_global,
          boot_rho_global
        )
      }
    )
    batch_rows <- do.call(rbind, batch)
    append_results(batch_rows)
    if (any(!batch_rows[["success"]] & batch_rows[["failure_category"]] == "STRUCTURAL_ERROR")) {
      bad <- batch_rows[!batch_rows[["success"]] & batch_rows[["failure_category"]] == "STRUCTURAL_ERROR", , drop = FALSE]
      parallel::stopCluster(cluster)
      we_stop_check("STOP — BOOTSTRAP STRUCTURAL ERROR", "BOOTSTRAP_FULL_RUN", bad[["failure_reason"]][1L])
    }
    n_success <- sum(attempt_log[["success"]])
    n_attempt <- nrow(attempt_log)
    n_fail <- n_attempt - n_success
    we_log(sprintf(
      "Bootstrap progress success=%d/%d attempts=%d failures=%d failure_rate=%.4f latest=%s",
      n_success, WE_BOOT_TARGET_SUCCESS, n_attempt, n_fail, n_fail / n_attempt,
      paste(attempts, collapse = ",")
    ))
    if (n_fail > (WE_BOOT_DRAWS - WE_BOOT_TARGET_SUCCESS)) break
  }
  parallel::stopCluster(cluster)
}

flush_bootstrap()
n_success <- sum(attempt_log[["success"]])
n_attempt <- nrow(attempt_log)
n_fail <- n_attempt - n_success
failure_rate <- if (n_attempt) n_fail / n_attempt else 1
if (n_success < WE_BOOT_TARGET_SUCCESS || n_attempt > WE_BOOT_DRAWS) {
  we_stop_check(
    "STOP — BOOTSTRAP FAILURE RATE TOO HIGH",
    "BOOTSTRAP_FINAL_check",
    sprintf("success=%d attempts=%d failures=%d", n_success, n_attempt, n_fail)
  )
}
if (failure_rate > WE_BOOT_MAX_FAILURE_RATE) {
  we_stop_check(
    "STOP — BOOTSTRAP FAILURE RATE TOO HIGH",
    "BOOTSTRAP_FINAL_check",
    sprintf("failure_rate=%.6f", failure_rate)
  )
}

success_values <- attempt_log[["contrast"]][attempt_log[["success"]]]
we_assert(length(success_values) == WE_BOOT_TARGET_SUCCESS, "Bootstrap success vector is not exactly 2000")
ci <- unname(quantile(success_values, c(0.025, 0.975), names = FALSE, type = 7))
summary <- data.frame(
  full_estimate = state[["model1_contrast"]],
  bootstrap_median = median(success_values),
  bootstrap_mean = mean(success_values),
  bootstrap_SD = sd(success_values),
  percentile_2_5 = ci[1L],
  percentile_97_5 = ci[2L],
  n_success = n_success,
  n_attempt = n_attempt,
  n_fail = n_fail,
  failure_rate = failure_rate,
  seed = WE_SEED,
  rho_used = rho_used,
  interval_method = "site-cluster bootstrap percentile",
  result_recorded = FALSE,
  stringsAsFactors = FALSE
)
we_write_csv(summary, we_path("10_BOOTSTRAP", "WE_MODEL1_PRIMARY_CONTRAST_SUMMARY.csv"))

png(
  we_path("10_BOOTSTRAP", "WE_MODEL1_BOOTSTRAP_CONTRAST_DISTRIBUTION.png"),
  width = 1800, height = 1400, res = 180
)
hist(
  success_values,
  breaks = "FD",
  col = "grey80",
  border = "white",
  xlab = "Prediction(A) - Prediction(B), g C m^-2 month^-1",
  main = "Model 1 site-cluster bootstrap contrast distribution"
)
abline(v = state[["model1_contrast"]], col = "red", lwd = 3)
abline(v = ci, col = "blue", lwd = 2, lty = 2)
legend(
  "topright",
  legend = c("Full Model 1 estimate", "Bootstrap percentile 95% limits"),
  col = c("red", "blue"), lwd = c(3, 2), lty = c(1, 2), bty = "n"
)
dev.off()

state[["bootstrap_complete"]] <- TRUE
state[["bootstrap_completed_at"]] <- format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z")
state[["bootstrap_summary"]] <- summary
saveRDS(state, state_path)
we_write_csv(
  data.frame(
    stage = "PRIMARY_SITE_CLUSTER_BOOTSTRAP",
    status = "PASS",
    n_success = n_success,
    n_attempt = n_attempt,
    n_fail = n_fail,
    failure_rate = failure_rate,
    stringsAsFactors = FALSE
  ),
  we_path("10_BOOTSTRAP", "WE_BOOTSTRAP_check_SUMMARY.csv")
)
we_log(sprintf(
  "Bootstrap complete success=%d attempts=%d fail=%d CI=[%.12g, %.12g]",
  n_success, n_attempt, n_fail, ci[1L], ci[2L]
))
