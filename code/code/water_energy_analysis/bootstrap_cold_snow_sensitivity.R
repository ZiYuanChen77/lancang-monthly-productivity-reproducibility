source(file.path(Sys.getenv("WATER_ENERGY_CODE_DIR", unset = getwd()), "cold_snow_common.R"))
suppressPackageStartupMessages(library(mgcv))

state_path <- we_path("09_LOGS", "WE_COLD_SNOW_STAGE1_STATE.rds")
if (!file.exists(state_path)) {
  we_stop_check("STOP — COLD/SNOW MODEL checkS NOT COMPLETE", "BOOTSTRAP_ENTRY_check")
}
state <- readRDS(state_path)
if (!isTRUE(state$model_complete)) {
  we_stop_check("STOP — COLD/SNOW MODEL checkS NOT PASSED", "BOOTSTRAP_ENTRY_check")
}
if (!identical(state$model$rho_used, WE_RHO_FIXED)) {
  we_stop_check("STOP — PRIMARY REFERENCE MISMATCH", "BOOTSTRAP_RHO_check")
}
if (!isTRUE(state$bootstrap_specified)) {
  we_write_csv(
    data.frame(
      analysis = WE_ALLOWED_ANALYSIS, status = "COLD_SNOW_FIXED_CONTRAST_UNSUPPORTED",
      n_success = 0L, n_attempt = 0L, n_fail = 0L,
      bootstrap_run = FALSE, stringsAsFactors = FALSE
    ),
    we_path("05_BOOTSTRAP", "WE_COLD_SNOW_BOOTSTRAP_NOT_RUN.csv")
  )
  state$bootstrap_complete <- FALSE
  state$bootstrap_status <- "COLD_SNOW_FIXED_CONTRAST_UNSUPPORTED"
  saveRDS(state, state_path)
  we_log("Cold/Snow bootstrap not run because fixed contrast support did not pass")
  quit(save = "no", status = 0L)
}

master <- we_read_master()
cold_full <- we_prepare_cold_snow(master)
required <- c(
  "point_id", "Year", "Month", "calendar_month_index", "npp_anomaly_mean_g",
  "precip_rank_baseline", "radiation_rank_baseline", WE_MODEL1_CONTROLS, "AR.start"
)
cold <- cold_full[, required, drop = FALSE]
site_ids <- levels(cold$point_id)
n_sites <- length(site_ids)
we_assert(n_sites == state$model$n_sites, "Cold/Snow bootstrap site-set mismatch")

draw_path <- we_path("05_BOOTSTRAP", "WE_COLD_SNOW_BOOTSTRAP_SITE_DRAW_SEQUENCE.csv")
if (file.exists(draw_path)) {
  draw_sequence <- read.csv(
    draw_path,
    colClasses = c(
      attempt = "integer", draw_id = "character", draw_position = "integer",
      original_point_id = "character", boot_series_id = "character"
    ), stringsAsFactors = FALSE
  )
  we_assert(nrow(draw_sequence) == WE_BOOT_DRAWS * n_sites,
              "Existing Cold/Snow draw sequence has wrong length")
  we_assert(identical(sort(unique(draw_sequence$attempt)), seq_len(WE_BOOT_DRAWS)),
              "Existing Cold/Snow draw sequence is incomplete")
  we_log("Loaded existing fixed Cold/Snow bootstrap draw sequence")
} else {
  we_log("Generating 1050 Cold/Snow site-cluster draws with seed", WE_SEED, "sites=", n_sites)
  draw_sequence <- we_bootstrap_draw_sequence(site_ids, WE_BOOT_DRAWS, WE_SEED)
  we_write_csv(draw_sequence, draw_path)
}

empty_attempt_log <- function() {
  data.frame(
    attempt = integer(), draw_id = character(), success = logical(),
    contrast = numeric(), convergence = character(), warning = character(),
    failure_reason = character(), failure_category = character(), n = integer(),
    n_boot_series = integer(), n_AR_sections = integer(), elapsed_seconds = numeric(),
    stringsAsFactors = FALSE
  )
}

checkpoint_path <- we_path("05_BOOTSTRAP", "WE_COLD_SNOW_BOOTSTRAP_CHECKPOINT.rds")
if (file.exists(checkpoint_path)) {
  checkpoint <- readRDS(checkpoint_path)
  attempt_log <- checkpoint$attempt_log
  we_log("Resuming Cold/Snow bootstrap attempts=", nrow(attempt_log),
           "success=", sum(attempt_log$success))
} else {
  attempt_log <- empty_attempt_log()
}

flush_bootstrap <- function() {
  if (nrow(attempt_log)) attempt_log <<- attempt_log[order(attempt_log$attempt), , drop = FALSE]
  success_rows <- attempt_log[attempt_log$success, , drop = FALSE]
  success_output <- data.frame(
    successful_replicate = seq_len(nrow(success_rows)),
    attempt = success_rows$attempt, draw_id = success_rows$draw_id,
    contrast_estimate = success_rows$contrast,
    rho_used = WE_RHO_FIXED, unit = "g C m^-2 month^-1",
    stringsAsFactors = FALSE
  )
  failure_rows <- attempt_log[!attempt_log$success, , drop = FALSE]
  we_write_csv(
    attempt_log,
    we_path("05_BOOTSTRAP", "WE_COLD_SNOW_BOOTSTRAP_ATTEMPT_LOG.csv")
  )
  we_write_csv(
    success_output,
    we_path("05_BOOTSTRAP", "WE_COLD_SNOW_CONTRAST_BOOTSTRAP.csv")
  )
  we_write_csv(
    failure_rows,
    we_path("05_BOOTSTRAP", "WE_COLD_SNOW_BOOTSTRAP_FAILURE_LOG.csv")
  )
  n_attempt <- nrow(attempt_log)
  n_success <- sum(attempt_log$success)
  progress <- data.frame(
    analysis = WE_ALLOWED_ANALYSIS,
    updated_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
    n_success = n_success, n_attempt = n_attempt,
    n_fail = n_attempt - n_success,
    failure_rate = if (n_attempt) (n_attempt - n_success) / n_attempt else 0,
    target_success = WE_BOOT_TARGET_SUCCESS, max_attempts = WE_BOOT_DRAWS,
    initial_batch_attempts = WE_BOOT_INITIAL_BATCH, fixed_seed = WE_SEED,
    fixed_rho = WE_RHO_FIXED, formula_changed = FALSE,
    k_changed = FALSE, family_changed = FALSE, stringsAsFactors = FALSE
  )
  we_write_csv(
    progress,
    we_path("05_BOOTSTRAP", "WE_COLD_SNOW_BOOTSTRAP_PROGRESS.csv")
  )
  saveRDS(list(attempt_log = attempt_log, progress = progress), checkpoint_path)
  invisible(progress)
}

append_results <- function(rows) {
  if (nrow(attempt_log)) rows <- rows[!(rows$attempt %in% attempt_log$attempt), , drop = FALSE]
  if (nrow(rows)) attempt_log <<- rbind(attempt_log, rows)
  flush_bootstrap()
}

completed <- if (nrow(attempt_log)) attempt_log$attempt else integer()
for (attempt in seq_len(WE_BOOT_INITIAL_BATCH)) {
  if (attempt %in% completed) next
  result <- we_run_cold_snow_boot_attempt(attempt, cold, draw_sequence)
  append_results(result)
  we_log(sprintf(
    "Cold/Snow initial_batch %d/%d status=%s contrast=%s seconds=%.2f",
    attempt, WE_BOOT_INITIAL_BATCH,
    if (result$success) "SUCCESS" else paste0("FAIL:", result$failure_category),
    if (is.finite(result$contrast)) format(result$contrast, digits = 10) else "NA",
    result$elapsed_seconds
  ))
  if (!result$success && result$failure_category == "STRUCTURAL_ERROR") {
    we_stop_check(
      "STOP — COLD/SNOW BOOTSTRAP INITIAL_BATCH FAILURE", "BOOTSTRAP_INITIAL_BATCH",
      result$failure_reason
    )
  }
}

initial_batch <- attempt_log[attempt_log$attempt <= WE_BOOT_INITIAL_BATCH, , drop = FALSE]
we_assert(nrow(initial_batch) == WE_BOOT_INITIAL_BATCH, "Cold/Snow initial_batch does not contain exactly 50 attempts")
initial_batch_fail <- sum(!initial_batch$success)
initial_batch_rate <- initial_batch_fail / WE_BOOT_INITIAL_BATCH
structural_ok <- all(initial_batch$n_boot_series[initial_batch$success] == n_sites) &&
  all(is.finite(initial_batch$contrast[initial_batch$success])) &&
  !any(initial_batch$failure_category == "STRUCTURAL_ERROR")
initial_batch_summary <- data.frame(
  analysis = WE_ALLOWED_ANALYSIS, attempts = WE_BOOT_INITIAL_BATCH,
  success = sum(initial_batch$success), fail = initial_batch_fail,
  failure_rate = initial_batch_rate,
  all_successful_contrasts_finite = all(is.finite(initial_batch$contrast[initial_batch$success])),
  all_successful_draws_have_expected_boot_series =
    all(initial_batch$n_boot_series[initial_batch$success] == n_sites),
  expected_boot_series = n_sites,
  structural_error_count = sum(initial_batch$failure_category == "STRUCTURAL_ERROR"),
  status = if (initial_batch_rate <= WE_BOOT_MAX_FAILURE_RATE && structural_ok)
    "PASS_CONTINUE" else "STOP",
  first_50_count_toward_final = TRUE, stringsAsFactors = FALSE
)
we_write_csv(
  initial_batch_summary,
  we_path("05_BOOTSTRAP", "WE_COLD_SNOW_BOOTSTRAP_COMPUTATIONAL_INITIAL_BATCH.csv")
)
if (!structural_ok || initial_batch_rate > WE_BOOT_MAX_FAILURE_RATE) {
  we_stop_check(
    "STOP — COLD/SNOW BOOTSTRAP INITIAL_BATCH FAILURE", "BOOTSTRAP_INITIAL_BATCH",
    sprintf("failure_rate=%.6f structural_ok=%s", initial_batch_rate, structural_ok)
  )
}
we_log(sprintf(
  "Cold/Snow initial_batch PASS success=%d fail=%d", sum(initial_batch$success), initial_batch_fail
))

if (sum(attempt_log$success) < WE_BOOT_TARGET_SUCCESS) {
  detected <- parallel::detectCores(logical = FALSE)
  if (is.na(detected) || detected < 1L) detected <- 1L
  worker_count <- min(3L, as.integer(detected))
  we_log("Starting local Cold/Snow PSOCK workers=", worker_count)
  cluster <- parallel::makePSOCKcluster(worker_count, outfile = "")
  cluster_open <- TRUE
  on.exit({
    if (isTRUE(cluster_open)) try(parallel::stopCluster(cluster), silent = TRUE)
  }, add = TRUE)
  parallel::clusterCall(cluster, function(path) { setwd(path); NULL }, WE_PROJECT_ROOT)
  parallel::clusterEvalQ(cluster, {
    source(file.path(Sys.getenv("WATER_ENERGY_CODE_DIR", unset = getwd()), "cold_snow_common.R"))
    suppressPackageStartupMessages(library(mgcv))
    NULL
  })
  cold_global <- cold
  draws_global <- draw_sequence
  parallel::clusterExport(cluster, c("cold_global", "draws_global"), envir = environment())

  repeat {
    n_success <- sum(attempt_log$success)
    if (n_success >= WE_BOOT_TARGET_SUCCESS) break
    next_attempt <- if (nrow(attempt_log)) max(attempt_log$attempt) + 1L else 1L
    if (next_attempt > WE_BOOT_DRAWS) break
    remaining_success <- WE_BOOT_TARGET_SUCCESS - n_success
    remaining_attempts <- WE_BOOT_DRAWS - next_attempt + 1L
    batch_n <- min(worker_count, remaining_success, remaining_attempts)
    attempts <- seq.int(next_attempt, length.out = batch_n)
    batch <- parallel::parLapply(cluster, attempts, function(i) {
      we_run_cold_snow_boot_attempt(i, cold_global, draws_global)
    })
    rows <- do.call(rbind, batch)
    append_results(rows)
    n_success <- sum(attempt_log$success)
    n_attempt <- nrow(attempt_log)
    n_fail <- n_attempt - n_success
    if (n_attempt %% 30L < batch_n || n_success >= WE_BOOT_TARGET_SUCCESS) {
      we_log(sprintf(
        "Cold/Snow bootstrap progress success=%d/%d attempts=%d fail=%d rate=%.4f",
        n_success, WE_BOOT_TARGET_SUCCESS, n_attempt, n_fail, n_fail / n_attempt
      ))
    }
    if (n_fail > (WE_BOOT_DRAWS - WE_BOOT_TARGET_SUCCESS)) break
  }
  parallel::stopCluster(cluster)
  cluster_open <- FALSE
}

flush_bootstrap()
n_success <- sum(attempt_log$success)
n_attempt <- nrow(attempt_log)
n_fail <- n_attempt - n_success
failure_rate <- if (n_attempt) n_fail / n_attempt else 1
if (n_success < WE_BOOT_TARGET_SUCCESS || n_attempt > WE_BOOT_DRAWS ||
    failure_rate > WE_BOOT_MAX_FAILURE_RATE) {
  we_stop_check(
    "STOP — COLD/SNOW BOOTSTRAP FAILURE RATE TOO HIGH", "BOOTSTRAP_FINAL_check",
    sprintf("success=%d attempts=%d fail=%d rate=%.6f", n_success, n_attempt, n_fail, failure_rate)
  )
}

values <- attempt_log$contrast[attempt_log$success]
we_assert(length(values) == WE_BOOT_TARGET_SUCCESS, "Cold/Snow success vector is not exactly 1000")
summary <- we_cold_snow_bootstrap_summary(values, state$model$contrast, n_attempt)
summary$status <- "PASS"
we_write_csv(
  summary,
  we_path("05_BOOTSTRAP", "WE_COLD_SNOW_CONTRAST_SUMMARY.csv")
)
we_write_csv(
  data.frame(
    analysis = WE_ALLOWED_ANALYSIS, stage = "SITE_CLUSTER_BOOTSTRAP", status = "PASS",
    n_success = n_success, n_attempt = n_attempt, n_fail = n_fail,
    failure_rate = failure_rate, stringsAsFactors = FALSE
  ),
  we_path("05_BOOTSTRAP", "WE_COLD_SNOW_BOOTSTRAP_check_SUMMARY.csv")
)

png(
  we_path("05_BOOTSTRAP", "WE_COLD_SNOW_BOOTSTRAP_CONTRAST_DISTRIBUTION.png"),
  width = 1800, height = 1400, res = 180
)
hist(values, breaks = "FD", col = "grey80", border = "white",
     xlab = "Prediction(A) - Prediction(B), g C m^-2 month^-1",
     main = "Cold/Snow site-cluster bootstrap contrast distribution")
abline(v = state$model$contrast, col = "red", lwd = 3)
abline(v = c(summary$percentile_2_5, summary$percentile_97_5),
       col = "blue", lwd = 2, lty = 2)
legend("topright", legend = c("Full-data estimate", "Bootstrap percentile 95% limits"),
       col = c("red", "blue"), lwd = c(3, 2), lty = c(1, 2), bty = "n")
dev.off()

check <- we_cold_snow_check(
  state$model$contrast, summary$percentile_97_5, support_pass = TRUE,
  diagnostic_stop = FALSE
)
summary$robustness_check <- check
we_write_csv(
  summary,
  we_path("05_BOOTSTRAP", "WE_COLD_SNOW_CONTRAST_SUMMARY.csv")
)
state$bootstrap_complete <- TRUE
state$bootstrap_completed_at <- format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z")
state$bootstrap_summary <- summary
state$robustness_check <- check
saveRDS(state, state_path)

we_log(sprintf(
  "Cold/Snow bootstrap complete success=%d attempts=%d fail=%d CI=[%.15g, %.15g] check=%s",
  n_success, n_attempt, n_fail, summary$percentile_2_5, summary$percentile_97_5, check
))

