source(file.path(Sys.getenv("WATER_ENERGY_CODE_DIR", unset = getwd()), "adjustment_common.R"))
suppressPackageStartupMessages(library(mgcv))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L || !toupper(args[1L]) %in% WE_MODEL_NAMES) {
  stop("Usage: we_adjustment_bootstrap.R MODEL2|MODEL3", call. = FALSE)
}
model_name <- toupper(args[1L])
prefix <- paste0("WE_", model_name)
we_log(model_name, "site-cluster bootstrap entry")

state_path <- we_path("09_LOGS", "WE_ADJUSTMENT_STAGE1_STATE.rds")
if (!file.exists(state_path)) {
  we_stop_check("STOP — MODEL2/3 DIAGNOSTICS NOT COMPLETE", "BOOTSTRAP_ENTRY_check")
}
state <- readRDS(state_path)
model_state <- state$model_results[[model_name]]
if (!isTRUE(state$diagnostics_complete) || is.null(model_state) ||
    !isTRUE(model_state$diagnostic_complete) || !isTRUE(model_state$bootstrap_specified)) {
  we_stop_check("STOP — MODEL2/3 DIAGNOSTIC checkS NOT PASSED", "BOOTSTRAP_ENTRY_check", model_name)
}
if (!identical(model_state$rho_used, WE_RHO_FIXED)) {
  we_stop_check("STOP — PRIMARY REFERENCE MISMATCH", "BOOTSTRAP_RHO_check", model_name)
}

master <- we_read_master()
data <- we_prepare_primary_adjustment_data(master)
we_assert(nrow(data) == 6992L && nlevels(data$point_id) == 247L,
            "Adjustment bootstrap Primary-domain mismatch")
site_ids <- levels(data$point_id)
n_sites <- length(site_ids)

draw_path <- we_path(
  "04_BOOTSTRAP", paste0(prefix, "_BOOTSTRAP_SITE_DRAW_SEQUENCE.csv")
)
if (file.exists(draw_path)) {
  draw_sequence <- read.csv(
    draw_path,
    colClasses = c(
      attempt = "integer", draw_id = "character", draw_position = "integer",
      original_point_id = "character", boot_series_id = "character"
    ), stringsAsFactors = FALSE
  )
  we_assert(nrow(draw_sequence) == WE_BOOT_DRAWS * n_sites,
              paste(model_name, "existing draw sequence has wrong length"))
  we_assert(identical(sort(unique(draw_sequence$attempt)), seq_len(WE_BOOT_DRAWS)),
              paste(model_name, "existing draw sequence is incomplete"))
  we_log("Loaded existing", model_name, "fixed draw sequence")
} else {
  we_log("Generating", WE_BOOT_DRAWS, model_name, "site draws seed=", WE_SEED)
  draw_sequence <- we_bootstrap_draw_sequence(site_ids, WE_BOOT_DRAWS, WE_SEED)
  draw_sequence$model <- model_name
  draw_sequence <- draw_sequence[, c("model", setdiff(names(draw_sequence), "model")), drop = FALSE]
  we_write_csv(draw_sequence, draw_path)
}

empty_attempt_log <- function() {
  data.frame(
    model = character(), attempt = integer(), draw_id = character(), success = logical(),
    contrast = numeric(), convergence = character(), warning = character(),
    failure_reason = character(), failure_category = character(), n = integer(),
    n_boot_series = integer(), n_AR_sections = integer(), elapsed_seconds = numeric(),
    stringsAsFactors = FALSE
  )
}

checkpoint_path <- we_path(
  "04_BOOTSTRAP", paste0(prefix, "_BOOTSTRAP_CHECKPOINT.rds")
)
if (file.exists(checkpoint_path)) {
  checkpoint <- readRDS(checkpoint_path)
  attempt_log <- checkpoint$attempt_log
  we_log(
    "Resuming", model_name, "attempts=", nrow(attempt_log),
    "success=", sum(attempt_log$success)
  )
} else {
  attempt_log <- empty_attempt_log()
}

flush_bootstrap <- function() {
  if (nrow(attempt_log)) attempt_log <<- attempt_log[order(attempt_log$attempt), , drop = FALSE]
  success_rows <- attempt_log[attempt_log$success, , drop = FALSE]
  success_output <- data.frame(
    model = model_name,
    successful_replicate = seq_len(nrow(success_rows)),
    attempt = success_rows$attempt, draw_id = success_rows$draw_id,
    contrast_estimate = success_rows$contrast,
    rho_used = WE_RHO_FIXED, unit = "g C m^-2 month^-1",
    stringsAsFactors = FALSE
  )
  failure_rows <- attempt_log[!attempt_log$success, , drop = FALSE]
  we_write_csv(
    attempt_log,
    we_path("04_BOOTSTRAP", paste0(prefix, "_BOOTSTRAP_ATTEMPT_LOG.csv"))
  )
  we_write_csv(
    success_output,
    we_path("04_BOOTSTRAP", paste0(prefix, "_CONTRAST_BOOTSTRAP.csv"))
  )
  we_write_csv(
    failure_rows,
    we_path("04_BOOTSTRAP", paste0(prefix, "_BOOTSTRAP_FAILURE_LOG.csv"))
  )
  n_attempt <- nrow(attempt_log)
  n_success <- sum(attempt_log$success)
  progress <- data.frame(
    model = model_name, updated_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
    n_success = n_success, n_attempt = n_attempt, n_fail = n_attempt - n_success,
    failure_rate = if (n_attempt) (n_attempt - n_success) / n_attempt else 0,
    target_success = WE_BOOT_TARGET_SUCCESS, max_attempts = WE_BOOT_DRAWS,
    initial_batch_attempts = WE_BOOT_INITIAL_BATCH, fixed_seed = WE_SEED,
    fixed_rho = WE_RHO_FIXED, formula_changed = FALSE,
    k_changed = FALSE, family_changed = FALSE, method_changed = FALSE,
    contrast_changed = FALSE, stringsAsFactors = FALSE
  )
  we_write_csv(
    progress,
    we_path("04_BOOTSTRAP", paste0(prefix, "_BOOTSTRAP_PROGRESS.csv"))
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
  result <- we_run_adjustment_boot_attempt(attempt, model_name, data, draw_sequence)
  append_results(result)
  if (attempt %% 5L == 0L || !result$success) {
    we_log(sprintf(
      "%s initial_batch %d/%d success=%d fail=%d last=%s seconds=%.2f",
      model_name, attempt, WE_BOOT_INITIAL_BATCH,
      sum(attempt_log$success), sum(!attempt_log$success),
      if (result$success) "SUCCESS" else paste0("FAIL:", result$failure_category),
      result$elapsed_seconds
    ))
  }
  if (!result$success && result$failure_category == "STRUCTURAL_ERROR") {
    we_stop_check(
      "STOP — MODEL2/3 BOOTSTRAP INITIAL_BATCH FAILURE", "BOOTSTRAP_INITIAL_BATCH",
      paste(model_name, result$failure_reason)
    )
  }
}

initial_batch <- attempt_log[attempt_log$attempt <= WE_BOOT_INITIAL_BATCH, , drop = FALSE]
we_assert(nrow(initial_batch) == WE_BOOT_INITIAL_BATCH,
            paste(model_name, "initial_batch does not contain exactly 50 attempts"))
initial_batch_fail <- sum(!initial_batch$success)
initial_batch_rate <- initial_batch_fail / WE_BOOT_INITIAL_BATCH
structural_ok <- all(initial_batch$n_boot_series[initial_batch$success] == n_sites) &&
  all(is.finite(initial_batch$contrast[initial_batch$success])) &&
  !any(initial_batch$failure_category == "STRUCTURAL_ERROR")
initial_batch_summary <- data.frame(
  model = model_name, attempts = WE_BOOT_INITIAL_BATCH,
  success = sum(initial_batch$success), fail = initial_batch_fail,
  failure_rate = initial_batch_rate,
  all_successful_contrasts_finite = all(is.finite(initial_batch$contrast[initial_batch$success])),
  all_successful_draws_have_expected_boot_series = all(
    initial_batch$n_boot_series[initial_batch$success] == n_sites
  ),
  expected_boot_series = n_sites,
  structural_error_count = sum(initial_batch$failure_category == "STRUCTURAL_ERROR"),
  status = if (initial_batch_rate <= WE_BOOT_MAX_FAILURE_RATE && structural_ok) {
    "PASS_CONTINUE"
  } else {
    "STOP"
  },
  first_50_count_toward_final = TRUE,
  stringsAsFactors = FALSE
)
we_write_csv(
  initial_batch_summary,
  we_path("04_BOOTSTRAP", paste0(prefix, "_BOOTSTRAP_COMPUTATIONAL_INITIAL_BATCH.csv"))
)
if (!structural_ok || initial_batch_rate > WE_BOOT_MAX_FAILURE_RATE) {
  we_stop_check(
    "STOP — MODEL2/3 BOOTSTRAP INITIAL_BATCH FAILURE", "BOOTSTRAP_INITIAL_BATCH",
    sprintf("%s failure_rate=%.6f structural_ok=%s", model_name, initial_batch_rate, structural_ok)
  )
}
we_log(model_name, "initial_batch PASS success=", sum(initial_batch$success), "fail=", initial_batch_fail)

if (sum(attempt_log$success) < WE_BOOT_TARGET_SUCCESS) {
  detected <- parallel::detectCores(logical = FALSE)
  if (is.na(detected) || detected < 1L) detected <- 1L
  worker_count <- min(3L, as.integer(detected))
  we_log("Starting", worker_count, "local PSOCK workers for", model_name)
  cluster <- parallel::makePSOCKcluster(worker_count, outfile = "")
  cluster_open <- TRUE
  on.exit({
    if (isTRUE(cluster_open)) try(parallel::stopCluster(cluster), silent = TRUE)
  }, add = TRUE)
  parallel::clusterCall(cluster, function(path) { setwd(path); NULL }, WE_PROJECT_ROOT)
  parallel::clusterEvalQ(cluster, {
    source(file.path(Sys.getenv("WATER_ENERGY_CODE_DIR", unset = getwd()), "adjustment_common.R"))
    suppressPackageStartupMessages(library(mgcv))
    NULL
  })
  data_global <- data
  draws_global <- draw_sequence
  model_name_global <- model_name
  parallel::clusterExport(
    cluster, c("data_global", "draws_global", "model_name_global"), envir = environment()
  )

  repeat {
    n_success <- sum(attempt_log$success)
    if (n_success >= WE_BOOT_TARGET_SUCCESS) break
    next_attempt <- if (nrow(attempt_log)) max(attempt_log$attempt) + 1L else 1L
    if (next_attempt > WE_BOOT_DRAWS) break
    remaining_success <- WE_BOOT_TARGET_SUCCESS - n_success
    remaining_attempts <- WE_BOOT_DRAWS - next_attempt + 1L
    batch_n <- min(worker_count * 5L, remaining_success, remaining_attempts)
    attempts <- seq.int(next_attempt, length.out = batch_n)
    batch <- parallel::parLapply(cluster, attempts, function(i) {
      we_run_adjustment_boot_attempt(i, model_name_global, data_global, draws_global)
    })
    rows <- do.call(rbind, batch)
    append_results(rows)
    n_success <- sum(attempt_log$success)
    n_attempt <- nrow(attempt_log)
    n_fail <- n_attempt - n_success
    we_log(sprintf(
      "%s bootstrap progress success=%d/%d attempts=%d fail=%d rate=%.4f",
      model_name, n_success, WE_BOOT_TARGET_SUCCESS, n_attempt, n_fail,
      if (n_attempt) n_fail / n_attempt else 0
    ))
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
    "STOP — MODEL2/3 BOOTSTRAP FAILURE RATE TOO HIGH", "BOOTSTRAP_FINAL_check",
    sprintf(
      "%s success=%d attempts=%d fail=%d rate=%.6f",
      model_name, n_success, n_attempt, n_fail, failure_rate
    )
  )
}

values <- attempt_log$contrast[attempt_log$success]
we_assert(length(values) == WE_BOOT_TARGET_SUCCESS,
            paste(model_name, "success vector is not exactly 1000"))
summary <- we_adjustment_bootstrap_summary(
  model_name, values, model_state$contrast, n_attempt
)
check <- we_adjustment_check(
  model_name, model_state$contrast, summary$percentile_97_5,
  support_pass = model_state$support_pass, diagnostic_stop = FALSE
)
summary$robustness_check <- check
summary$status <- "PASS"
we_write_csv(
  summary,
  we_path("04_BOOTSTRAP", paste0(prefix, "_CONTRAST_SUMMARY.csv"))
)
we_write_csv(
  data.frame(
    model = model_name, stage = "SITE_CLUSTER_BOOTSTRAP", status = "PASS",
    n_success = n_success, n_attempt = n_attempt, n_fail = n_fail,
    failure_rate = failure_rate, robustness_check = check,
    stringsAsFactors = FALSE
  ),
  we_path("04_BOOTSTRAP", paste0(prefix, "_BOOTSTRAP_check_SUMMARY.csv"))
)

we_plot_png(
  we_path("04_BOOTSTRAP", paste0(prefix, "_BOOTSTRAP_CONTRAST_DISTRIBUTION.png")),
  {
    hist(
      values, breaks = "FD", col = "grey80", border = "white",
      xlab = "Prediction(A) - Prediction(B), g C m^-2 month^-1",
      main = paste(model_name, "site-cluster bootstrap contrast distribution")
    )
    abline(v = model_state$contrast, col = "red", lwd = 3)
    abline(
      v = c(summary$percentile_2_5, summary$percentile_97_5),
      col = "blue", lwd = 2, lty = 2
    )
    legend(
      "topright", legend = c("Full-data estimate", "Bootstrap percentile 95% limits"),
      col = c("red", "blue"), lwd = c(3, 2), lty = c(1, 2), bty = "n"
    )
  }
)

state$bootstrap_complete[[model_name]] <- TRUE
state$bootstrap_summary[[model_name]] <- summary
state$robustness_check[[model_name]] <- check
state$bootstrap_completed_at[[model_name]] <- format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z")
saveRDS(state, state_path)

we_log(sprintf(
  "%s bootstrap complete success=%d attempts=%d fail=%d CI=[%.15g, %.15g] check=%s",
  model_name, n_success, n_attempt, n_fail,
  summary$percentile_2_5, summary$percentile_97_5, check
))

if (model_state$contrast > 0) {
  we_stop_check(
    paste0(model_name, "_SIGN_REVERSAL_ANALYSIS_CHECK_REQUIRED"),
    "DIRECTION_check", sprintf("contrast=%.15g", model_state$contrast)
  )
}
