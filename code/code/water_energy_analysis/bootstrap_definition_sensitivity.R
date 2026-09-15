source(file.path(Sys.getenv("WATER_ENERGY_CODE_DIR", unset = getwd()), "definition_common.R"))
suppressPackageStartupMessages(library(mgcv))

state_path <- we_path("10_LOGS", "WE_DEFINITION_STAGE1_STATE.rds")
if (!file.exists(state_path)) {
  we_stop_check("STOP — DEFINITION MODEL checkS NOT COMPLETE", "BOOTSTRAP_ENTRY_check")
}
state <- readRDS(state_path)
if (!isTRUE(state$stage1_complete)) {
  we_stop_check("STOP — DEFINITION MODEL checkS NOT PASSED", "BOOTSTRAP_ENTRY_check")
}
if (!identical(state$rho_used, WE_RHO_FIXED)) {
  we_stop_check("STOP — PRIMARY DOMAIN IDENTITY MISMATCH", "BOOTSTRAP_RHO_check")
}
if (is.null(state$raw_coordinates) || nrow(state$raw_coordinates) != 1L) {
  we_stop_check("STOP — RAW EXPOSURE ANOMALY NOT UNIQUELY RECONSTRUCTIBLE",
                 "BOOTSTRAP_RAW_COORDINATE_check")
}

master <- we_read_master()
primary <- we_prepare_primary_domain(master, raw_mapping = NULL)
reference_raw_mapping <- list(coordinates = state$raw_coordinates)
bootstrap_summaries <- list()

empty_attempt_log <- function() {
  data.frame(
    attempt = integer(), draw_id = character(), success = logical(),
    contrast = numeric(), convergence = character(), warning = character(),
    failure_reason = character(), failure_category = character(), n = integer(),
    n_boot_series = integer(), n_AR_sections = integer(), elapsed_seconds = numeric(),
    raw_coordinates_recomputed = logical(), stringsAsFactors = FALSE
  )
}

run_bootstrap_analysis <- function(analysis) {
  fit_state <- state$fit_results[[analysis]]
  if (is.null(fit_state) || !isTRUE(fit_state$support_pass)) {
    reason <- if (analysis == "MEDIAN_RESPONSE")
      "STOP — MEDIAN RESPONSE FIXED CONTRAST UNSUPPORTED" else
      "RAW_EXPOSURE_FIXED_CONTRAST_UNSUPPORTED"
    we_stop_check(reason, paste0(analysis, "_BOOTSTRAP_ENTRY_check"))
  }
  meta <- WE_ANALYSIS_META[[analysis]]
  mapping_arg <- if (analysis == "RAW_EXPOSURE") reference_raw_mapping else NULL
  required <- unique(c(
    "point_id", "Year", "Month", "calendar_month_index", meta$response,
    meta$exposure_x, meta$exposure_y, WE_MODEL1_CONTROLS, "AR.start"
  ))
  analysis_data <- primary[, required, drop = FALSE]
  site_ids <- levels(analysis_data$point_id)
  n_sites <- length(site_ids)
  prefix <- paste0("WE_", analysis)

  draw_path <- we_path("06_BOOTSTRAP", paste0(prefix, "_BOOTSTRAP_SITE_DRAW_SEQUENCE.csv"))
  if (file.exists(draw_path)) {
    draw_sequence <- read.csv(
      draw_path,
      colClasses = c(
        attempt = "integer", draw_id = "character", draw_position = "integer",
        original_point_id = "character", boot_series_id = "character"
      ), stringsAsFactors = FALSE
    )
    we_assert(nrow(draw_sequence) == WE_BOOT_DRAWS * n_sites,
                paste(analysis, "draw sequence length mismatch"))
    we_assert(identical(sort(unique(draw_sequence$attempt)), seq_len(WE_BOOT_DRAWS)),
                paste(analysis, "draw attempts incomplete"))
    we_log(analysis, "loaded existing fixed draw sequence")
  } else {
    we_log(analysis, "generating 1050 site draws seed=", WE_SEED, "sites=", n_sites)
    draw_sequence <- we_bootstrap_draw_sequence(site_ids, WE_BOOT_DRAWS, WE_SEED)
    we_write_csv(draw_sequence, draw_path)
  }

  checkpoint_path <- we_path("06_BOOTSTRAP", paste0(prefix, "_BOOTSTRAP_CHECKPOINT.rds"))
  if (file.exists(checkpoint_path)) {
    checkpoint <- readRDS(checkpoint_path)
    attempt_log <- checkpoint$attempt_log
    we_log(analysis, "resuming attempts=", nrow(attempt_log),
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
      contrast_estimate = success_rows$contrast, rho_used = WE_RHO_FIXED,
      raw_coordinates_recomputed = success_rows$raw_coordinates_recomputed,
      unit = "g C m^-2 month^-1", stringsAsFactors = FALSE
    )
    failures <- attempt_log[!attempt_log$success, , drop = FALSE]
    we_write_csv(attempt_log,
                   we_path("06_BOOTSTRAP", paste0(prefix, "_BOOTSTRAP_ATTEMPT_LOG.csv")))
    we_write_csv(success_output,
                   we_path("06_BOOTSTRAP", paste0(prefix, "_CONTRAST_BOOTSTRAP.csv")))
    we_write_csv(failures,
                   we_path("06_BOOTSTRAP", paste0(prefix, "_BOOTSTRAP_FAILURE_LOG.csv")))
    n_attempt <- nrow(attempt_log)
    n_success <- sum(attempt_log$success)
    progress <- data.frame(
      analysis = analysis, updated_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
      n_success = n_success, n_attempt = n_attempt, n_fail = n_attempt - n_success,
      failure_rate = if (n_attempt) (n_attempt - n_success) / n_attempt else 0,
      target_success = WE_BOOT_TARGET_SUCCESS, max_attempts = WE_BOOT_DRAWS,
      initial_batch_attempts = WE_BOOT_INITIAL_BATCH, fixed_seed = WE_SEED,
      fixed_rho = WE_RHO_FIXED, fixed_surface = "c(5,5)", fixed_controls_k = 5L,
      raw_coordinates_recomputed = FALSE, stringsAsFactors = FALSE
    )
    we_write_csv(progress,
                   we_path("06_BOOTSTRAP", paste0(prefix, "_BOOTSTRAP_PROGRESS.csv")))
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
    result <- we_run_definition_boot_attempt(
      attempt, analysis_data, draw_sequence, analysis, mapping_arg, WE_RHO_FIXED
    )
    append_results(result)
    we_log(sprintf(
      "%s initial_batch %d/%d status=%s contrast=%s seconds=%.2f",
      analysis, attempt, WE_BOOT_INITIAL_BATCH,
      if (result$success) "SUCCESS" else paste0("FAIL:", result$failure_category),
      if (is.finite(result$contrast)) format(result$contrast, digits = 10) else "NA",
      result$elapsed_seconds
    ))
    if (!result$success &&
        result$failure_category %in% c("STRUCTURAL_ERROR", "NONFINITE_RESULT")) {
      we_stop_check("STOP — BOOTSTRAP INITIAL_BATCH FAILURE",
                     paste0(analysis, "_BOOTSTRAP_INITIAL_BATCH"), result$failure_reason)
    }
  }

  initial_batch <- attempt_log[attempt_log$attempt <= WE_BOOT_INITIAL_BATCH, , drop = FALSE]
  we_assert(nrow(initial_batch) == WE_BOOT_INITIAL_BATCH,
              paste(analysis, "initial_batch does not contain 50 attempts"))
  initial_batch_fail <- sum(!initial_batch$success)
  initial_batch_rate <- initial_batch_fail / WE_BOOT_INITIAL_BATCH
  structural_ok <- all(initial_batch$n_boot_series[initial_batch$success] == n_sites) &&
    all(is.finite(initial_batch$contrast[initial_batch$success])) &&
    !any(initial_batch$failure_category %in% c("STRUCTURAL_ERROR", "NONFINITE_RESULT")) &&
    !any(initial_batch$raw_coordinates_recomputed)
  initial_batch_summary <- data.frame(
    analysis = analysis, attempts = WE_BOOT_INITIAL_BATCH,
    success = sum(initial_batch$success), fail = initial_batch_fail,
    failure_rate = initial_batch_rate,
    all_successful_contrasts_finite = all(is.finite(initial_batch$contrast[initial_batch$success])),
    expected_boot_series = n_sites,
    all_successful_draws_have_expected_boot_series =
      all(initial_batch$n_boot_series[initial_batch$success] == n_sites),
    structural_error_count = sum(initial_batch$failure_category == "STRUCTURAL_ERROR"),
    nonfinite_error_count = sum(initial_batch$failure_category == "NONFINITE_RESULT"),
    raw_coordinates_recomputed = any(initial_batch$raw_coordinates_recomputed),
    first_50_count_toward_final = TRUE,
    status = if (initial_batch_rate <= WE_BOOT_MAX_FAILURE_RATE && structural_ok)
      "PASS_CONTINUE" else "STOP",
    stringsAsFactors = FALSE
  )
  we_write_csv(
    initial_batch_summary,
    we_path("06_BOOTSTRAP", paste0(prefix, "_BOOTSTRAP_COMPUTATIONAL_INITIAL_BATCH.csv"))
  )
  if (!structural_ok || initial_batch_rate > WE_BOOT_MAX_FAILURE_RATE) {
    we_stop_check(
      "STOP — BOOTSTRAP INITIAL_BATCH FAILURE", paste0(analysis, "_BOOTSTRAP_INITIAL_BATCH"),
      sprintf("failure_rate=%.6f structural_ok=%s", initial_batch_rate, structural_ok)
    )
  }
  we_log(sprintf("%s initial_batch PASS success=%d fail=%d", analysis,
                   sum(initial_batch$success), initial_batch_fail))

  if (sum(attempt_log$success) < WE_BOOT_TARGET_SUCCESS) {
    detected <- parallel::detectCores(logical = FALSE)
    if (is.na(detected) || detected < 1L) detected <- 1L
    worker_count <- min(3L, as.integer(detected))
    we_log(analysis, "starting local PSOCK workers=", worker_count)
    cluster <- parallel::makePSOCKcluster(worker_count, outfile = "")
    cluster_open <- TRUE
    on.exit({
      if (isTRUE(cluster_open)) try(parallel::stopCluster(cluster), silent = TRUE)
    }, add = TRUE)
    parallel::clusterCall(cluster, function(path) { setwd(path); NULL }, WE_PROJECT_ROOT)
    parallel::clusterEvalQ(cluster, {
      source(file.path(Sys.getenv("WATER_ENERGY_CODE_DIR", unset = getwd()), "definition_common.R"))
      suppressPackageStartupMessages(library(mgcv))
      NULL
    })
    boot_data_global <- analysis_data
    boot_draw_global <- draw_sequence
    boot_analysis_global <- analysis
    boot_mapping_global <- mapping_arg
    parallel::clusterExport(
      cluster,
      c("boot_data_global", "boot_draw_global", "boot_analysis_global", "boot_mapping_global"),
      envir = environment()
    )

    repeat {
      n_success <- sum(attempt_log$success)
      if (n_success >= WE_BOOT_TARGET_SUCCESS) break
      next_attempt <- if (nrow(attempt_log)) max(attempt_log$attempt) + 1L else 1L
      if (next_attempt > WE_BOOT_DRAWS) break
      remaining_success <- WE_BOOT_TARGET_SUCCESS - n_success
      remaining_attempts <- WE_BOOT_DRAWS - next_attempt + 1L
      batch_n <- min(worker_count * 4L, remaining_success, remaining_attempts)
      attempts <- seq.int(next_attempt, length.out = batch_n)
      batch <- parallel::parLapply(cluster, attempts, function(i) {
        we_run_definition_boot_attempt(
          i, boot_data_global, boot_draw_global, boot_analysis_global,
          boot_mapping_global, WE_RHO_FIXED
        )
      })
      rows <- do.call(rbind, batch)
      append_results(rows)
      n_success <- sum(attempt_log$success)
      n_attempt <- nrow(attempt_log)
      n_fail <- n_attempt - n_success
      if (n_attempt %% 60L < batch_n || n_success >= WE_BOOT_TARGET_SUCCESS) {
        we_log(sprintf(
          "%s bootstrap progress success=%d/%d attempts=%d fail=%d rate=%.4f",
          analysis, n_success, WE_BOOT_TARGET_SUCCESS, n_attempt, n_fail,
          n_fail / n_attempt
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
      "STOP — DEFINITION SENSITIVITY BOOTSTRAP FAILURE",
      paste0(analysis, "_BOOTSTRAP_FINAL_check"),
      sprintf("success=%d attempts=%d fail=%d rate=%.6f", n_success, n_attempt, n_fail, failure_rate)
    )
  }
  values <- attempt_log$contrast[attempt_log$success]
  we_assert(length(values) == WE_BOOT_TARGET_SUCCESS,
              paste(analysis, "success vector is not exactly 1000"))
  summary <- we_bootstrap_summary_definition(
    values, fit_state$contrast, n_attempt, analysis
  )
  summary$raw_coordinates_recomputed <- FALSE
  summary$status <- "PASS"
  we_write_csv(summary,
                 we_path("06_BOOTSTRAP", paste0(prefix, "_CONTRAST_SUMMARY.csv")))
  we_write_csv(
    data.frame(
      analysis = analysis, stage = "SITE_CLUSTER_BOOTSTRAP", status = "PASS",
      n_success = n_success, n_attempt = n_attempt, n_fail = n_fail,
      failure_rate = failure_rate, raw_coordinates_recomputed = FALSE,
      stringsAsFactors = FALSE
    ),
    we_path("06_BOOTSTRAP", paste0(prefix, "_BOOTSTRAP_check_SUMMARY.csv"))
  )

  png(
    we_path("06_BOOTSTRAP", paste0(prefix, "_BOOTSTRAP_CONTRAST_DISTRIBUTION.png")),
    width = 1800, height = 1400, res = 180
  )
  hist(values, breaks = "FD", col = "grey80", border = "white",
       xlab = "Prediction(A) - Prediction(B), g C m^-2 month^-1",
       main = paste(analysis, "site-cluster bootstrap contrast distribution"))
  abline(v = fit_state$contrast, col = "red", lwd = 3)
  abline(v = c(summary$percentile_2_5, summary$percentile_97_5),
         col = "blue", lwd = 2, lty = 2)
  legend("topright", legend = c("Full-data estimate", "Bootstrap percentile 95% limits"),
         col = c("red", "blue"), lwd = c(3, 2), lty = c(1, 2), bty = "n")
  dev.off()
  we_log(sprintf(
    "%s bootstrap complete success=%d attempts=%d fail=%d CI=[%.15g, %.15g]",
    analysis, n_success, n_attempt, n_fail,
    summary$percentile_2_5, summary$percentile_97_5
  ))
  summary
}

for (analysis in c("MEDIAN_RESPONSE", "RAW_EXPOSURE")) {
  bootstrap_summaries[[analysis]] <- run_bootstrap_analysis(analysis)
  state$bootstrap_summaries <- bootstrap_summaries
  saveRDS(state, state_path)
}

all_boot <- do.call(rbind, bootstrap_summaries)
we_write_csv(all_boot,
               we_path("06_BOOTSTRAP", "WE_DEFINITION_BOOTSTRAP_SUMMARY.csv"))
state$bootstrap_complete <- TRUE
state$bootstrap_completed_at <- format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z")
state$bootstrap_summaries <- bootstrap_summaries
saveRDS(state, state_path)
we_log("All specified definition sensitivity bootstraps complete")
