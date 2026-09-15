source(file.path(Sys.getenv("WATER_ENERGY_CODE_DIR", unset = getwd()), "qa_dayweight_common.R"))
suppressPackageStartupMessages(library(mgcv))

we_log("water-energy QA/Day-weighted sensitivity stage 1 started")
we_write_csv(we_software_validation(), we_path("10_LOGS", "WE_R_MGCV_ENVIRONMENT_validation.csv"))
we_write_lines(capture.output(sessionInfo()), we_path("10_LOGS", "WE_R_SESSION_INFO.txt"))
if (!requireNamespace("mgcv", quietly = TRUE)) {
  we_stop_check("STOP — MGCV NOT AVAILABLE", "ENVIRONMENT_check")
}
if (as.character(packageVersion("mgcv")) != "1.9.4") {
  we_stop_check("STOP — PRIMARY REFERENCE MISMATCH", "ENVIRONMENT_check",
                 paste("mgcv", as.character(packageVersion("mgcv"))))
}

test_path <- we_path("09_TESTS", "WE_QA_DAYWEIGHT_TEST_RESULTS.csv")
if (!file.exists(test_path)) {
  we_stop_check("STOP — AUTOMATED TESTS NOT AVAILABLE", "PRE_MODEL_TEST_check")
}
tests <- read.csv(test_path, stringsAsFactors = FALSE)
if (!nrow(tests) || !all(tests$status == "PASS")) {
  we_stop_check("STOP — AUTOMATED TEST FAILURE", "PRE_MODEL_TEST_check")
}

primary_reference <- we_primary_reference_validation()
we_write_csv(primary_reference, we_path("10_LOGS", "WE_PRIMARY_REFERENCE_validation.csv"))
if (!all(primary_reference$status == "PASS")) {
  we_stop_check(
    "STOP — PRIMARY REFERENCE MISMATCH", "PRIMARY_REFERENCE_check",
    paste(primary_reference$check[primary_reference$status != "PASS"], collapse = ";")
  )
}
reference_values <- attr(primary_reference, "values")

master <- we_read_master()
primary <- we_primary_data(master)
if (nrow(primary) != 6992L || length(levels(primary$point_id)) != 247L) {
  we_stop_check(
    "STOP — PRIMARY REFERENCE MISMATCH", "MASTER_PRIMARY_REFERENCE_check",
    sprintf("master n=%d sites=%d", nrow(primary), length(levels(primary$point_id)))
  )
}

input_validation <- data.frame(
  master_path = normalizePath(WE_MASTER_PATH, winslash = "/", mustWork = TRUE),
  master_n = nrow(master), master_columns = ncol(master),
  primary_v2_path = normalizePath(WE_PRIMARY_ROOT, winslash = "/", mustWork = TRUE),
  primary_reference_refitted = FALSE, canonical_primary_n = 6992L,
  canonical_primary_sites = 247L, canonical_rho = WE_RHO_FIXED,
  output_root = normalizePath(WE_OUTPUT_ROOT, winslash = "/", mustWork = TRUE),
  network_download = FALSE, stringsAsFactors = FALSE
)
we_write_csv(input_validation, we_path("10_LOGS", "WE_INPUT_AND_IMMUTABILITY_validation.csv"))

identity_validations <- do.call(rbind, lapply(WE_ALLOWED_ANALYSES, function(a) {
  we_eligibility_identity_validation(master, a)
}))
we_write_csv(identity_validations, we_path("05_SUPPORT", "WE_SENSITIVITY_ELIGIBILITY_IDENTITY_validation.csv"))
we_write_csv(identity_validations[identity_validations$analysis == "QA1", , drop = FALSE],
               we_path("02_QA1", "WE_QA1_ELIGIBILITY_KEY_validation.csv"))
we_write_csv(identity_validations[identity_validations$analysis == "QA2", , drop = FALSE],
               we_path("03_QA2", "WE_QA2_ELIGIBILITY_KEY_validation.csv"))
we_write_csv(identity_validations[identity_validations$analysis == "DAYWEIGHTED", , drop = FALSE],
               we_path("04_DAYWEIGHTED", "WE_DAYWEIGHTED_ELIGIBILITY_DOMAIN_validation.csv"))
if (!all(identity_validations$status == "PASS")) {
  failing <- identity_validations$analysis[identity_validations$status != "PASS"]
  reason <- if ("QA1" %in% failing) "STOP — QA1 ELIGIBILITY MISMATCH" else
    if ("QA2" %in% failing) "STOP — QA2 ELIGIBILITY MISMATCH" else
      "STOP — DAYWEIGHTED ELIGIBILITY MISMATCH"
  we_stop_check(reason, "SENSITIVITY_ELIGIBILITY_check", paste(failing, collapse = ";"))
}

dw_domain <- we_dayweighted_domain_validation(master)
we_write_csv(dw_domain, we_path("04_DAYWEIGHTED", "WE_DAYWEIGHTED_DOMAIN_validation.csv"))
if (dw_domain$status != "PASS") {
  we_stop_check("STOP — DAYWEIGHTED ELIGIBILITY MISMATCH", "DAYWEIGHTED_DOMAIN_check")
}
dw_identity <- we_dayweighted_identity_change(master)
we_write_csv(
  dw_identity,
  we_path("05_SUPPORT", "WE_DAYWEIGHTED_DOMAIN_IDENTITY_CHANGE.csv")
)
identity_key_detail <- master[
  xor(we_flag_true(master$eligible_primary_model1), we_flag_true(master$eligible_dayweighted)),
  c("point_id", "Year", "Month", "calendar_month_index",
    "historically_active_canonical", "historically_active_dayweighted",
    "eligible_primary_model1", "eligible_dayweighted"), drop = FALSE
]
identity_key_detail$identity_class <- ifelse(
  we_flag_true(identity_key_detail$eligible_primary_model1), "primary_only", "dayweighted_only"
)
we_write_csv(
  identity_key_detail,
  we_path("05_SUPPORT", "WE_DAYWEIGHTED_DOMAIN_IDENTITY_KEY_DETAIL.csv")
)

data_list <- list(
  QA1 = we_prepare_sensitivity(master, "QA1"),
  DAYWEIGHTED = we_prepare_sensitivity(master, "DAYWEIGHTED"),
  QA2 = we_prepare_sensitivity(master, "QA2")
)

sample_summary <- do.call(rbind, lapply(names(data_list), function(a) {
  d <- data_list[[a]]
  data.frame(
    analysis = a, response = WE_ANALYSIS_META[[a]]$response,
    domain = WE_ANALYSIS_META[[a]]$domain,
    n = nrow(d), n_sites = length(levels(d$point_id)),
    n_years = length(unique(d$Year)),
    retention_vs_primary = nrow(d) / nrow(primary),
    rho_used = WE_RHO_FIXED, surface_k = "c(5,5)", controls_k = 5L,
    stringsAsFactors = FALSE
  )
}))
we_write_csv(sample_summary, we_path("08_COMPARISON", "WE_SENSITIVITY_SAMPLE_SUMMARY.csv"))

attrition <- we_mod17_attrition_by_exposure(master)
we_write_csv(attrition, we_path("05_SUPPORT", "WE_MOD17_QA_ATTRITION_BY_EXPOSURE.csv"))
support_attrition <- we_support_attrition(primary, data_list$QA1, data_list$QA2)
we_write_csv(
  support_attrition,
  we_path("05_SUPPORT", "WE_MOD17_QA_ATTRITION_NEAR_FIXED_CONTRAST.csv")
)

support <- do.call(rbind, lapply(names(data_list), function(a) {
  we_support_validation(data_list[[a]], a)
}))
we_write_csv(support, we_path("05_SUPPORT", "WE_FIXED_CONTRAST_SUPPORT_validation.csv"))
for (a in names(data_list)) {
  meta <- WE_ANALYSIS_META[[a]]
  we_write_csv(
    support[support$analysis == a, , drop = FALSE],
    we_path(meta$output_dir, paste0("WE_", a, "_FIXED_CONTRAST_SUPPORT_validation.csv"))
  )
}
qa1_support_pass <- all(support$status[support$analysis == "QA1"] == "PASS")
dw_support_pass <- all(support$status[support$analysis == "DAYWEIGHTED"] == "PASS")
qa2_support_pass <- all(support$status[support$analysis == "QA2"] == "PASS")
if (!qa1_support_pass) {
  we_stop_check("STOP — QA1 CONTRAST UNSUPPORTED", "QA1_SUPPORT_check")
}
if (!dw_support_pass) {
  we_stop_check("STOP — DAYWEIGHTED CONTRAST UNSUPPORTED", "DAYWEIGHTED_SUPPORT_check")
}

ar_validations <- list()
for (a in names(data_list)) {
  d <- data_list[[a]]
  ar <- we_ar_sequence_validation(d, "point_id")
  ar$summary$analysis <- a
  ar_validations[[a]] <- ar
  out_dir <- WE_ANALYSIS_META[[a]]$output_dir
  we_write_csv(ar$summary, we_path(out_dir, paste0("WE_", a, "_AR_SEQUENCE_validation.csv")))
  we_write_csv(ar$gaps, we_path(out_dir, paste0("WE_", a, "_AR_GAPS.csv")))
  we_write_csv(
    ar$gap_distribution,
    we_path(out_dir, paste0("WE_", a, "_AR_GAP_LENGTH_DISTRIBUTION.csv"))
  )
}
we_write_csv(
  do.call(rbind, lapply(ar_validations, `[[`, "summary")),
  we_path("07_DIAGNOSTICS", "WE_SENSITIVITY_AR_SEQUENCE_SUMMARY.csv")
)

we_health_check <- function(model, warnings, analysis) {
  health <- we_model_health(model, warnings)
  reasons <- we_model_health_stop_reason(health)
  if (length(reasons)) {
    if (any(grepl("NONFINITE", reasons))) {
      we_stop_check("STOP — NONFINITE SENSITIVITY MODEL ESTIMATE", paste0(analysis, "_MODEL_HEALTH"),
                     paste(reasons, collapse = ";"))
    }
    if ("RANK DEFICIENCY" %in% reasons) {
      we_stop_check("STOP — RANK DEFICIENCY", paste0(analysis, "_MODEL_HEALTH"),
                     paste(reasons, collapse = ";"))
    }
    if ("NUMERICAL IDENTIFIABILITY FAILURE" %in% reasons) {
      we_stop_check("STOP — IDENTIFIABILITY FAILURE", paste0(analysis, "_MODEL_HEALTH"),
                     paste(reasons, collapse = ";"))
    }
    we_stop_check("STOP — SENSITIVITY MODEL CONVERGENCE FAILURE", paste0(analysis, "_MODEL_HEALTH"),
                   paste(reasons, collapse = ";"))
  }
  health
}

we_plot_png <- function(filename, expression, width = 1800L, height = 1400L) {
  png(filename, width = width, height = height, res = 180)
  on.exit(dev.off(), add = TRUE)
  force(expression)
}

fit_results <- list()
fit_order <- c("QA1", "DAYWEIGHTED", "QA2")
for (a in fit_order) {
  if (a == "QA2" && !qa2_support_pass) {
    we_log("QA2 support is coverage-limited; QA2 fit and bootstrap are not specified")
    fit_results[[a]] <- list(skipped = TRUE, reason = "QA2_COVERAGE_LIMITED_FIXED_CONTRAST_UNSUPPORTED")
    next
  }
  d <- data_list[[a]]
  meta <- WE_ANALYSIS_META[[a]]
  form <- we_formula_sensitivity(meta$response, "point_id")
  we_log("Fitting", a, "canonical Model 1 structure with fixed rho", format(WE_RHO_FIXED, digits = 16))
  started <- proc.time()[["elapsed"]]
  fit <- we_fit_bam(form, d, WE_RHO_FIXED, d$AR.start)
  fit_seconds <- proc.time()[["elapsed"]] - started
  health <- we_health_check(fit$model, fit$warnings, a)
  model_name <- paste0(a, "_MODEL1")
  save_info <- we_save_model_outputs(
    fit$model, fit$warnings, model_name, meta$output_dir, d, WE_RHO_FIXED
  )

  post_ar <- we_ar_postfit_validation(fit$model, d, a)
  we_write_csv(post_ar, we_path(meta$output_dir, paste0("WE_", a, "_AR_POSTFIT_DIAGNOSTIC.csv")))
  if (post_ar$absolute_standardized_residual_lag1 > 0.20) {
    we_stop_check(
      "STOP — AR1 POSTFIT STANDARDIZED RESIDUAL CORRELATION",
      paste0(a, "_AR_POSTFIT_check"),
      sprintf("standardized_residual_lag1=%.15g", post_ar$standardized_residual_lag1)
    )
  }

  basis <- we_run_k_check(fit$model)
  basis$analysis <- a
  basis$surface_k <- "c(5,5)"
  basis$controls_k <- 5L
  basis$k_fallback_permitted <- FALSE
  we_write_csv(basis, we_path(meta$output_dir, paste0("WE_", a, "_BASIS_DIAGNOSTIC.csv")))
  if (any(basis$basis_dimension_suspect)) {
    we_stop_check(
      "STOP — SENSITIVITY BASIS FAILURE", paste0(a, "_BASIS_check"),
      paste(basis$smooth_name[basis$basis_dimension_suspect], collapse = ";")
    )
  }

  contrast <- we_contrast_invariance(fit$model, d)
  invariance <- contrast$validation
  invariance$analysis <- a
  we_write_csv(
    invariance,
    we_path(meta$output_dir, paste0("WE_", a, "_CONTRAST_INVARIANCE_validation.csv"))
  )
  contrast_summary <- data.frame(
    analysis = a, precip_rank_A = 0.75, radiation_rank_A = 0.25,
    precip_rank_B = 0.75, radiation_rank_B = 0.50,
    contrast_definition = "Prediction(A)-Prediction(B)",
    estimate = contrast$main_contrast, unit = "g C m^-2 month^-1",
    extraction = "predict(type='lpmatrix'); dX %*% coef(model)",
    invariance_max_absolute_difference = contrast$max_absolute_difference,
    invariance_status = if (contrast$pass) "PASS" else "STOP",
    stringsAsFactors = FALSE
  )
  we_write_csv(
    contrast_summary,
    we_path(meta$output_dir, paste0("WE_", a, "_FIXED_CONTRAST.csv"))
  )
  if (!is.finite(contrast$main_contrast)) {
    we_stop_check("STOP — NONFINITE SENSITIVITY CONTRAST", paste0(a, "_CONTRAST_check"))
  }
  if (!contrast$pass) {
    we_stop_check(
      "STOP — CONTRAST EXTRACTION FAILURE", paste0(a, "_CONTRAST_check"),
      sprintf("max_difference=%.15g", contrast$max_absolute_difference)
    )
  }

  concurvity <- we_concurvity_tidy(fit$model)
  concurvity$analysis <- a
  highest <- we_relevant_highest_concurvity(concurvity, a)
  we_write_csv(concurvity, we_path(meta$output_dir, paste0("WE_", a, "_CONCURVITY.csv")))
  we_write_csv(highest, we_path(meta$output_dir, paste0("WE_", a, "_HIGHEST_RELEVANT_CONCURVITY.csv")))

  raw <- as.numeric(residuals(fit$model, type = "response"))
  std <- as.numeric(fit$model$std.rsd)
  raw_acf <- we_sequence_acf(d, raw, 12L)
  std_acf <- we_sequence_acf(d, std, 12L)
  raw_acf$analysis <- a
  std_acf$analysis <- a
  we_write_csv(raw_acf, we_path(meta$output_dir, paste0("WE_", a, "_RESIDUAL_SEQUENCE_ACF.csv")))
  we_write_csv(std_acf, we_path(meta$output_dir, paste0("WE_", a, "_STANDARDIZED_RESIDUAL_SEQUENCE_ACF.csv")))

  plot_prefix <- we_path(meta$output_dir, paste0("WE_", a))
  we_plot_png(paste0(plot_prefix, "_RESIDUAL_VS_FITTED.png"), {
    plot(fitted(fit$model), raw, pch = 16, cex = 0.35, col = rgb(0, 0, 0, 0.23),
         xlab = "Fitted value", ylab = "Gaussian response residual",
         main = paste(a, "residual vs fitted"))
    abline(h = 0, col = "red", lwd = 2)
    lines(lowess(fitted(fit$model), raw), col = "blue", lwd = 2)
  })
  we_plot_png(paste0(plot_prefix, "_QQ_PLOT.png"), {
    qqnorm(std, pch = 16, cex = 0.35, col = rgb(0, 0, 0, 0.23),
           main = paste(a, "standardized residual QQ plot"))
    qqline(std, col = "red", lwd = 2)
  })
  we_plot_png(paste0(plot_prefix, "_RESIDUAL_HISTOGRAM.png"), {
    hist(raw, breaks = "FD", col = "grey80", border = "white",
         xlab = "Gaussian response residual", main = paste(a, "residual histogram"))
    abline(v = 0, col = "red", lwd = 2)
  })
  we_plot_png(paste0(plot_prefix, "_RESIDUAL_ACF.png"), {
    plot(raw_acf$lag, raw_acf$correlation, type = "h", lwd = 3, ylim = c(-1, 1),
         xlab = "Exact within-site calendar-month lag", ylab = "Correlation",
         main = paste(a, "residual ACF (valid pairs only)"))
    points(raw_acf$lag, raw_acf$correlation, pch = 16)
    abline(h = 0, col = "grey50")
  })
  we_plot_png(paste0(plot_prefix, "_STANDARDIZED_RESIDUAL_ACF.png"), {
    plot(std_acf$lag, std_acf$correlation, type = "h", lwd = 3, ylim = c(-1, 1),
         xlab = "Exact within-site calendar-month lag", ylab = "Correlation",
         main = paste(a, "standardized residual ACF (valid pairs only)"))
    points(std_acf$lag, std_acf$correlation, pch = 16)
    abline(h = 0, col = "grey50")
  })
  we_plot_png(paste0(plot_prefix, "_OBSERVED_SUPPORT_OVERLAY.png"), {
    plot(d$precip_rank_baseline, d$radiation_rank_baseline,
         pch = 16, cex = 0.34, col = rgb(0.1, 0.2, 0.5, 0.20),
         xlab = "Precipitation baseline rank", ylab = "Radiation baseline rank",
         main = paste(a, "observed support with fixed contrast"))
    theta <- seq(0, 2 * pi, length.out = 400L)
    lines(0.75 + 0.10 * cos(theta), 0.25 + 0.10 * sin(theta), col = "darkgreen", lwd = 2)
    lines(0.75 + 0.10 * cos(theta), 0.50 + 0.10 * sin(theta), col = "darkorange", lwd = 2)
    points(c(0.75, 0.75), c(0.25, 0.50), pch = c(17, 19), cex = 1.8,
           col = c("darkgreen", "darkorange"))
  })

  fit_results[[a]] <- list(
    skipped = FALSE, model_path = we_path(meta$output_dir, paste0("WE_", model_name, ".rds")),
    response = meta$response, n = nrow(d), n_sites = length(levels(d$point_id)),
    formula = paste(deparse(form), collapse = " "), fit_seconds = fit_seconds,
    health = health, post_ar = post_ar, basis = basis,
    contrast = contrast$main_contrast,
    contrast_invariance_max = contrast$max_absolute_difference,
    concurvity_highest = highest,
    support = support[support$analysis == a, , drop = FALSE]
  )
  we_log(sprintf(
    "%s fit complete n=%d sites=%d contrast=%.15g std_resid_lag1=%.15g seconds=%.2f",
    a, nrow(d), length(levels(d$point_id)), contrast$main_contrast,
    post_ar$standardized_residual_lag1, fit_seconds
  ))
}

highest_all <- do.call(rbind, lapply(fit_results, function(x) {
  if (isTRUE(x$skipped)) return(NULL)
  x$concurvity_highest
}))
we_write_csv(highest_all, we_path("07_DIAGNOSTICS", "WE_HIGHEST_RELEVANT_CONCURVITY_BY_SENSITIVITY.csv"))
post_ar_all <- do.call(rbind, lapply(fit_results, function(x) {
  if (isTRUE(x$skipped)) return(NULL)
  x$post_ar
}))
we_write_csv(post_ar_all, we_path("07_DIAGNOSTICS", "WE_AR_POSTFIT_DIAGNOSTIC_SUMMARY.csv"))
basis_all <- do.call(rbind, lapply(fit_results, function(x) {
  if (isTRUE(x$skipped)) return(NULL)
  x$basis
}))
we_write_csv(basis_all, we_path("07_DIAGNOSTICS", "WE_BASIS_DIAGNOSTIC_SUMMARY.csv"))

state <- list(
  stage1_complete = TRUE,
  created_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
  R_version = R.version.string, mgcv_version = as.character(packageVersion("mgcv")),
  rho_used = WE_RHO_FIXED,
  primary_reference = list(
    n = 6992L, n_sites = 247L, contrast = -2.94258224191719,
    ci_lower = -3.55908622452363, ci_upper = -2.21122658573456,
    refitted = FALSE
  ),
  sample_summary = sample_summary, support = support,
  qa2_support_pass = qa2_support_pass, fit_results = fit_results,
  bootstrap_complete = FALSE
)
saveRDS(state, we_path("10_LOGS", "WE_QA_DAYWEIGHT_STAGE1_STATE.rds"))

if (!isTRUE(fit_results$QA2$skipped) && fit_results$QA2$contrast > 0) {
  we_write_csv(
    data.frame(
      analysis = "QA2", contrast = fit_results$QA2$contrast,
      status = "QA2_SIGN_REVERSAL_ANALYSIS_CHECK_REQUIRED",
      bootstrap_run = FALSE, stringsAsFactors = FALSE
    ),
    we_path("03_QA2", "WE_QA2_SIGN_REVERSAL_STOP.csv")
  )
  we_stop_check(
    "QA2_SIGN_REVERSAL_ANALYSIS_CHECK_REQUIRED", "QA2_DIRECTION_check",
    sprintf("contrast=%.15g", fit_results$QA2$contrast)
  )
}

we_log("water-energy QA/Day-weighted sensitivity stage 1 complete; specified bootstraps may proceed")
