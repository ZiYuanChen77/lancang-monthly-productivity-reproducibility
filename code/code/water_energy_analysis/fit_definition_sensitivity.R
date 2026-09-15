source(file.path(Sys.getenv("WATER_ENERGY_CODE_DIR", unset = getwd()), "definition_common.R"))
suppressPackageStartupMessages(library(mgcv))

we_log("water-energy definition sensitivity model stage started")
if (!requireNamespace("mgcv", quietly = TRUE)) {
  we_stop_check("STOP — MGCV NOT AVAILABLE", "ENVIRONMENT_check")
}
if (as.character(packageVersion("mgcv")) != "1.9.4") {
  we_stop_check("STOP — PRIMARY DOMAIN IDENTITY MISMATCH", "ENVIRONMENT_check",
                 paste("mgcv", as.character(packageVersion("mgcv"))))
}

test_path <- we_path("09_TESTS", "WE_DEFINITION_SENSITIVITY_TEST_RESULTS.csv")
if (!file.exists(test_path)) {
  we_stop_check("STOP — AUTOMATED TESTS NOT AVAILABLE", "PRE_MODEL_TEST_check")
}
tests <- read.csv(test_path, stringsAsFactors = FALSE)
if (!nrow(tests) || !all(tests$status == "PASS")) {
  we_stop_check("STOP — AUTOMATED TEST FAILURE", "PRE_MODEL_TEST_check")
}
prepare_path <- we_path("10_LOGS", "WE_DEFINITION_prepare_STATE.rds")
if (!file.exists(prepare_path)) {
  we_stop_check("STOP — RAW EXPOSURE ANOMALY NOT UNIQUELY RECONSTRUCTIBLE",
                 "prepare_ENTRY_check")
}
prepare <- readRDS(prepare_path)
if (!isTRUE(prepare$complete) || prepare$raw_validation$centering_status != "PASS") {
  we_stop_check("STOP — RAW ANOMALY BASELINE CENTERING FAILURE", "prepare_ENTRY_check")
}
reference <- we_reference_reference_validation()
if (!all(reference$status == "PASS")) {
  we_stop_check("STOP — PRIMARY DOMAIN IDENTITY MISMATCH", "reference_REFERENCE_check")
}

master <- we_read_master()
identity <- we_primary_identity_validation(master)
if (identity$status != "PASS") {
  we_stop_check("STOP — PRIMARY DOMAIN IDENTITY MISMATCH", "PRIMARY_KEY_check")
}
primary <- we_prepare_primary_domain(master, raw_mapping = NULL)
raw_mapping <- list(coordinates = prepare$coordinates)
support <- read.csv(
  we_path("05_SUPPORT", "WE_DEFINITION_FIXED_CONTRAST_SUPPORT_validation.csv"),
  stringsAsFactors = FALSE
)
if (!all(support$status == "PASS")) {
  failing <- unique(support$analysis[support$status != "PASS"])
  reason <- if ("MEDIAN_RESPONSE" %in% failing)
    "STOP — MEDIAN RESPONSE FIXED CONTRAST UNSUPPORTED" else
    "RAW_EXPOSURE_FIXED_CONTRAST_UNSUPPORTED"
  we_stop_check(reason, "FIXED_CONTRAST_SUPPORT_check", paste(failing, collapse = ";"))
}

input_validation <- data.frame(
  master_path = normalizePath(WE_MASTER_PATH, winslash = "/", mustWork = TRUE),
  reference_historical_path = normalizePath(WE_RAW_MONTHLY_PATH, winslash = "/", mustWork = TRUE),
  primary_v2_path = normalizePath(WE_PRIMARY_ROOT, winslash = "/", mustWork = TRUE),
  target_robustness_path = normalizePath(WE_TARGET_ROOT, winslash = "/", mustWork = TRUE),
  domain_robustness_path = normalizePath(WE_DOMAIN_ROOT, winslash = "/", mustWork = TRUE),
  prior_models_refitted = FALSE, network_download = FALSE,
  analysis_definition_changed = FALSE, output_definition_changed = FALSE,
  raw_coordinates_recomputed_after_freeze = FALSE,
  output_root = normalizePath(WE_OUTPUT_ROOT, winslash = "/", mustWork = TRUE),
  stringsAsFactors = FALSE
)
we_write_csv(input_validation,
               we_path("10_LOGS", "WE_INPUT_AND_IMMUTABILITY_validation.csv"))

ar <- we_ar_sequence_validation(primary, "point_id")
for (a in WE_ALLOWED_ANALYSES) {
  out_dir <- WE_ANALYSIS_META[[a]]$output_dir
  summary <- ar$summary
  summary$analysis <- a
  summary$canonical_primary_AR_sections <- 988L
  summary$matches_canonical_primary <- summary$n_AR_sections == 988L
  we_write_csv(summary,
                 we_path(out_dir, paste0("WE_", a, "_AR_SEQUENCE_validation.csv")))
  we_write_csv(ar$gaps,
                 we_path(out_dir, paste0("WE_", a, "_AR_GAPS.csv")))
}
if (ar$summary$n_AR_sections != 988L) {
  we_stop_check("STOP — AR STRUCTURE MISMATCH", "AR_STRUCTURE_check")
}

we_definition_health_check <- function(model, warnings, analysis) {
  health <- we_model_health(model, warnings)
  reasons <- we_model_health_stop_reason(health)
  if (length(reasons)) {
    if (any(grepl("NONFINITE", reasons))) {
      we_stop_check("STOP — NONFINITE DEFINITION MODEL ESTIMATE",
                     paste0(analysis, "_MODEL_HEALTH"), paste(reasons, collapse = ";"))
    }
    if ("RANK DEFICIENCY" %in% reasons) {
      we_stop_check("STOP — RANK DEFICIENCY", paste0(analysis, "_MODEL_HEALTH"),
                     paste(reasons, collapse = ";"))
    }
    if ("NUMERICAL IDENTIFIABILITY FAILURE" %in% reasons) {
      we_stop_check("STOP — IDENTIFIABILITY FAILURE", paste0(analysis, "_MODEL_HEALTH"),
                     paste(reasons, collapse = ";"))
    }
    we_stop_check("STOP — DEFINITION MODEL CONVERGENCE FAILURE",
                   paste0(analysis, "_MODEL_HEALTH"), paste(reasons, collapse = ";"))
  }
  health
}

we_plot_png <- function(filename, expression, width = 1800L, height = 1400L) {
  png(filename, width = width, height = height, res = 180)
  on.exit(dev.off(), add = TRUE)
  force(expression)
}

fit_results <- list()
for (a in c("MEDIAN_RESPONSE", "RAW_EXPOSURE")) {
  meta <- WE_ANALYSIS_META[[a]]
  mapping_arg <- if (a == "RAW_EXPOSURE") raw_mapping else NULL
  form <- we_formula_definition(a, "point_id")
  we_log("Fitting", a, "n=", nrow(primary), "sites=", length(levels(primary$point_id)),
           "fixed rho=", format(WE_RHO_FIXED, digits = 16))
  started <- proc.time()[["elapsed"]]
  fit <- we_fit_bam(form, primary, WE_RHO_FIXED, primary$AR.start)
  fit_seconds <- proc.time()[["elapsed"]] - started
  health <- we_definition_health_check(fit$model, fit$warnings, a)
  model_name <- paste0(a, "_MODEL1")
  we_save_model_outputs(
    fit$model, fit$warnings, model_name, meta$output_dir, primary, WE_RHO_FIXED
  )

  post_ar <- we_ar_postfit_validation_definition(fit$model, primary, a)
  we_write_csv(post_ar,
                 we_path(meta$output_dir, paste0("WE_", a, "_AR_POSTFIT_DIAGNOSTIC.csv")))
  if (post_ar$absolute_standardized_residual_lag1 > 0.20) {
    we_stop_check("STOP — AR POSTFIT STANDARDIZED RESIDUAL CORRELATION",
                   paste0(a, "_AR_POSTFIT_check"),
                   sprintf("lag1=%.15g", post_ar$standardized_residual_lag1))
  }

  basis <- we_run_k_check(fit$model)
  basis$analysis <- a
  basis$surface_k <- "c(5,5)"
  basis$controls_k <- 5L
  basis$k_fallback_permitted <- FALSE
  we_write_csv(basis,
                 we_path(meta$output_dir, paste0("WE_", a, "_BASIS_DIAGNOSTIC.csv")))
  if (any(basis$basis_dimension_suspect)) {
    we_stop_check("STOP — DEFINITION SENSITIVITY BASIS FAILURE",
                   paste0(a, "_BASIS_check"),
                   paste(basis$smooth_name[basis$basis_dimension_suspect], collapse = ";"))
  }

  contrast <- we_definition_contrast_invariance(
    fit$model, primary, a, mapping_arg, "point_id"
  )
  invariance <- contrast$validation
  invariance$analysis <- a
  we_write_csv(
    invariance,
    we_path(meta$output_dir, paste0("WE_", a, "_CONTRAST_INVARIANCE_validation.csv"))
  )
  coords <- we_contrast_coordinates(a, mapping_arg)
  contrast_summary <- data.frame(
    analysis = a, exposure_x_field = meta$exposure_x, exposure_y_field = meta$exposure_y,
    A_x = coords$A[[meta$exposure_x]], A_y = coords$A[[meta$exposure_y]],
    B_x = coords$B[[meta$exposure_x]], B_y = coords$B[[meta$exposure_y]],
    contrast_definition = "Prediction(A)-Prediction(B)", estimate = contrast$main_contrast,
    unit = "g C m^-2 month^-1",
    extraction = "predict(type='lpmatrix'); dX %*% coef(model)",
    invariance_max_absolute_difference = contrast$max_absolute_difference,
    invariance_status = if (contrast$pass) "PASS" else "STOP",
    raw_coordinates_recomputed = FALSE, stringsAsFactors = FALSE
  )
  we_write_csv(
    contrast_summary,
    we_path(meta$output_dir, paste0("WE_", a, "_FIXED_CONTRAST.csv"))
  )
  if (!is.finite(contrast$main_contrast)) {
    we_stop_check("STOP — NONFINITE CONTRAST", paste0(a, "_CONTRAST_check"))
  }
  if (!contrast$pass) {
    reason <- if (a == "RAW_EXPOSURE")
      "STOP — RAW CONTRAST NOT INVARIANT" else "STOP — CONTRAST EXTRACTION FAILURE"
    we_stop_check(reason, paste0(a, "_CONTRAST_check"),
                   sprintf("max_difference=%.15g", contrast$max_absolute_difference))
  }

  concurvity <- we_concurvity_tidy(fit$model)
  concurvity$analysis <- a
  surface_concurvity <- we_surface_concurvity_summary(concurvity, a)
  highest <- we_highest_concurvity(concurvity, a)
  we_write_csv(concurvity,
                 we_path(meta$output_dir, paste0("WE_", a, "_CONCURVITY.csv")))
  we_write_csv(surface_concurvity,
                 we_path(meta$output_dir, paste0("WE_", a, "_SURFACE_CONCURVITY.csv")))
  we_write_csv(highest,
                 we_path(meta$output_dir, paste0("WE_", a, "_HIGHEST_CONCURVITY.csv")))

  residual_diag <- we_residual_diagnostics(fit$model)
  residual_diag$analysis <- a
  residual_diag$family_changed <- FALSE
  residual_diag$diagnostic_status <- "FINITE_MODEL_HEAVY_TAIL_RECORDED"
  we_write_csv(residual_diag,
                 we_path(meta$output_dir,
                           paste0("WE_", a, "_GAUSSIAN_RESIDUAL_DIAGNOSTICS.csv")))
  raw_residual <- as.numeric(residuals(fit$model, type = "response"))
  std_residual <- as.numeric(fit$model$std.rsd)
  std_acf <- we_sequence_acf(primary, std_residual, 12L)
  std_acf$analysis <- a
  we_write_csv(std_acf,
                 we_path(meta$output_dir,
                           paste0("WE_", a, "_STANDARDIZED_RESIDUAL_SEQUENCE_ACF.csv")))

  prefix <- we_path(meta$output_dir, paste0("WE_", a))
  we_plot_png(paste0(prefix, "_RESIDUAL_VS_FITTED.png"), {
    plot(fitted(fit$model), raw_residual, pch = 16, cex = 0.34,
         col = rgb(0, 0, 0, 0.22), xlab = "Fitted value",
         ylab = "Gaussian response residual", main = paste(a, "residual vs fitted"))
    abline(h = 0, col = "red", lwd = 2)
    lines(lowess(fitted(fit$model), raw_residual), col = "blue", lwd = 2)
  })
  we_plot_png(paste0(prefix, "_QQ_PLOT.png"), {
    qqnorm(std_residual, pch = 16, cex = 0.34, col = rgb(0, 0, 0, 0.22),
           main = paste(a, "standardized residual QQ plot"))
    qqline(std_residual, col = "red", lwd = 2)
  })
  we_plot_png(paste0(prefix, "_RESIDUAL_HISTOGRAM.png"), {
    hist(raw_residual, breaks = "FD", col = "grey80", border = "white",
         xlab = "Gaussian response residual", main = paste(a, "residual histogram"))
    abline(v = 0, col = "red", lwd = 2)
  })
  we_plot_png(paste0(prefix, "_STANDARDIZED_RESIDUAL_ACF.png"), {
    plot(std_acf$lag, std_acf$correlation, type = "h", lwd = 3, ylim = c(-1, 1),
         xlab = "Exact within-site calendar-month lag", ylab = "Correlation",
         main = paste(a, "standardized residual ACF"))
    points(std_acf$lag, std_acf$correlation, pch = 16)
    abline(h = 0, col = "grey50")
  })

  fit_results[[a]] <- list(
    model_path = we_path(meta$output_dir, paste0("WE_", model_name, ".rds")),
    response = meta$response, exposure_x = meta$exposure_x, exposure_y = meta$exposure_y,
    n = nrow(primary), n_sites = length(levels(primary$point_id)),
    formula = paste(deparse(form), collapse = " "), fit_seconds = fit_seconds,
    health = health, post_ar = post_ar, basis = basis,
    contrast = contrast$main_contrast,
    contrast_invariance_max = contrast$max_absolute_difference,
    surface_concurvity = surface_concurvity, highest_concurvity = highest,
    residual_diagnostics = residual_diag,
    support = support[support$analysis == a, , drop = FALSE],
    support_pass = all(support$status[support$analysis == a] == "PASS")
  )
  we_log(sprintf(
    "%s fit complete contrast=%.15g std_resid_lag1=%.15g seconds=%.2f",
    a, contrast$main_contrast, post_ar$standardized_residual_lag1, fit_seconds
  ))

  if (contrast$main_contrast > 0) {
    reason <- if (a == "MEDIAN_RESPONSE")
      "MEDIAN_RESPONSE_SIGN_REVERSAL_ANALYSIS_CHECK_REQUIRED" else
      "RAW_EXPOSURE_SIGN_REVERSAL_ANALYSIS_CHECK_REQUIRED"
    partial_state <- list(
      stage1_complete = FALSE, stopped_analysis = a, fit_results = fit_results,
      raw_coordinates = prepare$coordinates, rho_used = WE_RHO_FIXED
    )
    saveRDS(partial_state, we_path("10_LOGS", "WE_DEFINITION_STAGE1_STATE.rds"))
    we_stop_check(reason, paste0(a, "_DIRECTION_check"),
                   sprintf("contrast=%.15g", contrast$main_contrast))
  }
}

we_write_csv(
  do.call(rbind, lapply(fit_results, `[[`, "post_ar")),
  we_path("07_DIAGNOSTICS", "WE_DEFINITION_AR_POSTFIT_DIAGNOSTIC_SUMMARY.csv")
)
we_write_csv(
  do.call(rbind, lapply(fit_results, `[[`, "basis")),
  we_path("07_DIAGNOSTICS", "WE_DEFINITION_BASIS_DIAGNOSTIC_SUMMARY.csv")
)
we_write_csv(
  do.call(rbind, lapply(fit_results, `[[`, "surface_concurvity")),
  we_path("07_DIAGNOSTICS", "WE_DEFINITION_SURFACE_CONCURVITY_SUMMARY.csv")
)
we_write_csv(
  do.call(rbind, lapply(fit_results, `[[`, "residual_diagnostics")),
  we_path("07_DIAGNOSTICS", "WE_DEFINITION_GAUSSIAN_RESIDUAL_SUMMARY.csv")
)

state <- list(
  stage1_complete = TRUE,
  created_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
  R_version = R.version.string, mgcv_version = as.character(packageVersion("mgcv")),
  rho_used = WE_RHO_FIXED, primary_n = nrow(primary),
  primary_sites = length(levels(primary$point_id)), primary_AR_sections = 988L,
  raw_coordinates = prepare$coordinates, raw_mapping_summary = prepare$mapping_summary,
  support = support, fit_results = fit_results, bootstrap_complete = FALSE,
  prior_references_refitted = FALSE
)
saveRDS(state, we_path("10_LOGS", "WE_DEFINITION_STAGE1_STATE.rds"))
we_log("water-energy definition sensitivity stage 1 complete; bootstraps may proceed")
