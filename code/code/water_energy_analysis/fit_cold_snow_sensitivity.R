source(file.path(Sys.getenv("WATER_ENERGY_CODE_DIR", unset = getwd()), "cold_snow_common.R"))
suppressPackageStartupMessages(library(mgcv))

we_log("water-energy Cold/Snow model stage started")
prepare_path <- we_path("09_LOGS", "WE_COLD_SNOW_prepare_STATE.rds")
if (!file.exists(prepare_path)) {
  we_stop_check("STOP — COLD/SNOW prepare NOT AVAILABLE", "MODEL_ENTRY_check")
}
state <- readRDS(prepare_path)
if (!isTRUE(state$prepare_complete)) {
  we_stop_check("STOP — COLD/SNOW prepare NOT PASSED", "MODEL_ENTRY_check")
}
test_path <- we_path("08_TESTS", "WE_COLD_SNOW_TEST_RESULTS.csv")
if (!file.exists(test_path)) {
  we_stop_check("STOP — AUTOMATED TESTS NOT AVAILABLE", "PRE_MODEL_TEST_check")
}
tests <- read.csv(test_path, stringsAsFactors = FALSE)
if (!nrow(tests) || !all(tests$status == "PASS")) {
  we_stop_check(
    "STOP — AUTOMATED TEST FAILURE", "PRE_MODEL_TEST_check",
    paste(tests$test[tests$status != "PASS"], collapse = ";")
  )
}

master <- we_read_master()
cold <- we_prepare_cold_snow(master)
support <- we_cold_snow_support_validation(cold)
support_pass <- all(support$status == "PASS")
formula <- we_formula_model1(we_model1_k_initial(), site_variable = "point_id")
formula_text <- paste(deparse(formula, width.cutoff = 500L), collapse = " ")
we_write_lines(formula_text, we_path("03_MODEL", "WE_COLD_SNOW_MODEL1_FORMULA_PRE_FIT.txt"))

we_log(sprintf(
  "Fitting Cold/Snow Model 1 n=%d sites=%d fixed_rho=%.15g",
  nrow(cold), length(levels(cold$point_id)), WE_RHO_FIXED
))
started <- proc.time()[["elapsed"]]
fit <- we_fit_bam(formula, cold, WE_RHO_FIXED, cold$AR.start)
fit_seconds <- proc.time()[["elapsed"]] - started
health <- we_model_health(fit$model, fit$warnings)
reasons <- we_model_health_stop_reason(health)
if (length(reasons)) {
  if (any(grepl("NONFINITE", reasons))) {
    we_stop_check("STOP — NONFINITE COLD/SNOW MODEL ESTIMATE", "MODEL_HEALTH_check", paste(reasons, collapse = ";"))
  }
  if ("RANK DEFICIENCY" %in% reasons) {
    we_stop_check("STOP — RANK DEFICIENCY", "MODEL_HEALTH_check", paste(reasons, collapse = ";"))
  }
  if ("NUMERICAL IDENTIFIABILITY FAILURE" %in% reasons) {
    we_stop_check("STOP — IDENTIFIABILITY FAILURE", "MODEL_HEALTH_check", paste(reasons, collapse = ";"))
  }
  we_stop_check("STOP — COLD/SNOW MODEL CONVERGENCE FAILURE", "MODEL_HEALTH_check", paste(reasons, collapse = ";"))
}
save_info <- we_save_model_outputs(
  fit$model, fit$warnings, "COLD_SNOW_MODEL1", "03_MODEL", cold, WE_RHO_FIXED
)

post_ar <- we_ar_postfit_validation_cold_snow(fit$model, cold)
we_write_csv(
  post_ar,
  we_path("06_DIAGNOSTICS", "WE_COLD_SNOW_AR_POSTFIT_DIAGNOSTIC.csv")
)
if (!is.finite(post_ar$absolute_standardized_residual_lag1) ||
    post_ar$absolute_standardized_residual_lag1 > 0.20) {
  we_stop_check(
    "STOP — COLD/SNOW AR1 POSTFIT STANDARDIZED RESIDUAL CORRELATION",
    "AR_POSTFIT_check",
    sprintf("standardized_residual_lag1=%.15g", post_ar$standardized_residual_lag1)
  )
}

basis <- we_run_k_check(fit$model)
basis$analysis <- WE_ALLOWED_ANALYSIS
basis$surface_k <- "c(5,5)"
basis$controls_k <- 5L
basis$k_fallback_permitted <- FALSE
basis$k_fallback_used <- FALSE
we_write_csv(
  basis,
  we_path("06_DIAGNOSTICS", "WE_COLD_SNOW_BASIS_DIAGNOSTIC.csv")
)
if (any(basis$basis_dimension_suspect)) {
  we_stop_check(
    "STOP — COLD/SNOW BASIS FAILURE", "BASIS_check",
    paste(basis$smooth_name[basis$basis_dimension_suspect], collapse = ";")
  )
}

contrast <- we_contrast_invariance(fit$model, cold, group_col = "point_id")
invariance <- contrast$validation
invariance$analysis <- WE_ALLOWED_ANALYSIS
we_write_csv(
  invariance,
  we_path("03_MODEL", "WE_COLD_SNOW_CONTRAST_INVARIANCE_validation.csv")
)
contrast_summary <- data.frame(
  analysis = WE_ALLOWED_ANALYSIS,
  precip_rank_A = 0.75, radiation_rank_A = 0.25,
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
  we_path("03_MODEL", "WE_COLD_SNOW_FIXED_CONTRAST.csv")
)
if (!is.finite(contrast$main_contrast)) {
  we_stop_check("STOP — NONFINITE COLD/SNOW CONTRAST", "CONTRAST_check")
}
if (!contrast$pass) {
  we_stop_check(
    "STOP — COLD/SNOW CONTRAST EXTRACTION FAILURE", "CONTRAST_check",
    sprintf("max_difference=%.15g", contrast$max_absolute_difference)
  )
}

concurvity <- we_concurvity_tidy(fit$model)
concurvity$analysis <- WE_ALLOWED_ANALYSIS
concurvity_summary <- we_cold_snow_concurvity_summary(concurvity)
we_write_csv(
  concurvity,
  we_path("06_DIAGNOSTICS", "WE_COLD_SNOW_CONCURVITY.csv")
)
we_write_csv(
  concurvity_summary,
  we_path("06_DIAGNOSTICS", "WE_COLD_SNOW_CONCURVITY_SUMMARY.csv")
)

raw <- as.numeric(residuals(fit$model, type = "response"))
std <- as.numeric(fit$model$std.rsd)
if (is.null(fit$model$std.rsd)) std <- raw / sqrt(summary(fit$model)$scale)
raw_acf <- we_sequence_acf(cold, raw, 12L, "point_id")
std_acf <- we_sequence_acf(cold, std, 12L, "point_id")
we_write_csv(
  raw_acf,
  we_path("06_DIAGNOSTICS", "WE_COLD_SNOW_RESIDUAL_SEQUENCE_ACF.csv")
)
we_write_csv(
  std_acf,
  we_path("06_DIAGNOSTICS", "WE_COLD_SNOW_STANDARDIZED_RESIDUAL_SEQUENCE_ACF.csv")
)
residual_diagnostic <- we_residual_diagnostics(fit$model)
residual_diagnostic$heavy_tail_recorded <- residual_diagnostic$residual_excess_kurtosis > 0
residual_diagnostic$family_changed <- FALSE
residual_diagnostic$structural_numerical_failure <- FALSE
we_write_csv(
  residual_diagnostic,
  we_path("06_DIAGNOSTICS", "WE_COLD_SNOW_GAUSSIAN_RESIDUAL_DIAGNOSTICS.csv")
)

plot_prefix <- we_path("06_DIAGNOSTICS", "WE_COLD_SNOW")
we_plot_png(paste0(plot_prefix, "_RESIDUAL_VS_FITTED.png"), {
  plot(fitted(fit$model), raw, pch = 16, cex = 0.35, col = rgb(0, 0, 0, 0.23),
       xlab = "Fitted value", ylab = "Gaussian response residual",
       main = "Cold/Snow residual vs fitted")
  abline(h = 0, col = "red", lwd = 2)
  lines(lowess(fitted(fit$model), raw), col = "blue", lwd = 2)
})
we_plot_png(paste0(plot_prefix, "_QQ_PLOT.png"), {
  qqnorm(std, pch = 16, cex = 0.35, col = rgb(0, 0, 0, 0.23),
         main = "Cold/Snow standardized residual QQ plot")
  qqline(std, col = "red", lwd = 2)
})
we_plot_png(paste0(plot_prefix, "_RESIDUAL_HISTOGRAM.png"), {
  hist(raw, breaks = "FD", col = "grey80", border = "white",
       xlab = "Gaussian response residual", main = "Cold/Snow residual histogram")
  abline(v = 0, col = "red", lwd = 2)
})
we_plot_png(paste0(plot_prefix, "_STANDARDIZED_RESIDUAL_ACF.png"), {
  plot(std_acf$lag, std_acf$correlation, type = "h", lwd = 3, ylim = c(-1, 1),
       xlab = "Exact within-site calendar-month lag", ylab = "Correlation",
       main = "Cold/Snow standardized residual ACF")
  points(std_acf$lag, std_acf$correlation, pch = 16)
  abline(h = 0, col = "grey50")
})
we_plot_png(paste0(plot_prefix, "_RESIDUAL_ACF.png"), {
  plot(raw_acf$lag, raw_acf$correlation, type = "h", lwd = 3, ylim = c(-1, 1),
       xlab = "Exact within-site calendar-month lag", ylab = "Correlation",
       main = "Cold/Snow residual ACF")
  points(raw_acf$lag, raw_acf$correlation, pch = 16)
  abline(h = 0, col = "grey50")
})
we_plot_png(paste0(plot_prefix, "_OBSERVED_SUPPORT_OVERLAY.png"), {
  plot(cold$precip_rank_baseline, cold$radiation_rank_baseline,
       pch = 16, cex = 0.36, col = rgb(0.1, 0.2, 0.5, 0.22),
       xlab = "Precipitation baseline rank", ylab = "Radiation baseline rank",
       main = "Cold/Snow observed support with fixed contrast")
  theta <- seq(0, 2 * pi, length.out = 400L)
  lines(0.75 + 0.10 * cos(theta), 0.25 + 0.10 * sin(theta), col = "darkgreen", lwd = 2)
  lines(0.75 + 0.10 * cos(theta), 0.50 + 0.10 * sin(theta), col = "darkorange", lwd = 2)
  points(c(0.75, 0.75), c(0.25, 0.50), pch = c(17, 19), cex = 1.8,
         col = c("darkgreen", "darkorange"))
})

state$model_complete <- TRUE
state$model_completed_at <- format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z")
state$model <- list(
  n = nrow(cold), n_sites = length(levels(cold$point_id)),
  formula = formula_text, rho_used = WE_RHO_FIXED,
  fit_seconds = fit_seconds, health = health, post_ar = post_ar,
  basis = basis, contrast = contrast$main_contrast,
  contrast_invariance_max = contrast$max_absolute_difference,
  support = support, support_pass = support_pass,
  concurvity_summary = concurvity_summary,
  residual_diagnostic = residual_diagnostic,
  model_path = we_path("03_MODEL", "WE_COLD_SNOW_MODEL1.rds")
)
state$bootstrap_specified <- support_pass
saveRDS(state, we_path("09_LOGS", "WE_COLD_SNOW_STAGE1_STATE.rds"))

we_log(sprintf(
  "Cold/Snow model complete contrast=%.15g std_resid_lag1=%.15g support=%s seconds=%.2f",
  contrast$main_contrast, post_ar$standardized_residual_lag1,
  if (support_pass) "PASS" else "UNSUPPORTED", fit_seconds
))

if (contrast$main_contrast > 0) {
  we_write_csv(
    data.frame(
      contrast = contrast$main_contrast,
      status = "COLD_SNOW_SIGN_REVERSAL_ANALYSIS_CHECK_REQUIRED",
      bootstrap_run = FALSE, stringsAsFactors = FALSE
    ),
    we_path("03_MODEL", "WE_COLD_SNOW_SIGN_REVERSAL_STOP.csv")
  )
  we_stop_check(
    "COLD_SNOW_SIGN_REVERSAL_ANALYSIS_CHECK_REQUIRED", "DIRECTION_check",
    sprintf("contrast=%.15g", contrast$main_contrast)
  )
}

if (!support_pass) {
  we_write_csv(
    data.frame(
      status = "COLD_SNOW_FIXED_CONTRAST_UNSUPPORTED",
      bootstrap_run = FALSE, contrast_moved = FALSE, radius_changed = FALSE,
      stringsAsFactors = FALSE
    ),
    we_path("04_SUPPORT", "WE_COLD_SNOW_SUPPORT_LIMITATION_check.csv")
  )
  we_log("Cold/Snow fixed contrast unsupported; bootstrap is not specified")
} else {
  we_log("Cold/Snow model checks and support passed; bootstrap is specified")
}

