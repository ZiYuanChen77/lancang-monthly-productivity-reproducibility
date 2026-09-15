source(file.path(Sys.getenv("WATER_ENERGY_CODE_DIR", unset = getwd()), "primary_common.R"))
suppressPackageStartupMessages(library(mgcv))

we_log("water-energy GAMM primary analysis stage 1 started")
we_write_csv(we_software_validation(), we_path("14_LOGS", "WE_R_MGCV_ENVIRONMENT_validation.csv"))
we_write_lines(capture.output(sessionInfo()), we_path("14_LOGS", "WE_R_SESSION_INFO.txt"))

if (!identical(as.character(packageVersion("mgcv")), "1.9.4")) {
  we_log(sprintf("mgcv version %s differs from the reference environment (1.9-4)", as.character(packageVersion("mgcv"))))
}

test_results_path <- we_path("13_TESTS", "WE_GAMM_PRIMARY_TEST_RESULTS.csv")
if (!file.exists(test_results_path)) {
  we_stop_check("STOP — AUTOMATED TESTS NOT AVAILABLE", "PRE_MODEL_TEST_check")
}
test_results <- read.csv(test_results_path, stringsAsFactors = FALSE)
if (!all(test_results[["status"]] == "PASS")) {
  we_stop_check("STOP — AUTOMATED TEST FAILURE", "PRE_MODEL_TEST_check")
}

master <- we_read_master()
primary <- we_primary_data(master)
if (nrow(primary) != 6992L || length(levels(primary[["point_id"]])) != 247L) {
  we_stop_check(
    "STOP — PRIMARY DOMAIN COUNT MISMATCH",
    "PRIMARY_INPUT_check",
    sprintf("n=%d sites=%d", nrow(primary), length(levels(primary[["point_id"]])))
  )
}

input_validation <- data.frame(
  master_path = normalizePath(WE_MASTER_PATH, winslash = "/", mustWork = TRUE),
  master_n = nrow(master),
  master_columns = ncol(master),
  primary_selector = "eligible_primary_model1 == TRUE",
  primary_n = nrow(primary),
  primary_sites = length(levels(primary[["point_id"]])),
  primary_years = paste(sort(unique(primary[["Year"]])), collapse = ";"),
  primary_months = paste(sort(unique(primary[["Month"]])), collapse = ";"),
  stringsAsFactors = FALSE
)
we_write_csv(input_validation, we_path("14_LOGS", "WE_PRIMARY_INPUT_validation.csv"))

support <- we_support_recheck(primary)
we_write_csv(support, we_path("09_PRIMARY_CONTRAST", "WE_PRIMARY_CONTRAST_SUPPORT_RECHECK.csv"))
if (!all(support[["matches_reference_support"]])) {
  we_stop_check("STOP — CONTRAST A/B LOST PRE-reference SUPPORT", "PRIMARY_SUPPORT_check")
}

ar_validation <- we_ar_sequence_validation(primary)
we_write_csv(ar_validation[["summary"]], we_path("06_AR_DIAGNOSTICS", "WE_PRIMARY_AR_SEQUENCE_validation.csv"))
we_write_csv(ar_validation[["gaps"]], we_path("06_AR_DIAGNOSTICS", "WE_PRIMARY_AR_GAPS.csv"))
we_write_csv(ar_validation[["gap_distribution"]], we_path("06_AR_DIAGNOSTICS", "WE_PRIMARY_AR_GAP_LENGTH_DISTRIBUTION.csv"))

we_health_check <- function(model, warnings, stage, primary_model = FALSE) {
  health <- we_model_health(model, warnings)
  reasons <- we_model_health_stop_reason(health)
  if (length(reasons)) {
    if ("MODEL NOT CONVERGED" %in% reasons && primary_model) {
      we_stop_check("STOP — PRIMARY MODEL 1 NOT CONVERGED", stage, paste(reasons, collapse = ";"))
    }
    if (any(grepl("NONFINITE", reasons))) {
      we_stop_check("STOP — NONFINITE MODEL ESTIMATE", stage, paste(reasons, collapse = ";"))
    }
    if ("RANK DEFICIENCY" %in% reasons) {
      we_stop_check("STOP — RANK DEFICIENCY", stage, paste(reasons, collapse = ";"))
    }
    if ("NUMERICAL IDENTIFIABILITY FAILURE" %in% reasons) {
      we_stop_check("STOP — IDENTIFIABILITY FAILURE", stage, paste(reasons, collapse = ";"))
    }
    we_stop_check("STOP — MODEL CONVERGENCE FAILURE", stage, paste(reasons, collapse = ";"))
  }
  health
}

we_fit_model1_cycle <- function(k_config, tag) {
  formula1 <- we_formula_model1(k_config)
  we_log("Fitting Model1 rho=0 cycle", tag)
  t0 <- proc.time()[["elapsed"]]
  rho0_fit <- we_fit_bam(formula1, primary, 0, primary[["AR.start"]])
  rho0_seconds <- proc.time()[["elapsed"]] - t0
  we_log(sprintf("Model1 rho=0 %s completed in %.2f seconds", tag, rho0_seconds))
  we_health_check(rho0_fit[["model"]], rho0_fit[["warnings"]], paste0("MODEL1_RHO0_", tag), TRUE)
  saveRDS(
    rho0_fit[["model"]],
    we_path("03_MODEL1_PRIMARY", paste0("WE_MODEL1_RHO0_", tag, ".rds"))
  )
  rho_validation <- we_rho_validation(primary, residuals(rho0_fit[["model"]], type = "response"))
  rho_validation[["summary"]][["cycle"]] <- tag
  rho_validation[["summary"]][["rho0_fit_seconds"]] <- rho0_seconds
  we_write_csv(
    rho_validation[["summary"]],
    we_path("06_AR_DIAGNOSTICS", paste0("WE_RHO_ESTIMATION_validation_", tag, ".csv"))
  )
  pair_counts <- as.data.frame(table(rho_validation[["pairs"]][["series_id"]]), stringsAsFactors = FALSE)
  names(pair_counts) <- c("point_id", "n_consecutive_pairs")
  we_write_csv(
    pair_counts,
    we_path("06_AR_DIAGNOSTICS", paste0("WE_RHO_PAIR_COUNTS_BY_SITE_", tag, ".csv"))
  )
  rho_used <- rho_validation[["summary"]][["rho_raw"]][1L]
  if (!is.finite(rho_used)) {
    we_stop_check("STOP — NONFINITE AR1 RHO", paste0("RHO_check_", tag))
  }
  if (we_rho_is_extreme(rho_used)) {
    we_stop_check("STOP — EXTREME AR1 RHO", paste0("RHO_check_", tag), sprintf("rho_raw=%.12g", rho_used))
  }

  we_log(sprintf("Fitting Model1 AR(1) cycle %s with fixed rho %.12g", tag, rho_used))
  t1 <- proc.time()[["elapsed"]]
  ar_fit <- we_fit_bam(formula1, primary, rho_used, primary[["AR.start"]])
  ar_seconds <- proc.time()[["elapsed"]] - t1
  we_log(sprintf("Model1 AR(1) %s completed in %.2f seconds", tag, ar_seconds))
  health <- we_health_check(ar_fit[["model"]], ar_fit[["warnings"]], paste0("MODEL1_AR1_", tag), TRUE)
  saveRDS(
    ar_fit[["model"]],
    we_path("03_MODEL1_PRIMARY", paste0("WE_MODEL1_AR1_", tag, ".rds"))
  )
  if (is.null(ar_fit[["model"]][["std.rsd"]])) {
    we_stop_check("STOP — MODEL1 STD.RSD NOT AVAILABLE", paste0("MODEL1_AR1_", tag))
  }
  post_pairs <- we_continuous_pairs(primary, as.numeric(ar_fit[["model"]][["std.rsd"]]))
  lag1 <- we_rho_from_pairs(post_pairs)
  post_counts <- table(post_pairs[["series_id"]])
  ar_flag <- if (abs(lag1) <= 0.10) "PASS" else if (abs(lag1) <= 0.20) "WARN_RETAIN_MODEL_NO_RHO_RETUNING" else "STOP"
  post_validation <- data.frame(
    cycle = tag,
    lag1_std_residual_cor = lag1,
    absolute_lag1_std_residual_cor = abs(lag1),
    n_consecutive_pairs = nrow(post_pairs),
    n_contributing_sites = length(post_counts),
    diagnostic_flag = ar_flag,
    rho_reestimated_after_fit = FALSE,
    ar_fit_seconds = ar_seconds,
    stringsAsFactors = FALSE
  )
  we_write_csv(
    post_validation,
    we_path("06_AR_DIAGNOSTICS", paste0("WE_AR1_POSTFIT_DIAGNOSTIC_", tag, ".csv"))
  )
  if (abs(lag1) > 0.20) {
    we_stop_check(
      "STOP — AR1 POSTFIT STANDARDIZED RESIDUAL CORRELATION",
      paste0("MODEL1_AR1_", tag),
      sprintf("lag1_std_residual_cor=%.12g", lag1)
    )
  }
  basis <- we_run_k_check(ar_fit[["model"]])
  basis[["cycle"]] <- tag
  we_write_csv(
    basis,
    we_path("07_BASIS_DIAGNOSTICS", paste0("WE_MODEL1_BASIS_DIAGNOSTIC_", tag, ".csv"))
  )
  list(
    formula = formula1,
    rho0_model = rho0_fit[["model"]],
    rho0_warnings = rho0_fit[["warnings"]],
    model = ar_fit[["model"]],
    warnings = ar_fit[["warnings"]],
    health = health,
    rho_validation = rho_validation,
    rho_used = rho_used,
    post_validation = post_validation,
    basis = basis
  )
}

model1_k_initial <- we_model1_k_initial()
model1_initial <- we_fit_model1_cycle(model1_k_initial, "INITIAL_K5")
fallback_decision <- we_apply_model1_single_fallback(model1_k_initial, model1_initial[["basis"]])
fallback_used <- length(fallback_decision[["changed"]]) > 0L

if (fallback_used) {
  we_log("Model1 legal single basis fallback:", paste(fallback_decision[["changed"]], collapse = ";"))
  model1_final <- we_fit_model1_cycle(fallback_decision[["config"]], "SINGLE_FALLBACK")
  if (any(model1_final[["basis"]][["basis_dimension_suspect"]])) {
    we_stop_check(
      "STOP — BASIS DIMENSION FAILURE AFTER SINGLE FALLBACK",
      "MODEL1_FINAL_BASIS_check",
      paste(model1_final[["basis"]][["smooth_name"]][model1_final[["basis"]][["basis_dimension_suspect"]]], collapse = ";")
    )
  }
  model1_k_final <- fallback_decision[["config"]]
} else {
  model1_final <- model1_initial
  model1_k_final <- model1_k_initial
}

we_write_csv(model1_final[["rho_validation"]][["summary"]], we_path("06_AR_DIAGNOSTICS", "WE_RHO_ESTIMATION_validation.csv"))
we_write_csv(model1_final[["post_validation"]], we_path("06_AR_DIAGNOSTICS", "WE_AR1_POSTFIT_DIAGNOSTIC.csv"))
we_write_csv(model1_final[["basis"]], we_path("07_BASIS_DIAGNOSTICS", "WE_MODEL1_BASIS_DIAGNOSTIC.csv"))

fallback_validation <- data.frame(
  term = c("hydro_energy_surface", WE_MODEL1_CONTROLS),
  initial_k = 5L,
  final_k = c(model1_k_final[["surface"]], unname(model1_k_final[["controls"]][WE_MODEL1_CONTROLS])),
  fallback_used = c(model1_k_final[["surface"]] == 7L, unname(model1_k_final[["controls"]][WE_MODEL1_CONTROLS]) == 7L),
  fallback_limit = "SINGLE_5_TO_7_ONLY",
  stringsAsFactors = FALSE
)
we_write_csv(fallback_validation, we_path("07_BASIS_DIAGNOSTICS", "WE_MODEL1_K_FINAL_CONFIGURATION.csv"))

model1 <- model1_final[["model"]]
rho_used <- model1_final[["rho_used"]]
we_save_model_outputs(model1, model1_final[["warnings"]], "MODEL1_PRIMARY", "03_MODEL1_PRIMARY", primary, rho_used)

model1_contrast <- we_contrast_invariance(model1, primary)
we_write_csv(
  transform(model1_contrast[["validation"]], model = "Model1 PRIMARY"),
  we_path("09_PRIMARY_CONTRAST", "WE_MODEL1_CONTRAST_INVARIANCE_validation.csv")
)
we_write_csv(
  data.frame(
    model = "Model1 PRIMARY",
    precip_rank_A = 0.75, radiation_rank_A = 0.25,
    precip_rank_B = 0.75, radiation_rank_B = 0.50,
    contrast_definition = "Prediction(A)-Prediction(B)",
    estimate = model1_contrast[["main_contrast"]],
    unit = "g C m^-2 month^-1",
    extraction = "predict(type='lpmatrix'); dX %*% coef(model)",
    invariance_max_absolute_difference = model1_contrast[["max_absolute_difference"]],
    invariance_status = if (model1_contrast[["pass"]]) "PASS" else "STOP",
    stringsAsFactors = FALSE
  ),
  we_path("09_PRIMARY_CONTRAST", "WE_MODEL1_PRIMARY_FIXED_CONTRAST.csv")
)
if (!model1_contrast[["pass"]]) {
  we_stop_check(
    "STOP — CONTRAST EXTRACTION NOT INVARIANT",
    "MODEL1_CONTRAST_check",
    sprintf("max_difference=%.12g", model1_contrast[["max_absolute_difference"]])
  )
}

model1_concurvity <- we_concurvity_tidy(model1)
model1_concurvity[["model"]] <- "Model1 PRIMARY"
we_write_csv(model1_concurvity, we_path("08_CONCURVITY", "WE_MODEL1_CONCURVITY.csv"))

raw_residual <- as.numeric(residuals(model1, type = "response"))
std_residual <- as.numeric(model1[["std.rsd"]])
raw_acf <- we_sequence_acf(primary, raw_residual, 12L)
std_acf <- we_sequence_acf(primary, std_residual, 12L)
we_write_csv(raw_acf, we_path("06_AR_DIAGNOSTICS", "WE_MODEL1_RESIDUAL_SEQUENCE_ACF.csv"))
we_write_csv(std_acf, we_path("06_AR_DIAGNOSTICS", "WE_MODEL1_STANDARDIZED_RESIDUAL_SEQUENCE_ACF.csv"))

we_plot_png <- function(filename, expression, width = 1800L, height = 1400L) {
  png(filename, width = width, height = height, res = 180)
  on.exit(dev.off(), add = TRUE)
  force(expression)
}

plot_dir <- we_path("03_MODEL1_PRIMARY")
we_plot_png(file.path(plot_dir, "WE_MODEL1_RESIDUAL_VS_FITTED.png"), {
  plot(fitted(model1), raw_residual, pch = 16, cex = 0.35, col = rgb(0, 0, 0, 0.25),
       xlab = "Fitted value", ylab = "Gaussian response residual", main = "Model 1 residual vs fitted")
  abline(h = 0, col = "red", lwd = 2)
  lines(lowess(fitted(model1), raw_residual), col = "blue", lwd = 2)
})
we_plot_png(file.path(plot_dir, "WE_MODEL1_QQ_PLOT.png"), {
  qqnorm(std_residual, pch = 16, cex = 0.35, col = rgb(0, 0, 0, 0.25), main = "Model 1 standardized residual QQ plot")
  qqline(std_residual, col = "red", lwd = 2)
})
we_plot_png(file.path(plot_dir, "WE_MODEL1_RESIDUAL_HISTOGRAM.png"), {
  hist(raw_residual, breaks = "FD", col = "grey80", border = "white",
       xlab = "Gaussian response residual", main = "Model 1 residual histogram")
  abline(v = 0, col = "red", lwd = 2)
})
we_plot_png(file.path(plot_dir, "WE_MODEL1_RESIDUAL_ACF.png"), {
  plot(raw_acf[["lag"]], raw_acf[["correlation"]], type = "h", lwd = 3,
       xlab = "Exact within-site calendar-month lag", ylab = "Correlation",
       main = "Model 1 residual ACF (valid within-site pairs only)", ylim = c(-1, 1))
  points(raw_acf[["lag"]], raw_acf[["correlation"]], pch = 16)
  abline(h = 0, col = "grey50")
})
we_plot_png(file.path(plot_dir, "WE_MODEL1_STANDARDIZED_RESIDUAL_ACF.png"), {
  plot(std_acf[["lag"]], std_acf[["correlation"]], type = "h", lwd = 3,
       xlab = "Exact within-site calendar-month lag", ylab = "Correlation",
       main = "Model 1 standardized residual ACF (valid within-site pairs only)", ylim = c(-1, 1))
  points(std_acf[["lag"]], std_acf[["correlation"]], pch = 16)
  abline(h = 0, col = "grey50")
})

surface_ref <- we_reference_row(primary)
p_seq <- seq(min(primary[["precip_rank_baseline"]]), max(primary[["precip_rank_baseline"]]), length.out = 80L)
r_seq <- seq(min(primary[["radiation_rank_baseline"]]), max(primary[["radiation_rank_baseline"]]), length.out = 80L)
surface_grid <- expand.grid(precip_rank_baseline = p_seq, radiation_rank_baseline = r_seq)
surface_newdata <- surface_ref[rep(1L, nrow(surface_grid)), , drop = FALSE]
surface_newdata[["precip_rank_baseline"]] <- surface_grid[["precip_rank_baseline"]]
surface_newdata[["radiation_rank_baseline"]] <- surface_grid[["radiation_rank_baseline"]]
surface_grid[["fitted_npp_anomaly_g"]] <- as.numeric(predict(model1, newdata = surface_newdata, type = "response"))
we_write_csv(surface_grid, we_path("03_MODEL1_PRIMARY", "WE_MODEL1_HYDRO_ENERGY_FITTED_SURFACE_GRID.csv"))
surface_z <- matrix(surface_grid[["fitted_npp_anomaly_g"]], nrow = length(p_seq), ncol = length(r_seq))
we_plot_png(file.path(plot_dir, "WE_MODEL1_HYDRO_ENERGY_FITTED_SURFACE.png"), {
  image(p_seq, r_seq, surface_z, col = hcl.colors(100, "RdBu", rev = TRUE),
        xlab = "Precipitation baseline rank", ylab = "Radiation baseline rank",
        main = "Model 1 fitted hydro-energy surface at fixed reference controls")
  contour(p_seq, r_seq, surface_z, add = TRUE, drawlabels = TRUE)
  points(c(0.75, 0.75), c(0.25, 0.50), pch = c(17, 19), cex = 1.5, col = "black")
  text(c(0.75, 0.75), c(0.25, 0.50), labels = c(" A", " B"), pos = 4)
})
we_plot_png(file.path(plot_dir, "WE_MODEL1_OBSERVED_SUPPORT_OVERLAY.png"), {
  plot(primary[["precip_rank_baseline"]], primary[["radiation_rank_baseline"]],
       pch = 16, cex = 0.32, col = rgb(0.1, 0.2, 0.5, 0.20),
       xlab = "Precipitation baseline rank", ylab = "Radiation baseline rank",
       main = "Primary observed support with reference contrast points")
  theta <- seq(0, 2 * pi, length.out = 400L)
  lines(0.75 + WE_SUPPORT_RADIUS * cos(theta), 0.25 + WE_SUPPORT_RADIUS * sin(theta), col = "darkgreen", lwd = 2)
  lines(0.75 + WE_SUPPORT_RADIUS * cos(theta), 0.50 + WE_SUPPORT_RADIUS * sin(theta), col = "darkorange", lwd = 2)
  points(c(0.75, 0.75), c(0.25, 0.50), pch = c(17, 19), cex = 1.8, col = c("darkgreen", "darkorange"))
  text(c(0.75, 0.75), c(0.25, 0.50), labels = c("A", "B"), pos = 4)
})

we_fit_secondary <- function(model_name, formula, model_dir) {
  we_log("Fitting", model_name, "with shared rho", format(rho_used, digits = 12))
  t0 <- proc.time()[["elapsed"]]
  fit <- we_fit_bam(formula, primary, rho_used, primary[["AR.start"]])
  seconds <- proc.time()[["elapsed"]] - t0
  we_log(sprintf("%s completed in %.2f seconds", model_name, seconds))
  health <- we_health_check(fit[["model"]], fit[["warnings"]], model_name, FALSE)
  list(model = fit[["model"]], warnings = fit[["warnings"]], health = health, seconds = seconds)
}

we_validation_secondary <- function(fit, model_name, model_dir) {
  we_save_model_outputs(fit[["model"]], fit[["warnings"]], model_name, model_dir, primary, rho_used)
  basis <- we_run_k_check(fit[["model"]])
  basis[["model"]] <- model_name
  we_write_csv(basis, we_path("07_BASIS_DIAGNOSTICS", paste0("WE_", model_name, "_BASIS_DIAGNOSTIC.csv")))
  inv <- we_contrast_invariance(fit[["model"]], primary)
  inv_validation <- inv[["validation"]]
  inv_validation[["model"]] <- model_name
  we_write_csv(inv_validation, we_path("09_PRIMARY_CONTRAST", paste0("WE_", model_name, "_CONTRAST_INVARIANCE_validation.csv")))
  if (!inv[["pass"]]) {
    we_stop_check(
      "STOP — CONTRAST EXTRACTION NOT INVARIANT",
      paste0(model_name, "_CONTRAST_check"),
      sprintf("max_difference=%.12g", inv[["max_absolute_difference"]])
    )
  }
  conc <- we_concurvity_tidy(fit[["model"]])
  conc[["model"]] <- model_name
  we_write_csv(conc, we_path("08_CONCURVITY", paste0("WE_", model_name, "_CONCURVITY.csv")))
  list(basis = basis, contrast = inv, concurvity = conc)
}

model0_fit <- we_fit_secondary("MODEL0", we_formula_model0(model1_k_final[["surface"]]), "02_MODEL0")
model0_validation <- we_validation_secondary(model0_fit, "MODEL0", "02_MODEL0")

model2_added_initial <- we_added_k_initial(WE_MODEL2_ADDED)
model2_fit_initial <- we_fit_secondary("MODEL2_INITIAL", we_formula_model2(model1_k_final, model2_added_initial), "04_MODEL2")
model2_basis_initial <- we_run_k_check(model2_fit_initial[["model"]])
we_write_csv(model2_basis_initial, we_path("07_BASIS_DIAGNOSTICS", "WE_MODEL2_BASIS_DIAGNOSTIC_INITIAL.csv"))
model2_fallback <- we_apply_added_single_fallback(model2_added_initial, WE_MODEL2_ADDED, model2_basis_initial)
if (length(model2_fallback[["changed"]])) {
  saveRDS(model2_fit_initial[["model"]], we_path("04_MODEL2", "WE_MODEL2_INITIAL_BEFORE_FALLBACK.rds"))
  we_log("Model2 legal single basis fallback:", paste(model2_fallback[["changed"]], collapse = ";"))
  model2_fit <- we_fit_secondary("MODEL2", we_formula_model2(model1_k_final, model2_fallback[["config"]]), "04_MODEL2")
  model2_added_final <- model2_fallback[["config"]]
} else {
  model2_fit <- model2_fit_initial
  model2_added_final <- model2_added_initial
}
model2_validation <- we_validation_secondary(model2_fit, "MODEL2", "04_MODEL2")
if (any(vapply(WE_MODEL2_ADDED, function(v) we_basis_suspect_for(model2_validation[["basis"]], v), logical(1)))) {
  we_stop_check(
    "STOP — BASIS DIMENSION FAILURE AFTER SINGLE FALLBACK",
    "MODEL2_FINAL_BASIS_check",
    "Model2 added smooth remains suspect"
  )
}

model3_added_initial <- we_added_k_initial(WE_MODEL3_ADDED)
model3_fit_initial <- we_fit_secondary(
  "MODEL3_INITIAL",
  we_formula_model3(model1_k_final, model2_added_final, model3_added_initial),
  "05_MODEL3"
)
model3_basis_initial <- we_run_k_check(model3_fit_initial[["model"]])
we_write_csv(model3_basis_initial, we_path("07_BASIS_DIAGNOSTICS", "WE_MODEL3_BASIS_DIAGNOSTIC_INITIAL.csv"))
model3_fallback <- we_apply_added_single_fallback(model3_added_initial, WE_MODEL3_ADDED, model3_basis_initial)
if (length(model3_fallback[["changed"]])) {
  saveRDS(model3_fit_initial[["model"]], we_path("05_MODEL3", "WE_MODEL3_INITIAL_BEFORE_FALLBACK.rds"))
  we_log("Model3 legal single basis fallback:", paste(model3_fallback[["changed"]], collapse = ";"))
  model3_fit <- we_fit_secondary(
    "MODEL3",
    we_formula_model3(model1_k_final, model2_added_final, model3_fallback[["config"]]),
    "05_MODEL3"
  )
  model3_added_final <- model3_fallback[["config"]]
} else {
  model3_fit <- model3_fit_initial
  model3_added_final <- model3_added_initial
}
model3_validation <- we_validation_secondary(model3_fit, "MODEL3", "05_MODEL3")
if (any(vapply(WE_MODEL3_ADDED, function(v) we_basis_suspect_for(model3_validation[["basis"]], v), logical(1)))) {
  we_stop_check(
    "STOP — BASIS DIMENSION FAILURE AFTER SINGLE FALLBACK",
    "MODEL3_FINAL_BASIS_check",
    "Model3 added smooth remains suspect"
  )
}

model2_k_validation <- data.frame(
  term = WE_MODEL2_ADDED,
  initial_k = 5L,
  final_k = unname(model2_added_final[WE_MODEL2_ADDED]),
  fallback_used = unname(model2_added_final[WE_MODEL2_ADDED]) == 7L,
  fallback_limit = "SINGLE_5_TO_7_ONLY",
  stringsAsFactors = FALSE
)
model3_k_validation <- data.frame(
  term = WE_MODEL3_ADDED,
  initial_k = 5L,
  final_k = unname(model3_added_final[WE_MODEL3_ADDED]),
  fallback_used = unname(model3_added_final[WE_MODEL3_ADDED]) == 7L,
  fallback_limit = "SINGLE_5_TO_7_ONLY",
  stringsAsFactors = FALSE
)
we_write_csv(model2_k_validation, we_path("07_BASIS_DIAGNOSTICS", "WE_MODEL2_K_FINAL_CONFIGURATION.csv"))
we_write_csv(model3_k_validation, we_path("07_BASIS_DIAGNOSTICS", "WE_MODEL3_K_FINAL_CONFIGURATION.csv"))

contrast_rows <- data.frame(
  model = c("Model0", "Model1 PRIMARY", "Model2", "Model3"),
  contrast_estimate = c(
    model0_validation[["contrast"]][["main_contrast"]],
    model1_contrast[["main_contrast"]],
    model2_validation[["contrast"]][["main_contrast"]],
    model3_validation[["contrast"]][["main_contrast"]]
  ),
  direction = we_direction(c(
    model0_validation[["contrast"]][["main_contrast"]],
    model1_contrast[["main_contrast"]],
    model2_validation[["contrast"]][["main_contrast"]],
    model3_validation[["contrast"]][["main_contrast"]]
  )),
  n = nrow(primary),
  n_sites = length(levels(primary[["point_id"]])),
  rho_used = rho_used,
  diagnostics_flag = c(
    if (any(model0_validation[["basis"]][["basis_dimension_suspect"]])) "BASIS_CAUTION_MODEL0_NO_FALLBACK_specified" else "PASS",
    if (model1_final[["post_validation"]][["diagnostic_flag"]] == "WARN_RETAIN_MODEL_NO_RHO_RETUNING") "AR_WARN_RETAINED" else "PASS",
    if (any(model2_validation[["basis"]][["basis_dimension_suspect"]])) "BASIS_CAUTION_EXISTING_TERM" else "PASS",
    if (any(model3_validation[["basis"]][["basis_dimension_suspect"]])) "BASIS_CAUTION_EXISTING_TERM" else "PASS"
  ),
  unit = "g C m^-2 month^-1",
  result_recorded = FALSE,
  stringsAsFactors = FALSE
)
we_write_csv(contrast_rows, we_path("09_PRIMARY_CONTRAST", "WE_MODEL_CONTRAST_COMPARISON.csv"))

all_concurvity <- rbind(
  transform(model0_validation[["concurvity"]], model = "Model0"),
  model1_concurvity,
  transform(model2_validation[["concurvity"]], model = "Model2"),
  transform(model3_validation[["concurvity"]], model = "Model3")
)
highest <- do.call(rbind, lapply(
  split(all_concurvity, all_concurvity[["model"]]),
  function(x) we_highest_concurvity(x, unique(x[["model"]]))
))
we_write_csv(highest, we_path("08_CONCURVITY", "WE_HIGHEST_CONCURVITY_BY_MODEL.csv"))

gaussian_check <- data.frame(
  model = "Model1 PRIMARY",
  structural_finite_check = "PASS",
  residual_validation_status = "PENDING_FINITE_VALUE_CHECK",
  stringsAsFactors = FALSE
)
we_write_csv(gaussian_check, we_path("03_MODEL1_PRIMARY", "WE_MODEL1_GAUSSIAN_RESIDUAL_check.csv"))

stage1_state <- list(
  completed_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
  rho_used = rho_used,
  rho_raw = rho_used,
  model1_k_final = model1_k_final,
  model2_added_k_final = model2_added_final,
  model3_added_k_final = model3_added_final,
  model1_formula = formula(model1),
  model1_contrast = model1_contrast[["main_contrast"]],
  model0_contrast = model0_validation[["contrast"]][["main_contrast"]],
  model2_contrast = model2_validation[["contrast"]][["main_contrast"]],
  model3_contrast = model3_validation[["contrast"]][["main_contrast"]],
  lag1_std_residual_cor = model1_final[["post_validation"]][["lag1_std_residual_cor"]],
  ar_sections = ar_validation[["summary"]][["n_AR_sections"]],
  bootstrap_structural_checks_pass = TRUE,
  gaussian_finite_check = "PENDING"
)
saveRDS(stage1_state, we_path("14_LOGS", "WE_STAGE1_STATE.rds"))
we_write_csv(
  data.frame(
    stage = "MODEL0_TO_MODEL3_AND_PRIMARY_checkS",
    status = "PASS_PENDING_FINITE_VALUE_CHECK",
    detail = "Model fitting complete",
    stringsAsFactors = FALSE
  ),
  we_path("14_LOGS", "WE_STAGE1_check_SUMMARY.csv")
)
we_log("water-energy GAMM primary analysis stage 1 completed; bootstrap waits for Gaussian visual validation")
