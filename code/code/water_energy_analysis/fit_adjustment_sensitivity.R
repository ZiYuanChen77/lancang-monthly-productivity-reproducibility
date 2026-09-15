source(file.path(Sys.getenv("WATER_ENERGY_CODE_DIR", unset = getwd()), "adjustment_common.R"))
suppressPackageStartupMessages(library(mgcv))

we_log("water-energy Model2/Model3 canonical-model diagnostic stage started")
prepare_path <- we_path("09_LOGS", "WE_ADJUSTMENT_prepare_STATE.rds")
if (!file.exists(prepare_path)) {
  we_stop_check("STOP — ADJUSTMENT prepare NOT AVAILABLE", "DIAGNOSTIC_ENTRY_check")
}
prepare <- readRDS(prepare_path)
if (!isTRUE(prepare$prepare_complete)) {
  we_stop_check("STOP — ADJUSTMENT prepare NOT PASSED", "DIAGNOSTIC_ENTRY_check")
}

master <- we_read_master()
data <- we_prepare_primary_adjustment_data(master)
support <- we_support_validation_adjustment(data)
support_pass <- all(support$status == "PASS")
canonical <- we_canonical_contrast_table()
results <- list()

for (model_name in WE_MODEL_NAMES) {
  we_log("validationing canonical", model_name, "RDS without full-data refit")
  model <- readRDS(we_model_path(model_name))
  model_dir <- if (model_name == "MODEL2") "02_MODEL2" else "03_MODEL3"
  canonical_label <- if (model_name == "MODEL2") "Model2" else "Model3"
  canonical_contrast <- canonical$contrast_estimate[canonical$model == canonical_label]

  health <- we_model_health(model, character())
  canonical_info <- read.csv(we_model_info_path(model_name), stringsAsFactors = FALSE)
  reasons <- we_model_health_stop_reason(health)
  if (length(reasons) || canonical_info$convergence[1L] != "CONVERGED" ||
      !as.logical(canonical_info$full_rank[1L]) || canonical_info$warning_count[1L] != 0L) {
    we_stop_check(
      "STOP — canonical MODEL2/MODEL3 NUMERICAL HEALTH FAILURE", "MODEL_HEALTH_check",
      paste(model_name, paste(reasons, collapse = ";"))
    )
  }
  we_write_csv(
    cbind(
      data.frame(
        model = model_name, canonical_primary_object = TRUE,
        full_data_refitted = FALSE, source_RDS = normalizePath(
          we_model_path(model_name), winslash = "/"
        ), stringsAsFactors = FALSE
      ),
      health
    ),
    we_path(model_dir, paste0("WE_", model_name, "_canonical_MODEL_HEALTH.csv"))
  )
  we_write_csv(
    we_smooth_edf(model),
    we_path(model_dir, paste0("WE_", model_name, "_SMOOTH_EDF.csv"))
  )

  post_ar <- we_ar_postfit_adjustment(model, data, model_name)
  we_write_csv(
    post_ar,
    we_path("05_DIAGNOSTICS", paste0("WE_", model_name, "_AR_POSTFIT_DIAGNOSTIC.csv"))
  )
  if (!is.finite(post_ar$absolute_standardized_residual_lag1) ||
      post_ar$absolute_standardized_residual_lag1 > 0.20) {
    we_stop_check(
      "STOP — MODEL2/3 AR POSTFIT FAILURE", "AR_POSTFIT_check",
      sprintf("%s standardized_residual_lag1=%.15g", model_name, post_ar$standardized_residual_lag1)
    )
  }

  we_log("Running reference k.check for", model_name, "seed=20260803 n.rep=1000")
  basis <- we_run_k_check(model)
  basis$model <- model_name
  basis$surface_k <- "c(5,5)"
  basis$all_1D_smooth_k <- 5L
  basis$k_fallback_permitted <- FALSE
  basis$k_fallback_used <- FALSE
  basis$basis_status <- ifelse(
    basis$basis_dimension_suspect, "BASIS_DIMENSION_SUSPECT", "PASS"
  )
  we_write_csv(
    basis,
    we_path("05_DIAGNOSTICS", paste0("WE_", model_name, "_BASIS_DIAGNOSTIC.csv"))
  )
  if (any(basis$basis_dimension_suspect)) {
    we_stop_check(
      "STOP — MODEL2/3 BASIS FAILURE", "BASIS_check",
      paste(model_name, paste(basis$smooth_name[basis$basis_dimension_suspect], collapse = ";"))
    )
  }

  invariance <- we_adjustment_invariance(model, data, model_name)
  we_write_csv(
    invariance$validation,
    we_path(model_dir, paste0("WE_", model_name, "_CONTRAST_INVARIANCE_validation.csv"))
  )
  contrast_difference <- invariance$main_contrast - canonical_contrast
  contrast_summary <- data.frame(
    model = model_name,
    precip_rank_A = 0.75, radiation_rank_A = 0.25,
    precip_rank_B = 0.75, radiation_rank_B = 0.50,
    contrast_definition = "Prediction(A)-Prediction(B)",
    full_data_contrast = invariance$main_contrast,
    canonical_primary_contrast = canonical_contrast,
    difference_vs_primary = contrast_difference,
    absolute_difference_vs_primary = abs(contrast_difference),
    unit = "g C m^-2 month^-1",
    extraction = "predict(type='lpmatrix'); dX %*% coef(model)",
    invariance_max_absolute_difference = invariance$max_absolute_difference,
    invariance_status = if (invariance$pass) "PASS" else "STOP",
    full_data_refitted = FALSE,
    stringsAsFactors = FALSE
  )
  we_write_csv(
    contrast_summary,
    we_path(model_dir, paste0("WE_", model_name, "_FIXED_CONTRAST.csv"))
  )
  if (!is.finite(invariance$main_contrast)) {
    we_stop_check("STOP — NONFINITE CONTRAST", "CONTRAST_check", model_name)
  }
  if (!invariance$pass) {
    we_stop_check(
      "STOP — MODEL2/3 CONTRAST NOT INVARIANT", "CONTRAST_check",
      sprintf("%s max_difference=%.15g", model_name, invariance$max_absolute_difference)
    )
  }
  if (abs(contrast_difference) > 1e-8) {
    we_stop_check(
      "STOP — FULL-DATA MODEL2/3 REPRODUCTION MISMATCH", "CONTRAST_REFERENCE_check",
      sprintf("%s difference=%.15g", model_name, contrast_difference)
    )
  }
  if (invariance$main_contrast > 0) {
    we_stop_check(
      paste0(model_name, "_SIGN_REVERSAL_ANALYSIS_CHECK_REQUIRED"),
      "DIRECTION_check", sprintf("contrast=%.15g", invariance$main_contrast)
    )
  }

  we_log("Running full and pairwise concurvity for", model_name)
  concurvity <- we_concurvity_tidy(model)
  concurvity$model <- model_name
  concurvity_summary <- we_concurvity_summary_adjustment(concurvity, model_name)
  surface_control <- we_surface_control_concurvity(concurvity, model_name)
  we_write_csv(
    concurvity,
    we_path("05_DIAGNOSTICS", paste0("WE_", model_name, "_CONCURVITY.csv"))
  )
  we_write_csv(
    concurvity_summary,
    we_path("05_DIAGNOSTICS", paste0("WE_", model_name, "_CONCURVITY_SUMMARY.csv"))
  )
  we_write_csv(
    surface_control,
    we_path("05_DIAGNOSTICS", paste0("WE_", model_name, "_SURFACE_CONTROL_CONCURVITY.csv"))
  )

  vectors <- we_model_residual_vectors(model, data)
  raw_acf <- we_sequence_acf(data, vectors$raw, 12L, "point_id")
  std_acf <- we_sequence_acf(data, vectors$std, 12L, "point_id")
  residual_diagnostic <- we_residual_diagnostics(model)
  residual_diagnostic$model <- model_name
  residual_diagnostic$heavy_tail_recorded <- residual_diagnostic$residual_excess_kurtosis > 0
  residual_diagnostic$family_changed <- FALSE
  residual_diagnostic$structural_numerical_failure <- FALSE
  we_write_csv(
    residual_diagnostic,
    we_path("05_DIAGNOSTICS", paste0("WE_", model_name, "_GAUSSIAN_RESIDUAL_DIAGNOSTICS.csv"))
  )
  we_write_csv(
    raw_acf,
    we_path("05_DIAGNOSTICS", paste0("WE_", model_name, "_RESIDUAL_SEQUENCE_ACF.csv"))
  )
  we_write_csv(
    std_acf,
    we_path("05_DIAGNOSTICS", paste0("WE_", model_name, "_STANDARDIZED_RESIDUAL_SEQUENCE_ACF.csv"))
  )

  prefix <- we_path("05_DIAGNOSTICS", paste0("WE_", model_name))
  we_plot_png(paste0(prefix, "_RESIDUAL_VS_FITTED.png"), {
    plot(
      vectors$fitted, vectors$raw, pch = 16, cex = 0.35, col = rgb(0, 0, 0, 0.23),
      xlab = "Fitted value", ylab = "Gaussian response residual",
      main = paste(model_name, "residual vs fitted")
    )
    abline(h = 0, col = "red", lwd = 2)
    lines(lowess(vectors$fitted, vectors$raw), col = "blue", lwd = 2)
  })
  we_plot_png(paste0(prefix, "_QQ_PLOT.png"), {
    qqnorm(
      vectors$std, pch = 16, cex = 0.35, col = rgb(0, 0, 0, 0.23),
      main = paste(model_name, "standardized residual QQ plot")
    )
    qqline(vectors$std, col = "red", lwd = 2)
  })
  we_plot_png(paste0(prefix, "_RESIDUAL_HISTOGRAM.png"), {
    hist(
      vectors$raw, breaks = "FD", col = "grey80", border = "white",
      xlab = "Gaussian response residual", main = paste(model_name, "residual histogram")
    )
    abline(v = 0, col = "red", lwd = 2)
  })
  we_plot_png(paste0(prefix, "_STANDARDIZED_RESIDUAL_ACF.png"), {
    plot(
      std_acf$lag, std_acf$correlation, type = "h", lwd = 3, ylim = c(-1, 1),
      xlab = "Exact within-site calendar-month lag", ylab = "Correlation",
      main = paste(model_name, "standardized residual ACF")
    )
    points(std_acf$lag, std_acf$correlation, pch = 16)
    abline(h = 0, col = "grey50")
  })

  results[[model_name]] <- list(
    model = model_name, n = nrow(data), n_sites = nlevels(data$point_id),
    ar_sections = prepare$ar$n_AR_sections,
    formula = we_formula_text(formula(model)), rho_used = WE_RHO_FIXED,
    source_model_path = we_model_path(model_name), full_data_refitted = FALSE,
    health = health, post_ar = post_ar, basis = basis,
    contrast = invariance$main_contrast, canonical_contrast = canonical_contrast,
    contrast_difference = contrast_difference,
    contrast_invariance_max = invariance$max_absolute_difference,
    support = support, support_pass = support_pass,
    concurvity_summary = concurvity_summary,
    surface_control_concurvity = surface_control,
    residual_diagnostic = residual_diagnostic,
    diagnostic_complete = TRUE, bootstrap_specified = TRUE
  )
  we_log(sprintf(
    "%s diagnostics PASS contrast=%.15g diff_vs_primary=%.3g lag1=%.15g surface_concurvity=%.15g",
    model_name, invariance$main_contrast, contrast_difference,
    post_ar$standardized_residual_lag1, concurvity_summary$surface_full_worst
  ))
}

state <- list(
  diagnostics_complete = TRUE,
  completed_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
  model_results = results, support = support,
  full_data_refitted = FALSE, Model1_refitted = FALSE,
  bootstrap_complete = c(MODEL2 = FALSE, MODEL3 = FALSE)
)
saveRDS(state, we_path("09_LOGS", "WE_ADJUSTMENT_STAGE1_STATE.rds"))
we_log("Model2/Model3 canonical-model diagnostic stage PASS; both bootstraps specified")
