source(file.path(Sys.getenv("WATER_ENERGY_CODE_DIR", unset = getwd()), "domain_common.R"))
suppressPackageStartupMessages(library(mgcv))

we_log("water-energy domain sensitivity stage 1 started")
we_write_csv(we_software_validation(), we_path("09_LOGS", "WE_R_MGCV_ENVIRONMENT_validation.csv"))
we_write_lines(capture.output(sessionInfo()), we_path("09_LOGS", "WE_R_SESSION_INFO.txt"))
if (!requireNamespace("mgcv", quietly = TRUE)) {
  we_stop_check("STOP — MGCV NOT AVAILABLE", "ENVIRONMENT_check")
}
if (as.character(packageVersion("mgcv")) != "1.9.4") {
  we_stop_check("STOP — PRIMARY REFERENCE MISMATCH", "ENVIRONMENT_check",
                 paste("mgcv", as.character(packageVersion("mgcv"))))
}

test_path <- we_path("08_TESTS", "WE_DOMAIN_SENSITIVITY_TEST_RESULTS.csv")
if (!file.exists(test_path)) {
  we_stop_check("STOP — AUTOMATED TESTS NOT AVAILABLE", "PRE_MODEL_TEST_check")
}
tests <- read.csv(test_path, stringsAsFactors = FALSE)
if (!nrow(tests) || !all(tests$status == "PASS")) {
  we_stop_check("STOP — AUTOMATED TEST FAILURE", "PRE_MODEL_TEST_check")
}

reference_validation <- we_reference_reference_validation()
we_write_csv(reference_validation, we_path("09_LOGS", "WE_reference_REFERENCE_validation.csv"))
if (!all(reference_validation$status == "PASS")) {
  we_stop_check(
    "STOP — PRIMARY REFERENCE MISMATCH", "reference_REFERENCE_check",
    paste(reference_validation$check[reference_validation$status != "PASS"], collapse = ";")
  )
}

master <- we_read_master()
input_validation <- data.frame(
  master_path = normalizePath(WE_MASTER_PATH, winslash = "/", mustWork = TRUE),
  master_n = nrow(master), master_columns = ncol(master),
  primary_v2_path = normalizePath(WE_PRIMARY_ROOT, winslash = "/", mustWork = TRUE),
  target_robustness_path = normalizePath(WE_TARGET_ROBUSTNESS_ROOT, winslash = "/", mustWork = TRUE),
  primary_reference_refitted = FALSE, target_robustness_refitted = FALSE,
  network_download = FALSE, analysis_definition_changed = FALSE, output_definition_changed = FALSE,
  output_root = normalizePath(WE_OUTPUT_ROOT, winslash = "/", mustWork = TRUE),
  stringsAsFactors = FALSE
)
we_write_csv(input_validation, we_path("09_LOGS", "WE_INPUT_AND_IMMUTABILITY_validation.csv"))

flag_validations <- lapply(WE_ALLOWED_ANALYSES, function(a) we_domain_flag_validation(master, a))
names(flag_validations) <- WE_ALLOWED_ANALYSES
flag_summary <- do.call(rbind, lapply(flag_validations, `[[`, "summary"))
flag_detail <- do.call(rbind, lapply(flag_validations, function(x) x$detail))
we_write_csv(flag_summary, we_path("04_SUPPORT", "WE_DOMAIN_FLAG_IDENTITY_validation.csv"))
we_write_csv(flag_detail, we_path("04_SUPPORT", "WE_DOMAIN_FLAG_MISMATCH_KEY_DETAIL.csv"))
for (a in WE_ALLOWED_ANALYSES) {
  meta <- WE_DOMAIN_META[[a]]
  we_write_csv(flag_validations[[a]]$summary,
                 we_path(meta$output_dir, paste0("WE_", a, "_DOMAIN_FLAG_validation.csv")))
  we_write_csv(flag_validations[[a]]$detail,
                 we_path(meta$output_dir, paste0("WE_", a, "_DOMAIN_FLAG_MISMATCH_KEYS.csv")))
}
if (any(flag_summary$substantive_semantic_conflict)) {
  failing <- flag_summary$analysis[flag_summary$substantive_semantic_conflict]
  reason <- if ("FULL_YEAR" %in% failing)
    "STOP — FULL-YEAR DOMAIN FLAG SEMANTICS REQUIRE REVIEW" else
    "STOP — APR-SEP DOMAIN FLAG SEMANTICS REQUIRE REVIEW"
  we_stop_check(reason, "DOMAIN_FLAG_SEMANTICS_check", paste(failing, collapse = ";"))
}

data_list <- list(
  FULL_YEAR = we_prepare_domain(master, "FULL_YEAR"),
  APR_SEP = we_prepare_domain(master, "APR_SEP")
)
primary_selector <- we_flag_true(master$eligible_primary_model1)
primary <- master[primary_selector, , drop = FALSE]
primary$point_id <- sprintf("%03d", as.integer(primary$point_id))
primary <- primary[order(primary$point_id, primary$calendar_month_index), , drop = FALSE]
primary$point_id <- factor(primary$point_id, levels = sort(unique(primary$point_id)))
primary$AR.start <- we_build_ar_start(primary, "point_id")
if (nrow(primary) != 6992L || length(levels(primary$point_id)) != 247L) {
  we_stop_check("STOP — PRIMARY REFERENCE MISMATCH", "MASTER_PRIMARY_REFERENCE_check",
                 sprintf("master n=%d sites=%d", nrow(primary), length(levels(primary$point_id))))
}

sample_summary <- do.call(rbind, lapply(names(data_list), function(a) {
  d <- data_list[[a]]
  data.frame(
    analysis = a, domain_definition = WE_DOMAIN_META[[a]]$definition,
    response = "npp_anomaly_mean_g", n = nrow(d),
    n_sites = length(levels(d$point_id)), n_years = length(unique(d$Year)),
    n_months = length(unique(d$Month)), months = paste(sort(unique(d$Month)), collapse = ";"),
    historically_active_filter_used = FALSE, eligible_primary_model1_filter_used = FALSE,
    response_recalculated = FALSE, ranks_recalculated = FALSE,
    antecedent_controls_recalculated = FALSE, rho_used = WE_RHO_FIXED,
    surface_k = "c(5,5)", controls_k = 5L, k_fallback_allowed = FALSE,
    stringsAsFactors = FALSE
  )
}))
we_write_csv(sample_summary, we_path("07_COMPARISON", "WE_DOMAIN_SAMPLE_SUMMARY.csv"))

selector_list <- list(
  `Primary active` = primary_selector,
  `Full-year` = flag_validations$FULL_YEAR$explicit,
  `Apr-Sep` = flag_validations$APR_SEP$explicit
)
composition <- do.call(rbind, lapply(names(selector_list), function(label) {
  keep <- selector_list[[label]]
  active <- we_flag_true(master$historically_active_canonical)
  base <- master$baseline_median_npp_g
  data.frame(
    analysis = label, records = sum(keep),
    sites = length(unique(master$point_id[keep])), years = length(unique(master$Year[keep])),
    months_represented = paste(sort(unique(master$Month[keep])), collapse = ";"),
    n_months = length(unique(master$Month[keep])),
    historically_active_TRUE = sum(keep & active),
    historically_active_FALSE = sum(keep & !active),
    historically_active_fraction = mean(active[keep]),
    zero_historical_median_NPP = sum(keep & is.finite(base) & base == 0),
    nonzero_historical_median_NPP = sum(keep & is.finite(base) & base != 0),
    missing_historical_median_NPP = sum(keep & !is.finite(base)),
    intersection_with_primary = sum(keep & primary_selector),
    records_not_in_primary = sum(keep & !primary_selector),
    primary_records_not_in_domain = sum(primary_selector & !keep),
    stringsAsFactors = FALSE
  )
}))
we_write_csv(composition, we_path("04_SUPPORT", "WE_DOMAIN_COMPOSITION_validation.csv"))

active <- we_flag_true(master$historically_active_canonical)
month_composition <- do.call(rbind, lapply(1:12, function(m) {
  in_month <- master$Year %in% 2022:2025 & master$Month == m
  full_keep <- flag_validations$FULL_YEAR$explicit & master$Month == m
  apr_keep <- flag_validations$APR_SEP$explicit & master$Month == m
  data.frame(
    Month = m, n_full_year = sum(full_keep), n_apr_sep = sum(apr_keep),
    n_primary = sum(primary_selector & master$Month == m),
    n_historically_active = sum(full_keep & active),
    n_historically_inactive = sum(full_keep & !active),
    raw_scope_records = sum(in_month), stringsAsFactors = FALSE
  )
}))
we_write_csv(month_composition, we_path("04_SUPPORT", "WE_DOMAIN_MONTH_LEVEL_COMPOSITION.csv"))

support <- do.call(rbind, lapply(names(data_list), function(a) we_support_validation(data_list[[a]], a)))
we_write_csv(support, we_path("04_SUPPORT", "WE_DOMAIN_FIXED_CONTRAST_SUPPORT_validation.csv"))
for (a in names(data_list)) {
  we_write_csv(
    support[support$analysis == a, , drop = FALSE],
    we_path(WE_DOMAIN_META[[a]]$output_dir,
              paste0("WE_", a, "_FIXED_CONTRAST_SUPPORT_validation.csv"))
  )
}
support_pass <- setNames(vapply(names(data_list), function(a) {
  all(support$status[support$analysis == a] == "PASS")
}, logical(1)), names(data_list))

support_data <- c(list(PRIMARY = primary), data_list)
support_composition <- do.call(rbind, lapply(names(support_data), function(a) {
  d <- support_data[[a]]
  do.call(rbind, lapply(list(A = WE_CONTRAST_A, B = WE_CONTRAST_B), function(target) {
    distance <- sqrt(
      (d$precip_rank_baseline - target[["precip_rank_baseline"]])^2 +
        (d$radiation_rank_baseline - target[["radiation_rank_baseline"]])^2
    )
    keep <- is.finite(distance) & distance <= WE_SUPPORT_RADIUS
    active_d <- we_flag_true(d$historically_active_canonical)
    data.frame(
      analysis = a,
      contrast_point = if (target[["radiation_rank_baseline"]] == 0.25) "A" else "B",
      radius = WE_SUPPORT_RADIUS, records = sum(keep),
      sites = length(unique(as.character(d$point_id[keep]))),
      years = length(unique(d$Year[keep])),
      historically_active_records = sum(keep & active_d),
      historically_inactive_records = sum(keep & !active_d),
      historically_active_fraction = if (sum(keep)) mean(active_d[keep]) else NA_real_,
      stringsAsFactors = FALSE
    )
  }))
}))
we_write_csv(support_composition,
               we_path("04_SUPPORT", "WE_FIXED_CONTRAST_SUPPORT_COMPOSITION.csv"))

ar_validations <- list()
for (a in names(data_list)) {
  ar <- we_ar_sequence_validation(data_list[[a]], "point_id")
  ar$summary$analysis <- a
  ar_validations[[a]] <- ar
  out_dir <- WE_DOMAIN_META[[a]]$output_dir
  we_write_csv(ar$summary, we_path(out_dir, paste0("WE_", a, "_AR_SEQUENCE_validation.csv")))
  we_write_csv(ar$gaps, we_path(out_dir, paste0("WE_", a, "_AR_GAPS.csv")))
  we_write_csv(ar$gap_distribution,
                 we_path(out_dir, paste0("WE_", a, "_AR_GAP_LENGTH_DISTRIBUTION.csv")))
}
we_write_csv(
  do.call(rbind, lapply(ar_validations, `[[`, "summary")),
  we_path("06_DIAGNOSTICS", "WE_DOMAIN_AR_SEQUENCE_SUMMARY.csv")
)

we_domain_health_check <- function(model, warnings, analysis) {
  health <- we_model_health(model, warnings)
  reasons <- we_model_health_stop_reason(health)
  if (length(reasons)) {
    if (any(grepl("NONFINITE", reasons))) {
      we_stop_check("STOP — NONFINITE DOMAIN MODEL ESTIMATE", paste0(analysis, "_MODEL_HEALTH"),
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
    we_stop_check("STOP — DOMAIN MODEL CONVERGENCE FAILURE", paste0(analysis, "_MODEL_HEALTH"),
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
for (a in c("FULL_YEAR", "APR_SEP")) {
  d <- data_list[[a]]
  meta <- WE_DOMAIN_META[[a]]
  if (!support_pass[[a]]) {
    we_log(a, "fixed contrast support did not pass; model diagnostics proceed but bootstrap will not run")
  }
  form <- we_formula_domain("point_id")
  we_log("Fitting", a, "fixed Primary Model 1 structure; n=", nrow(d),
           "sites=", length(levels(d$point_id)), "rho=", format(WE_RHO_FIXED, digits = 16))
  started <- proc.time()[["elapsed"]]
  fit <- we_fit_bam(form, d, WE_RHO_FIXED, d$AR.start)
  fit_seconds <- proc.time()[["elapsed"]] - started
  health <- we_domain_health_check(fit$model, fit$warnings, a)
  model_name <- paste0(a, "_MODEL1")
  we_save_model_outputs(fit$model, fit$warnings, model_name, meta$output_dir,
                          d, WE_RHO_FIXED)

  post_ar <- we_ar_postfit_validation(fit$model, d, a)
  we_write_csv(post_ar,
                 we_path(meta$output_dir, paste0("WE_", a, "_AR_POSTFIT_DIAGNOSTIC.csv")))
  if (post_ar$absolute_standardized_residual_lag1 > 0.20) {
    we_stop_check(
      "STOP — DOMAIN AR1 POSTFIT STANDARDIZED RESIDUAL CORRELATION",
      paste0(a, "_AR_POSTFIT_check"),
      sprintf("standardized_residual_lag1=%.15g", post_ar$standardized_residual_lag1)
    )
  }

  basis <- we_run_k_check(fit$model)
  basis$analysis <- a
  basis$surface_k <- "c(5,5)"
  basis$controls_k <- 5L
  basis$k_fallback_permitted <- FALSE
  we_write_csv(basis,
                 we_path(meta$output_dir, paste0("WE_", a, "_BASIS_DIAGNOSTIC.csv")))
  if (any(basis$basis_dimension_suspect)) {
    we_stop_check(
      "STOP — DOMAIN SENSITIVITY BASIS FAILURE", paste0(a, "_BASIS_check"),
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
  we_write_csv(contrast_summary,
                 we_path(meta$output_dir, paste0("WE_", a, "_FIXED_CONTRAST.csv")))
  if (!is.finite(contrast$main_contrast)) {
    we_stop_check("STOP — NONFINITE DOMAIN CONTRAST", paste0(a, "_CONTRAST_check"))
  }
  if (!contrast$pass) {
    we_stop_check("STOP — CONTRAST EXTRACTION FAILURE", paste0(a, "_CONTRAST_check"),
                   sprintf("max_difference=%.15g", contrast$max_absolute_difference))
  }

  concurvity <- we_concurvity_tidy(fit$model)
  concurvity$analysis <- a
  hydro <- we_hydro_concurvity_summary(concurvity, a)
  highest <- we_highest_concurvity(concurvity, a)
  we_write_csv(concurvity, we_path(meta$output_dir, paste0("WE_", a, "_CONCURVITY.csv")))
  we_write_csv(hydro,
                 we_path(meta$output_dir, paste0("WE_", a, "_HYDRO_ENERGY_CONCURVITY.csv")))
  we_write_csv(highest,
                 we_path(meta$output_dir, paste0("WE_", a, "_HIGHEST_CONCURVITY.csv")))

  residual_diag <- we_residual_diagnostics(fit$model)
  residual_diag$analysis <- a
  residual_diag$gaussian_family_changed <- FALSE
  residual_diag$diagnostic_status <- "FINITE_MODEL_DIAGNOSTICS_RECORDED"
  we_write_csv(residual_diag,
                 we_path(meta$output_dir, paste0("WE_", a,
                                                  "_GAUSSIAN_RESIDUAL_DIAGNOSTICS.csv")))
  raw <- as.numeric(residuals(fit$model, type = "response"))
  std <- as.numeric(fit$model$std.rsd)
  raw_acf <- we_sequence_acf(d, raw, 12L)
  std_acf <- we_sequence_acf(d, std, 12L)
  raw_acf$analysis <- a
  std_acf$analysis <- a
  we_write_csv(raw_acf,
                 we_path(meta$output_dir, paste0("WE_", a, "_RESIDUAL_SEQUENCE_ACF.csv")))
  we_write_csv(std_acf,
                 we_path(meta$output_dir,
                           paste0("WE_", a, "_STANDARDIZED_RESIDUAL_SEQUENCE_ACF.csv")))

  plot_prefix <- we_path(meta$output_dir, paste0("WE_", a))
  we_plot_png(paste0(plot_prefix, "_RESIDUAL_VS_FITTED.png"), {
    plot(fitted(fit$model), raw, pch = 16, cex = 0.32, col = rgb(0, 0, 0, 0.20),
         xlab = "Fitted value", ylab = "Gaussian response residual",
         main = paste(a, "residual vs fitted"))
    abline(h = 0, col = "red", lwd = 2)
    lines(lowess(fitted(fit$model), raw), col = "blue", lwd = 2)
  })
  we_plot_png(paste0(plot_prefix, "_QQ_PLOT.png"), {
    qqnorm(std, pch = 16, cex = 0.32, col = rgb(0, 0, 0, 0.20),
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
         pch = 16, cex = 0.30, col = rgb(0.1, 0.2, 0.5, 0.18),
         xlab = "Precipitation baseline rank", ylab = "Radiation baseline rank",
         main = paste(a, "observed support with fixed contrast"))
    theta <- seq(0, 2 * pi, length.out = 400L)
    lines(0.75 + 0.10 * cos(theta), 0.25 + 0.10 * sin(theta), col = "darkgreen", lwd = 2)
    lines(0.75 + 0.10 * cos(theta), 0.50 + 0.10 * sin(theta), col = "darkorange", lwd = 2)
    points(c(0.75, 0.75), c(0.25, 0.50), pch = c(17, 19), cex = 1.8,
           col = c("darkgreen", "darkorange"))
  })

  fit_results[[a]] <- list(
    skipped = FALSE,
    model_path = we_path(meta$output_dir, paste0("WE_", model_name, ".rds")),
    response = "npp_anomaly_mean_g", n = nrow(d),
    n_sites = length(levels(d$point_id)), formula = paste(deparse(form), collapse = " "),
    fit_seconds = fit_seconds, health = health, post_ar = post_ar, basis = basis,
    contrast = contrast$main_contrast,
    contrast_invariance_max = contrast$max_absolute_difference,
    concurvity_highest = highest, hydro_concurvity = hydro,
    residual_diagnostics = residual_diag,
    support = support[support$analysis == a, , drop = FALSE],
    support_pass = support_pass[[a]]
  )
  we_log(sprintf(
    "%s fit complete n=%d sites=%d contrast=%.15g std_resid_lag1=%.15g seconds=%.2f",
    a, nrow(d), length(levels(d$point_id)), contrast$main_contrast,
    post_ar$standardized_residual_lag1, fit_seconds
  ))
}

we_write_csv(
  do.call(rbind, lapply(fit_results, `[[`, "post_ar")),
  we_path("06_DIAGNOSTICS", "WE_DOMAIN_AR_POSTFIT_DIAGNOSTIC_SUMMARY.csv")
)
we_write_csv(
  do.call(rbind, lapply(fit_results, `[[`, "basis")),
  we_path("06_DIAGNOSTICS", "WE_DOMAIN_BASIS_DIAGNOSTIC_SUMMARY.csv")
)
we_write_csv(
  do.call(rbind, lapply(fit_results, `[[`, "hydro_concurvity")),
  we_path("06_DIAGNOSTICS", "WE_DOMAIN_HYDRO_ENERGY_CONCURVITY_SUMMARY.csv")
)
we_write_csv(
  do.call(rbind, lapply(fit_results, `[[`, "residual_diagnostics")),
  we_path("06_DIAGNOSTICS", "WE_DOMAIN_GAUSSIAN_RESIDUAL_DIAGNOSTIC_SUMMARY.csv")
)

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
  target_references_refitted = FALSE, sample_summary = sample_summary,
  composition = composition, support = support, support_pass = support_pass,
  fit_results = fit_results, bootstrap_complete = FALSE
)
state_path <- we_path("09_LOGS", "WE_DOMAIN_STAGE1_STATE.rds")
saveRDS(state, state_path)

for (a in c("FULL_YEAR", "APR_SEP")) {
  if (fit_results[[a]]$contrast > 0) {
    check <- if (a == "FULL_YEAR")
      "FULL_YEAR_SIGN_REVERSAL_ANALYSIS_CHECK_REQUIRED" else
      "APR_SEP_SIGN_REVERSAL_ANALYSIS_CHECK_REQUIRED"
    we_write_csv(
      data.frame(analysis = a, contrast = fit_results[[a]]$contrast,
                 status = check, bootstrap_run = FALSE, stringsAsFactors = FALSE),
      we_path(WE_DOMAIN_META[[a]]$output_dir,
                paste0("WE_", a, "_SIGN_REVERSAL_STOP.csv"))
    )
    we_stop_check(check, paste0(a, "_DIRECTION_check"),
                   sprintf("contrast=%.15g", fit_results[[a]]$contrast))
  }
}

we_log("water-energy domain sensitivity stage 1 complete; supported bootstraps may proceed")
