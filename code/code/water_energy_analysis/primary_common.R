options(stringsAsFactors = FALSE, warn = 1)

WE_PROJECT_ROOT <- normalizePath(
  Sys.getenv("WATER_ENERGY_PROJECT_ROOT", unset = getwd()),
  winslash = "/", mustWork = TRUE
)
WE_INPUT_DIR <- Sys.getenv(
  "WATER_ENERGY_INPUT_DIR",
  unset = file.path(WE_PROJECT_ROOT, "data", "water_energy")
)
WE_OUTPUT_ROOT <- Sys.getenv(
  "WATER_ENERGY_PRIMARY_OUTPUT_DIR",
  unset = file.path(WE_PROJECT_ROOT, "outputs", "water_energy", "primary")
)
WE_MASTER_PATH <- Sys.getenv(
  "WATER_ENERGY_MASTER_PATH",
  unset = file.path(WE_INPUT_DIR, "water_energy_analysis_ready_2022_2025.csv")
)
WE_REFERENCE_SUPPORT_PATH <- Sys.getenv(
  "WATER_ENERGY_SUPPORT_PATH",
  unset = file.path(WE_INPUT_DIR, "primary_contrast_support.csv")
)

WE_SEED <- 20260803L
WE_BOOT_DRAWS <- 2100L
WE_BOOT_TARGET_SUCCESS <- 2000L
WE_BOOT_INITIAL_BATCH <- 50L
WE_BOOT_MAX_FAILURE_RATE <- 0.05
WE_CONTRAST_A <- c(precip_rank_baseline = 0.75, radiation_rank_baseline = 0.25)
WE_CONTRAST_B <- c(precip_rank_baseline = 0.75, radiation_rank_baseline = 0.50)
WE_SUPPORT_RADIUS <- 0.10

WE_DIRS <- c(
  "01_CODE", "02_MODEL0", "03_MODEL1_PRIMARY", "04_MODEL2",
  "05_MODEL3", "06_AR_DIAGNOSTICS", "07_BASIS_DIAGNOSTICS",
  "08_CONCURVITY", "09_PRIMARY_CONTRAST", "10_BOOTSTRAP",
  "11_LOYO", "12_VEGETATION_SUPPORT", "13_TESTS", "14_LOGS"
)

WE_MODEL1_CONTROLS <- c(
  "precip_antecedent3_sum_mm",
  "radiation_antecedent3_mean_MJ_m2",
  "soilwater_lag1",
  "ndvi_lag1_canonical"
)
WE_MODEL2_ADDED <- c("temperature_anomaly", "VPD_anomaly")
WE_MODEL3_ADDED <- c("soilwater_anomaly", "NDVI_anomaly")
WE_ALL_CONTROLS <- c(WE_MODEL1_CONTROLS, WE_MODEL2_ADDED, WE_MODEL3_ADDED)
WE_BANNED_PRIMARY_FIELDS <- c(
  "eligible_all12", "eligible_apr_sep", "eligible_mod17_QA1",
  "eligible_mod17_QA2", "eligible_ndvi_QA01", "eligible_ndvi_QA0",
  "eligible_cold_snow", "eligible_dayweighted", "npp_anomaly_median_g",
  "precip_anomaly_mm", "radiation_anomaly_MJ_m2"
)

we_ensure_dirs <- function() {
  dir.create(WE_OUTPUT_ROOT, recursive = TRUE, showWarnings = FALSE)
  invisible(vapply(
    WE_DIRS,
    function(x) dir.create(file.path(WE_OUTPUT_ROOT, x), recursive = TRUE, showWarnings = FALSE),
    logical(1)
  ))
}

we_path <- function(...) file.path(WE_OUTPUT_ROOT, ...)

we_write_csv <- function(x, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  write.csv(x, path, row.names = FALSE, na = "")
  invisible(path)
}

we_write_lines <- function(x, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  writeLines(enc2utf8(as.character(x)), path, useBytes = TRUE)
  invisible(path)
}

we_log <- function(..., file = we_path("14_LOGS", "WE_GAMM_PRIMARY_RUN.log")) {
  stamp <- format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z")
  msg <- paste(..., collapse = " ")
  cat(sprintf("[%s] %s\n", stamp, msg), file = file, append = TRUE)
  message(msg)
  invisible(msg)
}

we_stop_check <- function(reason, stage, detail = "") {
  status <- data.frame(
    timestamp = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
    stage = stage,
    reason = reason,
    detail = detail,
    final_status = paste0("STOPPED — analysis check required: ", reason),
    stringsAsFactors = FALSE
  )
  we_write_csv(status, we_path("14_LOGS", "WE_STOP_STATUS.csv"))
  we_log("STOP check:", reason, "stage=", stage, detail)
  stop(paste(reason, detail), call. = FALSE)
}

we_assert <- function(condition, message) {
  if (!isTRUE(condition)) stop(message, call. = FALSE)
  invisible(TRUE)
}

we_read_master <- function() {
  we_assert(file.exists(WE_MASTER_PATH), "Authoritative master is missing")
  d <- read.csv(
    WE_MASTER_PATH,
    colClasses = c(point_id = "character"),
    check.names = FALSE
  )
  d[["point_id"]] <- sprintf("%03d", as.integer(d[["point_id"]]))
  d
}

we_primary_data <- function(master) {
  we_assert("eligible_primary_model1" %in% names(master), "Primary eligibility flag missing")
  keep <- !is.na(master[["eligible_primary_model1"]]) & master[["eligible_primary_model1"]] == 1L
  d <- master[keep, , drop = FALSE]
  required <- c(
    "point_id", "Year", "Month", "calendar_month_index",
    "npp_anomaly_mean_g", "precip_rank_baseline", "radiation_rank_baseline",
    WE_ALL_CONTROLS, "daima"
  )
  missing_fields <- setdiff(required, names(d))
  we_assert(length(missing_fields) == 0L, paste("Required fields missing:", paste(missing_fields, collapse = ",")))
  numeric_required <- setdiff(required, c("point_id"))
  nonfinite <- vapply(
    numeric_required,
    function(v) any(!is.finite(d[[v]])),
    logical(1)
  )
  we_assert(!any(nonfinite), paste("Nonfinite Primary fields:", paste(names(nonfinite)[nonfinite], collapse = ",")))
  ord <- order(d[["point_id"]], d[["calendar_month_index"]])
  d <- d[ord, , drop = FALSE]
  rownames(d) <- NULL
  d[["point_id"]] <- factor(d[["point_id"]], levels = sort(unique(d[["point_id"]])))
  d[["AR.start"]] <- we_build_ar_start(d, "point_id")
  d
}

we_build_ar_start <- function(d, id_col = "point_id", time_col = "calendar_month_index") {
  n <- nrow(d)
  if (n == 0L) return(logical())
  ids <- as.character(d[[id_col]])
  tt <- as.integer(d[[time_col]])
  if (n > 1L) {
    within_same <- ids[-1L] == ids[-n]
    we_assert(all(diff(tt)[within_same] > 0L), "Time is not strictly increasing within series")
  }
  c(TRUE, ids[-1L] != ids[-n] | (tt[-1L] - tt[-n]) > 1L)
}

we_continuous_pairs <- function(d, values, id_col = "point_id", time_col = "calendar_month_index") {
  we_assert(length(values) == nrow(d), "Pair values length mismatch")
  n <- nrow(d)
  if (n < 2L) {
    return(data.frame(
      series_id = character(), calendar_month_index_t_minus_1 = integer(),
      calendar_month_index_t = integer(), value_t_minus_1 = numeric(),
      value_t = numeric(), stringsAsFactors = FALSE
    ))
  }
  ids <- as.character(d[[id_col]])
  tt <- as.integer(d[[time_col]])
  keep <- ids[-1L] == ids[-n] & (tt[-1L] - tt[-n]) == 1L &
    is.finite(values[-1L]) & is.finite(values[-n])
  data.frame(
    series_id = ids[-1L][keep],
    calendar_month_index_t_minus_1 = tt[-n][keep],
    calendar_month_index_t = tt[-1L][keep],
    value_t_minus_1 = values[-n][keep],
    value_t = values[-1L][keep],
    stringsAsFactors = FALSE
  )
}

we_rho_from_pairs <- function(pairs) {
  we_assert(nrow(pairs) >= 2L, "Insufficient consecutive residual pairs")
  unname(cor(pairs[["value_t_minus_1"]], pairs[["value_t"]]))
}

we_rho_is_extreme <- function(rho) is.finite(rho) && abs(rho) >= 0.90

we_rho_validation <- function(d, residuals) {
  pairs <- we_continuous_pairs(d, residuals)
  rho_raw <- we_rho_from_pairs(pairs)
  counts <- table(pairs[["series_id"]])
  summary <- data.frame(
    rho_raw = rho_raw,
    rho_used = rho_raw,
    n_consecutive_pairs = nrow(pairs),
    n_contributing_sites = length(counts),
    pair_count_per_site_min = min(as.numeric(counts)),
    pair_count_per_site_median = median(as.numeric(counts)),
    pair_count_per_site_mean = mean(as.numeric(counts)),
    pair_count_per_site_max = max(as.numeric(counts)),
    negative_rho_clamped = FALSE,
    rho_iteratively_reestimated = FALSE,
    stringsAsFactors = FALSE
  )
  list(summary = summary, pairs = pairs)
}

we_ar_sequence_validation <- function(d, id_col = "point_id") {
  starts <- we_build_ar_start(d, id_col)
  section_id <- cumsum(starts)
  section_lengths <- as.numeric(table(section_id))
  ids <- as.character(d[[id_col]])
  tt <- as.integer(d[["calendar_month_index"]])
  n <- nrow(d)
  same <- if (n > 1L) ids[-1L] == ids[-n] else logical()
  delta <- if (n > 1L) tt[-1L] - tt[-n] else integer()
  gap_delta <- delta[same & delta > 1L]
  gap_distribution <- if (length(gap_delta)) {
    tab <- table(gap_delta)
    paste(sprintf("delta_%s=%s", names(tab), as.integer(tab)), collapse = ";")
  } else {
    "NONE"
  }
  summary <- data.frame(
    n_records = nrow(d),
    n_sites = length(unique(ids)),
    n_AR_sections = sum(starts),
    section_length_min = min(section_lengths),
    section_length_median = median(section_lengths),
    section_length_mean = mean(section_lengths),
    section_length_max = max(section_lengths),
    gap_count = length(gap_delta),
    gap_length_distribution_calendar_delta = gap_distribution,
    stringsAsFactors = FALSE
  )
  gaps <- if (length(gap_delta)) {
    idx <- which(same & delta > 1L)
    data.frame(
      series_id = ids[idx],
      previous_calendar_month_index = tt[idx],
      current_calendar_month_index = tt[idx + 1L],
      calendar_month_delta = delta[idx],
      missing_month_count = delta[idx] - 1L,
      stringsAsFactors = FALSE
    )
  } else {
    data.frame(
      series_id = character(), previous_calendar_month_index = integer(),
      current_calendar_month_index = integer(), calendar_month_delta = integer(),
      missing_month_count = integer(), stringsAsFactors = FALSE
    )
  }
  gap_dist <- if (nrow(gaps)) {
    out <- as.data.frame(table(gaps[["calendar_month_delta"]]), stringsAsFactors = FALSE)
    names(out) <- c("calendar_month_delta", "n_gaps")
    out[["calendar_month_delta"]] <- as.integer(as.character(out[["calendar_month_delta"]]))
    out[["missing_month_count"]] <- out[["calendar_month_delta"]] - 1L
    out
  } else {
    data.frame(calendar_month_delta = integer(), n_gaps = integer(), missing_month_count = integer())
  }
  list(summary = summary, gaps = gaps, gap_distribution = gap_dist, starts = starts, section_id = section_id)
}

we_surface_term <- function(k) {
  sprintf("te(precip_rank_baseline, radiation_rank_baseline, k = c(%d, %d))", k, k)
}

we_smooth_term <- function(variable, k) sprintf("s(%s, k = %d)", variable, k)

we_model1_k_initial <- function() {
  list(surface = 5L, controls = setNames(rep(5L, length(WE_MODEL1_CONTROLS)), WE_MODEL1_CONTROLS))
}

we_added_k_initial <- function(variables) setNames(rep(5L, length(variables)), variables)

we_formula_from_terms <- function(terms, site_term = 's(point_id, bs = "re")') {
  as.formula(paste("npp_anomaly_mean_g ~", paste(c(terms, "factor(Month)", "factor(Year)", site_term), collapse = " + ")))
}

we_formula_model0 <- function(surface_k = 5L) {
  we_formula_from_terms(we_surface_term(surface_k))
}

we_formula_model1 <- function(k_config = we_model1_k_initial(), site_variable = "point_id") {
  terms <- c(
    we_surface_term(k_config[["surface"]]),
    vapply(
      WE_MODEL1_CONTROLS,
      function(v) we_smooth_term(v, k_config[["controls"]][[v]]),
      character(1)
    )
  )
  site_term <- sprintf('s(%s, bs = "re")', site_variable)
  we_formula_from_terms(terms, site_term)
}

we_formula_model2 <- function(k_config = we_model1_k_initial(), added_k = we_added_k_initial(WE_MODEL2_ADDED)) {
  terms <- c(
    we_surface_term(k_config[["surface"]]),
    vapply(WE_MODEL1_CONTROLS, function(v) we_smooth_term(v, k_config[["controls"]][[v]]), character(1)),
    vapply(WE_MODEL2_ADDED, function(v) we_smooth_term(v, added_k[[v]]), character(1))
  )
  we_formula_from_terms(terms)
}

we_formula_model3 <- function(
    k_config = we_model1_k_initial(),
    model2_added_k = we_added_k_initial(WE_MODEL2_ADDED),
    model3_added_k = we_added_k_initial(WE_MODEL3_ADDED)) {
  terms <- c(
    we_surface_term(k_config[["surface"]]),
    vapply(WE_MODEL1_CONTROLS, function(v) we_smooth_term(v, k_config[["controls"]][[v]]), character(1)),
    vapply(WE_MODEL2_ADDED, function(v) we_smooth_term(v, model2_added_k[[v]]), character(1)),
    vapply(WE_MODEL3_ADDED, function(v) we_smooth_term(v, model3_added_k[[v]]), character(1))
  )
  we_formula_from_terms(terms)
}

we_fit_bam <- function(formula, data, rho_used, ar_start) {
  we_assert(length(ar_start) == nrow(data), "AR.start length does not match model data")
  data[["AR.start"]] <- as.logical(ar_start)
  warnings <- character()
  model <- withCallingHandlers(
    mgcv::bam(
      formula = formula,
      data = data,
      family = gaussian(link = "identity"),
      method = "fREML",
      rho = rho_used,
      AR.start = AR.start,
      discrete = FALSE
    ),
    warning = function(w) {
      warnings <<- c(warnings, conditionMessage(w))
      invokeRestart("muffleWarning")
    }
  )
  list(model = model, warnings = unique(warnings))
}

we_model_convergence <- function(model) {
  if (!is.null(model[["converged"]])) return(isTRUE(model[["converged"]]))
  if (!is.null(model[["mgcv.conv"]][["fully.converged"]])) {
    return(isTRUE(model[["mgcv.conv"]][["fully.converged"]]))
  }
  if (!is.null(model[["outer.info"]][["conv"]])) {
    return(grepl("converg", as.character(model[["outer.info"]][["conv"]]), ignore.case = TRUE))
  }
  NA
}

we_model_health <- function(model, warnings = character()) {
  cf <- coef(model)
  fv <- fitted(model)
  rr <- residuals(model, type = "response")
  rank_value <- if (is.null(model[["rank"]])) NA_integer_ else as.integer(model[["rank"]])
  coefficient_count <- length(cf)
  full_rank <- if (is.na(rank_value)) NA else rank_value == coefficient_count
  conv <- we_model_convergence(model)
  numerical_warning <- any(grepl(
    "rank deficient|not finite|non-finite|singular|identif|failed to converge|iteration limit|not positive definite",
    warnings,
    ignore.case = TRUE
  ))
  data.frame(
    convergence = if (isTRUE(conv)) "CONVERGED" else if (identical(conv, FALSE)) "NOT_CONVERGED" else "NOT_EXPLICITLY_REPORTED",
    coefficients_finite = all(is.finite(cf)),
    fitted_finite = all(is.finite(fv)),
    residuals_finite = all(is.finite(rr)),
    coefficient_count = coefficient_count,
    model_rank = rank_value,
    full_rank = full_rank,
    numerical_identifiability_warning = numerical_warning,
    warning_count = length(warnings),
    warnings = paste(warnings, collapse = " | "),
    stringsAsFactors = FALSE
  )
}

we_model_health_stop_reason <- function(health) {
  reasons <- character()
  if (health[["convergence"]][1] == "NOT_CONVERGED") reasons <- c(reasons, "MODEL NOT CONVERGED")
  if (!isTRUE(health[["coefficients_finite"]][1])) reasons <- c(reasons, "NONFINITE COEFFICIENT")
  if (!isTRUE(health[["fitted_finite"]][1])) reasons <- c(reasons, "NONFINITE FITTED VALUE")
  if (!isTRUE(health[["residuals_finite"]][1])) reasons <- c(reasons, "NONFINITE RESIDUAL")
  if (identical(health[["full_rank"]][1], FALSE)) reasons <- c(reasons, "RANK DEFICIENCY")
  if (isTRUE(health[["numerical_identifiability_warning"]][1])) reasons <- c(reasons, "NUMERICAL IDENTIFIABILITY FAILURE")
  unique(reasons)
}

we_basis_triple_check <- function(edf, k_prime, k_index, p_value) {
  is.finite(k_index) & is.finite(p_value) & is.finite(edf) & is.finite(k_prime) &
    k_index < 0.90 & p_value < 0.05 & edf >= 0.80 * k_prime
}

we_run_k_check <- function(model) {
  set.seed(WE_SEED)
  warnings <- character()
  raw <- withCallingHandlers(
    mgcv::k.check(model, subsample = 20000, n.rep = 1000),
    warning = function(w) {
      warnings <<- c(warnings, conditionMessage(w))
      invokeRestart("muffleWarning")
    }
  )
  out <- as.data.frame(raw, check.names = FALSE, stringsAsFactors = FALSE)
  out[["smooth_name"]] <- rownames(raw)
  rownames(out) <- NULL
  names(out)[names(out) == "k'"] <- "k_prime"
  names(out)[names(out) == "k-index"] <- "k_index"
  names(out)[names(out) == "p-value"] <- "p_value"
  names(out)[names(out) == "edf"] <- "edf"
  out <- out[, c("smooth_name", "edf", "k_prime", "k_index", "p_value"), drop = FALSE]
  out[["basis_dimension_suspect"]] <- with(out, we_basis_triple_check(edf, k_prime, k_index, p_value))
  out[["basis_flag"]] <- ifelse(out[["basis_dimension_suspect"]], "BASIS_DIMENSION_SUSPECT", "PASS_OR_NOT_APPLICABLE")
  attr(out, "warnings") <- unique(warnings)
  out
}

we_basis_suspect_for <- function(basis, variable) {
  any(basis[["basis_dimension_suspect"]] & grepl(variable, basis[["smooth_name"]], fixed = TRUE))
}

we_basis_surface_suspect <- function(basis) {
  any(basis[["basis_dimension_suspect"]] &
    grepl("precip_rank_baseline", basis[["smooth_name"]], fixed = TRUE) &
    grepl("radiation_rank_baseline", basis[["smooth_name"]], fixed = TRUE))
}

we_apply_model1_single_fallback <- function(k_config, basis) {
  updated <- k_config
  changed <- character()
  if (we_basis_surface_suspect(basis)) {
    we_assert(updated[["surface"]] == 5L, "Surface fallback is not a single 5-to-7 change")
    updated[["surface"]] <- 7L
    changed <- c(changed, "hydro_energy_surface:5->7")
  }
  for (v in WE_MODEL1_CONTROLS) {
    if (we_basis_suspect_for(basis, v)) {
      we_assert(updated[["controls"]][[v]] == 5L, paste("Fallback is not a single 5-to-7 change for", v))
      updated[["controls"]][[v]] <- 7L
      changed <- c(changed, paste0(v, ":5->7"))
    }
  }
  list(config = updated, changed = changed)
}

we_apply_added_single_fallback <- function(added_k, variables, basis) {
  updated <- added_k
  changed <- character()
  for (v in variables) {
    if (we_basis_suspect_for(basis, v)) {
      we_assert(updated[[v]] == 5L, paste("Fallback is not a single 5-to-7 change for", v))
      updated[[v]] <- 7L
      changed <- c(changed, paste0(v, ":5->7"))
    }
  }
  list(config = updated, changed = changed)
}

we_smooth_edf <- function(model) {
  st <- summary(model)[["s.table"]]
  if (is.null(st)) return(data.frame())
  out <- as.data.frame(st, check.names = FALSE, stringsAsFactors = FALSE)
  out[["smooth_name"]] <- rownames(st)
  rownames(out) <- NULL
  out[, c("smooth_name", setdiff(names(out), "smooth_name")), drop = FALSE]
}

we_residual_diagnostics <- function(model) {
  r <- as.numeric(residuals(model, type = "response"))
  f <- as.numeric(fitted(model))
  scale_value <- summary(model)[["scale"]]
  std <- model[["std.rsd"]]
  standardized_source <- "model$std.rsd"
  if (is.null(std)) {
    std <- r / sqrt(scale_value)
    standardized_source <- "fallback_residual_div_sqrt_scale"
  }
  std <- as.numeric(std)
  skew <- function(x) mean((x - mean(x))^3) / sd(x)^3
  excess_kurtosis <- function(x) mean((x - mean(x))^4) / sd(x)^4 - 3
  data.frame(
    n = length(r),
    residual_mean = mean(r),
    residual_sd = sd(r),
    residual_min = min(r),
    residual_q01 = unname(quantile(r, 0.01)),
    residual_q25 = unname(quantile(r, 0.25)),
    residual_median = median(r),
    residual_q75 = unname(quantile(r, 0.75)),
    residual_q99 = unname(quantile(r, 0.99)),
    residual_max = max(r),
    residual_skewness = skew(r),
    residual_excess_kurtosis = excess_kurtosis(r),
    standardized_residual_min = min(std),
    standardized_residual_max = max(std),
    max_abs_standardized_residual = max(abs(std)),
    largest_squared_residual_share = max(r^2) / sum(r^2),
    fitted_min = min(f),
    fitted_max = max(f),
    residual_fitted_correlation = cor(r, f),
    nonfinite_residual_count = sum(!is.finite(r)),
    nonfinite_fitted_count = sum(!is.finite(f)),
    standardized_residual_source = standardized_source,
    stringsAsFactors = FALSE
  )
}

we_reference_row <- function(data, group_col = "point_id", quantile_prob = 0.50) {
  ref <- data[1L, , drop = FALSE]
  controls <- intersect(WE_ALL_CONTROLS, names(ref))
  for (v in controls) ref[[v]] <- unname(quantile(data[[v]], quantile_prob, na.rm = TRUE))
  ref[["precip_rank_baseline"]] <- WE_CONTRAST_A[["precip_rank_baseline"]]
  ref[["radiation_rank_baseline"]] <- WE_CONTRAST_A[["radiation_rank_baseline"]]
  ref[["Month"]] <- sort(unique(data[["Month"]]))[1L]
  ref[["Year"]] <- sort(unique(data[["Year"]]))[1L]
  group_values <- levels(data[[group_col]])
  if (is.null(group_values)) group_values <- sort(unique(as.character(data[[group_col]])))
  ref[[group_col]] <- factor(group_values[1L], levels = group_values)
  ref
}

we_lpmatrix_contrast <- function(model, reference_row) {
  a <- reference_row
  b <- reference_row
  a[["precip_rank_baseline"]] <- WE_CONTRAST_A[["precip_rank_baseline"]]
  a[["radiation_rank_baseline"]] <- WE_CONTRAST_A[["radiation_rank_baseline"]]
  b[["precip_rank_baseline"]] <- WE_CONTRAST_B[["precip_rank_baseline"]]
  b[["radiation_rank_baseline"]] <- WE_CONTRAST_B[["radiation_rank_baseline"]]
  xa <- predict(model, newdata = a, type = "lpmatrix")
  xb <- predict(model, newdata = b, type = "lpmatrix")
  dx <- xa - xb
  estimate <- as.numeric(dx %*% coef(model))
  list(estimate = estimate, dX = dx, X_A = xa, X_B = xb, newdata_A = a, newdata_B = b)
}

we_contrast_invariance <- function(model, data, group_col = "point_id") {
  main_ref <- we_reference_row(data, group_col, 0.50)
  main <- we_lpmatrix_contrast(model, main_ref)[["estimate"]]
  months <- unique(c(min(data[["Month"]]), median(sort(unique(data[["Month"]]))), max(data[["Month"]])))
  months <- as.integer(months[seq_len(min(3L, length(months)))])
  years <- sort(unique(data[["Year"]]))
  years <- years[c(1L, length(years))]
  group_values <- levels(data[[group_col]])
  if (is.null(group_values)) group_values <- sort(unique(as.character(data[[group_col]])))
  site_idx <- unique(as.integer(round(seq(1, length(group_values), length.out = 3L))))
  groups <- group_values[site_idx]
  probs <- c(0.25, 0.50, 0.75)
  rows <- list()
  z <- 0L
  for (m in months) for (y in years) for (q in probs) for (g in groups) {
    ref <- we_reference_row(data, group_col, q)
    ref[["Month"]] <- m
    ref[["Year"]] <- y
    ref[[group_col]] <- factor(g, levels = group_values)
    value <- we_lpmatrix_contrast(model, ref)[["estimate"]]
    z <- z + 1L
    rows[[z]] <- data.frame(
      Month = m, Year = y, continuous_control_quantile = q,
      group_value = g, contrast = value,
      difference_from_main = value - main,
      absolute_difference_from_main = abs(value - main),
      stringsAsFactors = FALSE
    )
  }
  validation <- do.call(rbind, rows)
  max_diff <- max(validation[["absolute_difference_from_main"]])
  list(
    main_contrast = main,
    validation = validation,
    max_absolute_difference = max_diff,
    pass = is.finite(main) && all(is.finite(validation[["contrast"]])) && max_diff <= 1e-8
  )
}

we_support_recheck <- function(primary) {
  points <- list(A = WE_CONTRAST_A, B = WE_CONTRAST_B)
  out <- do.call(rbind, lapply(names(points), function(label) {
    target <- points[[label]]
    dist <- sqrt(
      (primary[["precip_rank_baseline"]] - target[["precip_rank_baseline"]])^2 +
        (primary[["radiation_rank_baseline"]] - target[["radiation_rank_baseline"]])^2
    )
    keep <- is.finite(dist) & dist <= WE_SUPPORT_RADIUS
    data.frame(
      contrast_point = label,
      precip_rank_target = target[["precip_rank_baseline"]],
      radiation_rank_target = target[["radiation_rank_baseline"]],
      radius = WE_SUPPORT_RADIUS,
      n_records = sum(keep),
      n_sites = length(unique(as.character(primary[["point_id"]][keep]))),
      n_years = length(unique(primary[["Year"]][keep])),
      stringsAsFactors = FALSE
    )
  }))
  reference <- read.csv(WE_REFERENCE_SUPPORT_PATH, stringsAsFactors = FALSE)
  reference <- reference[match(out[["contrast_point"]], reference[["contrast_point"]]), , drop = FALSE]
  out[["reference_n_records"]] <- reference[["n_records"]]
  out[["reference_n_sites"]] <- reference[["n_sites"]]
  out[["reference_n_years"]] <- reference[["n_years"]]
  out[["matches_reference_support"]] <- with(
    out,
    n_records == reference_n_records & n_sites == reference_n_sites & n_years == reference_n_years
  )
  out
}

we_concurvity_label <- function(x) {
  ifelse(x > 0.90, "HIGH_CONCURVITY", ifelse(x >= 0.80, "CAUTION", "ROUTINE"))
}

we_concurvity_tidy <- function(model) {
  full_raw <- mgcv::concurvity(model, full = TRUE)
  full <- as.matrix(full_raw)
  full_rows <- list()
  z <- 0L
  for (i in seq_len(nrow(full))) for (j in seq_len(ncol(full))) {
    z <- z + 1L
    full_rows[[z]] <- data.frame(
      mode = "full", metric = rownames(full)[i], term_from = "ALL_OTHER_TERMS",
      term_to = colnames(full)[j], value = as.numeric(full[i, j]),
      stringsAsFactors = FALSE
    )
  }
  pair_raw <- mgcv::concurvity(model, full = FALSE)
  pair_rows <- list()
  z <- 0L
  for (metric in names(pair_raw)) {
    mat <- as.matrix(pair_raw[[metric]])
    for (i in seq_len(nrow(mat))) for (j in seq_len(ncol(mat))) {
      z <- z + 1L
      pair_rows[[z]] <- data.frame(
        mode = "pairwise", metric = metric, term_from = rownames(mat)[i],
        term_to = colnames(mat)[j], value = as.numeric(mat[i, j]),
        stringsAsFactors = FALSE
      )
    }
  }
  out <- rbind(do.call(rbind, full_rows), do.call(rbind, pair_rows))
  out[["is_pairwise_diagonal"]] <- out[["mode"]] == "pairwise" & out[["term_from"]] == out[["term_to"]]
  out[["validation_label"]] <- ifelse(is.finite(out[["value"]]), we_concurvity_label(out[["value"]]), "NOT_AVAILABLE")
  out
}

we_highest_concurvity <- function(concurvity_table, model_name) {
  keep <- is.finite(concurvity_table[["value"]]) & !concurvity_table[["is_pairwise_diagonal"]]
  x <- concurvity_table[keep, , drop = FALSE]
  if (!nrow(x)) return(data.frame())
  row <- x[which.max(x[["value"]]), , drop = FALSE]
  row[["model"]] <- model_name
  row[, c("model", setdiff(names(row), "model")), drop = FALSE]
}

we_sequence_acf <- function(d, values, max_lag = 12L, id_col = "point_id") {
  ids <- as.character(d[[id_col]])
  tt <- as.integer(d[["calendar_month_index"]])
  out <- lapply(seq_len(max_lag), function(lag) {
    if (nrow(d) <= lag) return(data.frame(lag = lag, correlation = NA_real_, n_pairs = 0L))
    left <- seq_len(nrow(d) - lag)
    right <- left + lag
    keep <- ids[left] == ids[right] & (tt[right] - tt[left]) == lag &
      is.finite(values[left]) & is.finite(values[right])
    data.frame(
      lag = lag,
      correlation = if (sum(keep) >= 2L) cor(values[left][keep], values[right][keep]) else NA_real_,
      n_pairs = sum(keep),
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, out)
}

we_save_model_outputs <- function(model, warnings, model_name, model_dir, data, rho_used) {
  dir <- we_path(model_dir)
  saveRDS(model, file.path(dir, paste0("WE_", model_name, ".rds")))
  formula_text <- paste(deparse(formula(model), width.cutoff = 500L), collapse = " ")
  we_write_lines(formula_text, file.path(dir, paste0("WE_", model_name, "_FORMULA.txt")))
  health <- we_model_health(model, warnings)
  sm <- summary(model)
  info <- cbind(
    data.frame(
      model = model_name,
      n = nrow(data),
      n_sites = if ("point_id" %in% names(data)) length(unique(as.character(data[["point_id"]]))) else NA_integer_,
      rho_used = rho_used,
      family = model[["family"]][["family"]],
      link = model[["family"]][["link"]],
      method = model[["method"]],
      discrete = FALSE,
      deviance_explained = sm[["dev.expl"]],
      adjusted_r_squared = sm[["r.sq"]],
      scale = sm[["scale"]],
      formula = formula_text,
      stringsAsFactors = FALSE
    ),
    health
  )
  we_write_csv(info, file.path(dir, paste0("WE_", model_name, "_MODEL_INFO.csv")))
  we_write_csv(we_smooth_edf(model), file.path(dir, paste0("WE_", model_name, "_SMOOTH_EDF.csv")))
  we_write_csv(we_residual_diagnostics(model), file.path(dir, paste0("WE_", model_name, "_GAUSSIAN_RESIDUAL_DIAGNOSTICS.csv")))
  invisible(list(info = info, health = health))
}

we_boot_series_ids <- function(sampled_sites) {
  occurrence <- ave(seq_along(sampled_sites), sampled_sites, FUN = seq_along)
  paste0(sampled_sites, "_copy", sprintf("%02d", occurrence))
}

we_bootstrap_draw_sequence <- function(site_ids, n_draws = WE_BOOT_DRAWS, seed = WE_SEED) {
  set.seed(seed)
  rows <- vector("list", n_draws)
  for (attempt in seq_len(n_draws)) {
    sampled <- sample(site_ids, size = length(site_ids), replace = TRUE)
    rows[[attempt]] <- data.frame(
      attempt = attempt,
      draw_id = sprintf("draw_%04d", attempt),
      draw_position = seq_along(sampled),
      original_point_id = sampled,
      boot_series_id = we_boot_series_ids(sampled),
      stringsAsFactors = FALSE
    )
  }
  do.call(rbind, rows)
}

we_build_boot_data <- function(primary, draw_rows) {
  ids <- as.character(primary[["point_id"]])
  copies <- lapply(seq_len(nrow(draw_rows)), function(i) {
    x <- primary[ids == draw_rows[["original_point_id"]][i], , drop = FALSE]
    x[["boot_series_id"]] <- draw_rows[["boot_series_id"]][i]
    x
  })
  out <- do.call(rbind, copies)
  ord <- order(out[["boot_series_id"]], out[["calendar_month_index"]])
  out <- out[ord, , drop = FALSE]
  rownames(out) <- NULL
  out[["boot_series_id"]] <- factor(out[["boot_series_id"]], levels = sort(unique(out[["boot_series_id"]])))
  out[["AR.start"]] <- we_build_ar_start(out, "boot_series_id")
  out
}

we_boot_failure_category <- function(message) {
  if (!nzchar(message)) return("NONE")
  if (grepl(
    "object .* not found|length mismatch|factor has new level|contrasts can be applied|model frame|AR.start length|not strictly increasing",
    message,
    ignore.case = TRUE
  )) return("STRUCTURAL_ERROR")
  if (grepl("rank|identif|singular", message, ignore.case = TRUE)) return("NUMERICAL_IDENTIFIABILITY_FAILURE")
  if (grepl("nonfinite|non-finite|not finite|Inf|NaN", message, ignore.case = TRUE)) return("NONFINITE_RESULT")
  if (grepl("converg|iteration", message, ignore.case = TRUE)) return("CONVERGENCE_FAILURE")
  "FIT_ERROR"
}

we_run_boot_attempt <- function(attempt, primary, draw_sequence, k_config, rho_used) {
  started <- proc.time()[["elapsed"]]
  draw_rows <- draw_sequence[draw_sequence[["attempt"]] == attempt, , drop = FALSE]
  base <- list(
    attempt = as.integer(attempt),
    draw_id = sprintf("draw_%04d", as.integer(attempt)),
    success = FALSE,
    contrast = NA_real_,
    convergence = "NOT_FIT",
    warning = "",
    failure_reason = "",
    failure_category = "",
    n = NA_integer_,
    n_boot_series = NA_integer_,
    n_AR_sections = NA_integer_,
    elapsed_seconds = NA_real_
  )
  result <- tryCatch({
    boot <- we_build_boot_data(primary, draw_rows)
    formula <- we_formula_model1(k_config, site_variable = "boot_series_id")
    fit <- we_fit_bam(formula, boot, rho_used, boot[["AR.start"]])
    health <- we_model_health(fit[["model"]], fit[["warnings"]])
    reasons <- we_model_health_stop_reason(health)
    contrast <- NA_real_
    if (!length(reasons)) {
      ref <- we_reference_row(boot, group_col = "boot_series_id", quantile_prob = 0.50)
      contrast <- we_lpmatrix_contrast(fit[["model"]], ref)[["estimate"]]
      if (!is.finite(contrast)) reasons <- c(reasons, "NONFINITE CONTRAST")
    }
    reason_text <- paste(reasons, collapse = ";")
    list(
      attempt = as.integer(attempt),
      draw_id = sprintf("draw_%04d", as.integer(attempt)),
      success = length(reasons) == 0L && is.finite(contrast),
      contrast = contrast,
      convergence = health[["convergence"]][1L],
      warning = paste(fit[["warnings"]], collapse = " | "),
      failure_reason = reason_text,
      failure_category = we_boot_failure_category(reason_text),
      n = nrow(boot),
      n_boot_series = length(levels(boot[["boot_series_id"]])),
      n_AR_sections = sum(boot[["AR.start"]]),
      elapsed_seconds = proc.time()[["elapsed"]] - started
    )
  }, error = function(e) {
    msg <- conditionMessage(e)
    base[["failure_reason"]] <- msg
    base[["failure_category"]] <- we_boot_failure_category(msg)
    base[["elapsed_seconds"]] <- proc.time()[["elapsed"]] - started
    base
  })
  gc(verbose = FALSE)
  as.data.frame(result, stringsAsFactors = FALSE)
}

we_vegetation_support_qualified <- function(a_records, a_sites, a_years, b_records, b_sites, b_years) {
  all(c(a_records, b_records) >= 50L) && all(c(a_sites, b_sites) >= 25L) && all(c(a_years, b_years) >= 3L)
}

we_vegetation_support <- function(primary) {
  groups <- sort(unique(primary[["daima"]]))
  rows <- lapply(groups, function(group) {
    x <- primary[primary[["daima"]] == group, , drop = FALSE]
    counts <- lapply(list(A = WE_CONTRAST_A, B = WE_CONTRAST_B), function(target) {
      dist <- sqrt(
        (x[["precip_rank_baseline"]] - target[["precip_rank_baseline"]])^2 +
          (x[["radiation_rank_baseline"]] - target[["radiation_rank_baseline"]])^2
      )
      keep <- is.finite(dist) & dist <= WE_SUPPORT_RADIUS
      c(
        records = sum(keep),
        sites = length(unique(as.character(x[["point_id"]][keep]))),
        years = length(unique(x[["Year"]][keep]))
      )
    })
    qualified <- we_vegetation_support_qualified(
      counts[["A"]][["records"]], counts[["A"]][["sites"]], counts[["A"]][["years"]],
      counts[["B"]][["records"]], counts[["B"]][["sites"]], counts[["B"]][["years"]]
    )
    data.frame(
      vegetation_group = as.character(group),
      radius = WE_SUPPORT_RADIUS,
      A_n_records = counts[["A"]][["records"]],
      A_n_sites = counts[["A"]][["sites"]],
      A_n_years = counts[["A"]][["years"]],
      B_n_records = counts[["B"]][["records"]],
      B_n_sites = counts[["B"]][["sites"]],
      B_n_years = counts[["B"]][["years"]],
      status = if (qualified) "QUALIFIED_FOR_SUPPLEMENTARY_VEGETATION_MODEL" else "NOT_ESTIMABLE_DUE_TO_SUPPORT",
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}

we_formula_variables <- function(formula) unique(all.vars(formula))

we_direction <- function(x) ifelse(x > 0, "POSITIVE", ifelse(x < 0, "NEGATIVE", "ZERO"))

we_software_validation <- function() {
  data.frame(
    R_version_string = R.version.string,
    mgcv_version = as.character(utils::packageVersion("mgcv")),
    mgcv_loadable = requireNamespace("mgcv", quietly = TRUE),
    network_download = FALSE,
    install_packages_called = FALSE,
    update_packages_called = FALSE,
    stringsAsFactors = FALSE
  )
}

we_ensure_dirs()
