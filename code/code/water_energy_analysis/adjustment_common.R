options(stringsAsFactors = FALSE, warn = 1)

WE_CODE_DIR <- Sys.getenv("WATER_ENERGY_CODE_DIR", unset = getwd())
source(file.path(WE_CODE_DIR, "primary_common.R"))

WE_PRIMARY_ROOT <- Sys.getenv(
  "WATER_ENERGY_PRIMARY_OUTPUT_DIR",
  unset = file.path(WE_PROJECT_ROOT, "outputs", "water_energy", "primary")
)
WE_INPUT_DIR <- Sys.getenv(
  "WATER_ENERGY_INPUT_DIR",
  unset = file.path(WE_PROJECT_ROOT, "data", "water_energy")
)
WE_OUTPUT_ROOT <- Sys.getenv(
  "WATER_ENERGY_ADJUSTMENT_OUTPUT_DIR",
  unset = file.path(WE_PROJECT_ROOT, "outputs", "water_energy", "adjustment")
)
WE_MASTER_PATH <- Sys.getenv(
  "WATER_ENERGY_MASTER_PATH",
  unset = file.path(WE_INPUT_DIR, "water_energy_analysis_ready_2022_2025.csv")
)
WE_REFERENCE_SUPPORT_PATH <- Sys.getenv("WATER_ENERGY_SUPPORT_PATH", unset = file.path(WE_INPUT_DIR, "primary_contrast_support.csv"))

WE_DIRS <- c(
  "01_CODE", "02_MODEL2", "03_MODEL3", "04_BOOTSTRAP", "05_DIAGNOSTICS",
  "06_SUPPORT", "07_COMPARISON", "08_TESTS", "09_LOGS"
)

WE_SEED <- 20260803L
WE_BOOT_DRAWS <- 1050L
WE_BOOT_TARGET_SUCCESS <- 1000L
WE_BOOT_INITIAL_BATCH <- 50L
WE_BOOT_MAX_FAILURE_RATE <- 0.05
WE_RHO_FIXED <- 0.036623315093487
WE_SUPPORT_RADIUS <- 0.10
WE_SUPPORT_MIN_RECORDS <- 100L
WE_SUPPORT_MIN_SITES <- 50L
WE_SUPPORT_MIN_YEARS <- 3L
WE_CONTRAST_A <- c(precip_rank_baseline = 0.75, radiation_rank_baseline = 0.25)
WE_CONTRAST_B <- c(precip_rank_baseline = 0.75, radiation_rank_baseline = 0.50)
WE_MODEL1_CONTROLS <- c(
  "precip_antecedent3_sum_mm", "radiation_antecedent3_mean_MJ_m2",
  "soilwater_lag1", "ndvi_lag1_canonical"
)
WE_MODEL2_ADDED <- c("temperature_anomaly", "VPD_anomaly")
WE_MODEL3_ADDED <- c("soilwater_anomaly", "NDVI_anomaly")
WE_ALL_CONTROLS <- c(WE_MODEL1_CONTROLS, WE_MODEL2_ADDED, WE_MODEL3_ADDED)
WE_ALLOWED_ANALYSES <- c(
  "MODEL2_ADJUSTMENT_ROBUSTNESS", "MODEL3_HIGH_ADJUSTMENT_PRESSURE_TEST"
)
WE_MODEL_NAMES <- c("MODEL2", "MODEL3")
WE_EXPECTED_CONTRAST <- c(
  MODEL2 = -4.05456162116481,
  MODEL3 = -3.73188716350626
)
WE_PRIMARY_REFERENCE <- list(
  n = 6992L, n_sites = 247L, ar_sections = 988L,
  rho = WE_RHO_FIXED,
  contrast = -2.94258224191719,
  ci_lower = -3.55908622452363,
  ci_upper = -2.21122658573456
)

we_ensure_dirs()

we_log <- function(..., file = we_path("09_LOGS", "WE_ADJUSTMENT_ROBUSTNESS_RUN.log")) {
  stamp <- format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z")
  msg <- paste(..., collapse = " ")
  cat(sprintf("[%s] %s\n", stamp, msg), file = file, append = TRUE)
  message(msg)
  invisible(msg)
}

we_stop_check <- function(reason, stage, detail = "") {
  status <- data.frame(
    timestamp = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"), stage = stage,
    reason = reason, detail = detail,
    final_status = paste0("STOPPED — analysis check required: ", reason),
    stringsAsFactors = FALSE
  )
  we_write_csv(status, we_path("09_LOGS", "WE_STOP_STATUS.csv"))
  we_log("STOP check:", reason, "stage=", stage, detail)
  stop(paste(reason, detail), call. = FALSE)
}

we_normalize_point_id <- function(x) sprintf("%03d", as.integer(as.character(x)))

we_flag_true <- function(x) !is.na(x) & (x == TRUE | x == 1L)

we_key <- function(d) {
  paste(
    we_normalize_point_id(d$point_id), as.integer(d$Year), as.integer(d$Month),
    sep = "|"
  )
}

we_model_frame_key <- function(model) {
  mf <- model$model
  required <- c("point_id", "factor(Year)", "factor(Month)")
  we_assert(all(required %in% names(mf)), "canonical model frame key fields are missing")
  paste(
    we_normalize_point_id(mf$point_id),
    as.integer(as.character(mf[["factor(Year)"]])),
    as.integer(as.character(mf[["factor(Month)"]])), sep = "|"
  )
}

we_model_path <- function(model_name) {
  model_name <- toupper(model_name)
  we_assert(model_name %in% WE_MODEL_NAMES, "Unknown adjustment model")
  model_dir <- if (model_name == "MODEL2") "04_MODEL2" else "05_MODEL3"
  file.path(WE_PRIMARY_ROOT, model_dir, paste0("WE_", model_name, ".rds"))
}

we_formula_path <- function(model_name) {
  model_name <- toupper(model_name)
  model_dir <- if (model_name == "MODEL2") "04_MODEL2" else "05_MODEL3"
  file.path(WE_PRIMARY_ROOT, model_dir, paste0("WE_", model_name, "_FORMULA.txt"))
}

we_model_info_path <- function(model_name) {
  model_name <- toupper(model_name)
  model_dir <- if (model_name == "MODEL2") "04_MODEL2" else "05_MODEL3"
  file.path(WE_PRIMARY_ROOT, model_dir, paste0("WE_", model_name, "_MODEL_INFO.csv"))
}

we_adjustment_formula <- function(model_name, site_variable = "point_id") {
  model_name <- toupper(model_name)
  we_assert(model_name %in% WE_MODEL_NAMES, "Unknown adjustment model formula")
  controls <- c(WE_MODEL1_CONTROLS, WE_MODEL2_ADDED)
  if (model_name == "MODEL3") controls <- c(controls, WE_MODEL3_ADDED)
  terms <- c(
    "te(precip_rank_baseline, radiation_rank_baseline, k = c(5, 5))",
    sprintf("s(%s, k = 5)", controls),
    "factor(Month)", "factor(Year)", sprintf("s(%s, bs = \"re\")", site_variable)
  )
  as.formula(paste("npp_anomaly_mean_g ~", paste(terms, collapse = " + ")))
}

we_formula_text <- function(formula) paste(deparse(formula, width.cutoff = 500L), collapse = " ")

we_normalize_formula <- function(x) gsub("[[:space:]]+", "", paste(x, collapse = ""))

we_prepare_primary_adjustment_data <- function(master) {
  d <- we_primary_data(master)
  d$point_id <- factor(
    we_normalize_point_id(d$point_id),
    levels = sort(unique(we_normalize_point_id(d$point_id)))
  )
  d$AR.start <- we_build_ar_start(d, "point_id")
  d
}

we_primary_reference_validation <- function() {
  comparison_path <- file.path(
    WE_PRIMARY_ROOT, "09_PRIMARY_CONTRAST", "WE_MODEL_CONTRAST_COMPARISON.csv"
  )
  primary_boot_path <- file.path(
    WE_PRIMARY_ROOT, "10_BOOTSTRAP", "WE_MODEL1_PRIMARY_CONTRAST_SUMMARY.csv"
  )
  comparison <- read.csv(comparison_path, stringsAsFactors = FALSE)
  primary_boot <- read.csv(primary_boot_path, stringsAsFactors = FALSE)
  m1 <- comparison[comparison$model == "Model1 PRIMARY", , drop = FALSE]
  checks <- c(
    primary_n = nrow(m1) == 1L && m1$n == WE_PRIMARY_REFERENCE$n,
    primary_sites = nrow(m1) == 1L && m1$n_sites == WE_PRIMARY_REFERENCE$n_sites,
    primary_rho = nrow(m1) == 1L && isTRUE(all.equal(
      m1$rho_used, WE_PRIMARY_REFERENCE$rho, tolerance = 1e-15
    )),
    primary_contrast = nrow(m1) == 1L && isTRUE(all.equal(
      m1$contrast_estimate, WE_PRIMARY_REFERENCE$contrast, tolerance = 1e-14
    )),
    primary_ci_lower = isTRUE(all.equal(
      primary_boot$percentile_2_5[1L], WE_PRIMARY_REFERENCE$ci_lower, tolerance = 1e-14
    )),
    primary_ci_upper = isTRUE(all.equal(
      primary_boot$percentile_97_5[1L], WE_PRIMARY_REFERENCE$ci_upper, tolerance = 1e-14
    ))
  )
  data.frame(
    check = names(checks), status = ifelse(checks, "PASS", "FAIL"),
    primary_refitted = FALSE, stringsAsFactors = FALSE
  )
}

we_canonical_contrast_table <- function() {
  x <- read.csv(file.path(
    WE_PRIMARY_ROOT, "09_PRIMARY_CONTRAST", "WE_MODEL_CONTRAST_COMPARISON.csv"
  ), stringsAsFactors = FALSE)
  x[x$model %in% c("Model1 PRIMARY", "Model2", "Model3"), , drop = FALSE]
}

we_domain_identity_validation <- function(data, models) {
  primary_keys <- we_key(data)
  rows <- lapply(names(models), function(model_name) {
    model_keys <- we_model_frame_key(models[[model_name]])
    missing <- setdiff(primary_keys, model_keys)
    extra <- setdiff(model_keys, primary_keys)
    data.frame(
      model = model_name, primary_n = length(primary_keys), model_n = length(model_keys),
      primary_sites = length(unique(as.character(data$point_id))),
      model_sites = length(unique(as.character(models[[model_name]]$model$point_id))),
      duplicate_primary_keys = sum(duplicated(primary_keys)),
      duplicate_model_keys = sum(duplicated(model_keys)),
      missing_keys = length(missing), extra_keys = length(extra),
      ordered_key_identity = identical(primary_keys, model_keys),
      set_key_identity = setequal(primary_keys, model_keys),
      status = if (
        length(primary_keys) == 6992L && length(unique(as.character(data$point_id))) == 247L &&
        !anyDuplicated(primary_keys) && !anyDuplicated(model_keys) &&
        !length(missing) && !length(extra)
      ) "PASS" else "FAIL",
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}

we_formula_validation <- function(models) {
  do.call(rbind, lapply(names(models), function(model_name) {
    expected <- we_formula_text(we_adjustment_formula(model_name, "point_id"))
    rds_formula <- we_formula_text(formula(models[[model_name]]))
    file_formula <- paste(readLines(we_formula_path(model_name), warn = FALSE), collapse = " ")
    variables <- all.vars(formula(models[[model_name]]))
    expected_variables <- all.vars(we_adjustment_formula(model_name, "point_id"))
    data.frame(
      model = model_name, expected_formula = expected, rds_formula = rds_formula,
      formula_file = file_formula,
      rds_matches_expected = identical(we_normalize_formula(rds_formula), we_normalize_formula(expected)),
      file_matches_expected = identical(we_normalize_formula(file_formula), we_normalize_formula(expected)),
      variable_set_identity = setequal(variables, expected_variables),
      formula_uniquely_determined = TRUE,
      status = if (
        identical(we_normalize_formula(rds_formula), we_normalize_formula(expected)) &&
        identical(we_normalize_formula(file_formula), we_normalize_formula(expected)) &&
        setequal(variables, expected_variables)
      ) "PASS" else "FAIL",
      stringsAsFactors = FALSE
    )
  }))
}

we_model_reference_validation <- function(models) {
  canonical <- we_canonical_contrast_table()
  do.call(rbind, lapply(names(models), function(model_name) {
    label <- if (model_name == "MODEL2") "Model2" else "Model3"
    row <- canonical[canonical$model == label, , drop = FALSE]
    info <- read.csv(we_model_info_path(model_name), stringsAsFactors = FALSE)
    expected_controls <- if (model_name == "MODEL2") {
      c(WE_MODEL1_CONTROLS, WE_MODEL2_ADDED)
    } else {
      c(WE_MODEL1_CONTROLS, WE_MODEL2_ADDED, WE_MODEL3_ADDED)
    }
    data.frame(
      model = model_name, canonical_contrast = row$contrast_estimate[1L],
      reference_expected_contrast = unname(WE_EXPECTED_CONTRAST[[model_name]]),
      contrast_reference_difference = row$contrast_estimate[1L] - WE_EXPECTED_CONTRAST[[model_name]],
      n = info$n[1L], n_sites = info$n_sites[1L], rho = info$rho_used[1L],
      family = info$family[1L], link = info$link[1L], method = info$method[1L],
      discrete = info$discrete[1L], convergence = info$convergence[1L],
      full_rank = info$full_rank[1L], warning_count = info$warning_count[1L],
      added_fields_complete = all(vapply(
        expected_controls, function(v) v %in% names(models[[model_name]]$model), logical(1)
      )),
      full_data_refitted = FALSE,
      status = if (
        nrow(row) == 1L && info$n[1L] == 6992L && info$n_sites[1L] == 247L &&
        isTRUE(all.equal(info$rho_used[1L], WE_RHO_FIXED, tolerance = 1e-15)) &&
        info$family[1L] == "gaussian" && info$link[1L] == "identity" &&
        info$method[1L] == "fREML" && !as.logical(info$discrete[1L]) &&
        info$convergence[1L] == "CONVERGED" && as.logical(info$full_rank[1L]) &&
        info$warning_count[1L] == 0L &&
        isTRUE(all.equal(row$contrast_estimate[1L], WE_EXPECTED_CONTRAST[[model_name]], tolerance = 1e-14))
      ) "PASS" else "FAIL",
      stringsAsFactors = FALSE
    )
  }))
}

we_ar_postfit_adjustment <- function(model, data, model_name) {
  model_keys <- we_model_frame_key(model)
  data_keys <- we_key(data)
  idx <- match(data_keys, model_keys)
  we_assert(all(!is.na(idx)), paste(model_name, "residual-key mapping failed"))
  std <- model$std.rsd
  source <- "canonical_model$std.rsd"
  if (is.null(std)) {
    std <- residuals(model, type = "response") / sqrt(summary(model)$scale)
    source <- "fallback_residual_div_sqrt_scale"
  }
  std <- as.numeric(std)[idx]
  pairs <- we_continuous_pairs(data, std, "point_id", "calendar_month_index")
  r <- if (nrow(pairs) >= 2L) cor(pairs$value_t_minus_1, pairs$value_t) else NA_real_
  abs_r <- abs(r)
  data.frame(
    model = model_name, standardized_residual_lag1 = r,
    absolute_standardized_residual_lag1 = abs_r,
    n_consecutive_pairs = nrow(pairs),
    n_contributing_sites = length(unique(pairs$series_id)),
    residual_source = source,
    check = if (!is.finite(abs_r)) "STOP" else if (abs_r <= 0.10) "PASS" else if (abs_r <= 0.20) "WARN" else "STOP",
    rho_reestimated = FALSE, rho_used = WE_RHO_FIXED,
    stringsAsFactors = FALSE
  )
}

we_adjustment_invariance <- function(model, data, model_name) {
  model_name <- toupper(model_name)
  used_controls <- if (model_name == "MODEL2") {
    c(WE_MODEL1_CONTROLS, WE_MODEL2_ADDED)
  } else {
    c(WE_MODEL1_CONTROLS, WE_MODEL2_ADDED, WE_MODEL3_ADDED)
  }
  base <- we_reference_row(data, "point_id", 0.50)
  main <- we_lpmatrix_contrast(model, base)$estimate
  rows <- list()
  add_scenario <- function(name, variable, value, ref) {
    est <- we_lpmatrix_contrast(model, ref)$estimate
    rows[[length(rows) + 1L]] <<- data.frame(
      model = model_name, scenario = name, varied_variable = variable,
      varied_value = as.character(value), contrast = est,
      difference_from_main = est - main,
      absolute_difference_from_main = abs(est - main),
      stringsAsFactors = FALSE
    )
  }
  add_scenario("main", "none", NA, base)
  month_values <- unique(as.integer(round(seq(min(data$Month), max(data$Month), length.out = 3L))))
  for (v in month_values) {
    ref <- base; ref$Month <- v; add_scenario(paste0("Month_", v), "Month", v, ref)
  }
  year_values <- range(data$Year)
  for (v in year_values) {
    ref <- base; ref$Year <- v; add_scenario(paste0("Year_", v), "Year", v, ref)
  }
  point_values <- levels(data$point_id)[unique(as.integer(round(seq(1, nlevels(data$point_id), length.out = 3L))))]
  for (v in point_values) {
    ref <- base; ref$point_id <- factor(v, levels = levels(data$point_id))
    add_scenario(paste0("point_id_", v), "point_id", v, ref)
  }
  for (variable in used_controls) for (prob in c(0.25, 0.50, 0.75)) {
    value <- unname(quantile(data[[variable]], prob, na.rm = TRUE, type = 7))
    ref <- base; ref[[variable]] <- value
    add_scenario(
      paste0(variable, "_q", format(prob, nsmall = 2L)), variable,
      format(value, digits = 16), ref
    )
  }
  validation <- do.call(rbind, rows)
  max_diff <- max(validation$absolute_difference_from_main)
  list(
    main_contrast = main, validation = validation,
    max_absolute_difference = max_diff,
    pass = is.finite(main) && all(is.finite(validation$contrast)) && max_diff <= 1e-8
  )
}

we_support_validation_adjustment <- function(data) {
  points <- list(A = WE_CONTRAST_A, B = WE_CONTRAST_B)
  expected <- data.frame(
    contrast_point = c("A", "B"), expected_records = c(401L, 339L),
    expected_sites = c(215L, 182L), expected_years = c(4L, 4L),
    stringsAsFactors = FALSE
  )
  out <- do.call(rbind, lapply(names(points), function(label) {
    target <- points[[label]]
    distance <- sqrt(
      (data$precip_rank_baseline - target[["precip_rank_baseline"]])^2 +
        (data$radiation_rank_baseline - target[["radiation_rank_baseline"]])^2
    )
    keep <- is.finite(distance) & distance <= WE_SUPPORT_RADIUS
    data.frame(
      contrast_point = label,
      precip_rank_target = target[["precip_rank_baseline"]],
      radiation_rank_target = target[["radiation_rank_baseline"]],
      radius = WE_SUPPORT_RADIUS, n_records = sum(keep),
      n_sites = length(unique(as.character(data$point_id[keep]))),
      n_years = length(unique(data$Year[keep])),
      minimum_records = WE_SUPPORT_MIN_RECORDS,
      minimum_sites = WE_SUPPORT_MIN_SITES,
      minimum_years = WE_SUPPORT_MIN_YEARS,
      stringsAsFactors = FALSE
    )
  }))
  out <- merge(out, expected, by = "contrast_point", sort = FALSE)
  out$support_check_pass <- with(
    out, n_records >= minimum_records & n_sites >= minimum_sites & n_years >= minimum_years
  )
  out$matches_canonical_primary <- with(
    out, n_records == expected_records & n_sites == expected_sites & n_years == expected_years
  )
  out$status <- ifelse(
    out$support_check_pass & out$matches_canonical_primary, "PASS", "FAIL"
  )
  out[match(c("A", "B"), out$contrast_point), , drop = FALSE]
}

we_term_has <- function(x, variable) grepl(variable, x, fixed = TRUE)

we_concurvity_summary_adjustment <- function(concurvity_table, model_name) {
  surface_pattern <- "precip_rank_baseline"
  is_surface_from <- we_term_has(concurvity_table$term_from, surface_pattern)
  is_surface_to <- we_term_has(concurvity_table$term_to, surface_pattern)
  non_diagonal <- !concurvity_table$is_pairwise_diagonal
  finite_max <- function(x) if (any(is.finite(x))) max(x[is.finite(x)]) else NA_real_
  surface_full <- concurvity_table$mode == "full" & concurvity_table$metric == "worst" & is_surface_to
  surface_pair <- concurvity_table$mode == "pairwise" & concurvity_table$metric == "worst" &
    non_diagonal & (is_surface_from | is_surface_to)
  highest_rows <- concurvity_table[
    is.finite(concurvity_table$value) & non_diagonal & concurvity_table$metric == "worst", , drop = FALSE
  ]
  highest <- highest_rows[which.max(highest_rows$value), , drop = FALSE]
  controls <- if (toupper(model_name) == "MODEL2") {
    c(WE_MODEL1_CONTROLS, WE_MODEL2_ADDED)
  } else {
    c(WE_MODEL1_CONTROLS, WE_MODEL2_ADDED, WE_MODEL3_ADDED)
  }
  control_mask <- Reduce(`|`, lapply(controls, function(v) {
    we_term_has(concurvity_table$term_from, v) | we_term_has(concurvity_table$term_to, v)
  }))
  control_rows <- concurvity_table[
    is.finite(concurvity_table$value) & non_diagonal &
      concurvity_table$metric == "worst" & control_mask, , drop = FALSE
  ]
  control_max <- control_rows[which.max(control_rows$value), , drop = FALSE]
  surface_full_value <- finite_max(concurvity_table$value[surface_full])
  surface_pair_value <- finite_max(concurvity_table$value[surface_pair])
  data.frame(
    model = toupper(model_name),
    surface_full_worst = surface_full_value,
    surface_full_label = we_concurvity_label(surface_full_value),
    surface_pairwise_max = surface_pair_value,
    surface_pairwise_label = we_concurvity_label(surface_pair_value),
    highest_overall_term_from = highest$term_from[1L],
    highest_overall_term_to = highest$term_to[1L],
    highest_overall_concurvity = highest$value[1L],
    highest_overall_label = we_concurvity_label(highest$value[1L]),
    highest_control_term_from = control_max$term_from[1L],
    highest_control_term_to = control_max$term_to[1L],
    highest_control_concurvity = control_max$value[1L],
    highest_control_label = we_concurvity_label(control_max$value[1L]),
    stringsAsFactors = FALSE
  )
}

we_surface_control_concurvity <- function(concurvity_table, model_name) {
  controls <- if (toupper(model_name) == "MODEL2") {
    c(WE_MODEL2_ADDED)
  } else {
    c(WE_MODEL2_ADDED, WE_MODEL3_ADDED)
  }
  do.call(rbind, lapply(controls, function(variable) {
    surface_from <- we_term_has(concurvity_table$term_from, "precip_rank_baseline")
    surface_to <- we_term_has(concurvity_table$term_to, "precip_rank_baseline")
    control_from <- we_term_has(concurvity_table$term_from, variable)
    control_to <- we_term_has(concurvity_table$term_to, variable)
    keep <- concurvity_table$mode == "pairwise" & concurvity_table$metric == "worst" &
      !concurvity_table$is_pairwise_diagonal &
      ((surface_from & control_to) | (control_from & surface_to))
    rows <- concurvity_table[keep & is.finite(concurvity_table$value), , drop = FALSE]
    if (!nrow(rows)) {
      return(data.frame(
        model = toupper(model_name), control = variable,
        max_pairwise_worst = NA_real_, label = "NOT_AVAILABLE",
        term_from = NA_character_, term_to = NA_character_, stringsAsFactors = FALSE
      ))
    }
    row <- rows[which.max(rows$value), , drop = FALSE]
    data.frame(
      model = toupper(model_name), control = variable,
      max_pairwise_worst = row$value[1L],
      label = we_concurvity_label(row$value[1L]),
      term_from = row$term_from[1L], term_to = row$term_to[1L],
      stringsAsFactors = FALSE
    )
  }))
}

we_model_residual_vectors <- function(model, data) {
  model_keys <- we_model_frame_key(model)
  idx <- match(we_key(data), model_keys)
  we_assert(all(!is.na(idx)), "canonical model residual-key mapping failed")
  raw <- as.numeric(residuals(model, type = "response"))[idx]
  fitted_values <- as.numeric(fitted(model))[idx]
  std <- model$std.rsd
  if (is.null(std)) std <- residuals(model, type = "response") / sqrt(summary(model)$scale)
  std <- as.numeric(std)[idx]
  list(raw = raw, fitted = fitted_values, std = std)
}

we_plot_png <- function(filename, expression, width = 1800L, height = 1400L) {
  png(filename, width = width, height = height, res = 180)
  on.exit(dev.off(), add = TRUE)
  force(expression)
}

we_run_adjustment_boot_attempt <- function(attempt, model_name, data, draw_sequence) {
  started <- proc.time()[["elapsed"]]
  model_name <- toupper(model_name)
  draw_rows <- draw_sequence[draw_sequence$attempt == attempt, , drop = FALSE]
  base <- list(
    model = model_name, attempt = as.integer(attempt),
    draw_id = sprintf("draw_%04d", as.integer(attempt)), success = FALSE,
    contrast = NA_real_, convergence = "NOT_FIT", warning = "",
    failure_reason = "", failure_category = "", n = NA_integer_,
    n_boot_series = NA_integer_, n_AR_sections = NA_integer_,
    elapsed_seconds = NA_real_
  )
  result <- tryCatch({
    boot <- we_build_boot_data(data, draw_rows)
    formula <- we_adjustment_formula(model_name, "boot_series_id")
    fit <- we_fit_bam(formula, boot, WE_RHO_FIXED, boot$AR.start)
    health <- we_model_health(fit$model, fit$warnings)
    reasons <- we_model_health_stop_reason(health)
    contrast <- NA_real_
    if (!length(reasons)) {
      ref <- we_reference_row(boot, "boot_series_id", 0.50)
      contrast <- we_lpmatrix_contrast(fit$model, ref)$estimate
      if (!is.finite(contrast)) reasons <- c(reasons, "NONFINITE CONTRAST")
    }
    reason_text <- paste(unique(reasons), collapse = ";")
    list(
      model = model_name, attempt = as.integer(attempt),
      draw_id = sprintf("draw_%04d", as.integer(attempt)),
      success = length(reasons) == 0L && is.finite(contrast),
      contrast = contrast, convergence = health$convergence[1L],
      warning = paste(fit$warnings, collapse = " | "),
      failure_reason = reason_text,
      failure_category = we_boot_failure_category(reason_text),
      n = nrow(boot), n_boot_series = nlevels(boot$boot_series_id),
      n_AR_sections = sum(boot$AR.start),
      elapsed_seconds = proc.time()[["elapsed"]] - started
    )
  }, error = function(e) {
    msg <- conditionMessage(e)
    base$failure_reason <- msg
    base$failure_category <- we_boot_failure_category(msg)
    base$elapsed_seconds <- proc.time()[["elapsed"]] - started
    base
  })
  gc(verbose = FALSE)
  as.data.frame(result, stringsAsFactors = FALSE)
}

we_adjustment_bootstrap_summary <- function(model_name, values, full_estimate, n_attempt) {
  data.frame(
    model = toupper(model_name), full_estimate = full_estimate,
    bootstrap_mean = mean(values), bootstrap_median = median(values),
    bootstrap_SD = sd(values),
    percentile_2_5 = unname(quantile(values, 0.025, type = 7)),
    percentile_97_5 = unname(quantile(values, 0.975, type = 7)),
    bootstrap_min = min(values), bootstrap_max = max(values),
    negative_replicate_fraction = mean(values < 0),
    positive_replicate_fraction = mean(values > 0),
    zero_replicate_fraction = mean(values == 0),
    n_success = length(values), n_attempt = n_attempt,
    n_fail = n_attempt - length(values),
    failure_fraction = (n_attempt - length(values)) / n_attempt,
    seed = WE_SEED, rho_used = WE_RHO_FIXED,
    interval_method = "site-cluster bootstrap percentile 2.5% and 97.5%",
    stringsAsFactors = FALSE
  )
}

we_adjustment_check <- function(model_name, contrast, ci_upper, support_pass, diagnostic_stop = FALSE) {
  model_name <- toupper(model_name)
  if (diagnostic_stop) return(paste0(model_name, "_DIAGNOSTIC_STOP"))
  if (!support_pass) return(paste0(model_name, "_FIXED_CONTRAST_UNSUPPORTED"))
  if (contrast > 0) return(paste0(model_name, "_SIGN_REVERSAL_ANALYSIS_CHECK_REQUIRED"))
  if (model_name == "MODEL2") {
    if (contrast < 0 && ci_upper < 0) return("MODEL2_ADJUSTMENT_ROBUSTNESS_PASS")
    if (contrast < 0) return("MODEL2_DIRECTIONALLY_CONSISTENT_BUT_IMPRECISE")
  } else {
    if (contrast < 0 && ci_upper < 0) {
      return("MODEL3_HIGH_ADJUSTMENT_DIRECTION_AND_INTERVAL_CONSISTENT")
    }
    if (contrast < 0) return("MODEL3_DIRECTIONALLY_CONSISTENT_BUT_IMPRECISE")
  }
  paste0(model_name, "_ZERO_CONTRAST_ANALYSIS_CHECK_REQUIRED")
}

we_analysis_selector_validation <- function() {
  data.frame(
    item = c(
      "analyses_run", "new_sensitivity", "Primary_Model1_refit", "variable_selection",
      "variable_deletion", "rho_changed", "k_changed", "contrast_changed", "LOYO",
      "analysis_definition_changed", "network_download"
    ),
    value = c(
      paste(WE_ALLOWED_ANALYSES, collapse = ";"), rep("NO", 10L)
    ),
    status = "PASS", stringsAsFactors = FALSE
  )
}
