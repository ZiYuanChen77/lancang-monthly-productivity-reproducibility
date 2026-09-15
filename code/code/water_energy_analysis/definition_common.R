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
  "WATER_ENERGY_DEFINITION_OUTPUT_DIR",
  unset = file.path(WE_PROJECT_ROOT, "outputs", "water_energy", "definition")
)
WE_MASTER_PATH <- Sys.getenv(
  "WATER_ENERGY_MASTER_PATH",
  unset = file.path(WE_INPUT_DIR, "water_energy_analysis_ready_2022_2025.csv")
)
WE_TARGET_ROOT <- Sys.getenv("WATER_ENERGY_QA_DAYWEIGHT_OUTPUT_DIR", unset = file.path(WE_PROJECT_ROOT, "outputs", "water_energy", "qa_dayweight"))
WE_DOMAIN_ROOT <- Sys.getenv("WATER_ENERGY_DOMAIN_OUTPUT_DIR", unset = file.path(WE_PROJECT_ROOT, "outputs", "water_energy", "domain"))
WE_RAW_MONTHLY_PATH <- Sys.getenv("WATER_ENERGY_RAW_MONTHLY_PATH", unset = file.path(WE_INPUT_DIR, "monthly_model_input.csv"))
WE_COVARIATE_BASELINE_PATH <- Sys.getenv("WATER_ENERGY_COVARIATE_BASELINE_PATH", unset = file.path(WE_INPUT_DIR, "covariate_baseline_site_month_2005_2018.csv"))
WE_CANONICAL_BASELINE_PATH <- Sys.getenv("WATER_ENERGY_CANONICAL_BASELINE_PATH", unset = file.path(WE_INPUT_DIR, "canonical_baseline_site_month_2005_2018.csv"))

WE_DIRS <- c(
  "01_CODE", "02_MEDIAN_RESPONSE", "03_RAW_EXPOSURE", "04_RAW_BASELINE_MAPPING",
  "05_SUPPORT", "06_BOOTSTRAP", "07_DIAGNOSTICS", "08_COMPARISON",
  "09_TESTS", "10_LOGS"
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
WE_MODEL1_CONTROLS <- c(
  "precip_antecedent3_sum_mm", "radiation_antecedent3_mean_MJ_m2",
  "soilwater_lag1", "ndvi_lag1_canonical"
)
WE_ALL_CONTROLS <- WE_MODEL1_CONTROLS
WE_ALLOWED_ANALYSES <- c("MEDIAN_RESPONSE", "RAW_EXPOSURE")
WE_RAW_ACTUAL_FIELDS <- c(
  precip = "precip_anomaly_mm", radiation = "radiation_anomaly_MJ_m2"
)
WE_RAW_SOURCE_FIELDS <- c(
  precip = "Precipitation_mm", radiation = "surface_solar_radiation_downwards_sum"
)
WE_RANK_CONTRAST_A <- c(
  precip_rank_baseline = 0.75, radiation_rank_baseline = 0.25
)
WE_RANK_CONTRAST_B <- c(
  precip_rank_baseline = 0.75, radiation_rank_baseline = 0.50
)
WE_ANALYSIS_META <- list(
  MEDIAN_RESPONSE = list(
    output_dir = "02_MEDIAN_RESPONSE", response = "npp_anomaly_median_g",
    exposure_x = "precip_rank_baseline", exposure_y = "radiation_rank_baseline",
    response_definition = "canonical monthly productivity minus 2005-2018 same point_id x Month historical median",
    exposure_definition = "canonical site-month baseline-relative precipitation and radiation ranks",
    role = "one-factor median-response definition sensitivity"
  ),
  RAW_EXPOSURE = list(
    output_dir = "03_RAW_EXPOSURE", response = "npp_anomaly_mean_g",
    exposure_x = "precip_anomaly_mm", exposure_y = "radiation_anomaly_MJ_m2",
    response_definition = "canonical Primary canonical mean anomaly",
    exposure_definition = "current-month raw value minus 2005-2018 same point_id x Month historical mean",
    role = "one-factor raw-exposure parameterization sensitivity"
  )
)

we_ensure_dirs()

we_log <- function(..., file = we_path("10_LOGS", "WE_DEFINITION_SENSITIVITY_RUN.log")) {
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
  we_write_csv(status, we_path("10_LOGS", "WE_STOP_STATUS.csv"))
  we_log("STOP check:", reason, "stage=", stage, detail)
  stop(paste(reason, detail), call. = FALSE)
}

we_flag_true <- function(x) !is.na(x) & (x == TRUE | x == 1L)

we_normalize_point_id <- function(x) sprintf("%03d", as.integer(as.character(x)))

we_key <- function(d) {
  paste(we_normalize_point_id(d$point_id), as.integer(d$Year), as.integer(d$Month), sep = "|")
}

we_primary_identity_validation <- function(master) {
  explicit <- we_flag_true(master$eligible_primary_model1)
  primary_keys <- sort(we_key(master[explicit, , drop = FALSE]))
  model_path <- file.path(
    WE_PRIMARY_ROOT, "03_MODEL1_PRIMARY", "WE_MODEL1_PRIMARY.rds"
  )
  model <- readRDS(model_path)
  mf <- model$model
  required <- c("point_id", "factor(Year)", "factor(Month)")
  missing <- setdiff(required, names(mf))
  if (length(missing)) {
    return(data.frame(
      check = "canonical_primary_model_frame_keys", master_primary_n = length(primary_keys),
      canonical_model_n = nrow(mf), missing_keys = NA_integer_, extra_keys = NA_integer_,
      duplicate_master_keys = anyDuplicated(primary_keys),
      duplicate_model_keys = NA_integer_, status = "FAIL_MODEL_FRAME_FIELDS_MISSING",
      detail = paste(missing, collapse = ";"), stringsAsFactors = FALSE
    ))
  }
  model_keys <- sort(paste(
    we_normalize_point_id(mf$point_id),
    as.integer(as.character(mf[["factor(Year)"]])),
    as.integer(as.character(mf[["factor(Month)"]])), sep = "|"
  ))
  missing_keys <- setdiff(primary_keys, model_keys)
  extra_keys <- setdiff(model_keys, primary_keys)
  pass <- length(primary_keys) == 6992L && length(unique(sub("\\|.*", "", primary_keys))) == 247L &&
    !anyDuplicated(primary_keys) && !anyDuplicated(model_keys) &&
    !length(missing_keys) && !length(extra_keys)
  data.frame(
    check = "canonical_primary_model_frame_keys", master_primary_n = length(primary_keys),
    canonical_model_n = length(model_keys), missing_keys = length(missing_keys),
    extra_keys = length(extra_keys), duplicate_master_keys = anyDuplicated(primary_keys),
    duplicate_model_keys = anyDuplicated(model_keys),
    status = if (pass) "PASS" else "FAIL", detail = "exact point_id-Year-Month set comparison",
    stringsAsFactors = FALSE
  )
}

we_reference_reference_validation <- function() {
  primary_info <- read.csv(file.path(
    WE_PRIMARY_ROOT, "03_MODEL1_PRIMARY", "WE_MODEL1_PRIMARY_MODEL_INFO.csv"
  ), stringsAsFactors = FALSE)
  primary_boot <- read.csv(file.path(
    WE_PRIMARY_ROOT, "10_BOOTSTRAP", "WE_MODEL1_PRIMARY_CONTRAST_SUMMARY.csv"
  ), stringsAsFactors = FALSE)
  primary_k <- read.csv(file.path(
    WE_PRIMARY_ROOT, "07_BASIS_DIAGNOSTICS", "WE_MODEL1_K_FINAL_CONFIGURATION.csv"
  ), stringsAsFactors = FALSE)
  target <- read.csv(file.path(
    WE_TARGET_ROOT, "08_COMPARISON", "WE_QA_DAYWEIGHT_CORE_COMPARISON.csv"
  ), stringsAsFactors = FALSE)
  domain <- read.csv(file.path(
    WE_DOMAIN_ROOT, "07_COMPARISON", "WE_DOMAIN_SENSITIVITY_CORE_COMPARISON.csv"
  ), stringsAsFactors = FALSE)
  tv <- function(a, f) target[[f]][target$analysis == a]
  dv <- function(a, f) domain[[f]][domain$analysis == a]
  checks <- c(
    primary_n = primary_info$n == 6992L,
    primary_sites = primary_info$n_sites == 247L,
    primary_rho = isTRUE(all.equal(primary_info$rho_used, WE_RHO_FIXED, tolerance = 1e-15)),
    primary_contrast = isTRUE(all.equal(primary_boot$full_estimate, -2.94258224191719, tolerance = 1e-14)),
    primary_ci = isTRUE(all.equal(
      c(primary_boot$percentile_2_5, primary_boot$percentile_97_5),
      c(-3.55908622452363, -2.21122658573456), tolerance = 1e-14
    )),
    primary_surface_k = primary_k$final_k[primary_k$term == "hydro_energy_surface"] == 5L,
    primary_controls_k = all(primary_k$final_k[match(WE_MODEL1_CONTROLS, primary_k$term)] == 5L),
    QA1 = isTRUE(all.equal(c(tv("QA1", "contrast"), tv("QA1", "CI_lower"), tv("QA1", "CI_upper")),
                            c(-2.74933816863286, -3.26893919769731, -1.73797044794133), tolerance = 1e-14)),
    QA2 = isTRUE(all.equal(c(tv("QA2", "contrast"), tv("QA2", "CI_lower"), tv("QA2", "CI_upper")),
                            c(-1.27535872111437, -2.41855680785968, -0.368551825715992), tolerance = 1e-14)),
    dayweighted = isTRUE(all.equal(c(tv("Day-weighted", "contrast"), tv("Day-weighted", "CI_lower"), tv("Day-weighted", "CI_upper")),
                                    c(-3.79465651925158, -4.56331310321137, -3.08974619145371), tolerance = 1e-14)),
    full_year = isTRUE(all.equal(c(dv("Full-year", "contrast"), dv("Full-year", "CI_lower"), dv("Full-year", "CI_upper")),
                                  c(-1.73973148523244, -2.38440010037769, -1.18741300668684), tolerance = 1e-14)),
    apr_sep = isTRUE(all.equal(c(dv("Apr-Sep", "contrast"), dv("Apr-Sep", "CI_lower"), dv("Apr-Sep", "CI_upper")),
                                c(-3.09098284750588, -3.95091986389233, -2.09774303555844), tolerance = 1e-14))
  )
  data.frame(
    check = names(checks), status = ifelse(checks, "PASS", "FAIL"),
    refitted = FALSE, stringsAsFactors = FALSE
  )
}

we_read_reference_raw <- function() {
  if (!file.exists(WE_RAW_MONTHLY_PATH)) {
    we_stop_check("STOP — RAW EXPOSURE ANOMALY NOT UNIQUELY RECONSTRUCTIBLE",
                   "RAW_SOURCE_check", WE_RAW_MONTHLY_PATH)
  }
  raw <- read.csv(WE_RAW_MONTHLY_PATH, check.names = TRUE, stringsAsFactors = FALSE)
  required <- c("point_id", "Year", "Month", unname(WE_RAW_SOURCE_FIELDS))
  missing <- setdiff(required, names(raw))
  if (length(missing)) {
    we_stop_check("STOP — RAW EXPOSURE ANOMALY NOT UNIQUELY RECONSTRUCTIBLE",
                   "RAW_SOURCE_check", paste(missing, collapse = ";"))
  }
  raw$point_id <- we_normalize_point_id(raw$point_id)
  raw$Year <- as.integer(raw$Year)
  raw$Month <- as.integer(raw$Month)
  raw$precipitation_mm <- suppressWarnings(as.numeric(raw[[WE_RAW_SOURCE_FIELDS[["precip"]]]]))
  raw$radiation_MJ_m2 <- suppressWarnings(as.numeric(raw[[WE_RAW_SOURCE_FIELDS[["radiation"]]]]))
  raw$precipitation_mm[raw$precipitation_mm == -9999] <- NA_real_
  raw$radiation_MJ_m2[raw$radiation_MJ_m2 == -9999] <- NA_real_
  raw
}

we_build_raw_mapping <- function() {
  raw <- we_read_reference_raw()
  covar <- read.csv(WE_COVARIATE_BASELINE_PATH, stringsAsFactors = FALSE)
  active <- read.csv(WE_CANONICAL_BASELINE_PATH, stringsAsFactors = FALSE)
  covar$point_id <- we_normalize_point_id(covar$point_id)
  active$point_id <- we_normalize_point_id(active$point_id)
  key <- c("point_id", "Month")
  historical <- raw[raw$Year >= 2005L & raw$Year <= 2018L,
                    c("point_id", "Year", "Month", "precipitation_mm", "radiation_MJ_m2"),
                    drop = FALSE]
  expected_mean <- stats::aggregate(
    cbind(precipitation_mm, radiation_MJ_m2) ~ point_id + Month,
    historical, mean, na.rm = TRUE
  )
  names(expected_mean)[names(expected_mean) == "precipitation_mm"] <- "precip_mean_recomputed"
  names(expected_mean)[names(expected_mean) == "radiation_MJ_m2"] <- "radiation_mean_recomputed"
  baseline_validation <- merge(
    covar[, c(key, "precipitation_mm_baseline_mean", "radiation_MJ_m2_baseline_mean")],
    expected_mean, by = key, all = TRUE, sort = FALSE
  )
  baseline_validation$precip_baseline_difference <- with(
    baseline_validation, precipitation_mm_baseline_mean - precip_mean_recomputed
  )
  baseline_validation$radiation_baseline_difference <- with(
    baseline_validation, radiation_MJ_m2_baseline_mean - radiation_mean_recomputed
  )
  historical <- merge(
    historical,
    covar[, c(key, "precipitation_mm_baseline_mean", "radiation_MJ_m2_baseline_mean",
              "precipitation_mm_baseline_n", "radiation_MJ_m2_baseline_n")],
    by = key, all.x = TRUE, sort = FALSE
  )
  historical <- merge(
    historical, active[, c(key, "historically_active_canonical")],
    by = key, all.x = TRUE, sort = FALSE
  )
  historical$precip_anomaly_mm <- with(
    historical, precipitation_mm - precipitation_mm_baseline_mean
  )
  historical$radiation_anomaly_MJ_m2 <- with(
    historical, radiation_MJ_m2 - radiation_MJ_m2_baseline_mean
  )
  active_history <- historical[we_flag_true(historical$historically_active_canonical), , drop = FALSE]
  pool <- active_history[
    is.finite(active_history$precip_anomaly_mm) &
      is.finite(active_history$radiation_anomaly_MJ_m2), , drop = FALSE
  ]
  pool <- pool[order(pool$point_id, pool$Month, pool$Year), , drop = FALSE]
  rownames(pool) <- NULL
  if (!nrow(pool)) {
    we_stop_check("STOP — RAW EXPOSURE ANOMALY NOT UNIQUELY RECONSTRUCTIBLE",
                   "RAW_BASELINE_POOL_check", "empty baseline pool")
  }
  group_key <- paste(pool$point_id, pool$Month, sep = "|")
  group_counts <- as.numeric(table(group_key))
  precip_group_mean <- tapply(pool$precip_anomaly_mm, group_key, mean)
  radiation_group_mean <- tapply(pool$radiation_anomaly_MJ_m2, group_key, mean)
  max_precip_center <- max(abs(precip_group_mean))
  max_radiation_center <- max(abs(radiation_group_mean))
  max_center <- max(max_precip_center, max_radiation_center)
  centering_status <- if (max_center <= 1e-10) "PASS" else "FAIL_SYSTEMATIC"
  validation <- data.frame(
    n_records = nrow(pool), n_sites = length(unique(pool$point_id)),
    n_site_month_groups = length(unique(group_key)),
    years_represented = paste(sort(unique(pool$Year)), collapse = ";"),
    records_per_group_min = min(group_counts), records_per_group_median = median(group_counts),
    records_per_group_mean = mean(group_counts), records_per_group_max = max(group_counts),
    active_historical_records_before_finite_check = nrow(active_history),
    missing_precipitation_anomaly = sum(!is.finite(active_history$precip_anomaly_mm)),
    missing_radiation_anomaly = sum(!is.finite(active_history$radiation_anomaly_MJ_m2)),
    max_abs_precip_group_mean_anomaly = max_precip_center,
    max_abs_radiation_group_mean_anomaly = max_radiation_center,
    centering_tolerance = 1e-10, centering_status = centering_status,
    baseline_year_start = 2005L, baseline_year_end = 2018L,
    evaluation_years_used_for_coordinates = FALSE,
    historically_active_groups_only = TRUE, stringsAsFactors = FALSE
  )
  pwet <- unname(quantile(pool$precip_anomaly_mm, probs = 0.75, type = 1, na.rm = TRUE))
  rdim <- unname(quantile(pool$radiation_anomaly_MJ_m2, probs = 0.25, type = 1, na.rm = TRUE))
  rmid <- unname(quantile(pool$radiation_anomaly_MJ_m2, probs = 0.50, type = 1, na.rm = TRUE))
  empirical_rank <- function(x, baseline) {
    sorted <- sort(baseline[is.finite(baseline)])
    less_equal <- findInterval(x, sorted)
    less <- findInterval(x, sorted, left.open = TRUE)
    (less + 0.5 * (less_equal - less) + 0.5) / (length(sorted) + 1)
  }
  coords <- data.frame(
    Pwet_raw_mm = pwet, Rdim_raw_MJ_m2 = rdim, Rmid_raw_MJ_m2 = rmid,
    baseline_pool_n = nrow(pool), quantile_type = 1L,
    derivation_years = "2005-2018",
    Pwet_raw_support_rank_actual = empirical_rank(pwet, pool$precip_anomaly_mm),
    Rdim_raw_support_rank_actual = empirical_rank(rdim, pool$radiation_anomaly_MJ_m2),
    Rmid_raw_support_rank_actual = empirical_rank(rmid, pool$radiation_anomaly_MJ_m2),
    coordinate_method = "baseline_quantiles",
    stringsAsFactors = FALSE
  )
  list(
    raw = raw, historical = historical, active_history = active_history, pool = pool,
    baseline_validation = baseline_validation, validation = validation, coordinates = coords,
    empirical_rank = empirical_rank
  )
}

we_prepare_primary_domain <- function(master, raw_mapping = NULL) {
  if (!"npp_anomaly_median_g" %in% names(master)) {
    we_stop_check("STOP — APPROVED MEDIAN RESPONSE FIELD NOT FOUND", "MEDIAN_FIELD_check")
  }
  required <- c(
    "point_id", "Year", "Month", "calendar_month_index", "eligible_primary_model1",
    "npp_anomaly_mean_g", "npp_anomaly_median_g", "precip_rank_baseline",
    "radiation_rank_baseline", unname(WE_RAW_ACTUAL_FIELDS), WE_MODEL1_CONTROLS
  )
  missing <- setdiff(required, names(master))
  if (length(missing)) {
    reason <- if (any(unname(WE_RAW_ACTUAL_FIELDS) %in% missing))
      "STOP — RAW EXPOSURE ANOMALY NOT UNIQUELY RECONSTRUCTIBLE" else
      "STOP — PRIMARY DOMAIN IDENTITY MISMATCH"
    we_stop_check(reason, "PRIMARY_DOMAIN_FIELD_check", paste(missing, collapse = ";"))
  }
  keep <- we_flag_true(master$eligible_primary_model1)
  d <- master[keep, , drop = FALSE]
  if (nrow(d) != 6992L || length(unique(d$point_id)) != 247L) {
    we_stop_check("STOP — PRIMARY DOMAIN IDENTITY MISMATCH", "PRIMARY_DOMAIN_COUNT_check",
                   sprintf("n=%d sites=%d", nrow(d), length(unique(d$point_id))))
  }
  numeric_required <- setdiff(required, c("point_id", "eligible_primary_model1"))
  nonfinite <- vapply(numeric_required, function(v) any(!is.finite(d[[v]])), logical(1))
  if (any(nonfinite)) {
    we_stop_check("STOP — PRIMARY DOMAIN IDENTITY MISMATCH", "PRIMARY_DOMAIN_FINITE_check",
                   paste(names(nonfinite)[nonfinite], collapse = ";"))
  }
  d$point_id <- we_normalize_point_id(d$point_id)
  d <- d[order(d$point_id, d$calendar_month_index), , drop = FALSE]
  rownames(d) <- NULL
  d$point_id <- factor(d$point_id, levels = sort(unique(d$point_id)))
  d$AR.start <- we_build_ar_start(d, "point_id")
  if (!is.null(raw_mapping)) {
    pool <- raw_mapping$pool
    d$precip_raw_global_support_rank <- raw_mapping$empirical_rank(
      d$precip_anomaly_mm, pool$precip_anomaly_mm
    )
    d$radiation_raw_global_support_rank <- raw_mapping$empirical_rank(
      d$radiation_anomaly_MJ_m2, pool$radiation_anomaly_MJ_m2
    )
  }
  d
}

we_formula_definition <- function(analysis, site_variable = "point_id") {
  meta <- WE_ANALYSIS_META[[analysis]]
  if (is.null(meta)) stop("Unknown definition sensitivity", call. = FALSE)
  terms <- c(
    sprintf("te(%s, %s, k = c(5, 5))", meta$exposure_x, meta$exposure_y),
    vapply(WE_MODEL1_CONTROLS, function(v) sprintf("s(%s, k = 5)", v), character(1)),
    "factor(Month)", "factor(Year)", sprintf('s(%s, bs = "re")', site_variable)
  )
  as.formula(paste(meta$response, "~", paste(terms, collapse = " + ")))
}

we_contrast_coordinates <- function(analysis, raw_mapping = NULL) {
  if (analysis == "MEDIAN_RESPONSE") {
    return(list(A = WE_RANK_CONTRAST_A, B = WE_RANK_CONTRAST_B))
  }
  if (is.null(raw_mapping)) stop("Raw mapping is required", call. = FALSE)
  c <- raw_mapping$coordinates
  list(
    A = c(precip_anomaly_mm = c$Pwet_raw_mm, radiation_anomaly_MJ_m2 = c$Rdim_raw_MJ_m2),
    B = c(precip_anomaly_mm = c$Pwet_raw_mm, radiation_anomaly_MJ_m2 = c$Rmid_raw_MJ_m2)
  )
}

we_definition_reference_row <- function(data, analysis, raw_mapping = NULL,
                                          group_col = "point_id", quantile_prob = 0.50) {
  ref <- data[1L, , drop = FALSE]
  for (v in WE_MODEL1_CONTROLS) {
    ref[[v]] <- unname(quantile(data[[v]], quantile_prob, na.rm = TRUE))
  }
  coords <- we_contrast_coordinates(analysis, raw_mapping)
  for (v in names(coords$A)) ref[[v]] <- coords$A[[v]]
  ref$Month <- sort(unique(data$Month))[1L]
  ref$Year <- sort(unique(data$Year))[1L]
  groups <- levels(data[[group_col]])
  if (is.null(groups)) groups <- sort(unique(as.character(data[[group_col]])))
  ref[[group_col]] <- factor(groups[1L], levels = groups)
  ref
}

we_definition_lpmatrix_contrast <- function(model, reference_row, analysis, raw_mapping = NULL) {
  coords <- we_contrast_coordinates(analysis, raw_mapping)
  a <- reference_row
  b <- reference_row
  for (v in names(coords$A)) a[[v]] <- coords$A[[v]]
  for (v in names(coords$B)) b[[v]] <- coords$B[[v]]
  xa <- predict(model, newdata = a, type = "lpmatrix")
  xb <- predict(model, newdata = b, type = "lpmatrix")
  dx <- xa - xb
  list(
    estimate = as.numeric(dx %*% coef(model)), dX = dx, X_A = xa, X_B = xb,
    newdata_A = a, newdata_B = b
  )
}

we_definition_contrast_invariance <- function(model, data, analysis, raw_mapping = NULL,
                                                group_col = "point_id") {
  main_ref <- we_definition_reference_row(data, analysis, raw_mapping, group_col, 0.50)
  main <- we_definition_lpmatrix_contrast(model, main_ref, analysis, raw_mapping)$estimate
  months <- sort(unique(data$Month))[unique(round(seq(1, length(unique(data$Month)), length.out = 3L)))]
  years <- sort(unique(data$Year))
  years <- years[c(1L, length(years))]
  groups <- levels(data[[group_col]])
  if (is.null(groups)) groups <- sort(unique(as.character(data[[group_col]])))
  group_idx <- unique(as.integer(round(seq(1, length(groups), length.out = 3L))))
  rows <- list()
  z <- 0L
  for (m in months) for (y in years) for (q in c(0.25, 0.50, 0.75)) for (g in groups[group_idx]) {
    ref <- we_definition_reference_row(data, analysis, raw_mapping, group_col, q)
    ref$Month <- m
    ref$Year <- y
    ref[[group_col]] <- factor(g, levels = groups)
    value <- we_definition_lpmatrix_contrast(model, ref, analysis, raw_mapping)$estimate
    z <- z + 1L
    rows[[z]] <- data.frame(
      Month = m, Year = y, continuous_control_quantile = q, group_value = g,
      contrast = value, difference_from_main = value - main,
      absolute_difference_from_main = abs(value - main), stringsAsFactors = FALSE
    )
  }
  validation <- do.call(rbind, rows)
  max_diff <- max(validation$absolute_difference_from_main)
  list(
    main_contrast = main, validation = validation, max_absolute_difference = max_diff,
    pass = is.finite(main) && all(is.finite(validation$contrast)) && max_diff <= 1e-8
  )
}

we_definition_support_validation <- function(data, analysis, raw_mapping = NULL) {
  if (analysis == "MEDIAN_RESPONSE") {
    centers <- list(
      A = c(x = 0.75, y = 0.25), B = c(x = 0.75, y = 0.50)
    )
    x <- data$precip_rank_baseline
    y <- data$radiation_rank_baseline
    scale_name <- "SITE_MONTH_BASELINE_RELATIVE_RANK"
  } else {
    if (is.null(raw_mapping)) stop("Raw mapping required", call. = FALSE)
    c <- raw_mapping$coordinates
    centers <- list(
      A = c(x = c$Pwet_raw_support_rank_actual, y = c$Rdim_raw_support_rank_actual),
      B = c(x = c$Pwet_raw_support_rank_actual, y = c$Rmid_raw_support_rank_actual)
    )
    x <- data$precip_raw_global_support_rank
    y <- data$radiation_raw_global_support_rank
    scale_name <- "GLOBAL_2005_2018_ACTIVE_POOL_EMPIRICAL_SUPPORT_RANK"
  }
  do.call(rbind, lapply(names(centers), function(label) {
    center <- centers[[label]]
    distance <- sqrt((x - center[["x"]])^2 + (y - center[["y"]])^2)
    keep <- is.finite(distance) & distance <= WE_SUPPORT_RADIUS
    n_records <- sum(keep)
    n_sites <- length(unique(as.character(data$point_id[keep])))
    n_years <- length(unique(data$Year[keep]))
    pass <- n_records >= WE_SUPPORT_MIN_RECORDS &&
      n_sites >= WE_SUPPORT_MIN_SITES && n_years >= WE_SUPPORT_MIN_YEARS
    data.frame(
      analysis = analysis, contrast_point = label, support_scale = scale_name,
      center_precip = center[["x"]], center_radiation = center[["y"]],
      radius = WE_SUPPORT_RADIUS, distance_rule = "EUCLIDEAN",
      n_records = n_records, n_sites = n_sites, n_years = n_years,
      min_records = WE_SUPPORT_MIN_RECORDS, min_sites = WE_SUPPORT_MIN_SITES,
      min_years = WE_SUPPORT_MIN_YEARS, status = if (pass) "PASS" else "FAIL",
      stringsAsFactors = FALSE
    )
  }))
}

we_ar_postfit_validation_definition <- function(model, data, analysis) {
  if (is.null(model$std.rsd)) stop("model$std.rsd unavailable", call. = FALSE)
  pairs <- we_continuous_pairs(data, as.numeric(model$std.rsd))
  lag1 <- we_rho_from_pairs(pairs)
  counts <- table(pairs$series_id)
  flag <- if (abs(lag1) <= 0.10) "PASS" else if (abs(lag1) <= 0.20) "WARN" else "STOP"
  data.frame(
    analysis = analysis, standardized_residual_lag1 = lag1,
    absolute_standardized_residual_lag1 = abs(lag1),
    n_consecutive_pairs = nrow(pairs), n_contributing_sites = length(counts),
    diagnostic_flag = flag, rho_used = WE_RHO_FIXED, rho_reestimated = FALSE,
    stringsAsFactors = FALSE
  )
}

we_surface_concurvity_summary <- function(table, analysis) {
  meta <- WE_ANALYSIS_META[[analysis]]
  surface <- sprintf("te(%s,%s)", meta$exposure_x, meta$exposure_y)
  full <- table[table$mode == "full" & table$term_to == surface & is.finite(table$value), , drop = FALSE]
  pair <- table[table$mode == "pairwise" & !table$is_pairwise_diagonal &
                  (table$term_from == surface | table$term_to == surface) &
                  is.finite(table$value), , drop = FALSE]
  fw <- full[full$metric == "worst", , drop = FALSE]
  pm <- if (nrow(pair)) pair[which.max(pair$value), , drop = FALSE] else NULL
  get_pair <- function(control) {
    term <- sprintf("s(%s)", control)
    z <- pair[(pair$term_from == surface & pair$term_to == term) |
                (pair$term_from == term & pair$term_to == surface), , drop = FALSE]
    if (!nrow(z)) return(NA_real_)
    max(z$value, na.rm = TRUE)
  }
  data.frame(
    analysis = analysis, surface_term = surface,
    surface_full_worst = if (nrow(fw)) fw$value[1L] else NA_real_,
    surface_full_worst_label = if (nrow(fw)) fw$validation_label[1L] else "NOT_AVAILABLE",
    surface_pairwise_max = if (!is.null(pm)) pm$value[1L] else NA_real_,
    surface_pairwise_label = if (!is.null(pm)) pm$validation_label[1L] else "NOT_AVAILABLE",
    pairwise_max_metric = if (!is.null(pm)) pm$metric[1L] else NA_character_,
    pairwise_max_from = if (!is.null(pm)) pm$term_from[1L] else NA_character_,
    pairwise_max_to = if (!is.null(pm)) pm$term_to[1L] else NA_character_,
    vs_antecedent_precipitation_max = get_pair("precip_antecedent3_sum_mm"),
    vs_antecedent_radiation_max = get_pair("radiation_antecedent3_mean_MJ_m2"),
    stringsAsFactors = FALSE
  )
}

we_boot_series_ids <- function(sampled_sites) {
  occurrence <- ave(seq_along(sampled_sites), sampled_sites, FUN = seq_along)
  paste0(sampled_sites, "_copy", sprintf("%02d", occurrence))
}

we_run_definition_boot_attempt <- function(
    attempt, analysis_data, draw_sequence, analysis, raw_mapping = NULL,
    rho_used = WE_RHO_FIXED) {
  started <- proc.time()[["elapsed"]]
  draw_rows <- draw_sequence[draw_sequence$attempt == attempt, , drop = FALSE]
  base <- list(
    attempt = as.integer(attempt), draw_id = sprintf("draw_%04d", as.integer(attempt)),
    success = FALSE, contrast = NA_real_, convergence = "NOT_FIT", warning = "",
    failure_reason = "", failure_category = "", n = NA_integer_,
    n_boot_series = NA_integer_, n_AR_sections = NA_integer_, elapsed_seconds = NA_real_,
    raw_coordinates_recomputed = FALSE
  )
  result <- tryCatch({
    boot <- we_build_boot_data(analysis_data, draw_rows)
    form <- we_formula_definition(analysis, "boot_series_id")
    fit <- we_fit_bam(form, boot, rho_used, boot$AR.start)
    health <- we_model_health(fit$model, fit$warnings)
    reasons <- we_model_health_stop_reason(health)
    value <- NA_real_
    if (!length(reasons)) {
      ref <- we_definition_reference_row(
        boot, analysis, raw_mapping, "boot_series_id", 0.50
      )
      value <- we_definition_lpmatrix_contrast(
        fit$model, ref, analysis, raw_mapping
      )$estimate
      if (!is.finite(value)) reasons <- c(reasons, "NONFINITE CONTRAST")
    }
    reason_text <- paste(reasons, collapse = ";")
    list(
      attempt = as.integer(attempt), draw_id = sprintf("draw_%04d", as.integer(attempt)),
      success = length(reasons) == 0L && is.finite(value), contrast = value,
      convergence = health$convergence[1L], warning = paste(fit$warnings, collapse = " | "),
      failure_reason = reason_text, failure_category = we_boot_failure_category(reason_text),
      n = nrow(boot), n_boot_series = length(levels(boot$boot_series_id)),
      n_AR_sections = sum(boot$AR.start),
      elapsed_seconds = proc.time()[["elapsed"]] - started,
      raw_coordinates_recomputed = FALSE
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

we_bootstrap_summary_definition <- function(values, full_estimate, n_attempt, analysis) {
  ci <- unname(quantile(values, c(0.025, 0.975), type = 7, names = FALSE))
  data.frame(
    analysis = analysis, full_data_contrast = full_estimate,
    bootstrap_mean = mean(values), bootstrap_median = median(values),
    bootstrap_SD = sd(values), percentile_2_5 = ci[1L], percentile_97_5 = ci[2L],
    negative_replicate_fraction = mean(values < 0),
    positive_replicate_fraction = mean(values > 0),
    n_success = length(values), n_attempt = n_attempt,
    n_fail = n_attempt - length(values),
    failure_fraction = (n_attempt - length(values)) / n_attempt,
    seed = WE_SEED, rho_used = WE_RHO_FIXED,
    interval_method = "site-cluster bootstrap percentile", stringsAsFactors = FALSE
  )
}

we_definition_check <- function(analysis, contrast, ci_upper, support_pass, diagnostic_stop) {
  if (!support_pass) {
    return(if (analysis == "RAW_EXPOSURE")
      "RAW_EXPOSURE_FIXED_CONTRAST_UNSUPPORTED" else
      "MEDIAN_RESPONSE_FIXED_CONTRAST_UNSUPPORTED")
  }
  if (diagnostic_stop || !is.finite(contrast)) {
    return(paste0(analysis, "_DIAGNOSTIC_STOP"))
  }
  if (analysis == "MEDIAN_RESPONSE") {
    if (contrast > 0) return("MEDIAN_RESPONSE_SIGN_REVERSAL_ANALYSIS_CHECK_REQUIRED")
    if (is.finite(ci_upper) && ci_upper < 0) return("MEDIAN_RESPONSE_ROBUSTNESS_PASS")
    return("MEDIAN_RESPONSE_DIRECTIONALLY_CONSISTENT_BUT_IMPRECISE")
  }
  if (contrast > 0) return("RAW_EXPOSURE_SIGN_REVERSAL_ANALYSIS_CHECK_REQUIRED")
  if (is.finite(ci_upper) && ci_upper < 0) return("RAW_EXPOSURE_DIRECTION_AND_INTERVAL_CONSISTENT")
  "RAW_EXPOSURE_DIRECTIONALLY_CONSISTENT_BUT_IMPRECISE"
}

we_software_validation <- function() {
  data.frame(
    R_version_string = R.version.string,
    mgcv_version = as.character(utils::packageVersion("mgcv")),
    mgcv_loadable = requireNamespace("mgcv", quietly = TRUE),
    fixed_primary_rho = WE_RHO_FIXED, network_download = FALSE,
    install_packages_called = FALSE, update_packages_called = FALSE,
    stringsAsFactors = FALSE
  )
}
