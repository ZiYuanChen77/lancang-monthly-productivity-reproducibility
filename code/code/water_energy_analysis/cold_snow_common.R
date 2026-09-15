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
  "WATER_ENERGY_COLD_SNOW_OUTPUT_DIR",
  unset = file.path(WE_PROJECT_ROOT, "outputs", "water_energy", "cold_snow")
)
WE_MASTER_PATH <- Sys.getenv(
  "WATER_ENERGY_MASTER_PATH",
  unset = file.path(WE_INPUT_DIR, "water_energy_analysis_ready_2022_2025.csv")
)
WE_NDVI_MONTHLY_PATH <- Sys.getenv("WATER_ENERGY_NDVI_QA_PATH", unset = file.path(WE_INPUT_DIR, "ndvi_monthly_qa_2001_2025.csv"))

WE_DIRS <- c(
  "01_CODE", "02_ELIGIBILITY", "03_MODEL", "04_SUPPORT", "05_BOOTSTRAP",
  "06_DIAGNOSTICS", "07_COMPARISON", "08_TESTS", "09_LOGS"
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
WE_CONTRAST_A <- c(precip_rank_baseline = 0.75, radiation_rank_baseline = 0.25)
WE_CONTRAST_B <- c(precip_rank_baseline = 0.75, radiation_rank_baseline = 0.50)
WE_TEMPERATURE_FIELD <- "temperature_2m"
WE_TEMPERATURE_UNIT <- "degrees Celsius"
WE_SNOW_KNOWN_FIELD <- "snow_ice_qa_known"
WE_SNOW_STATUS_FIELD <- "target_month_any_snow_ice"
WE_ELIGIBILITY_FIELD <- "eligible_cold_snow"
WE_ALLOWED_ANALYSIS <- "COLD_SNOW_EXCLUSION"

we_ensure_dirs()

we_log <- function(..., file = we_path("09_LOGS", "WE_COLD_SNOW_RUN.log")) {
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

we_flag_true <- function(x) !is.na(x) & (x == TRUE | x == 1L)

we_normalize_point_id <- function(x) sprintf("%03d", as.integer(as.character(x)))

we_key <- function(d) {
  paste(we_normalize_point_id(d$point_id), as.integer(d$Year), as.integer(d$Month), sep = "|")
}

we_primary_identity_validation <- function(master) {
  primary <- master[we_flag_true(master$eligible_primary_model1), , drop = FALSE]
  primary_keys <- sort(we_key(primary))
  model <- readRDS(file.path(
    WE_PRIMARY_ROOT, "03_MODEL1_PRIMARY", "WE_MODEL1_PRIMARY.rds"
  ))
  mf <- model$model
  required <- c("point_id", "factor(Year)", "factor(Month)")
  missing_fields <- setdiff(required, names(mf))
  if (length(missing_fields)) {
    return(data.frame(
      check = "canonical_primary_model_frame_key_identity", master_primary_n = length(primary_keys),
      canonical_model_n = nrow(mf), master_sites = length(unique(primary$point_id)),
      missing_keys = NA_integer_, extra_keys = NA_integer_, status = "FAIL",
      detail = paste("model fields missing", paste(missing_fields, collapse = ";")),
      stringsAsFactors = FALSE
    ))
  }
  model_keys <- sort(paste(
    we_normalize_point_id(mf$point_id),
    as.integer(as.character(mf[["factor(Year)"]])),
    as.integer(as.character(mf[["factor(Month)"]])), sep = "|"
  ))
  missing_keys <- setdiff(primary_keys, model_keys)
  extra_keys <- setdiff(model_keys, primary_keys)
  pass <- length(primary_keys) == 6992L && length(unique(primary$point_id)) == 247L &&
    !anyDuplicated(primary_keys) && !anyDuplicated(model_keys) &&
    !length(missing_keys) && !length(extra_keys)
  data.frame(
    check = "canonical_primary_model_frame_key_identity", master_primary_n = length(primary_keys),
    canonical_model_n = length(model_keys), master_sites = length(unique(primary$point_id)),
    missing_keys = length(missing_keys), extra_keys = length(extra_keys),
    status = if (pass) "PASS" else "FAIL",
    detail = "exact point_id-Year-Month set comparison; Primary model not refitted",
    stringsAsFactors = FALSE
  )
}

we_primary_reference_validation <- function() {
  info <- read.csv(file.path(
    WE_PRIMARY_ROOT, "03_MODEL1_PRIMARY", "WE_MODEL1_PRIMARY_MODEL_INFO.csv"
  ), stringsAsFactors = FALSE)
  boot <- read.csv(file.path(
    WE_PRIMARY_ROOT, "10_BOOTSTRAP", "WE_MODEL1_PRIMARY_CONTRAST_SUMMARY.csv"
  ), stringsAsFactors = FALSE)
  k <- read.csv(file.path(
    WE_PRIMARY_ROOT, "07_BASIS_DIAGNOSTICS", "WE_MODEL1_K_FINAL_CONFIGURATION.csv"
  ), stringsAsFactors = FALSE)
  checks <- c(
    primary_n = info$n == 6992L,
    primary_sites = info$n_sites == 247L,
    primary_rho = isTRUE(all.equal(info$rho_used, WE_RHO_FIXED, tolerance = 1e-15)),
    primary_contrast = isTRUE(all.equal(boot$full_estimate, -2.94258224191719, tolerance = 1e-14)),
    primary_ci_lower = isTRUE(all.equal(boot$percentile_2_5, -3.55908622452363, tolerance = 1e-14)),
    primary_ci_upper = isTRUE(all.equal(boot$percentile_97_5, -2.21122658573456, tolerance = 1e-14)),
    primary_surface_k = k$final_k[k$term == "hydro_energy_surface"] == 5L,
    primary_control_k = all(k$final_k[match(WE_MODEL1_CONTROLS, k$term)] == 5L)
  )
  data.frame(
    check = names(checks), status = ifelse(checks, "PASS", "FAIL"),
    primary_refitted = FALSE, stringsAsFactors = FALSE
  )
}

we_equal_with_na <- function(x, y, tolerance = 0) {
  both_na <- is.na(x) & is.na(y)
  both_value <- !is.na(x) & !is.na(y)
  out <- both_na
  if (tolerance == 0) out[both_value] <- x[both_value] == y[both_value]
  else out[both_value] <- abs(x[both_value] - y[both_value]) <= tolerance
  out
}

we_temperature_semantics_validation <- function(master) {
  required <- c(WE_TEMPERATURE_FIELD, "temperature_anomaly")
  missing <- setdiff(required, names(master))
  primary <- master[we_flag_true(master$eligible_primary_model1), , drop = FALSE]
  finite_ok <- !length(missing) && all(is.finite(primary[[WE_TEMPERATURE_FIELD]]))
  range_ok <- finite_ok && max(primary[[WE_TEMPERATURE_FIELD]]) < 100 && min(primary[[WE_TEMPERATURE_FIELD]]) > -100
  data.frame(
    actual_field = WE_TEMPERATURE_FIELD,
    semantic_role = "target-month ERA5-Land 2 m air temperature",
    source_product = "ECMWF/ERA5_LAND/MONTHLY_AGGR",
    unit = WE_TEMPERATURE_UNIT,
    separate_anomaly_field = "temperature_anomaly",
    threshold_operator = ">", threshold_C = 0,
    primary_min_C = if (finite_ok) min(primary[[WE_TEMPERATURE_FIELD]]) else NA_real_,
    primary_max_C = if (finite_ok) max(primary[[WE_TEMPERATURE_FIELD]]) else NA_real_,
    status = if (!length(missing) && range_ok) "PASS" else paste0("FAIL:", paste(missing, collapse = ";")),
    stringsAsFactors = FALSE
  )
}

we_snow_semantics_validation <- function(master) {
  ndvi <- read.csv(
    WE_NDVI_MONTHLY_PATH, colClasses = c(point_id = "character"), check.names = FALSE
  )
  ndvi$point_id <- we_normalize_point_id(ndvi$point_id)
  master$point_id <- we_normalize_point_id(master$point_id)
  mk <- we_key(master)
  nk <- we_key(ndvi)
  idx <- match(mk, nk)
  matched <- !is.na(idx)
  d <- ndvi[idx, , drop = FALSE]
  reconstructed_known <- d$any_snow_ice_flag %in% c(0, 1) &
    is.finite(d$ndvi_total_composite_count) & d$ndvi_total_composite_count > 0
  reconstructed_target <- ifelse(reconstructed_known, d$any_snow_ice_flag, NA_real_)
  known_match <- matched & we_equal_with_na(as.numeric(master$snow_ice_qa_known), as.numeric(reconstructed_known))
  target_match <- matched & we_equal_with_na(
    as.numeric(master$target_month_any_snow_ice), as.numeric(reconstructed_target)
  )
  underlying_flag_match <- matched & we_equal_with_na(
    as.numeric(master$any_snow_ice_flag), as.numeric(d$any_snow_ice_flag)
  )
  composite_count_match <- matched & we_equal_with_na(
    as.numeric(master$ndvi_total_composite_count), as.numeric(d$ndvi_total_composite_count)
  )
  checks <- data.frame(
    check = c(
      "NDVI monthly key coverage", "snow-known deterministic reconstruction",
      "target-month snow/ice deterministic reconstruction", "underlying snow flag identity",
      "NDVI total composite count identity"
    ),
    n_checked = rep(nrow(master), 5L),
    n_mismatch = c(
      sum(!matched), sum(!known_match), sum(!target_match),
      sum(!underlying_flag_match), sum(!composite_count_match)
    ),
    status = "PASS",
    stringsAsFactors = FALSE
  )
  checks$status[checks$n_mismatch > 0L] <- "FAIL"
  attr(checks, "detail") <- data.frame(
    point_id = master$point_id[!(matched & known_match & target_match)],
    Year = master$Year[!(matched & known_match & target_match)],
    Month = master$Month[!(matched & known_match & target_match)],
    master_snow_known = master$snow_ice_qa_known[!(matched & known_match & target_match)],
    reconstructed_snow_known = as.integer(reconstructed_known[!(matched & known_match & target_match)]),
    master_target_snow = master$target_month_any_snow_ice[!(matched & known_match & target_match)],
    reconstructed_target_snow = reconstructed_target[!(matched & known_match & target_match)],
    stringsAsFactors = FALSE
  )
  checks
}

we_cold_snow_flags <- function(master) {
  primary <- we_flag_true(master$eligible_primary_model1)
  temperature_known <- is.finite(master$temperature_2m)
  temperature_gt0 <- temperature_known & master$temperature_2m > 0
  temperature_le0 <- temperature_known & master$temperature_2m <= 0
  snow_known <- we_flag_true(master$snow_ice_qa_known)
  snow_absent <- snow_known & !is.na(master$target_month_any_snow_ice) &
    master$target_month_any_snow_ice == 0
  snow_present <- snow_known & !is.na(master$target_month_any_snow_ice) &
    master$target_month_any_snow_ice == 1
  explicit <- primary & temperature_gt0 & snow_known & snow_absent
  reference <- we_flag_true(master$eligible_cold_snow)
  data.frame(
    primary = primary, temperature_known = temperature_known,
    temperature_gt0 = temperature_gt0, temperature_le0 = temperature_le0,
    temperature_unknown = !temperature_known, snow_known = snow_known,
    snow_unknown = !snow_known, snow_absent = snow_absent,
    snow_present = snow_present, explicit = explicit, reference = reference,
    stringsAsFactors = FALSE
  )
}

we_cold_snow_eligibility_validation <- function(master) {
  f <- we_cold_snow_flags(master)
  p <- f$primary
  explicit_keys <- sort(we_key(master[f$explicit, , drop = FALSE]))
  reference_keys <- sort(we_key(master[f$reference, , drop = FALSE]))
  retained <- master[f$explicit, , drop = FALSE]
  validation <- data.frame(
    primary_n = sum(p),
    temperature_gt_0_n = sum(p & f$temperature_gt0),
    temperature_le_0_n = sum(p & f$temperature_le0),
    temperature_unknown_n = sum(p & f$temperature_unknown),
    snow_known_n = sum(p & f$snow_known),
    snow_unknown_n = sum(p & f$snow_unknown),
    snow_present_n = sum(p & f$snow_present),
    snow_absent_n = sum(p & f$snow_absent),
    final_cold_snow_eligible_n = sum(f$explicit),
    reference_flag_n = sum(f$reference),
    retention_vs_primary = sum(f$explicit) / sum(p),
    n_sites = length(unique(retained$point_id)),
    n_years = length(unique(retained$Year)),
    n_months = length(unique(retained$Month)),
    explicit_only = length(setdiff(explicit_keys, reference_keys)),
    flag_only = length(setdiff(reference_keys, explicit_keys)),
    mismatch = length(setdiff(explicit_keys, reference_keys)) + length(setdiff(reference_keys, explicit_keys)),
    status = if (identical(explicit_keys, reference_keys)) "PASS" else "FAIL",
    stringsAsFactors = FALSE
  )
  detail_keep <- xor(f$explicit, f$reference)
  detail <- master[detail_keep, c(
    "point_id", "Year", "Month", "calendar_month_index", "eligible_primary_model1",
    "temperature_2m", "snow_ice_qa_known", "target_month_any_snow_ice",
    "eligible_cold_snow"
  ), drop = FALSE]
  if (nrow(detail)) detail$explicit_cold_snow <- f$explicit[detail_keep]
  attr(validation, "mismatch_detail") <- detail
  validation
}

we_prepare_cold_snow <- function(master) {
  f <- we_cold_snow_flags(master)
  explicit_keys <- sort(we_key(master[f$explicit, , drop = FALSE]))
  reference_keys <- sort(we_key(master[f$reference, , drop = FALSE]))
  we_assert(identical(explicit_keys, reference_keys), "Cold/Snow explicit and reference key sets differ")
  d <- master[f$explicit, , drop = FALSE]
  required <- c(
    "point_id", "Year", "Month", "calendar_month_index", "npp_anomaly_mean_g",
    "precip_rank_baseline", "radiation_rank_baseline", WE_MODEL1_CONTROLS
  )
  missing_fields <- setdiff(required, names(d))
  we_assert(!length(missing_fields), paste("Cold/Snow model fields missing:", paste(missing_fields, collapse = ",")))
  numeric_required <- setdiff(required, "point_id")
  bad <- vapply(numeric_required, function(v) any(!is.finite(d[[v]])), logical(1))
  we_assert(!any(bad), paste("Cold/Snow nonfinite model fields:", paste(names(bad)[bad], collapse = ",")))
  d <- d[order(d$point_id, d$calendar_month_index), , drop = FALSE]
  rownames(d) <- NULL
  d$point_id <- factor(we_normalize_point_id(d$point_id), levels = sort(unique(we_normalize_point_id(d$point_id))))
  d$AR.start <- we_build_ar_start(d, "point_id")
  d
}

we_cold_snow_attrition_by_reason <- function(master) {
  f <- we_cold_snow_flags(master)
  pidx <- which(f$primary)
  reasons <- cbind(
    temperature_le_0 = f$temperature_le0[pidx],
    temperature_unknown = f$temperature_unknown[pidx],
    snow_present = f$snow_present[pidx],
    snow_unknown = f$snow_unknown[pidx]
  )
  n_reason <- rowSums(reasons)
  category <- rep("retained", length(pidx))
  category[n_reason > 1L] <- "excluded_multiple_reasons"
  category[n_reason == 1L & reasons[, "temperature_le_0"]] <- "excluded_temperature_le_0"
  category[n_reason == 1L & reasons[, "temperature_unknown"]] <- "excluded_temperature_unknown"
  category[n_reason == 1L & reasons[, "snow_present"]] <- "excluded_snow_present"
  category[n_reason == 1L & reasons[, "snow_unknown"]] <- "excluded_snow_unknown"
  levels_out <- c(
    "retained", "excluded_temperature_le_0", "excluded_temperature_unknown",
    "excluded_snow_present", "excluded_snow_unknown", "excluded_multiple_reasons"
  )
  counts <- table(factor(category, levels = levels_out))
  data.frame(
    attrition_category = levels_out, n_records = as.integer(counts),
    fraction_of_primary = as.integer(counts) / length(pidx),
    mutually_exclusive = TRUE, stringsAsFactors = FALSE
  )
}

we_cold_snow_attrition_by_group <- function(master, group_field) {
  f <- we_cold_snow_flags(master)
  pidx <- which(f$primary)
  groups <- sort(unique(master[[group_field]][pidx]))
  do.call(rbind, lapply(groups, function(g) {
    idx <- pidx[master[[group_field]][pidx] == g]
    n <- length(idx)
    data.frame(
      grouping = group_field, group_value = as.character(g), n_primary = n,
      n_retained = sum(f$explicit[idx]), retention = sum(f$explicit[idx]) / n,
      temperature_exclusion_n = sum(f$temperature_le0[idx] | f$temperature_unknown[idx]),
      temperature_exclusion_fraction = sum(f$temperature_le0[idx] | f$temperature_unknown[idx]) / n,
      snow_exclusion_n = sum(f$snow_present[idx] | f$snow_unknown[idx]),
      snow_exclusion_fraction = sum(f$snow_present[idx] | f$snow_unknown[idx]) / n,
      unknown_snow_n = sum(f$snow_unknown[idx]),
      unknown_snow_fraction = sum(f$snow_unknown[idx]) / n,
      stringsAsFactors = FALSE
    )
  }))
}

we_cold_snow_support_validation <- function(data) {
  points <- list(A = WE_CONTRAST_A, B = WE_CONTRAST_B)
  do.call(rbind, lapply(names(points), function(label) {
    target <- points[[label]]
    distance <- sqrt(
      (data$precip_rank_baseline - target[["precip_rank_baseline"]])^2 +
        (data$radiation_rank_baseline - target[["radiation_rank_baseline"]])^2
    )
    keep <- is.finite(distance) & distance <= WE_SUPPORT_RADIUS
    n_records <- sum(keep)
    n_sites <- length(unique(as.character(data$point_id[keep])))
    n_years <- length(unique(data$Year[keep]))
    pass <- n_records >= WE_SUPPORT_MIN_RECORDS && n_sites >= WE_SUPPORT_MIN_SITES &&
      n_years >= WE_SUPPORT_MIN_YEARS
    data.frame(
      contrast_point = label,
      precip_rank_target = target[["precip_rank_baseline"]],
      radiation_rank_target = target[["radiation_rank_baseline"]],
      radius = WE_SUPPORT_RADIUS, n_records = n_records,
      n_sites = n_sites, n_years = n_years,
      minimum_records = WE_SUPPORT_MIN_RECORDS,
      minimum_sites = WE_SUPPORT_MIN_SITES,
      minimum_years = WE_SUPPORT_MIN_YEARS,
      status = if (pass) "PASS" else "COLD_SNOW_FIXED_CONTRAST_UNSUPPORTED",
      stringsAsFactors = FALSE
    )
  }))
}

we_cold_snow_support_attrition <- function(master) {
  f <- we_cold_snow_flags(master)
  primary <- master[f$primary, , drop = FALSE]
  fp <- f[f$primary, , drop = FALSE]
  points <- list(A = WE_CONTRAST_A, B = WE_CONTRAST_B)
  do.call(rbind, lapply(names(points), function(label) {
    target <- points[[label]]
    distance <- sqrt(
      (primary$precip_rank_baseline - target[["precip_rank_baseline"]])^2 +
        (primary$radiation_rank_baseline - target[["radiation_rank_baseline"]])^2
    )
    near <- is.finite(distance) & distance <= WE_SUPPORT_RADIUS
    retained <- near & fp$explicit
    data.frame(
      contrast_point = label, radius = WE_SUPPORT_RADIUS,
      primary_records = sum(near),
      primary_sites = length(unique(primary$point_id[near])),
      primary_years = length(unique(primary$Year[near])),
      cold_snow_records = sum(retained),
      cold_snow_sites = length(unique(primary$point_id[retained])),
      cold_snow_years = length(unique(primary$Year[retained])),
      record_retention = sum(retained) / sum(near),
      temperature_exclusions = sum(near & (fp$temperature_le0 | fp$temperature_unknown)),
      snow_present_exclusions = sum(near & fp$snow_present),
      snow_unknown_exclusions = sum(near & fp$snow_unknown),
      stringsAsFactors = FALSE
    )
  }))
}

we_ar_postfit_validation_cold_snow <- function(model, data) {
  std <- as.numeric(model$std.rsd)
  if (is.null(model$std.rsd)) {
    std <- as.numeric(residuals(model, type = "response")) / sqrt(summary(model)$scale)
  }
  pairs <- we_continuous_pairs(data, std, "point_id", "calendar_month_index")
  r <- if (nrow(pairs) >= 2L) cor(pairs$value_t_minus_1, pairs$value_t) else NA_real_
  abs_r <- abs(r)
  data.frame(
    analysis = WE_ALLOWED_ANALYSIS,
    standardized_residual_lag1 = r,
    absolute_standardized_residual_lag1 = abs_r,
    n_consecutive_pairs = nrow(pairs),
    n_contributing_sites = length(unique(pairs$series_id)),
    check = if (!is.finite(abs_r)) "STOP" else if (abs_r <= 0.10) "PASS" else if (abs_r <= 0.20) "WARN" else "STOP",
    rho_reestimated = FALSE, rho_used = WE_RHO_FIXED,
    stringsAsFactors = FALSE
  )
}

we_cold_snow_concurvity_summary <- function(concurvity_table) {
  surface <- grepl("precip_rank_baseline", concurvity_table$term_from, fixed = TRUE) |
    grepl("precip_rank_baseline", concurvity_table$term_to, fixed = TRUE)
  surface_full <- concurvity_table$mode == "full" &
    grepl("precip_rank_baseline", concurvity_table$term_to, fixed = TRUE)
  surface_pair <- concurvity_table$mode == "pairwise" & surface &
    !concurvity_table$is_pairwise_diagonal
  control_pattern <- paste(WE_MODEL1_CONTROLS, collapse = "|")
  control <- (grepl(control_pattern, concurvity_table$term_from) |
    grepl(control_pattern, concurvity_table$term_to)) &
    !concurvity_table$is_pairwise_diagonal
  finite_max <- function(x) if (any(is.finite(x))) max(x[is.finite(x)]) else NA_real_
  full_value <- finite_max(concurvity_table$value[surface_full])
  pair_value <- finite_max(concurvity_table$value[surface_pair])
  control_rows <- concurvity_table[control & is.finite(concurvity_table$value), , drop = FALSE]
  if (nrow(control_rows)) control_row <- control_rows[which.max(control_rows$value), , drop = FALSE]
  else control_row <- data.frame(term_from = NA_character_, term_to = NA_character_, value = NA_real_)
  data.frame(
    surface_full_worst = full_value,
    surface_full_label = we_concurvity_label(full_value),
    surface_pairwise_max = pair_value,
    surface_pairwise_label = we_concurvity_label(pair_value),
    highest_control_term_from = control_row$term_from[1L],
    highest_control_term_to = control_row$term_to[1L],
    highest_control_concurvity = control_row$value[1L],
    highest_control_label = we_concurvity_label(control_row$value[1L]),
    stringsAsFactors = FALSE
  )
}

we_run_cold_snow_boot_attempt <- function(attempt, data, draw_sequence) {
  we_run_boot_attempt(
    attempt = attempt, primary = data, draw_sequence = draw_sequence,
    k_config = we_model1_k_initial(), rho_used = WE_RHO_FIXED
  )
}

we_cold_snow_bootstrap_summary <- function(values, full_estimate, n_attempt) {
  data.frame(
    analysis = WE_ALLOWED_ANALYSIS,
    full_estimate = full_estimate,
    bootstrap_mean = mean(values), bootstrap_median = median(values),
    bootstrap_SD = sd(values),
    percentile_2_5 = unname(quantile(values, 0.025, type = 7)),
    percentile_97_5 = unname(quantile(values, 0.975, type = 7)),
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

we_cold_snow_check <- function(contrast, ci_upper, support_pass, diagnostic_stop = FALSE) {
  if (diagnostic_stop) return("COLD_SNOW_DIAGNOSTIC_STOP")
  if (!support_pass) return("COLD_SNOW_FIXED_CONTRAST_UNSUPPORTED")
  if (contrast > 0) return("COLD_SNOW_SIGN_REVERSAL_ANALYSIS_CHECK_REQUIRED")
  if (contrast < 0 && ci_upper < 0) return("COLD_SNOW_ROBUSTNESS_PASS")
  if (contrast < 0) return("COLD_SNOW_DIRECTIONALLY_CONSISTENT_BUT_IMPRECISE")
  "COLD_SNOW_ZERO_CONTRAST_ANALYSIS_CHECK_REQUIRED"
}

we_plot_png <- function(filename, expression, width = 1800L, height = 1400L) {
  png(filename, width = width, height = height, res = 180)
  on.exit(dev.off(), add = TRUE)
  force(expression)
}

we_analysis_selector_validation <- function() {
  data.frame(
    item = c(
      "analysis_run", "QA_stacking", "dayweighted_stacking", "median_raw_stacking",
      "Model0_2_3", "LOYO", "threshold_search", "other_sensitivity",
      "analysis_definition_changed", "network_download"
    ),
    value = c(
      WE_ALLOWED_ANALYSIS, rep("NO", 9L)
    ),
    status = "PASS", stringsAsFactors = FALSE
  )
}
