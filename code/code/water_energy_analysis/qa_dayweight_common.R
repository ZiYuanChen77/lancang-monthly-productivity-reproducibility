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
  "WATER_ENERGY_QA_DAYWEIGHT_OUTPUT_DIR",
  unset = file.path(WE_PROJECT_ROOT, "outputs", "water_energy", "qa_dayweight")
)
WE_MASTER_PATH <- Sys.getenv(
  "WATER_ENERGY_MASTER_PATH",
  unset = file.path(WE_INPUT_DIR, "water_energy_analysis_ready_2022_2025.csv")
)

WE_DIRS <- c(
  "01_CODE", "02_QA1", "03_QA2", "04_DAYWEIGHTED", "05_SUPPORT",
  "06_BOOTSTRAP", "07_DIAGNOSTICS", "08_COMPARISON", "09_TESTS", "10_LOGS"
)
WE_SEED <- 20260803L
WE_BOOT_DRAWS <- 1050L
WE_BOOT_TARGET_SUCCESS <- 1000L
WE_BOOT_INITIAL_BATCH <- 50L
WE_BOOT_MAX_FAILURE_RATE <- 0.05
WE_RHO_FIXED <- 0.036623315093487
WE_CONTRAST_A <- c(precip_rank_baseline = 0.75, radiation_rank_baseline = 0.25)
WE_CONTRAST_B <- c(precip_rank_baseline = 0.75, radiation_rank_baseline = 0.50)
WE_SUPPORT_RADIUS <- 0.10
WE_SUPPORT_MIN_RECORDS <- 100L
WE_SUPPORT_MIN_SITES <- 50L
WE_SUPPORT_MIN_YEARS <- 3L
WE_MODEL1_CONTROLS <- c(
  "precip_antecedent3_sum_mm", "radiation_antecedent3_mean_MJ_m2",
  "soilwater_lag1", "ndvi_lag1_canonical"
)
WE_ALL_CONTROLS <- WE_MODEL1_CONTROLS
WE_ALLOWED_ANALYSES <- c("QA1", "QA2", "DAYWEIGHTED")
WE_ANALYSIS_META <- list(
  QA1 = list(
    response = "npp_anomaly_mean_g", output_dir = "02_QA1",
    domain = "eligible_primary_model1 AND mod17_QA1",
    role = "primary MOD17 QA1 one-factor sensitivity"
  ),
  QA2 = list(
    response = "npp_anomaly_mean_g", output_dir = "03_QA2",
    domain = "eligible_primary_model1 AND mod17_QA2",
    role = "coverage-limited strict pressure test"
  ),
  DAYWEIGHTED = list(
    response = "dayweighted_anomaly_mean_g", output_dir = "04_DAYWEIGHTED",
    domain = "eligible_dayweighted",
    role = "primary day-weighted target one-factor sensitivity"
  )
)

we_ensure_dirs()

we_log <- function(..., file = we_path("10_LOGS", "WE_QA_DAYWEIGHT_RUN.log")) {
  stamp <- format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z")
  msg <- paste(..., collapse = " ")
  cat(sprintf("[%s] %s\n", stamp, msg), file = file, append = TRUE)
  message(msg)
  invisible(msg)
}

we_stop_check <- function(reason, stage, detail = "") {
  status <- data.frame(
    timestamp = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
    stage = stage, reason = reason, detail = detail,
    final_status = paste0("STOPPED — analysis check required: ", reason),
    stringsAsFactors = FALSE
  )
  we_write_csv(status, we_path("10_LOGS", "WE_STOP_STATUS.csv"))
  we_log("STOP check:", reason, "stage=", stage, detail)
  stop(paste(reason, detail), call. = FALSE)
}

we_flag_true <- function(x) !is.na(x) & (x == TRUE | x == 1L)

we_key <- function(d) {
  paste(sprintf("%03d", as.integer(as.character(d[["point_id"]]))),
        as.integer(d[["Year"]]), as.integer(d[["Month"]]), sep = "|")
}

we_model_required_fields <- function(response) {
  c(
    "point_id", "Year", "Month", "calendar_month_index", response,
    "precip_rank_baseline", "radiation_rank_baseline", WE_MODEL1_CONTROLS
  )
}

we_primary_reference_validation <- function() {
  info_path <- file.path(
    WE_PRIMARY_ROOT, "03_MODEL1_PRIMARY", "WE_MODEL1_PRIMARY_MODEL_INFO.csv"
  )
  k_path <- file.path(
    WE_PRIMARY_ROOT, "07_BASIS_DIAGNOSTICS", "WE_MODEL1_K_FINAL_CONFIGURATION.csv"
  )
  contrast_path <- file.path(
    WE_PRIMARY_ROOT, "09_PRIMARY_CONTRAST", "WE_MODEL_CONTRAST_COMPARISON.csv"
  )
  boot_path <- file.path(
    WE_PRIMARY_ROOT, "10_BOOTSTRAP", "WE_MODEL1_PRIMARY_CONTRAST_SUMMARY.csv"
  )
  test_path <- file.path(
    WE_PRIMARY_ROOT, "13_TESTS", "WE_GAMM_PRIMARY_TEST_RESULTS.csv"
  )
  required <- c(info_path, k_path, contrast_path, boot_path, test_path)
  we_assert(all(file.exists(required)), "primary analysis reference files are incomplete")
  info <- read.csv(info_path, stringsAsFactors = FALSE, check.names = FALSE)
  kk <- read.csv(k_path, stringsAsFactors = FALSE, check.names = FALSE)
  contrast <- read.csv(contrast_path, stringsAsFactors = FALSE, check.names = FALSE)
  boot <- read.csv(boot_path, stringsAsFactors = FALSE, check.names = FALSE)
  primary_tests <- read.csv(test_path, stringsAsFactors = FALSE, check.names = FALSE)
  m1 <- contrast[contrast[["model"]] == "Model1 PRIMARY", , drop = FALSE]
  checks <- c(
    primary_n = nrow(info) == 1L && info[["n"]][1L] == 6992L,
    primary_sites = nrow(info) == 1L && info[["n_sites"]][1L] == 247L,
    rho = nrow(info) == 1L && isTRUE(all.equal(info[["rho_used"]][1L], WE_RHO_FIXED, tolerance = 1e-15)),
    contrast = nrow(m1) == 1L && isTRUE(all.equal(m1[["contrast_estimate"]][1L], -2.94258224191719, tolerance = 1e-14)),
    ci_lower = nrow(boot) == 1L && isTRUE(all.equal(boot[["percentile_2_5"]][1L], -3.55908622452363, tolerance = 1e-14)),
    ci_upper = nrow(boot) == 1L && isTRUE(all.equal(boot[["percentile_97_5"]][1L], -2.21122658573456, tolerance = 1e-14)),
    surface_k = nrow(kk[kk[["term"]] == "hydro_energy_surface", , drop = FALSE]) == 1L &&
      kk[["final_k"]][kk[["term"]] == "hydro_energy_surface"] == 5L,
    controls_k = all(kk[["final_k"]][match(WE_MODEL1_CONTROLS, kk[["term"]])] == 5L),
    primary_tests = nrow(primary_tests) > 0L && all(primary_tests[["status"]] == "PASS")
  )
  out <- data.frame(
    check = names(checks), status = ifelse(checks, "PASS", "FAIL"),
    expected_reference = c(
      "6992", "247", format(WE_RHO_FIXED, digits = 16),
      "-2.94258224191719", "-3.55908622452363", "-2.21122658573456",
      "c(5,5)", "all four controls k=5", "all applicable primary analysis tests PASS"
    ),
    stringsAsFactors = FALSE
  )
  attr(out, "values") <- list(info = info, k = kk, contrast = m1, boot = boot)
  out
}

we_explicit_eligibility <- function(master, analysis) {
  primary <- we_flag_true(master[["eligible_primary_model1"]])
  switch(
    analysis,
    QA1 = primary & we_flag_true(master[["mod17_QA1"]]),
    QA2 = primary & we_flag_true(master[["mod17_QA2"]]),
    DAYWEIGHTED = {
      req <- we_model_required_fields("dayweighted_anomaly_mean_g")
      numeric_req <- setdiff(req, "point_id")
      complete <- Reduce(`&`, lapply(numeric_req, function(v) is.finite(master[[v]])))
      we_flag_true(master[["historically_active_dayweighted"]]) &
        master[["Year"]] >= 2022L & master[["Year"]] <= 2025L & complete
    },
    stop("Unknown analysis", call. = FALSE)
  )
}

we_eligibility_identity_validation <- function(master, analysis) {
  explicit <- we_explicit_eligibility(master, analysis)
  flag_name <- switch(
    analysis, QA1 = "eligible_mod17_QA1", QA2 = "eligible_mod17_QA2",
    DAYWEIGHTED = "eligible_dayweighted"
  )
  flagged <- we_flag_true(master[[flag_name]])
  keys <- we_key(master)
  missing_keys <- setdiff(keys[explicit], keys[flagged])
  extra_keys <- setdiff(keys[flagged], keys[explicit])
  mismatch <- xor(explicit, flagged)
  data.frame(
    analysis = analysis, explicit_definition = switch(
      analysis,
      QA1 = "eligible_primary_model1 == TRUE & mod17_QA1 == TRUE",
      QA2 = "eligible_primary_model1 == TRUE & mod17_QA2 == TRUE",
      DAYWEIGHTED = paste(
        "historically_active_dayweighted == TRUE; 2022-2025;",
        "dayweighted response and reference Model 1 covariates finite"
      )
    ),
    eligibility_flag = flag_name,
    explicit_n = sum(explicit), flagged_n = sum(flagged),
    missing_key_count = length(missing_keys), extra_key_count = length(extra_keys),
    mismatch_flag_count = sum(mismatch),
    status = if (length(missing_keys) == 0L && length(extra_keys) == 0L && !any(mismatch)) "PASS" else "FAIL",
    stringsAsFactors = FALSE
  )
}

we_prepare_sensitivity <- function(master, analysis) {
  meta <- WE_ANALYSIS_META[[analysis]]
  if (is.null(meta)) stop("Unknown sensitivity analysis", call. = FALSE)
  identity <- we_eligibility_identity_validation(master, analysis)
  if (identity[["status"]][1L] != "PASS") {
    reason <- switch(
      analysis,
      QA1 = "STOP — QA1 ELIGIBILITY MISMATCH",
      QA2 = "STOP — QA2 ELIGIBILITY MISMATCH",
      DAYWEIGHTED = "STOP — DAYWEIGHTED ELIGIBILITY MISMATCH"
    )
    we_stop_check(reason, paste0(analysis, "_ELIGIBILITY_check"))
  }
  flag <- switch(
    analysis,
    QA1 = we_flag_true(master[["eligible_mod17_QA1"]]),
    QA2 = we_flag_true(master[["eligible_mod17_QA2"]]),
    DAYWEIGHTED = we_flag_true(master[["eligible_dayweighted"]])
  )
  d <- master[flag, , drop = FALSE]
  required <- we_model_required_fields(meta[["response"]])
  missing_fields <- setdiff(required, names(d))
  we_assert(length(missing_fields) == 0L,
              paste("Missing sensitivity fields:", paste(missing_fields, collapse = ",")))
  numeric_required <- setdiff(required, "point_id")
  nonfinite <- vapply(numeric_required, function(v) any(!is.finite(d[[v]])), logical(1))
  we_assert(!any(nonfinite),
              paste("Nonfinite sensitivity fields:", paste(names(nonfinite)[nonfinite], collapse = ",")))
  d[["point_id"]] <- sprintf("%03d", as.integer(d[["point_id"]]))
  d <- d[order(d[["point_id"]], d[["calendar_month_index"]]), , drop = FALSE]
  rownames(d) <- NULL
  d[["point_id"]] <- factor(d[["point_id"]], levels = sort(unique(d[["point_id"]])))
  d[["AR.start"]] <- we_build_ar_start(d, "point_id")
  d
}

we_formula_sensitivity <- function(response, site_variable = "point_id") {
  terms <- c(
    "te(precip_rank_baseline, radiation_rank_baseline, k = c(5, 5))",
    vapply(WE_MODEL1_CONTROLS, function(v) sprintf("s(%s, k = 5)", v), character(1)),
    "factor(Month)", "factor(Year)", sprintf('s(%s, bs = "re")', site_variable)
  )
  as.formula(paste(response, "~", paste(terms, collapse = " + ")))
}

we_support_validation <- function(data, analysis) {
  points <- list(A = WE_CONTRAST_A, B = WE_CONTRAST_B)
  do.call(rbind, lapply(names(points), function(label) {
    target <- points[[label]]
    distance <- sqrt(
      (data[["precip_rank_baseline"]] - target[["precip_rank_baseline"]])^2 +
        (data[["radiation_rank_baseline"]] - target[["radiation_rank_baseline"]])^2
    )
    keep <- is.finite(distance) & distance <= WE_SUPPORT_RADIUS
    n_records <- sum(keep)
    n_sites <- length(unique(as.character(data[["point_id"]][keep])))
    n_years <- length(unique(data[["Year"]][keep]))
    pass <- n_records >= WE_SUPPORT_MIN_RECORDS &&
      n_sites >= WE_SUPPORT_MIN_SITES && n_years >= WE_SUPPORT_MIN_YEARS
    data.frame(
      analysis = analysis, contrast_point = label,
      precip_rank_target = target[["precip_rank_baseline"]],
      radiation_rank_target = target[["radiation_rank_baseline"]],
      radius = WE_SUPPORT_RADIUS, distance_rule = "EUCLIDEAN_RANK_SPACE",
      n_records = n_records, n_sites = n_sites, n_years = n_years,
      min_records = WE_SUPPORT_MIN_RECORDS, min_sites = WE_SUPPORT_MIN_SITES,
      min_years = WE_SUPPORT_MIN_YEARS,
      status = if (pass) "PASS" else "FAIL",
      stringsAsFactors = FALSE
    )
  }))
}

we_dayweighted_domain_validation <- function(master) {
  primary <- we_flag_true(master[["eligible_primary_model1"]])
  dw <- we_flag_true(master[["eligible_dayweighted"]])
  anomaly_identity <- with(master,
    !is.finite(dayweighted_anomaly_mean_g) |
      abs(dayweighted_anomaly_mean_g -
            (PsnNet_dayweighted_gC_m2_month - dayweighted_baseline_mean_g)) <= 1e-10
  )
  active_identity <- we_flag_true(master[["historically_active_dayweighted"]]) ==
    (is.finite(master[["dayweighted_baseline_median_g"]]) &
       master[["dayweighted_baseline_median_g"]] > 0)
  data.frame(
    master_n = nrow(master), primary_n = sum(primary), dayweighted_n = sum(dw),
    intersection = sum(primary & dw), primary_only = sum(primary & !dw),
    dayweighted_only = sum(!primary & dw),
    dayweighted_year_min = min(master[["Year"]][dw]),
    dayweighted_year_max = max(master[["Year"]][dw]),
    dayweighted_sites = length(unique(master[["point_id"]][dw])),
    response_identity_mismatch = sum(!anomaly_identity),
    active_definition_mismatch = sum(!active_identity),
    canonical_active_used_as_selector = FALSE,
    direct_eligible_dayweighted_selector = TRUE,
    status = if (all(anomaly_identity) && all(active_identity)) "PASS" else "FAIL",
    stringsAsFactors = FALSE
  )
}

we_dayweighted_identity_change <- function(master) {
  primary <- we_flag_true(master[["eligible_primary_model1"]])
  dw <- we_flag_true(master[["eligible_dayweighted"]])
  summary <- data.frame(
    record_type = "SUMMARY",
    category = c("primary_only", "dayweighted_only", "intersection"),
    count = c(sum(primary & !dw), sum(!primary & dw), sum(primary & dw)),
    point_id = NA_character_, calendar_month_index = NA_integer_, Year = NA_integer_,
    Month = NA_integer_, historically_active_canonical = NA,
    historically_active_dayweighted = NA, eligible_primary_model1 = NA,
    eligible_dayweighted = NA, stringsAsFactors = FALSE
  )
  active_diff <- we_flag_true(master[["historically_active_canonical"]]) !=
    we_flag_true(master[["historically_active_dayweighted"]])
  detail <- master[active_diff, c(
    "point_id", "calendar_month_index", "Year", "Month",
    "historically_active_canonical", "historically_active_dayweighted",
    "eligible_primary_model1", "eligible_dayweighted"
  ), drop = FALSE]
  if (nrow(detail)) {
    detail <- transform(
      detail, record_type = "ACTIVE_STATUS_SWAP", category = "active_identity_exchange",
      count = NA_integer_
    )
    detail <- detail[, names(summary), drop = FALSE]
  } else {
    detail <- summary[FALSE, , drop = FALSE]
  }
  rbind(summary, detail)
}

we_mod17_attrition_by_exposure <- function(master) {
  primary <- we_flag_true(master[["eligible_primary_model1"]])
  d <- master[primary, , drop = FALSE]
  d[["QA1"]] <- we_flag_true(d[["mod17_QA1"]])
  d[["QA2"]] <- we_flag_true(d[["mod17_QA2"]])
  rank_breaks <- c(-Inf, 0.25, 0.50, 0.75, Inf)
  labels <- c("Q1_[0,0.25]", "Q2_(0.25,0.50]", "Q3_(0.50,0.75]", "Q4_(0.75,1]")
  strata <- list(
    precip_rank_quartile = cut(d[["precip_rank_baseline"]], breaks = rank_breaks,
                               labels = labels, include.lowest = TRUE, right = TRUE),
    radiation_rank_quartile = cut(d[["radiation_rank_baseline"]], breaks = rank_breaks,
                                  labels = labels, include.lowest = TRUE, right = TRUE),
    Year = factor(d[["Year"]], levels = sort(unique(d[["Year"]]))),
    Month = factor(d[["Month"]], levels = sort(unique(d[["Month"]])))
  )
  rows <- list()
  z <- 0L
  for (nm in names(strata)) {
    s <- strata[[nm]]
    for (lev in levels(s)) {
      keep <- !is.na(s) & s == lev
      z <- z + 1L
      rows[[z]] <- data.frame(
        stratum_type = nm, stratum_value = lev,
        n_primary = sum(keep), n_QA1 = sum(d[["QA1"]][keep]),
        QA1_fraction = if (sum(keep)) mean(d[["QA1"]][keep]) else NA_real_,
        n_QA2 = sum(d[["QA2"]][keep]),
        QA2_fraction = if (sum(keep)) mean(d[["QA2"]][keep]) else NA_real_,
        interpretation = "DESCRIPTIVE_SELECTION_validation_ONLY",
        stringsAsFactors = FALSE
      )
    }
  }
  do.call(rbind, rows)
}

we_support_attrition <- function(primary, qa1, qa2) {
  domains <- list(Primary = primary, QA1 = qa1, QA2 = qa2)
  rows <- list()
  z <- 0L
  for (point in c("A", "B")) {
    target <- if (point == "A") WE_CONTRAST_A else WE_CONTRAST_B
    primary_records <- NA_integer_
    for (nm in names(domains)) {
      d <- domains[[nm]]
      distance <- sqrt(
        (d[["precip_rank_baseline"]] - target[["precip_rank_baseline"]])^2 +
          (d[["radiation_rank_baseline"]] - target[["radiation_rank_baseline"]])^2
      )
      keep <- is.finite(distance) & distance <= WE_SUPPORT_RADIUS
      n_records <- sum(keep)
      if (nm == "Primary") primary_records <- n_records
      z <- z + 1L
      rows[[z]] <- data.frame(
        contrast_point = point, domain = nm, radius = WE_SUPPORT_RADIUS,
        records = n_records,
        sites = length(unique(as.character(d[["point_id"]][keep]))),
        years = length(unique(d[["Year"]][keep])),
        retention_vs_primary_records = n_records / primary_records,
        stringsAsFactors = FALSE
      )
    }
  }
  do.call(rbind, rows)
}

we_ar_postfit_validation <- function(model, data, analysis) {
  we_assert(!is.null(model[["std.rsd"]]), "model$std.rsd is unavailable")
  pairs <- we_continuous_pairs(data, as.numeric(model[["std.rsd"]]))
  lag1 <- we_rho_from_pairs(pairs)
  counts <- table(pairs[["series_id"]])
  flag <- if (abs(lag1) <= 0.10) "PASS" else if (abs(lag1) <= 0.20) "WARN" else "STOP"
  data.frame(
    analysis = analysis, standardized_residual_lag1 = lag1,
    absolute_standardized_residual_lag1 = abs(lag1),
    n_consecutive_pairs = nrow(pairs), n_contributing_sites = length(counts),
    diagnostic_flag = flag, rho_used = WE_RHO_FIXED,
    rho_reestimated = FALSE, stringsAsFactors = FALSE
  )
}

we_relevant_highest_concurvity <- function(table, analysis) {
  target_terms <- c(
    "te(precip_rank_baseline,radiation_rank_baseline)",
    sprintf("s(%s)", WE_MODEL1_CONTROLS)
  )
  keep <- is.finite(table[["value"]]) & !table[["is_pairwise_diagonal"]] &
    table[["term_to"]] %in% target_terms
  x <- table[keep, , drop = FALSE]
  if (!nrow(x)) return(data.frame())
  out <- x[which.max(x[["value"]]), , drop = FALSE]
  out[["analysis"]] <- analysis
  out[, c("analysis", setdiff(names(out), "analysis")), drop = FALSE]
}

we_boot_series_ids <- function(sampled_sites) {
  occurrence <- ave(seq_along(sampled_sites), sampled_sites, FUN = seq_along)
  paste0(sampled_sites, "_copy", sprintf("%02d", occurrence))
}

we_run_sensitivity_boot_attempt <- function(
    attempt, analysis_data, draw_sequence, response, rho_used = WE_RHO_FIXED) {
  started <- proc.time()[["elapsed"]]
  draw_rows <- draw_sequence[draw_sequence[["attempt"]] == attempt, , drop = FALSE]
  base <- list(
    attempt = as.integer(attempt), draw_id = sprintf("draw_%04d", as.integer(attempt)),
    success = FALSE, contrast = NA_real_, convergence = "NOT_FIT", warning = "",
    failure_reason = "", failure_category = "", n = NA_integer_,
    n_boot_series = NA_integer_, n_AR_sections = NA_integer_, elapsed_seconds = NA_real_
  )
  result <- tryCatch({
    boot <- we_build_boot_data(analysis_data, draw_rows)
    form <- we_formula_sensitivity(response, "boot_series_id")
    fit <- we_fit_bam(form, boot, rho_used, boot[["AR.start"]])
    health <- we_model_health(fit[["model"]], fit[["warnings"]])
    reasons <- we_model_health_stop_reason(health)
    value <- NA_real_
    if (!length(reasons)) {
      ref <- we_reference_row(boot, "boot_series_id", 0.50)
      value <- we_lpmatrix_contrast(fit[["model"]], ref)[["estimate"]]
      if (!is.finite(value)) reasons <- c(reasons, "NONFINITE CONTRAST")
    }
    reason_text <- paste(reasons, collapse = ";")
    list(
      attempt = as.integer(attempt), draw_id = sprintf("draw_%04d", as.integer(attempt)),
      success = length(reasons) == 0L && is.finite(value), contrast = value,
      convergence = health[["convergence"]][1L],
      warning = paste(fit[["warnings"]], collapse = " | "),
      failure_reason = reason_text,
      failure_category = we_boot_failure_category(reason_text),
      n = nrow(boot), n_boot_series = length(levels(boot[["boot_series_id"]])),
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

we_bootstrap_summary <- function(values, full_estimate, n_attempt, analysis) {
  ci <- unname(quantile(values, c(0.025, 0.975), type = 7, names = FALSE))
  data.frame(
    analysis = analysis, full_data_contrast = full_estimate,
    bootstrap_mean = mean(values), bootstrap_median = median(values),
    bootstrap_SD = sd(values), percentile_2_5 = ci[1L], percentile_97_5 = ci[2L],
    n_success = length(values), n_attempt = n_attempt,
    n_fail = n_attempt - length(values),
    failure_fraction = (n_attempt - length(values)) / n_attempt,
    negative_replicate_fraction = mean(values < 0),
    positive_replicate_fraction = mean(values > 0),
    seed = WE_SEED, rho_used = WE_RHO_FIXED,
    interval_method = "site-cluster bootstrap percentile",
    stringsAsFactors = FALSE
  )
}

we_check_label <- function(analysis, contrast, ci_upper, support_pass, diagnostic_stop) {
  if (analysis == "QA1") {
    if (!support_pass) return("QA1_SUPPORT_FAIL")
    if (diagnostic_stop) return("QA1_DIAGNOSTIC_STOP")
    if (!is.finite(contrast) || contrast >= 0) return("QA1_DIRECTION_FAIL")
    if (!is.finite(ci_upper) || ci_upper >= 0) return("QA1_CI_CROSSES_ZERO")
    return("QA1_ROBUSTNESS_PASS")
  }
  if (analysis == "DAYWEIGHTED") {
    if (!support_pass) return("DAYWEIGHTED_SUPPORT_FAIL")
    if (diagnostic_stop) return("DAYWEIGHTED_DIAGNOSTIC_STOP")
    if (!is.finite(contrast) || contrast >= 0) return("DAYWEIGHTED_DIRECTION_FAIL")
    if (!is.finite(ci_upper) || ci_upper >= 0) return("DAYWEIGHTED_CI_CROSSES_ZERO")
    return("DAYWEIGHTED_ROBUSTNESS_PASS")
  }
  if (!support_pass) return("QA2_COVERAGE_LIMITED_FIXED_CONTRAST_UNSUPPORTED")
  if (!is.finite(contrast)) return("QA2_DIAGNOSTIC_STOP")
  if (contrast > 0) return("QA2_SIGN_REVERSAL_ANALYSIS_CHECK_REQUIRED")
  if (is.finite(ci_upper) && ci_upper < 0) return("QA2_DIRECTION_AND_INTERVAL_CONSISTENT")
  "QA2_DIRECTIONALLY_CONSISTENT_COVERAGE_LIMITED"
}

we_analysis_slug <- function(analysis) if (analysis == "DAYWEIGHTED") "DAYWEIGHTED" else analysis

we_software_validation <- function() {
  data.frame(
    R_version_string = R.version.string,
    mgcv_version = as.character(utils::packageVersion("mgcv")),
    mgcv_loadable = requireNamespace("mgcv", quietly = TRUE),
    fixed_primary_rho = WE_RHO_FIXED,
    network_download = FALSE, install_packages_called = FALSE,
    update_packages_called = FALSE, stringsAsFactors = FALSE
  )
}
