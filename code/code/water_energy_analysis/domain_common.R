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
  "WATER_ENERGY_DOMAIN_OUTPUT_DIR",
  unset = file.path(WE_PROJECT_ROOT, "outputs", "water_energy", "domain")
)
WE_MASTER_PATH <- Sys.getenv(
  "WATER_ENERGY_MASTER_PATH",
  unset = file.path(WE_INPUT_DIR, "water_energy_analysis_ready_2022_2025.csv")
)
WE_TARGET_ROBUSTNESS_ROOT <- Sys.getenv("WATER_ENERGY_QA_DAYWEIGHT_OUTPUT_DIR", unset = file.path(WE_PROJECT_ROOT, "outputs", "water_energy", "qa_dayweight"))

WE_DIRS <- c(
  "01_CODE", "02_FULL_YEAR", "03_APR_SEP", "04_SUPPORT", "05_BOOTSTRAP",
  "06_DIAGNOSTICS", "07_COMPARISON", "08_TESTS", "09_LOGS"
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
WE_REQUIRED_FIELDS <- c(
  "point_id", "Year", "Month", "calendar_month_index", "npp_anomaly_mean_g",
  "precip_rank_baseline", "radiation_rank_baseline", WE_MODEL1_CONTROLS
)
WE_ALLOWED_ANALYSES <- c("FULL_YEAR", "APR_SEP")
WE_DOMAIN_META <- list(
  FULL_YEAR = list(
    output_dir = "02_FULL_YEAR", months = 1:12,
    domain_flag = "domain_all12", eligibility_flag = "eligible_all12",
    definition = "Year 2022-2025; Month 1-12; reference Model 1 complete case; no active filter",
    role = "domain pressure test"
  ),
  APR_SEP = list(
    output_dir = "03_APR_SEP", months = 4:9,
    domain_flag = "domain_apr_sep", eligibility_flag = "eligible_apr_sep",
    definition = "Year 2022-2025; Month 4-9 exactly; reference Model 1 complete case; no active filter",
    role = "fixed April-September robustness window"
  )
)

we_ensure_dirs()

we_log <- function(..., file = we_path("09_LOGS", "WE_DOMAIN_SENSITIVITY_RUN.log")) {
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

we_key <- function(d) {
  paste(sprintf("%03d", as.integer(as.character(d$point_id))),
        as.integer(d$Year), as.integer(d$Month), sep = "|")
}

we_complete_case <- function(master) {
  missing <- setdiff(WE_REQUIRED_FIELDS, names(master))
  we_assert(length(missing) == 0L,
              paste("Required domain fields missing:", paste(missing, collapse = ",")))
  numeric_fields <- setdiff(WE_REQUIRED_FIELDS, "point_id")
  !is.na(master$point_id) & Reduce(`&`, lapply(numeric_fields, function(v) is.finite(master[[v]])))
}

we_explicit_domain <- function(master, analysis) {
  meta <- WE_DOMAIN_META[[analysis]]
  if (is.null(meta)) stop("Unknown domain analysis", call. = FALSE)
  master$Year %in% 2022:2025 & master$Month %in% meta$months & we_complete_case(master)
}

we_domain_flag_validation <- function(master, analysis) {
  meta <- WE_DOMAIN_META[[analysis]]
  explicit <- we_explicit_domain(master, analysis)
  domain_flag <- we_flag_true(master[[meta$domain_flag]])
  eligibility_flag <- we_flag_true(master[[meta$eligibility_flag]])
  key <- we_key(master)
  explicit_only <- explicit & !domain_flag
  flag_only <- !explicit & domain_flag
  complete <- we_complete_case(master)
  requested_scope <- master$Year %in% 2022:2025 & master$Month %in% meta$months
  flag_only_explained_by_complete_case <- flag_only & requested_scope & !complete
  unexplained_flag_only <- flag_only & !flag_only_explained_by_complete_case
  substantive_conflict <- any(explicit_only) || any(unexplained_flag_only) ||
    any(xor(explicit, eligibility_flag))
  summary <- data.frame(
    analysis = analysis, explicit_definition = meta$definition,
    modeling_selector = "EXPLICIT_TASK_LOGIC",
    explicit_n = sum(explicit), domain_flag = meta$domain_flag,
    domain_flag_n = sum(domain_flag), intersection = sum(explicit & domain_flag),
    explicit_only = sum(explicit_only), flag_only = sum(flag_only),
    mismatch_keys = sum(xor(explicit, domain_flag)),
    flag_only_explained_by_complete_case = sum(flag_only_explained_by_complete_case),
    unexplained_flag_only = sum(unexplained_flag_only),
    eligibility_flag = meta$eligibility_flag,
    eligibility_flag_n = sum(eligibility_flag),
    explicit_vs_eligibility_mismatch = sum(xor(explicit, eligibility_flag)),
    historically_active_used = FALSE, eligible_primary_model1_used = FALSE,
    substantive_semantic_conflict = substantive_conflict,
    status = if (substantive_conflict) "STOP_REVIEW" else "PASS_EXPLAINED_SEMANTICS",
    stringsAsFactors = FALSE
  )
  detail_keep <- xor(explicit, domain_flag) | xor(explicit, eligibility_flag)
  detail <- master[detail_keep, intersect(c(
    "point_id", "Year", "Month", "calendar_month_index", meta$domain_flag,
    meta$eligibility_flag, "historically_active_canonical", "eligible_primary_model1",
    WE_REQUIRED_FIELDS[-1L], "analysis_ready_missing_reasons"
  ), names(master)), drop = FALSE]
  if (nrow(detail)) {
    detail$explicit_eligible <- explicit[detail_keep]
    detail$domain_flag_value <- domain_flag[detail_keep]
    detail$eligibility_flag_value <- eligibility_flag[detail_keep]
    detail$flag_only_explained_by_complete_case <- flag_only_explained_by_complete_case[detail_keep]
    detail$analysis <- analysis
  }
  list(summary = summary, detail = detail, explicit = explicit)
}

we_prepare_domain <- function(master, analysis) {
  validation <- we_domain_flag_validation(master, analysis)
  if (isTRUE(validation$summary$substantive_semantic_conflict)) {
    reason <- if (analysis == "FULL_YEAR")
      "STOP — FULL-YEAR DOMAIN FLAG SEMANTICS REQUIRE REVIEW" else
      "STOP — APR-SEP DOMAIN FLAG SEMANTICS REQUIRE REVIEW"
    we_stop_check(reason, paste0(analysis, "_DOMAIN_FLAG_check"))
  }
  d <- master[validation$explicit, , drop = FALSE]
  nonfinite <- vapply(setdiff(WE_REQUIRED_FIELDS, "point_id"),
                      function(v) any(!is.finite(d[[v]])), logical(1))
  we_assert(!any(nonfinite),
              paste("Nonfinite domain fields:", paste(names(nonfinite)[nonfinite], collapse = ",")))
  d$point_id <- sprintf("%03d", as.integer(d$point_id))
  d <- d[order(d$point_id, d$calendar_month_index), , drop = FALSE]
  rownames(d) <- NULL
  d$point_id <- factor(d$point_id, levels = sort(unique(d$point_id)))
  d$AR.start <- we_build_ar_start(d, "point_id")
  d
}

we_formula_domain <- function(site_variable = "point_id") {
  terms <- c(
    "te(precip_rank_baseline, radiation_rank_baseline, k = c(5, 5))",
    vapply(WE_MODEL1_CONTROLS, function(v) sprintf("s(%s, k = 5)", v), character(1)),
    "factor(Month)", "factor(Year)", sprintf('s(%s, bs = "re")', site_variable)
  )
  as.formula(paste("npp_anomaly_mean_g ~", paste(terms, collapse = " + ")))
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
  target_core <- read.csv(file.path(
    WE_TARGET_ROBUSTNESS_ROOT, "08_COMPARISON", "WE_QA_DAYWEIGHT_CORE_COMPARISON.csv"
  ), stringsAsFactors = FALSE)
  ref <- function(name, field) target_core[[field]][target_core$analysis == name]
  checks <- c(
    primary_n = primary_info$n == 6992L,
    primary_sites = primary_info$n_sites == 247L,
    primary_rho = isTRUE(all.equal(primary_info$rho_used, WE_RHO_FIXED, tolerance = 1e-15)),
    primary_contrast = isTRUE(all.equal(primary_boot$full_estimate, -2.94258224191719, tolerance = 1e-14)),
    primary_ci_lower = isTRUE(all.equal(primary_boot$percentile_2_5, -3.55908622452363, tolerance = 1e-14)),
    primary_ci_upper = isTRUE(all.equal(primary_boot$percentile_97_5, -2.21122658573456, tolerance = 1e-14)),
    primary_surface_k = primary_k$final_k[primary_k$term == "hydro_energy_surface"] == 5L,
    primary_controls_k = all(primary_k$final_k[match(WE_MODEL1_CONTROLS, primary_k$term)] == 5L),
    QA1_contrast = isTRUE(all.equal(ref("QA1", "contrast"), -2.74933816863286, tolerance = 1e-14)),
    QA1_ci = isTRUE(all.equal(c(ref("QA1", "CI_lower"), ref("QA1", "CI_upper")),
                              c(-3.26893919769731, -1.73797044794133), tolerance = 1e-14)),
    QA2_contrast = isTRUE(all.equal(ref("QA2", "contrast"), -1.27535872111437, tolerance = 1e-14)),
    QA2_ci = isTRUE(all.equal(c(ref("QA2", "CI_lower"), ref("QA2", "CI_upper")),
                              c(-2.41855680785968, -0.368551825715992), tolerance = 1e-14)),
    dayweighted_contrast = isTRUE(all.equal(ref("Day-weighted", "contrast"), -3.79465651925158, tolerance = 1e-14)),
    dayweighted_ci = isTRUE(all.equal(c(ref("Day-weighted", "CI_lower"), ref("Day-weighted", "CI_upper")),
                                      c(-4.56331310321137, -3.08974619145371), tolerance = 1e-14))
  )
  data.frame(
    check = names(checks), status = ifelse(checks, "PASS", "FAIL"),
    Primary_refitted = FALSE, target_robustness_refitted = FALSE,
    stringsAsFactors = FALSE
  )
}

we_support_validation <- function(data, analysis) {
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
    pass <- n_records >= WE_SUPPORT_MIN_RECORDS &&
      n_sites >= WE_SUPPORT_MIN_SITES && n_years >= WE_SUPPORT_MIN_YEARS
    data.frame(
      analysis = analysis, contrast_point = label,
      precip_rank_target = target[["precip_rank_baseline"]],
      radiation_rank_target = target[["radiation_rank_baseline"]],
      radius = WE_SUPPORT_RADIUS, distance_rule = "EUCLIDEAN_RANK_SPACE",
      n_records = n_records, n_sites = n_sites, n_years = n_years,
      min_records = WE_SUPPORT_MIN_RECORDS, min_sites = WE_SUPPORT_MIN_SITES,
      min_years = WE_SUPPORT_MIN_YEARS, status = if (pass) "PASS" else "FAIL",
      stringsAsFactors = FALSE
    )
  }))
}

we_ar_postfit_validation <- function(model, data, analysis) {
  we_assert(!is.null(model$std.rsd), "model$std.rsd is unavailable")
  pairs <- we_continuous_pairs(data, as.numeric(model$std.rsd))
  lag1 <- we_rho_from_pairs(pairs)
  counts <- table(pairs$series_id)
  flag <- if (abs(lag1) <= 0.10) "PASS" else if (abs(lag1) <= 0.20) "WARN" else "STOP"
  data.frame(
    analysis = analysis, standardized_residual_lag1 = lag1,
    absolute_standardized_residual_lag1 = abs(lag1),
    n_consecutive_pairs = nrow(pairs), n_contributing_sites = length(counts),
    diagnostic_flag = flag, rho_used = WE_RHO_FIXED,
    rho_reestimated = FALSE, stringsAsFactors = FALSE
  )
}

we_hydro_concurvity_summary <- function(table, analysis) {
  te_term <- "te(precip_rank_baseline,radiation_rank_baseline)"
  full <- table[table$mode == "full" & table$term_to == te_term & is.finite(table$value), , drop = FALSE]
  pair <- table[table$mode == "pairwise" & !table$is_pairwise_diagonal &
                  (table$term_from == te_term | table$term_to == te_term) &
                  is.finite(table$value), , drop = FALSE]
  full_worst <- full[full$metric == "worst", , drop = FALSE]
  pair_max <- if (nrow(pair)) pair[which.max(pair$value), , drop = FALSE] else NULL
  data.frame(
    analysis = analysis,
    hydro_full_worst = if (nrow(full_worst)) full_worst$value[1L] else NA_real_,
    hydro_full_worst_label = if (nrow(full_worst)) full_worst$validation_label[1L] else "NOT_AVAILABLE",
    hydro_full_observed = full$value[match("observed", full$metric)],
    hydro_full_estimate = full$value[match("estimate", full$metric)],
    hydro_pairwise_max = if (!is.null(pair_max)) pair_max$value[1L] else NA_real_,
    hydro_pairwise_metric = if (!is.null(pair_max)) pair_max$metric[1L] else NA_character_,
    hydro_pairwise_from = if (!is.null(pair_max)) pair_max$term_from[1L] else NA_character_,
    hydro_pairwise_to = if (!is.null(pair_max)) pair_max$term_to[1L] else NA_character_,
    hydro_pairwise_label = if (!is.null(pair_max)) pair_max$validation_label[1L] else "NOT_AVAILABLE",
    stringsAsFactors = FALSE
  )
}

we_boot_series_ids <- function(sampled_sites) {
  occurrence <- ave(seq_along(sampled_sites), sampled_sites, FUN = seq_along)
  paste0(sampled_sites, "_copy", sprintf("%02d", occurrence))
}

we_run_domain_boot_attempt <- function(
    attempt, analysis_data, draw_sequence, rho_used = WE_RHO_FIXED) {
  started <- proc.time()[["elapsed"]]
  draw_rows <- draw_sequence[draw_sequence$attempt == attempt, , drop = FALSE]
  base <- list(
    attempt = as.integer(attempt), draw_id = sprintf("draw_%04d", as.integer(attempt)),
    success = FALSE, contrast = NA_real_, convergence = "NOT_FIT", warning = "",
    failure_reason = "", failure_category = "", n = NA_integer_,
    n_boot_series = NA_integer_, n_AR_sections = NA_integer_, elapsed_seconds = NA_real_
  )
  result <- tryCatch({
    boot <- we_build_boot_data(analysis_data, draw_rows)
    form <- we_formula_domain("boot_series_id")
    fit <- we_fit_bam(form, boot, rho_used, boot$AR.start)
    health <- we_model_health(fit$model, fit$warnings)
    reasons <- we_model_health_stop_reason(health)
    value <- NA_real_
    if (!length(reasons)) {
      ref <- we_reference_row(boot, "boot_series_id", 0.50)
      value <- we_lpmatrix_contrast(fit$model, ref)$estimate
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

we_bootstrap_summary <- function(values, full_estimate, n_attempt, analysis) {
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

we_domain_check <- function(analysis, contrast, ci_upper, support_pass, diagnostic_stop) {
  if (!support_pass) return("DOMAIN_SENSITIVITY_FIXED_CONTRAST_UNSUPPORTED")
  if (diagnostic_stop || !is.finite(contrast)) return("DOMAIN_SENSITIVITY_DIAGNOSTIC_STOP")
  if (analysis == "FULL_YEAR") {
    if (contrast > 0) return("FULL_YEAR_SIGN_REVERSAL_ANALYSIS_CHECK_REQUIRED")
    if (is.finite(ci_upper) && ci_upper < 0) return("FULL_YEAR_DIRECTION_AND_INTERVAL_CONSISTENT")
    return("FULL_YEAR_DIRECTIONALLY_CONSISTENT_BUT_IMPRECISE")
  }
  if (contrast > 0) return("APR_SEP_SIGN_REVERSAL_ANALYSIS_CHECK_REQUIRED")
  if (is.finite(ci_upper) && ci_upper < 0) return("APR_SEP_ROBUSTNESS_PASS")
  "APR_SEP_DIRECTIONALLY_CONSISTENT_BUT_IMPRECISE"
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
