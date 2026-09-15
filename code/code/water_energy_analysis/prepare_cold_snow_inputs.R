source(file.path(Sys.getenv("WATER_ENERGY_CODE_DIR", unset = getwd()), "cold_snow_common.R"))

we_log("water-energy Cold/Snow prepare started")
if (!requireNamespace("mgcv", quietly = TRUE)) {
  we_stop_check("STOP — MGCV NOT AVAILABLE", "ENVIRONMENT_check")
}
environment_validation <- we_software_validation()
we_write_csv(
  environment_validation,
  we_path("09_LOGS", "WE_R_MGCV_ENVIRONMENT_validation.csv")
)
we_write_lines(capture.output(sessionInfo()), we_path("09_LOGS", "WE_R_SESSION_INFO.txt"))


master <- we_read_master()
primary_identity <- we_primary_identity_validation(master)
we_write_csv(
  primary_identity,
  we_path("09_LOGS", "WE_PRIMARY_DOMAIN_KEY_IDENTITY_validation.csv")
)
if (!all(primary_identity$status == "PASS")) {
  we_stop_check(
    "STOP — PRIMARY DOMAIN IDENTITY MISMATCH", "PRIMARY_DOMAIN_check",
    primary_identity$detail[primary_identity$status != "PASS"]
  )
}

reference_validation <- we_primary_reference_validation()
we_write_csv(
  reference_validation,
  we_path("09_LOGS", "WE_PRIMARY_REFERENCE_validation.csv")
)
if (!all(reference_validation$status == "PASS")) {
  we_stop_check(
    "STOP — PRIMARY REFERENCE MISMATCH", "PRIMARY_REFERENCE_check",
    paste(reference_validation$check[reference_validation$status != "PASS"], collapse = ";")
  )
}

temperature_validation <- we_temperature_semantics_validation(master)
we_write_csv(
  temperature_validation,
  we_path("02_ELIGIBILITY", "WE_TARGET_TEMPERATURE_FIELD_AND_UNIT_validation.csv")
)
if (!all(temperature_validation$status == "PASS")) {
  we_stop_check(
    "STOP — TARGET-MONTH ERA5 TEMPERATURE FIELD AMBIGUOUS", "TEMPERATURE_SEMANTICS_check",
    temperature_validation$status
  )
}

snow_validation <- we_snow_semantics_validation(master)
snow_detail <- attr(snow_validation, "detail")
we_write_csv(
  snow_validation,
  we_path("02_ELIGIBILITY", "WE_SNOW_STATUS_SEMANTICS_validation.csv")
)
we_write_csv(
  snow_detail,
  we_path("02_ELIGIBILITY", "WE_SNOW_STATUS_SEMANTICS_MISMATCH_DETAIL.csv")
)
if (!all(snow_validation$status == "PASS")) {
  we_stop_check(
    "STOP — SNOW STATUS NOT UNIQUELY RECONSTRUCTIBLE", "SNOW_SEMANTICS_check",
    paste(snow_validation$check[snow_validation$status != "PASS"], collapse = ";")
  )
}

eligibility <- we_cold_snow_eligibility_validation(master)
eligibility_detail <- attr(eligibility, "mismatch_detail")
we_write_csv(
  eligibility,
  we_path("02_ELIGIBILITY", "WE_COLD_SNOW_ELIGIBILITY_validation.csv")
)
we_write_csv(
  data.frame(
    explicit_n = eligibility$final_cold_snow_eligible_n,
    reference_flag_n = eligibility$reference_flag_n,
    explicit_only = eligibility$explicit_only,
    flag_only = eligibility$flag_only,
    mismatch = eligibility$mismatch,
    status = eligibility$status,
    stringsAsFactors = FALSE
  ),
  we_path("02_ELIGIBILITY", "WE_COLD_SNOW_ELIGIBILITY_KEY_IDENTITY_validation.csv")
)
we_write_csv(
  eligibility_detail,
  we_path("02_ELIGIBILITY", "WE_COLD_SNOW_ELIGIBILITY_KEY_MISMATCH_DETAIL.csv")
)
if (eligibility$status != "PASS") {
  we_stop_check(
    "STOP — COLD/SNOW ELIGIBILITY MISMATCH", "ELIGIBILITY_check",
    sprintf(
      "explicit=%d flag=%d explicit_only=%d flag_only=%d",
      eligibility$final_cold_snow_eligible_n, eligibility$reference_flag_n,
      eligibility$explicit_only, eligibility$flag_only
    )
  )
}

attrition_reason <- we_cold_snow_attrition_by_reason(master)
attrition_month <- we_cold_snow_attrition_by_group(master, "Month")
attrition_year <- we_cold_snow_attrition_by_group(master, "Year")
we_write_csv(
  attrition_reason,
  we_path("02_ELIGIBILITY", "WE_COLD_SNOW_ATTRITION_BY_REASON.csv")
)
we_write_csv(
  attrition_month,
  we_path("02_ELIGIBILITY", "WE_COLD_SNOW_ATTRITION_BY_MONTH.csv")
)
we_write_csv(
  attrition_year,
  we_path("02_ELIGIBILITY", "WE_COLD_SNOW_ATTRITION_BY_YEAR.csv")
)

cold <- we_prepare_cold_snow(master)
primary <- we_primary_data(master)
support <- we_cold_snow_support_validation(cold)
support_attrition <- we_cold_snow_support_attrition(master)
we_write_csv(
  support,
  we_path("04_SUPPORT", "WE_COLD_SNOW_FIXED_CONTRAST_SUPPORT_validation.csv")
)
we_write_csv(
  support_attrition,
  we_path("04_SUPPORT", "WE_COLD_SNOW_CONTRAST_SUPPORT_ATTRITION.csv")
)

ar <- we_ar_sequence_validation(cold, "point_id")
ar$summary$analysis <- WE_ALLOWED_ANALYSIS
we_write_csv(
  ar$summary,
  we_path("06_DIAGNOSTICS", "WE_COLD_SNOW_AR_SEQUENCE_validation.csv")
)
we_write_csv(
  ar$gaps,
  we_path("06_DIAGNOSTICS", "WE_COLD_SNOW_AR_GAPS.csv")
)
we_write_csv(
  ar$gap_distribution,
  we_path("06_DIAGNOSTICS", "WE_COLD_SNOW_AR_GAP_LENGTH_DISTRIBUTION.csv")
)

selector_validation <- we_analysis_selector_validation()
we_write_csv(
  selector_validation,
  we_path("09_LOGS", "WE_NO_UNAPPROVED_ANALYSIS_validation.csv")
)
input_validation <- data.frame(
  master_path = normalizePath(WE_MASTER_PATH, winslash = "/", mustWork = TRUE),
  ndvi_monthly_path = normalizePath(WE_NDVI_MONTHLY_PATH, winslash = "/", mustWork = TRUE),
  primary_v2_path = normalizePath(WE_PRIMARY_ROOT, winslash = "/", mustWork = TRUE),
  master_rows = nrow(master), primary_n = nrow(primary), primary_sites = length(levels(primary$point_id)),
  cold_snow_n = nrow(cold), cold_snow_sites = length(levels(cold$point_id)),
  canonical_rho = WE_RHO_FIXED, primary_refitted = FALSE,
  analysis_definition_changed = FALSE, network_download = FALSE,
  stringsAsFactors = FALSE
)
we_write_csv(
  input_validation,
  we_path("09_LOGS", "WE_INPUT_AND_IMMUTABILITY_validation.csv")
)

state <- list(
  prepare_complete = TRUE,
  created_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
  R_version = R.version.string,
  mgcv_version = as.character(packageVersion("mgcv")),
  primary_reference = list(
    n = 6992L, n_sites = 247L, contrast = -2.94258224191719,
    ci_lower = -3.55908622452363, ci_upper = -2.21122658573456,
    rho_used = WE_RHO_FIXED, refitted = FALSE
  ),
  temperature_validation = temperature_validation,
  snow_validation = snow_validation,
  eligibility = eligibility,
  attrition_reason = attrition_reason,
  support = support,
  support_attrition = support_attrition,
  support_pass = all(support$status == "PASS"),
  ar_summary = ar$summary,
  model_complete = FALSE,
  bootstrap_complete = FALSE
)
saveRDS(state, we_path("09_LOGS", "WE_COLD_SNOW_prepare_STATE.rds"))

we_log(sprintf(
  paste(
    "Cold/Snow prepare complete primary=%d retained=%d sites=%d retention=%.6f",
    "AR_sections=%d support=%s"
  ),
  eligibility$primary_n, eligibility$final_cold_snow_eligible_n,
  eligibility$n_sites, eligibility$retention_vs_primary,
  ar$summary$n_AR_sections, if (all(support$status == "PASS")) "PASS" else "UNSUPPORTED"
))
