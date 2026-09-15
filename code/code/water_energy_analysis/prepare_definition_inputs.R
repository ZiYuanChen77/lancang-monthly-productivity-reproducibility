source(file.path(Sys.getenv("WATER_ENERGY_CODE_DIR", unset = getwd()), "definition_common.R"))

we_log("water-energy definition sensitivity pre-validation started")
we_write_csv(we_software_validation(),
               we_path("10_LOGS", "WE_R_MGCV_ENVIRONMENT_validation.csv"))
we_write_lines(capture.output(sessionInfo()),
                 we_path("10_LOGS", "WE_R_SESSION_INFO.txt"))

reference <- we_reference_reference_validation()
we_write_csv(reference, we_path("10_LOGS", "WE_reference_REFERENCE_validation.csv"))
if (!all(reference$status == "PASS")) {
  we_stop_check("STOP — PRIMARY DOMAIN IDENTITY MISMATCH", "reference_REFERENCE_check",
                 paste(reference$check[reference$status != "PASS"], collapse = ";"))
}

master <- we_read_master()
identity <- we_primary_identity_validation(master)
we_write_csv(identity, we_path("10_LOGS", "WE_PRIMARY_DOMAIN_KEY_IDENTITY_validation.csv"))
if (identity$status != "PASS") {
  we_stop_check("STOP — PRIMARY DOMAIN IDENTITY MISMATCH", "PRIMARY_KEY_check",
                 paste("missing", identity$missing_keys, "extra", identity$extra_keys))
}

raw_mapping <- we_build_raw_mapping()
if (raw_mapping$validation$centering_status != "PASS") {
  we_write_csv(raw_mapping$validation,
                 we_path("04_RAW_BASELINE_MAPPING", "WE_RAW_EXPOSURE_BASELINE_POOL_validation.csv"))
  we_stop_check("STOP — RAW ANOMALY BASELINE CENTERING FAILURE",
                 "RAW_BASELINE_CENTERING_check",
                 sprintf("max_precip=%.17g max_radiation=%.17g",
                         raw_mapping$validation$max_abs_precip_group_mean_anomaly,
                         raw_mapping$validation$max_abs_radiation_group_mean_anomaly))
}
primary <- we_prepare_primary_domain(master, raw_mapping)

primary_flag <- we_flag_true(master$eligible_primary_model1)
median_flag <- we_flag_true(master$eligible_median_response)
raw_flag <- we_flag_true(master$eligible_raw_exposure)
domain_validation <- data.frame(
  analysis = c("PRIMARY_REFERENCE", "MEDIAN_RESPONSE", "RAW_EXPOSURE"),
  selector_used = c(
    "eligible_primary_model1", "eligible_primary_model1", "eligible_primary_model1"
  ),
  n = c(sum(primary_flag), nrow(primary), nrow(primary)),
  n_sites = c(length(unique(master$point_id[primary_flag])),
              length(levels(primary$point_id)), length(levels(primary$point_id))),
  year_min = c(min(master$Year[primary_flag]), min(primary$Year), min(primary$Year)),
  year_max = c(max(master$Year[primary_flag]), max(primary$Year), max(primary$Year)),
  eligible_sensitivity_flag_mismatch = c(
    NA_integer_, sum(xor(primary_flag, median_flag)), sum(xor(primary_flag, raw_flag))
  ),
  complete_case_reformed_domain = FALSE,
  status = c(
    if (sum(primary_flag) == 6992L) "PASS" else "FAIL",
    if (nrow(primary) == 6992L && sum(xor(primary_flag, median_flag)) == 0L) "PASS" else "FAIL",
    if (nrow(primary) == 6992L && sum(xor(primary_flag, raw_flag)) == 0L) "PASS" else "FAIL"
  ),
  stringsAsFactors = FALSE
)
we_write_csv(domain_validation,
               we_path("05_SUPPORT", "WE_PRIMARY_DOMAIN_IDENTITY_validation.csv"))
if (!all(domain_validation$status == "PASS")) {
  we_stop_check("STOP — PRIMARY DOMAIN IDENTITY MISMATCH", "SENSITIVITY_DOMAIN_check")
}

median_validation <- data.frame(
  approved_field = "npp_anomaly_median_g",
  field_present = "npp_anomaly_median_g" %in% names(master),
  primary_records = nrow(primary),
  finite_primary_values = sum(is.finite(primary$npp_anomaly_median_g)),
  values_different_from_mean_response = sum(
    abs(primary$npp_anomaly_median_g - primary$npp_anomaly_mean_g) > 1e-12
  ),
  recalculated = FALSE,
  source = normalizePath(WE_MASTER_PATH, winslash = "/", mustWork = TRUE),
  definition = "canonical monthly productivity minus 2005-2018 same point_id x Month historical median",
  status = if (all(is.finite(primary$npp_anomaly_median_g)) &&
                any(abs(primary$npp_anomaly_median_g - primary$npp_anomaly_mean_g) > 1e-12))
    "PASS" else "FAIL",
  stringsAsFactors = FALSE
)
we_write_csv(median_validation,
               we_path("02_MEDIAN_RESPONSE", "WE_MEDIAN_RESPONSE_FIELD_validation.csv"))
if (median_validation$status != "PASS") {
  we_stop_check("STOP — APPROVED MEDIAN RESPONSE FIELD NOT FOUND", "MEDIAN_FIELD_check")
}

we_write_csv(raw_mapping$validation,
               we_path("04_RAW_BASELINE_MAPPING", "WE_RAW_EXPOSURE_BASELINE_POOL_validation.csv"))
we_write_csv(raw_mapping$coordinates,
               we_path("04_RAW_BASELINE_MAPPING", "WE_RAW_CONTRAST_COORDINATES.csv"))
we_write_csv(raw_mapping$baseline_validation,
               we_path("04_RAW_BASELINE_MAPPING", "WE_RAW_BASELINE_MEAN_VALIDATION_BY_GROUP.csv"))

pool_output <- raw_mapping$pool[, c(
  "point_id", "Year", "Month", "precipitation_mm", "radiation_MJ_m2",
  "precipitation_mm_baseline_mean", "radiation_MJ_m2_baseline_mean",
  "precip_anomaly_mm", "radiation_anomaly_MJ_m2",
  "historically_active_canonical"
), drop = FALSE]
pool_output$source_monthly_table <- basename(WE_RAW_MONTHLY_PATH)
pool_output$baseline_key <- paste(pool_output$point_id, pool_output$Month, sep = "|")
we_write_csv(pool_output,
               we_path("04_RAW_BASELINE_MAPPING", "WE_RAW_EXPOSURE_BASELINE_POOL_DERIVED.csv"))

group_key <- paste(raw_mapping$pool$point_id, raw_mapping$pool$Month, sep = "|")
group_center <- data.frame(
  baseline_key = names(table(group_key)),
  n = as.integer(table(group_key)),
  precip_anomaly_group_mean = as.numeric(tapply(
    raw_mapping$pool$precip_anomaly_mm, group_key, mean
  )),
  radiation_anomaly_group_mean = as.numeric(tapply(
    raw_mapping$pool$radiation_anomaly_MJ_m2, group_key, mean
  )),
  stringsAsFactors = FALSE
)
group_center$centering_pass <- with(
  group_center,
  abs(precip_anomaly_group_mean) <= 1e-10 &
    abs(radiation_anomaly_group_mean) <= 1e-10
)
we_write_csv(group_center,
               we_path("04_RAW_BASELINE_MAPPING", "WE_RAW_ANOMALY_CENTERING_BY_GROUP.csv"))

evaluation_direct <- data.frame(
  point_id = as.character(primary$point_id), Year = primary$Year, Month = primary$Month,
  calendar_month_index = primary$calendar_month_index,
  precipitation_mm = primary$precipitation_mm,
  precipitation_baseline_mean = primary$precipitation_mm_baseline_mean,
  precip_anomaly_mm = primary$precip_anomaly_mm,
  precip_direct_reconstruction = primary$precipitation_mm - primary$precipitation_mm_baseline_mean,
  radiation_MJ_m2 = primary$radiation_MJ_m2,
  radiation_baseline_mean = primary$radiation_MJ_m2_baseline_mean,
  radiation_anomaly_MJ_m2 = primary$radiation_anomaly_MJ_m2,
  radiation_direct_reconstruction = primary$radiation_MJ_m2 - primary$radiation_MJ_m2_baseline_mean,
  precip_raw_global_support_rank = primary$precip_raw_global_support_rank,
  radiation_raw_global_support_rank = primary$radiation_raw_global_support_rank,
  stringsAsFactors = FALSE
)
evaluation_direct$precip_reconstruction_difference <- with(
  evaluation_direct, precip_anomaly_mm - precip_direct_reconstruction
)
evaluation_direct$radiation_reconstruction_difference <- with(
  evaluation_direct, radiation_anomaly_MJ_m2 - radiation_direct_reconstruction
)
we_write_csv(evaluation_direct,
               we_path("04_RAW_BASELINE_MAPPING", "WE_RAW_EXPOSURE_EVALUATION_DERIVED_ANALYSIS_TABLE.csv"))
raw_eval_validation <- data.frame(
  actual_precip_anomaly_field = WE_RAW_ACTUAL_FIELDS[["precip"]],
  actual_radiation_anomaly_field = WE_RAW_ACTUAL_FIELDS[["radiation"]],
  precip_source_field = WE_RAW_SOURCE_FIELDS[["precip"]],
  radiation_source_field = WE_RAW_SOURCE_FIELDS[["radiation"]],
  precip_unit = "mm", radiation_unit = "MJ m^-2",
  n = nrow(evaluation_direct),
  max_abs_precip_reconstruction_difference = max(abs(evaluation_direct$precip_reconstruction_difference)),
  max_abs_radiation_reconstruction_difference = max(abs(evaluation_direct$radiation_reconstruction_difference)),
  old_200mm_threshold_used = FALSE,
  status = if (max(abs(evaluation_direct$precip_reconstruction_difference)) <= 1e-12 &&
                max(abs(evaluation_direct$radiation_reconstruction_difference)) <= 1e-12)
    "PASS" else "FAIL",
  stringsAsFactors = FALSE
)
we_write_csv(raw_eval_validation,
               we_path("04_RAW_BASELINE_MAPPING", "WE_RAW_EXPOSURE_EVALUATION_FIELD_validation.csv"))
if (raw_eval_validation$status != "PASS") {
  we_stop_check("STOP — RAW EXPOSURE ANOMALY NOT UNIQUELY RECONSTRUCTIBLE",
                 "RAW_EVALUATION_FIELD_check")
}

mapping_summary <- data.frame(
  actual_precip_field = WE_RAW_ACTUAL_FIELDS[["precip"]],
  actual_radiation_field = WE_RAW_ACTUAL_FIELDS[["radiation"]],
  precipitation_units = "mm", radiation_units = "MJ m^-2",
  baseline_years = "2005-2018", baseline_pool_n = nrow(raw_mapping$pool),
  baseline_group_count = raw_mapping$validation$n_site_month_groups,
  baseline_site_count = raw_mapping$validation$n_sites,
  quantile_type = raw_mapping$coordinates$quantile_type,
  Pwet_raw = raw_mapping$coordinates$Pwet_raw_mm,
  Rdim_raw = raw_mapping$coordinates$Rdim_raw_MJ_m2,
  Rmid_raw = raw_mapping$coordinates$Rmid_raw_MJ_m2,
  Pwet_support_rank_actual = raw_mapping$coordinates$Pwet_raw_support_rank_actual,
  Rdim_support_rank_actual = raw_mapping$coordinates$Rdim_raw_support_rank_actual,
  Rmid_support_rank_actual = raw_mapping$coordinates$Rmid_raw_support_rank_actual,
  baseline_start_year = 2005L, baseline_end_year = 2018L,
  stringsAsFactors = FALSE
)
we_write_csv(mapping_summary,
               we_path("04_RAW_BASELINE_MAPPING", "WE_RAW_EXPOSURE_MAPPING_SUMMARY.csv"))

provenance <- c(
  "# water-energy Raw Exposure Mapping Provenance", "",
  sprintf("- Actual precipitation anomaly field: `%s` (mm).", WE_RAW_ACTUAL_FIELDS[["precip"]]),
  sprintf("- Actual radiation anomaly field: `%s` (MJ m^-2).", WE_RAW_ACTUAL_FIELDS[["radiation"]]),
  sprintf("- reference monthly source fields: `%s` and `%s`.",
          WE_RAW_SOURCE_FIELDS[["precip"]], WE_RAW_SOURCE_FIELDS[["radiation"]]),
  "- Each anomaly is the current-month raw value minus the reference 2005-2018 same point_id x calendar-Month historical mean.",
  "- The historical baseline pool contains only 2005-2018 canonical monthly records belonging to historically-active canonical point_id x Month groups with both anomalies finite.",
  "- Pwet_raw, Rdim_raw, and Rmid_raw use quantile type 1 at probabilities 0.75, 0.25, and 0.50, respectively.",
  "- Raw contrast coordinates use the 2005-2018 baseline pool.",
  "- Empirical support ranks define the radius-0.10 support check."
)
we_write_lines(provenance,
                 we_path("04_RAW_BASELINE_MAPPING", "WE_RAW_EXPOSURE_MAPPING_PROVENANCE.md"))

support <- rbind(
  we_definition_support_validation(primary, "MEDIAN_RESPONSE", raw_mapping),
  we_definition_support_validation(primary, "RAW_EXPOSURE", raw_mapping)
)
we_write_csv(support,
               we_path("05_SUPPORT", "WE_DEFINITION_FIXED_CONTRAST_SUPPORT_validation.csv"))
we_write_csv(support[support$analysis == "MEDIAN_RESPONSE", , drop = FALSE],
               we_path("02_MEDIAN_RESPONSE", "WE_MEDIAN_RESPONSE_SUPPORT_validation.csv"))
we_write_csv(support[support$analysis == "RAW_EXPOSURE", , drop = FALSE],
               we_path("03_RAW_EXPOSURE", "WE_RAW_EXPOSURE_SUPPORT_validation.csv"))
if (!all(support$status[support$analysis == "MEDIAN_RESPONSE"] == "PASS")) {
  we_stop_check("STOP — MEDIAN RESPONSE FIXED CONTRAST UNSUPPORTED",
                 "MEDIAN_SUPPORT_check")
}
if (!all(support$status[support$analysis == "RAW_EXPOSURE"] == "PASS")) {
  we_stop_check("RAW_EXPOSURE_FIXED_CONTRAST_UNSUPPORTED", "RAW_SUPPORT_check")
}

ar <- we_ar_sequence_validation(primary, "point_id")
ar$summary$analysis <- "PRIMARY_DOMAIN_SHARED"
primary_ar <- read.csv(file.path(
  WE_PRIMARY_ROOT, "06_AR_DIAGNOSTICS", "WE_PRIMARY_AR_SEQUENCE_validation.csv"
), stringsAsFactors = FALSE)
ar_match <- ar$summary$n_records == primary_ar$n_records &&
  ar$summary$n_sites == primary_ar$n_sites &&
  ar$summary$n_AR_sections == primary_ar$n_AR_sections &&
  ar$summary$gap_count == primary_ar$gap_count
ar$summary$canonical_primary_AR_sections <- primary_ar$n_AR_sections
ar$summary$canonical_primary_gap_count <- primary_ar$gap_count
ar$summary$matches_canonical_primary <- ar_match
we_write_csv(ar$summary,
               we_path("07_DIAGNOSTICS", "WE_DEFINITION_AR_SEQUENCE_validation.csv"))
we_write_csv(ar$gaps,
               we_path("07_DIAGNOSTICS", "WE_DEFINITION_AR_GAPS.csv"))
we_write_csv(ar$gap_distribution,
               we_path("07_DIAGNOSTICS", "WE_DEFINITION_AR_GAP_LENGTH_DISTRIBUTION.csv"))
if (!ar_match || ar$summary$n_AR_sections != 988L) {
  we_stop_check("STOP — AR STRUCTURE MISMATCH", "AR_SEQUENCE_check")
}

prepare_state <- list(
  complete = TRUE, created_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
  primary_identity = identity, domain_validation = domain_validation, median_validation = median_validation,
  raw_validation = raw_mapping$validation, coordinates = raw_mapping$coordinates,
  mapping_summary = mapping_summary, support = support, ar_summary = ar$summary,
  R_version = R.version.string, mgcv_version = as.character(packageVersion("mgcv"))
)
saveRDS(prepare_state,
        we_path("10_LOGS", "WE_DEFINITION_prepare_STATE.rds"))

we_log(sprintf(
  paste0("Pre-validation PASS primary=%d sites=%d AR=%d pool=%d groups=%d ",
         "coords=[%.15g,%.15g,%.15g] support Median A/B=%d/%d Raw A/B=%d/%d"),
  nrow(primary), length(levels(primary$point_id)), ar$summary$n_AR_sections,
  nrow(raw_mapping$pool), raw_mapping$validation$n_site_month_groups,
  raw_mapping$coordinates$Pwet_raw_mm, raw_mapping$coordinates$Rdim_raw_MJ_m2,
  raw_mapping$coordinates$Rmid_raw_MJ_m2,
  support$n_records[support$analysis == "MEDIAN_RESPONSE" & support$contrast_point == "A"],
  support$n_records[support$analysis == "MEDIAN_RESPONSE" & support$contrast_point == "B"],
  support$n_records[support$analysis == "RAW_EXPOSURE" & support$contrast_point == "A"],
  support$n_records[support$analysis == "RAW_EXPOSURE" & support$contrast_point == "B"]
))
