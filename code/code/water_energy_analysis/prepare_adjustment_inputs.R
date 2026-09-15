source(file.path(Sys.getenv("WATER_ENERGY_CODE_DIR", unset = getwd()), "adjustment_common.R"))

we_log("water-energy Model2/Model3 adjustment robustness prepare started")

environment_validation <- data.frame(
  item = c("R.version.string", "packageVersion('mgcv')", "mgcv_available"),
  value = c(
    R.version.string,
    if (requireNamespace("mgcv", quietly = TRUE)) as.character(packageVersion("mgcv")) else NA_character_,
    as.character(requireNamespace("mgcv", quietly = TRUE))
  ),
  status = c(
    if (grepl("R version 4.6.1", R.version.string, fixed = TRUE)) "PASS" else "FAIL",
    if (requireNamespace("mgcv", quietly = TRUE) &&
        as.character(packageVersion("mgcv")) == "1.9.4") "PASS" else "FAIL",
    if (requireNamespace("mgcv", quietly = TRUE)) "PASS" else "FAIL"
  ),
  package_install_or_update_performed = FALSE,
  stringsAsFactors = FALSE
)
we_write_csv(
  environment_validation,
  we_path("09_LOGS", "WE_ADJUSTMENT_ENVIRONMENT_validation.csv")
)
if (!requireNamespace("mgcv", quietly = TRUE)) {
  we_stop_check("STOP — MGCV NOT AVAILABLE", "ENVIRONMENT_check")
}
if (any(environment_validation$status != "PASS")) {
  we_stop_check(
    "STOP — R/mgcv environment differs from recorded analysis environment", "ENVIRONMENT_check",
    paste(environment_validation$item[environment_validation$status != "PASS"], collapse = ";")
  )
}

required_paths <- c(
  WE_MASTER_PATH,
  we_model_path("MODEL2"), we_model_path("MODEL3"),
  we_formula_path("MODEL2"), we_formula_path("MODEL3"),
  we_model_info_path("MODEL2"), we_model_info_path("MODEL3"),
  file.path(WE_PRIMARY_ROOT, "09_PRIMARY_CONTRAST", "WE_MODEL_CONTRAST_COMPARISON.csv"),
  file.path(WE_PRIMARY_ROOT, "10_BOOTSTRAP", "WE_MODEL1_PRIMARY_CONTRAST_SUMMARY.csv")
)
input_validation <- data.frame(
  path = normalizePath(required_paths, winslash = "/", mustWork = FALSE),
  exists = file.exists(required_paths), reference_input = TRUE,
  status = ifelse(file.exists(required_paths), "PASS", "FAIL"),
  stringsAsFactors = FALSE
)
we_write_csv(input_validation, we_path("09_LOGS", "WE_ADJUSTMENT_INPUT_validation.csv"))
if (any(!input_validation$exists)) {
  we_stop_check(
    "STOP — canonical MODEL2/MODEL3 PRIMARY REFERENCE NOT FOUND", "REFERENCE_check",
    paste(input_validation$path[!input_validation$exists], collapse = ";")
  )
}

models <- list(
  MODEL2 = readRDS(we_model_path("MODEL2")),
  MODEL3 = readRDS(we_model_path("MODEL3"))
)
master <- we_read_master()
data <- we_prepare_primary_adjustment_data(master)

primary_reference <- we_primary_reference_validation()
domain <- we_domain_identity_validation(data, models)
formula_validation <- we_formula_validation(models)
reference <- we_model_reference_validation(models)
ar <- we_ar_sequence_validation(data, "point_id")
support <- we_support_validation_adjustment(data)

finite_fields <- do.call(rbind, lapply(WE_MODEL_NAMES, function(model_name) {
  fields <- if (model_name == "MODEL2") {
    c(
      "npp_anomaly_mean_g", "precip_rank_baseline", "radiation_rank_baseline",
      WE_MODEL1_CONTROLS, WE_MODEL2_ADDED
    )
  } else {
    c(
      "npp_anomaly_mean_g", "precip_rank_baseline", "radiation_rank_baseline",
      WE_MODEL1_CONTROLS, WE_MODEL2_ADDED, WE_MODEL3_ADDED
    )
  }
  do.call(rbind, lapply(fields, function(field) {
    data.frame(
      model = model_name, field = field, n = nrow(data),
      missing_count = sum(is.na(data[[field]])),
      nonfinite_count = sum(!is.finite(data[[field]])),
      status = if (all(is.finite(data[[field]]))) "PASS" else "FAIL",
      stringsAsFactors = FALSE
    )
  }))
}))

health <- do.call(rbind, lapply(WE_MODEL_NAMES, function(model_name) {
  h <- we_model_health(models[[model_name]], character())
  h$model <- model_name
  h$canonical_primary_object <- TRUE
  h$full_data_refitted <- FALSE
  h[, c("model", setdiff(names(h), "model")), drop = FALSE]
}))

we_write_csv(primary_reference, we_path("09_LOGS", "WE_PRIMARY_REFERENCE_validation.csv"))
we_write_csv(domain, we_path("09_LOGS", "WE_MODEL2_MODEL3_DOMAIN_KEY_IDENTITY.csv"))
we_write_csv(formula_validation, we_path("09_LOGS", "WE_MODEL2_MODEL3_FORMULA_validation.csv"))
we_write_csv(reference, we_path("09_LOGS", "WE_MODEL2_MODEL3_PRIMARY_REFERENCE_validation.csv"))
we_write_csv(finite_fields, we_path("09_LOGS", "WE_MODEL2_MODEL3_FIELD_COMPLETENESS.csv"))
we_write_csv(health, we_path("09_LOGS", "WE_MODEL2_MODEL3_canonical_HEALTH_validation.csv"))
we_write_csv(ar$summary, we_path("05_DIAGNOSTICS", "WE_ADJUSTMENT_AR_SEQUENCE_validation.csv"))
we_write_csv(ar$gaps, we_path("05_DIAGNOSTICS", "WE_ADJUSTMENT_AR_GAPS.csv"))
we_write_csv(ar$gap_distribution, we_path("05_DIAGNOSTICS", "WE_ADJUSTMENT_AR_GAP_DISTRIBUTION.csv"))
we_write_csv(support, we_path("06_SUPPORT", "WE_MODEL2_MODEL3_CONTRAST_SUPPORT_validation.csv"))
we_write_csv(
  we_analysis_selector_validation(),
  we_path("09_LOGS", "WE_NO_UNAPPROVED_ANALYSIS_validation.csv")
)
for (model_name in WE_MODEL_NAMES) {
  model_dir <- if (model_name == "MODEL2") "02_MODEL2" else "03_MODEL3"
  we_write_lines(
    we_formula_text(formula(models[[model_name]])),
    we_path(model_dir, paste0("WE_", model_name, "_EXACT_canonical_FORMULA.txt"))
  )
  we_write_csv(
    data.frame(
      model = model_name,
      canonical_RDS_path = normalizePath(we_model_path(model_name), winslash = "/"),
      canonical_formula_path = normalizePath(we_formula_path(model_name), winslash = "/"),
      full_data_refitted = FALSE, primary_output_overwritten = FALSE,
      stringsAsFactors = FALSE
    ),
    we_path(model_dir, paste0("WE_", model_name, "_READ_ONLY_REUSE_validation.csv"))
  )
}

if (any(primary_reference$status != "PASS")) {
  we_stop_check("STOP — PRIMARY REFERENCE MISMATCH", "PRIMARY_REFERENCE_check")
}
if (nrow(data) != 6992L || nlevels(data$point_id) != 247L || any(domain$status != "PASS")) {
  we_stop_check(
    "STOP — MODEL2/3 DOMAIN NOT IDENTICAL TO PRIMARY", "DOMAIN_check",
    sprintf("n=%d sites=%d", nrow(data), nlevels(data$point_id))
  )
}
if (any(formula_validation$status != "PASS") || any(reference$status != "PASS")) {
  we_stop_check(
    "STOP — canonical MODEL2/MODEL3 PRIMARY REFERENCE NOT FOUND", "REFERENCE_FORMULA_check"
  )
}
if (any(finite_fields$status != "PASS")) {
  we_stop_check(
    "STOP — MODEL2/3 DOMAIN NOT IDENTICAL TO PRIMARY", "FIELD_COMPLETENESS_check",
    paste(finite_fields$field[finite_fields$status != "PASS"], collapse = ";")
  )
}
if (ar$summary$n_AR_sections != 988L) {
  we_stop_check(
    "STOP — MODEL2/3 AR STRUCTURE MISMATCH", "AR_STRUCTURE_check",
    sprintf("sections=%d", ar$summary$n_AR_sections)
  )
}
if (any(support$status != "PASS")) {
  we_stop_check("STOP — MODEL2/3 FIXED CONTRAST SUPPORT INCONSISTENT", "SUPPORT_check")
}
if (any(vapply(WE_MODEL_NAMES, function(model_name) {
  length(we_model_health_stop_reason(health[health$model == model_name, , drop = FALSE])) > 0L
}, logical(1)))) {
  we_stop_check("STOP — canonical MODEL2/MODEL3 NUMERICAL HEALTH FAILURE", "HEALTH_check")
}

state <- list(
  prepare_complete = TRUE,
  completed_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
  n = nrow(data), n_sites = nlevels(data$point_id), ar = ar$summary,
  support = support, primary_reference = primary_reference,
  domain = domain, formula_validation = formula_validation, reference = reference,
  full_data_refitted = FALSE
)
saveRDS(state, we_path("09_LOGS", "WE_ADJUSTMENT_prepare_STATE.rds"))

we_log(sprintf(
  "prepare PASS n=%d sites=%d AR_sections=%d Model2_ref=%.15g Model3_ref=%.15g",
  nrow(data), nlevels(data$point_id), ar$summary$n_AR_sections,
  reference$canonical_contrast[reference$model == "MODEL2"],
  reference$canonical_contrast[reference$model == "MODEL3"]
))
