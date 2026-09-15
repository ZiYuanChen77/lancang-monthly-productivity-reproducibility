source(file.path(Sys.getenv("WATER_ENERGY_CODE_DIR", unset = getwd()), "definition_common.R"))

results <- list()
test_index <- 0L
run_test <- function(name, code) {
  test_index <<- test_index + 1L
  started <- proc.time()[["elapsed"]]
  outcome <- tryCatch({
    value <- force(code)
    if (!isTRUE(value)) stop("assertion returned non-TRUE", call. = FALSE)
    list(status = "PASS", detail = "")
  }, error = function(e) list(status = "FAIL", detail = conditionMessage(e)))
  results[[test_index]] <<- data.frame(
    test_id = test_index, test_name = name, status = outcome$status,
    detail = outcome$detail, elapsed_seconds = proc.time()[["elapsed"]] - started,
    stringsAsFactors = FALSE
  )
  invisible(outcome$status == "PASS")
}

master <- we_read_master()
raw_mapping <- we_build_raw_mapping()
primary <- we_prepare_primary_domain(master, raw_mapping)
primary_flag <- we_flag_true(master$eligible_primary_model1)

run_test("Primary domain exact 6992", nrow(primary) == 6992L)
run_test("Primary site count 247", length(levels(primary$point_id)) == 247L)
run_test("Median response field exists", "npp_anomaly_median_g" %in% names(master))
run_test("Median response is not mean response", {
  sum(abs(primary$npp_anomaly_median_g - primary$npp_anomaly_mean_g) > 1e-12) > 0L
})
run_test("Median only changes response", {
  identical(
    attr(terms(we_formula_definition("MEDIAN_RESPONSE")), "term.labels"),
    attr(terms(we_formula_model1()), "term.labels")
  ) && all.vars(we_formula_definition("MEDIAN_RESPONSE"))[1L] == "npp_anomaly_median_g"
})
run_test("Median fixed rank exposure", {
  text <- paste(deparse(we_formula_definition("MEDIAN_RESPONSE")), collapse = " ")
  grepl("te\\(precip_rank_baseline, radiation_rank_baseline", text)
})
run_test("Median fixed contrast", {
  identical(unname(WE_RANK_CONTRAST_A), c(0.75, 0.25)) &&
    identical(unname(WE_RANK_CONTRAST_B), c(0.75, 0.50))
})
run_test("Raw response remains mean anomaly", {
  all.vars(we_formula_definition("RAW_EXPOSURE"))[1L] == "npp_anomaly_mean_g"
})
run_test("Raw exposure uses raw anomalies", {
  text <- paste(deparse(we_formula_definition("RAW_EXPOSURE")), collapse = " ")
  grepl("te\\(precip_anomaly_mm, radiation_anomaly_MJ_m2", text)
})
run_test("Raw surface excludes Primary ranks", {
  first_term <- attr(terms(we_formula_definition("RAW_EXPOSURE")), "term.labels")[1L]
  !grepl("rank_baseline", first_term)
})
run_test("Raw units mm and MJ m-2", {
  identical(WE_RAW_ACTUAL_FIELDS[["precip"]], "precip_anomaly_mm") &&
    identical(WE_RAW_ACTUAL_FIELDS[["radiation"]], "radiation_anomaly_MJ_m2")
})
run_test("Baseline pool only 2005-2018", {
  identical(range(raw_mapping$pool$Year), c(2005L, 2018L)) &&
    length(unique(raw_mapping$pool$Year)) == 14L
})
run_test("Baseline pool active groups only", {
  all(we_flag_true(raw_mapping$pool$historically_active_canonical))
})
run_test("Raw coordinate baseline period", {
  !any(raw_mapping$pool$Year %in% 2022:2025) &&
    identical(raw_mapping$coordinates$derivation_years, "2005-2018")
})
run_test("Raw quantile type 1", raw_mapping$coordinates$quantile_type == 1L)
run_test("Pwet equals historical Q75 type1", {
  isTRUE(all.equal(
    raw_mapping$coordinates$Pwet_raw_mm,
    unname(quantile(raw_mapping$pool$precip_anomaly_mm, 0.75, type = 1)),
    tolerance = 0
  ))
})
run_test("Rdim equals historical Q25 type1", {
  isTRUE(all.equal(
    raw_mapping$coordinates$Rdim_raw_MJ_m2,
    unname(quantile(raw_mapping$pool$radiation_anomaly_MJ_m2, 0.25, type = 1)),
    tolerance = 0
  ))
})
run_test("Rmid equals historical Q50 type1", {
  isTRUE(all.equal(
    raw_mapping$coordinates$Rmid_raw_MJ_m2,
    unname(quantile(raw_mapping$pool$radiation_anomaly_MJ_m2, 0.50, type = 1)),
    tolerance = 0
  ))
})
run_test("Baseline-quantile coordinate method", {
  identical(raw_mapping$coordinates$coordinate_method, "baseline_quantiles")
})
run_test("Raw support ranks excluded from model formula", {
  !any(c("precip_raw_global_support_rank", "radiation_raw_global_support_rank") %in%
         all.vars(we_formula_definition("RAW_EXPOSURE")))
})
run_test("Actual mapped support centers used", {
  s <- we_definition_support_validation(primary, "RAW_EXPOSURE", raw_mapping)
  isTRUE(all.equal(
    s$center_precip[1L], raw_mapping$coordinates$Pwet_raw_support_rank_actual,
    tolerance = 0
  )) && isTRUE(all.equal(
    s$center_radiation[s$contrast_point == "A"],
    raw_mapping$coordinates$Rdim_raw_support_rank_actual, tolerance = 0
  )) && isTRUE(all.equal(
    s$center_radiation[s$contrast_point == "B"],
    raw_mapping$coordinates$Rmid_raw_support_rank_actual, tolerance = 0
  ))
})
run_test("Support radius 0.10", identical(WE_SUPPORT_RADIUS, 0.10))
all_support <- rbind(
  we_definition_support_validation(primary, "MEDIAN_RESPONSE", raw_mapping),
  we_definition_support_validation(primary, "RAW_EXPOSURE", raw_mapping)
)
run_test("Support records check", all(all_support$n_records >= 100L))
run_test("Support sites check", all(all_support$n_sites >= 50L))
run_test("Support years check", all(all_support$n_years >= 3L))
run_test("Fixed Primary rho", identical(WE_RHO_FIXED, 0.036623315093487))
run_test("AR exact Primary structure", {
  ar <- we_ar_sequence_validation(primary, "point_id")$summary
  canonical <- read.csv(file.path(
    WE_PRIMARY_ROOT, "06_AR_DIAGNOSTICS", "WE_PRIMARY_AR_SEQUENCE_validation.csv"
  ), stringsAsFactors = FALSE)
  ar$n_AR_sections == 988L && ar$n_AR_sections == canonical$n_AR_sections &&
    ar$gap_count == canonical$gap_count
})
run_test("Surface fixed 5 by 5", all(vapply(WE_ALLOWED_ANALYSES, function(a) {
  grepl("k = c\\(5, 5\\)", paste(deparse(we_formula_definition(a)), collapse = " "))
}, logical(1))))
run_test("Controls fixed k5", all(vapply(WE_ALLOWED_ANALYSES, function(a) {
  text <- gsub("[[:space:]]+", " ", paste(deparse(we_formula_definition(a)), collapse = " "))
  all(vapply(WE_MODEL1_CONTROLS, function(v) {
    grepl(sprintf("s\\(%s, k = 5\\)", v), text)
  }, logical(1)))
}, logical(1))))
run_test("Fixed basis dimensions", !exists("we_definition_k_fallback", mode = "function"))

canonical_model <- readRDS(file.path(
  WE_PRIMARY_ROOT, "03_MODEL1_PRIMARY", "WE_MODEL1_PRIMARY.rds"
))
reference_row <- we_definition_reference_row(primary, "MEDIAN_RESPONSE", raw_mapping)
lptest <- we_definition_lpmatrix_contrast(
  canonical_model, reference_row, "MEDIAN_RESPONSE", raw_mapping
)
run_test("Lpmatrix contrast extraction", {
  isTRUE(all.equal(lptest$estimate, as.numeric(lptest$dX %*% coef(canonical_model)),
                   tolerance = 1e-14))
})
canonical_invariance <- we_definition_contrast_invariance(
  canonical_model, primary, "MEDIAN_RESPONSE", raw_mapping
)
run_test("Contrast invariance implementation", {
  canonical_invariance$pass && canonical_invariance$max_absolute_difference <= 1e-8
})
run_test("Bootstrap target 1000", identical(WE_BOOT_TARGET_SUCCESS, 1000L))
run_test("Bootstrap maximum 1050", identical(WE_BOOT_DRAWS, 1050L))
run_test("Duplicate-site independent IDs", {
  identical(
    we_boot_series_ids(c("029", "029", "030", "029")),
    c("029_copy01", "029_copy02", "030_copy01", "029_copy03")
  )
})
run_test("Bootstrap fixed rho", {
  identical(formals(we_run_definition_boot_attempt)$rho_used, quote(WE_RHO_FIXED))
})
run_test("Fixed bootstrap contrast coordinates", {
  body_text <- paste(deparse(body(we_run_definition_boot_attempt)), collapse = " ")
  !grepl("we_build_raw_mapping|quantile\\(", body_text)
})
run_test("Model selector", {
  !any(c("MODEL0", "MODEL2", "MODEL3") %in% WE_ALLOWED_ANALYSES)
})
run_test("Analysis-stage selector", !"LOYO" %in% WE_ALLOWED_ANALYSES)
run_test("Sensitivity analysis selector", identical(
  WE_ALLOWED_ANALYSES, c("MEDIAN_RESPONSE", "RAW_EXPOSURE")
))
run_test("Sensitivity selector list is fixed", {
  identical(sort(WE_ALLOWED_ANALYSES), sort(c("MEDIAN_RESPONSE", "RAW_EXPOSURE")))
})
run_test("Local input workflow", {
  we_software_validation()$network_download == FALSE
})
run_test("Primary key identity exact", we_primary_identity_validation(master)$status == "PASS")
run_test("Median eligibility flag identity", {
  !any(xor(primary_flag, we_flag_true(master$eligible_median_response)))
})
run_test("Raw eligibility flag identity", {
  !any(xor(primary_flag, we_flag_true(master$eligible_raw_exposure)))
})
run_test("Evaluation raw anomaly fields match deterministic definition", {
  max(abs(primary$precip_anomaly_mm -
            (primary$precipitation_mm - primary$precipitation_mm_baseline_mean))) <= 1e-12 &&
    max(abs(primary$radiation_anomaly_MJ_m2 -
            (primary$radiation_MJ_m2 - primary$radiation_MJ_m2_baseline_mean))) <= 1e-12
})
run_test("Historical anomaly centering pass", {
  raw_mapping$validation$centering_status == "PASS" &&
    raw_mapping$validation$max_abs_precip_group_mean_anomaly <= 1e-10 &&
    raw_mapping$validation$max_abs_radiation_group_mean_anomaly <= 1e-10
})
run_test("Historical pool complete 14 records per group", {
  raw_mapping$validation$records_per_group_min == 14L &&
    raw_mapping$validation$records_per_group_max == 14L
})
run_test("Empirical support rank formula including ties", {
  isTRUE(all.equal(raw_mapping$empirical_rank(2, c(1, 2, 2, 3)), 0.5, tolerance = 0))
})
run_test("reference reference values all match", all(we_reference_reference_validation()$status == "PASS"))
run_test("Applicable primary analysis tests remain PASS", {
  x <- read.csv(file.path(
    WE_PRIMARY_ROOT, "13_TESTS", "WE_GAMM_PRIMARY_TEST_RESULTS.csv"
  ), stringsAsFactors = FALSE)
  nrow(x) > 0L && all(x$status == "PASS")
})
run_test("Target robustness tests remain PASS", {
  x <- read.csv(file.path(
    WE_TARGET_ROOT, "09_TESTS", "WE_QA_DAYWEIGHT_ALL_TEST_RESULTS.csv"
  ), stringsAsFactors = FALSE)
  nrow(x) > 0L && all(x$status == "PASS")
})
run_test("Domain robustness tests remain PASS", {
  x <- read.csv(file.path(
    WE_DOMAIN_ROOT, "08_TESTS", "WE_DOMAIN_SENSITIVITY_ALL_TEST_RESULTS.csv"
  ), stringsAsFactors = FALSE)
  nrow(x) > 0L && all(x$status == "PASS")
})
run_test("Environment has reference mgcv", {
  requireNamespace("mgcv", quietly = TRUE) &&
    as.character(packageVersion("mgcv")) == "1.9.4"
})
run_test("Formula interaction specification", all(vapply(WE_ALLOWED_ANALYSES, function(a) {
  !any(grepl(":", attr(terms(we_formula_definition(a)), "term.labels"), fixed = TRUE))
}, logical(1))))

test_results <- do.call(rbind, results)
we_write_csv(test_results,
               we_path("09_TESTS", "WE_DEFINITION_SENSITIVITY_TEST_RESULTS.csv"))
summary <- data.frame(
  tests_pass = sum(test_results$status == "PASS"), total_tests = nrow(test_results),
  tests_fail = sum(test_results$status == "FAIL"),
  status = if (all(test_results$status == "PASS")) "PASS" else "FAIL",
  stringsAsFactors = FALSE
)
we_write_csv(summary,
               we_path("09_TESTS", "WE_DEFINITION_SENSITIVITY_TEST_SUMMARY.csv"))
we_write_lines(sprintf("%d/%d PASS", summary$tests_pass, summary$total_tests),
                 we_path("09_TESTS", "WE_DEFINITION_SENSITIVITY_TEST_RESULTS.txt"))
if (!all(test_results$status == "PASS")) {
  we_stop_check(
    "STOP — AUTOMATED TEST FAILURE", "PRE_MODEL_TEST_check",
    paste(test_results$test_name[test_results$status == "FAIL"], collapse = ";")
  )
}
we_log(sprintf("Automated tests PASS %d/%d", summary$tests_pass, summary$total_tests))
