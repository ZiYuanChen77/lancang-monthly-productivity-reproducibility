source(file.path(Sys.getenv("WATER_ENERGY_CODE_DIR", unset = getwd()), "qa_dayweight_common.R"))

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
    test_id = test_index, test_name = name, status = outcome[["status"]],
    detail = outcome[["detail"]], elapsed_seconds = proc.time()[["elapsed"]] - started,
    stringsAsFactors = FALSE
  )
  invisible(outcome[["status"]] == "PASS")
}

master <- we_read_master()
primary_flag <- we_flag_true(master[["eligible_primary_model1"]])
qa1_flag <- we_flag_true(master[["eligible_mod17_QA1"]])
qa2_flag <- we_flag_true(master[["eligible_mod17_QA2"]])
dw_flag <- we_flag_true(master[["eligible_dayweighted"]])
qa1_explicit <- primary_flag & we_flag_true(master[["mod17_QA1"]])
qa2_explicit <- primary_flag & we_flag_true(master[["mod17_QA2"]])
qa1 <- we_prepare_sensitivity(master, "QA1")
qa2 <- we_prepare_sensitivity(master, "QA2")
dw <- we_prepare_sensitivity(master, "DAYWEIGHTED")

run_test("QA1 explicit intersection", identical(qa1_explicit, we_explicit_eligibility(master, "QA1")))
run_test("QA1 eligibility key identity", {
  x <- we_eligibility_identity_validation(master, "QA1")
  x$status == "PASS" && x$missing_key_count == 0L && x$extra_key_count == 0L && x$mismatch_flag_count == 0L
})
run_test("QA2 explicit intersection", identical(qa2_explicit, we_explicit_eligibility(master, "QA2")))
run_test("QA2 eligibility key identity", {
  x <- we_eligibility_identity_validation(master, "QA2")
  x$status == "PASS" && x$missing_key_count == 0L && x$extra_key_count == 0L && x$mismatch_flag_count == 0L
})
run_test("QA1 does not recompute response", {
  identical(as.numeric(qa1$npp_anomaly_mean_g),
            as.numeric(master$npp_anomaly_mean_g[qa1_flag][order(
              sprintf("%03d", as.integer(master$point_id[qa1_flag])),
              master$calendar_month_index[qa1_flag]
            )]))
})
run_test("QA2 does not recompute response", {
  identical(as.numeric(qa2$npp_anomaly_mean_g),
            as.numeric(master$npp_anomaly_mean_g[qa2_flag][order(
              sprintf("%03d", as.integer(master$point_id[qa2_flag])),
              master$calendar_month_index[qa2_flag]
            )]))
})
run_test("dayweighted uses dayweighted response", {
  identical(all.vars(we_formula_sensitivity("dayweighted_anomaly_mean_g"))[1L],
            "dayweighted_anomaly_mean_g") && all(is.finite(dw$dayweighted_anomaly_mean_g))
})
run_test("dayweighted uses dayweighted active", {
  all(we_flag_true(dw$historically_active_dayweighted)) &&
    identical(dw_flag, we_explicit_eligibility(master, "DAYWEIGHTED"))
})
run_test("dayweighted is not primary-only response replacement", {
  sum(primary_flag & !dw_flag) > 0L && sum(!primary_flag & dw_flag) > 0L &&
    nrow(dw) == sum(dw_flag)
})
run_test("fixed canonical Primary rho", identical(WE_RHO_FIXED, 0.036623315093487))

synthetic_gap <- data.frame(
  point_id = c("001", "001", "001"), calendar_month_index = c(100L, 101L, 103L)
)
run_test("QA month gap restarts AR", identical(we_build_ar_start(synthetic_gap), c(TRUE, FALSE, TRUE)))
run_test("dayweighted month gap restarts AR", identical(we_build_ar_start(synthetic_gap), c(TRUE, FALSE, TRUE)))

forms <- lapply(WE_ALLOWED_ANALYSES, function(a) {
  we_formula_sensitivity(WE_ANALYSIS_META[[a]]$response)
})
form_text <- vapply(forms, function(f) {
  gsub("[[:space:]]+", " ", paste(deparse(f), collapse = " "))
}, character(1))
run_test("fixed 5 by 5 surface", all(grepl(
  "te\\(precip_rank_baseline, radiation_rank_baseline, k = c\\(5, 5\\)\\)", form_text
)))
run_test("fixed k5 controls", all(vapply(WE_MODEL1_CONTROLS, function(v) {
  all(grepl(sprintf("s\\(%s, k = 5\\)", v), form_text))
}, logical(1))))
run_test("fixed contrast A and B", {
  identical(unname(WE_CONTRAST_A), c(0.75, 0.25)) &&
    identical(unname(WE_CONTRAST_B), c(0.75, 0.50))
})
run_test("Euclidean radius support", {
  toy <- data.frame(
    point_id = factor(c("001", "002", "003")), Year = c(2022L, 2022L, 2022L),
    precip_rank_baseline = c(0.75, 0.75, 0.75),
    radiation_rank_baseline = c(0.25, 0.34, 0.36)
  )
  x <- we_support_validation(toy, "TOY")
  x$n_records[x$contrast_point == "A"] == 2L
})
run_test("QA1 primary support check", {
  x <- we_support_validation(qa1, "QA1")
  all(x$status == "PASS") && all(x$n_records >= 100L) && all(x$n_sites >= 50L) && all(x$n_years >= 3L)
})
run_test("QA2 coverage-limited unsupported check", {
  identical(we_check_label("QA2", -1, NA_real_, FALSE, FALSE),
            "QA2_COVERAGE_LIMITED_FIXED_CONTRAST_UNSUPPORTED")
})
run_test("dayweighted primary support check", {
  x <- we_support_validation(dw, "DAYWEIGHTED")
  all(x$status == "PASS") && all(x$n_records >= 100L) && all(x$n_sites >= 50L) && all(x$n_years >= 3L)
})
run_test("bootstrap target B1000", identical(WE_BOOT_TARGET_SUCCESS, 1000L))
run_test("bootstrap maximum 1050 attempts", identical(WE_BOOT_DRAWS, 1050L))
run_test("duplicate-site bootstrap IDs are independent", {
  ids <- we_boot_series_ids(c("029", "029", "030", "029"))
  identical(ids, c("029_copy01", "029_copy02", "030_copy01", "029_copy03")) &&
    length(unique(ids)) == length(ids)
})
run_test("bootstrap uses fixed rho", {
  identical(formals(we_run_sensitivity_boot_attempt)$rho_used, quote(WE_RHO_FIXED))
})
run_test("no LOYO specified", !"LOYO" %in% WE_ALLOWED_ANALYSES)
run_test("no Model0 Model2 Model3 specified", !any(c("MODEL0", "MODEL2", "MODEL3") %in% WE_ALLOWED_ANALYSES))
run_test("Sensitivity analysis selector", identical(WE_ALLOWED_ANALYSES, c("QA1", "QA2", "DAYWEIGHTED")))
run_test("QA2 sign reversal STOP label", {
  identical(we_check_label("QA2", 0.01, 0.2, TRUE, FALSE),
            "QA2_SIGN_REVERSAL_ANALYSIS_CHECK_REQUIRED")
})
run_test("QA1 bootstrap CI check", {
  we_check_label("QA1", -1, -0.1, TRUE, FALSE) == "QA1_ROBUSTNESS_PASS" &&
    we_check_label("QA1", -1, 0.1, TRUE, FALSE) == "QA1_CI_CROSSES_ZERO"
})
run_test("dayweighted bootstrap CI check", {
  we_check_label("DAYWEIGHTED", -1, -0.1, TRUE, FALSE) == "DAYWEIGHTED_ROBUSTNESS_PASS" &&
    we_check_label("DAYWEIGHTED", -1, 0.1, TRUE, FALSE) == "DAYWEIGHTED_CI_CROSSES_ZERO"
})

primary_model_path <- file.path(
  WE_PRIMARY_ROOT, "03_MODEL1_PRIMARY", "WE_MODEL1_PRIMARY.rds"
)
primary_mtime_before <- file.info(primary_model_path)$mtime
primary_model <- readRDS(primary_model_path)
primary <- we_primary_data(master)
primary_contrast <- we_lpmatrix_contrast(primary_model, we_reference_row(primary))$estimate
primary_mtime_after <- file.info(primary_model_path)$mtime
run_test("Primary reference input identity", {
  identical(primary_mtime_before, primary_mtime_after) &&
    isTRUE(all.equal(primary_contrast, -2.94258224191719, tolerance = 1e-12))
})
run_test("AR.start continuous month", identical(we_build_ar_start(synthetic_gap)[1:2], c(TRUE, FALSE)))
run_test("AR.start new site", {
  x <- data.frame(point_id = c("001", "002"), calendar_month_index = c(100L, 101L))
  identical(we_build_ar_start(x), c(TRUE, TRUE))
})
run_test("lpmatrix contrast extraction", {
  x <- we_lpmatrix_contrast(primary_model, we_reference_row(primary))
  isTRUE(all.equal(x$estimate, as.numeric(x$dX %*% coef(primary_model)), tolerance = 1e-14))
})
run_test("contrast invariance", {
  x <- we_contrast_invariance(primary_model, primary)
  isTRUE(x$pass) && x$max_absolute_difference <= 1e-8
})
run_test("duplicate-site copies have independent AR chains", {
  toy <- primary[as.character(primary$point_id) == levels(primary$point_id)[1L],
                 c("point_id", "Year", "Month", "calendar_month_index",
                   "npp_anomaly_mean_g", "precip_rank_baseline", "radiation_rank_baseline",
                   WE_MODEL1_CONTROLS), drop = FALSE]
  draw <- data.frame(
    attempt = 1L, draw_id = "draw_0001", draw_position = 1:2,
    original_point_id = rep(as.character(toy$point_id[1L]), 2),
    boot_series_id = c("x_copy01", "x_copy02"), stringsAsFactors = FALSE
  )
  b <- we_build_boot_data(toy, draw)
  all(tapply(b$AR.start, b$boot_series_id, function(v) isTRUE(v[1L]))) &&
    length(levels(b$boot_series_id)) == 2L
})
run_test("real QA sample counts", nrow(qa1) == 5280L && nrow(qa2) == 3358L)
run_test("real dayweighted direct-domain count", nrow(dw) == 6992L && length(levels(dw$point_id)) == 247L)
run_test("primary analysis regression tests remain PASS", {
  x <- we_primary_reference_validation()
  all(x$status == "PASS")
})
run_test("reference formula has no added interaction", {
  all(vapply(forms, function(f) {
    vars <- attr(terms(f), "term.labels")
    !any(grepl(":", vars, fixed = TRUE))
  }, logical(1)))
})
run_test("environment has reference mgcv", {
  requireNamespace("mgcv", quietly = TRUE) && as.character(packageVersion("mgcv")) == "1.9.4"
})

test_results <- do.call(rbind, results)
we_write_csv(test_results, we_path("09_TESTS", "WE_QA_DAYWEIGHT_TEST_RESULTS.csv"))
summary <- data.frame(
  tests_pass = sum(test_results$status == "PASS"), total_tests = nrow(test_results),
  tests_fail = sum(test_results$status == "FAIL"),
  status = if (all(test_results$status == "PASS")) "PASS" else "FAIL",
  stringsAsFactors = FALSE
)
we_write_csv(summary, we_path("09_TESTS", "WE_QA_DAYWEIGHT_TEST_SUMMARY.csv"))
we_write_lines(
  sprintf("%d/%d PASS", summary$tests_pass, summary$total_tests),
  we_path("09_TESTS", "WE_QA_DAYWEIGHT_TEST_RESULTS.txt")
)
if (!all(test_results$status == "PASS")) {
  failed <- test_results$test_name[test_results$status == "FAIL"]
  we_stop_check("STOP — AUTOMATED TEST FAILURE", "PRE_MODEL_TEST_check", paste(failed, collapse = ";"))
}
we_log(sprintf("Automated tests PASS %d/%d", summary$tests_pass, summary$total_tests))
