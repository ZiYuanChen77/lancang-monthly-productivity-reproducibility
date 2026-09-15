source(file.path(Sys.getenv("WATER_ENERGY_CODE_DIR", unset = getwd()), "domain_common.R"))

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
full_flag <- we_explicit_domain(master, "FULL_YEAR")
apr_flag <- we_explicit_domain(master, "APR_SEP")
full <- we_prepare_domain(master, "FULL_YEAR")
apr <- we_prepare_domain(master, "APR_SEP")
active <- we_flag_true(master$historically_active_canonical)

run_test("Full-year no active filter", {
  sum(full_flag & !active) > 0L && !"historically_active_canonical" %in% all.vars(we_formula_domain())
})
run_test("Full-year all 12 months", identical(sort(unique(full$Month)), 1:12))
run_test("Full-year explicit complete-case", {
  identical(full_flag, master$Year %in% 2022:2025 & master$Month %in% 1:12 & we_complete_case(master))
})
run_test("Full-year max-key validation", {
  nrow(full) == 11855L && nrow(full) <= 247L * 48L &&
    length(unique(we_key(full))) == nrow(full)
})
run_test("Apr-Sep no active filter", sum(apr_flag & !active) == 4L)
run_test("Apr-Sep months exactly 4 to 9", identical(sort(unique(apr$Month)), 4:9))
run_test("Apr-Sep explicit complete-case", {
  identical(apr_flag, master$Year %in% 2022:2025 & master$Month %in% 4:9 & we_complete_case(master))
})
run_test("Calendar-month selector", {
  identical(WE_DOMAIN_META$FULL_YEAR$months, 1:12) &&
    identical(WE_DOMAIN_META$APR_SEP$months, 4:9)
})

ordered_master_values <- function(flag, variable) {
  idx <- which(flag)
  idx <- idx[order(sprintf("%03d", as.integer(master$point_id[idx])), master$calendar_month_index[idx])]
  master[[variable]][idx]
}
run_test("Canonical response identity", {
  identical(as.numeric(full$npp_anomaly_mean_g), as.numeric(ordered_master_values(full_flag, "npp_anomaly_mean_g"))) &&
    identical(as.numeric(apr$npp_anomaly_mean_g), as.numeric(ordered_master_values(apr_flag, "npp_anomaly_mean_g")))
})
run_test("Reference rank identity", all(vapply(c("precip_rank_baseline", "radiation_rank_baseline"), function(v) {
  identical(as.numeric(full[[v]]), as.numeric(ordered_master_values(full_flag, v))) &&
    identical(as.numeric(apr[[v]]), as.numeric(ordered_master_values(apr_flag, v)))
}, logical(1))))
run_test("Antecedent and lag-field identity", all(vapply(WE_MODEL1_CONTROLS, function(v) {
  identical(as.numeric(full[[v]]), as.numeric(ordered_master_values(full_flag, v))) &&
    identical(as.numeric(apr[[v]]), as.numeric(ordered_master_values(apr_flag, v)))
}, logical(1))))
run_test("Fixed canonical Primary rho", identical(WE_RHO_FIXED, 0.036623315093487))

run_test("Full-year AR start at new site", {
  x <- data.frame(point_id = c("001", "002"), calendar_month_index = c(100L, 101L))
  identical(we_build_ar_start(x), c(TRUE, TRUE))
})
run_test("Full-year AR start at month gap", {
  x <- data.frame(point_id = c("001", "001", "001"), calendar_month_index = c(100L, 101L, 103L))
  identical(we_build_ar_start(x), c(TRUE, FALSE, TRUE))
})
run_test("Apr-Sep September to April AR gap", {
  x <- data.frame(point_id = c("001", "001"), calendar_month_index = c(2022L * 12L + 9L, 2023L * 12L + 4L))
  identical(we_build_ar_start(x), c(TRUE, TRUE))
})

form_text <- gsub("[[:space:]]+", " ", paste(deparse(we_formula_domain()), collapse = " "))
run_test("Fixed 5 by 5 surface", grepl(
  "te\\(precip_rank_baseline, radiation_rank_baseline, k = c\\(5, 5\\)\\)", form_text
))
run_test("Fixed control k5", all(vapply(WE_MODEL1_CONTROLS, function(v) {
  grepl(sprintf("s\\(%s, k = 5\\)", v), form_text)
}, logical(1))))
run_test("Fixed basis dimensions", !exists("we_apply_domain_fallback", mode = "function"))
run_test("Fixed contrast", {
  identical(unname(WE_CONTRAST_A), c(0.75, 0.25)) &&
    identical(unname(WE_CONTRAST_B), c(0.75, 0.50))
})

primary_model_path <- file.path(WE_PRIMARY_ROOT, "03_MODEL1_PRIMARY", "WE_MODEL1_PRIMARY.rds")
primary_mtime_before <- file.info(primary_model_path)$mtime
primary_model <- readRDS(primary_model_path)
primary <- we_primary_data(master)
primary_ref <- we_reference_row(primary)
primary_lp <- we_lpmatrix_contrast(primary_model, primary_ref)
primary_inv <- we_contrast_invariance(primary_model, primary)
primary_mtime_after <- file.info(primary_model_path)$mtime
run_test("Lpmatrix extraction", {
  isTRUE(all.equal(primary_lp$estimate, as.numeric(primary_lp$dX %*% coef(primary_model)), tolerance = 1e-14))
})
run_test("Contrast invariance", primary_inv$pass && primary_inv$max_absolute_difference <= 1e-8)

run_test("Euclidean radius 0.10", {
  toy <- data.frame(
    point_id = factor(c("001", "002", "003")), Year = rep(2022L, 3),
    precip_rank_baseline = c(0.75, 0.75, 0.75),
    radiation_rank_baseline = c(0.25, 0.34, 0.36)
  )
  x <- we_support_validation(toy, "TOY")
  x$n_records[x$contrast_point == "A"] == 2L
})
full_support <- we_support_validation(full, "FULL_YEAR")
apr_support <- we_support_validation(apr, "APR_SEP")
run_test("Support records check", all(c(full_support$n_records, apr_support$n_records) >= 100L))
run_test("Support sites check", all(c(full_support$n_sites, apr_support$n_sites) >= 50L))
run_test("Support years check", all(c(full_support$n_years, apr_support$n_years) >= 3L))
run_test("Bootstrap B1000", identical(WE_BOOT_TARGET_SUCCESS, 1000L))
run_test("Bootstrap max1050", identical(WE_BOOT_DRAWS, 1050L))
run_test("Duplicate site boot IDs", {
  ids <- we_boot_series_ids(c("029", "029", "030", "029"))
  identical(ids, c("029_copy01", "029_copy02", "030_copy01", "029_copy03"))
})
run_test("Independent AR boot series", {
  toy <- full[as.character(full$point_id) == levels(full$point_id)[1L],
              c(WE_REQUIRED_FIELDS, "AR.start"), drop = FALSE]
  draw <- data.frame(
    attempt = 1L, draw_id = "draw_0001", draw_position = 1:2,
    original_point_id = rep(as.character(toy$point_id[1L]), 2),
    boot_series_id = c("x_copy01", "x_copy02"), stringsAsFactors = FALSE
  )
  b <- we_build_boot_data(toy, draw)
  all(tapply(b$AR.start, b$boot_series_id, function(v) isTRUE(v[1L]))) &&
    length(levels(b$boot_series_id)) == 2L
})
run_test("Fixed rho in bootstrap", {
  identical(formals(we_run_domain_boot_attempt)$rho_used, quote(WE_RHO_FIXED))
})
run_test("Domain sensitivity selector", !"QA_COMBINATION" %in% WE_ALLOWED_ANALYSES)
run_test("Domain response selector", !"DAYWEIGHTED_COMBINATION" %in% WE_ALLOWED_ANALYSES)
run_test("Model selector", !any(c("MODEL0", "MODEL2", "MODEL3") %in% WE_ALLOWED_ANALYSES))
run_test("Analysis-stage selector", !"LOYO" %in% WE_ALLOWED_ANALYSES)
run_test("Sensitivity analysis selector", identical(WE_ALLOWED_ANALYSES, c("FULL_YEAR", "APR_SEP")))
run_test("Primary not refitted", {
  identical(primary_mtime_before, primary_mtime_after) &&
    isTRUE(all.equal(primary_lp$estimate, -2.94258224191719, tolerance = 1e-12))
})

target_core_path <- file.path(
  WE_TARGET_ROBUSTNESS_ROOT, "08_COMPARISON", "WE_QA_DAYWEIGHT_CORE_COMPARISON.csv"
)
target_mtime_before <- file.info(target_core_path)$mtime
reference_refs <- we_reference_reference_validation()
target_mtime_after <- file.info(target_core_path)$mtime
run_test("QA and Day-weighted references not refitted", {
  identical(target_mtime_before, target_mtime_after) && all(reference_refs$status == "PASS")
})
run_test("Full-year domain flag semantics explained", {
  x <- we_domain_flag_validation(master, "FULL_YEAR")$summary
  x$flag_only == 1L && x$flag_only_explained_by_complete_case == 1L &&
    x$unexplained_flag_only == 0L && !x$substantive_semantic_conflict
})
run_test("Apr-Sep domain flag exact", {
  x <- we_domain_flag_validation(master, "APR_SEP")$summary
  x$mismatch_keys == 0L && x$explicit_vs_eligibility_mismatch == 0L &&
    !x$substantive_semantic_conflict
})
run_test("Explicit model domains ignore eligible_primary_model1", {
  sum(full_flag & !we_flag_true(master$eligible_primary_model1)) > 0L &&
    sum(apr_flag & !we_flag_true(master$eligible_primary_model1)) > 0L
})
run_test("Canonical response is formula response", all.vars(we_formula_domain())[1L] == "npp_anomaly_mean_g")
run_test("Formula interaction specification", !any(grepl(":", attr(terms(we_formula_domain()), "term.labels"), fixed = TRUE)))
run_test("Full-year theoretical maximum documented", 247L * 48L == 11856L && nrow(full) == 11856L - 1L)
run_test("Apr-Sep theoretical maximum achieved", 247L * 4L * 6L == 5928L && nrow(apr) == 5928L)
run_test("All applicable primary analysis tests remain PASS", {
  x <- read.csv(file.path(WE_PRIMARY_ROOT, "13_TESTS", "WE_GAMM_PRIMARY_TEST_RESULTS.csv"),
                stringsAsFactors = FALSE)
  nrow(x) > 0L && all(x$status == "PASS")
})
run_test("All target-robustness tests remain PASS", {
  x <- read.csv(file.path(WE_TARGET_ROBUSTNESS_ROOT, "09_TESTS", "WE_QA_DAYWEIGHT_ALL_TEST_RESULTS.csv"),
                stringsAsFactors = FALSE)
  nrow(x) > 0L && all(x$status == "PASS")
})
run_test("Environment has reference mgcv", {
  requireNamespace("mgcv", quietly = TRUE) && as.character(packageVersion("mgcv")) == "1.9.4"
})

test_results <- do.call(rbind, results)
we_write_csv(test_results, we_path("08_TESTS", "WE_DOMAIN_SENSITIVITY_TEST_RESULTS.csv"))
summary <- data.frame(
  tests_pass = sum(test_results$status == "PASS"), total_tests = nrow(test_results),
  tests_fail = sum(test_results$status == "FAIL"),
  status = if (all(test_results$status == "PASS")) "PASS" else "FAIL",
  stringsAsFactors = FALSE
)
we_write_csv(summary, we_path("08_TESTS", "WE_DOMAIN_SENSITIVITY_TEST_SUMMARY.csv"))
we_write_lines(sprintf("%d/%d PASS", summary$tests_pass, summary$total_tests),
                 we_path("08_TESTS", "WE_DOMAIN_SENSITIVITY_TEST_RESULTS.txt"))
if (!all(test_results$status == "PASS")) {
  we_stop_check(
    "STOP — AUTOMATED TEST FAILURE", "PRE_MODEL_TEST_check",
    paste(test_results$test_name[test_results$status == "FAIL"], collapse = ";")
  )
}
we_log(sprintf("Automated tests PASS %d/%d", summary$tests_pass, summary$total_tests))
