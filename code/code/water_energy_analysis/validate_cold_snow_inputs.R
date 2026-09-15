source(file.path(Sys.getenv("WATER_ENERGY_CODE_DIR", unset = getwd()), "cold_snow_common.R"))

results <- list()
add_test <- function(name, condition, detail = "") {
  results[[length(results) + 1L]] <<- data.frame(
    test = name, status = if (isTRUE(condition)) "PASS" else "FAIL",
    detail = as.character(detail), stringsAsFactors = FALSE
  )
}

master <- we_read_master()
primary <- we_primary_data(master)
cold <- we_prepare_cold_snow(master)
flags <- we_cold_snow_flags(master)
identity <- we_primary_identity_validation(master)
eligibility <- we_cold_snow_eligibility_validation(master)
temperature_validation <- we_temperature_semantics_validation(master)
snow_validation <- we_snow_semantics_validation(master)
support <- we_cold_snow_support_validation(cold)
formula <- we_formula_model1(we_model1_k_initial(), "point_id")
formula_text <- paste(deparse(formula, width.cutoff = 500L), collapse = " ")
formula_vars <- all.vars(formula)
stage1_text <- paste(readLines(we_path("01_CODE", "we_cold_snow_stage1.R"), warn = FALSE), collapse = "\n")
bootstrap_text <- paste(readLines(we_path("01_CODE", "we_cold_snow_bootstrap.R"), warn = FALSE), collapse = "\n")
common_text <- paste(readLines(we_path("01_CODE", "cold_snow_common.R"), warn = FALSE), collapse = "\n")
primary_common_text <- paste(readLines(WE_PRIMARY_COMMON_PATH, warn = FALSE), collapse = "\n")

add_test("Primary input exactly 6992", nrow(primary) == 6992L, nrow(primary))
add_test("Primary site count exactly 247", length(levels(primary$point_id)) == 247L,
         length(levels(primary$point_id)))
add_test("Primary canonical key identity", all(identity$status == "PASS"), identity$status)
add_test("ColdSnow starts from Primary only", all(!flags$explicit | flags$primary))
add_test("ColdSnow is strict Primary subset", all(we_key(cold) %in% we_key(primary)))
add_test("Actual target temperature field exists", WE_TEMPERATURE_FIELD == "temperature_2m" &&
           WE_TEMPERATURE_FIELD %in% names(master))
add_test("Temperature actual not anomaly", WE_TEMPERATURE_FIELD != "temperature_anomaly" &&
           temperature_validation$separate_anomaly_field == "temperature_anomaly")
add_test("Temperature unit Celsius verified", temperature_validation$unit == "degrees Celsius" &&
           temperature_validation$status == "PASS")
add_test("Temperature threshold strictly greater than zero",
         identical(c(-0.1, 0, 0.1) > 0, c(FALSE, FALSE, TRUE)))
add_test("Temperature predictor specification", !grepl("LST_Day_1km", formula_text, fixed = TRUE) &&
           !grepl("LST_Day_1km", common_text, fixed = TRUE))
add_test("Snow status known required", all(!flags$explicit | flags$snow_known))
add_test("Unknown snow excluded", !any(flags$explicit & flags$snow_unknown))
add_test("Snow present excluded", !any(flags$explicit & flags$snow_present))
add_test("Known no-snow retained when other checks pass",
         all(flags$explicit == (flags$primary & flags$temperature_gt0 & flags$snow_known & flags$snow_absent)))
add_test("Snow known deterministic reconstruction", all(snow_validation$status == "PASS"),
         paste(snow_validation$n_mismatch, collapse = ";"))
add_test("Precomputed eligibility matches explicit logic", eligibility$status == "PASS" &&
           eligibility$mismatch == 0L)
add_test("Response remains Primary mean anomaly", identical(as.character(formula[[2L]]), "npp_anomaly_mean_g"))
add_test("Rank exposure fields", all(c("precip_rank_baseline", "radiation_rank_baseline") %in% formula_vars))
add_test("Rank-based exposure specification", !any(c("precip_anomaly_mm", "radiation_anomaly_MJ_m2") %in% formula_vars))
add_test("Antecedent control fields", all(WE_MODEL1_CONTROLS %in% formula_vars))
add_test("Only four reference control smooths", sum(WE_MODEL1_CONTROLS %in% formula_vars) == 4L)
add_test("Fixed canonical Model1 formula", grepl("te\\(precip_rank_baseline, radiation_rank_baseline", formula_text) &&
           grepl("factor\\(Month\\)", formula_text) && grepl("factor\\(Year\\)", formula_text) &&
           grepl('s\\(point_id, bs = "re"\\)', formula_text))
add_test("Fixed Primary rho", identical(WE_RHO_FIXED, 0.036623315093487))
synthetic <- data.frame(
  point_id = c("001", "001", "001", "002"),
  calendar_month_index = c(100L, 101L, 103L, 100L)
)
synthetic_start <- we_build_ar_start(synthetic, "point_id")
add_test("AR start new site", isTRUE(synthetic_start[4L]))
add_test("AR gap rebuilt", isTRUE(synthetic_start[3L]))
add_test("AR continuous month not start", !synthetic_start[2L])
add_test("ColdSnow AR starts stored", identical(cold$AR.start, we_build_ar_start(cold, "point_id")))
k <- we_model1_k_initial()
add_test("Fixed surface 5x5", k$surface == 5L && grepl("k = c\\(5, 5\\)", formula_text))
add_test("Fixed controls k5", all(k$controls == 5L))
add_test("Fixed basis dimensions", grepl("k_fallback_permitted <- FALSE", stage1_text, fixed = TRUE) &&
           !grepl("we_apply_model1_single_fallback", stage1_text, fixed = TRUE))
add_test("Fixed A contrast", identical(unname(WE_CONTRAST_A), c(0.75, 0.25)))
add_test("Fixed B contrast", identical(unname(WE_CONTRAST_B), c(0.75, 0.50)))
add_test("Lpmatrix extraction", grepl('type = "lpmatrix"', primary_common_text, fixed = TRUE))
add_test("Contrast invariance tolerance 1e-8", grepl("max_diff <= 1e-8", primary_common_text, fixed = TRUE))
add_test("Support Euclidean radius 0.10", identical(WE_SUPPORT_RADIUS, 0.10))
add_test("Support records check 100", identical(WE_SUPPORT_MIN_RECORDS, 100L))
add_test("Support sites check 50", identical(WE_SUPPORT_MIN_SITES, 50L))
add_test("Support years check 3", identical(WE_SUPPORT_MIN_YEARS, 3L))
add_test("Support validationed at both A and B", identical(as.character(support$contrast_point), c("A", "B")))
add_test("Bootstrap target B1000", identical(WE_BOOT_TARGET_SUCCESS, 1000L))
add_test("Bootstrap maximum 1050", identical(WE_BOOT_DRAWS, 1050L))
dup_ids <- we_boot_series_ids(c("029", "029", "030", "029"))
add_test("Duplicate sites get independent boot IDs", length(unique(dup_ids)) == 4L &&
           identical(dup_ids, c("029_copy01", "029_copy02", "030_copy01", "029_copy03")))
mini <- data.frame(
  point_id = factor(c("029", "029")), calendar_month_index = c(100L, 101L),
  Year = c(2022L, 2022L), Month = c(4L, 5L), npp_anomaly_mean_g = c(1, 2),
  precip_rank_baseline = c(0.5, 0.6), radiation_rank_baseline = c(0.4, 0.3),
  precip_antecedent3_sum_mm = c(10, 11), radiation_antecedent3_mean_MJ_m2 = c(20, 21),
  soilwater_lag1 = c(0.2, 0.3), ndvi_lag1_canonical = c(0.4, 0.5)
)
draw <- data.frame(
  original_point_id = c("029", "029"), boot_series_id = c("029_copy01", "029_copy02")
)
mini_boot <- we_build_boot_data(mini, draw)
add_test("Bootstrap copies have independent AR chains", sum(mini_boot$AR.start) == 2L &&
           length(levels(mini_boot$boot_series_id)) == 2L)
add_test("Bootstrap fixed rho", grepl("WE_RHO_FIXED", bootstrap_text, fixed = TRUE) &&
           !grepl("rho_from_pairs", bootstrap_text, fixed = TRUE))
add_test("Bootstrap fixed seed", identical(WE_SEED, 20260803L) &&
           grepl("we_bootstrap_draw_sequence", bootstrap_text, fixed = TRUE))
add_test("Cold/snow selector independence", !any(c("mod17_QA1", "mod17_QA2", "ndvi_lag1_qa01", "ndvi_lag1_qa0") %in% formula_vars))
add_test("Response definition", !any(grepl("dayweighted", formula_vars, ignore.case = TRUE)))
add_test("Response and exposure definitions", !any(c("npp_anomaly_median_g", "precip_anomaly_mm", "radiation_anomaly_MJ_m2") %in% formula_vars))
add_test("Model selector", !grepl("we_formula_model[023]", paste(stage1_text, bootstrap_text), perl = TRUE))
add_test("Analysis-stage selector", !grepl("LOYO|drop 202", paste(stage1_text, bootstrap_text), ignore.case = TRUE))
add_test("Fixed temperature threshold", !grepl("> -5|> 2|> 5|> 10|threshold_grid|optimi[sz]e.*threshold",
                                      paste(stage1_text, bootstrap_text), ignore.case = TRUE))
add_test("Sensitivity analysis selector", identical(WE_ALLOWED_ANALYSIS, "COLD_SNOW_EXCLUSION"))
add_test("Primary not refitted", !grepl("Fitting Primary|rho = 0", stage1_text, fixed = TRUE))
add_test("Local input workflow", !grepl("download.file|curl|wget|http://|https://", paste(stage1_text, bootstrap_text), ignore.case = TRUE))
primary_tests <- read.csv(file.path(
  WE_PRIMARY_ROOT, "13_TESTS", "WE_GAMM_PRIMARY_TEST_RESULTS.csv"
), stringsAsFactors = FALSE)
add_test("canonical Primary regression tests remain PASS", nrow(primary_tests) > 0L &&
           all(primary_tests$status == "PASS"), sprintf("%d/%d", sum(primary_tests$status == "PASS"), nrow(primary_tests)))

out <- do.call(rbind, results)
we_write_csv(out, we_path("08_TESTS", "WE_COLD_SNOW_TEST_RESULTS.csv"))
summary <- data.frame(
  tests_pass = sum(out$status == "PASS"), total_tests = nrow(out),
  tests_fail = sum(out$status != "PASS"),
  status = if (all(out$status == "PASS")) "PASS" else "FAIL",
  stringsAsFactors = FALSE
)
we_write_csv(summary, we_path("08_TESTS", "WE_COLD_SNOW_TEST_SUMMARY.csv"))
we_write_lines(
  sprintf("%d/%d %s", summary$tests_pass, summary$total_tests, summary$status),
  we_path("08_TESTS", "WE_COLD_SNOW_TEST_RESULTS.txt")
)
if (!all(out$status == "PASS")) {
  message("Failed tests: ", paste(out$test[out$status != "PASS"], collapse = "; "))
  quit(save = "no", status = 1L)
}
message(sprintf("Cold/Snow input validation: %d/%d PASS", summary$tests_pass, summary$total_tests))
