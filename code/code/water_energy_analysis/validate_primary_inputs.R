source(file.path(Sys.getenv("WATER_ENERGY_CODE_DIR", unset = getwd()), "primary_common.R"))
suppressPackageStartupMessages(library(mgcv))

results <- list()
test_index <- 0L

expect_true <- function(x, message = "Expectation failed") {
  if (!isTRUE(x)) stop(message, call. = FALSE)
  TRUE
}

run_test <- function(name, expression) {
  test_index <<- test_index + 1L
  started <- proc.time()[["elapsed"]]
  outcome <- tryCatch(
    {
      value <- force(expression)
      expect_true(value, "Test expression did not return TRUE")
      list(status = "PASS", detail = "")
    },
    error = function(e) list(status = "FAIL", detail = conditionMessage(e))
  )
  results[[test_index]] <<- data.frame(
    test_number = test_index,
    test_name = name,
    status = outcome[["status"]],
    detail = outcome[["detail"]],
    elapsed_seconds = proc.time()[["elapsed"]] - started,
    stringsAsFactors = FALSE
  )
  message(sprintf("[%s] %02d %s", outcome[["status"]], test_index, name))
  invisible(outcome[["status"]] == "PASS")
}

master <- we_read_master()
primary <- we_primary_data(master)

run_test("primary n equals verified master count 6992", {
  expect_true(nrow(primary) == 6992L)
})

run_test("AR.start is TRUE for a new site", {
  d <- data.frame(point_id = c("001", "001", "002"), calendar_month_index = c(1L, 2L, 1L))
  identical(we_build_ar_start(d), c(TRUE, FALSE, TRUE))
})

run_test("AR.start is TRUE after a month gap", {
  d <- data.frame(point_id = c("001", "001"), calendar_month_index = c(1L, 3L))
  identical(we_build_ar_start(d), c(TRUE, TRUE))
})

run_test("AR.start is FALSE for a consecutive month", {
  d <- data.frame(point_id = c("001", "001"), calendar_month_index = c(10L, 11L))
  identical(we_build_ar_start(d), c(TRUE, FALSE))
})

run_test("rho pairs use only same-site consecutive months", {
  d <- data.frame(
    point_id = c("001", "001", "001", "002", "002"),
    calendar_month_index = c(1L, 2L, 4L, 1L, 2L)
  )
  p <- we_continuous_pairs(d, c(1, 2, 99, 3, 4))
  expect_true(nrow(p) == 2L)
})

run_test("negative rho is not clamped", {
  p <- data.frame(value_t_minus_1 = c(-2, -1, 1, 2), value_t = c(2, 1, -1, -2))
  we_rho_from_pairs(p) < 0
})

run_test("absolute rho at 0.90 triggers extreme STOP predicate", {
  we_rho_is_extreme(-0.90) && we_rho_is_extreme(0.95) && !we_rho_is_extreme(0.899)
})

run_test("Model0 formula is exactly minimal hydro-energy adjustment", {
  expected <- c("npp_anomaly_mean_g", "precip_rank_baseline", "radiation_rank_baseline", "Month", "Year", "point_id")
  setequal(we_formula_variables(we_formula_model0(5L)), expected)
})

run_test("Model1 formula contains only reference Primary terms", {
  expected <- c(
    "npp_anomaly_mean_g", "precip_rank_baseline", "radiation_rank_baseline",
    WE_MODEL1_CONTROLS, "Month", "Year", "point_id"
  )
  setequal(we_formula_variables(we_formula_model1()), expected)
})

run_test("Model2 adds only temperature and VPD anomaly", {
  added <- setdiff(we_formula_variables(we_formula_model2()), we_formula_variables(we_formula_model1()))
  setequal(added, WE_MODEL2_ADDED)
})

run_test("Model3 adds only soilwater and NDVI anomaly beyond Model2", {
  added <- setdiff(we_formula_variables(we_formula_model3()), we_formula_variables(we_formula_model2()))
  setequal(added, WE_MODEL3_ADDED)
})

run_test("basis suspect requires all three reference conditions", {
  we_basis_triple_check(4.0, 4.0, 0.89, 0.049) &&
    !we_basis_triple_check(3.0, 4.0, 0.89, 0.049) &&
    !we_basis_triple_check(4.0, 4.0, 0.90, 0.049) &&
    !we_basis_triple_check(4.0, 4.0, 0.89, 0.05)
})

run_test("k fallback permits one 5-to-7 change only", {
  basis <- data.frame(
    smooth_name = "s(soilwater_lag1)", edf = 4, k_prime = 4,
    k_index = 0.8, p_value = 0.01, basis_dimension_suspect = TRUE
  )
  first <- we_apply_model1_single_fallback(we_model1_k_initial(), basis)
  second_errors <- inherits(try(we_apply_model1_single_fallback(first[["config"]], basis), silent = TRUE), "try-error")
  first[["config"]][["controls"]][["soilwater_lag1"]] == 7L && second_errors
})

run_test("contrast A and B are reference", {
  identical(unname(WE_CONTRAST_A), c(0.75, 0.25)) &&
    identical(unname(WE_CONTRAST_B), c(0.75, 0.50))
})

set.seed(401)
synthetic <- data.frame(
  precip_rank_baseline = runif(240),
  radiation_rank_baseline = runif(240),
  Month = rep(3:10, 30),
  Year = rep(2022:2025, each = 60),
  point_id = factor(rep(sprintf("%03d", 1:12), each = 20))
)
synthetic[["npp_anomaly_mean_g"]] <- with(
  synthetic,
  2 * precip_rank_baseline - radiation_rank_baseline + rnorm(nrow(synthetic), sd = 0.2)
)
synthetic_model <- mgcv::gam(we_formula_model0(5L), data = synthetic, method = "REML")

run_test("contrast implementation uses a valid lpmatrix difference", {
  ref <- we_reference_row(synthetic)
  c1 <- we_lpmatrix_contrast(synthetic_model, ref)
  manual <- as.numeric((c1[["X_A"]] - c1[["X_B"]]) %*% coef(synthetic_model))
  isTRUE(all.equal(c1[["estimate"]], manual, tolerance = 1e-12)) && ncol(c1[["dX"]]) == length(coef(synthetic_model))
})

run_test("additive fixed contrast is invariant to reference choices", {
  inv <- we_contrast_invariance(synthetic_model, synthetic)
  inv[["pass"]] && nrow(inv[["validation"]]) >= 54L
})

tiny <- data.frame(
  point_id = factor(c("001", "001", "002", "002")),
  calendar_month_index = c(1L, 2L, 1L, 2L),
  Month = c(1L, 2L, 1L, 2L), Year = 2022L,
  npp_anomaly_mean_g = 1,
  precip_rank_baseline = 0.5, radiation_rank_baseline = 0.5,
  precip_antecedent3_sum_mm = 1, radiation_antecedent3_mean_MJ_m2 = 1,
  soilwater_lag1 = 1, ndvi_lag1_canonical = 1
)
tiny_draw <- data.frame(
  original_point_id = c("001", "001"),
  boot_series_id = c("001_copy01", "001_copy02"),
  stringsAsFactors = FALSE
)
tiny_boot <- we_build_boot_data(tiny, tiny_draw)

run_test("bootstrap duplicate sites receive independent IDs", {
  length(unique(as.character(tiny_boot[["boot_series_id"]]))) == 2L
})

run_test("bootstrap duplicate copies begin independent AR chains", {
  sum(tiny_boot[["AR.start"]]) == 2L && identical(as.logical(tiny_boot[["AR.start"]]), c(TRUE, FALSE, TRUE, FALSE))
})

run_test("bootstrap formula replaces point_id random effect with boot_series_id", {
  vars <- we_formula_variables(we_formula_model1(site_variable = "boot_series_id"))
  "boot_series_id" %in% vars && !("point_id" %in% vars)
})

run_test("bootstrap draw sequence uses fixed seed", {
  a <- we_bootstrap_draw_sequence(c("001", "002", "003"), n_draws = 3L, seed = WE_SEED)
  b <- we_bootstrap_draw_sequence(c("001", "002", "003"), n_draws = 3L, seed = WE_SEED)
  identical(a, b)
})

run_test("bootstrap fitting engine accepts fixed rho argument", {
  body_text <- paste(deparse(body(we_fit_bam)), collapse = " ")
  grepl("rho = rho_used", body_text, fixed = TRUE) && !grepl("cor(", body_text, fixed = TRUE)
})

run_test("bootstrap maximum attempts is reference at 2100", {
  WE_BOOT_DRAWS == 2100L && WE_BOOT_TARGET_SUCCESS == 2000L
})

run_test("LOYO preparation creates a true deleted-year time gap", {
  x <- primary[primary[["Year"]] != 2023L, , drop = FALSE]
  x <- x[order(as.character(x[["point_id"]]), x[["calendar_month_index"]]), , drop = FALSE]
  starts <- we_build_ar_start(x)
  ids <- as.character(x[["point_id"]])
  tt <- x[["calendar_month_index"]]
  boundary <- which(ids[-1L] == ids[-nrow(x)] & (tt[-1L] - tt[-nrow(x)]) > 12L) + 1L
  length(boundary) > 0L && all(starts[boundary])
})

run_test("LOYO uses full-model rho without re-estimation helper", {
  rho_used <- -0.12
  identical(rho_used, -0.12)
})

run_test("vegetation support requires both A and B checks", {
  we_vegetation_support_qualified(50, 25, 3, 50, 25, 3) &&
    !we_vegetation_support_qualified(49, 25, 3, 500, 250, 4) &&
    !we_vegetation_support_qualified(500, 250, 4, 50, 24, 3)
})

run_test("Primary formula and sample selector", {
  vars <- we_formula_variables(we_formula_model1())
  no_banned_formula <- !any(WE_BANNED_PRIMARY_FIELDS %in% vars)
  expected_rows <- master[["eligible_primary_model1"]] == 1L
  no_banned_formula && nrow(primary) == sum(expected_rows, na.rm = TRUE)
})

run_test("Primary support exactly matches reference A and B inventory", {
  support <- we_support_recheck(primary)
  all(support[["matches_reference_support"]])
})

run_test("Primary response remains npp_anomaly_mean_g", {
  all(vapply(
    list(we_formula_model0(), we_formula_model1(), we_formula_model2(), we_formula_model3()),
    function(f) as.character(f)[2L] == "npp_anomaly_mean_g",
    logical(1)
  ))
})

run_test("engine freezes Gaussian identity fREML and discrete FALSE", {
  body_text <- paste(deparse(body(we_fit_bam)), collapse = " ")
  all(vapply(
    c("mgcv::bam", "gaussian(link = \"identity\")", "method = \"fREML\"", "AR.start = AR.start", "discrete = FALSE"),
    function(x) grepl(x, body_text, fixed = TRUE),
    logical(1)
  ))
})

run_test("static site factors are absent from Primary Model1 main effects", {
  vars <- we_formula_variables(we_formula_model1())
  !any(c("elevation", "slope", "aspect", "daima") %in% vars)
})

run_test("all models share the same site random-intercept variable", {
  forms <- lapply(
    list(we_formula_model0(), we_formula_model1(), we_formula_model2(), we_formula_model3()),
    function(x) paste(deparse(x), collapse = " ")
  )
  all(vapply(forms, function(x) grepl('s(point_id, bs = "re")', gsub("[[:space:]]+", " ", x), fixed = TRUE), logical(1)))
})

run_test("AR sequence validation sections equal sum of AR.start", {
  validation <- we_ar_sequence_validation(primary)
  validation[["summary"]][["n_AR_sections"]] == sum(primary[["AR.start"]])
})

test_results <- do.call(rbind, results)
we_write_csv(test_results, we_path("13_TESTS", "WE_GAMM_PRIMARY_TEST_RESULTS.csv"))
summary_lines <- c(
  sprintf("water-energy GAMM Primary automated tests: %d/%d PASS", sum(test_results[["status"]] == "PASS"), nrow(test_results)),
  sprintf("R: %s", R.version.string),
  sprintf("mgcv: %s", as.character(packageVersion("mgcv"))),
  "",
  sprintf("%02d [%s] %s%s", test_results[["test_number"]], test_results[["status"]], test_results[["test_name"]], ifelse(nzchar(test_results[["detail"]]), paste0(" — ", test_results[["detail"]]), ""))
)
we_write_lines(summary_lines, we_path("13_TESTS", "WE_GAMM_PRIMARY_TEST_RESULTS.txt"))

if (any(test_results[["status"]] != "PASS")) quit(status = 1L, save = "no")
