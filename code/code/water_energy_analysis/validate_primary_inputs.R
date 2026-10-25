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







run_test("Observed monthly series and consecutive pairs", {
  ids <- as.character(primary$point_id)
  month <- primary$calendar_month_index
  expected <- c(TRUE, ids[-1L] != ids[-length(ids)] | diff(month) > 1L)
  pairs <- we_continuous_pairs(primary, primary$npp_anomaly_mean_g)
  identical(we_build_ar_start(primary), expected) &&
    nrow(pairs) == sum(!expected) &&
    all(pairs$calendar_month_index_t - pairs$calendar_month_index_t_minus_1 == 1L)
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





run_test("contrast A and B are reference", {
  identical(unname(WE_CONTRAST_A), c(0.75, 0.25)) &&
    identical(unname(WE_CONTRAST_B), c(0.75, 0.50))
})

validation_model <- mgcv::gam(we_formula_model0(5L), data = primary, method = "REML")

run_test("contrast implementation uses a valid lpmatrix difference", {
  ref <- we_reference_row(primary)
  c1 <- we_lpmatrix_contrast(validation_model, ref)
  manual <- as.numeric((c1[["X_A"]] - c1[["X_B"]]) %*% coef(validation_model))
  isTRUE(all.equal(c1[["estimate"]], manual, tolerance = 1e-12)) && ncol(c1[["dX"]]) == length(coef(validation_model))
})

run_test("additive fixed contrast is invariant to reference choices", {
  inv <- we_contrast_invariance(validation_model, primary)
  inv[["pass"]] && nrow(inv[["validation"]]) >= 54L
})

validation_draw <- we_bootstrap_draw_sequence(levels(primary$point_id), n_draws = 1L, seed = WE_SEED)
validation_boot <- we_build_boot_data(primary, validation_draw)

run_test("Observed-site bootstrap copies have unique IDs", {
  length(unique(validation_boot$boot_series_id)) == nrow(validation_draw)
})

run_test("Observed-site bootstrap copies restart AR chains", {
  all(tapply(validation_boot$AR.start, validation_boot$boot_series_id, function(x) isTRUE(x[1L])))
})

run_test("bootstrap formula replaces point_id random effect with boot_series_id", {
  vars <- we_formula_variables(we_formula_model1(site_variable = "boot_series_id"))
  "boot_series_id" %in% vars && !("point_id" %in% vars)
})

run_test("Observed-site bootstrap sequence uses fixed seed", {
  identical(validation_draw, we_bootstrap_draw_sequence(levels(primary$point_id), n_draws = 1L, seed = WE_SEED))
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
