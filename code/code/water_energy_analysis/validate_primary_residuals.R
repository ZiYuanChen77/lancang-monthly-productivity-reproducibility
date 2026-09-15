source(file.path(Sys.getenv("WATER_ENERGY_CODE_DIR", unset = getwd()), "primary_common.R"))

diagnostic_path <- we_path(
  "03_MODEL1_PRIMARY",
  "WE_MODEL1_PRIMARY_GAUSSIAN_RESIDUAL_DIAGNOSTICS.csv"
)
state_path <- we_path("14_LOGS", "WE_STAGE1_STATE.rds")
we_assert(file.exists(diagnostic_path), "Gaussian residual diagnostics are missing")
we_assert(file.exists(state_path), "Stage 1 state is missing")

d <- read.csv(diagnostic_path, stringsAsFactors = FALSE)
we_assert(d[["nonfinite_residual_count"]][1L] == 0L, "Nonfinite residual detected")
we_assert(d[["nonfinite_fitted_count"]][1L] == 0L, "Nonfinite fitted value detected")

check <- data.frame(
  model = "Model1 PRIMARY",
  structural_finite_check = "PASS",
  residual_validation_status = "PASS_FINITE_VALUES",
  residual_skewness = d[["residual_skewness"]][1L],
  residual_excess_kurtosis = d[["residual_excess_kurtosis"]][1L],
  largest_squared_residual_share = d[["largest_squared_residual_share"]][1L],
  stringsAsFactors = FALSE
)
we_write_csv(
  check,
  we_path("03_MODEL1_PRIMARY", "WE_MODEL1_GAUSSIAN_RESIDUAL_check.csv")
)

state <- readRDS(state_path)
state[["gaussian_finite_check"]] <- "PASS_FINITE_VALUES"
state[["gaussian_residual_validation_at"]] <- format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z")
saveRDS(state, state_path)
we_log("Gaussian residual and fitted-value finiteness check: PASS")
