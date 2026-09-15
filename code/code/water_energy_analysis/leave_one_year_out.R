source(file.path(Sys.getenv("WATER_ENERGY_CODE_DIR", unset = getwd()), "primary_common.R"))
suppressPackageStartupMessages(library(mgcv))

state_path <- we_path("14_LOGS", "WE_STAGE1_STATE.rds")
if (!file.exists(state_path)) {
  we_stop_check("STOP — PRIMARY STATE NOT AVAILABLE", "LOYO_ENTRY_check")
}
state <- readRDS(state_path)
if (!isTRUE(state[["bootstrap_complete"]])) {
  we_stop_check("STOP — PRIMARY BOOTSTRAP NOT COMPLETE", "LOYO_ENTRY_check")
}

master <- we_read_master()
primary <- we_primary_data(master)
rho_used <- state[["rho_used"]]
model1_formula <- we_formula_model1(state[["model1_k_final"]])
full_contrast <- state[["model1_contrast"]]
years <- c(2022L, 2023L, 2024L, 2025L)
loyo_rows <- list()

for (i in seq_along(years)) {
  omitted <- years[i]
  d <- primary[primary[["Year"]] != omitted, , drop = FALSE]
  d <- d[order(as.character(d[["point_id"]]), d[["calendar_month_index"]]), , drop = FALSE]
  rownames(d) <- NULL
  d[["point_id"]] <- droplevels(d[["point_id"]])
  d[["AR.start"]] <- we_build_ar_start(d, "point_id")
  ar <- we_ar_sequence_validation(d)
  we_write_csv(
    ar[["summary"]],
    we_path("11_LOYO", sprintf("WE_LOYO_DROP_%d_AR_SEQUENCE_validation.csv", omitted))
  )
  we_write_csv(
    ar[["gaps"]],
    we_path("11_LOYO", sprintf("WE_LOYO_DROP_%d_AR_GAPS.csv", omitted))
  )
  we_log("Fitting LOYO drop", omitted, "with fixed full-Primary rho", format(rho_used, digits = 12))
  fit <- tryCatch(
    we_fit_bam(model1_formula, d, rho_used, d[["AR.start"]]),
    error = function(e) e
  )
  if (inherits(fit, "error")) {
    we_stop_check(
      "STOP — LOYO REQUIRES analysis check",
      paste0("LOYO_DROP_", omitted),
      conditionMessage(fit)
    )
  }
  health <- we_model_health(fit[["model"]], fit[["warnings"]])
  reasons <- we_model_health_stop_reason(health)
  if (length(reasons)) {
    we_stop_check(
      "STOP — LOYO REQUIRES analysis check",
      paste0("LOYO_DROP_", omitted),
      paste(reasons, collapse = ";")
    )
  }
  formula_unchanged <- identical(
    paste(deparse(formula(fit[["model"]]), width.cutoff = 500L), collapse = " "),
    paste(deparse(model1_formula, width.cutoff = 500L), collapse = " ")
  )
  if (!formula_unchanged) {
    we_stop_check(
      "STOP — LOYO REQUIRES FORMULA/K/CONTRAST CHANGE",
      paste0("LOYO_DROP_", omitted)
    )
  }
  contrast <- we_contrast_invariance(fit[["model"]], d)
  if (!contrast[["pass"]] || !is.finite(contrast[["main_contrast"]])) {
    we_stop_check(
      "STOP — CONTRAST EXTRACTION NOT INVARIANT",
      paste0("LOYO_DROP_", omitted),
      sprintf("max_difference=%.12g", contrast[["max_absolute_difference"]])
    )
  }
  saveRDS(
    fit[["model"]],
    we_path("11_LOYO", sprintf("WE_MODEL1_LOYO_DROP_%d.rds", omitted))
  )
  we_write_lines(
    paste(deparse(formula(fit[["model"]]), width.cutoff = 500L), collapse = " "),
    we_path("11_LOYO", sprintf("WE_MODEL1_LOYO_DROP_%d_FORMULA.txt", omitted))
  )
  we_write_csv(
    we_smooth_edf(fit[["model"]]),
    we_path("11_LOYO", sprintf("WE_MODEL1_LOYO_DROP_%d_SMOOTH_EDF.csv", omitted))
  )
  we_write_csv(
    we_residual_diagnostics(fit[["model"]]),
    we_path("11_LOYO", sprintf("WE_MODEL1_LOYO_DROP_%d_RESIDUAL_DIAGNOSTICS.csv", omitted))
  )
  invariance_validation <- contrast[["validation"]]
  invariance_validation[["omitted_year"]] <- omitted
  we_write_csv(
    invariance_validation,
    we_path("11_LOYO", sprintf("WE_MODEL1_LOYO_DROP_%d_CONTRAST_INVARIANCE.csv", omitted))
  )
  value <- contrast[["main_contrast"]]
  loyo_rows[[i]] <- data.frame(
    omitted_year = omitted,
    n = nrow(d),
    n_sites = length(levels(d[["point_id"]])),
    contrast = value,
    sign = we_direction(value),
    absolute_difference_from_full = abs(value - full_contrast),
    relative_difference_from_full = abs(value - full_contrast) / abs(full_contrast),
    convergence = health[["convergence"]][1L],
    diagnostic_flags = "PASS",
    rho_used = rho_used,
    rho_reestimated = FALSE,
    formula_changed = FALSE,
    k_changed = FALSE,
    contrast_changed = FALSE,
    n_AR_sections = ar[["summary"]][["n_AR_sections"]],
    stringsAsFactors = FALSE
  )
  we_log(sprintf("LOYO drop %d complete contrast=%.12g", omitted, value))
}

loyo <- do.call(rbind, loyo_rows)
we_write_csv(loyo, we_path("11_LOYO", "WE_MODEL1_LOYO_CONTRAST.csv"))
