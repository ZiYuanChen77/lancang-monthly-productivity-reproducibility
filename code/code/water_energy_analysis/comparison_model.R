# Gaussian GAMM fitting and complete-site bootstrap sampling.
WE_SEED <- 20260803L
WE_BOOT_DRAWS <- 2100L
WE_CONTRAST_A <- c(precip_rank_baseline = 0.75, radiation_rank_baseline = 0.25)
WE_MODEL1_CONTROLS <- c('precip_antecedent3_sum_mm', 'radiation_antecedent3_mean_MJ_m2', 'soilwater_lag1', 'ndvi_lag1_canonical')
WE_ALL_CONTROLS <- WE_MODEL1_CONTROLS

we_assert <- function(condition, message) {
    if (!isTRUE(condition)) 
        stop(message, call. = FALSE)
    invisible(TRUE)
}

we_build_ar_start <- function(d, id_col = "point_id", time_col = "calendar_month_index") {
    n <- nrow(d)
    if (n == 0L) 
        return(logical())
    ids <- as.character(d[[id_col]])
    tt <- as.integer(d[[time_col]])
    if (n > 1L) {
        within_same <- ids[-1L] == ids[-n]
        we_assert(all(diff(tt)[within_same] > 0L), "Time is not strictly increasing within series")
    }
    c(TRUE, ids[-1L] != ids[-n] | (tt[-1L] - tt[-n]) > 1L)
}

we_surface_term <- function(k) {
    sprintf("te(precip_rank_baseline, radiation_rank_baseline, k = c(%d, %d))", k, k)
}

we_smooth_term <- function(variable, k) sprintf("s(%s, k = %d)", variable, k)

we_model1_k_initial <- function() {
    list(surface = 5L, controls = setNames(rep(5L, length(WE_MODEL1_CONTROLS)), WE_MODEL1_CONTROLS))
}

we_formula_from_terms <- function(terms, site_term = "s(point_id, bs = \"re\")") {
    as.formula(paste("npp_anomaly_mean_g ~", paste(c(terms, "factor(Month)", "factor(Year)", site_term), 
        collapse = " + ")))
}

we_formula_model1 <- function(k_config = we_model1_k_initial(), site_variable = "point_id") {
    terms <- c(we_surface_term(k_config[["surface"]]), vapply(WE_MODEL1_CONTROLS, function(v) we_smooth_term(v, 
        k_config[["controls"]][[v]]), character(1)))
    site_term <- sprintf("s(%s, bs = \"re\")", site_variable)
    we_formula_from_terms(terms, site_term)
}

we_fit_bam <- function(formula, data, rho_used, ar_start) {
    we_assert(length(ar_start) == nrow(data), "AR.start length does not match model data")
    data[["AR.start"]] <- as.logical(ar_start)
    warnings <- character()
    model <- withCallingHandlers(mgcv::bam(formula = formula, data = data, family = gaussian(link = "identity"), 
        method = "fREML", rho = rho_used, AR.start = AR.start, discrete = FALSE), warning = function(w) {
        warnings <<- c(warnings, conditionMessage(w))
        invokeRestart("muffleWarning")
    })
    list(model = model, warnings = unique(warnings))
}

we_model_convergence <- function(model) {
    if (!is.null(model[["converged"]])) 
        return(isTRUE(model[["converged"]]))
    if (!is.null(model[["mgcv.conv"]][["fully.converged"]])) {
        return(isTRUE(model[["mgcv.conv"]][["fully.converged"]]))
    }
    if (!is.null(model[["outer.info"]][["conv"]])) {
        return(grepl("converg", as.character(model[["outer.info"]][["conv"]]), ignore.case = TRUE))
    }
    NA
}

we_model_health <- function(model, warnings = character()) {
    cf <- coef(model)
    fv <- fitted(model)
    rr <- residuals(model, type = "response")
    rank_value <- if (is.null(model[["rank"]])) 
        NA_integer_
    else as.integer(model[["rank"]])
    coefficient_count <- length(cf)
    full_rank <- if (is.na(rank_value)) 
        NA
    else rank_value == coefficient_count
    conv <- we_model_convergence(model)
    numerical_warning <- any(grepl("rank deficient|not finite|non-finite|singular|identif|failed to converge|iteration limit|not positive definite", 
        warnings, ignore.case = TRUE))
    data.frame(convergence = if (isTRUE(conv)) 
        "CONVERGED"
    else if (identical(conv, FALSE)) 
        "NOT_CONVERGED"
    else "NOT_EXPLICITLY_REPORTED", coefficients_finite = all(is.finite(cf)), fitted_finite = all(is.finite(fv)), 
        residuals_finite = all(is.finite(rr)), coefficient_count = coefficient_count, model_rank = rank_value, 
        full_rank = full_rank, numerical_identifiability_warning = numerical_warning, warning_count = length(warnings), 
        warnings = paste(warnings, collapse = " | "), stringsAsFactors = FALSE)
}

we_model_health_stop_reason <- function(health) {
    reasons <- character()
    if (health[["convergence"]][1] == "NOT_CONVERGED") 
        reasons <- c(reasons, "MODEL NOT CONVERGED")
    if (!isTRUE(health[["coefficients_finite"]][1])) 
        reasons <- c(reasons, "NONFINITE COEFFICIENT")
    if (!isTRUE(health[["fitted_finite"]][1])) 
        reasons <- c(reasons, "NONFINITE FITTED VALUE")
    if (!isTRUE(health[["residuals_finite"]][1])) 
        reasons <- c(reasons, "NONFINITE RESIDUAL")
    if (identical(health[["full_rank"]][1], FALSE)) 
        reasons <- c(reasons, "RANK DEFICIENCY")
    if (isTRUE(health[["numerical_identifiability_warning"]][1])) 
        reasons <- c(reasons, "NUMERICAL IDENTIFIABILITY FAILURE")
    unique(reasons)
}

we_reference_row <- function(data, group_col = "point_id", quantile_prob = 0.5) {
    ref <- data[1L, , drop = FALSE]
    controls <- intersect(WE_ALL_CONTROLS, names(ref))
    for (v in controls) ref[[v]] <- unname(quantile(data[[v]], quantile_prob, na.rm = TRUE))
    ref[["precip_rank_baseline"]] <- WE_CONTRAST_A[["precip_rank_baseline"]]
    ref[["radiation_rank_baseline"]] <- WE_CONTRAST_A[["radiation_rank_baseline"]]
    ref[["Month"]] <- sort(unique(data[["Month"]]))[1L]
    ref[["Year"]] <- sort(unique(data[["Year"]]))[1L]
    group_values <- levels(data[[group_col]])
    if (is.null(group_values)) 
        group_values <- sort(unique(as.character(data[[group_col]])))
    ref[[group_col]] <- factor(group_values[1L], levels = group_values)
    ref
}

we_boot_series_ids <- function(sampled_sites) {
    occurrence <- ave(seq_along(sampled_sites), sampled_sites, FUN = seq_along)
    paste0(sampled_sites, "_copy", sprintf("%02d", occurrence))
}

we_bootstrap_draw_sequence <- function(site_ids, n_draws = WE_BOOT_DRAWS, seed = WE_SEED) {
    set.seed(seed)
    rows <- vector("list", n_draws)
    for (attempt in seq_len(n_draws)) {
        sampled <- sample(site_ids, size = length(site_ids), replace = TRUE)
        rows[[attempt]] <- data.frame(attempt = attempt, draw_id = sprintf("draw_%04d", attempt), draw_position = seq_along(sampled), 
            original_point_id = sampled, boot_series_id = we_boot_series_ids(sampled), stringsAsFactors = FALSE)
    }
    do.call(rbind, rows)
}

we_build_boot_data <- function(primary, draw_rows) {
    ids <- as.character(primary[["point_id"]])
    copies <- lapply(seq_len(nrow(draw_rows)), function(i) {
        x <- primary[ids == draw_rows[["original_point_id"]][i], , drop = FALSE]
        x[["boot_series_id"]] <- draw_rows[["boot_series_id"]][i]
        x
    })
    out <- do.call(rbind, copies)
    ord <- order(out[["boot_series_id"]], out[["calendar_month_index"]])
    out <- out[ord, , drop = FALSE]
    rownames(out) <- NULL
    out[["boot_series_id"]] <- factor(out[["boot_series_id"]], levels = sort(unique(out[["boot_series_id"]])))
    out[["AR.start"]] <- we_build_ar_start(out, "boot_series_id")
    out
}

