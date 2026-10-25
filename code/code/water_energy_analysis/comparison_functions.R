# Six differences across seven historical water-energy positions.
comparison_positions <- function() {
  data.frame(position = 1:7,
             P = c(.25, .50, .50, .50, .75, .75, .25),
             R = c(.50, .25, .50, .75, .25, .50, .75))
}

comparison_definitions <- function() {
  data.frame(
    contrast = c("P_mid_minus_low_R50", "P_high_minus_mid_R50", "P_mid_minus_low_R75",
                 "R_mid_minus_low_P50", "R_high_minus_mid_P50", "R_mid_minus_low_P75"),
    start = c(1L, 3L, 7L, 2L, 3L, 5L), end = c(3L, 6L, 4L, 3L, 4L, 6L),
    variable = c(rep("precipitation", 3L), rep("radiation", 3L)),
    exploratory = c(FALSE, FALSE, TRUE, FALSE, FALSE, FALSE)
  )
}

read_comparison_data <- function(path) {
  d <- read.csv(path, colClasses = c(point_id = "character"), check.names = FALSE)
  required <- c("point_id", "Year", "Month", "calendar_month_index",
                "npp_anomaly_mean_g", "precip_rank_baseline", "radiation_rank_baseline",
                WE_MODEL1_CONTROLS, "eligible_primary_model1")
  we_assert(all(required %in% names(d)), "Required analysis fields are missing")
  we_assert(!anyDuplicated(d[c("point_id", "Year", "Month")]), "Duplicate site-month keys")
  d <- d[!is.na(d$eligible_primary_model1) & d$eligible_primary_model1 == 1L, required]
  site_numbers <- suppressWarnings(as.integer(d$point_id))
  we_assert(all(is.finite(site_numbers)), "Invalid site identifiers")
  d$point_id <- sprintf("%03d", site_numbers)
  num <- setdiff(required, "point_id")
  we_assert(all(vapply(d[num], function(x) all(is.finite(x)), logical(1))),
            "Nonfinite analysis inputs")
  we_assert(!anyNA(d$point_id), "Invalid site identifiers")
  we_assert(nrow(d) == 6992L && length(unique(d$point_id)) == 247L &&
              setequal(unique(d$Year), 2022:2025), "Unexpected analysis domain")
  ranks <- unlist(d[c("precip_rank_baseline", "radiation_rank_baseline")])
  we_assert(all(ranks >= 0 & ranks <= 1), "Ranks must be in [0,1]")
  d <- d[order(d$point_id, d$calendar_month_index), ]
  rownames(d) <- NULL
  d$point_id <- factor(d$point_id, levels = sort(unique(d$point_id)))
  d$AR.start <- we_build_ar_start(d)
  d
}

comparison_lpmatrix <- function(model, data, group = "point_id", q = .5,
                               last_site = FALSE, last_time = FALSE) {
  ref <- we_reference_row(data, group, q)
  for (v in c("Month", "Year")) {
    lev <- model$xlevels[[paste0("factor(", v, ")")]]
    if (!length(lev)) lev <- sort(unique(data[[v]]))
    ref[[v]] <- as.numeric(if (last_time) tail(lev, 1L) else lev[1L])
  }
  lev <- model$xlevels[[group]]
  if (!length(lev)) lev <- levels(data[[group]])
  ref[[group]] <- factor(if (last_site) tail(lev, 1L) else lev[1L], levels = lev)
  pos <- comparison_positions()
  nd <- do.call(rbind, lapply(seq_len(nrow(pos)), function(i) {
    r <- ref
    r$precip_rank_baseline <- pos$P[i]
    r$radiation_rank_baseline <- pos$R[i]
    r
  }))
  predict(model, newdata = nd, type = "lpmatrix")
}

comparison_values <- function(model, data, group = "point_id") {
  X <- comparison_lpmatrix(model, data, group)
  defs <- comparison_definitions()
  dx <- X[defs$end, , drop = FALSE] - X[defs$start, , drop = FALSE]
  setNames(as.numeric(dx %*% coef(model)), defs$contrast)
}

fit_comparison_model <- function(data, site_variable = "point_id") {
  fit <- we_fit_bam(we_formula_model1(site_variable = site_variable), data,
                    rho_used = 0.036623315093487, ar_start = data$AR.start)
  reasons <- we_model_health_stop_reason(we_model_health(fit$model, fit$warnings))
  we_assert(length(reasons) == 0L, paste(reasons, collapse = "; "))
  fit$model
}

summarize_comparisons <- function(estimates, bootstrap) {
  defs <- comparison_definitions()
  we_assert(all(c("contrast", "estimate") %in% names(estimates)), "Missing estimates")
  we_assert(!anyDuplicated(estimates$contrast) &&
              setequal(estimates$contrast, defs$contrast), "Comparison identifiers differ")
  we_assert(all(c("attempt", defs$contrast) %in% names(bootstrap)), "Missing bootstrap fields")
  we_assert(!anyDuplicated(bootstrap$attempt), "Duplicate bootstrap attempts")
  estimates <- estimates[match(defs$contrast, estimates$contrast), , drop = FALSE]
  draws <- as.matrix(bootstrap[, defs$contrast, drop = FALSE])
  we_assert(nrow(draws) >= 2L && all(is.finite(draws)) &&
              all(is.finite(estimates$estimate)), "Invalid bootstrap values")
  sds <- apply(draws, 2L, sd)
  we_assert(all(sds > 0), "Bootstrap SD must be positive")
  z <- sweep(sweep(draws, 2L, estimates$estimate, "-"), 2L, sds, "/")
  critical <- unname(quantile(apply(abs(z), 1L, max), .95, type = 7))
  marginal <- apply(draws, 2L, quantile, probs = c(.025, .975), type = 7)
  estimates$bootstrap_SD <- sds
  estimates$marginal_lower <- marginal[1L, ]
  estimates$marginal_upper <- marginal[2L, ]
  estimates$simultaneous_lower <- estimates$estimate - critical * sds
  estimates$simultaneous_upper <- estimates$estimate + critical * sds
  estimates$simultaneous_critical <- critical
  estimates$replicates <- nrow(draws)
  rownames(estimates) <- NULL
  estimates
}

comparison_options <- function(args, defaults) {
  if ("--help" %in% args) {
    cat("Options:\n", paste(names(defaults), defaults, sep = " = ", collapse = "\n"), "\n")
    quit(status = 0L)
  }
  we_assert(length(args) %% 2L == 0L, "Options require a value")
  if (!length(args)) return(defaults)
  for (i in seq.int(1L, length(args), by = 2L)) {
    key <- args[i]
    we_assert(key %in% names(defaults), paste("Unknown option:", key))
    defaults[[key]] <- args[i + 1L]
  }
  defaults
}
