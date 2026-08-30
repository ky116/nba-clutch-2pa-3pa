#!/usr/bin/env Rscript

ensure_pkg <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop(
      sprintf(
        "Required R package '%s' is missing in current .libPaths(). Set R_LIBS_USER to your installed library path.",
        pkg
      )
    )
  }
}

fread_maybe_gz <- function(path, ...) {
  p <- as.character(path)
  if (grepl("\\.gz$", p, ignore.case = TRUE)) {
    return(fread(cmd = sprintf("gzip -dc %s", shQuote(p)), ...))
  }
  fread(p, ...)
}

suppressPackageStartupMessages({
  ensure_pkg("data.table")
  ensure_pkg("mgcv")
  library(data.table)
  library(mgcv)
})

set.seed(123)

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag, default = NULL) {
  if (!(flag %in% args)) return(default)
  idx <- match(flag, args)
  if (is.na(idx) || idx == length(args)) return(default)
  args[[idx + 1]]
}
has_flag <- function(flag) flag %in% args
clip_prob <- function(p, eps = 1e-6) pmin(pmax(as.numeric(p), eps), 1 - eps)
inv_logit <- function(x) 1 / (1 + exp(-x))
safe_qlogis <- function(p, eps = 1e-6) qlogis(clip_prob(p, eps = eps))

coalesce_elo_diff_pregame <- function(dt, preferred_k = NA_real_, context = "data") {
  if ("elo_diff_pregame" %in% names(dt)) {
    dt[, elo_diff_pregame := as.numeric(elo_diff_pregame)]
    return(invisible(NULL))
  }
  k_label <- function(k) {
    if (!is.finite(k)) return(NA_character_)
    if (abs(k - round(k)) < 1e-9) return(as.character(as.integer(round(k))))
    gsub("\\.", "p", sub("0+$", "", sub("\\.$", "", sprintf("%.6f", k))))
  }
  cands <- character()
  if (is.finite(preferred_k)) {
    cands <- c(cands, sprintf("elo_diff_pregame_k%s", k_label(preferred_k)))
  }
  cands <- c(cands, grep("^elo_diff_pregame_k", names(dt), value = TRUE))
  cands <- unique(cands[cands %in% names(dt)])
  if (length(cands) == 0L) return(invisible(NULL))
  dt[, elo_diff_pregame := as.numeric(get(cands[[1]]))]
  cat(sprintf("Mapped %s -> elo_diff_pregame in %s.\n", cands[[1]], context))
  invisible(NULL)
}

load_platt_map <- function(path, calibration = "platt") {
  if (is.null(path) || identical(path, "")) return(NULL)
  if (!file.exists(path)) stop(sprintf("Platt coefficient file not found: %s", path))
  coef_dt <- fread(path)
  need <- c("calibration", "term", "estimate")
  miss <- setdiff(need, names(coef_dt))
  if (length(miss) > 0L) stop(sprintf("Missing columns in Platt coefficient csv: %s", paste(miss, collapse = ", ")))
  cal_name <- as.character(calibration)
  sub <- coef_dt[as.character(coef_dt$calibration) == cal_name]
  if (nrow(sub) == 0L) stop(sprintf("No coefficient rows for calibration='%s' in %s", cal_name, path))
  alpha <- suppressWarnings(as.numeric(sub[term == "(Intercept)", estimate][1]))
  beta  <- suppressWarnings(as.numeric(sub[term == "eta", estimate][1]))
  if (!is.finite(alpha) || !is.finite(beta)) {
    stop(sprintf("Calibration '%s' must include finite '(Intercept)' and 'eta' in %s", cal_name, path))
  }
  list(alpha = alpha, beta = beta, calibration = cal_name, path = path)
}

apply_platt_prob <- function(p, alpha, beta, eps = 1e-6) {
  clip_prob(inv_logit(alpha + beta * safe_qlogis(p, eps = eps)), eps = eps)
}

derive_after_off_reb_from_shot_sequence <- function(x) {
  as.integer(as.numeric(x) > 1)
}
ensure_after_off_reb <- function(dt, context = "data") {
  if (!("after_off_reb" %in% names(dt))) {
    if ("after_off_reb_state" %in% names(dt)) {
      dt[, after_off_reb := as.integer(after_off_reb_state)]
      cat(sprintf("Mapped after_off_reb_state -> after_off_reb in %s.\n", context))
    } else if ("shot_sequence" %in% names(dt)) {
      dt[, after_off_reb := derive_after_off_reb_from_shot_sequence(shot_sequence)]
      cat(sprintf("Derived after_off_reb from shot_sequence in %s.\n", context))
    } else if ("shot_sequence_at_state" %in% names(dt)) {
      dt[, after_off_reb := derive_after_off_reb_from_shot_sequence(shot_sequence_at_state)]
      cat(sprintf("Derived after_off_reb from shot_sequence_at_state in %s.\n", context))
    } else {
      dt[, after_off_reb := 0L]
      warning(sprintf("No `after_off_reb`/`after_off_reb_state`/`shot_sequence`/`shot_sequence_at_state` in %s; using after_off_reb=0.", context))
    }
  }
  dt[, after_off_reb := as.integer(after_off_reb)]
  dt[is.na(after_off_reb), after_off_reb := 0L]
}
season_to_era <- function(season) {
  fifelse(season <= 2003, "transition_pre2004",
    fifelse(season <= 2013, "post_handcheck_pre3p",
      "pace_space_3p"))
}
season_to_era4 <- function(season) {
  fifelse(season <= 2003, "transition_pre2004",
    fifelse(season <= 2014, "post_handcheck_pre3p",
      fifelse(season <= 2017, "pace_space_early", "modern_3p")))
}
START_TYPE_DEAD <- c(
  "off dead ball",
  "off timeout",
  "off ft make",
  "off at rim make",
  "off long mid-range make",
  "off short mid-range make",
  "off arc 3 make",
  "off corner 3 make"
)
START_TYPE_LIVE <- c(
  "off steal",
  "off ft miss",
  "off long mid-range miss",
  "off short mid-range miss",
  "off arc 3 miss",
  "off corner 3 miss",
  "off at rim miss",
  "off at rim block",
  "off block"
)
classify_start_type_group <- function(x) {
  s <- tolower(trimws(as.character(x)))
  s[is.na(s)] <- ""
  out <- rep("dead_ball", length(s))
  out[s %in% START_TYPE_DEAD] <- "dead_ball"
  out[s %in% START_TYPE_LIVE] <- "live_ball"
  out[grepl("inbound|jump ball", s)] <- "dead_ball"
  out[s == "" | s == "unknown"] <- "dead_ball"
  out
}
start_type_to_group <- function(x) {
  out <- classify_start_type_group(x)
  factor(out, levels = c("live_ball", "dead_ball"))
}
start_type_to_dead_ball_indicator <- function(x) {
  as.integer(classify_start_type_group(x) == "dead_ball")
}
build_start_type_group <- function(x) {
  factor(classify_start_type_group(x), levels = c("live_ball", "dead_ball"))
}
add_late_close_tail <- function(dt, time_sec = 60, score_abs = 3) {
  dt[, late_close_tail := as.numeric(
    is.finite(time_left_game) &
      is.finite(score_diff) &
      time_left_game <= as.numeric(time_sec) &
      abs(score_diff) <= as.numeric(score_abs)
  )]
  dt[, late_time_band := fifelse(
    late_close_tail == 1 & time_left_game <= 30,
    "t0_30",
    fifelse(late_close_tail == 1 & time_left_game <= 60, "t30_60", "other")
  )]
  dt[, late_time_band := factor(late_time_band, levels = c("other", "t0_30", "t30_60"))]
}
late_tail_terms <- function(variant) {
  if (identical(variant, "none")) return(character())
  if (identical(variant, "main")) return("late_close_tail")
  if (identical(variant, "band")) return("late_time_band")
  if (identical(variant, "score-band")) return(c("late_time_band", "score_diff:late_time_band"))
  if (identical(variant, "surface")) {
    return(c("late_close_tail", "ti(score_diff, time_left_game, by=late_close_tail, bs=c('ts','ts'), k=c(6,6), id='tail')"))
  }
  stop(sprintf("Unknown late-tail variant: %s", variant))
}
season_to_era_from_levels <- function(season, levels_hint = NULL) {
  if (!is.null(levels_hint) && any(levels_hint %in% c("pace_space_early", "modern_3p"))) {
    return(season_to_era4(season))
  }
  season_to_era(season)
}
check_home_away_orientation <- function(dt, score_col = "score_diff", y_col = "final_home_win") {
  ok <- dt[!is.na(get(score_col)) & !is.na(get(y_col))]
  if (nrow(ok) < 1000L) {
    warning("Orientation check skipped: too few non-missing rows.")
    return(invisible(NULL))
  }
  corr <- suppressWarnings(cor(ok[[score_col]], ok[[y_col]]))
  cat(sprintf("Orientation check: cor(%s, %s)=%.4f\n", score_col, y_col, corr))
  if (is.na(corr) || corr <= 0) {
    stop(
      sprintf(
        "Detected non-home-away orientation (cor(%s,%s)=%.4f). ",
        score_col, y_col, corr
      ),
      "Expected score_diff as home-away and y as final_home_win."
    )
  }
}
detect_model_direction <- function(model_obj) {
  fm_vars <- all.vars(formula(model_obj))
  d <- data.table(score_diff = c(-5, 5), time_left_game = c(120, 120), OT_flag = c(0L, 0L), home_possession = c(1L, 1L))
  if ("after_off_reb" %in% fm_vars) {
    d[, after_off_reb := 0L]
  }
  if ("start_type" %in% fm_vars) {
    lv <- NULL
    if (!is.null(model_obj$model) && "start_type" %in% names(model_obj$model)) lv <- levels(model_obj$model$start_type)
    if (is.null(lv) || length(lv) == 0) lv <- c("UNKNOWN")
    d[, start_type := factor(lv[[1]], levels = lv)]
  }
  if ("state_type" %in% fm_vars) {
    lv <- NULL
    if (!is.null(model_obj$model) && "state_type" %in% names(model_obj$model)) lv <- levels(model_obj$model$state_type)
    if (is.null(lv) || length(lv) == 0) lv <- c("poss_start", "shot_state", "off_reb")
    pick <- if ("poss_start" %in% lv) "poss_start" else lv[[1]]
    d[, state_type := factor(pick, levels = lv)]
  }
  if ("state_type3" %in% fm_vars) {
    lv <- NULL
    if (!is.null(model_obj$model) && "state_type3" %in% names(model_obj$model)) lv <- levels(model_obj$model$state_type3)
    if (is.null(lv) || length(lv) == 0) lv <- c("poss_start", "shot_state", "off_reb")
    pick <- if ("poss_start" %in% lv) "poss_start" else lv[[1]]
    d[, state_type3 := factor(pick, levels = lv)]
  }
  if ("era" %in% fm_vars) {
    lv <- NULL
    if (!is.null(model_obj$model) && "era" %in% names(model_obj$model)) lv <- levels(model_obj$model$era)
    if (is.null(lv) || length(lv) == 0) lv <- c("pace_space_3p")
    d[, era := factor(lv[[1]], levels = lv)]
  }
  if ("start_type_group" %in% fm_vars) {
    lv <- NULL
    if (!is.null(model_obj$model) && "start_type_group" %in% names(model_obj$model)) lv <- levels(model_obj$model$start_type_group)
    if (is.null(lv) || length(lv) == 0) lv <- c("dead_ball", "live_ball")
    d[, start_type_group := factor(lv[[1]], levels = lv)]
  }
  if ("home_possession_f" %in% fm_vars) {
    lv <- NULL
    if (!is.null(model_obj$model) && "home_possession_f" %in% names(model_obj$model)) lv <- levels(model_obj$model$home_possession_f)
    if (is.null(lv) || length(lv) == 0) lv <- c("away", "home")
    d[, home_possession_f := factor("home", levels = lv)]
  }
  if ("dead_ball_indicator" %in% fm_vars) {
    d[, dead_ball_indicator := 1L]
  }
  if ("elo_diff_pregame" %in% fm_vars) {
    d[, elo_diff_pregame := 0.0]
  }
  if ("late_close_tail" %in% fm_vars) {
    d[, late_close_tail := 0.0]
  }
  p <- suppressWarnings(as.numeric(predict(model_obj, newdata = d, type = "response")))
  if (length(p) != 2 || any(!is.finite(p))) {
    warning("Could not detect model direction reliably; keep as-is.")
    return("unknown")
  }
  if (p[2] > p[1]) return("home")
  if (p[2] < p[1]) return("away")
  "unknown"
}

report_model_spec <- function(model_obj) {
  fm <- formula(model_obj)
  vars <- all.vars(fm)
  smooth_labels <- character()
  if (!is.null(model_obj$smooth) && length(model_obj$smooth) > 0) {
    smooth_labels <- vapply(model_obj$smooth, function(s) s$label, character(1))
  }
  has_te <- any(grepl("^te\\(score_diff,time_left_game\\)$", smooth_labels))
  cat(sprintf("Model spec check: has_te(score_diff,time_left_game)=%s\n", has_te))
  cat(sprintf("Model spec check: has_start_type=%s has_state_type=%s has_state_type3=%s has_after_off_reb=%s has_era=%s has_elo_diff_pregame=%s\n",
              "start_type" %in% vars, "state_type" %in% vars, "state_type3" %in% vars, "after_off_reb" %in% vars, "era" %in% vars, "elo_diff_pregame" %in% vars))
  cat("Model formula (deparse):\n")
  cat(paste(deparse(fm), collapse = "\n"), "\n")
}

extract_model_meta <- function(model_obj) {
  vars <- all.vars(formula(model_obj))
  out <- list(
    uses_start_type = "start_type" %in% vars,
    uses_state_type = "state_type" %in% vars,
    uses_state_type3 = "state_type3" %in% vars,
    uses_after_off_reb = "after_off_reb" %in% vars,
    uses_era = "era" %in% vars,
    uses_start_type_group = "start_type_group" %in% vars,
    uses_home_possession_f = "home_possession_f" %in% vars,
    uses_dead_ball_indicator = "dead_ball_indicator" %in% vars,
    uses_elo_diff_pregame = "elo_diff_pregame" %in% vars,
    uses_late_close_tail = "late_close_tail" %in% vars,
    late_tail_time_sec = attr(model_obj, "late_tail_time_sec", exact = TRUE),
    late_tail_score_abs = attr(model_obj, "late_tail_score_abs", exact = TRUE),
    start_levels = NULL,
    state_type_levels = NULL,
    state_type3_levels = NULL,
    era_levels = NULL,
    start_group_levels = NULL,
    home_possession_f_levels = NULL
  )
  if (!is.null(model_obj$model) && "start_type" %in% names(model_obj$model)) {
    out$start_levels <- levels(model_obj$model$start_type)
  }
  if (!is.null(model_obj$model) && "state_type" %in% names(model_obj$model)) {
    out$state_type_levels <- levels(model_obj$model$state_type)
  }
  if (!is.null(model_obj$model) && "state_type3" %in% names(model_obj$model)) {
    out$state_type3_levels <- levels(model_obj$model$state_type3)
  }
  if (!is.null(model_obj$model) && "era" %in% names(model_obj$model)) {
    out$era_levels <- levels(model_obj$model$era)
  }
  if (!is.null(model_obj$model) && "start_type_group" %in% names(model_obj$model)) {
    out$start_group_levels <- levels(model_obj$model$start_type_group)
  }
  if (!is.null(model_obj$model) && "home_possession_f" %in% names(model_obj$model)) {
    out$home_possession_f_levels <- levels(model_obj$model$home_possession_f)
  }
  if (length(out$late_tail_time_sec) == 0L || !is.finite(as.numeric(out$late_tail_time_sec))) out$late_tail_time_sec <- 60
  if (length(out$late_tail_score_abs) == 0L || !is.finite(as.numeric(out$late_tail_score_abs))) out$late_tail_score_abs <- 3
  out$model_direction <- detect_model_direction(model_obj)
  out$model_is_reversed <- identical(out$model_direction, "away")
  out
}

prepare_states_for_model <- function(dt_in, meta, context = "state rows") {
  dt <- copy(dt_in)
  if (!("score_diff" %in% names(dt)) && ("score_diff_home" %in% names(dt))) {
    dt[, score_diff := as.numeric(score_diff_home)]
  }
  need <- c("score_diff", "time_left_game", "OT_flag", "home_possession")
  miss <- setdiff(need, names(dt))
  if (length(miss) > 0L) stop(sprintf("Missing in %s: %s", context, paste(miss, collapse = ", ")))

  dt[, `:=`(
    score_diff = as.numeric(score_diff),
    time_left_game = as.numeric(time_left_game),
    OT_flag = as.integer(OT_flag),
    home_possession = as.integer(home_possession)
  )]

  if (meta$uses_start_type) {
    if (!("start_type" %in% names(dt))) dt[, start_type := "UNKNOWN"]
    dt[, start_type := trimws(as.character(start_type))]
    dt[is.na(start_type) | start_type == "", start_type := "UNKNOWN"]
    if (!is.null(meta$start_levels) && length(meta$start_levels) > 0) {
      fallback_level <- if ("UNKNOWN" %in% meta$start_levels) "UNKNOWN" else meta$start_levels[[1]]
      dt[!(start_type %in% meta$start_levels), start_type := fallback_level]
      dt[, start_type := factor(start_type, levels = meta$start_levels)]
    } else {
      dt[, start_type := as.factor(start_type)]
    }
  }
  if (meta$uses_state_type) {
    if (!("state_type" %in% names(dt))) dt[, state_type := "poss_start"]
    dt[, state_type := trimws(as.character(state_type))]
    dt[is.na(state_type) | state_type == "", state_type := "poss_start"]
    if (!is.null(meta$state_type_levels) && length(meta$state_type_levels) > 0) {
      fallback_state <- if ("poss_start" %in% meta$state_type_levels) "poss_start" else meta$state_type_levels[[1]]
      dt[!(state_type %in% meta$state_type_levels), state_type := fallback_state]
      dt[, state_type := factor(state_type, levels = meta$state_type_levels)]
    } else {
      dt[, state_type := as.factor(state_type)]
    }
  }
  if (meta$uses_state_type3) {
    if (!("state_type3" %in% names(dt))) {
      if ("state_type" %in% names(dt)) {
        dt[, state_type3 := as.character(state_type)]
      } else {
        dt[, state_type3 := "poss_start"]
      }
    }
    dt[, state_type3 := trimws(as.character(state_type3))]
    dt[is.na(state_type3) | state_type3 == "", state_type3 := "poss_start"]
    if (!is.null(meta$state_type3_levels) && length(meta$state_type3_levels) > 0) {
      fallback_state3 <- if ("poss_start" %in% meta$state_type3_levels) "poss_start" else meta$state_type3_levels[[1]]
      dt[!(state_type3 %in% meta$state_type3_levels), state_type3 := fallback_state3]
      dt[, state_type3 := factor(state_type3, levels = meta$state_type3_levels)]
    } else {
      dt[, state_type3 := as.factor(state_type3)]
    }
  }
  if (meta$uses_after_off_reb) ensure_after_off_reb(dt, context = context)
  if (meta$uses_elo_diff_pregame) {
    coalesce_elo_diff_pregame(dt, context = context)
    if (!("elo_diff_pregame" %in% names(dt))) {
      stop(sprintf("Model requires elo_diff_pregame, but %s has no elo_diff_pregame(_k*) column.", context))
    }
    dt[, elo_diff_pregame := as.numeric(elo_diff_pregame)]
  }
  if (meta$uses_era) {
    if ("season" %in% names(dt)) {
      dt[, era := season_to_era_from_levels(as.integer(season), meta$era_levels)]
    } else if (!("era" %in% names(dt))) {
      dt[, era := "pace_space_3p"]
    }
    dt[, era := trimws(as.character(era))]
    dt[is.na(era) | era == "", era := "pace_space_3p"]
    if (!is.null(meta$era_levels) && length(meta$era_levels) > 0) {
      fallback_era <- if ("pace_space_3p" %in% meta$era_levels) "pace_space_3p" else meta$era_levels[[1]]
      dt[!(era %in% meta$era_levels), era := fallback_era]
      dt[, era := factor(era, levels = meta$era_levels)]
    } else {
      dt[, era := as.factor(era)]
    }
  }
  if (meta$uses_start_type_group) {
    if (!("start_type" %in% names(dt))) dt[, start_type := "UNKNOWN"]
    dt[, start_type_group := start_type_to_group(start_type)]
    if (!is.null(meta$start_group_levels) && length(meta$start_group_levels) > 0) {
      dt[, start_type_group := factor(as.character(start_type_group), levels = meta$start_group_levels)]
      fallback_group <- if ("dead_ball" %in% meta$start_group_levels) "dead_ball" else meta$start_group_levels[[1]]
      dt[is.na(start_type_group), start_type_group := fallback_group]
      dt[, start_type_group := factor(as.character(start_type_group), levels = meta$start_group_levels)]
    }
  }
  if (meta$uses_home_possession_f) {
    lv <- if (!is.null(meta$home_possession_f_levels) && length(meta$home_possession_f_levels) > 0) meta$home_possession_f_levels else c("away", "home")
    dt[, home_possession_f := factor(fifelse(as.integer(home_possession) == 1L, "home", "away"), levels = lv)]
  }
  if (meta$uses_dead_ball_indicator) {
    if (!("start_type" %in% names(dt))) dt[, start_type := "UNKNOWN"]
    dt[, dead_ball_indicator := start_type_to_dead_ball_indicator(start_type)]
  }
  if (isTRUE(meta$uses_late_close_tail)) {
    add_late_close_tail(dt, time_sec = meta$late_tail_time_sec, score_abs = meta$late_tail_score_abs)
  }
  dt
}

score_states_with_model <- function(dt_in, model_obj, context = "state rows") {
  meta <- extract_model_meta(model_obj)
  dt <- prepare_states_for_model(dt_in, meta = meta, context = context)
  pred_dt <- dt[, .(
    score_diff = score_diff,
    time_left_game = time_left_game,
    OT_flag = OT_flag,
    home_possession = home_possession
  )]
  if (meta$uses_start_type) pred_dt[, start_type := dt$start_type]
  if (meta$uses_state_type) pred_dt[, state_type := dt$state_type]
  if (meta$uses_state_type3) pred_dt[, state_type3 := dt$state_type3]
  if (meta$uses_after_off_reb) pred_dt[, after_off_reb := dt$after_off_reb]
  if (meta$uses_elo_diff_pregame) pred_dt[, elo_diff_pregame := dt$elo_diff_pregame]
  if (meta$uses_era) pred_dt[, era := dt$era]
  if (meta$uses_start_type_group) pred_dt[, start_type_group := dt$start_type_group]
  if (meta$uses_home_possession_f) pred_dt[, home_possession_f := dt$home_possession_f]
  if (meta$uses_dead_ball_indicator) pred_dt[, dead_ball_indicator := dt$dead_ball_indicator]
  if (isTRUE(meta$uses_late_close_tail)) pred_dt[, late_close_tail := dt$late_close_tail]

  p <- suppressWarnings(as.numeric(predict(model_obj, newdata = pred_dt, type = "response")))
  p <- clip_prob(p)
  if (isTRUE(meta$model_is_reversed)) p <- 1.0 - p
  dt[, wp_hat_raw := p]
  dt
}

fit_platt_glm_ab <- function(y, p_hat) {
  yv <- as.integer(y)
  pv <- clip_prob(as.numeric(p_hat))
  keep <- is.finite(yv) & is.finite(pv) & (yv %in% c(0L, 1L))
  yv <- yv[keep]
  pv <- pv[keep]
  if (length(yv) < 20L || length(unique(yv)) < 2L) return(NULL)
  eta <- safe_qlogis(pv)
  fit <- tryCatch(glm(yv ~ eta, family = binomial()), error = function(e) NULL)
  if (is.null(fit)) return(NULL)
  cf <- coef(fit)
  alpha <- suppressWarnings(as.numeric(cf[[1]]))
  beta <- suppressWarnings(as.numeric(cf[[2]]))
  if (!is.finite(alpha) || !is.finite(beta)) return(NULL)
  list(alpha = alpha, beta = beta)
}

season_oof_platt_mask <- function(dt, subset = "clutch", clutch_time_sec = 300, clutch_score_diff = 10) {
  if (identical(subset, "all")) {
    return(rep(TRUE, nrow(dt)))
  }
  if (!(subset %in% c("clutch"))) {
    stop(sprintf("Unsupported season-OOF Platt subset: %s", subset))
  }
  time_left <- suppressWarnings(as.numeric(dt$time_left_game))
  score_diff <- suppressWarnings(as.numeric(dt$score_diff))
  is.finite(time_left) & is.finite(score_diff) &
    time_left <= as.numeric(clutch_time_sec) &
    abs(score_diff) <= as.numeric(clutch_score_diff)
}

prepare_shots_for_model <- function(shots_in, meta) {
  shots <- copy(shots_in)
  if (meta$uses_start_type) {
    shots[, before_start_type := trimws(as.character(before_start_type))]
    shots[, next_start_type := trimws(as.character(next_start_type))]
    shots[is.na(before_start_type) | before_start_type == "", before_start_type := "UNKNOWN"]
    shots[is.na(next_start_type) | next_start_type == "", next_start_type := "UNKNOWN"]
    if (!is.null(meta$start_levels) && length(meta$start_levels) > 0) {
      fallback_level <- if ("UNKNOWN" %in% meta$start_levels) "UNKNOWN" else meta$start_levels[[1]]
      shots[!(before_start_type %in% meta$start_levels), before_start_type := fallback_level]
      shots[!(next_start_type %in% meta$start_levels), next_start_type := fallback_level]
    }
  }
  if (meta$uses_state_type) {
    if (!("before_state_type" %in% names(shots))) {
      shots[, before_state_type := "shot_state"]
    }
    shots[, before_state_type := trimws(as.character(before_state_type))]
    shots[is.na(before_state_type) | before_state_type == "", before_state_type := "shot_state"]
    shots[, next_state_type := fifelse(next_type == "off_reb", "off_reb", fifelse(next_type == "next_poss_start", "poss_start", NA_character_))]
    if (!is.null(meta$state_type_levels) && length(meta$state_type_levels) > 0) {
      fallback_state <- if ("poss_start" %in% meta$state_type_levels) "poss_start" else meta$state_type_levels[[1]]
      shots[!(before_state_type %in% meta$state_type_levels), before_state_type := fallback_state]
      shots[!(next_state_type %in% meta$state_type_levels), next_state_type := fallback_state]
      shots[, before_state_type := factor(before_state_type, levels = meta$state_type_levels)]
      shots[, next_state_type := factor(next_state_type, levels = meta$state_type_levels)]
    } else {
      shots[, before_state_type := as.factor(before_state_type)]
      shots[, next_state_type := as.factor(next_state_type)]
    }
  }
  if (meta$uses_state_type3) {
    if (!("before_state_type" %in% names(shots))) {
      shots[, before_state_type := "shot_state"]
    }
    shots[, before_state_type3 := trimws(as.character(before_state_type))]
    shots[is.na(before_state_type3) | before_state_type3 == "", before_state_type3 := "shot_state"]
    shots[, next_state_type3 := fifelse(next_type == "off_reb", "off_reb", fifelse(next_type == "next_poss_start", "poss_start", NA_character_))]
    if (!is.null(meta$state_type3_levels) && length(meta$state_type3_levels) > 0) {
      fallback_state3 <- if ("poss_start" %in% meta$state_type3_levels) "poss_start" else meta$state_type3_levels[[1]]
      shots[!(before_state_type3 %in% meta$state_type3_levels), before_state_type3 := fallback_state3]
      shots[!(next_state_type3 %in% meta$state_type3_levels), next_state_type3 := fallback_state3]
      shots[, before_state_type3 := factor(before_state_type3, levels = meta$state_type3_levels)]
      shots[, next_state_type3 := factor(next_state_type3, levels = meta$state_type3_levels)]
    } else {
      shots[, before_state_type3 := as.factor(before_state_type3)]
      shots[, next_state_type3 := as.factor(next_state_type3)]
    }
  }
  if (meta$uses_after_off_reb) {
    shots[, after_off_reb := as.integer(after_off_reb)]
    shots[is.na(after_off_reb), after_off_reb := 0L]
  }
  if (meta$uses_elo_diff_pregame) {
    coalesce_elo_diff_pregame(shots, context = "shot states")
    if (!("elo_diff_pregame" %in% names(shots))) {
      stop("Model requires elo_diff_pregame, but no elo_diff_pregame(_k*) column was found in shot states.")
    }
    shots[, elo_diff_pregame := as.numeric(elo_diff_pregame)]
  }
  if (meta$uses_era) {
    if ("season" %in% names(shots)) {
      shots[, era := season_to_era_from_levels(as.integer(season), meta$era_levels)]
    } else if (!("era" %in% names(shots))) {
      shots[, era := "pace_space_3p"]
    }
    shots[, era := trimws(as.character(era))]
    shots[is.na(era) | era == "", era := "pace_space_3p"]
    if (!is.null(meta$era_levels) && length(meta$era_levels) > 0) {
      fallback_era <- if ("pace_space_3p" %in% meta$era_levels) "pace_space_3p" else meta$era_levels[[1]]
      shots[!(era %in% meta$era_levels), era := fallback_era]
      shots[, era := factor(era, levels = meta$era_levels)]
    } else {
      shots[, era := as.factor(era)]
    }
  }
  if (meta$uses_start_type_group) {
    shots[, before_start_type_group := start_type_to_group(before_start_type)]
    shots[, next_start_type_group := start_type_to_group(next_start_type)]
    if (!is.null(meta$start_group_levels) && length(meta$start_group_levels) > 0) {
      shots[, before_start_type_group := factor(as.character(before_start_type_group), levels = meta$start_group_levels)]
      shots[, next_start_type_group := factor(as.character(next_start_type_group), levels = meta$start_group_levels)]
      fallback_group <- if ("dead_ball" %in% meta$start_group_levels) "dead_ball" else meta$start_group_levels[[1]]
      shots[is.na(before_start_type_group), before_start_type_group := fallback_group]
      shots[is.na(next_start_type_group), next_start_type_group := fallback_group]
      shots[, before_start_type_group := factor(as.character(before_start_type_group), levels = meta$start_group_levels)]
      shots[, next_start_type_group := factor(as.character(next_start_type_group), levels = meta$start_group_levels)]
    }
  }
  if (meta$uses_home_possession_f) {
    lv <- if (!is.null(meta$home_possession_f_levels) && length(meta$home_possession_f_levels) > 0) meta$home_possession_f_levels else c("away", "home")
    shots[, before_home_possession_f := factor(fifelse(as.integer(before_home_possession) == 1L, "home", "away"), levels = lv)]
    shots[, next_home_possession_f := factor(fifelse(as.integer(next_home_possession) == 1L, "home", "away"), levels = lv)]
  }
  if (meta$uses_dead_ball_indicator) {
    shots[, before_dead_ball_indicator := start_type_to_dead_ball_indicator(before_start_type)]
    shots[, next_dead_ball_indicator := start_type_to_dead_ball_indicator(next_start_type)]
  }
  shots
}

score_shots_with_model <- function(shots_in, model_obj) {
  meta <- extract_model_meta(model_obj)
  shots <- prepare_shots_for_model(shots_in, meta)

  before_dt <- shots[, .(
    score_diff = before_score_diff,
    time_left_game = before_time_left_game,
    OT_flag = before_OT_flag,
    home_possession = before_home_possession
  )]
  if (meta$uses_start_type) {
    before_dt[, start_type := shots$before_start_type]
    if (!is.null(meta$start_levels) && length(meta$start_levels) > 0) {
      before_dt[, start_type := factor(start_type, levels = meta$start_levels)]
    } else {
      before_dt[, start_type := as.factor(start_type)]
    }
  }
  if (meta$uses_state_type) before_dt[, state_type := shots$before_state_type]
  if (meta$uses_state_type3) before_dt[, state_type3 := shots$before_state_type3]
  if (meta$uses_after_off_reb) before_dt[, after_off_reb := shots$after_off_reb]
  if (meta$uses_elo_diff_pregame) before_dt[, elo_diff_pregame := shots$elo_diff_pregame]
  if (meta$uses_era) before_dt[, era := shots$era]
  if (meta$uses_start_type_group) before_dt[, start_type_group := shots$before_start_type_group]
  if (meta$uses_home_possession_f) before_dt[, home_possession_f := shots$before_home_possession_f]
  if (meta$uses_dead_ball_indicator) before_dt[, dead_ball_indicator := shots$before_dead_ball_indicator]
  if (isTRUE(meta$uses_late_close_tail)) {
    add_late_close_tail(before_dt, time_sec = meta$late_tail_time_sec, score_abs = meta$late_tail_score_abs)
  }

  shots[, wp_before := predict(model_obj, newdata = before_dt, type = "response")]
  if (meta$model_is_reversed) shots[, wp_before := 1.0 - wp_before]

  shots[, wp_next := as.numeric(NA)]
  shots[, max_game_event_id := max(GAME_EVENT_ID, na.rm = TRUE), by = GAME_ID]
  is_true_game_end <- (
    shots$next_is_terminal == 1L &
      shots$next_type == "terminal" &
      !is.na(shots$GAME_EVENT_ID) &
      shots$GAME_EVENT_ID == shots$max_game_event_id
  )
  has_terminal_wp <- is_true_game_end & !is.na(shots$final_home_win)
  if (any(has_terminal_wp, na.rm = TRUE)) {
    shots[has_terminal_wp, wp_next := as.numeric(final_home_win)]
  }
  shots[, max_game_event_id := NULL]

  is_terminal_state <- (
    shots$next_is_terminal == 1L |
      shots$next_type == "terminal"
  )

  term_mask <- (
    !has_terminal_wp &
      (
        is_terminal_state |
          is.na(shots$next_score_diff) |
          is.na(shots$next_time_left_game) |
          is.na(shots$next_OT_flag) |
          is.na(shots$next_home_possession)
      )
  )
  idx_pred <- which(!has_terminal_wp & !term_mask)
  if (length(idx_pred) > 0) {
    next_dt <- shots[idx_pred, .(
      score_diff = next_score_diff,
      time_left_game = next_time_left_game,
      OT_flag = next_OT_flag,
      home_possession = next_home_possession
    )]
    if (meta$uses_start_type) {
      next_dt[, start_type := shots[idx_pred]$next_start_type]
      if (!is.null(meta$start_levels) && length(meta$start_levels) > 0) {
        next_dt[, start_type := factor(start_type, levels = meta$start_levels)]
      } else {
        next_dt[, start_type := as.factor(start_type)]
      }
    }
    if (meta$uses_state_type) next_dt[, state_type := shots[idx_pred]$next_state_type]
    if (meta$uses_state_type3) next_dt[, state_type3 := shots[idx_pred]$next_state_type3]
    if (meta$uses_after_off_reb) next_dt[, after_off_reb := as.integer(shots[idx_pred]$next_type == "off_reb")]
    if (meta$uses_elo_diff_pregame) next_dt[, elo_diff_pregame := shots[idx_pred]$elo_diff_pregame]
    if (meta$uses_era) next_dt[, era := shots[idx_pred]$era]
    if (meta$uses_start_type_group) next_dt[, start_type_group := shots[idx_pred]$next_start_type_group]
    if (meta$uses_home_possession_f) next_dt[, home_possession_f := shots[idx_pred]$next_home_possession_f]
    if (meta$uses_dead_ball_indicator) next_dt[, dead_ball_indicator := shots[idx_pred]$next_dead_ball_indicator]
    if (isTRUE(meta$uses_late_close_tail)) {
      add_late_close_tail(next_dt, time_sec = meta$late_tail_time_sec, score_abs = meta$late_tail_score_abs)
    }
    shots[idx_pred, wp_next := predict(model_obj, newdata = next_dt, type = "response")]
    if (meta$model_is_reversed) shots[idx_pred, wp_next := 1.0 - wp_next]
  }

  shots[, delta_wp := wp_next - wp_before]
  list(
    scored = shots[!term_mask],
    n_terminal = sum(term_mask, na.rm = TRUE),
    n_terminal_kept = sum(has_terminal_wp, na.rm = TRUE)
  )
}

protocol_m0_formula <- function(use_elo_base = FALSE, use_late_close_tail = FALSE) {
  terms <- c(
    "era",
    "OT_flag",
    "home_possession",
    "s(score_diff, bs='cr', k=15)",
    "s(time_left_game, bs='cr', k=20)",
    "ti(score_diff, time_left_game, bs=c('cr','cr'), k=c(12,12))",
    "ti(score_diff, time_left_game, by=home_possession, bs=c('ts','ts'), k=c(6,6), id='pos')",
    "state_type",
    "s(score_diff, state_type3, bs='fs', k=6)",
    "s(time_left_game, state_type3, bs='fs', k=8)",
    "start_type_group",
    "ti(score_diff, time_left_game, by=start_type_group, bs=c('ts','ts'), k=c(5,5), id='st')",
    "after_off_reb",
    "ti(score_diff, time_left_game, by=after_off_reb, bs=c('ts','ts'), k=c(5,5), id='orb')"
  )
  if (isTRUE(use_elo_base)) {
    terms <- c(terms, "s(elo_diff_pregame, bs='cr', k=10)")
  }
  if (isTRUE(use_late_close_tail)) {
    terms <- c(terms, "late_close_tail", "ti(score_diff, time_left_game, by=late_close_tail, bs=c('ts','ts'), k=c(6,6), id='tail')")
  }
  as.formula(paste("final_home_win ~", paste(terms, collapse = " + ")))
}

protocol_m0_template_meta <- function(
  use_elo_base = FALSE,
  use_late_close_tail = FALSE,
  late_tail_time_sec = 60,
  late_tail_score_abs = 3
) {
  list(
    uses_start_type = FALSE,
    uses_state_type = TRUE,
    uses_state_type3 = TRUE,
    uses_after_off_reb = TRUE,
    uses_era = TRUE,
    uses_start_type_group = TRUE,
    uses_home_possession_f = FALSE,
    uses_dead_ball_indicator = FALSE,
    uses_elo_diff_pregame = isTRUE(use_elo_base),
    uses_late_close_tail = isTRUE(use_late_close_tail),
    late_tail_time_sec = as.numeric(late_tail_time_sec),
    late_tail_score_abs = as.numeric(late_tail_score_abs),
    start_levels = NULL,
    state_type_levels = c("off_reb", "poss_start", "shot_state"),
    state_type3_levels = c("off_reb", "poss_start", "shot_state"),
    era_levels = c("modern_3p", "pace_space_early", "post_handcheck_pre3p", "transition_pre2004"),
    start_group_levels = c("live_ball", "dead_ball"),
    home_possession_f_levels = NULL,
    model_direction = "home",
    model_is_reversed = FALSE
  )
}

start_season <- as.integer(get_arg("--start-season", "2000"))
end_season   <- as.integer(get_arg("--end-season", "2024"))
seasontype   <- get_arg("--seasontype", "rs")
model_out    <- get_arg("--model-out", "models/wp_gam_leagueavg_unified_ot.rds")
train_path_override <- get_arg("--train-path", NULL)
model_in     <- get_arg("--model-in", NULL)
skip_shots   <- !is.null(get_arg("--skip-shot-scoring", NULL))
use_start_type <- !has_flag("--no-start-type")
use_state_type <- !has_flag("--no-state-type")
use_after_off_reb <- !has_flag("--no-after-off-reb")
use_era_smooth_interaction <- !has_flag("--no-era-smooth-interaction")
use_start_type_re <- has_flag("--start-type-re")
use_home_possession_by_surface <- has_flag("--home-possession-by-surface")
use_start_type_group_smooth <- has_flag("--start-type-group-smooth")
use_dead_ball_by_surface <- has_flag("--dead-ball-by-surface")
use_protocol_m0_spec <- has_flag("--protocol-m0-spec")
use_protocol_m0_elo_base <- has_flag("--protocol-m0-use-elo-base") || (as.integer(get_arg("--use-elo-base", "0")) != 0L)
use_oof_protocol_m0_template <- has_flag("--oof-template-protocol-m0-spec")
use_late_close_tail_surface <- has_flag("--late-close-tail-surface")
late_tail_time_sec <- suppressWarnings(as.numeric(get_arg("--late-tail-time-sec", "60")))
late_tail_score_abs <- suppressWarnings(as.numeric(get_arg("--late-tail-score-abs", "3")))
shot_path_override <- get_arg("--shot-path", NULL)
season_oof_predict <- !has_flag("--no-season-oof-predict")
season_oof_jobs <- suppressWarnings(as.integer(get_arg("--season-oof-jobs", Sys.getenv("SEASON_OOF_JOBS", "1"))))
if (!is.finite(season_oof_jobs) || season_oof_jobs < 1L) season_oof_jobs <- 1L
bam_nthreads <- suppressWarnings(as.integer(get_arg("--bam-nthreads", Sys.getenv("BAM_NTHREADS", "1"))))
if (!is.finite(bam_nthreads) || bam_nthreads < 1L) bam_nthreads <- 1L
elo_k_target <- suppressWarnings(as.numeric(get_arg("--elo-k-target", "NA")))
platt_coef_csv <- get_arg("--platt-coef-csv", "")
platt_calibration <- get_arg("--platt-calibration", "platt")
season_oof_platt <- has_flag("--season-oof-platt")
season_oof_platt_fit_path <- get_arg("--season-oof-platt-fit-path", NULL)
season_oof_platt_subset <- tolower(get_arg("--season-oof-platt-subset", "clutch"))
season_oof_platt_clutch_time_sec <- as.numeric(get_arg("--season-oof-platt-clutch-time-sec", "300"))
season_oof_platt_clutch_score_diff <- as.numeric(get_arg("--season-oof-platt-clutch-score-diff", "10"))
season_oof_platt_coef_out <- get_arg("--season-oof-platt-coef-out", "")
full_out_override <- get_arg("--full-out", NULL)
wp_out_override <- get_arg("--wp-out", NULL)
dml_out_override <- get_arg("--dml-out", NULL)
gam_wp <- NULL
wp_formula_template <- NULL

if (isTRUE(use_oof_protocol_m0_template)) {
  if (!isTRUE(season_oof_predict)) {
    stop("--oof-template-protocol-m0-spec requires season-OOF prediction.")
  }
  if (!is.null(model_in)) {
    stop("Use --oof-template-protocol-m0-spec without --model-in; the point is to avoid a full-data model template.")
  }
  if (!isTRUE(use_protocol_m0_spec)) {
    stop("--oof-template-protocol-m0-spec requires --protocol-m0-spec.")
  }
}

# -----------------------------
# 1) WP学習データ（poss start）読み込み
# -----------------------------
if (!is.null(model_in)) {
  cat("Loading model from:", model_in, "\n")
  gam_wp <- readRDS(model_in)
} else if (isTRUE(use_oof_protocol_m0_template)) {
  wp_formula_template <- protocol_m0_formula(use_protocol_m0_elo_base, use_late_close_tail_surface)
  cat("Using protocol M0 formula template for season-OOF WP inference; no full-data WP model is fit or loaded.\n")
  cat("OOF template formula:\n")
  cat(paste(deparse(wp_formula_template), collapse = "\n"), "\n")
} else {
  if (!is.null(train_path_override)) {
    train_path <- train_path_override
  } else {
    train_path <- sprintf("data/wp/wp_states_%d_%d_%s.csv.gz", start_season, end_season, seasontype)
    if (!file.exists(train_path)) {
      train_path <- sprintf("data/wp/wp_states_2000_2024_%s.csv.gz", seasontype)
    }
  }
  cat("Reading train:", train_path, "\n")

  dt_train <- fread_maybe_gz(train_path, colClasses = c(game_id="character"))
  if ("season" %in% names(dt_train)) {
    dt_train <- dt_train[season >= start_season & season <= end_season]
  }

  if (!("score_diff" %in% names(dt_train)) && ("score_diff_home" %in% names(dt_train))) {
    dt_train[, score_diff := as.numeric(score_diff_home)]
  }
  # 必須列チェック
  need_train <- c("score_diff", "time_left_game", "OT_flag", "home_possession", "final_home_win")
  miss <- setdiff(need_train, names(dt_train))
  if (length(miss) > 0) stop(paste("Missing in train:", paste(miss, collapse=", ")))

  # 型
  dt_train[, `:=`(
    score_diff     = as.numeric(score_diff),
    time_left_game = as.numeric(time_left_game),
    OT_flag        = as.integer(OT_flag),
    home_possession = as.integer(home_possession),
    final_home_win = as.integer(final_home_win)
  )]

  # 範囲フィルタ（あなたの定義のまま）
  dt_train <- dt_train[
    is.finite(score_diff) & is.finite(time_left_game) & is.finite(final_home_win) & !is.na(home_possession) &
    final_home_win %in% c(0L, 1L) &
    time_left_game >= 0 & time_left_game <= 2880
  ]
  check_home_away_orientation(dt_train, score_col = "score_diff", y_col = "final_home_win")
  if (isTRUE(use_late_close_tail_surface)) {
    add_late_close_tail(dt_train, time_sec = late_tail_time_sec, score_abs = late_tail_score_abs)
    cat(sprintf(
      "Using late-close tail surface: time_left_game <= %.1f and abs(score_diff) <= %.1f (rows=%d).\n",
      late_tail_time_sec, late_tail_score_abs, sum(dt_train$late_close_tail == 1, na.rm = TRUE)
    ))
  }
  if (use_protocol_m0_spec) {
    if (!("start_type" %in% names(dt_train))) dt_train[, start_type := "UNKNOWN"]
    if (isTRUE(use_protocol_m0_elo_base)) {
      coalesce_elo_diff_pregame(dt_train, preferred_k = elo_k_target, context = "train data")
      if (!("elo_diff_pregame" %in% names(dt_train))) {
        stop("Protocol M0 + Elo base requested, but train data has no elo_diff_pregame(_k*) column.")
      }
      dt_train[, elo_diff_pregame := as.numeric(elo_diff_pregame)]
      dt_train <- dt_train[is.finite(elo_diff_pregame)]
      cat("Using protocol M0 base Elo term (elo_diff_pregame).\n")
    }
    ensure_after_off_reb(dt_train, context = "train data")
    dt_train[, start_type_group := build_start_type_group(start_type)]
    dt_train[, era := factor(season_to_era4(as.integer(season)))]
    use_protocol_m0_era_eff <- length(unique(na.omit(dt_train$era))) > 1L
    if (!isTRUE(use_protocol_m0_era_eff)) {
      warning("Protocol M0: era has no variation; skipping era term.")
    }
    cat("Using protocol M0 spec (fit_wp_model.r).\n")
  }

  use_start_type_eff <- FALSE
  if (use_start_type) {
    if ("start_type" %in% names(dt_train)) {
      dt_train[, start_type := trimws(as.character(start_type))]
      dt_train[is.na(start_type) | start_type == "", start_type := "UNKNOWN"]
      dt_train[, start_type := as.factor(start_type)]
      use_start_type_eff <- TRUE
      cat("Using start_type in WP model.\n")
    } else {
      warning("Default start_type usage is enabled, but train data has no `start_type`; fitting without it.")
    }
  }
  use_state_type_eff <- FALSE
  if (use_state_type) {
    if ("state_type" %in% names(dt_train)) {
      dt_train[, state_type := trimws(as.character(state_type))]
      dt_train[is.na(state_type) | state_type == "", state_type := "poss_start"]
      dt_train[, state_type := as.factor(state_type)]
      if (length(unique(dt_train$state_type)) > 1) {
        use_state_type_eff <- TRUE
        dt_train[, state_type3 := factor(as.character(state_type), levels = levels(state_type))]
        cat("Using state_type in WP model.\n")
      } else {
        warning("state_type has no variation in train data; fitting without it.")
      }
    } else {
      warning("Default state_type usage is enabled, but train data has no `state_type`; fitting without it.")
    }
  }
  use_after_off_reb_eff <- FALSE
  if (use_after_off_reb) {
    ensure_after_off_reb(dt_train, context = "train data")
    if (length(unique(dt_train$after_off_reb)) > 1) {
      use_after_off_reb_eff <- TRUE
      cat("Using after_off_reb in WP model.\n")
    } else {
      warning("after_off_reb has no variation in train data; fitting without it.")
    }
  }
  use_era_smooth_eff <- FALSE
  if (use_era_smooth_interaction) {
    if ("season" %in% names(dt_train)) {
      dt_train[, era := season_to_era(as.integer(season))]
      cat("Derived era from season in train data (3-era scheme).\n")
    } else if (!("era" %in% names(dt_train))) {
      warning("No `era` or `season` in train data; fitting without era smooth interaction.")
    }
    if ("era" %in% names(dt_train)) {
      dt_train[, era := trimws(as.character(era))]
      dt_train[is.na(era) | era == "", era := "pace_space_3p"]
      dt_train[, era := as.factor(era)]
      if (length(unique(dt_train$era)) > 1) {
        use_era_smooth_eff <- TRUE
        cat("Using era smooth interaction in WP model.\n")
      } else {
        warning("era has no variation in train data; fitting without era smooth interaction.")
      }
    }
  }
  use_home_possession_by_surface_eff <- FALSE
  if (use_home_possession_by_surface) {
    dt_train[, home_possession_f := factor(
      fifelse(as.integer(home_possession) == 1L, "home", "away"),
      levels = c("away", "home")
    )]
    if (length(unique(dt_train$home_possession_f)) > 1) {
      use_home_possession_by_surface_eff <- TRUE
      cat("Using possession-specific te(score_diff,time_left_game) by home_possession_f.\n")
    } else {
      warning("home_possession_f has no variation; fitting without possession-specific surface.")
    }
  }
  use_start_type_group_smooth_eff <- FALSE
  if (use_start_type_group_smooth) {
    if (!("start_type" %in% names(dt_train))) {
      dt_train[, start_type := "UNKNOWN"]
    }
    dt_train[, start_type_group := start_type_to_group(start_type)]
    if (length(unique(dt_train$start_type_group)) > 1) {
      use_start_type_group_smooth_eff <- TRUE
      cat("Using coarse start_type_group smooth interaction.\n")
    } else {
      warning("start_type_group has no variation; fitting without start_type_group smooth interaction.")
    }
  }
  use_dead_ball_by_surface_eff <- FALSE
  if (use_dead_ball_by_surface) {
    if (!("start_type" %in% names(dt_train))) {
      dt_train[, start_type := "UNKNOWN"]
    }
    dt_train[, dead_ball_indicator := start_type_to_dead_ball_indicator(start_type)]
    if (length(unique(dt_train$dead_ball_indicator)) > 1) {
      use_dead_ball_by_surface_eff <- TRUE
      cat("Using dead_ball_indicator-specific te(score_diff,time_left_game).\n")
    } else {
      warning("dead_ball_indicator has no variation; fitting without dead-ball-specific surface.")
    }
  }

  cat("Training rows:", nrow(dt_train), "\n")

  # -----------------------------
  # 2) 統合GAMフィット
  # -----------------------------
  if (use_protocol_m0_spec) {
    terms <- c(
      "OT_flag",
      "home_possession",
      "s(score_diff, bs='cr', k=15)",
      "s(time_left_game, bs='cr', k=20)",
      "ti(score_diff, time_left_game, bs=c('cr','cr'), k=c(12,12))"
    )
    if (isTRUE(use_protocol_m0_era_eff)) {
      terms <- c("era", terms)
    }
    if (length(unique(na.omit(dt_train$home_possession))) > 1) {
      terms <- c(terms, "ti(score_diff, time_left_game, by=home_possession, bs=c('ts','ts'), k=c(6,6), id='pos')")
    } else {
      warning("Protocol M0: home_possession has no variation; skipping by-home_possession ti term.")
    }
    if (use_state_type_eff) {
      terms <- c(
        terms,
        "state_type",
        "s(score_diff, state_type3, bs='fs', k=6)",
        "s(time_left_game, state_type3, bs='fs', k=8)"
      )
    }
    if ("start_type_group" %in% names(dt_train) && length(unique(na.omit(dt_train$start_type_group))) > 1) {
      terms <- c(terms, "start_type_group", "ti(score_diff, time_left_game, by=start_type_group, bs=c('ts','ts'), k=c(5,5), id='st')")
    } else {
      warning("Protocol M0: start_type_group has no variation; skipping start_type_group terms.")
    }
    if ("after_off_reb" %in% names(dt_train) && length(unique(na.omit(dt_train$after_off_reb))) > 1) {
      terms <- c(terms, "after_off_reb", "ti(score_diff, time_left_game, by=after_off_reb, bs=c('ts','ts'), k=c(5,5), id='orb')")
    } else {
      warning("Protocol M0: after_off_reb has no variation; skipping after_off_reb terms.")
    }
    if (isTRUE(use_protocol_m0_elo_base)) {
      if (!("elo_diff_pregame" %in% names(dt_train))) {
        stop("Protocol M0 + Elo base requested but elo_diff_pregame missing after preprocessing.")
      }
      terms <- c(terms, "s(elo_diff_pregame, bs='cr', k=10)")
    }
    if (isTRUE(use_late_close_tail_surface)) {
      terms <- c(terms, "late_close_tail", "ti(score_diff, time_left_game, by=late_close_tail, bs=c('ts','ts'), k=c(6,6), id='tail')")
    }
  } else {
    terms <- c("OT_flag", "home_possession")
    if (use_home_possession_by_surface_eff) {
      terms <- c(terms, "te(score_diff, time_left_game, by = home_possession_f, k = c(12, 12))")
    } else {
      terms <- c(terms, "te(score_diff, time_left_game, k = c(15, 15))")
    }
    if (use_start_type_eff) {
      if (use_start_type_re) {
        terms <- c(terms, "s(start_type, bs = 're')")
      } else {
        terms <- c(terms, "start_type")
      }
    }
    if (use_state_type_eff) {
      terms <- c(
        terms,
        "state_type",
        "s(score_diff, state_type3, bs='fs', k=6)",
        "s(time_left_game, state_type3, bs='fs', k=8)"
      )
    }
    if (use_after_off_reb_eff) {
      terms <- c(terms, "after_off_reb")
    }
    if (use_era_smooth_eff) {
      terms <- c(terms, "era", "s(score_diff, by = era, k = 8)", "s(time_left_game, by = era, k = 8)")
    }
    if (use_start_type_group_smooth_eff) {
      terms <- c(terms, "start_type_group", "s(score_diff, by = start_type_group, k = 6)", "s(time_left_game, by = start_type_group, k = 6)")
    }
    if (use_dead_ball_by_surface_eff) {
      terms <- c(terms, "dead_ball_indicator", "te(score_diff, time_left_game, by = dead_ball_indicator, k = c(10, 10))")
    }
    if (isTRUE(use_late_close_tail_surface)) {
      terms <- c(terms, "late_close_tail", "ti(score_diff, time_left_game, by=late_close_tail, bs=c('ts','ts'), k=c(6,6), id='tail')")
    }
  }
  wp_formula <- as.formula(paste("final_home_win ~", paste(terms, collapse = " + ")))
  cat("Training formula:\n")
  cat(paste(deparse(wp_formula), collapse = "\n"), "\n")
  gam_wp <- bam(
    wp_formula,
    data     = dt_train,
    family   = binomial(link = "logit"),
    method   = "fREML",
    discrete = TRUE,
    nthreads = bam_nthreads
  )
  if (isTRUE(use_late_close_tail_surface)) {
    attr(gam_wp, "late_tail_time_sec") <- late_tail_time_sec
    attr(gam_wp, "late_tail_score_abs") <- late_tail_score_abs
  }
  print(summary(gam_wp))

  dir.create("models", showWarnings = FALSE)
  saveRDS(gam_wp, file = model_out)
  cat("Saved model:", model_out, "\n")
}

if (is.null(gam_wp)) {
  base_meta <- protocol_m0_template_meta(
    use_protocol_m0_elo_base,
    use_late_close_tail_surface,
    late_tail_time_sec,
    late_tail_score_abs
  )
  cat("Model spec check: using protocol M0 template metadata for OOF refits.\n")
  cat(sprintf("Model direction probe: %s\n", base_meta$model_direction))
} else {
  report_model_spec(gam_wp)
  base_meta <- extract_model_meta(gam_wp)
  cat(sprintf("Model direction probe: %s\n", base_meta$model_direction))
  if (isTRUE(base_meta$model_is_reversed)) {
    warning("Loaded WP model appears to output away-win probability. Auto-converting to home-win via (1-p).")
  }
}
platt_map <- load_platt_map(platt_coef_csv, calibration = platt_calibration)
if (!is.null(platt_map)) {
  if (is.null(gam_wp)) {
    stop("Platt calibration from a fixed base model is not available with --oof-template-protocol-m0-spec.")
  }
  cat(sprintf("Platt calibration: %s (alpha=%.6f, beta=%.6f) from %s\n",
              platt_map$calibration, platt_map$alpha, platt_map$beta, platt_map$path))
}

if (skip_shots) {
  cat("Skipping shot scoring (--skip-shot-scoring).\n")
  quit(save = "no")
}

# -----------------------------
# 3) shot_decision_states をスコアリングして ΔWP を作る（C用）
# -----------------------------
if (!is.null(shot_path_override)) {
  shot_path <- shot_path_override
} else {
  shot_path <- sprintf("data/wp/shot_decision_states_%d_%d_%s.csv.gz", start_season, end_season, seasontype)
  if (!file.exists(shot_path)) {
    shot_path <- sprintf("data/wp/shot_decision_states_2000_2024_%s.csv.gz", seasontype)
  }
}
cat("Reading shots:", shot_path, "\n")

shots <- fread_maybe_gz(shot_path, colClasses = c(GAME_ID="character"))
if ("season" %in% names(shots)) {
  shots <- shots[season >= start_season & season <= end_season]
}

need_shots <- c(
  "GAME_ID","GAME_EVENT_ID","shot_made","final_home_win",
  "before_score_diff","before_time_left_game","before_OT_flag",
  "before_home_possession",
  "next_type","next_is_terminal","next_score_diff","next_time_left_game","next_OT_flag","next_home_possession"
)
miss2 <- setdiff(need_shots, names(shots))
if (length(miss2) > 0) stop(paste("Missing in shots:", paste(miss2, collapse=", ")))

# shot_zone_choice は後段で追加される場合があるため任意扱い
if (!("shot_zone_choice" %in% names(shots))) {
  shots[, shot_zone_choice := NA_character_]
}
if (!("before_start_type" %in% names(shots))) {
  shots[, before_start_type := NA_character_]
}
if (!("next_start_type" %in% names(shots))) {
  shots[, next_start_type := NA_character_]
}
if (!("after_off_reb" %in% names(shots))) {
  if ("shot_sequence" %in% names(shots)) {
    shots[, after_off_reb := derive_after_off_reb_from_shot_sequence(shot_sequence)]
  } else {
    shots[, after_off_reb := 0L]
  }
}

# 型
shots[, `:=`(
  before_score_diff     = as.numeric(before_score_diff),
  before_time_left_game = as.numeric(before_time_left_game),
  before_OT_flag        = as.integer(before_OT_flag),
  before_home_possession = as.integer(before_home_possession),

  next_score_diff       = as.numeric(next_score_diff),
  next_time_left_game   = as.numeric(next_time_left_game),
  next_OT_flag          = as.integer(next_OT_flag),
  next_home_possession  = as.integer(next_home_possession),

  final_home_win        = as.integer(final_home_win),
  next_is_terminal      = as.integer(next_is_terminal)
)]

if (isTRUE(season_oof_predict)) {
  if (!("season" %in% names(shots))) {
    stop("season-OOF prediction requires `season` column in shot states.")
  }
  if (!is.null(train_path_override)) {
    train_path_for_oof <- train_path_override
  } else {
    train_path_for_oof <- sprintf("data/wp/wp_states_%d_%d_%s.csv.gz", start_season, end_season, seasontype)
    if (!file.exists(train_path_for_oof)) {
      train_path_for_oof <- sprintf("data/wp/wp_states_2000_2024_%s.csv.gz", seasontype)
    }
  }
  if (!file.exists(train_path_for_oof)) {
    stop(sprintf("season-OOF prediction requires train data, but not found: %s", train_path_for_oof))
  }
  cat(sprintf("Season-OOF WP inference: ON (train=%s)\n", train_path_for_oof))
  dt_oof <- fread_maybe_gz(train_path_for_oof, colClasses = c(game_id = "character"))
  if (!("score_diff" %in% names(dt_oof)) && ("score_diff_home" %in% names(dt_oof))) {
    dt_oof[, score_diff := as.numeric(score_diff_home)]
  }
  need_oof <- c("season", "score_diff", "time_left_game", "OT_flag", "home_possession", "final_home_win")
  miss_oof <- setdiff(need_oof, names(dt_oof))
  if (length(miss_oof) > 0) stop(paste("Missing in OOF train:", paste(miss_oof, collapse = ", ")))
  dt_oof[, `:=`(
    season = as.integer(season),
    score_diff = as.numeric(score_diff),
    time_left_game = as.numeric(time_left_game),
    OT_flag = as.integer(OT_flag),
    home_possession = as.integer(home_possession),
    final_home_win = as.integer(final_home_win)
  )]
  dt_oof <- dt_oof[
    season >= start_season & season <= end_season &
      is.finite(score_diff) & is.finite(time_left_game) & is.finite(final_home_win) & !is.na(home_possession) &
      final_home_win %in% c(0L, 1L) &
      time_left_game >= 0 & time_left_game <= 2880
  ]
  if (base_meta$uses_elo_diff_pregame) {
    coalesce_elo_diff_pregame(dt_oof, preferred_k = elo_k_target, context = "OOF train data")
    if (!("elo_diff_pregame" %in% names(dt_oof))) {
      stop("Loaded model requires elo_diff_pregame for OOF refits, but OOF train data has no elo_diff_pregame(_k*) column.")
    }
    dt_oof <- dt_oof[!is.na(elo_diff_pregame)]
  }
  if (base_meta$uses_start_type) {
    if (!("start_type" %in% names(dt_oof))) dt_oof[, start_type := "UNKNOWN"]
    dt_oof[, start_type := trimws(as.character(start_type))]
    dt_oof[is.na(start_type) | start_type == "", start_type := "UNKNOWN"]
  }
  if (base_meta$uses_state_type) {
    if (!("state_type" %in% names(dt_oof))) dt_oof[, state_type := "poss_start"]
    dt_oof[, state_type := trimws(as.character(state_type))]
    dt_oof[is.na(state_type) | state_type == "", state_type := "poss_start"]
    if (!is.null(base_meta$state_type_levels) && length(base_meta$state_type_levels) > 0) {
      fallback_state <- if ("poss_start" %in% base_meta$state_type_levels) "poss_start" else base_meta$state_type_levels[[1]]
      dt_oof[!(state_type %in% base_meta$state_type_levels), state_type := fallback_state]
      dt_oof[, state_type := factor(state_type, levels = base_meta$state_type_levels)]
    } else {
      dt_oof[, state_type := as.factor(state_type)]
    }
  }
  if (base_meta$uses_state_type3) {
    if (!("state_type3" %in% names(dt_oof))) {
      if ("state_type" %in% names(dt_oof)) {
        dt_oof[, state_type3 := as.character(state_type)]
      } else {
        dt_oof[, state_type3 := "poss_start"]
      }
    }
    dt_oof[, state_type3 := trimws(as.character(state_type3))]
    dt_oof[is.na(state_type3) | state_type3 == "", state_type3 := "poss_start"]
    if (!is.null(base_meta$state_type3_levels) && length(base_meta$state_type3_levels) > 0) {
      fallback_state3 <- if ("poss_start" %in% base_meta$state_type3_levels) "poss_start" else base_meta$state_type3_levels[[1]]
      dt_oof[!(state_type3 %in% base_meta$state_type3_levels), state_type3 := fallback_state3]
      dt_oof[, state_type3 := factor(state_type3, levels = base_meta$state_type3_levels)]
    } else {
      dt_oof[, state_type3 := as.factor(state_type3)]
    }
  }
  if (base_meta$uses_after_off_reb) ensure_after_off_reb(dt_oof, context = "OOF train data")
  if (base_meta$uses_era) {
    dt_oof[, era := season_to_era_from_levels(as.integer(season), base_meta$era_levels)]
    dt_oof[, era := trimws(as.character(era))]
    dt_oof[is.na(era) | era == "", era := "pace_space_3p"]
    if (!is.null(base_meta$era_levels) && length(base_meta$era_levels) > 0) {
      fallback_era <- if ("pace_space_3p" %in% base_meta$era_levels) "pace_space_3p" else base_meta$era_levels[[1]]
      dt_oof[!(era %in% base_meta$era_levels), era := fallback_era]
      dt_oof[, era := factor(era, levels = base_meta$era_levels)]
    } else {
      dt_oof[, era := as.factor(era)]
    }
  }
  if (base_meta$uses_start_type_group) {
    if (!("start_type" %in% names(dt_oof))) dt_oof[, start_type := "UNKNOWN"]
    dt_oof[, start_type_group := start_type_to_group(start_type)]
  }
  if (base_meta$uses_home_possession_f) {
    lv <- if (!is.null(base_meta$home_possession_f_levels) && length(base_meta$home_possession_f_levels) > 0) base_meta$home_possession_f_levels else c("away", "home")
    dt_oof[, home_possession_f := factor(fifelse(as.integer(home_possession) == 1L, "home", "away"), levels = lv)]
  }
  if (base_meta$uses_dead_ball_indicator) {
    if (!("start_type" %in% names(dt_oof))) dt_oof[, start_type := "UNKNOWN"]
    dt_oof[, dead_ball_indicator := start_type_to_dead_ball_indicator(start_type)]
  }
  if (isTRUE(base_meta$uses_late_close_tail)) {
    add_late_close_tail(dt_oof, time_sec = base_meta$late_tail_time_sec, score_abs = base_meta$late_tail_score_abs)
    cat(sprintf(
      "OOF train uses late_close_tail: time_left_game <= %.1f and abs(score_diff) <= %.1f (rows=%d).\n",
      base_meta$late_tail_time_sec, base_meta$late_tail_score_abs, sum(dt_oof$late_close_tail == 1, na.rm = TRUE)
    ))
  }

  oof_seasons <- sort(unique(as.integer(shots$season)))
  n_oof <- length(oof_seasons)
  oof_formula <- if (is.null(gam_wp)) wp_formula_template else formula(gam_wp)
  oof_worker <- function(i) {
    s <- oof_seasons[[i]]
    fold_started <- Sys.time()
    cat(sprintf("  [OOF] start season=%d (%d/%d) at %s\n", s, i, n_oof, format(fold_started, "%Y-%m-%d %H:%M:%S")))
    flush.console()
    shots_s <- shots[season == s]
    dt_fit <- dt_oof[season != s]
    if (nrow(dt_fit) < 1000L) {
      if (is.null(gam_wp)) {
        stop(sprintf("OOF training rows too small for season=%d (n=%d), and no base model is available in template-only mode.", s, nrow(dt_fit)))
      }
      warning(sprintf("OOF training rows too small for season=%d (n=%d). Using base model.", s, nrow(dt_fit)))
      scored <- score_shots_with_model(shots_s, gam_wp)$scored
      cat(sprintf("  [OOF] done season=%d rows=%d elapsed=%.1f min (base model fallback)\n",
                  s, nrow(scored), as.numeric(difftime(Sys.time(), fold_started, units = "mins"))))
      flush.console()
      return(scored)
    }
    fit_oof <- bam(
      oof_formula,
      data = dt_fit,
      family = binomial(link = "logit"),
      method = "fREML",
      discrete = TRUE,
      nthreads = bam_nthreads
    )
    scored <- score_shots_with_model(shots_s, fit_oof)$scored
    cat(sprintf("  [OOF] done season=%d rows=%d elapsed=%.1f min\n",
                s, nrow(scored), as.numeric(difftime(Sys.time(), fold_started, units = "mins"))))
    flush.console()
    scored
  }

  if (season_oof_jobs > 1L) {
    if (.Platform$OS.type != "unix") {
      warning("--season-oof-jobs > 1 requires Unix fork support; falling back to sequential OOF scoring.")
      season_oof_jobs <- 1L
    }
  }
  cat(sprintf("Season-OOF scoring config: jobs=%d bam_nthreads=%d seasons=%d\n", season_oof_jobs, bam_nthreads, n_oof))
  if (season_oof_jobs > 1L) {
    scored_parts <- parallel::mclapply(seq_along(oof_seasons), oof_worker, mc.cores = season_oof_jobs)
  } else {
    scored_parts <- lapply(seq_along(oof_seasons), oof_worker)
  }
  shots <- rbindlist(scored_parts, use.names = TRUE, fill = TRUE)
} else {
  if (is.null(gam_wp)) {
    stop("Single-model prediction requires a fitted or loaded model; disable --oof-template-protocol-m0-spec.")
  }
  cat("Season-OOF WP inference: OFF (single model prediction)\n")
  shots <- score_shots_with_model(shots, gam_wp)$scored
}

if (isTRUE(season_oof_platt) && !is.null(platt_map)) {
  stop("Use either --season-oof-platt or --platt-coef-csv, not both.")
}

if (isTRUE(season_oof_platt)) {
  if (!("season" %in% names(shots))) {
    stop("season-OOF Platt requires `season` in shot_decision_states input.")
  }
  if (isTRUE(season_oof_predict)) {
    warning("season-OOF Platt is being applied after season-OOF WP inference. This mixes fold-specific base predictions with a fixed-base Platt fit path unless you intentionally matched them.")
  }
  if (is.null(gam_wp)) {
    stop("Season-OOF Platt is not available with --oof-template-protocol-m0-spec because no fixed base model is loaded.")
  }
  if (is.null(season_oof_platt_fit_path) || identical(season_oof_platt_fit_path, "")) {
    season_oof_platt_fit_path <- sprintf("data/wp/wp_states_%d_%d_po.csv.gz", start_season, end_season)
  }
  if (!file.exists(season_oof_platt_fit_path)) {
    stop(sprintf("season-OOF Platt fit path not found: %s", season_oof_platt_fit_path))
  }
  cat(sprintf("Season-OOF Platt: ON (fit=%s subset=%s)\n", season_oof_platt_fit_path, season_oof_platt_subset))
  cal_dt <- fread_maybe_gz(season_oof_platt_fit_path, colClasses = c(game_id = "character"))
  if (!("season" %in% names(cal_dt))) {
    stop("season-OOF Platt fit data requires `season` column.")
  }
  if ("season" %in% names(cal_dt)) {
    cal_dt <- cal_dt[season >= start_season & season <= end_season]
  }
  if (!("final_home_win" %in% names(cal_dt))) {
    stop("season-OOF Platt fit data requires final_home_win.")
  }
  cal_dt <- score_states_with_model(cal_dt, gam_wp, context = "season-OOF Platt fit data")
  cal_dt[, final_home_win := as.integer(final_home_win)]

  holdout_seasons <- sort(unique(as.integer(shots$season)))
  coef_rows <- vector("list", length(holdout_seasons))
  for (i in seq_along(holdout_seasons)) {
    s <- holdout_seasons[[i]]
    idx_shots_s <- which(as.integer(shots$season) == s)
    if (length(idx_shots_s) == 0L) next
    cal_train <- cal_dt[as.integer(season) != s]
    cal_mask <- season_oof_platt_mask(
      cal_train,
      subset = season_oof_platt_subset,
      clutch_time_sec = season_oof_platt_clutch_time_sec,
      clutch_score_diff = season_oof_platt_clutch_score_diff
    )
    cal_fit <- cal_train[cal_mask]
    ab <- fit_platt_glm_ab(cal_fit$final_home_win, cal_fit$wp_hat_raw)
    if (is.null(ab)) {
      warning(sprintf("season-OOF Platt fit failed for season=%d (subset n=%d). Leaving raw wp for this season.", s, nrow(cal_fit)))
      coef_rows[[i]] <- data.table(
        holdout_season = as.integer(s),
        alpha = NA_real_, beta = NA_real_,
        n_train_all = as.integer(nrow(cal_train)),
        n_train_subset = as.integer(nrow(cal_fit)),
        subset = as.character(season_oof_platt_subset)
      )
      next
    }
    shots[idx_shots_s, wp_before_raw := as.numeric(wp_before)]
    shots[idx_shots_s, wp_next_raw := as.numeric(wp_next)]
    shots[idx_shots_s, wp_before := apply_platt_prob(wp_before_raw, ab$alpha, ab$beta)]
    shots[idx_shots_s, wp_next := apply_platt_prob(wp_next_raw, ab$alpha, ab$beta)]
    shots[idx_shots_s, delta_wp := wp_next - wp_before]
    coef_rows[[i]] <- data.table(
      holdout_season = as.integer(s),
      alpha = as.numeric(ab$alpha), beta = as.numeric(ab$beta),
      n_train_all = as.integer(nrow(cal_train)),
      n_train_subset = as.integer(nrow(cal_fit)),
      subset = as.character(season_oof_platt_subset)
    )
    cat(sprintf("  [OOF-Platt] season=%d alpha=%.6f beta=%.6f n_subset=%d\n", s, ab$alpha, ab$beta, nrow(cal_fit)))
  }
  if (length(coef_rows) > 0L) {
    coef_dt <- rbindlist(coef_rows, use.names = TRUE, fill = TRUE)
    if (!identical(season_oof_platt_coef_out, "")) {
      dir.create(dirname(season_oof_platt_coef_out), recursive = TRUE, showWarnings = FALSE)
      fwrite(coef_dt, season_oof_platt_coef_out)
      cat(sprintf("Saved season-OOF Platt fold coefficients: %s\n", season_oof_platt_coef_out))
    }
  }
}

if (!is.null(platt_map)) {
  shots[, wp_before_raw := as.numeric(wp_before)]
  shots[, wp_next_raw := as.numeric(wp_next)]
  shots[, wp_before := apply_platt_prob(wp_before_raw, platt_map$alpha, platt_map$beta)]
  shots[, wp_next := apply_platt_prob(wp_next_raw, platt_map$alpha, platt_map$beta)]
  shots[, delta_wp := wp_next - wp_before]
  cat("Applied Platt calibration to wp_before/wp_next; recomputed delta_wp.\n")
}

na_delta <- sum(is.na(shots$delta_wp))
if (na_delta > 0) warning(paste("delta_wp has NAs:", na_delta))

# ------------------------------------------------------------
# 出力（2系統）
#   1) full: 監査・デバッグ用（DMLには使わない）
#   2) dml : DML投入用（post-treatment列を排除）
#   3) wp  : ショットパネル用（with_wp）
# ------------------------------------------------------------

# 1) 監査用フル出力（※DMLでは絶対に使わない）
out_full <- shots[, .(
  GAME_ID, GAME_EVENT_ID,
  shot_zone_choice,

  # 結果・分岐（post-treatment）※監査目的のみ
  shot_made, next_type, next_is_terminal,

  # before / next の状態（next_* は post-treatment）※監査目的のみ
  after_off_reb,
  before_score_diff, before_time_left_game, before_OT_flag, before_home_possession, before_start_type,
  next_score_diff,   next_time_left_game,   next_OT_flag,   next_home_possession, next_start_type,

  # WPと差分（delta_wpが最終アウトカム）
  wp_before, wp_next, delta_wp,

  # 終端処理チェック用（post-treatment）
  final_home_win
)]
if ("shot_sequence" %in% names(shots)) out_full[, shot_sequence := shots$shot_sequence]

full_path <- sprintf("data/wp/shot_decision_states_%d_%d_%s_with_wp_full.csv.gz",
                     start_season, end_season, seasontype)
if (!is.null(full_out_override) && !identical(full_out_override, "")) {
  full_path <- full_out_override
}
fwrite(out_full, full_path, compress = "gzip")
cat("Saved FULL (debug) output to:", full_path, "rows=", nrow(out_full), "\n")

# 2) ショットパネル用（with_wp）
out_wp <- shots[, .(
  GAME_ID, GAME_EVENT_ID,
  shot_zone_choice,
  shot_made, next_type, next_is_terminal,
  after_off_reb,
  before_score_diff, before_time_left_game, before_OT_flag, before_home_possession, before_start_type,
  next_score_diff,   next_time_left_game,   next_OT_flag,   next_home_possession, next_start_type,
  wp_before, wp_next, delta_wp,
  final_home_win
)]
if ("shot_sequence" %in% names(shots)) out_wp[, shot_sequence := shots$shot_sequence]

# もし season / seasontype が shots に存在するなら保持（識別子として安全）
if ("season" %in% names(shots)) out_wp[, season := shots$season]
if ("seasontype" %in% names(shots)) out_wp[, seasontype := shots$seasontype]

wp_path <- sprintf("data/wp/shot_decision_states_%d_%d_%s_with_wp.csv.gz",
                   start_season, end_season, seasontype)
if (!is.null(wp_out_override) && !identical(wp_out_override, "")) {
  wp_path <- wp_out_override
}
fwrite(out_wp, wp_path, compress = "gzip")
cat("Saved WP output to:", wp_path, "rows=", nrow(out_wp), "\n")


# 3) DML投入用（post-treatment列を除外）
#    - treatment: shot_zone_choice
#    - outcome  : delta_wp = WP(next decision) - WP(shot before)
#    - covariates (C): “shot before” の状態のみ（pre-treatment）
#    ここに next_* / shot_made / next_type / wp_next などを入れない
out_dml <- shots[, .(
  GAME_ID, GAME_EVENT_ID,
  shot_zone_choice,
  delta_wp,

  # pre-treatment covariates（shot before state）
  before_score_diff,
  before_time_left_game,
  before_OT_flag,
  before_home_possession,
  start_type = before_start_type
)]
if ("shot_sequence" %in% names(shots)) out_dml[, shot_sequence := shots$shot_sequence]

# もし season / seasontype が shots に存在するなら保持（識別子として安全）
if ("season" %in% names(shots)) out_dml[, season := shots$season]
if ("seasontype" %in% names(shots)) out_dml[, seasontype := shots$seasontype]

dml_path <- sprintf("data/wp/shot_decision_panel_%d_%d_%s_dml.csv.gz",
                    start_season, end_season, seasontype)
if (!is.null(dml_out_override) && !identical(dml_out_override, "")) {
  dml_path <- dml_out_override
}
fwrite(out_dml, dml_path, compress = "gzip")
cat("Saved DML-ready output to:", dml_path, "rows=", nrow(out_dml), "\n")

# 念のため注意喚起
cat("\n[NOTE] Use ONLY the DML-ready file for DML/DR:\n  ", dml_path, "\n")
cat("[NOTE] The FULL file contains post-treatment columns and is for debugging only.\n\n")
