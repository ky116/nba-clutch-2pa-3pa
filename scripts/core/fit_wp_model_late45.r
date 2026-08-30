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
coalesce_elo_diff_pregame <- function(dt, preferred_k = NA_real_, context = "data") {
  if ("elo_diff_pregame" %in% names(dt)) {
    dt[, elo_diff_pregame := as.numeric(elo_diff_pregame)]
    return(invisible(TRUE))
  }
  k_label <- function(k) {
    if (!is.finite(k)) return(NA_character_)
    if (abs(k - round(k)) < 1e-9) return(as.character(as.integer(round(k))))
    gsub("\\.", "p", sub("0+$", "", sub("\\.$", "", sprintf("%.6f", k))))
  }
  cands <- character()
  if (is.finite(preferred_k)) cands <- c(cands, sprintf("elo_diff_pregame_k%s", k_label(preferred_k)))
  cands <- c(cands, grep("^elo_diff_pregame_k", names(dt), value = TRUE))
  cands <- unique(cands[cands %in% names(dt)])
  if (length(cands) == 0L) return(invisible(FALSE))
  dt[, elo_diff_pregame := as.numeric(get(cands[[1]]))]
  cat(sprintf("Mapped %s -> elo_diff_pregame in %s.\n", cands[[1]], context))
  invisible(TRUE)
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
add_late_close_tail <- function(dt, time_sec = 45, score_abs = 7) {
  dt[, late_close_tail := as.numeric(
    is.finite(time_left_game) &
      is.finite(score_diff) &
      time_left_game <= as.numeric(time_sec) &
      abs(score_diff) <= as.numeric(score_abs)
  )]
  dt[, late_time_band := fifelse(
    late_close_tail == 1 & time_left_game <= 30,
    "t0_30",
    fifelse(late_close_tail == 1 & time_left_game <= 45, "t30_45", "other")
  )]
  dt[, late_time_band := factor(late_time_band, levels = c("other", "t0_30", "t30_45"))]
  dt[, late_score_t0_30 := fifelse(late_time_band == "t0_30", score_diff, 0.0)]
  dt[, late_score_t30_45 := fifelse(late_time_band == "t30_45", score_diff, 0.0)]
}
late_tail_terms <- function(variant) {
  if (identical(variant, "none")) return(character())
  if (identical(variant, "main")) return("late_close_tail")
  if (identical(variant, "band")) return("late_time_band")
  if (identical(variant, "score-band")) return(c("late_time_band", "late_score_t0_30", "late_score_t30_45"))
  if (identical(variant, "surface")) {
    return(c("late_close_tail", "ti(score_diff, time_left_game, by=late_close_tail, bs=c('ts','ts'), k=c(6,6), id='tail')"))
  }
  stop(sprintf("Unknown late-tail variant: %s", variant))
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
  if ("late_time_band" %in% fm_vars) {
    d[, late_time_band := factor("other", levels = c("other", "t0_30", "t30_45"))]
  }
  if ("late_score_t0_30" %in% fm_vars) d[, late_score_t0_30 := 0.0]
  if ("late_score_t30_45" %in% fm_vars) d[, late_score_t30_45 := 0.0]
  p <- suppressWarnings(as.numeric(predict(model_obj, newdata = d, type = "response")))
  if (length(p) != 2 || any(!is.finite(p))) return("unknown")
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

start_season <- as.integer(get_arg("--start-season", "2000"))
end_season   <- as.integer(get_arg("--end-season", "2024"))
seasontype   <- get_arg("--seasontype", "rs")
train_path_override <- get_arg("--train-path", NULL)
model_out    <- get_arg("--model-out", "models/wp_gam_leagueavg_unified_ot.rds")
use_start_type <- !has_flag("--no-start-type")
use_state_type <- !has_flag("--no-state-type")
use_after_off_reb <- !has_flag("--no-after-off-reb")
use_era_smooth_interaction <- !has_flag("--no-era-smooth-interaction")
use_start_type_re <- has_flag("--start-type-re")
use_home_possession_by_surface <- has_flag("--home-possession-by-surface")
use_start_type_group_smooth <- has_flag("--start-type-group-smooth")
use_dead_ball_by_surface <- has_flag("--dead-ball-by-surface")
use_era_te_by_surface <- has_flag("--era-te-by-surface")
use_protocol_m0_spec <- has_flag("--protocol-m0-spec")
use_elo_pregame <- has_flag("--use-elo-pregame")
elo_k_target <- suppressWarnings(as.numeric(get_arg("--elo-k-target", "NA")))
late_tail_variant <- tolower(get_arg("--late-tail-variant", "none"))
if (has_flag("--late-close-tail-surface")) late_tail_variant <- "surface"
if (!(late_tail_variant %in% c("none", "main", "band", "score-band", "surface"))) {
  stop("--late-tail-variant must be one of: none, main, band, score-band, surface")
}
use_late_close_tail_surface <- !identical(late_tail_variant, "none")
late_tail_time_sec <- suppressWarnings(as.numeric(get_arg("--late-tail-time-sec", "45")))
late_tail_score_abs <- suppressWarnings(as.numeric(get_arg("--late-tail-score-abs", "7")))
if (late_tail_variant %in% c("band", "score-band") &&
    (!is.finite(late_tail_time_sec) || abs(late_tail_time_sec - 45) > 1e-9)) {
  stop("late-tail variants 'band' and 'score-band' are defined for 0--30 and 30--45 s; use --late-tail-time-sec 45.")
}

if (!is.null(train_path_override)) {
  train_path <- train_path_override
} else {
  train_path <- sprintf("data/wp/wp_states_%d_%d_%s.csv.gz", start_season, end_season, seasontype)
  if (!file.exists(train_path)) {
    train_path <- sprintf("data/wp/wp_states_2000_2024_%s.csv.gz", seasontype)
  }
}

cat("Reading train:", train_path, "\n")
dt_train <- fread_maybe_gz(train_path, colClasses = c(game_id = "character"))
if ("season" %in% names(dt_train)) {
  dt_train <- dt_train[season >= start_season & season <= end_season]
}

if (!("score_diff" %in% names(dt_train)) && ("score_diff_home" %in% names(dt_train))) {
  dt_train[, score_diff := as.numeric(score_diff_home)]
}
need_train <- c("score_diff", "time_left_game", "OT_flag", "home_possession", "final_home_win")
miss <- setdiff(need_train, names(dt_train))
if (length(miss) > 0) stop(paste("Missing in train:", paste(miss, collapse = ", ")))
if ("score_diff_home" %in% names(dt_train)) {
  dt_train[, score_diff := as.numeric(score_diff_home)]
}

dt_train[, `:=`(
  score_diff = as.numeric(score_diff),
  time_left_game = as.numeric(time_left_game),
  OT_flag = as.integer(OT_flag),
  home_possession = as.integer(home_possession),
  final_home_win = as.integer(final_home_win)
)]
dt_train <- dt_train[
  !is.na(score_diff) & !is.na(time_left_game) & !is.na(final_home_win) & !is.na(home_possession) &
    time_left_game >= 0 & time_left_game <= 2880
]
use_elo_pregame_eff <- FALSE
if (use_elo_pregame) {
  ok_elo <- coalesce_elo_diff_pregame(dt_train, preferred_k = elo_k_target, context = "train data")
  if (isTRUE(ok_elo) && "elo_diff_pregame" %in% names(dt_train)) {
    dt_train <- dt_train[is.finite(elo_diff_pregame)]
    if (nrow(dt_train) > 0L && length(unique(na.omit(dt_train$elo_diff_pregame))) > 1L) {
      use_elo_pregame_eff <- TRUE
      cat("Using elo_diff_pregame in WP model.\n")
    } else {
      warning("elo_diff_pregame has no variation; fitting without Elo term.")
    }
  } else {
    warning("`--use-elo-pregame` specified but no elo_diff_pregame(_k*) column found; fitting without Elo term.")
  }
}
check_home_away_orientation(dt_train, score_col = "score_diff", y_col = "final_home_win")
if (use_late_close_tail_surface) {
  add_late_close_tail(dt_train, time_sec = late_tail_time_sec, score_abs = late_tail_score_abs)
  cat(sprintf(
    "Using late-tail variant=%s: time_left_game <= %.1f and abs(score_diff) <= %.1f (rows=%d).\n",
    late_tail_variant, late_tail_time_sec, late_tail_score_abs, sum(dt_train$late_close_tail == 1, na.rm = TRUE)
  ))
}

if (use_protocol_m0_spec) {
  if (!("start_type" %in% names(dt_train))) dt_train[, start_type := "UNKNOWN"]
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
    dt_train[, state_type := factor(state_type)]
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
  if (isTRUE(use_elo_pregame_eff)) {
    terms <- c(terms, "s(elo_diff_pregame, bs='cr', k=10)")
  }
  terms <- c(terms, late_tail_terms(late_tail_variant))
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
  if (isTRUE(use_elo_pregame_eff)) {
    terms <- c(terms, "s(elo_diff_pregame, bs='cr', k=10)")
  }
  if (use_era_smooth_eff) {
    if (use_era_te_by_surface) {
      terms <- c(terms, "era", "te(score_diff, time_left_game, by = era, k = c(8, 8))")
    } else {
      terms <- c(terms, "era", "s(score_diff, by = era, k = 8)", "s(time_left_game, by = era, k = 8)")
    }
  }
  if (use_start_type_group_smooth_eff) {
    terms <- c(terms, "start_type_group", "s(score_diff, by = start_type_group, k = 6)", "s(time_left_game, by = start_type_group, k = 6)")
  }
  if (use_dead_ball_by_surface_eff) {
    terms <- c(terms, "dead_ball_indicator", "te(score_diff, time_left_game, by = dead_ball_indicator, k = c(10, 10))")
  }
  terms <- c(terms, late_tail_terms(late_tail_variant))
}
wp_formula <- as.formula(paste("final_home_win ~", paste(terms, collapse = " + ")))
cat("Training formula:\n")
cat(paste(deparse(wp_formula), collapse = "\n"), "\n")

gam_wp <- bam(
  wp_formula,
  data = dt_train,
  family = binomial(link = "logit"),
  method = "fREML",
  discrete = TRUE
)
if (isTRUE(use_late_close_tail_surface)) {
  attr(gam_wp, "late_tail_time_sec") <- late_tail_time_sec
  attr(gam_wp, "late_tail_score_abs") <- late_tail_score_abs
  attr(gam_wp, "late_tail_variant") <- late_tail_variant
}
print(summary(gam_wp))
report_model_spec(gam_wp)
dir_probe <- detect_model_direction(gam_wp)
cat(sprintf("Model direction probe: %s\n", dir_probe))
if (identical(dir_probe, "away")) {
  stop("Fitted model appears to output away-win probability. Aborting save to keep home-away standard.")
}

dir.create("models", showWarnings = FALSE, recursive = TRUE)
saveRDS(gam_wp, file = model_out)
cat("Saved model:", model_out, "\n")
