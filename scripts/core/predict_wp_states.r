#!/usr/bin/env Rscript
# Apply a trained WP model to possession-start states and save wp_hat.

suppressPackageStartupMessages({
  library(data.table)
  library(mgcv)
})

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag, default = NULL) {
  if (!(flag %in% args)) return(default)
  idx <- match(flag, args)
  if (is.na(idx) || idx == length(args)) return(default)
  args[[idx + 1]]
}
derive_after_off_reb_from_start_type <- function(x) {
  s <- tolower(trimws(as.character(x)))
  s[is.na(s)] <- ""
  as.integer(grepl("(miss|block)$", s))
}
season_to_era <- function(season) {
  fifelse(season <= 2003, "transition_pre2004",
    fifelse(season <= 2013, "post_handcheck_pre3p",
      "pace_space_3p"))
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

model_path <- get_arg("--model-path", "models/wp_gam_leagueavg_unified_ot.rds")
in_path    <- get_arg("--in-path", "data/wp/wp_states_2000_2024_rs.csv.gz")
out_path   <- get_arg("--out-path", "data/wp/wp_states_2000_2024_rs_with_wp.csv.gz")
min_season <- get_arg("--min-season", NULL)
max_season <- get_arg("--max-season", NULL)

cat("Loading trained model from:", model_path, "\n")
gam_wp <- readRDS(model_path)
model_uses_start_type <- "start_type" %in% all.vars(formula(gam_wp))
model_uses_state_type <- "state_type" %in% all.vars(formula(gam_wp))
model_uses_state_type3 <- "state_type3" %in% all.vars(formula(gam_wp))
model_uses_after_off_reb <- "after_off_reb" %in% all.vars(formula(gam_wp))
model_uses_era <- "era" %in% all.vars(formula(gam_wp))
model_uses_start_type_group <- "start_type_group" %in% all.vars(formula(gam_wp))
model_uses_home_possession_f <- "home_possession_f" %in% all.vars(formula(gam_wp))
model_uses_dead_ball_indicator <- "dead_ball_indicator" %in% all.vars(formula(gam_wp))
start_levels <- NULL
state_type_levels <- NULL
state_type3_levels <- NULL
era_levels <- NULL
start_group_levels <- NULL
home_possession_f_levels <- NULL
if (model_uses_start_type) {
  cat("Loaded WP model expects start_type.\n")
  if (!is.null(gam_wp$model) && "start_type" %in% names(gam_wp$model)) {
    start_levels <- levels(gam_wp$model$start_type)
  }
}
if (model_uses_state_type) {
  cat("Loaded WP model expects state_type.\n")
  if (!is.null(gam_wp$model) && "state_type" %in% names(gam_wp$model)) {
    state_type_levels <- levels(gam_wp$model$state_type)
  }
}
if (model_uses_state_type3) {
  cat("Loaded WP model expects state_type3.\n")
  if (!is.null(gam_wp$model) && "state_type3" %in% names(gam_wp$model)) {
    state_type3_levels <- levels(gam_wp$model$state_type3)
  }
}
if (model_uses_after_off_reb) {
  cat("Loaded WP model expects after_off_reb.\n")
}
if (model_uses_era) {
  cat("Loaded WP model expects era.\n")
  if (!is.null(gam_wp$model) && "era" %in% names(gam_wp$model)) {
    era_levels <- levels(gam_wp$model$era)
  }
}
if (model_uses_start_type_group) {
  cat("Loaded WP model expects start_type_group.\n")
  if (!is.null(gam_wp$model) && "start_type_group" %in% names(gam_wp$model)) {
    start_group_levels <- levels(gam_wp$model$start_type_group)
  }
}
if (model_uses_home_possession_f) {
  cat("Loaded WP model expects home_possession_f.\n")
  if (!is.null(gam_wp$model) && "home_possession_f" %in% names(gam_wp$model)) {
    home_possession_f_levels <- levels(gam_wp$model$home_possession_f)
  }
}
if (model_uses_dead_ball_indicator) {
  cat("Loaded WP model expects dead_ball_indicator.\n")
}

cat("Reading data from:", in_path, "\n")
dt <- fread(in_path, colClasses = c(game_id = "character"))
cat("Rows:", nrow(dt), "\n")

if (!is.null(min_season) && "season" %in% names(dt)) {
  dt <- dt[season >= as.integer(min_season)]
}
if (!is.null(max_season) && "season" %in% names(dt)) {
  dt <- dt[season <= as.integer(max_season)]
}
cat("Rows after season filter:", nrow(dt), "\n")

dt[, `:=`(
  score_diff = as.numeric(score_diff),
  time_left_game = as.numeric(time_left_game),
  final_home_win = as.integer(final_home_win),
  OT_flag = as.integer(OT_flag),
  home_possession = as.integer(home_possession)
)]

if (model_uses_start_type) {
  if (!("start_type" %in% names(dt))) {
    dt[, start_type := "UNKNOWN"]
  }
  dt[, start_type := trimws(as.character(start_type))]
  dt[is.na(start_type) | start_type == "", start_type := "UNKNOWN"]
  if (!is.null(start_levels) && length(start_levels) > 0) {
    fallback_level <- if ("UNKNOWN" %in% start_levels) "UNKNOWN" else start_levels[[1]]
    dt[!(start_type %in% start_levels), start_type := fallback_level]
    dt[, start_type := factor(start_type, levels = start_levels)]
  } else {
    dt[, start_type := as.factor(start_type)]
  }
}
if (model_uses_state_type) {
  if (!("state_type" %in% names(dt))) {
    dt[, state_type := "poss_start"]
  }
  dt[, state_type := trimws(as.character(state_type))]
  dt[is.na(state_type) | state_type == "", state_type := "poss_start"]
  if (!is.null(state_type_levels) && length(state_type_levels) > 0) {
    fallback_state <- if ("poss_start" %in% state_type_levels) "poss_start" else state_type_levels[[1]]
    dt[!(state_type %in% state_type_levels), state_type := fallback_state]
    dt[, state_type := factor(state_type, levels = state_type_levels)]
  } else {
    dt[, state_type := as.factor(state_type)]
  }
}
if (model_uses_state_type3) {
  if (!("state_type3" %in% names(dt))) {
    if ("state_type" %in% names(dt)) {
      dt[, state_type3 := as.character(state_type)]
    } else {
      dt[, state_type3 := "poss_start"]
    }
  }
  dt[, state_type3 := trimws(as.character(state_type3))]
  dt[is.na(state_type3) | state_type3 == "", state_type3 := "poss_start"]
  if (!is.null(state_type3_levels) && length(state_type3_levels) > 0) {
    fallback_state3 <- if ("poss_start" %in% state_type3_levels) "poss_start" else state_type3_levels[[1]]
    dt[!(state_type3 %in% state_type3_levels), state_type3 := fallback_state3]
    dt[, state_type3 := factor(state_type3, levels = state_type3_levels)]
  } else {
    dt[, state_type3 := as.factor(state_type3)]
  }
}
if (model_uses_after_off_reb) {
  if (!("after_off_reb" %in% names(dt))) {
    if ("start_type" %in% names(dt)) {
      dt[, after_off_reb := derive_after_off_reb_from_start_type(start_type)]
    } else {
      dt[, after_off_reb := 0L]
    }
  }
  dt[, after_off_reb := as.integer(after_off_reb)]
  dt[is.na(after_off_reb), after_off_reb := 0L]
}
if (model_uses_era) {
  if ("season" %in% names(dt)) {
    dt[, era := season_to_era(as.integer(season))]
  } else if (!("era" %in% names(dt))) {
    dt[, era := "pace_space_3p"]
  }
  dt[, era := trimws(as.character(era))]
  dt[is.na(era) | era == "", era := "pace_space_3p"]
  if (!is.null(era_levels) && length(era_levels) > 0) {
    fallback_era <- if ("pace_space_3p" %in% era_levels) "pace_space_3p" else era_levels[[1]]
    dt[!(era %in% era_levels), era := fallback_era]
    dt[, era := factor(era, levels = era_levels)]
  } else {
    dt[, era := as.factor(era)]
  }
}
if (model_uses_start_type_group) {
  if (!("start_type_group" %in% names(dt))) {
    if ("start_type" %in% names(dt)) {
      dt[, start_type_group := start_type_to_group(start_type)]
    } else {
      dt[, start_type_group := factor("dead_ball", levels = c("live_ball", "dead_ball"))]
    }
  }
  if (!is.null(start_group_levels) && length(start_group_levels) > 0) {
    dt[, start_type_group := factor(as.character(start_type_group), levels = start_group_levels)]
    fallback_group <- if ("dead_ball" %in% start_group_levels) "dead_ball" else if ("unknown" %in% start_group_levels) "unknown" else start_group_levels[[1]]
    dt[is.na(start_type_group), start_type_group := fallback_group]
    dt[, start_type_group := factor(as.character(start_type_group), levels = start_group_levels)]
  } else {
    dt[, start_type_group := as.factor(start_type_group)]
  }
}
if (model_uses_home_possession_f) {
  if (!("home_possession_f" %in% names(dt))) {
    dt[, home_possession_f := factor(
      fifelse(as.integer(home_possession) == 1L, "home", "away"),
      levels = if (!is.null(home_possession_f_levels) && length(home_possession_f_levels) > 0) home_possession_f_levels else c("away", "home")
    )]
  } else {
    dt[, home_possession_f := as.character(home_possession_f)]
    dt[is.na(home_possession_f) | home_possession_f == "", home_possession_f := "away"]
    dt[, home_possession_f := factor(
      home_possession_f,
      levels = if (!is.null(home_possession_f_levels) && length(home_possession_f_levels) > 0) home_possession_f_levels else c("away", "home")
    )]
  }
}
if (model_uses_dead_ball_indicator) {
  if (!("dead_ball_indicator" %in% names(dt))) {
    if ("start_type" %in% names(dt)) {
      dt[, dead_ball_indicator := start_type_to_dead_ball_indicator(start_type)]
    } else {
      dt[, dead_ball_indicator := 0L]
    }
  }
  dt[, dead_ball_indicator := as.integer(dead_ball_indicator)]
  dt[is.na(dead_ball_indicator), dead_ball_indicator := 0L]
}

dt <- dt[
  !is.na(score_diff) & !is.na(time_left_game) & !is.na(final_home_win) & !is.na(home_possession) &
    time_left_game >= 0 & time_left_game <= 2880
]
cat("Valid rows after cleaning:", nrow(dt), "\n")

cat("Predicting wp_hat...\n")
dt[, wp_hat := predict(gam_wp, newdata = dt, type = "response")]

fwrite(dt, out_path)
cat("Saved:", out_path, "\n")
