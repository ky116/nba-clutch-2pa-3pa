#!/usr/bin/env sh
set -eu

START_SEASON="${START_SEASON:-2000}"
END_SEASON="${END_SEASON:-2024}"
SEASONTYPE="${SEASONTYPE:-rs}"
LOGDIR="${LOGDIR:-logs}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "$LOGDIR" data/analysis

STAMP="$(date +%Y%m%d_%H%M%S)"

TEAM_STATS_OUT="data/analysis/team_shot_stats_${START_SEASON}_${END_SEASON}.parquet"
TEAM_FOULS_OUT="data/analysis/cumulative_team_fouls_${START_SEASON}_${END_SEASON}_${SEASONTYPE}.parquet"

TEAM_STATS_LOG="$LOGDIR/nohup_construct_team_shot_stats_${START_SEASON}_${END_SEASON}_${SEASONTYPE}_${STAMP}.out"
TEAM_FOULS_LOG="$LOGDIR/nohup_construct_cumulative_team_foul_${START_SEASON}_${END_SEASON}_${SEASONTYPE}_${STAMP}.out"

TEAM_STATS_PIDF="$LOGDIR/nohup_construct_team_shot_stats_${START_SEASON}_${END_SEASON}_${SEASONTYPE}_${STAMP}.pid"
TEAM_FOULS_PIDF="$LOGDIR/nohup_construct_cumulative_team_foul_${START_SEASON}_${END_SEASON}_${SEASONTYPE}_${STAMP}.pid"

nohup "$PYTHON_BIN" scripts/core/construct_team_shot_stats.py \
  --start-season "$START_SEASON" \
  --end-season "$END_SEASON" \
  --seasontype "$SEASONTYPE" \
  --out "$TEAM_STATS_OUT" \
  > "$TEAM_STATS_LOG" 2>&1 &
echo $! > "$TEAM_STATS_PIDF"

nohup "$PYTHON_BIN" scripts/core/construct_cumulative_team_foul.py \
  --start-season "$START_SEASON" \
  --end-season "$END_SEASON" \
  --seasontype "$SEASONTYPE" \
  --output "$TEAM_FOULS_OUT" \
  > "$TEAM_FOULS_LOG" 2>&1 &
echo $! > "$TEAM_FOULS_PIDF"

echo "started team_shot_stats pid=$(cat "$TEAM_STATS_PIDF") log=$TEAM_STATS_LOG"
echo "started cumulative_team_foul pid=$(cat "$TEAM_FOULS_PIDF") log=$TEAM_FOULS_LOG"
echo ""
echo "monitor:"
echo "  tail -f $TEAM_STATS_LOG"
echo "  tail -f $TEAM_FOULS_LOG"
echo ""
echo "outputs:"
echo "  $TEAM_STATS_OUT"
echo "  $TEAM_FOULS_OUT"
