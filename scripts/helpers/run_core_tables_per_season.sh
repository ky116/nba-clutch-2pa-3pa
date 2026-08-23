#!/usr/bin/env sh
set -eu

# Run scripts/core/build_core_tables.py one season at a time.
# Example:
#   bash scripts/helpers/run_core_tables_per_season.sh --start-season 2000 --end-season 2024 --seasontype rs --jobs 8

START_SEASON=2000
END_SEASON=2024
SEASONTYPE="rs"
PYTHON_BIN="${PYTHON_BIN:-python3}"
JOBS=25
REFRESH_ELO=1
ELO_K=20
ELO_H=0
ELO_CARRY=0.75
ELO_MEAN=1500
ELO_INIT=1500

while [ "$#" -gt 0 ]; do
  case "$1" in
    --start-season)
      START_SEASON="$2"
      shift 2
      ;;
    --end-season)
      END_SEASON="$2"
      shift 2
      ;;
    --seasontype)
      SEASONTYPE="$2"
      shift 2
      ;;
    --python-bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --jobs)
      JOBS="$2"
      shift 2
      ;;
    --refresh-elo)
      REFRESH_ELO="$2"
      shift 2
      ;;
    --elo-k)
      ELO_K="$2"
      shift 2
      ;;
    --elo-h)
      ELO_H="$2"
      shift 2
      ;;
    --elo-carry)
      ELO_CARRY="$2"
      shift 2
      ;;
    --elo-mean)
      ELO_MEAN="$2"
      shift 2
      ;;
    --elo-init)
      ELO_INIT="$2"
      shift 2
      ;;
    -h|--help)
      cat <<'EOF'
Usage:
  bash scripts/helpers/run_core_tables_per_season.sh [options]

Options:
  --start-season <int>   First season to run (default: 2000)
  --end-season <int>     Last season to run (default: 2024)
  --seasontype <rs>      Season type (default: rs)
  --python-bin <path>    Python executable (default: python3)
  --jobs <int>           Parallel jobs across seasons (default: 1)
  --refresh-elo <0|1>    Recompute linked Elo after run (default: 1)
  --elo-k <float>        Elo K (default: 20)
  --elo-h <float>        Elo home advantage H (default: 0)
  --elo-carry <float>    Season carry (default: 0.75)
  --elo-mean <float>     Elo mean (default: 1500)
  --elo-init <float>     Elo init for new team (default: 1500)
EOF
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

if [ "$SEASONTYPE" != "rs" ]; then
  echo "--seasontype must be 'rs'" >&2
  exit 1
fi

if [ "$START_SEASON" -gt "$END_SEASON" ]; then
  echo "--start-season must be <= --end-season" >&2
  exit 1
fi

if [ "$JOBS" -lt 1 ] 2>/dev/null; then
  echo "--jobs must be >= 1" >&2
  exit 1
fi

echo "Running scripts/core/build_core_tables.py per season"
echo "  seasontype: $SEASONTYPE"
echo "  seasons:    $START_SEASON .. $END_SEASON"
echo "  python:     $PYTHON_BIN"
echo "  jobs:       $JOBS"
echo "  refresh_elo:$REFRESH_ELO"
echo "  elo:        k=$ELO_K h=$ELO_H carry=$ELO_CARRY mean=$ELO_MEAN init=$ELO_INIT"
echo ""

tmp_seasons="$(mktemp)"
trap 'rm -f "$tmp_seasons"' EXIT INT TERM

season="$START_SEASON"
while [ "$season" -le "$END_SEASON" ]; do
  echo "$season" >> "$tmp_seasons"
  season=$((season + 1))
done

if [ "$JOBS" -eq 1 ]; then
  while IFS= read -r season; do
    echo "[run] season=$season seasontype=$SEASONTYPE"
    "$PYTHON_BIN" scripts/core/build_core_tables.py \
      --start-season "$season" \
      --end-season "$season" \
      --seasontype "$SEASONTYPE" \
      --refresh-elo "$REFRESH_ELO" \
      --elo-k "$ELO_K" \
      --elo-h "$ELO_H" \
      --elo-carry "$ELO_CARRY" \
      --elo-mean "$ELO_MEAN" \
      --elo-init "$ELO_INIT"
  done < "$tmp_seasons"
else
  xargs -P "$JOBS" -n 1 -I {} sh -c '
    season="$1"
    seasontype="$2"
    python_bin="$3"
    echo "[run] season=$season seasontype=$seasontype"
    "$python_bin" scripts/core/build_core_tables.py \
      --start-season "$season" \
      --end-season "$season" \
      --seasontype "$seasontype" \
      --refresh-elo "$4" \
      --elo-k "$5" \
      --elo-h "$6" \
      --elo-carry "$7" \
      --elo-mean "$8" \
      --elo-init "$9"
  ' sh {} "$SEASONTYPE" "$PYTHON_BIN" "$REFRESH_ELO" "$ELO_K" "$ELO_H" "$ELO_CARRY" "$ELO_MEAN" "$ELO_INIT" < "$tmp_seasons"
fi

echo ""
echo "[done] completed seasons $START_SEASON .. $END_SEASON ($SEASONTYPE)"
