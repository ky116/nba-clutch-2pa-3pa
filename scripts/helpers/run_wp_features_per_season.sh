#!/usr/bin/env sh
set -eu

# Run scripts/core/build_wp_features.py one season at a time.
# Example:
#   bash scripts/helpers/run_wp_features_per_season.sh --start-season 2000 --end-season 2024 --seasontype rs

START_SEASON=2000
END_SEASON=2024
SEASONTYPE="rs"
PYTHON_BIN="${PYTHON_BIN:-python3}"
JOBS=25
PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

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
    -h|--help)
      cat <<'EOF'
Usage:
  bash scripts/helpers/run_wp_features_per_season.sh [options]

Options:
  --start-season <int>   First season to run (default: 2000)
  --end-season <int>     Last season to run (default: 2024)
  --seasontype <rs>      Season type (default: rs)
  --python-bin <path>    Python executable (default: python3)
  --jobs <int>           Parallel jobs across seasons (default: 1)
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

echo "Running scripts/core/build_wp_features.py per season"
echo "  seasontype: $SEASONTYPE"
echo "  seasons:    $START_SEASON .. $END_SEASON"
echo "  python:     $PYTHON_BIN"
echo "  jobs:       $JOBS"
echo "  unbuffered: $PYTHONUNBUFFERED"
echo ""

# Safety: old workflows left rs outputs as symlinks (often rs -> rg).
# Remove them up front so outputs are always real files.
RS_SYMLINKS="$(find data/wp -maxdepth 1 -type l -name '*_rs*.csv.gz' 2>/dev/null || true)"
if [ -n "$RS_SYMLINKS" ]; then
  echo "[cleanup] removing rs symlinks before run"
  echo "$RS_SYMLINKS" | while IFS= read -r p; do
    [ -n "$p" ] && rm -f "$p"
  done
fi

export PYTHONUNBUFFERED

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
    "$PYTHON_BIN" scripts/core/build_wp_features.py \
      --start-season "$season" \
      --end-season "$season" \
      --seasontype "$SEASONTYPE"
  done < "$tmp_seasons"
else
  xargs -P "$JOBS" -I {} sh -c '
    season="$1"
    seasontype="$2"
    python_bin="$3"
    echo "[run] season=$season seasontype=$seasontype"
    "$python_bin" scripts/core/build_wp_features.py \
      --start-season "$season" \
      --end-season "$season" \
      --seasontype "$seasontype"
  ' sh {} "$SEASONTYPE" "$PYTHON_BIN" < "$tmp_seasons"
fi

echo ""
echo "[done] completed seasons $START_SEASON .. $END_SEASON ($SEASONTYPE)"
echo "[next] combine if needed:"
echo "  python3 scripts/core/combine_wp_states.py --start-season $START_SEASON --end-season $END_SEASON --seasontype $SEASONTYPE"
