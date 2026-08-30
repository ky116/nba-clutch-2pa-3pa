#!/usr/bin/env bash
set -euo pipefail

THIS_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${PROJECT_DIR:-}" ]]; then
  for cand in "${SLURM_SUBMIT_DIR:-}" "${PWD}" "${PWD}/../.." "${THIS_SCRIPT_DIR}/../.."; do
    if [[ -n "${cand}" && -f "${cand}/scripts/core/fit_wp_model.r" ]]; then
      PROJECT_DIR="$(cd "${cand}" && pwd -P)"
      break
    fi
  done
fi
if [[ -z "${PROJECT_DIR:-}" || ! -f "${PROJECT_DIR}/scripts/core/fit_wp_model.r" ]]; then
  echo "[error] cannot resolve PROJECT_DIR from PWD=${PWD} SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-} THIS_SCRIPT_DIR=${THIS_SCRIPT_DIR}" >&2
  exit 1
fi
cd "${PROJECT_DIR}"

THROUGH="panel"
FROM_STAGE="raw"
SUBMIT_SLURM=0
DRY_RUN=0
CHECK_INPUTS_ONLY=0
SKIP_EXISTING=1
PYTHON_BIN="${PYTHON_BIN:-python3}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/pipelines/run_reproduction_release.sh [options]

Options:
  --check-inputs          Check required raw/intermediate inputs and exit.
  --from <stage>          First stage to run: raw, core, wp, context, wp_score, panel, wf, full, figures.
  --through <stage>       Last stage to run: inputs, core, wp, context, wp_score, panel, validate, wf, full, figures.
                          Default: panel.
  --submit-slurm          Submit Slurm jobs for panel/WF/full stages where applicable.
  --no-skip-existing      Rebuild stages even when their expected outputs already exist.
  --dry-run               Print commands without executing.
  --python-bin <path>     Python executable for local commands.
  -h, --help              Show this help.

Examples:
  bash scripts/pipelines/run_reproduction_release.sh --check-inputs
  bash scripts/pipelines/run_reproduction_release.sh --through panel
  bash scripts/pipelines/run_reproduction_release.sh --through figures
  bash scripts/pipelines/run_reproduction_release.sh --submit-slurm --from wp_score --through full
  bash scripts/pipelines/run_reproduction_release.sh --submit-slurm --from panel --through full
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check-inputs)
      CHECK_INPUTS_ONLY=1
      shift
      ;;
    --from)
      FROM_STAGE="$2"
      shift 2
      ;;
    --through)
      THROUGH="$2"
      shift 2
      ;;
    --submit-slurm)
      SUBMIT_SLURM=1
      shift
      ;;
    --no-skip-existing)
      SKIP_EXISTING=0
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --python-bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[error] unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

stage_rank() {
  case "$1" in
    raw|inputs) echo 0 ;;
    core) echo 1 ;;
    wp) echo 2 ;;
    context) echo 3 ;;
    wp_score) echo 4 ;;
    panel) echo 5 ;;
    validate) echo 6 ;;
    wf) echo 7 ;;
    full) echo 8 ;;
    figures) echo 9 ;;
    *)
      echo "[error] unknown stage: $1" >&2
      exit 1
      ;;
  esac
}

FROM_RANK="$(stage_rank "${FROM_STAGE}")"
THROUGH_RANK="$(stage_rank "${THROUGH}")"
if [[ "${THROUGH_RANK}" -lt "${FROM_RANK}" ]]; then
  echo "[error] --through must be >= --from" >&2
  exit 1
fi

in_range() {
  local rank
  rank="$(stage_rank "$1")"
  [[ "${rank}" -ge "${FROM_RANK}" && "${rank}" -le "${THROUGH_RANK}" ]]
}

run_cmd() {
  printf '[run]'
  printf ' %q' "$@"
  printf '\n'
  if [[ "${DRY_RUN}" == "0" ]]; then
    "$@"
  fi
}

run_shell() {
  echo "[run] $*"
  if [[ "${DRY_RUN}" == "0" ]]; then
    bash -lc "$*"
  fi
}

check_contract() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[check] reproduction contract stage=$1 (dry-run skipped)"
    return 0
  fi
  run_cmd "${PYTHON_BIN}" scripts/helpers/check_reproduction_contracts.py --stage "$1"
}

expect_file() {
  if [[ ! -f "$1" ]]; then
    echo "[error] missing expected file: $1" >&2
    exit 1
  fi
}

stage_done() {
  case "$1" in
    core)
      [[ -f data/processed/games_2000_rs.parquet && -f data/processed/games_2024_rs.parquet ]]
      ;;
    wp)
      [[ -f data/wp/wp_states_2000_2024_rs.csv.gz && -f data/wp/shot_decision_states_2000_2024_rs.csv.gz ]]
      ;;
    context)
      [[ -f data/analysis/team_shot_stats_2000_2024.parquet && -f data/analysis/cumulative_team_fouls_2000_2024_rs.parquet ]]
      ;;
    wp_score)
      [[ -f data/wp/shot_decision_states_2000_2024_rs_with_wp.csv.gz ]]
      ;;
    panel)
      [[ -f data/analysis/shotchoice_panel_clutch_rs.parquet ]]
      ;;
    wf)
      [[ -d results/nested_wf_catboost_gpu && -d results/nested_wf_xgb && -d results/nested_wf_lgbm ]]
      ;;
    full)
      [[ -d results/full_data_ensemble_state_fixed_loso ]]
      ;;
    figures)
      [[ -f figures/figure2_full_data_t30_300_cate_surface.png && -f figures/figure3_outer_fold_t30_300_cate_surface.png && -f figures/figures1_cate_surface_0_30s_masked_n50.png ]]
      ;;
    *)
      return 1
      ;;
  esac
}

maybe_skip() {
  local stage="$1"
  if [[ "${SKIP_EXISTING}" == "1" ]] && stage_done "${stage}"; then
    if [[ "${stage}" == "panel" && "${DRY_RUN}" == "0" ]]; then
      if ! "${PYTHON_BIN}" scripts/helpers/check_reproduction_contracts.py --stage panel; then
        echo "[rebuild] panel: existing outputs failed contract; rebuilding"
        return 1
      fi
    fi
    echo "[skip] ${stage}: expected outputs already exist"
    return 0
  fi
  return 1
}

check_raw_inputs() {
  local missing=0
  for season in $(seq 2000 2024); do
    for kind in nbastats pbpstats shotdetail; do
      if [[ ! -f "data/nba_raw/${kind}_${season}.csv" ]]; then
        echo "[missing] data/nba_raw/${kind}_${season}.csv" >&2
        missing=1
      fi
    done
  done
  if [[ "${missing}" != "0" ]]; then
    echo "[error] raw NBA CSV inputs are incomplete" >&2
    exit 1
  fi
  echo "[ok] raw NBA CSV inputs found for nbastats/pbpstats/shotdetail, seasons 2000-2024"
}

submit_job() {
  local out job_id
  echo "[submit] $*" >&2
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRYRUN"
    return 0
  fi
  out="$("$@")"
  echo "${out}" >&2
  job_id="$(printf '%s\n' "${out}" | awk '/Submitted batch job/ {print $4}' | tail -n 1)"
  if [[ -z "${job_id}" ]]; then
    echo "[error] could not parse Slurm job id from: ${out}" >&2
    exit 1
  fi
  printf '%s\n' "${job_id}"
}

mkdir -p data/processed data/wp data/analysis models results logs tmp

echo "[config] project=${PROJECT_DIR}"
echo "[config] from=${FROM_STAGE} through=${THROUGH} submit_slurm=${SUBMIT_SLURM} skip_existing=${SKIP_EXISTING} dry_run=${DRY_RUN}"
echo "[config] python=${PYTHON_BIN}"

if [[ "${CHECK_INPUTS_ONLY}" == "1" || "${THROUGH}" == "inputs" ]] || in_range raw; then
  check_raw_inputs
  check_contract raw
fi
if [[ "${CHECK_INPUTS_ONLY}" == "1" || "${THROUGH}" == "inputs" ]]; then
  exit 0
fi

PANEL_JOB=""
WP_SCORE_JOB=""
VALIDATE_JOB=""
WF_CAT_JOB=""
WF_XGB_JOB=""
WF_LGBM_JOB=""
FULL_WRAPPER_JOB=""
FIGURE_JOB=""

if in_range core; then
  if ! maybe_skip core; then
    run_cmd bash scripts/helpers/run_core_tables_per_season.sh --python-bin "${PYTHON_BIN}"
  fi
  check_contract core
fi

if in_range wp; then
  if ! maybe_skip wp; then
    run_cmd bash scripts/helpers/run_wp_features_per_season.sh --python-bin "${PYTHON_BIN}"
    run_cmd "${PYTHON_BIN}" scripts/core/combine_wp_states.py --start-season 2000 --end-season 2024 --seasontype rs
  fi
  check_contract wp
fi

if in_range context; then
  if ! maybe_skip context; then
    run_cmd "${PYTHON_BIN}" scripts/core/construct_team_shot_stats.py \
      --start-season 2000 \
      --end-season 2024 \
      --input-start-season 2000 \
      --input-end-season 2024 \
      --seasontype rs \
      --out data/analysis/team_shot_stats_2000_2024.parquet
    run_cmd "${PYTHON_BIN}" scripts/core/construct_cumulative_team_foul.py \
      --start-season 2000 \
      --end-season 2024 \
      --seasontype rs \
      --output data/analysis/cumulative_team_fouls_2000_2024_rs.parquet
  fi
  check_contract context
fi

if in_range wp_score; then
  if ! maybe_skip wp_score; then
    if [[ "${SUBMIT_SLURM}" == "1" ]]; then
      WP_SCORE_JOB="$(submit_job sbatch scripts/slurm/run_rs_m0_k0_wp_scoring_slurm.sh)"
      echo "[submitted] wp_score=${WP_SCORE_JOB}"
    else
      run_shell "TMPDIR='${PROJECT_DIR}/tmp' OOF_TEMPLATE_PROTOCOL_M0=1 PYTHON_BIN='${PYTHON_BIN}' bash scripts/pipelines/run_rs_m0_k0_wp_scoring.sh"
    fi
  fi
fi

if in_range panel; then
  if ! maybe_skip panel; then
    if [[ "${SUBMIT_SLURM}" == "1" ]]; then
      dep_args=()
      if [[ -n "${WP_SCORE_JOB}" ]]; then
        dep_args=(--dependency="afterok:${WP_SCORE_JOB}")
      fi
      PANEL_JOB="$(submit_job sbatch "${dep_args[@]}" scripts/slurm/run_rs_m0_k0_shot_panel_slurm.sh)"
      echo "[submitted] panel=${PANEL_JOB}"
    else
      run_shell "PYTHON_BIN='${PYTHON_BIN}' bash scripts/pipelines/run_rs_m0_k0_panel_from_wp.sh"
    fi
  fi
  if [[ "${SUBMIT_SLURM}" != "1" || -z "${PANEL_JOB}" ]]; then
    check_contract panel
  fi
fi

if in_range validate; then
  if [[ "${SUBMIT_SLURM}" == "1" && -n "${PANEL_JOB}" ]]; then
    VALIDATE_JOB="$(submit_job sbatch --dependency="afterok:${PANEL_JOB}" --job-name=nba_panel_validate --cpus-per-task=4 --mem=24G --time=04:00:00 --chdir="${PROJECT_DIR}" -o "${PROJECT_DIR}/logs/%j.txt" --wrap "cd '${PROJECT_DIR}' && '${PYTHON_BIN}' scripts/helpers/validate_wp_scored_shots.py --with-wp data/wp/shot_decision_states_2000_2024_rs_with_wp.csv.gz --panel data/analysis/shotchoice_panel_clutch_rs.parquet && '${PYTHON_BIN}' scripts/helpers/check_reproduction_contracts.py --stage panel")"
    echo "[submitted] validate=${VALIDATE_JOB}"
  else
    run_cmd "${PYTHON_BIN}" scripts/helpers/validate_wp_scored_shots.py \
      --with-wp data/wp/shot_decision_states_2000_2024_rs_with_wp.csv.gz \
      --panel data/analysis/shotchoice_panel_clutch_rs.parquet
    check_contract panel
  fi
fi

if in_range wf; then
  if ! maybe_skip wf; then
    if [[ "${SUBMIT_SLURM}" == "1" ]]; then
      dep_args=()
      if [[ -n "${VALIDATE_JOB}" ]]; then
        dep_args=(--dependency="afterok:${VALIDATE_JOB}")
      elif [[ -n "${PANEL_JOB}" ]]; then
        dep_args=(--dependency="afterok:${PANEL_JOB}")
      fi
      WF_CAT_JOB="$(submit_job sbatch "${dep_args[@]}" scripts/pipelines/run_nested_walk_forward_catboost.sh)"
      WF_XGB_JOB="$(submit_job sbatch "${dep_args[@]}" scripts/slurm/run_nested_walk_forward_xgb_slurm.sh)"
      WF_LGBM_JOB="$(submit_job sbatch "${dep_args[@]}" scripts/slurm/run_nested_walk_forward_lgbm_slurm.sh)"
      echo "[submitted] wf_catboost=${WF_CAT_JOB}"
      echo "[submitted] wf_xgb=${WF_XGB_JOB}"
      echo "[submitted] wf_lgbm=${WF_LGBM_JOB}"
    else
      run_cmd bash scripts/pipelines/run_nested_walk_forward_catboost.sh
      run_cmd bash scripts/slurm/run_nested_walk_forward_xgb_slurm.sh
      run_cmd bash scripts/slurm/run_nested_walk_forward_lgbm_slurm.sh
    fi
  fi
  if [[ "${SUBMIT_SLURM}" != "1" || -z "${WF_CAT_JOB}${WF_XGB_JOB}${WF_LGBM_JOB}" ]]; then
    check_contract wf
  fi
fi

if in_range full; then
  if ! maybe_skip full; then
    if [[ "${SUBMIT_SLURM}" == "1" ]]; then
      dep=""
      auto_figures=0
      if [[ "${THROUGH_RANK}" -ge "$(stage_rank figures)" ]]; then
        auto_figures=1
      fi
      if [[ -n "${WF_CAT_JOB}" && -n "${WF_XGB_JOB}" && -n "${WF_LGBM_JOB}" ]]; then
        dep="${WF_CAT_JOB}:${WF_XGB_JOB}:${WF_LGBM_JOB}"
      fi
      if [[ -n "${dep}" ]]; then
        FULL_WRAPPER_JOB="$(submit_job sbatch --dependency="afterok:${dep}" --job-name=nba_full_data_submit --cpus-per-task=1 --mem=4G --time=02:00:00 --chdir="${PROJECT_DIR}" -o "${PROJECT_DIR}/logs/%j.txt" --wrap "cd '${PROJECT_DIR}' && AUTO_SUBMIT_FIGURES='${auto_figures}' PYTHON_BIN='${PYTHON_BIN}' bash scripts/pipelines/run_full_data_pipeline.sh")"
      else
        run_shell "AUTO_SUBMIT_FIGURES='${auto_figures}' PYTHON_BIN='${PYTHON_BIN}' bash scripts/pipelines/run_full_data_pipeline.sh"
      fi
      [[ -n "${FULL_WRAPPER_JOB}" ]] && echo "[submitted] full_wrapper=${FULL_WRAPPER_JOB}"
    else
      run_cmd bash scripts/pipelines/run_full_data_pipeline.sh
    fi
  fi
  if [[ "${SUBMIT_SLURM}" != "1" || -z "${FULL_WRAPPER_JOB}" ]]; then
    check_contract full
  fi
fi

if in_range figures; then
  if [[ "${SUBMIT_SLURM}" == "1" && -n "${FULL_WRAPPER_JOB}" ]]; then
    echo "[skip] figures: full-data wrapper job ${FULL_WRAPPER_JOB} will submit figure assembly after the ensemble job"
  elif ! maybe_skip figures; then
    if [[ "${SUBMIT_SLURM}" == "1" ]]; then
      FIGURE_JOB="$(submit_job sbatch --job-name=nba_figures --cpus-per-task=8 --mem=32G --time=24:00:00 --chdir="${PROJECT_DIR}" -o "${PROJECT_DIR}/logs/%j.txt" --wrap "cd '${PROJECT_DIR}' && '${PYTHON_BIN}' scripts/helpers/check_reproduction_contracts.py --stage full && '${PYTHON_BIN}' scripts/helpers/rebuild_wf_cate_surfaces_recalibrated.py --outdir results/wf_cate_surfaces --catboost-dir results/nested_wf_catboost_gpu --xgb-dir results/nested_wf_xgb --lgbm-dir results/nested_wf_lgbm && '${PYTHON_BIN}' scripts/helpers/summarize_cate_time_window_stability.py --rows results/wf_cate_surfaces/outer_fold_ensemble_tau_test_rows.parquet --outdir results/wf_cate_surfaces/time_window_stability && '${PYTHON_BIN}' scripts/helpers/assemble_manuscript_cate_figures.py --ensemble-dir results/full_data_ensemble_state_fixed_loso --wf-dir results/wf_cate_surfaces --figure-source-dir results/figure_source_data --panel data/analysis/shotchoice_panel_clutch_rs.parquet --outdir figures && '${PYTHON_BIN}' scripts/helpers/summarize_wp_model_dependence_sensitivity.py && '${PYTHON_BIN}' scripts/helpers/check_reproduction_contracts.py --stage figures")"
      echo "[submitted] figures=${FIGURE_JOB}"
    else
      run_cmd "${PYTHON_BIN}" scripts/helpers/rebuild_wf_cate_surfaces_recalibrated.py \
      --outdir results/wf_cate_surfaces \
      --catboost-dir results/nested_wf_catboost_gpu \
      --xgb-dir results/nested_wf_xgb \
      --lgbm-dir results/nested_wf_lgbm
    run_cmd "${PYTHON_BIN}" scripts/helpers/summarize_cate_time_window_stability.py \
      --rows results/wf_cate_surfaces/outer_fold_ensemble_tau_test_rows.parquet \
      --outdir results/wf_cate_surfaces/time_window_stability
	    run_cmd "${PYTHON_BIN}" scripts/helpers/assemble_manuscript_cate_figures.py \
	      --ensemble-dir results/full_data_ensemble_state_fixed_loso \
	      --wf-dir results/wf_cate_surfaces \
	      --figure-source-dir results/figure_source_data \
	      --panel data/analysis/shotchoice_panel_clutch_rs.parquet \
	      --outdir figures
	    run_cmd "${PYTHON_BIN}" scripts/helpers/summarize_wp_model_dependence_sensitivity.py
	    fi
  fi
  if [[ "${SUBMIT_SLURM}" != "1" || -z "${FULL_WRAPPER_JOB}${FIGURE_JOB}" ]]; then
    check_contract figures
  fi
fi

echo "[done] reproduction orchestration reached --through ${THROUGH}"
