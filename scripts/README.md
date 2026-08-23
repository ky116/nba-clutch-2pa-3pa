# Scripts Layout

This directory contains runnable shell wrappers grouped by role for the
regular-season CATE manuscript workflow.

## Directories

- `pipelines/`: workflow-level entry points for panel generation, nested
  walk-forward, full-data training, and ensemble CATE summaries.
- `helpers/`: reusable narrower wrappers called by pipelines or used for
  targeted data/model refreshes.
- `slurm/`: Slurm submission wrappers and Slurm-oriented job scripts.
- `core/`: Python and R analysis programs called by the wrappers.

## Current Rerun Path

1. `scripts/pipelines/run_rs_m0_k0_shot_panel.sh`

   Server/Slurm entry: `scripts/slurm/run_rs_m0_k0_shot_panel_slurm.sh`.
   Strict season-OOF scoring is parallelized with `SEASON_OOF_JOBS` and `BAM_NTHREADS`; keep `SEASON_OOF_JOBS * BAM_NTHREADS <= THREADS`.
2. `scripts/helpers/validate_wp_scored_shots.py`

   Post-run validation for the teacher signal. It checks that terminal shot rows are retained with `wp_next = final_home_win`, that `delta_wp = wp_next - wp_before`, and that the late 3/4-point trailing region has visible row and treatment counts.
3. `scripts/pipelines/run_nested_walk_forward_catboost.sh`
4. `scripts/pipelines/run_nested_walk_forward_xgb_lgbm.sh`
5. `scripts/pipelines/run_full_data_pipeline.sh`
6. `scripts/helpers/rebuild_wf_cate_surfaces_recalibrated.py`
7. `scripts/helpers/assemble_manuscript_cate_figures.py`

For direct non-Slurm reruns, `scripts/pipelines/rerun_rs_panel_and_validate.sh` runs steps 1 and 2 in sequence. On the server, prefer submitting the Slurm wrapper first, then running the validator after the job completes.

## Expected Outputs

After `run_rs_m0_k0_shot_panel.sh`:

- `data/wp/shot_decision_states_2000_2024_rs_with_wp.csv.gz`: shot states with `wp_before`, `wp_next`, and `delta_wp`.
- `data/analysis/shotchoice_panel_clutch_rs.parquet`: DML/CATE input panel for RS clutch shots.
- Additional strict/extended panel outputs from `scripts/core/build_shotchoice_panel_from_wp.py`, depending on its current defaults.
- `logs/<job_id>.txt` or `logs/run_rs_m0_k0_shot_panel_*.log`: execution log.

After `validate_wp_scored_shots.py`:

- Console summary only; it does not write data.
- Expected success line: `[ok] WP-scored shot validation completed`.
- Key reported checks: terminal rows kept/scored, max delta identity error, out-of-bounds counts, and row/treatment counts for `time_left_game <= 15` and `-5 < score_diff <= -2`.

## Compatibility Rule

Keep shell entry points in this layout. New internal calls should resolve
`PROJECT_DIR` relative to the script location and should reference analysis
programs through `scripts/core/<name>`.
