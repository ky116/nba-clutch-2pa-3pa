# Reproducible Result Tables

The lightweight result artifacts in this directory are retained so that the
paper's numerical results can be inspected directly on GitHub. Large model,
Parquet, and compressed raw-data artifacts remain local-only.

## Primary CATE surfaces

- [Full-data 30--300 second surface](full_data_ensemble_state_fixed_loso/full_data_t30_300_cate_surface_equal_weight.csv)
- [Full-data 0--30 second diagnostic surface](full_data_ensemble_state_fixed_loso/full_data_t0_30_cate_surface_equal_weight.csv)
- [Walk-forward 30--300 second surfaces](wf_cate_surfaces/outer_fold_ensemble_cate_surface_30_300.csv)
- [Walk-forward surface summary](wf_cate_surfaces/outer_fold_ensemble_cate_surface_summary.csv)
- [CATE time-window stability](wf_cate_surfaces/time_window_stability/cate_time_window_primary_candidate_comparison.csv)

## Support and learner diagnostics

- [Full-data cell support](full_data_ensemble_state_fixed_loso/cate_surface_support/full_data_t30_300_cate_surface_cell_counts.csv)
- [Learner sign agreement](full_data_ensemble_state_fixed_loso/cate_surface_sign_agreement/full_data_t30_300_cate_surface_sign_agreement.csv)
- [Fixed hyperparameters](fixed_hparams_catboost_majority/fixed_tau_params.json)

## WP calibration and sensitivity

- [WP model-dependence surface](wp_calibration/model_dependence_sensitivity/wp_model_dependence_sensitivity_surface.csv)
- [WP model-dependence summary](wp_calibration/model_dependence_sensitivity/wp_model_dependence_sensitivity_summary.csv)
- [WP model-dependence by score](wp_calibration/model_dependence_sensitivity/wp_model_dependence_sensitivity_by_score.csv)
- [Final WP late-tail comparison](wp_calibration/final_wp_audit/overall_late_tail_calibration_comparison.csv)
- [WP calibration by season](wp_calibration/wp_calibration_none_vs_surface_by_season_metrics.csv)

## Figure source data

- [Figure 2 source data](figure_source_data/figure2_full_data_t30_300_cate_surface_source_data.csv)
- [Supplementary CATE source data](figure_source_data/figures1_cate_surface_0_30s_masked_n50_source_data.csv)
