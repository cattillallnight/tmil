$ErrorActionPreference = "SilentlyContinue"

# Delete PG-EGAE and old results
$filesToDelete = @(
    "results\pg_gae_step01_clusters.json",
    "results\pg_gae_step02_normal_graphs.pt",
    "results\pg_gae_step02_phisher_graphs.pt",
    "results\pg_gae_step05_results.json",
    "results\step02b_features_hybrid.pkl",
    "results\step02d_features_hybrid_norm.pkl",
    "results\step02c_mse_distribution.png",
    "results\step04_windows_stats.json",
    "results\step08_sidak_thresholds.json",
    "results\step09_gt_targeted_cv.json",
    "results\step09_nested_cv_results.json",
    "results\step13_time_aware_localization_metrics.csv",
    "results\step15_perturbation_study.json",
    "archive\flat_refactor.ps1",
    "archive\fix_flat_imports.py"
)

foreach ($file in $filesToDelete) {
    Remove-Item -Path $file -Force
}

# Clean up archive root
$archiveScripts = @(
    "archive\cohen_kappa_evaluation.py",
    "archive\etherscan_label_scraper.py",
    "archive\fetch_brian_labels.py",
    "archive\fetch_labels.py",
    "archive\human_annotation_sheet_builder.py",
    "archive\orthogonality_validation.py"
)

foreach ($script in $archiveScripts) {
    Move-Item -Path $script -Destination "archive\legacy_scripts\" -Force
}

# Remove empty data subfolders if any
# Actually let's keep data/raw etc. The user wanted them.

Write-Host "Deep cleanup completed!"
