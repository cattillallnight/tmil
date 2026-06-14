$ErrorActionPreference = "SilentlyContinue"

# Create directories
New-Item -ItemType Directory -Force -Path "dataset_builders"
New-Item -ItemType Directory -Force -Path "archive\pg_egae_project"
New-Item -ItemType Directory -Force -Path "archive\old_experiments"

# Move dataset_builders
$datasetBuilders = @(
    "step16_etherscan_tc_crawler.py",
    "step16_sanity_test.py",
    "step18_chainabuse_inference.py",
    "step18_chainabuse_pipeline.py",
    "step19_free_gt_collection.py",
    "step20_txhash_mining.py",
    "step21_build_scamsniffer_dataset.py",
    "step21b_refine_scamsniffer.py",
    "step28a_scamsniffer_bert_extraction.py",
    "step28c_scamsniffer_tmil_inference.py"
)
foreach ($file in $datasetBuilders) { Move-Item -Path $file -Destination "dataset_builders" }

# Move pg_egae_project
$pgEgae = @(
    "pg_gae_step01_clustering.py",
    "pg_gae_step02_graph_builder.py",
    "pg_gae_step03_model.py",
    "pg_gae_step04_training.py",
    "pg_gae_step05_evaluation.py",
    "pg_gae_step07_injection_experiment.py",
    "pg_gae_validation.py",
    "step02b_inject_gae_mse.py",
    "step02c_analyze_mse_distribution.py",
    "step02d_normalize_mse.py",
    "step07b_training_hybrid.py",
    "step22_scamsniffer_gae_eval.py",
    "step23_tornado_gae_eval.py",
    "step23_tornado_gae_eval_fixed.py",
    "step24_ablation_noise_mse.py",
    "step25_evaluate_hybrid_tmil.py",
    "step25b_evaluate_hybrid_scamsniffer.py",
    "step25c_evaluate_hybrid_auc.py",
    "step28b_scamsniffer_pgegae_extraction.py"
)
foreach ($file in $pgEgae) { Move-Item -Path $file -Destination "archive\pg_egae_project" }

# Move old_experiments
$oldExps = @(
    "step04_sliding_window.py",
    "step08_sidak_correction.py",
    "step09_nested_cv.py",
    "step10_baselines.py",
    "step11_ablation_study.py",
    "step12_time_aware_gt_builder.py",
    "step13_onchain_localization_eval.py",
    "step14_statistical_gt_builder.py",
    "step15_perturbation_study.py",
    "step17_case_study_validation.py",
    "step26_train_random_ablation.py",
    "step26_out.txt",
    "step27_attention_saliency.py"
)
foreach ($file in $oldExps) { Move-Item -Path $file -Destination "archive\old_experiments" }

Write-Host "Cleanup completed successfully!"
