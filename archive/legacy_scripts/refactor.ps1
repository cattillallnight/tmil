$ErrorActionPreference = "SilentlyContinue"

# Create directories
New-Item -ItemType Directory -Force -Path "core"
New-Item -ItemType Directory -Force -Path "analysis"
New-Item -ItemType Directory -Force -Path "experiments"
New-Item -ItemType Directory -Force -Path "archive\legacy_ground_truth"

# 1. step27
Move-Item -Path "archive\old_experiments\step27_attention_saliency.py" -Destination "analysis\step27_attention_saliency.py"

# 2. step11
Move-Item -Path "archive\old_experiments\step11_ablation_study.py" -Destination "experiments\step11_ablation_study.py"

# 3. Legacy GT
Move-Item -Path "archive\old_experiments\step12_time_aware_gt_builder.py" -Destination "archive\legacy_ground_truth\step12_time_aware_gt_builder.py"
Move-Item -Path "archive\old_experiments\step14_statistical_gt_builder.py" -Destination "archive\legacy_ground_truth\step14_statistical_gt_builder.py"

# 4. Core Pipeline rename and move
Move-Item -Path "step01_dataset_analysis.py" -Destination "core\01_dataset_analysis.py"
Move-Item -Path "step02_feature_extraction.py" -Destination "core\02_feature_extraction.py"
Move-Item -Path "step05_model_architecture.py" -Destination "core\05_tmil_architecture.py"
Move-Item -Path "step07_training.py" -Destination "core\07_train_tmil.py"
Move-Item -Path "step25_evaluate_original_tmil.py" -Destination "core\25_evaluate_tmil.py"
Move-Item -Path "utils.py" -Destination "core\utils.py"

# 5. Ground Truth (Move contents of dataset_builders to existing ground_truth folder, then delete dataset_builders)
Get-ChildItem -Path "dataset_builders" | Move-Item -Destination "ground_truth"
Remove-Item -Path "dataset_builders" -Force

Write-Host "Refactoring completed!"
