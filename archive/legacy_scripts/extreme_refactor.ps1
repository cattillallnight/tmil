$ErrorActionPreference = "SilentlyContinue"

# 1. Create Directories
New-Item -ItemType Directory -Force -Path "src"
New-Item -ItemType Directory -Force -Path "scripts"
New-Item -ItemType Directory -Force -Path "data\raw"
New-Item -ItemType Directory -Force -Path "data\processed"
New-Item -ItemType Directory -Force -Path "data\ground_truth"
New-Item -ItemType Directory -Force -Path "results\models"
New-Item -ItemType Directory -Force -Path "results\logs"
New-Item -ItemType Directory -Force -Path "results\evaluation"
New-Item -ItemType Directory -Force -Path "archive\legacy_scripts"

# 2. Move to src
Move-Item -Path "core\02_feature_extraction.py" -Destination "src\feature_extraction.py" -Force
Move-Item -Path "core\05_tmil_architecture.py" -Destination "src\model.py" -Force
Move-Item -Path "core\07_train_tmil.py" -Destination "src\train.py" -Force
Move-Item -Path "core\25_evaluate_tmil.py" -Destination "src\evaluate.py" -Force
Move-Item -Path "core\utils.py" -Destination "src\utils.py" -Force

# 3. Rename Ground Truth builders
Move-Item -Path "ground_truth\step16_etherscan_tc_crawler.py" -Destination "ground_truth\build_tornado_cash_gt.py" -Force
Move-Item -Path "ground_truth\step21_build_scamsniffer_dataset.py" -Destination "ground_truth\build_scamsniffer_gt.py" -Force
Move-Item -Path "ground_truth\step18_chainabuse_pipeline.py" -Destination "ground_truth\build_chainabuse_gt.py" -Force

# 4. Move to scripts
Move-Item -Path "core\01_dataset_analysis.py" -Destination "scripts\dataset_analysis.py" -Force
Move-Item -Path "experiments\step11_ablation_study.py" -Destination "scripts\ablation.py" -Force
Move-Item -Path "analysis\step27_attention_saliency.py" -Destination "scripts\attention_analysis.py" -Force

# 5. Archive everything else in ground_truth that hasn't been renamed
Get-ChildItem -Path "ground_truth" -Filter "step*.py" | Move-Item -Destination "archive\legacy_scripts" -Force

# 6. Archive everything in old_experiments
Get-ChildItem -Path "archive\old_experiments" | Move-Item -Destination "archive\legacy_scripts" -Force

# 7. Cleanup old directories
Remove-Item -Path "core" -Force -Recurse
Remove-Item -Path "analysis" -Force -Recurse
Remove-Item -Path "experiments" -Force -Recurse
Remove-Item -Path "archive\old_experiments" -Force -Recurse

Write-Host "Extreme refactoring completed!"
