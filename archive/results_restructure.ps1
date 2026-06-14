$ErrorActionPreference = "SilentlyContinue"

# 1. Archive Window Ablation Features
New-Item -ItemType Directory -Force -Path "archive\window_ablations"
Move-Item -Path "results\step02_features_W20_S5.pkl" -Destination "archive\window_ablations\" -Force
Move-Item -Path "results\step02_features_W50_S10.pkl" -Destination "archive\window_ablations\" -Force

# 2. Restructure results/
New-Item -ItemType Directory -Force -Path "results\main_paper"
New-Item -ItemType Directory -Force -Path "results\ablations"
New-Item -ItemType Directory -Force -Path "results\archive"

# Move remaining PG-EGAE models (if any weren't caught in deep cleanup) to results/archive/
Move-Item -Path "results\checkpoints\pg_gae_*.pt" -Destination "results\archive\" -Force
Move-Item -Path "results\checkpoints\tmil_hybrid_*.pt" -Destination "results\archive\" -Force
Move-Item -Path "results\checkpoints\tmil_random_*.pt" -Destination "results\archive\" -Force

# Move main canonical checkpoint
Move-Item -Path "results\checkpoints\tmil_eth_final.pt" -Destination "results\main_paper\" -Force
Move-Item -Path "results\checkpoints\tmil_phase1_best.pt" -Destination "results\main_paper\" -Force

# Cleanup empty results/checkpoints if empty
Remove-Item -Path "results\checkpoints" -Force -Recurse

# Move logs, evaluation, figures into results/archive/ unless needed
Move-Item -Path "results\logs" -Destination "results\archive\logs" -Force
Move-Item -Path "results\evaluation" -Destination "results\archive\evaluation" -Force
Move-Item -Path "results\figures" -Destination "results\archive\figures" -Force
Move-Item -Path "results\plots" -Destination "results\archive\plots" -Force
Move-Item -Path "results\saliency_maps" -Destination "results\archive\saliency_maps" -Force

Write-Host "Results folder restructuring completed!"
