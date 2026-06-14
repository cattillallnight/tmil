# C-TMIL: Counterparty-Aware Temporal Multiple Instance Learning for Phishing Transaction Localization

This repository contains the official implementation for the paper: **"Counterparty-Aware Temporal Multiple Instance Learning for Phishing Transaction Localization"**.

## 📌 Overview
Detecting phishing accounts on Ethereum is typically treated as a node classification problem. However, forensic analysis and law enforcement require **localizing the exact cashout transactions** (e.g., transfers to Tornado Cash) to track illicit fund flows.

This project introduces **C-TMIL**, a novel decoupled architecture that combines:
1. **Self-Supervised Sequence Embeddings (BERT4ETH)**
2. **Counterparty-Aware Multiple Instance Learning (C-TMIL)**
3. **Outbound-Only Attention Masking** to prevent weak-supervision collapse (MIL leakage).

## 🚀 Quick Start & Project Pipeline

The project is structured into 10 sequential steps for full reproducibility.

### Part 1: Ground Truth & Data Engineering
- `step01_build_tornado_cash_gt.py`: Harvests Tornado Cash interaction labels.
- `step02_build_scamsniffer_gt.py`: Builds the core ground-truth dataset from ScamSniffer.
- `step03_feature_extraction.py`: Extracts handcrafted features (z_amount, density, etc.) and integrates `BERT4ETH` outputs.

### Part 2: Baseline TMIL Architecture
- `step04_train_baseline.py`: Trains the standard TMIL baseline.
- `step05_evaluate_baseline.py`: Evaluates Bag-level classification and Instance-level localization (Hit@K).
- `step06_ablation_study.py`: Evaluates the contribution of various feature subsets.
- `step07_attention_analysis.py`: Analyzes the learned attention distributions.
- `step08_forensic_experiments.py`: Runs deep forensic case studies on detected cashouts.

### Part 3: C-TMIL (The Proposed Method)
- `step09_build_counterparty_vocab.py`: Builds a vocabulary of the top 50,000 transaction counterparties.
- `step10_ctmil_experiment.py`: Trains the **C-TMIL** model with **Outbound-Only Masking**, achieving the state-of-the-art localization Hit@10 score.

## 🛠 Model Architecture (`tmil_architecture.py`)
- Standard Dual-Stream Gated Attention MIL.
- Advanced C-TMIL with `nn.Embedding` for Counterparty routing and `masked_fill_` for Outbound constraints.

## 📦 Requirements
Install dependencies using:
```bash
pip install -r requirements.txt
```

## 📊 Dataset Notice
Due to privacy and size constraints, the raw Ethereum transaction graphs (several GBs) and the original `embed.tfrecord` files are not included in this repository. Ensure you have the `../data/` directory properly populated before running step 1.
