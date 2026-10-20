# Trustworthy Neuroprognostication using CNN–BiGRU on I-CARE BCI

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official code implementation for:  
**Uncertainty-Aware Multicenter Neuroprognostication with Longitudinal Brain Continuity Index**

This repository provides the complete experimental code to evaluate multimodal deep learning and classical machine learning models for post–cardiac arrest neurological outcome prediction. The pipeline operates on the 72-hour **Brain Continuity Index (BCI)** representation from the I-CARE consortium under strict **Leave-One-Hospital-Out (LOHO)** cross-validation across 607 patients.

---

## Key Features

1. **Multimodal CNN–BiGRU**:
   - 1D convolutional downsampling (~100× sequence compression from 25,920 to 259 steps) coupled with bidirectional GRU.
   - Fused with 6 clinical covariates (age, sex, ROSC duration, OHCA, shockable rhythm, TTM).
   - Active Monte Carlo dropout ($S = 50$ stochastic passes) for predictive uncertainty quantification.

2. **Benchmarked Baselines**:
   - **Deep Sequence Model**: Temporal Convolutional Network (TCN) with dilated causal convolutions.
   - **Classical ML Baselines**: XGBoost, Random Forest, and Regularized Logistic Regression (L1 and L2 penalties evaluated in tuning) trained on 21 engineered temporal summaries + 6 clinical covariates.

3. **Trustworthy AI Audits**:
   - **Safety Operating Point**: True positive rate evaluated at fixed false positive rate $\le 0.05$ (PhysioNet Challenge benchmark).
   - **Probabilistic Calibration**: Expected Calibration Error (ECE) and Brier scores evaluated pre- and post-temperature scaling.
   - **Demographic Fairness**: Stratified evaluation across sex and age groups with 1,000-sample bootstrap 95% confidence intervals.
   - **Selective Prediction**: Risk–coverage analysis driven by Monte Carlo predictive entropy (total predictive uncertainty).

---

## Summary of Results (Pooled $N=607$)

| Model | AUROC [95% CI] | PR-AUC [95% CI] | Positive F1 [95% CI] | TPR @ FPR $\le$ 0.05 | ECE (Calibrated) | Brier (Calibrated) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Multimodal CNN–BiGRU** | 0.736 [0.696, 0.775] | **0.830** [0.793, 0.864] | 0.739 [0.703, 0.774] | **0.272** | 0.170 | 0.231 |
| **Logistic Regression**   | **0.747** [0.703, 0.787] | 0.800 [0.752, 0.846] | 0.749 [0.718, 0.784] | 0.152 | 0.117 | 0.208 |
| **XGBoost**               | 0.742 [0.700, 0.782] | 0.802 [0.759, 0.843] | **0.763** [0.729, 0.796] | 0.160 | **0.044** | **0.193** |
| **Random Forest**         | 0.724 [0.680, 0.764] | 0.786 [0.740, 0.830] | 0.759 [0.724, 0.788] | 0.157 | 0.062 | 0.200 |
| **TCN**                   | 0.727 [0.687, 0.766] | 0.814 [0.773, 0.853] | 0.755 [0.727, 0.785] | 0.221 | 0.112 | 0.204 |

---

## Repository Structure

```text
├── icare_project/
│   ├── run_nested_tuning.py        # Complete nested LOHO tuning & evaluation across all 5 models
│   ├── model/
│   │   └── cnn_bigru_model/
│   │       ├── model.py            # Multimodal CNN-BiGRU PyTorch architecture
│   │       └── data_loader.py      # BCI and clinical metadata data loader
│   ├── data/
│   │   └── unified_loader.py       # Deterministic LOHO hospital splits
│   ├── experiments/
│   │   ├── tcn/
│   │   │   └── model_tcn.py        # TCN alternative deep model
│   │   └── ablations/
│   │       ├── model_ablations.py  # Ablation variants (CNN-only, GRU-only, No-clinical)
│   │       └── train_eval_ablations.py # Structural ablation training
│   ├── features/
│   │   └── summary_features.py     # 21 engineered temporal BCI features
│   ├── stats/
│   │   ├── delong_tests.py         # DeLong test for AUROC comparison
│   │   └── bootstrap_tests.py      # Paired bootstrap tests for F1 and CIs
│   ├── tables/                     # Output CSV performance tables (with 95% CIs)
│   └── figures/
│       └── paper_ready/            # High-resolution publication figures (ROC, Calibration, etc.)
├── requirements.txt
└── README.md
```

---

## Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/paaaraaaz/icare-trustworthy-ai.git
   cd icare-trustworthy-ai
   ```

2. **Install requirements**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Data Access**:
   - The Brain Continuity Index (BCI) dataset is publicly available on Dryad: [**Dryad DOI: 10.5061/dryad.2fqz612zv**](https://doi.org/10.5061/dryad.2fqz612zv).
   - Clinical metadata can be downloaded from the PhysioNet I-CARE challenge repository: [**PhysioNet I-CARE Database**](https://doi.org/10.13026/m33r-bj81).

---

## Running the Pipeline

To execute the complete nested Leave-One-Hospital-Out evaluation across all 5 model families:
```bash
python icare_project/run_nested_tuning.py
```

To run the structural ablation experiments:
```bash
python icare_project/experiments/ablations/train_eval_ablations.py
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
