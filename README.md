# Trustworthy Neuroprognostication using CNN-BiGRU on I-CARE BCI

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official code implementation for the paper:  
*An Uncertainty-Aware Neuroprognostication Pipeline Using Longitudinal Brain Continuity Index and Leave-One-Center-Out Validation* (Under review).

We present a hybrid, trustworthy **CNN+BiGRU** deep learning architecture and traditional Machine Learning (ML) ablation models designed to predict neurological outcomes in patients following cardiac arrest. This repository uses the publicly available Brain-Computer Interface (BCI) extracted features from the **I-CARE** dataset hosted on Dryad.

## Features
- **Deep Learning Model:** A robust 1D Convolutional Neural Network combined with a Bidirectional GRU (`CNN+BiGRU`) tailored for continuous time-series EEG features.
- **Traditional ML Ablation Models:** Implementation of XGBoost, Random Forest, and SVM baselines for direct comparative analysis.
- **Trustworthy AI Pipeline:** Built-in scripts for Uncertainty Quantification (Monte Carlo Dropout), Reliability/Calibration Analysis (Platt Scaling), and Demographic Fairness benchmarking across Age and Sex subgroups.
- **Out-of-Distribution (OOD) Generalization:** Designed utilizing a Leave-One-Hospital-Out (LOHO) or equivalent Cross-Validation validation strategy to rigorously test clinical generalization.

## Repository Structure

The core files provided in this repository map directly to the methodology presented in the manuscript:

```text
├── icare_project/
│   ├── model/
│   │   └── cnn_bigru_model/
│   │       ├── model.py                # Core PyTorch CNN+BiGRU architecture
│   │       ├── data_loader.py          # PyTorch dataset & data loaders for Dryad BCI features
│   │       ├── train_eval.py           # Training, validation, testing, and metrics logic
│   │       ├── main.py                 # Main execution script to run LOOCV pipeline
│   │       ├── calibrate.py            # Post-hoc confidence calibration scripts
│   │       ├── calibrate_confidence.py # Trustworthy AI confidence metrics
│   │       └── generate_visuals.py     # Reproduces paper's visualization graphs
│   ├── experiments/
│   │   └── ablations/
│   │       ├── model_ablations.py      # Traditional ML model instantiations (XGBoost, etc.)
│   │       └── train_eval_ablations.py # Training pipeline for ablation models
│   ├── tables/                         # CSV outputs of fold-level and aggregated performance
│   └── figures/
│       └── paper_ready/                # High-resolution generated manuscript visuals (ROC, Risk-Coverage, Reliability)
└── README.md
```

## Dataset Access

This project utilizes the processed Brain Continuity Index (BCI) features extracted from the I-CARE (International Cardiac Arrest REsearch) consortium clinical EEG dataset. 

Because we rely on the derived BCI dataset hosted on **Dryad**, all data used in our experiments is completely open-access and publicly available. 

To reproduce these results:
1. Download the isolated I-CARE BCI dataset from Dryad here: [**Dryad DOI:10.5061/dryad.2fqz612zv**](https://datadryad.org/dataset/doi:10.5061/dryad.2fqz612zv).
2. For context on the original clinical dataset, you may refer to the full I-CARE corpus on PhysioNet here: [**I-CARE v2.1**](https://physionet.org/content/i-care/2.1/).
3. Place the downloaded `.csv` BCI files into the `icare_project/data/` directory.
4. Update the data paths within `icare_project/model/cnn_bigru_model/data_loader.py` as needed.

## Installation and Requirements

1. Clone this repository:
```bash
git clone https://github.com/YourUsername/trustworthy-neuroprognostication.git
cd trustworthy-neuroprognostication
```

2. Install dependencies:
```bash
pip install torch torchvision torchaudio numpy pandas scikit-learn matplotlib seaborn xgboost
```

## Usage Instructions

### 1. Training the CNN-BiGRU framework
To run the full Cross-Validation training loop, uncertainty estimation, and evaluation for the deep neural network:
```bash
python icare_project/model/cnn_bigru_model/main.py
```

### 2. Running ML Baseline Ablations
To execute the traditional machine learning baselines:
```bash
python icare_project/experiments/ablations/train_eval_ablations.py
```

### 3. Generating Trustworthy AI Metrics & Visuals
To generate the exact figures seen in the paper (Calibration curves, Fairness bar plots, Risk-Coverage Tradeoff curves):
```bash
python icare_project/model/cnn_bigru_model/generate_visuals.py
```

## Citation
If you find this code or methodology useful in your research, please cite our paper:
> *Citation details will be updated upon publication.*

## License
This project is licensed under the MIT License - see the LICENSE file for details.


