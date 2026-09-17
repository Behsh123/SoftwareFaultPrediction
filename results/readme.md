# From Correlation to Causation: ATE-FS for Interpretable Software Fault Prediction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8%2B-orange)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen)]()

Official implementation of the paper:

> **From Correlation to Causation: Average Treatment Effect-Based Feature Selection for Interpretable Software Fault Prediction**
> Behrooz Shahi, Hooman Tahayori
> Shiraz University, Shiraz, Iran

---

## 📌 Overview

Software fault prediction (SFP) is critical for ensuring software reliability and allocating testing resources effectively. However, existing approaches suffer from three key limitations:

1. **Static granulation thresholds** that do not adapt to individual features
2. **Black-box deep learning models** that provide predictions without explaining *why* certain modules are flagged as high-risk
3. **Correlation-based feature selection** that may retain spurious relationships rather than genuine causal mechanisms

To address these limitations, we introduce **ATE-FS** (*Average Treatment Effect-based Feature Selection*), a novel causal feature selection method that quantifies the **causal effect** of each software metric on fault occurrence. Unlike existing methods that only identify *whether* a relationship exists (e.g., PC, NOTEARS), ATE-FS provides a **quantitative estimate** of causal strength, enabling interpretable and actionable insights for developers.

---

## ✨ Key Contributions

1. **Adaptive Data Granulation:** A dynamic granulation mechanism that determines optimal abstraction levels for each feature based on Mutual Information with the target variable, effectively mitigating noise and outliers.

2. **Causal Feature Selection via ATE-FS (Core Novelty):** A novel method that estimates the Average Treatment Effect of each software metric on fault occurrence using logistic regression with bootstrap confidence intervals. Unlike correlation-based methods, ATE-FS provides interpretable and actionable insights by estimating *how much* each metric directly influences fault probability.

3. **Weighted Multi-Criteria Feature Selection:** A weighted optimization problem balancing predictive accuracy (F1-Score), model parsimony (number of selected features), and feature importance (SHAP-based scores), solved via a multi-start Tabu search algorithm.

4. **Comprehensive Empirical Validation:** Evaluation on **16 real-world projects** from the TRAVIS (10 projects) and PROMISE (6 projects) repositories, comparing against **7 baseline methods** with statistical validation using the Wilcoxon test.

5. **Actionable Interpretability:** Quantitative causal effects for each selected feature. For example, in the graylog2-server project, reducing Feature 14 from High to Low could decrease fault probability by approximately **18 percentage points** — enabling developers to prioritize refactoring efforts.

---

## 📊 Key Results

### Within-Project Results

| Dataset | Method | F1-Score | AUC | Precision | Recall |
|---------|--------|----------|-----|-----------|--------|
| **TRAVIS** | **Hybrid (ATE-FS+Granulation)** | **87.03%** | 55.99% | 80.91% | **94.64%** |
| TRAVIS | Hybrid (RF+Granulation) | 87.23% | 67.44% | 82.12% | 93.25% |
| TRAVIS | GRU Baseline | 84.28% | 70.80% | 82.91% | 86.22% |
| TRAVIS | LSTM | 84.75% | 70.46% | 83.11% | 87.13% |
| **PROMISE** | **Hybrid (ATE-FS+Granulation)** | **86.79%** | 82.61% | 86.97% | 88.78% |
| PROMISE | GRU Baseline | 87.20% | 83.41% | 87.49% | 87.51% |
| PROMISE | LSTM | 88.33% | 83.88% | 86.08% | 91.09% |

### Cross-Project Results (Leave-One-Out)

| Dataset | Average F1-Score |
|---------|------------------|
| **TRAVIS** | **70.78%** |
| **PROMISE** | **77.33%** |

### Statistical Significance (Wilcoxon Test)

| Dataset | Comparison | p-value | Result |
|---------|------------|---------|--------|
| TRAVIS | Proposed vs GRU | 0.0244 | **SIG** |
| TRAVIS | Proposed vs LSTM | 0.0186 | **SIG** |
| TRAVIS | Proposed vs ML Methods | < 0.001 | **SIG** |
| PROMISE | Proposed vs GRU | 0.5781 | NOT-SIG |
| PROMISE | Proposed vs LSTM | 0.6563 | NOT-SIG |
| PROMISE | Proposed vs ML Methods | < 0.05 | **SIG** |

### Key Highlights

- ✅ **Highest Recall on TRAVIS (94.64%)** — robust detection of actual faults
- ✅ **Significant feature reduction (73.7%)** — improved computational efficiency
- ✅ **Statistically significant improvements** (Wilcoxon test, p < 0.05)
- ✅ **Actionable causal insights** via ATE-FS

---

## 🗂️ Repository Structure
.
├── main.py # Main pipeline (ATE-FS implementation)
├── requirements.txt # Python dependencies
├── LICENSE # MIT License
├── README.md # This file
├── data/
│ ├── Travis/ # TRAVIS dataset (10 projects)
│ │ ├── cloudify.csv
│ │ ├── graylog2-server.csv
│ │ ├── jackrabbit-oak.csv
│ │ ├── jruby.csv
│ │ ├── metasploit-framework.csv
│ │ ├── open-build-service.csv
│ │ ├── openproject.csv
│ │ ├── rails.csv
│ │ ├── ruby.csv
│ │ └── sonarqube.csv
│ └── Promise/ # PROMISE dataset (6 projects)
│ ├── ant-1.7.csv
│ ├── camel-1.6.csv
│ ├── ivy.csv
│ ├── jEdit-4.2.csv
│ ├── log4j-1.2.csv
│ └── xerces-1.4.csv
├── results/
│ ├── figures/ # Generated figures (PNG)
│ └── tables/ # Generated CSV results
└── .gitignore

text

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/behroozshahi/software-fault-prediction-atefs.git
cd software-fault-prediction-atefs
2. Create a Virtual Environment (Recommended)
bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
# or
venv\Scripts\activate           # Windows
3. Install Dependencies
bash
pip install -r requirements.txt
4. Prepare the Datasets
Place the datasets in the following directories:

text
data/
├── Travis/              # Place TRAVIS CSV files here
└── Promise/             # Place PROMISE CSV files here
Dataset sources:

TRAVIS (TravisTorrent): https://travistorrent.testingwise.com/

PROMISE: http://promise.site.uottawa.ca/

5. Run the Pipeline
bash
python main.py
The pipeline will:

Load and preprocess the datasets (time-series transformation + SMOTE-Tomek)

Apply dynamic granulation based on Mutual Information

Perform causal feature selection via ATE-FS

Select optimal features via multi-start Tabu search

Train GRU classifiers and evaluate performance

Generate figures and tables in results/

📦 Requirements
text
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
imbalanced-learn>=0.10.0
xgboost>=1.5.0
tensorflow>=2.8.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
tqdm>=4.62.0
🔬 Methodology
The proposed framework consists of four integrated phases:

Phase 1: Preprocessing
Time-series transformation: Weekly aggregated records with sliding windows (window size = 4, step = 1)

SMOTE-Tomek balancing: Applied to training data only, within each project independently, to avoid temporal leakage

Phase 2: Granulation + Causal Feature Selection
Dynamic Granulation: Quartile-based with Mutual Information optimization. Each feature is transformed into three symbolic levels: Low (0), Medium (1), High (2).

ATE-FS: Estimates the Average Treatment Effect of each granulated feature on fault occurrence using logistic regression (S-Learner) with 30 bootstrap iterations. Features with 95% CI excluding zero and ATE > 0.005 are retained.

Phase 3: Feature Selection + Classification
Multi-start Tabu search: Weighted fitness function balancing F1-Score, model parsimony, and interpretability (SHAP-based).

GRU classifier: Three recurrent layers (128, 64, 32 units) with batch normalization and dropout (0.3).

Phase 4: Output
Fault prediction: F1-Score, AUC, Accuracy, Precision, Recall

Causal insights: ATE values for each selected feature

📈 Output Files
After running main.py, the following files are generated:

results/tables/
summary_TRAVIS.csv — Within-project results on TRAVIS

summary_PROMISE.csv — Within-project results on PROMISE

cross_TRAVIS.csv — Cross-project F1-Score matrix (TRAVIS)

cross_PROMISE.csv — Cross-project F1-Score matrix (PROMISE)

results/figures/
comparison_TRAVIS.png — F1-Score and AUC comparison (TRAVIS)

comparison_PROMISE.png — F1-Score and AUC comparison (PROMISE)

cross_heatmap_TRAVIS.png — Cross-project heatmap (TRAVIS)

cross_heatmap_PROMISE.png — Cross-project heatmap (PROMISE)

results/
final_summary.txt — Complete text summary of all results

📖 Citation
If you find this work useful, please cite:


