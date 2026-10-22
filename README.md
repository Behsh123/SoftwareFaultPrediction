# Adaptive Treatment-Effect-Based Framework for Software Fault Prediction

This repository contains the official implementation of the paper: **"An Adaptive Treatment-Effect-Based Framework for Software Fault Prediction: Insights from Temporal and Static Datasets"**.

## ?? Overview
The framework dynamically routes the prediction pipeline based on the nature of the software data:
- **Temporal Data** (e.g., TRAVIS): Granulation ? ATE-FS ? Tabu Search ? SMOTE-Tomek ? GRU.
- **Static Data** (e.g., PROMISE, ZMD-296k): Feature Engineering ? ATE-FS ? SMOTE ? Ensemble (XGBoost + Random Forest).

## ?? Dataset Preparation
Before running the code, place your datasets in the `data/` folder as follows:
- `data/Travis/` : Contains CSV files for the TRAVIS dataset.
- `data/Promise/` : Contains CSV files for the PROMISE dataset.
- `data/ZMD-296k/` : Contains CSV files for the Zenodo Modern Dataset.

## ?? How to Run
1. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   python main.py