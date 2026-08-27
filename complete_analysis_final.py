#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Complete Unified Analysis - Final Version with Cross-Project and Statistical Tests
Includes: TRAVIS (10 projects) + ECLIPSE (3 projects)
Methods: Hybrid, GRU, LSTM, XGBoost, Random Forest, Decision Tree, SVM, Naive Bayes
Evaluation: Within-Project + Cross-Project + Wilcoxon + Visualizations
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
from sklearn.feature_selection import mutual_info_classif
from imblearn.combine import SMOTETomek
from scipy import stats
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, LSTM, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings('ignore')
np.random.seed(42)
tf.random.set_seed(42)

# ================================================================
# CONFIGURATION
# ================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
TRAVIS_DIR = os.path.join(DATA_DIR, 'Travis')
ECLIPSE_DIR = os.path.join(DATA_DIR, 'eclipse')
RESULTS_DIR = os.path.join(BASE_DIR, 'results_final_complete')

for d in [RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

print("="*100)
print("Complete Unified Analysis - Final Version")
print("Includes: Within-Project + Cross-Project + Wilcoxon + Visualizations")
print("="*100)

# ================================================================
# HELPER FUNCTIONS
# ================================================================

def load_travis_data(file_path):
    try:
        df = pd.read_csv(file_path, header=None)
        cols_to_drop = []
        for col in df.columns:
            sample = df[col].head(5).astype(str)
            if sample.str.contains('/').any() or sample.str.contains(':').any():
                cols_to_drop.append(col)
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
        X = df.iloc[:, 1:].values.astype(np.float32)
        y = df.iloc[:, 0].values.astype(np.int32)
        return X, y
    except:
        return None, None

def load_eclipse_data(file_path):
    try:
        df = pd.read_csv(file_path)
        y = df.iloc[:, -1].values.astype(np.int32)
        X = df.iloc[:, :-1].values.astype(np.float32)
        return X, y
    except:
        return None, None

def dynamic_granulation(X, y):
    n_samples, n_features = X.shape
    X_granular = np.zeros_like(X, dtype=np.int32)
    for feat_idx in range(n_features):
        feature_values = X[:, feat_idx]
        Q1 = np.percentile(feature_values, 25)
        Q3 = np.percentile(feature_values, 75)
        iqr = Q3 - Q1
        if iqr == 0:
            X_granular[:, feat_idx] = 1
            continue
        best_mi = -1
        best_p = 0.1
        for p in np.linspace(0.01, 0.5, 10):
            low_threshold = Q1 + p * iqr
            high_threshold = Q3 - p * iqr
            temp = np.zeros(n_samples, dtype=np.int32)
            for i, val in enumerate(feature_values):
                if val < low_threshold:
                    temp[i] = 0
                elif val > high_threshold:
                    temp[i] = 2
                else:
                    temp[i] = 1
            mi = mutual_info_classif(temp.reshape(-1, 1), y, random_state=42)[0]
            if mi > best_mi:
                best_mi = mi
                best_p = p
        low_threshold = Q1 + best_p * iqr
        high_threshold = Q3 - best_p * iqr
        for i, val in enumerate(feature_values):
            if val < low_threshold:
                X_granular[i, feat_idx] = 0
            elif val > high_threshold:
                X_granular[i, feat_idx] = 2
            else:
                X_granular[i, feat_idx] = 1
    return X_granular

def build_causal_graph(X, y, threshold=0.02):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    importances = rf.feature_importances_
    important = np.where(importances > threshold)[0]
    if len(important) == 0:
        important = np.argsort(importances)[-5:]
    elif len(important) < 5:
        top = np.argsort(importances)[-5:]
        important = list(important) + [i for i in top if i not in important]
    return list(important)

def create_sequences(X, y, window_size=4):
    X_ts, y_ts = [], []
    for i in range(window_size, len(X)):
        X_ts.append(X[i-window_size:i])
        y_ts.append(1 if np.sum(y[i-window_size:i]) > 0 else 0)
    return np.array(X_ts), np.array(y_ts)

def build_gru_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        GRU(64, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),
        GRU(32, return_sequences=False),
        BatchNormalization(),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_lstm_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(32, return_sequences=False),
        BatchNormalization(),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def preprocess_for_deep_learning(X, y, use_hybrid=False):
    if use_hybrid:
        X = dynamic_granulation(X, y)
        causal_features = build_causal_graph(X, y)
        X = X[:, causal_features]
    else:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    smt = SMOTETomek(random_state=42, sampling_strategy=1.0)
    X_bal, y_bal = smt.fit_resample(X, y)
    
    window_size = min(4, len(X_bal) // 10)
    if window_size < 2:
        window_size = 2
    
    X_ts, y_ts = create_sequences(X_bal, y_bal, window_size)
    return X_ts, y_ts, window_size

def train_deep_model(X_train, y_train, X_test, y_test, model_type='gru', use_hybrid=False):
    X_train_ts, y_train_ts, window_size = preprocess_for_deep_learning(X_train, y_train, use_hybrid)
    X_test_ts, y_test_ts, _ = preprocess_for_deep_learning(X_test, y_test, use_hybrid)
    
    if len(X_train_ts) == 0 or len(X_test_ts) == 0:
        return None
    
    if X_train_ts.shape[2] != X_test_ts.shape[2]:
        min_features = min(X_train_ts.shape[2], X_test_ts.shape[2])
        X_train_ts = X_train_ts[:, :, :min_features]
        X_test_ts = X_test_ts[:, :, :min_features]
    
    X_tr, X_val, y_tr, y_val = train_test_split(X_train_ts, y_train_ts, test_size=0.3, random_state=42)
    
    input_shape = (window_size, X_tr.shape[2])
    if model_type == 'gru':
        model = build_gru_model(input_shape)
    else:
        model = build_lstm_model(input_shape)
    
    callbacks = [EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]
    
    try:
        model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=30,
                  batch_size=min(64, len(X_tr)//10), callbacks=callbacks, verbose=0)
    except:
        return None
    
    y_pred_prob = model.predict(X_test_ts, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    return {
        'f1': f1_score(y_test_ts, y_pred),
        'auc': roc_auc_score(y_test_ts, y_pred_prob),
        'accuracy': accuracy_score(y_test_ts, y_pred),
        'precision': precision_score(y_test_ts, y_pred),
        'recall': recall_score(y_test_ts, y_pred)
    }

def train_ml_model(X_train, y_train, X_test, y_test, model_type):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    smt = SMOTETomek(random_state=42, sampling_strategy=1.0)
    X_bal, y_bal = smt.fit_resample(X_train_scaled, y_train)
    
    models = {
        'xgboost': xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss'),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'decision_tree': DecisionTreeClassifier(random_state=42),
        'svm': SVC(kernel='rbf', probability=True, random_state=42),
        'naive_bayes': GaussianNB()
    }
    
    model = models.get(model_type)
    if model is None:
        return None
    
    try:
        model.fit(X_bal, y_bal)
        y_pred = model.predict(X_test_scaled)
        y_pred_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        return {
            'f1': f1_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_pred_prob),
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred)
        }
    except:
        return None

# ================================================================
# LOAD DATASETS
# ================================================================
print("\nLoading datasets...")

travis_projects = {}
for file in os.listdir(TRAVIS_DIR):
    if file.endswith('.csv') and not file.startswith('new'):
        name = file.replace('.csv', '')
        file_path = os.path.join(TRAVIS_DIR, file)
        X, y = load_travis_data(file_path)
        if X is not None:
            travis_projects[name] = {'X': X, 'y': y}
            print(f"   TRAVIS - {name}: {X.shape[0]:,} samples, {X.shape[1]} features")

eclipse_projects = {}
for file in os.listdir(ECLIPSE_DIR):
    if file.endswith('.csv'):
        name = file.replace('.csv', '')
        file_path = os.path.join(ECLIPSE_DIR, file)
        X, y = load_eclipse_data(file_path)
        if X is not None:
            eclipse_projects[name] = {'X': X, 'y': y}
            print(f"   ECLIPSE - {name}: {X.shape[0]:,} samples, {X.shape[1]} features")

print(f"\nTRAVIS: {len(travis_projects)} projects, ECLIPSE: {len(eclipse_projects)} projects")

# ================================================================
# DEFINE METHODS
# ================================================================
methods = {
    'hybrid': {'type': 'deep', 'model': 'gru', 'use_hybrid': True, 'label': 'Our Method'},
    'gru_base': {'type': 'deep', 'model': 'gru', 'use_hybrid': False, 'label': 'GRU Baseline'},
    'lstm': {'type': 'deep', 'model': 'lstm', 'use_hybrid': False, 'label': 'LSTM'},
    'xgboost': {'type': 'ml', 'model': 'xgboost', 'label': 'XGBoost'},
    'random_forest': {'type': 'ml', 'model': 'random_forest', 'label': 'Random Forest'},
    'decision_tree': {'type': 'ml', 'model': 'decision_tree', 'label': 'Decision Tree'},
    'svm': {'type': 'ml', 'model': 'svm', 'label': 'SVM'},
    'naive_bayes': {'type': 'ml', 'model': 'naive_bayes', 'label': 'Naive Bayes'}
}

# ================================================================
# SECTION 1: Within-Project Evaluation
# ================================================================
print("\n" + "="*100)
print("SECTION 1: Within-Project Evaluation")
print("="*100)

all_results = {}
all_datasets = {'TRAVIS': travis_projects, 'ECLIPSE': eclipse_projects}

for dataset_name, projects in all_datasets.items():
    print(f"\nDataset: {dataset_name} ({len(projects)} projects)")
    print('-'*80)
    
    dataset_results = {}
    
    for method_name, method_config in methods.items():
        print(f"\n   Method: {method_config['label']}...")
        method_results = []
        
        for proj_name, data in projects.items():
            X, y = data['X'], data['y']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
            
            if method_config['type'] == 'deep':
                result = train_deep_model(X_train, y_train, X_test, y_test, 
                                         method_config['model'], method_config.get('use_hybrid', False))
            else:
                result = train_ml_model(X_train, y_train, X_test, y_test, method_config['model'])
            
            if result is not None:
                method_results.append({'project': proj_name, **result})
        
        if method_results:
            df = pd.DataFrame(method_results)
            dataset_results[method_name] = df
    
    all_results[dataset_name] = dataset_results

# ================================================================
# SECTION 2: Cross-Project Evaluation (Our Method Only)
# ================================================================
print("\n" + "="*100)
print("SECTION 2: Cross-Project Evaluation (Our Method)")
print("="*100)

cross_results = {}

for dataset_name, projects in all_datasets.items():
    project_names = list(projects.keys())
    n = len(project_names)
    
    if n < 2:
        print(f"\nWARNING: {dataset_name}: Not enough projects for Cross-Project evaluation.")
        continue
    
    print(f"\nDataset: {dataset_name} ({n} projects)")
    print('-'*80)
    
    cross_matrix = np.zeros((n, n))
    
    for i, train_name in enumerate(project_names):
        train_X = projects[train_name]['X']
        train_y = projects[train_name]['y']
        
        for j, test_name in enumerate(project_names):
            test_X = projects[test_name]['X']
            test_y = projects[test_name]['y']
            
            print(f"   Train: {train_name} -> Test: {test_name}...", end=" ")
            
            result = train_deep_model(train_X, train_y, test_X, test_y, 'gru', use_hybrid=True)
            
            if result is not None:
                cross_matrix[i, j] = result['f1']
                print(f"F1={result['f1']:.4f}")
            else:
                cross_matrix[i, j] = 0.0
                print("ERROR")
    
    cross_df = pd.DataFrame(cross_matrix, index=project_names, columns=project_names)
    cross_results[dataset_name] = cross_df
    
    print(f"\nCross-Project Matrix - {dataset_name}:")
    print(cross_df.round(4))
    
    diag = np.diag(cross_matrix)
    off_diag = cross_matrix[~np.eye(n, dtype=bool)]
    
    within_avg = np.mean(diag) if len(diag) > 0 else 0
    cross_avg = np.mean(off_diag) if len(off_diag) > 0 else 0
    
    print(f"\nCross-Project Statistics - {dataset_name}:")
    print(f"   Average Within-Project: {within_avg:.4f}")
    print(f"   Average Cross-Project: {cross_avg:.4f}")
    print(f"   Difference: {within_avg - cross_avg:.4f}")

# ================================================================
# SECTION 3: Wilcoxon Statistical Test
# ================================================================
print("\n" + "="*100)
print("SECTION 3: Wilcoxon Statistical Test")
print("="*100)

wilcoxon_results = {}

for dataset_name, dataset_results in all_results.items():
    print(f"\nDataset: {dataset_name}")
    print('-'*80)
    
    hybrid_df = dataset_results.get('hybrid')
    if hybrid_df is None:
        print("   WARNING: Our Method data not available.")
        continue
    
    hybrid_f1 = hybrid_df['f1'].values
    
    for method_name, df in dataset_results.items():
        if method_name == 'hybrid':
            continue
        
        if len(df) == 0:
            continue
        
        other_f1 = df['f1'].values
        
        if len(hybrid_f1) >= 5 and len(other_f1) >= 5:
            try:
                stat, p_value = stats.wilcoxon(hybrid_f1, other_f1, alternative='greater')
                
                wilcoxon_results[f"{dataset_name}_{method_name}"] = {
                    'method': methods[method_name]['label'],
                    'statistic': stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
                
                print(f"\n   Our Method vs {methods[method_name]['label']}:")
                print(f"      W-statistic: {stat:.4f}")
                print(f"      p-value: {p_value:.6f}")
                print(f"      {'Significant' if p_value < 0.05 else 'Not significant'}")
            except:
                print(f"   WARNING: Error in Wilcoxon test for {method_name}")

# ================================================================
# SECTION 4: Final Results Display
# ================================================================
print("\n" + "="*100)
print("SECTION 4: Final Results")
print("="*100)

summary_tables = {}

for dataset_name, dataset_results in all_results.items():
    print(f"\n{'='*80}")
    print(f"Dataset: {dataset_name}")
    print('='*80)
    
    summary = []
    for method_name, df in dataset_results.items():
        if len(df) > 0:
            summary.append({
                'method': methods[method_name]['label'],
                'f1': df['f1'].mean(),
                'auc': df['auc'].mean(),
                'accuracy': df['accuracy'].mean(),
                'precision': df['precision'].mean(),
                'recall': df['recall'].mean()
            })
    
    summary_df = pd.DataFrame(summary)
    summary_df = summary_df.sort_values('f1', ascending=False)
    summary_tables[dataset_name] = summary_df
    
    print(summary_df.to_string(index=False, float_format='{:.4f}'.format))
    
    csv_path = os.path.join(RESULTS_DIR, f'summary_{dataset_name}.csv')
    summary_df.to_csv(csv_path, index=False)
    print(f"\nSaved {dataset_name} results: {csv_path}")

for dataset_name, cross_df in cross_results.items():
    csv_path = os.path.join(RESULTS_DIR, f'cross_{dataset_name}.csv')
    cross_df.to_csv(csv_path)
    print(f"Saved Cross-Project matrix {dataset_name}: {csv_path}")

# ================================================================
# SECTION 5: Visualizations
# ================================================================
print("\n" + "="*100)
print("SECTION 5: Visualizations")
print("="*100)

for dataset_name, summary_df in summary_tables.items():
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Method Comparison - {dataset_name}', fontsize=16, fontweight='bold')
    
    axes[0].bar(summary_df['method'], summary_df['f1'], color='skyblue', edgecolor='black')
    axes[0].axhline(y=summary_df['f1'].max(), color='red', linestyle='--', 
                    linewidth=2, label=f'Best: {summary_df["f1"].max():.3f}')
    axes[0].set_title('F1-Score', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('F1-Score')
    axes[0].set_ylim([0, 1])
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis='x', rotation=45)
    
    metrics_melt = summary_df.melt(id_vars=['method'], 
                                   value_vars=['f1', 'auc', 'accuracy', 'precision', 'recall'],
                                   var_name='Metric', value_name='Score')
    sns.barplot(data=metrics_melt, x='method', y='Score', hue='Metric', ax=axes[1])
    axes[1].set_title('All Metrics Comparison', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Score')
    axes[1].set_ylim([0, 1])
    axes[1].legend(loc='lower right')
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, f'comparison_{dataset_name}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot {dataset_name}: {plot_path}")
    
    if dataset_name in cross_results:
        cross_df = cross_results[dataset_name]
        if len(cross_df) > 1:
            plt.figure(figsize=(12, 10))
            sns.heatmap(cross_df, annot=True, fmt='.3f', cmap='RdYlGn', 
                        center=0.5, vmin=0, vmax=1,
                        cbar_kws={'label': 'F1-Score'})
            plt.title(f'Cross-Project Matrix - {dataset_name}', fontsize=16, fontweight='bold')
            plt.xlabel('Test Project')
            plt.ylabel('Train Project')
            plt.tight_layout()
            plot_path = os.path.join(RESULTS_DIR, f'cross_heatmap_{dataset_name}.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Saved Cross-Project heatmap {dataset_name}: {plot_path}")

# ================================================================
# SECTION 6: Final Summary
# ================================================================
print("\n" + "="*100)
print("SECTION 6: Final Summary for Paper")
print("="*100)

summary_path = os.path.join(RESULTS_DIR, 'final_summary.txt')
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write("="*100 + "\n")
    f.write("FINAL SUMMARY - COMPLETE ANALYSIS\n")
    f.write("="*100 + "\n\n")
    
    for dataset_name, summary_df in summary_tables.items():
        f.write(f"\n{'='*80}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write('='*80 + "\n\n")
        f.write(summary_df.to_string(index=False, float_format='{:.4f}'.format))
        
        best = summary_df.iloc[0]
        f.write(f"\n\nBest Method: {best['method']}")
        f.write(f"\n   F1-Score: {best['f1']:.4f} ({best['f1']*100:.2f}%)")
        f.write("\n")
    
    f.write(f"\n\n{'='*80}\n")
    f.write("CROSS-PROJECT SUMMARY\n")
    f.write('='*80 + "\n")
    
    for dataset_name, cross_df in cross_results.items():
        if len(cross_df) > 1:
            diag = np.diag(cross_df.values)
            off_diag = cross_df.values[~np.eye(len(cross_df), dtype=bool)]
            
            f.write(f"\n{dataset_name}:\n")
            f.write(f"   Average Within-Project: {np.mean(diag):.4f}\n")
            f.write(f"   Average Cross-Project: {np.mean(off_diag):.4f}\n")
            f.write(f"   Difference: {np.mean(diag) - np.mean(off_diag):.4f}\n")
    
    f.write(f"\n\n{'='*80}\n")
    f.write("WILCOXON TEST (Our Method vs Other Methods)\n")
    f.write('='*80 + "\n")
    
    for key, result in wilcoxon_results.items():
        dataset, method = key.split('_')
        f.write(f"\n{dataset} - {result['method']}:\n")
        f.write(f"   p-value: {result['p_value']:.6f}\n")
        f.write(f"   {'Significant' if result['significant'] else 'Not significant'}\n")

print(f"\nFinal summary saved: {summary_path}")

# ================================================================
# COMPLETION
# ================================================================
print("\n" + "="*100)
print("ANALYSIS COMPLETE!")
print("="*100)

print("\nOutput files generated:")
print(f"   - Final summary: {summary_path}")
print(f"   - Summary tables: {RESULTS_DIR}/summary_*.csv")
print(f"   - Cross-Project matrices: {RESULTS_DIR}/cross_*.csv")
print(f"   - Visualizations: {RESULTS_DIR}/*.png")
print("="*100)