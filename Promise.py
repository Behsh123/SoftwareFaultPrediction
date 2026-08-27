#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import warnings
import numpy as np
import pandas as pd

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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
from imblearn.over_sampling import SMOTE
from scipy import stats
import xgboost as xgb
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, LSTM, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings('ignore')
np.random.seed(42)
tf.random.set_seed(42)

BASE_DIR = '/mnt/c/Users/Behrooz/Desktop/deeplearningAricle'
DATA_DIR = os.path.join(BASE_DIR, 'data')
PROMISE_DIR = os.path.join(DATA_DIR, 'promise')
RESULTS_DIR = os.path.join(BASE_DIR, 'results_promise')
os.makedirs(RESULTS_DIR, exist_ok=True)

print("="*80)
print("Software Defect Prediction - Promise Dataset")
print("="*80)
print(f"Base Directory: {BASE_DIR}")
print(f"Data Directory: {PROMISE_DIR}")
print(f"Results Directory: {RESULTS_DIR}")
print("="*80)

print("\nLoading Promise datasets...")

def load_promise_data(file_path):
    try:
        df = pd.read_csv(file_path)
        
        non_numeric_cols = []
        for col in df.columns:
            if col != df.columns[-1]:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    if df[col].isna().all():
                        non_numeric_cols.append(col)
                except:
                    non_numeric_cols.append(col)
        
        if non_numeric_cols:
            df = df.drop(columns=non_numeric_cols)
            print(f"      Removed non-numeric columns: {non_numeric_cols}")
        
        df = df.dropna()
        
        if len(df) == 0:
            return None, None
        
        y = df.iloc[:, -1].values.astype(np.int32)
        X = df.iloc[:, :-1].values.astype(np.float32)
        
        y_binary = np.where(y > 0, 1, 0)
        
        print(f"      Original label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        print(f"      Binary label distribution: {dict(zip(*np.unique(y_binary, return_counts=True)))}")
        
        return X, y_binary
    except Exception as e:
        print(f"   Error loading {file_path}: {e}")
        return None, None

promise_projects = {}

if os.path.exists(PROMISE_DIR):
    csv_files = [f for f in os.listdir(PROMISE_DIR) if f.endswith('.csv')]
    
    selected_projects = ['ant-1.7', 'camel-1.6', 'ivy', 'jEdit-4.2', 'log4j-1.2', 'xerces-1.4']
    csv_files = [f for f in csv_files if f.replace('.csv', '') in selected_projects]
    
    print(f"   Found {len(csv_files)} CSV files: {csv_files}")
    
    if len(csv_files) == 0:
        print(f"   No CSV files found in {PROMISE_DIR}!")
        sys.exit(1)
    
    for file in csv_files:
        name = file.replace('.csv', '')
        file_path = os.path.join(PROMISE_DIR, file)
        print(f"   Loading dataset: {file}")
        X, y = load_promise_data(file_path)
        if X is not None and len(X) > 0:
            promise_projects[name] = {
                'X': X, 
                'y': y, 
                'n_samples': len(X),
                'n_features': X.shape[1]
            }
            print(f"   Loaded {name}: {X.shape[0]:,} samples, {X.shape[1]} features")
        else:
            print(f"   Failed to load {name}")
else:
    print(f"   Data directory {PROMISE_DIR} does not exist!")
    sys.exit(1)

print(f"\nSuccessfully loaded {len(promise_projects)} datasets")

if len(promise_projects) == 0:
    print("\nNo datasets loaded successfully!")
    sys.exit(1)

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
        GRU(32, return_sequences=True),
        BatchNormalization(),
        Dropout(0.2),
        GRU(16, return_sequences=False),
        BatchNormalization(),
        Dropout(0.2),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_lstm_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(32, return_sequences=True),
        BatchNormalization(),
        Dropout(0.2),
        LSTM(16, return_sequences=False),
        BatchNormalization(),
        Dropout(0.2),
        Dense(8, activation='relu'),
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
    
    n_pos = len(X[y==1])
    k_neighbors = min(5, n_pos - 1) if n_pos > 1 else 1
    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
    X_bal, y_bal = smote.fit_resample(X, y)
    
    window_size = min(4, len(X_bal) // 10)
    if window_size < 2:
        window_size = 2
    
    X_ts, y_ts = create_sequences(X_bal, y_bal, window_size)
    return X_ts, y_ts, window_size

def train_deep_model(X_train, y_train, X_test, y_test, model_type='gru', use_hybrid=False):
    try:
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
        
        callbacks = [EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]
        
        model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=10,
                  batch_size=min(16, len(X_tr)//5), callbacks=callbacks, verbose=0)
        
        y_pred_prob = model.predict(X_test_ts, verbose=0)
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        return {
            'f1': f1_score(y_test_ts, y_pred),
            'auc': roc_auc_score(y_test_ts, y_pred_prob),
            'accuracy': accuracy_score(y_test_ts, y_pred),
            'precision': precision_score(y_test_ts, y_pred),
            'recall': recall_score(y_test_ts, y_pred)
        }
    except Exception as e:
        print(f"      Error in deep model training: {e}")
        return None

def train_ml_model(X_train, y_train, X_test, y_test, model_type):
    try:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        n_pos = len(y_train[y_train==1])
        k_neighbors = min(5, n_pos - 1) if n_pos > 1 else 1
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_bal, y_bal = smote.fit_resample(X_train_scaled, y_train)
        
        models = {
            'xgboost': xgb.XGBClassifier(n_estimators=50, random_state=42, use_label_encoder=False, eval_metric='logloss'),
            'random_forest': RandomForestClassifier(n_estimators=50, random_state=42),
            'decision_tree': DecisionTreeClassifier(random_state=42),
            'svm': SVC(kernel='rbf', probability=True, random_state=42),
            'naive_bayes': GaussianNB()
        }
        
        model = models.get(model_type)
        if model is None:
            return None
        
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
    except Exception as e:
        print(f"      Error in ML model training: {e}")
        return None

methods = {
    'hybrid': {'type': 'deep', 'model': 'gru', 'use_hybrid': True, 'label': 'Our Method (GRU+Granulation+Causal)'},
    'gru_base': {'type': 'deep', 'model': 'gru', 'use_hybrid': False, 'label': 'GRU Baseline'},
    'xgboost': {'type': 'ml', 'model': 'xgboost', 'label': 'XGBoost'},
    'random_forest': {'type': 'ml', 'model': 'random_forest', 'label': 'Random Forest'},
}

print("\n" + "="*80)
print("Part 1: Within-Project Evaluation")
print("="*80)
print(f"Number of datasets: {len(promise_projects)}")
print('-'*80)

dataset_results = {}

for method_name, method_config in methods.items():
    print(f"\n   Testing: {method_config['label']}...")
    method_results = []
    
    for proj_name, data in promise_projects.items():
        X, y = data['X'], data['y']
        
        if len(X) < 20:
            print(f"      Skipping {proj_name}: insufficient samples ({len(X)} < 20)")
            continue
        
        if np.sum(y == 1) < 2:
            print(f"      Skipping {proj_name}: less than 2 positive samples")
            continue
            
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        except ValueError:
            print(f"      {proj_name}: Stratify failed, using simple split")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        if method_config['type'] == 'deep':
            result = train_deep_model(X_train, y_train, X_test, y_test, 
                                     method_config['model'], method_config.get('use_hybrid', False))
        else:
            result = train_ml_model(X_train, y_train, X_test, y_test, method_config['model'])
        
        if result is not None:
            method_results.append({'project': proj_name, **result})
            print(f"      {proj_name}: F1={result['f1']:.4f}")
        else:
            print(f"      {proj_name}: Failed")
    
    if method_results:
        df = pd.DataFrame(method_results)
        dataset_results[method_name] = df
        avg_f1 = df['f1'].mean()
        print(f"      Average for {method_config['label']}: F1={avg_f1:.4f}")

print("\n" + "="*80)
print("Part 2: Cross-Project Evaluation")
print("="*80)
print("   Training on one project, testing on another")
print("-"*80)

cross_results = []

if len(promise_projects) >= 2:
    best_project = 'xerces-1.4'
    if best_project not in promise_projects:
        best_project = list(promise_projects.keys())[0]
    
    print(f"\nTraining project = {best_project} ({promise_projects[best_project]['n_samples']:,} samples)")
    
    train_X = promise_projects[best_project]['X']
    train_y = promise_projects[best_project]['y']
    
    for test_name, data in promise_projects.items():
        if test_name == best_project:
            continue
        
        test_X = data['X']
        test_y = data['y']
        
        print(f"\n   Training: {best_project} -> Testing: {test_name} ({len(test_X)} samples)")
        
        result = train_deep_model(train_X, train_y, test_X, test_y, 'gru', use_hybrid=True)
        
        if result is not None:
            cross_results.append({
                'train': best_project,
                'test': test_name,
                'f1': result['f1'],
                'auc': result['auc'],
                'accuracy': result['accuracy']
            })
            print(f"      Result: F1={result['f1']:.4f}, AUC={result['auc']:.4f}")
        else:
            print(f"      Failed")
else:
    print("   Not enough projects for Cross-Project evaluation.")

print("\n" + "="*80)
print("Part 3: Summary Results")
print("="*80)

summary_df = pd.DataFrame()

for method_name, df in dataset_results.items():
    if len(df) > 0:
        temp_df = pd.DataFrame({
            'method': [methods[method_name]['label']],
            'f1': [df['f1'].mean() * 100],
            'auc': [df['auc'].mean() * 100],
            'accuracy': [df['accuracy'].mean() * 100],
            'precision': [df['precision'].mean() * 100],
            'recall': [df['recall'].mean() * 100]
        })
        summary_df = pd.concat([summary_df, temp_df], ignore_index=True)

if len(summary_df) > 0:
    summary_df = summary_df.sort_values('f1', ascending=False)
    
    print("\n" + "="*80)
    print("Performance Summary - Promise Dataset")
    print('='*80)
    print(summary_df.to_string(index=False, float_format='{:.2f}'.format))
    
    csv_path = os.path.join(RESULTS_DIR, 'summary_promise.csv')
    summary_df.to_csv(csv_path, index=False)
    print(f"\nSummary saved to: {csv_path}")
else:
    print("No results to summarize.")

if cross_results:
    cross_df = pd.DataFrame(cross_results)
    csv_path = os.path.join(RESULTS_DIR, 'cross_promise.csv')
    cross_df.to_csv(csv_path, index=False)
    print(f"Cross-Project results saved to: {csv_path}")

print("\n" + "="*80)
print("Execution Complete!")
print("="*80)

print("\nGenerated files:")
print(f"   - Summary CSV: {RESULTS_DIR}/summary_promise.csv")
if cross_results:
    print(f"   - Cross-Project CSV: {RESULTS_DIR}/cross_promise.csv")
print("="*80)