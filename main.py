#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Software Fault Prediction with ATE-FS (Average Treatment Effect Feature Selection)
TRAVIS (10) + PROMISE (6) | Within-Project + Cross-Project
"""

import os
import sys
import time
import gc
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (f1_score, roc_auc_score, accuracy_score,
                              precision_score, recall_score)
from sklearn.feature_selection import mutual_info_classif
from imblearn.combine import SMOTETomek
from scipy import stats
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (GRU, LSTM, Dense, Dropout,
                                       Input, BatchNormalization)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tqdm import tqdm

warnings.filterwarnings('ignore')
np.random.seed(42)
tf.random.set_seed(42)

print("[OK] All libraries loaded")

# ================================================================
# Paths
# ================================================================
BASE_DIR = "/mnt/c/Users/Behrooz/Desktop/deeplearningAricle"
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAVIS_DIR = os.path.join(DATA_DIR, "Travis")
PROMISE_DIR = os.path.join(DATA_DIR, "Promise")
RESULTS_DIR = os.path.join(BASE_DIR, "results_ate")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")

for d in [RESULTS_DIR, FIGURES_DIR, TABLES_DIR]:
    os.makedirs(d, exist_ok=True)


# ================================================================
# Data loading
# ================================================================

def load_travis_data(file_path):
    """TRAVIS: no header, col 0 = target, last col = date"""
    try:
        df = pd.read_csv(file_path, header=None)
        df = df.iloc[:, :-1]
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.dropna(axis=1, how='all').dropna()
        
        y_raw = df.iloc[:, 0].values
        X = df.iloc[:, 1:].values.astype(np.float32)
        
        if len(np.unique(y_raw)) > 2:
            y = (y_raw > 0).astype(np.int32)
        else:
            y = y_raw.astype(np.int32)
        
        return X, y
    except Exception as e:
        print("   [ERR] " + str(e))
        return None, None


def load_promise_data(file_path):
    """PROMISE: header, drop name/version/name.1, target=bug (binary)"""
    try:
        df = pd.read_csv(file_path)
        
        cols_to_drop = ['name', 'version', 'name.1']
        existing = [c for c in cols_to_drop if c in df.columns]
        df = df.drop(columns=existing)
        
        object_cols = df.select_dtypes(include=['object']).columns.tolist()
        df = df.drop(columns=object_cols).dropna()
        
        target_col = 'bug' if 'bug' in df.columns else df.columns[-1]
        
        y_raw = df[target_col].values
        y = (y_raw > 0).astype(np.int32)
        
        X = df.drop(columns=[target_col]).values.astype(np.float32)
        
        return X, y
    except Exception as e:
        print("   [ERR] " + str(e))
        return None, None


# ================================================================
# Granulation
# ================================================================

def dynamic_granulation(X, y):
    n_samples, n_features = X.shape
    X_granular = np.zeros_like(X, dtype=np.int32)
    
    for feat_idx in range(n_features):
        fv = X[:, feat_idx]
        Q1 = np.percentile(fv, 25)
        Q3 = np.percentile(fv, 75)
        iqr = Q3 - Q1
        
        if iqr == 0:
            X_granular[:, feat_idx] = 1
            continue
        
        best_mi, best_p = -1, 0.1
        for p in np.linspace(0.01, 0.5, 10):
            lo = Q1 + p * iqr
            hi = Q3 - p * iqr
            
            temp = np.ones(n_samples, dtype=np.int32)
            temp[fv < lo] = 0
            temp[fv > hi] = 2
            
            try:
                mi = mutual_info_classif(temp.reshape(-1, 1), y,
                                          random_state=42)[0]
            except:
                mi = 0
            
            if mi > best_mi:
                best_mi, best_p = mi, p
        
        lo = Q1 + best_p * iqr
        hi = Q3 - best_p * iqr
        
        X_granular[fv < lo, feat_idx] = 0
        X_granular[fv > hi, feat_idx] = 2
        X_granular[(fv >= lo) & (fv <= hi), feat_idx] = 1
    
    return X_granular


# ================================================================
# Causal Discovery - ATE-FS (NEW)
# ================================================================

def estimate_ate_feature(X, y, feature_idx, n_bootstrap=30):
    """
    تخمین Average Treatment Effect برای یک feature
    
    Treatment: High (2) vs Low (0)
    """
    fv = X[:, feature_idx]
    
    treatment = (fv == 2).astype(int)
    control = (fv == 0).astype(int)
    
    mask = (treatment == 1) | (control == 1)
    if mask.sum() < 30:
        return 0.0, 0.0, 0.0
    
    X_masked = X[mask]
    y_masked = y[mask]
    t_masked = treatment[mask]
    
    # اطمینان از دوتایی بودن
    if len(np.unique(t_masked)) < 2:
        return 0.0, 0.0, 0.0
    
    if len(np.unique(y_masked)) < 2:
        return 0.0, 0.0, 0.0
    
    try:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_masked)
        
        # اضافه کردن treatment به featureها
        X_with_t = np.column_stack([X_scaled, t_masked])
        
        ates = []
        for _ in range(n_bootstrap):
            idx = np.random.choice(len(y_masked), len(y_masked), replace=True)
            X_boot = X_with_t[idx]
            y_boot = y_masked[idx]
            
            if len(np.unique(y_boot)) < 2:
                continue
            
            try:
                model = LogisticRegression(max_iter=500, random_state=42)
                model.fit(X_boot, y_boot)
                
                X_t1 = X_boot.copy()
                X_t1[:, -1] = 1
                X_t0 = X_boot.copy()
                X_t0[:, -1] = 0
                
                p1 = model.predict_proba(X_t1)[:, 1]
                p0 = model.predict_proba(X_t0)[:, 1]
                
                ate = np.mean(p1 - p0)
                ates.append(ate)
            except:
                continue
        
        if len(ates) < 5:
            return 0.0, 0.0, 0.0
        
        ate_mean = np.mean(ates)
        ci_lower = np.percentile(ates, 2.5)
        ci_upper = np.percentile(ates, 97.5)
        
        return ate_mean, ci_lower, ci_upper
    except:
        return 0.0, 0.0, 0.0


def build_causal_graph_rf(X, y, threshold=0.02):
    """Random Forest importance (fallback)"""
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    imp = rf.feature_importances_
    
    important = np.where(imp > threshold)[0]
    if len(important) < 5:
        top = np.argsort(imp)[-5:]
        important = list(set(list(important) + list(top)))
    
    return list(important)


def build_causal_graph_ate(X, y, min_features=5, max_features=15,
                            min_ate=0.005, min_ci=0.0):
    """
    ATE-FS: انتخاب feature بر اساس Average Treatment Effect
    
    معیار انتخاب: 
    - ATE > min_ate
    - ci_lower > min_ci (یعنی CI شامل 0 نباشه)
    """
    n_features = X.shape[1]
    ate_scores = {}
    
    print("      [ATE-FS] Computing ATE for " + str(n_features) + " features...")
    
    for i in range(n_features):
        ate, ci_lower, ci_upper = estimate_ate_feature(X, y, i)
        
        # فقط اگه CI شامل 0 نباشه و ATE معنادار باشه
        if ci_lower > min_ci and abs(ate) > min_ate:
            ate_scores[i] = abs(ate)
    
    sorted_features = sorted(ate_scores.keys(),
                              key=lambda x: ate_scores[x],
                              reverse=True)
    
    print("      [ATE-FS] " + str(len(sorted_features)) + 
          " features with significant ATE", flush=True)
    
    # نمایش top features
    for f in sorted_features[:5]:
        print("        Feature " + str(f) + ": ATE=" + 
              str(round(ate_scores[f], 4)), flush=True)
    
    # اطمینان از حداقل تعداد
    if len(sorted_features) < min_features:
        print("      [ATE-FS] Only " + str(len(sorted_features)) + 
              ", supplementing with RF", flush=True)
        rf_features = build_causal_graph_rf(X, y)
        for f in rf_features:
            if f not in sorted_features:
                sorted_features.append(f)
            if len(sorted_features) >= min_features:
                break
    
    if len(sorted_features) > max_features:
        sorted_features = sorted_features[:max_features]
    
    return sorted_features


# ================================================================
# Time Series
# ================================================================

def create_sequences(X, y, window_size=4):
    X_ts, y_ts = [], []
    for i in range(window_size, len(X)):
        X_ts.append(X[i-window_size:i])
        y_ts.append(1 if np.sum(y[i-window_size:i]) > 0 else 0)
    return np.array(X_ts), np.array(y_ts)


# ================================================================
# Preprocessing
# ================================================================

def preprocess_for_deep_learning(X, y, use_hybrid=False, method='ate'):
    if use_hybrid:
        X = dynamic_granulation(X, y)
        
        if method == 'ate':
            causal = build_causal_graph_ate(X, y)
        elif method == 'rf':
            causal = build_causal_graph_rf(X, y)
        else:
            causal = build_causal_graph_ate(X, y)
        
        X = X[:, causal]
    else:
        X = StandardScaler().fit_transform(X)
    
    try:
        smt = SMOTETomek(random_state=42, sampling_strategy=1.0)
        X_bal, y_bal = smt.fit_resample(X, y)
    except:
        X_bal, y_bal = X, y
    
    ws = max(2, min(4, len(X_bal) // 10))
    X_ts, y_ts = create_sequences(X_bal, y_bal, ws)
    return X_ts, y_ts, ws


# ================================================================
# Models
# ================================================================

def build_gru(input_shape):
    m = Sequential([
        Input(shape=input_shape),
        GRU(64, return_sequences=True),
        BatchNormalization(), Dropout(0.3),
        GRU(32), BatchNormalization(), Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    m.compile(optimizer=Adam(0.001), loss='binary_crossentropy',
              metrics=['accuracy'])
    return m


def build_lstm(input_shape):
    m = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=True),
        BatchNormalization(), Dropout(0.3),
        LSTM(32), BatchNormalization(), Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    m.compile(optimizer=Adam(0.001), loss='binary_crossentropy',
              metrics=['accuracy'])
    return m


def train_deep(X_train, y_train, X_test, y_test, model_type='gru',
               use_hybrid=False, method='ate'):
    try:
        X_tr_ts, y_tr_ts, ws = preprocess_for_deep_learning(
            X_train, y_train, use_hybrid, method)
        X_te_ts, y_te_ts, _ = preprocess_for_deep_learning(
            X_test, y_test, use_hybrid, method)
        
        if len(X_tr_ts) == 0 or len(X_te_ts) == 0:
            return None
        
        if X_tr_ts.shape[2] != X_te_ts.shape[2]:
            m = min(X_tr_ts.shape[2], X_te_ts.shape[2])
            X_tr_ts, X_te_ts = X_tr_ts[:,:,:m], X_te_ts[:,:,:m]
        
        if len(X_te_ts.shape) == 2:
            X_te_ts = X_te_ts.reshape(X_te_ts.shape[0], X_te_ts.shape[1], 1)
        
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_tr_ts, y_tr_ts, test_size=0.3, random_state=42)
        
        input_shape = (X_tr.shape[1], X_tr.shape[2])
        model = build_gru(input_shape) if model_type == 'gru' \
                else build_lstm(input_shape)
        
        cb = [EarlyStopping(monitor='val_loss', patience=10,
                             restore_best_weights=True)]
        
        model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
                  epochs=30, batch_size=min(64, max(8, len(X_tr)//10)),
                  callbacks=cb, verbose=0)
        
        y_prob = model.predict(X_te_ts, verbose=0).flatten()
        y_pred = (y_prob > 0.5).astype(int)
        
        return {
            'f1': f1_score(y_te_ts, y_pred, zero_division=0),
            'auc': roc_auc_score(y_te_ts, y_prob),
            'accuracy': accuracy_score(y_te_ts, y_pred),
            'precision': precision_score(y_te_ts, y_pred, zero_division=0),
            'recall': recall_score(y_te_ts, y_pred, zero_division=0)
        }
    except Exception as e:
        print("      [DEEP-ERR] " + str(e))
        return None


def train_ml(X_train, y_train, X_test, y_test, model_type):
    try:
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_train)
        X_te = sc.transform(X_test)
        
        try:
            smt = SMOTETomek(random_state=42, sampling_strategy=1.0)
            X_bal, y_bal = smt.fit_resample(X_tr, y_train)
        except:
            X_bal, y_bal = X_tr, y_train
        
        models = {
            'xgboost': xgb.XGBClassifier(n_estimators=100, random_state=42,
                                          use_label_encoder=False,
                                          eval_metric='logloss'),
            'random_forest': RandomForestClassifier(n_estimators=100,
                                                     random_state=42),
            'decision_tree': DecisionTreeClassifier(random_state=42),
            'svm': SVC(kernel='rbf', probability=True, random_state=42),
            'naive_bayes': GaussianNB()
        }
        
        model = models.get(model_type)
        if model is None:
            return None
        
        model.fit(X_bal, y_bal)
        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)[:,1] if hasattr(model,
                                                            'predict_proba') else y_pred
        
        return {
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc': roc_auc_score(y_test, y_prob),
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0)
        }
    except Exception as e:
        print("      [ML-ERR] " + str(e))
        return None


# ================================================================
# Load data
# ================================================================

print("\n[LOAD] Loading datasets...")

travis_projects = {}
promise_projects = {}

if os.path.exists(TRAVIS_DIR):
    files = [f for f in os.listdir(TRAVIS_DIR) if f.endswith('.csv')]
    print("\n   TRAVIS: " + str(len(files)) + " files")
    for file in files:
        name = file.replace('.csv', '')
        X, y = load_travis_data(os.path.join(TRAVIS_DIR, file))
        if X is not None and len(X) > 50:
            travis_projects[name] = {'X': X, 'y': y}
            print("   [OK] " + name + ": " + str(X.shape[0]) + " x " + 
                  str(X.shape[1]))

if os.path.exists(PROMISE_DIR):
    files = [f for f in os.listdir(PROMISE_DIR) if f.endswith('.csv')]
    print("\n   PROMISE: " + str(len(files)) + " files")
    for file in files:
        name = file.replace('.csv', '')
        X, y = load_promise_data(os.path.join(PROMISE_DIR, file))
        if X is not None and len(X) > 50:
            promise_projects[name] = {'X': X, 'y': y}
            print("   [OK] " + name + ": " + str(X.shape[0]) + " x " + 
                  str(X.shape[1]))

print("\n[TOTAL] TRAVIS: " + str(len(travis_projects)) + 
      " | PROMISE: " + str(len(promise_projects)))

if len(travis_projects) == 0 and len(promise_projects) == 0:
    print("\n[ERROR] No data loaded!")
    sys.exit(1)


# ================================================================
# Methods
# ================================================================

methods = {
    'hybrid_ate': {'type':'deep','model':'gru','use_hybrid':True,
                    'method':'ate','label':'Hybrid (ATE-FS+Granulation)'},
    'hybrid_rf': {'type':'deep','model':'gru','use_hybrid':True,
                   'method':'rf','label':'Hybrid (RF+Granulation)'},
    'gru_base': {'type':'deep','model':'gru','use_hybrid':False,
                  'method':'ate','label':'GRU Baseline'},
    'lstm': {'type':'deep','model':'lstm','use_hybrid':False,
             'method':'ate','label':'LSTM'},
    'xgboost': {'type':'ml','model':'xgboost','label':'XGBoost'},
    'random_forest': {'type':'ml','model':'random_forest','label':'Random Forest'},
    'decision_tree': {'type':'ml','model':'decision_tree','label':'Decision Tree'},
    'svm': {'type':'ml','model':'svm','label':'SVM'},
    'naive_bayes': {'type':'ml','model':'naive_bayes','label':'Naive Bayes'}
}


# ================================================================
# Part 1: Within-Project
# ================================================================

print("\n" + "="*100)
print("[PART 1] Within-Project Evaluation")
print("="*100)

all_results = {}
all_datasets = {'TRAVIS': travis_projects, 'PROMISE': promise_projects}

for ds_name, projects in all_datasets.items():
    if not projects:
        continue
    
    print("\n[DATASET] " + ds_name + " (" + str(len(projects)) + " projects)")
    print('-'*80)
    
    ds_results = {}
    
    for m_name, m_cfg in methods.items():
        print("\n   [METHOD] " + m_cfg['label'])
        results = []
        
        for p_name, data in tqdm(projects.items(), desc="      " + m_name):
            X, y = data['X'], data['y']
            if len(np.unique(y)) < 2:
                continue
            
            try:
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X, y, test_size=0.3, random_state=42, stratify=y)
            except:
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X, y, test_size=0.3, random_state=42)
            
            if m_cfg['type'] == 'deep':
                r = train_deep(X_tr, y_tr, X_te, y_te, m_cfg['model'],
                                m_cfg.get('use_hybrid',False),
                                m_cfg.get('method','ate'))
            else:
                r = train_ml(X_tr, y_tr, X_te, y_te, m_cfg['model'])
            
            if r is not None:
                results.append({'project': p_name, **r})
        
        if results:
            ds_results[m_name] = pd.DataFrame(results)
    
    all_results[ds_name] = ds_results


# ================================================================
# Part 2: Cross-Project (Optimized)
# ================================================================

print("\n" + "="*100)
print("[PART 2] Cross-Project Evaluation (Optimized)")
print("="*100)

cross_results = {}

for ds_name, projects in all_datasets.items():
    names = list(projects.keys())
    n = len(names)
    if n < 2:
        continue
    
    print("\n[DATASET] " + ds_name + " (" + str(n) + " projects)")
    print('-'*80)
    
    cm = np.zeros((n, n))
    
    for i, tr_name in enumerate(names):
        print("\n   [TRAIN] " + tr_name + "...")
        
        tr_X = projects[tr_name]['X']
        tr_y = projects[tr_name]['y']
        
        # Granulation + ATE-FS فقط ۱ بار
        try:
            tr_X_gran = dynamic_granulation(tr_X, tr_y)
            causal_features = build_causal_graph_ate(tr_X_gran, tr_y)
            tr_X_sel = tr_X_gran[:, causal_features]
            n_causal = len(causal_features)
            print("   [ATE-FS] " + str(n_causal) + " features selected")
        except Exception as e:
            print("   [ERR] ATE-FS failed: " + str(e))
            continue
        
        # SMOTE + time series
        try:
            smt = SMOTETomek(random_state=42, sampling_strategy=1.0)
            tr_X_bal, tr_y_bal = smt.fit_resample(tr_X_sel, tr_y)
            ws = max(2, min(4, len(tr_X_bal) // 10))
            tr_X_ts, tr_y_ts = create_sequences(tr_X_bal, tr_y_bal, ws)
        except Exception as e:
            print("   [ERR] SMOTE failed: " + str(e))
            continue
        
        if len(tr_X_ts) == 0:
            continue
        
        # Train GRU فقط ۱ بار
        try:
            X_tr, X_val, y_tr, y_val = train_test_split(
                tr_X_ts, tr_y_ts, test_size=0.3, random_state=42)
            
            input_shape = (X_tr.shape[1], X_tr.shape[2])
            model = build_gru(input_shape)
            
            cb = [EarlyStopping(monitor='val_loss', patience=10,
                                 restore_best_weights=True)]
            
            model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
                      epochs=30, batch_size=min(64, max(8, len(X_tr)//10)),
                      callbacks=cb, verbose=0)
            
            print("   [GRU] trained")
        except Exception as e:
            print("   [ERR] GRU failed: " + str(e))
            continue
        
        # تست روی همه پروژه‌های دیگه
        for j, te_name in enumerate(names):
            if i == j:
                cm[i,j] = np.nan
                continue
            
            print("      -> " + te_name + "...", end=" ", flush=True)
            
            try:
                te_X = projects[te_name]['X']
                te_y = projects[te_name]['y']
                
                te_X_gran = dynamic_granulation(te_X, te_y)
                
                # تطابق ابعاد
                if te_X_gran.shape[1] != tr_X_gran.shape[1]:
                    m = min(te_X_gran.shape[1], tr_X_gran.shape[1])
                    te_X_gran = te_X_gran[:, :m]
                    causal_features_adj = [c for c in causal_features if c < m]
                else:
                    causal_features_adj = causal_features
                
                te_X_sel = te_X_gran[:, causal_features_adj]
                
                try:
                    smt_te = SMOTETomek(random_state=42, sampling_strategy=1.0)
                    te_X_bal, te_y_bal = smt_te.fit_resample(te_X_sel, te_y)
                except:
                    te_X_bal, te_y_bal = te_X_sel, te_y
                
                ws_te = max(2, min(ws, len(te_X_bal) // 10))
                te_X_ts, te_y_ts = create_sequences(te_X_bal, te_y_bal, ws_te)
                
                if len(te_X_ts) == 0:
                    cm[i,j] = np.nan
                    print("no data")
                    continue
                
                if te_X_ts.shape[1] != tr_X_ts.shape[1]:
                    m_ws = min(te_X_ts.shape[1], tr_X_ts.shape[1])
                    te_X_ts = te_X_ts[:, :m_ws, :]
                
                if te_X_ts.shape[2] != tr_X_ts.shape[2]:
                    m_f = min(te_X_ts.shape[2], tr_X_ts.shape[2])
                    te_X_ts = te_X_ts[:, :, :m_f]
                
                y_prob = model.predict(te_X_ts, verbose=0).flatten()
                y_pred = (y_prob > 0.5).astype(int)
                
                f1 = f1_score(te_y_ts, y_pred, zero_division=0)
                cm[i,j] = f1
                print("F1=" + str(round(f1,4)))
            
            except Exception as e:
                cm[i,j] = np.nan
                print("FAIL: " + str(e)[:50])
        
        # پاک‌سازی RAM
        try:
            del model, tr_X_ts, tr_y_ts, tr_X_bal, tr_y_bal
        except:
            pass
        gc.collect()
        tf.keras.backend.clear_session()
    
    cross_df = pd.DataFrame(cm, index=names, columns=names)
    cross_results[ds_name] = cross_df
    cross_df.to_csv(os.path.join(TABLES_DIR, 'cross_' + ds_name + '.csv'))
    
    off = cm[~np.isnan(cm)]
    if len(off) > 0:
        print("\n[AVG] " + str(round(np.mean(off),4)))


# ================================================================
# Part 3: Wilcoxon
# ================================================================

print("\n" + "="*100)
print("[PART 3] Wilcoxon Test")
print("="*100)

wilcoxon = {}

for ds_name, ds_results in all_results.items():
    if 'hybrid_ate' not in ds_results:
        continue
    
    print("\n[DATASET] " + ds_name)
    print('-'*80)
    
    h = ds_results['hybrid_ate']['f1'].values
    
    for m_name, df in ds_results.items():
        if m_name == 'hybrid_ate' or len(df) == 0:
            continue
        
        o = df['f1'].values
        ml = min(len(h), len(o))
        
        if ml >= 5:
            try:
                stat, p = stats.wilcoxon(h[:ml], o[:ml], alternative='greater')
                wilcoxon[ds_name + "_" + m_name] = {
                    'method': methods[m_name]['label'],
                    'p_value': p,
                    'significant': p < 0.05
                }
                sig = "SIG" if p < 0.05 else "NOT-SIG"
                print("   vs " + methods[m_name]['label'] + 
                      ": p=" + str(round(p,6)) + " [" + sig + "]")
            except Exception as e:
                print("   [ERR] " + m_name + ": " + str(e))


# ================================================================
# Part 4: Summary
# ================================================================

print("\n" + "="*100)
print("[PART 4] Summary")
print("="*100)

summary_tables = {}

for ds_name, ds_results in all_results.items():
    print("\n" + "="*80)
    print("[DATASET] " + ds_name)
    print("="*80)
    
    s = []
    for m_name, df in ds_results.items():
        if len(df) > 0:
            s.append({
                'Method': methods[m_name]['label'],
                'F1': df['f1'].mean(),
                'AUC': df['auc'].mean(),
                'Accuracy': df['accuracy'].mean(),
                'Precision': df['precision'].mean(),
                'Recall': df['recall'].mean()
            })
    
    if s:
        sdf = pd.DataFrame(s).sort_values('F1', ascending=False)
        summary_tables[ds_name] = sdf
        print(sdf.to_string(index=False, float_format='{:.4f}'.format))
        sdf.to_csv(os.path.join(TABLES_DIR, 'summary_' + ds_name + '.csv'),
                    index=False)


# ================================================================
# Part 5: Figures
# ================================================================

for ds_name, sdf in summary_tables.items():
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('Comparison - ' + ds_name, fontsize=16, fontweight='bold')
    
    axes[0].bar(range(len(sdf)), sdf['F1'], color='skyblue', edgecolor='black')
    axes[0].set_xticks(range(len(sdf)))
    axes[0].set_xticklabels(sdf['Method'], rotation=45, ha='right')
    axes[0].set_ylabel('F1-Score')
    axes[0].set_title('F1-Score')
    axes[0].set_ylim([0, 1])
    axes[0].axhline(sdf['F1'].max(), color='red', linestyle='--',
                    label='Best: ' + str(round(sdf['F1'].max(),3)))
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    melt = sdf.melt(id_vars=['Method'],
                     value_vars=['F1','AUC','Accuracy','Precision','Recall'],
                     var_name='Metric', value_name='Score')
    sns.barplot(data=melt, x='Method', y='Score', hue='Metric', ax=axes[1])
    axes[1].set_title('All Metrics')
    axes[1].set_ylim([0, 1])
    axes[1].legend(loc='lower right')
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'comparison_' + ds_name + '.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    
    if ds_name in cross_results and len(cross_results[ds_name]) > 1:
        plt.figure(figsize=(12, 10))
        sns.heatmap(cross_results[ds_name], annot=True, fmt='.3f',
                    cmap='RdYlGn', center=0.5, vmin=0, vmax=1)
        plt.title('Cross-Project - ' + ds_name)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'cross_heatmap_' + ds_name + '.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()


# ================================================================
# Part 6: Text Summary
# ================================================================

sp = os.path.join(RESULTS_DIR, 'final_summary.txt')
with open(sp, 'w', encoding='utf-8') as f:
    f.write("="*100 + "\n")
    f.write("SFP with ATE-FS - Final Summary\n")
    f.write("="*100 + "\n\n")
    
    for ds_name, sdf in summary_tables.items():
        f.write("\n" + "="*80 + "\n")
        f.write(ds_name + "\n")
        f.write("="*80 + "\n\n")
        f.write(sdf.to_string(index=False, float_format='{:.4f}'.format))
        if len(sdf) > 0:
            b = sdf.iloc[0]
            f.write("\n\nBest: " + b['Method'] + " F1=" + str(round(b['F1'],4)) + "\n")
    
    f.write("\n\n" + "="*80 + "\n")
    f.write("Wilcoxon\n")
    f.write("="*80 + "\n")
    for k, v in wilcoxon.items():
        sig = "SIG" if v['significant'] else "NOT-SIG"
        f.write("\n" + k + ": p=" + str(round(v['p_value'],6)) + " [" + sig + "]\n")

print("\n[OK] Summary: " + sp)
print("\n" + "="*100)
print("[DONE]")
print("="*100)