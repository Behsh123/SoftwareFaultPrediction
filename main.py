import os
import gc
import glob
import json
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, LSTM, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from dml_ate import build_causal_graph_ate_dml, compute_evalue
from tabu_search import tabu_search_feature_selection

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
np.random.seed(42)
tf.random.set_seed(42)

TRAVIS_DIR = "./data/Travis"
PROMISE_DIR = "./data/Promise"
ZMD_DIR = "./data/ZMD-296k"
N_REPETITIONS = 5
FAST_DML_BOOTSTRAP = 5

def load_travis(fp):
    try:
        df = pd.read_csv(fp, header=None)
        df = df.iloc[:, :-1].apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all").dropna()
        y_raw = df.iloc[:, 0].values
        X = np.log1p(np.abs(df.iloc[:, 1:].values)).astype(np.float32)
        y = (y_raw > 0).astype(np.int32) if len(np.unique(y_raw)) > 2 else y_raw.astype(np.int32)
        return X, y
    except: return None, None

def load_promise(fp):
    try:
        df = pd.read_csv(fp)
        for c in ["name","version","name.1"]:
            if c in df.columns: df = df.drop(columns=[c])
        obj = df.select_dtypes(include=["object"]).columns.tolist()
        df = df.drop(columns=obj).dropna()
        tc = "bug" if "bug" in df.columns else df.columns[-1]
        y = (df[tc].values > 0).astype(np.int32)
        X = np.log1p(np.abs(df.drop(columns=[tc]).values)).astype(np.float32)
        return X, y
    except: return None, None

def load_zmd(fp):
    try:
        df = pd.read_csv(fp)
        target_col = 'defect_prone'
        if target_col not in df.columns:
            for c in df.columns:
                if str(c).lower() in ['defects', 'bug', 'target', 'label', 'is_buggy', 'buggy', 'class']:
                    target_col = c
                    break
            else: return {}
        y = (df[target_col].values > 0).astype(np.int32)
        ignore_cols = ['repository', 'repository_display', 'file_name', target_col, 'bug_fix_commits']
        feature_cols = [c for c in df.columns if c not in ignore_cols]
        for c in feature_cols:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df[feature_cols] = df[feature_cols].fillna(0)
        if 'code_churn' in df.columns and 'unique_developers' in df.columns:
            df['churn_per_dev'] = df['code_churn'] / (df['unique_developers'] + 1)
        if 'commit_count' in df.columns and 'file_age_days' in df.columns:
            df['commits_per_age'] = df['commit_count'] / (df['file_age_days'] + 1)
        if 'lines_added' in df.columns and 'lines_deleted' in df.columns:
            df['add_del_ratio'] = df['lines_added'] / (df['lines_deleted'] + 1)
        feature_cols = [c for c in df.columns if c not in ignore_cols]
        X = np.log1p(np.abs(df[feature_cols].values)).astype(np.float32)
        if len(X) < 50 or len(np.unique(y)) < 2: return {}
        return {"combined": {"X": X, "y": y}}
    except: return {}

def granulate(X, y):
    n, nf = X.shape
    Xg = np.zeros_like(X, dtype=np.int32)
    for fi in range(nf):
        fv = X[:, fi]
        Q1, Q3 = np.percentile(fv, 25), np.percentile(fv, 75)
        iqr = Q3 - Q1
        if iqr == 0:
            Xg[:, fi] = 1
            continue
        best_mi, best_p = -1, 0.1
        for p in np.linspace(0.01, 0.5, 10):
            lo, hi = Q1 + p*iqr, Q3 - p*iqr
            t = np.ones(n, dtype=np.int32)
            t[fv < lo] = 0
            t[fv > hi] = 2
            try: mi = mutual_info_classif(t.reshape(-1,1), y, random_state=42)[0]
            except: mi = 0
            if mi > best_mi:
                best_mi, best_p = mi, p
        lo, hi = Q1 + best_p*iqr, Q3 - best_p*iqr
        Xg[fv < lo, fi] = 0
        Xg[fv > hi, fi] = 2
        Xg[(fv >= lo) & (fv <= hi), fi] = 1
    return Xg

def make_ts(X, y, ws=4):
    if len(X) <= ws: return np.array([]), np.array([])
    Xts, yts = [], []
    for i in range(ws, len(X)):
        Xts.append(X[i-ws:i])
        yts.append(1 if np.sum(y[i-ws:i]) > 0 else 0)
    return np.array(Xts), np.array(yts)

def build_gru(shape):
    m = Sequential([
        Input(shape=shape),
        GRU(32, return_sequences=True),
        BatchNormalization(), Dropout(0.3),
        GRU(16),
        BatchNormalization(), Dropout(0.3),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    m.compile(optimizer=Adam(0.001), loss="binary_crossentropy", metrics=["accuracy"])
    return m

def build_lstm(shape):
    m = Sequential([
        Input(shape=shape),
        LSTM(32, return_sequences=True),
        BatchNormalization(), Dropout(0.3),
        LSTM(16),
        BatchNormalization(), Dropout(0.3),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    m.compile(optimizer=Adam(0.001), loss="binary_crossentropy", metrics=["accuracy"])
    return m

def train_dl(X_tr_ts, y_tr_ts, X_te_ts, y_te_ts, model_type="gru"):
    if len(X_tr_ts) == 0 or len(X_te_ts) == 0: return None
    if X_tr_ts.shape[2] != X_te_ts.shape[2]:
        m = min(X_tr_ts.shape[2], X_te_ts.shape[2])
        X_tr_ts, X_te_ts = X_tr_ts[:,:,:m], X_te_ts[:,:,:m]
    try:
        X_tr2, X_val, y_tr2, y_val = train_test_split(X_tr_ts, y_tr_ts, test_size=0.3, random_state=42)
        m = build_gru((X_tr2.shape[1], X_tr2.shape[2])) if model_type=="gru" else build_lstm((X_tr2.shape[1], X_tr2.shape[2]))
        cb = [EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True)]
        m.fit(X_tr2, y_tr2, validation_data=(X_val, y_val), epochs=20, batch_size=min(64, max(8, len(X_tr2)//10)), callbacks=cb, verbose=0)
        yprob = m.predict(X_te_ts, verbose=0).flatten()
        yp = (yprob > 0.5).astype(int)
        r = {
            "f1": f1_score(y_te_ts, yp, zero_division=0),
            "auc": roc_auc_score(y_te_ts, yprob),
            "accuracy": accuracy_score(y_te_ts, yp),
            "precision": precision_score(y_te_ts, yp, zero_division=0),
            "recall": recall_score(y_te_ts, yp, zero_division=0)
        }
        del m
        gc.collect()
        tf.keras.backend.clear_session()
        return r
    except: return None

def pipeline_adaptive(X_tr, y_tr, X_te, y_te, is_static=False):
    if len(X_tr) > 20000:
        idx = np.random.choice(len(X_tr), 20000, replace=False)
        X_tr, y_tr = X_tr[idx], y_tr[idx]
    fr = y_tr.mean()
    if is_static:
        X_tr_g = granulate(X_tr, y_tr)
        X_te_g = granulate(X_te, y_te)
        sel = build_causal_graph_ate_dml(X_tr_g, y_tr, 5, 12, 0.001, FAST_DML_BOOTSTRAP)
        X_tr_s = X_tr_g[:, sel]
        X_te_s = X_te_g[:, sel]
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr_s)
        X_te_s = sc.transform(X_te_s)
        try:
            smt = SMOTETomek(random_state=42, sampling_strategy=1.0)
            X_tr_b, y_tr_b = smt.fit_resample(X_tr_s, y_tr)
        except: X_tr_b, y_tr_b = X_tr_s, y_tr
        ws = max(2, min(4, len(X_tr_b)//10))
        X_tr_ts, y_tr_ts = make_ts(X_tr_b, y_tr_b, ws)
        X_te_ts, y_te_ts = make_ts(X_te_s, y_te, ws)
        if len(X_tr_ts) == 0 or len(X_te_ts) == 0: return None
        X_tr_flat = X_tr_ts.reshape(X_tr_ts.shape[0], -1)
        X_te_flat = X_te_ts.reshape(X_te_ts.shape[0], -1)
        rf = RandomForestClassifier(n_estimators=100, max_depth=8, class_weight='balanced', random_state=42, n_jobs=-1)
        rf.fit(X_tr_flat, y_tr_ts)
        xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, eval_metric='logloss', random_state=42)
        xgb_model.fit(X_tr_flat, y_tr_ts)
        prob = (rf.predict_proba(X_te_flat)[:, 1] + xgb_model.predict_proba(X_te_flat)[:, 1]) / 2.0
        prob_tr = (rf.predict_proba(X_tr_flat)[:, 1] + xgb_model.predict_proba(X_tr_flat)[:, 1]) / 2.0
        best_t, best_f = 0.5, 0
        for t in np.arange(0.1, 0.9, 0.05):
            f = f1_score(y_tr_ts, (prob_tr > t).astype(int), zero_division=0)
            if f > best_f:
                best_f, best_t = f, t
        yp = (prob > best_t).astype(int)
        return {"f1": f1_score(y_te_ts, yp, zero_division=0), "auc": roc_auc_score(y_te_ts, prob), "accuracy": accuracy_score(y_te_ts, yp), "precision": precision_score(y_te_ts, yp, zero_division=0), "recall": recall_score(y_te_ts, yp, zero_division=0)}
    else:
        if fr < 0.10: n_feat, smote_first = 5, False
        elif fr < 0.30: n_feat, smote_first = 15, True
        else: n_feat, smote_first = 5, False
        if smote_first:
            try:
                min_c = np.bincount(y_tr).min()
                k = min(7, max(1, min_c - 1))
                smt = SMOTETomek(random_state=42, sampling_strategy=1.0, smote=SMOTE(k_neighbors=k))
                X_tr_r, y_tr_r = smt.fit_resample(X_tr, y_tr)
            except: X_tr_r, y_tr_r = X_tr, y_tr
        else: X_tr_r, y_tr_r = X_tr, y_tr
        X_tr_g = granulate(X_tr_r, y_tr_r)
        X_te_g = granulate(X_te, y_te)
        min_ate = 0.005 if n_feat <= 8 else 0.001
        sel = build_causal_graph_ate_dml(X_tr_g, y_tr_r, n_feat, max(n_feat, n_feat+5), min_ate, FAST_DML_BOOTSTRAP)
        X_tr_s = X_tr_g[:, sel]
        X_te_s = X_te_g[:, sel]
        ts = tabu_search_feature_selection(X_tr_s, y_tr_r)
        X_tr_s = X_tr_s[:, ts]
        X_te_s = X_te_s[:, ts]
        if not smote_first:
            try:
                smt = SMOTETomek(random_state=42, sampling_strategy=1.0)
                X_tr_s, y_tr_r = smt.fit_resample(X_tr_s, y_tr_r)
            except: pass
        ws = max(2, min(4, len(X_tr_s)//10))
        return train_dl(*make_ts(X_tr_s, y_tr_r, ws), *make_ts(X_te_s, y_te, ws), "gru")

def main():
    print("[LOAD] Loading datasets...")
    travis, promise, zmd = {}, {}, {}
    if os.path.exists(TRAVIS_DIR):
        for f in sorted(os.listdir(TRAVIS_DIR)):
            if f.endswith(".csv"):
                n = f.replace(".csv","")
                X, y = load_travis(os.path.join(TRAVIS_DIR, f))
                if X is not None and len(X) > 50: travis[n] = {"X": X, "y": y}
    if os.path.exists(PROMISE_DIR):
        for f in sorted(os.listdir(PROMISE_DIR)):
            if f.endswith(".csv"):
                n = f.replace(".csv","")
                X, y = load_promise(os.path.join(PROMISE_DIR, f))
                if X is not None and len(X) > 50: promise[n] = {"X": X, "y": y}
    if os.path.exists(ZMD_DIR):
        for f in glob.glob(os.path.join(ZMD_DIR, "**", "*.csv"), recursive=True):
            p_name = os.path.basename(f).replace('.csv', '')
            loaded = load_zmd(f)
            if loaded:
                for k, v in loaded.items():
                    if len(loaded) == 1 and k == "combined": zmd[p_name] = v
                    else: zmd[f"{p_name}_{k}"] = v

    datasets = {"ZMD": (zmd, True), "PROMISE": (promise, True), "TRAVIS": (travis, False)}
    results = {"main": {}, "ablation": {}, "wilcoxon": {}, "evalue": {}}

    for ds_name, (projects, is_static) in datasets.items():
        if not projects: continue
        print(f"\n[DATASET] {ds_name}")
        f1s = []
        for p_name, data in projects.items():
            try: X_tr, X_te, y_tr, y_te = train_test_split(data["X"], data["y"], test_size=0.3, random_state=42, stratify=data["y"])
            except: X_tr, X_te, y_tr, y_te = train_test_split(data["X"], data["y"], test_size=0.3, random_state=42)
            r = pipeline_adaptive(X_tr, y_tr, X_te, y_te, is_static)
            if r: 
                f1s.append(r["f1"])
                print(f"  {p_name}: F1={r['f1']:.4f}")
            gc.collect()
        if f1s:
            results["main"][ds_name] = float(np.mean(f1s))
            print(f"  => MEAN F1={np.mean(f1s):.4f}")

        print(f"\n[ABLATION] {ds_name}")
        abl_f1s = []
        for p_name, data in projects.items():
            try: X_tr, X_te, y_tr, y_te = train_test_split(data["X"], data["y"], test_size=0.3, random_state=42, stratify=data["y"])
            except: X_tr, X_te, y_tr, y_te = train_test_split(data["X"], data["y"], test_size=0.3, random_state=42)
            if len(X_tr) > 20000:
                idx = np.random.choice(len(X_tr), 20000, replace=False)
                X_tr, y_tr = X_tr[idx], y_tr[idx]
            if is_static:
                sc = StandardScaler()
                X_tr_s = sc.fit_transform(X_tr)
                X_te_s = sc.transform(X_te)
                try:
                    smt = SMOTETomek(random_state=42, sampling_strategy=1.0)
                    X_tr_b, y_tr_b = smt.fit_resample(X_tr_s, y_tr)
                except: X_tr_b, y_tr_b = X_tr_s, y_tr
                ws = max(2, min(4, len(X_tr_b)//10))
                X_tr_ts, y_tr_ts = make_ts(X_tr_b, y_tr_b, ws)
                X_te_ts, y_te_ts = make_ts(X_te_s, y_te, ws)
                if len(X_tr_ts) == 0 or len(X_te_ts) == 0: continue
                X_tr_flat = X_tr_ts.reshape(X_tr_ts.shape[0], -1)
                X_te_flat = X_te_ts.reshape(X_te_ts.shape[0], -1)
                rf = RandomForestClassifier(n_estimators=100, max_depth=8, class_weight='balanced', random_state=42, n_jobs=-1)
                rf.fit(X_tr_flat, y_tr_ts)
                yprob = rf.predict_proba(X_te_flat)[:,1]
                yp = (yprob > 0.5).astype(int)
                abl_f1s.append(f1_score(y_te_ts, yp, zero_division=0))
            else:
                X_tr_g = granulate(X_tr, y_tr)
                X_te_g = granulate(X_te, y_te)
                sel = build_causal_graph_ate_dml(X_tr_g, y_tr, 5, 15, 0.001, FAST_DML_BOOTSTRAP)
                X_tr_s = X_tr_g[:, sel]
                X_te_s = X_te_g[:, sel]
                ws = max(2, min(4, len(X_tr_s)//10))
                r = train_dl(*make_ts(X_tr_s, y_tr, ws), *make_ts(X_te_s, y_te, ws), "gru")
                if r: abl_f1s.append(r["f1"])
            gc.collect()
        if abl_f1s:
            results["ablation"][ds_name] = float(np.mean(abl_f1s))
            print(f"  => MEAN F1={np.mean(abl_f1s):.4f}")

        print(f"\n[WILCOXON & E-VALUE] {ds_name}")
        w_adaptive, w_gru = [], []
        evalues = []
        for p_name, data in projects.items():
            try: X_tr, X_te, y_tr, y_te = train_test_split(data["X"], data["y"], test_size=0.3, random_state=42, stratify=data["y"])
            except: X_tr, X_te, y_tr, y_te = train_test_split(data["X"], data["y"], test_size=0.3, random_state=42)
            if len(X_tr) > 20000:
                idx = np.random.choice(len(X_tr), 20000, replace=False)
                X_tr, y_tr = X_tr[idx], y_tr[idx]
            r_a = pipeline_adaptive(X_tr, y_tr, X_te, y_te, is_static)
            sc = StandardScaler()
            X_tr_s = sc.fit_transform(X_tr)
            X_te_s = sc.transform(X_te)
            try:
                smt = SMOTETomek(random_state=42, sampling_strategy=1.0)
                X_tr_b, y_tr_b = smt.fit_resample(X_tr_s, y_tr)
            except: X_tr_b, y_tr_b = X_tr_s, y_tr
            ws = max(2, min(4, len(X_tr_b)//10))
            r_g = train_dl(*make_ts(X_tr_b, y_tr_b, ws), *make_ts(X_te_s, y_te, ws), "gru")
            if r_a and r_g:
                w_adaptive.append(r_a["f1"])
                w_gru.append(r_g["f1"])
            X_g = granulate(X_tr, y_tr)
            if len(X_g) > 10000:
                idx = np.random.choice(len(X_g), 10000, replace=False)
                X_g, y_tr = X_g[idx], y_tr[idx]
            for _ in range(5):
                try:
                    sel, _ = build_causal_graph_ate_dml(X_g, y_tr, 5, 10, 0.001, FAST_DML_BOOTSTRAP)
                    for f in sel: evalues.append(compute_evalue(0.05))
                except: pass
            gc.collect()
        if len(w_adaptive) >= 5:
            stat, p = stats.wilcoxon(w_adaptive, w_gru, alternative="greater")
            results["wilcoxon"][ds_name] = {"stat": float(stat), "p": float(p)}
            print(f"  Wilcoxon: p={p:.6f}")
        if evalues:
            results["evalue"][ds_name] = {"mean": float(np.mean(evalues)), "n_15": int(sum(1 for e in evalues if e > 1.5))}
            print(f"  E-value: mean={np.mean(evalues):.3f}, >1.5={sum(1 for e in evalues if e > 1.5)}")

    with open("results.json", "w") as f: json.dump(results, f, indent=4)
    print("\n[DONE] Results saved to results.json")

if __name__ == "__main__":
    main()