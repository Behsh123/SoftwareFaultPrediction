import warnings
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

def _cross_fit_nuisance(X, T, Y, n_folds=3, seed=42):
    n = len(Y)
    Y_res = np.zeros(n)
    T_res = np.zeros(n)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr, te in kf.split(X):
        my = RandomForestRegressor(n_estimators=20, max_depth=6, random_state=seed, n_jobs=-1)
        my.fit(X[tr], Y[tr])
        Y_res[te] = Y[te] - my.predict(X[te])
        mt = RandomForestRegressor(n_estimators=20, max_depth=6, random_state=seed, n_jobs=-1)
        mt.fit(X[tr], T[tr])
        T_res[te] = T[te] - mt.predict(X[te])
    return Y_res, T_res

def _dml_ate(X, T, Y, n_folds=3, seed=42):
    Y_res, T_res = _cross_fit_nuisance(X, T, Y, n_folds, seed)
    d = np.mean(T_res**2)
    if d < 1e-8: return 0.0
    return float(np.clip(np.mean(T_res*Y_res)/d, -1.0, 1.0))

def estimate_ate_dml(X, y, fi, n_bootstrap=5, n_folds=3):
    fv = X[:, fi]
    tm = (fv == 2)
    cm = (fv == 0)
    if tm.sum() < 20 or cm.sum() < 20: return 0.0, 0.0, 0.0
    m = tm | cm
    Xm = X[m].astype(np.float64)
    ym = y[m].astype(np.float64)
    tm2 = tm[m].astype(np.float64)
    ci = [i for i in range(X.shape[1]) if i != fi]
    if not ci: return 0.0, 0.0, 0.0
    Xc = StandardScaler().fit_transform(Xm[:, ci])
    ates = []
    for b in range(n_bootstrap):
        rng = np.random.RandomState(42+b)
        idx = rng.choice(len(ym), len(ym), replace=True)
        Xb, Tb, Yb = Xc[idx], tm2[idx], ym[idx]
        if len(np.unique(Tb)) < 2 or len(np.unique(Yb)) < 2: continue
        try: ates.append(_dml_ate(Xb, Tb, Yb, n_folds, 42+b))
        except: continue
    if len(ates) < 3: return 0.0, 0.0, 0.0
    return float(np.mean(ates)), float(np.percentile(ates, 2.5)), float(np.percentile(ates, 97.5))

def build_causal_graph_ate_dml(X, y, min_features=5, max_features=15, min_ate=0.001, n_bootstrap=5):
    n = X.shape[1]
    scores = {}
    for i in range(n):
        try:
            a, cl, cu = estimate_ate_dml(X, y, i, n_bootstrap, 3)
            if cl > 0 and abs(a) > min_ate:
                scores[i] = abs(a)
        except: continue
    sorted_f = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    if len(sorted_f) < min_features:
        rf = RandomForestClassifier(n_estimators=30, random_state=42, n_jobs=-1).fit(X, y)
        top = np.argsort(rf.feature_importances_)[::-1].tolist()
        for f in top:
            if f not in sorted_f: sorted_f.append(f)
            if len(sorted_f) >= min_features: break
    return sorted_f[:max_features]

def compute_evalue(ate):
    if abs(ate) < 1e-8: return 1.0
    rr = np.exp(abs(ate))
    if rr < 1.0: rr = 1.0/rr
    return float(np.sqrt(rr) + np.sqrt(rr*(np.sqrt(rr)-1)))