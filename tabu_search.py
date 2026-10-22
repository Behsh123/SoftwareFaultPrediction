import warnings
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")
MIN_FEATURES = 4

def _fitness(sel, X, y, imp, w1, w2, w3, seed=42):
    if len(sel) < MIN_FEATURES: return -10.0
    sel = sorted(set(sel))
    Xs = X[:, sel]
    try:
        Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.3, random_state=seed, stratify=y)
        clf = RandomForestClassifier(n_estimators=30, max_depth=6, random_state=seed, n_jobs=-1)
        clf.fit(Xtr, ytr)
        f1 = f1_score(yte, clf.predict(Xte), zero_division=0)
    except: return -1.0
    return float(w1*f1 - w2*(len(sel)/X.shape[1]) + w3*np.mean(imp[sel]))

def _neighbors(cur, nf, size, rng):
    nbs = []
    cs = set(cur)
    for _ in range(size):
        nb = set(cs)
        op = rng.choice(["flip", "swap"], p=[0.5, 0.5])
        if op == "flip":
            f = rng.randint(nf)
            if f in nb:
                if len(nb) > MIN_FEATURES: nb.remove(f)
            else: nb.add(f)
        else:
            if len(nb) >= MIN_FEATURES:
                fo = rng.choice(list(nb))
                fi = rng.randint(nf)
                a = 0
                while fi in nb and a < 10:
                    fi = rng.randint(nf)
                    a += 1
                if fi not in nb:
                    nb.remove(fo)
                    nb.add(fi)
        if len(nb) >= MIN_FEATURES: nbs.append(sorted(nb))
    return nbs

def tabu_search_feature_selection(X, y, w1=0.7, w2=0.05, w3=0.25, max_iter=15, tabu_tenure=5, neighbor_size=8, n_restarts=1, patience=5, seed=42):
    nf = X.shape[1]
    if nf <= MIN_FEATURES: return list(range(nf))
    imp = RandomForestClassifier(n_estimators=30, random_state=seed, n_jobs=-1).fit(X, y).feature_importances_
    imp = imp / (imp.sum() + 1e-9)
    rng = np.random.RandomState(seed)
    best_g, best_gf = None, -np.inf
    for r in range(n_restarts):
        ns = max(MIN_FEATURES, min(nf//2, 10))
        cur = sorted(rng.choice(nf, ns, replace=False).tolist())
        cf = _fitness(cur, X, y, imp, w1, w2, w3, seed)
        tl = []
        ni = 0
        lb = cur.copy()
        lbf = cf
        for it in range(max_iter):
            nbs = _neighbors(cur, nf, neighbor_size, rng)
            bn, bnf = None, -np.inf
            for nb in nbs:
                k = frozenset(nb)
                is_t = any(kk == k and e > it for kk, e in tl)
                f = _fitness(nb, X, y, imp, w1, w2, w3, seed)
                if is_t and f <= best_gf: continue
                if f > bnf:
                    bnf = f
                    bn = nb
            if bn is None:
                ns = max(MIN_FEATURES, min(nf//2, 10))
                cur = sorted(rng.choice(nf, ns, replace=False).tolist())
                continue
            cur = bn
            cf = bnf
            tl.append((frozenset(cur), it + tabu_tenure))
            tl = [(kk, e) for kk, e in tl if e > it]
            if cf > lbf:
                lbf = cf
                lb = cur.copy()
                ni = 0
            else:
                ni += 1
                if ni >= patience: break
        if lbf > best_gf:
            best_gf = lbf
            best_g = lb.copy()
    if best_g is None: best_g = list(range(min(MIN_FEATURES, nf)))
    return sorted(best_g, key=lambda i: -imp[i])