#!/usr/bin/env python

import argparse, json, os, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.tree import DecisionTreeClassifier

NC = 40
MASK_PIX = np.zeros((64, 64), bool)
MASK_PIX[24:40, 24:40] = True
MASK_PIX = MASK_PIX.ravel()
OUT = "results"


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    with open(os.path.join(OUT, "run_log.txt"), "a") as f:
        f.write(s + "\n")


def jdump(obj, name):
    def conv(o):
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        raise TypeError(type(o))
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(obj, f, indent=2, default=conv)


def rs(x):
    return int(x % (2 ** 31 - 1))


def load_data(synthetic=False):
    if synthetic:
        r = np.random.default_rng(1)
        y = np.repeat(np.arange(NC), 10)
        means = r.random((NC, 4096))
        X = np.clip(means[y] + 0.25 * r.standard_normal((400, 4096)), 0, 1)
        keep = np.zeros(4096, bool); keep[r.choice(4096, 150, replace=False)] = True
        keep[np.where(MASK_PIX)[0][:40]] = True
        X[:, ~keep] = 0.0
        return X.astype(np.float32), y
    d = fetch_olivetti_faces()
    return d.data.astype(np.float32), d.target


def make_split(y, rng):
    tr, va, te = [], [], []
    for c in range(NC):
        idx = rng.permutation(np.where(y == c)[0])
        tr += list(idx[:6]); va += list(idx[6:8]); te += list(idx[8:10])
    return np.array(tr), np.array(va), np.array(te)


def mask_faces(X):
    Xm = X.copy().reshape(-1, 64, 64)
    Xm[:, 24:40, 24:40] = 0.0
    return Xm.reshape(len(X), -1)


def new_tree(md, msl, seed):
    return DecisionTreeClassifier(max_depth=md, min_samples_leaf=msl, random_state=rs(seed))


def leaf_stats(t):
    tt = t.tree_
    leaf = tt.children_left == -1
    ns = tt.n_node_samples[leaf]
    return dict(n_leaves=int(leaf.sum()), mean_samples_per_leaf=float(ns.mean()),
                frac_singleton_leaves=float((ns == 1).mean()))


def tree_proba(t, X, K=NC):
    p = np.zeros((len(X), K)); p[:, t.classes_] = t.predict_proba(X); return p


def tree_pc(t, X):
    p = t.predict_proba(X)
    return t.classes_[p.argmax(1)], p.max(1)


def bag_fit(X, y, params, B, rng):
    n = len(X); trees, boots = [], []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        t = new_tree(params["max_depth"], params["min_samples_leaf"], rng.integers(0, 2 ** 31 - 1)).fit(X[idx], y[idx])
        trees.append(t); boots.append(idx)
    return trees, boots


def tree_preds(trees, X):
    return np.stack([t.predict(X) for t in trees])


def votes(P, K=NC):
    V = np.zeros((P.shape[1], K)); cols = np.arange(P.shape[1])
    for p in P: V[cols, p] += 1
    return V


def bag_eval(trees, X):
    V = votes(tree_preds(trees, X))
    return V.argmax(1), V.max(1) / len(trees), V


def oob_accuracy(Ptr, boots, y, B):
    n = len(y); V = np.zeros((n, NC)); cnt = np.zeros(n)
    for b in range(B):
        oob = np.ones(n, bool); oob[boots[b]] = False
        ii = np.where(oob)[0]
        V[ii, Ptr[b][ii]] += 1; cnt[ii] += 1
    cov = cnt > 0
    return float((V[cov].argmax(1) == y[cov]).mean()), float(cov.mean())


def pairwise_disagreement(P):
    B = P.shape[0]
    if B < 2: return np.nan
    tot, cnt = 0.0, 0
    for i in range(B - 1):
        tot += (P[i + 1:] != P[i]).mean(axis=1).sum(); cnt += B - 1 - i
    return tot / cnt


def mean_margin(V):
    f = np.sort(V / V.sum(1, keepdims=True), axis=1)
    return float((f[:, -1] - f[:, -2]).mean())

def samme_fit(X, y, K, depth, T, seed, verbose=False):
    """Multiclass AdaBoost (SAMME)"""
    n = len(y); w = np.full(n, 1.0 / n)
    L, A = [], []
    H = dict(w_used=[], pred=[], err=[], alpha=[], w_next=[], stopped=None)
    for t in range(T):
        h = DecisionTreeClassifier(max_depth=depth, random_state=rs(seed + t)).fit(X, y, sample_weight=w)
        pred = h.predict(X); wrong = pred != y
        err = float(np.sum(w * wrong) / np.sum(w))
        if err >= 1.0 - 1.0 / K:
            H["stopped"] = f"round {t+1}: weighted error {err:.4f} >= 1-1/K = {1-1/K:.4f}; learner discarded, boosting stopped"
            if verbose: print(H["stopped"])
            break
        e = max(err, 1e-10)
        alpha = float(np.log((1 - e) / e) + np.log(K - 1))
        w_new = w * np.exp(alpha * wrong); w_new /= w_new.sum()
        L.append(h); A.append(alpha)
        H["w_used"].append(w.copy()); H["pred"].append(pred); H["err"].append(err)
        H["alpha"].append(alpha); H["w_next"].append(w_new.copy())
        if verbose:
            print(f"--- round {t+1} ---")
            print("weights used     :", np.round(w, 4))
            print("true labels      :", y)
            print("weak predictions :", pred)
            print(f"weighted error   : {err:.4f}")
            print(f"alpha            : {alpha:.4f}  = log((1-err)/err)+log(K-1)")
            print("updated weights  :", np.round(w_new, 4), "(sum=%.4f)" % w_new.sum())
        w = w_new
        if err == 0:
            H["stopped"] = f"round {t+1}: zero weighted error, boosting stopped"; break
    for k in ("w_used", "pred", "w_next"): H[k] = np.array(H[k])
    H["err"] = np.array(H["err"]); H["alpha"] = np.array(H["alpha"])
    return L, A, H


def samme_scores(L, A, X, K):
    S = np.zeros((len(X), K)); cols = np.arange(len(X))
    for h, a in zip(L, A): S[cols, h.predict(X)] += a
    return S


def samme_staged_acc(L, A, X, y, K):
    S = np.zeros((len(X), K)); cols = np.arange(len(X)); out = []
    for h, a in zip(L, A):
        S[cols, h.predict(X)] += a; out.append((S.argmax(1) == y).mean())
    return np.array(out)


def boost_eval(L, A, X):
    q = samme_scores(L, A, X, NC) / np.sum(A)
    return q.argmax(1), q.max(1), q


def masked_path_share(t, x):
    nodes = t.decision_path(x[None]).indices
    feat = t.tree_.feature[nodes]; internal = feat >= 0
    if internal.sum() == 0: return 0.0
    return float(MASK_PIX[feat[internal]].sum() / internal.sum())


def masked_split_share(t):
    f = t.tree_.feature; f = f[f >= 0]
    return float(MASK_PIX[f].mean()) if len(f) else 0.0


def show_faces(items, path, ncols=4):
    n = len(items); nr = int(np.ceil(n / ncols))
    fig, axs = plt.subplots(nr, ncols, figsize=(3 * ncols, 3.3 * nr)); axs = np.atleast_1d(axs).ravel()
    for ax in axs: ax.axis("off")
    for ax, (img, title) in zip(axs, items):
        ax.imshow(img.reshape(64, 64), cmap="gray"); ax.set_title(title, fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, path), dpi=150); plt.close()


def bias_var(Pr, y):                       # Pr: (R, N, NC)
    pbar = Pr.mean(0); E = np.eye(NC)[y]
    return float(((pbar - E) ** 2).sum(1).mean()), float(((Pr - pbar) ** 2).sum(2).mean())


def acc_stats(Pr, y):
    a = np.array([(Pr[r].argmax(1) == y).mean() for r in range(len(Pr))])
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


_cache = {}
def tree_bag_study(Xall, yall, resamples, params, B, Xtests, yte, seed):
    key = (params["max_depth"], params["min_samples_leaf"], B)
    if key in _cache: return _cache[key]
    T = [[] for _ in Xtests]; G = [[] for _ in Xtests]
    leafs, bleafs, dis, depths = [], [], [], []
    for r, idx in enumerate(resamples):
        Xs, ys = Xall[idx], yall[idx]
        t = new_tree(params["max_depth"], params["min_samples_leaf"], seed).fit(Xs, ys)
        leafs.append(leaf_stats(t)); depths.append(t.get_depth())
        trees, _ = bag_fit(Xs, ys, params, B, np.random.default_rng([seed, 31, r]))
        for j, Xt in enumerate(Xtests):
            T[j].append(tree_proba(t, Xt))
            P = tree_preds(trees, Xt); G[j].append(votes(P) / B)
            if j == 0: dis.append(pairwise_disagreement(P))
        ls = [leaf_stats(tt) for tt in trees]
        bleafs.append({k: float(np.mean([l[k] for l in ls])) for k in ls[0]})
    out = dict(tree=[np.stack(a) for a in T], bag=[np.stack(a) for a in G],
               tree_leaf=pd.DataFrame(leafs).mean().to_dict(), bag_leaf=pd.DataFrame(bleafs).mean().to_dict(),
               disagreement=float(np.mean(dis)), mean_depth=float(np.mean(depths)))
    _cache[key] = out
    return out


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--roll", required=True, help="full roll number; its digits are the seed")
    ap.add_argument("--out", default="results")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--synthetic", action="store_true", help="smoke test only, no download")
    a = ap.parse_args()
    OUT = a.out; os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "run_log.txt"), "w").close()
    seed = int("".join(c for c in a.roll if c.isdigit()))
    log("seed =", seed)


    X, y = load_data(a.synthetic)
    tr, va, te = make_split(y, np.random.default_rng(seed))
    np.savez(os.path.join(OUT, "split.npz"), train=tr, val=va, test=te, seed=seed)
    Xtr, ytr, Xva, yva, Xte, yte = X[tr], y[tr], X[va], y[va], X[te], y[te]
    Xte_m = mask_faces(Xte)
    log("train/val/test sizes:", len(tr), len(va), len(te))


    log("\n===== Q1(a) single decision tree =====")
    rows = []
    for md in [2, 4, 8, 16, None]:
        for msl in [1, 2, 4]:
            t = new_tree(md, msl, seed).fit(Xtr, ytr)
            rows.append(dict(max_depth=md, min_samples_leaf=msl, train=t.score(Xtr, ytr), val=t.score(Xva, yva),
                             test=t.score(Xte, yte), actual_depth=t.get_depth(), **leaf_stats(t)))
    d1a = pd.DataFrame(rows); d1a.to_csv(os.path.join(OUT, "q1a_grid.csv"), index=False)
    log(d1a.round(4).to_string())
    best = max(rows, key=lambda r: (r["val"], -r["n_leaves"]))
    frozen = dict(max_depth=best["max_depth"], min_samples_leaf=best["min_samples_leaf"])
    log("FROZEN tree setting:", frozen)
    tree0 = new_tree(frozen["max_depth"], frozen["min_samples_leaf"], seed).fit(Xtr, ytr)

    log("\n===== Q1(b),(c) bagging =====")
    Bs = [1, 5, 10, 25, 50, 100, 200]
    trees, boots = bag_fit(Xtr, ytr, frozen, 200, np.random.default_rng([seed, 1]))
    Ptr, Pva, Pte = tree_preds(trees, Xtr), tree_preds(trees, Xva), tree_preds(trees, Xte)
    n = len(ytr); theory = n * (1 - (1 - 1 / n) ** n)
    rows = []
    for B in Bs:
        oob, cov = oob_accuracy(Ptr, boots, ytr, B)
        Vva = votes(Pva[:B])
        rows.append(dict(B=B, train=(votes(Ptr[:B]).argmax(1) == ytr).mean(), val=(Vva.argmax(1) == yva).mean(),
                         test=(votes(Pte[:B]).argmax(1) == yte).mean(),
                         distinct_per_bootstrap=np.mean([len(np.unique(boots[b])) for b in range(B)]),
                         theory_distinct=theory, oob_acc=oob, oob_coverage=cov,
                         disagreement_val=pairwise_disagreement(Pva[:B]), mean_margin_val=mean_margin(Vva)))
    d1b = pd.DataFrame(rows); d1b.to_csv(os.path.join(OUT, "q1b_bagging.csv"), index=False)
    log(d1b.round(4).to_string())
    bsel = int(max(rows, key=lambda r: (r["val"], -r["B"]))["B"])
    log("FINAL bagging size B =", bsel)
    bag_trees = trees[:bsel]

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    ax[0].plot(d1b.B, d1b.oob_acc, "o-", label="OOB"); ax[0].plot(d1b.B, d1b.val, "s--", label="validation")
    ax[0].set_ylabel("accuracy"); ax[0].legend()
    ax[1].plot(d1b.B, d1b.disagreement_val, "o-"); ax[1].set_ylabel("pairwise disagreement (val)")
    ax[2].plot(d1b.B, d1b.mean_margin_val, "o-"); ax[2].set_ylabel("mean vote margin (val)")
    for x in ax: x.set_xscale("log"); x.set_xlabel("B (number of trees)"); x.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_q1c.png"), dpi=200); plt.close()

    log("\n===== Q1(d) tree vs bagging, clean & masked =====")
    tp_c, tc_c = tree_pc(tree0, Xte); bp_c, bc_c, Vb_c = bag_eval(bag_trees, Xte)
    tp_m, tc_m = tree_pc(tree0, Xte_m); bp_m, bc_m, _ = bag_eval(bag_trees, Xte_m)
    t1d = pd.DataFrame([
        dict(model="single tree", test="clean", acc=(tp_c == yte).mean(), mean_conf=tc_c.mean()),
        dict(model="single tree", test="masked", acc=(tp_m == yte).mean(), mean_conf=tc_m.mean()),
        dict(model="bagging", test="clean", acc=(bp_c == yte).mean(), mean_conf=bc_c.mean()),
        dict(model="bagging", test="masked", acc=(bp_m == yte).mean(), mean_conf=bc_m.mean())])
    t1d.to_csv(os.path.join(OUT, "q1d_table.csv"), index=False); log(t1d.round(4).to_string())
    fixed = np.where((bp_c == yte) & (tp_c != yte))[0]; fixed = fixed[np.argsort(-bc_c[fixed])][:2]
    both = np.where((bp_c != yte) & (tp_c != yte))[0]; both = both[np.argsort(-bc_c[both])][:2]
    cases, items = [], []
    for kind, ids in (("bagging fixes tree", fixed), ("both wrong", both)):
        for i in ids:
            top = np.argsort(-Vb_c[i])[:3]
            c = dict(kind=kind, test_index=int(i), true=int(yte[i]), tree_pred=int(tp_c[i]), tree_conf=float(tc_c[i]),
                     bag_top3_votes={int(k): int(Vb_c[i, k]) for k in top}, votes_for_true_class=int(Vb_c[i, yte[i]]), B=bsel)
            cases.append(c); log(c)
            items.append((Xte[i], f"{kind}\ntrue {yte[i]} | tree {tp_c[i]}\nbag votes {c['bag_top3_votes']}"))
    jdump(cases, "q1d_cases.json")
    if items: show_faces(items, "fig_q1d_faces.png")

    
    log("\n===== Q2(a) toy SAMME round (same function as faces) =====")
    Xt = np.array([[1, 1], [2, 2], [1, 3], [5, 1], [6, 2], [5, 3], [3, 6], [4, 7], [2, 6]], float)
    yt = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        Lt, At, Ht = samme_fit(Xt, yt, 3, depth=1, T=3, seed=seed, verbose=True)
    log(buf.getvalue())
    e0 = Ht["err"][0]; assert np.isclose(Ht["alpha"][0], np.log((1 - e0) / e0) + np.log(2))
    St = samme_scores(Lt, At, Xt, 3); log("toy ensemble predictions:", St.argmax(1), " truth:", yt)

    log("\n===== Q2(b) boosting grid =====")
    models, rows = {}, []
    for d in [1, 2, 3, 4]:
        L, A, H = samme_fit(Xtr, ytr, NC, d, 100, seed); models[d] = (L, A, H)
        if H["stopped"]: log(f"depth {d}:", H["stopped"])
        atr, ava, ate = (samme_staged_acc(L, A, Xs, ys, NC) for Xs, ys in ((Xtr, ytr), (Xva, yva), (Xte, yte)))
        for R in [10, 25, 50, 100]:
            m = min(R, len(L))
            rows.append(dict(depth=d, rounds=R, rounds_used=m, train=atr[m - 1], val=ava[m - 1], test=ate[m - 1]))
    d2b = pd.DataFrame(rows); d2b.to_csv(os.path.join(OUT, "q2b_grid.csv"), index=False); log(d2b.round(4).to_string())
    bb = max(rows, key=lambda r: (r["val"], -r["depth"], -r["rounds_used"]))
    dsel, Rsel = bb["depth"], bb["rounds_used"]
    log(f"SELECTED boosting: depth={dsel}, rounds={Rsel}")
    Lall, Aall, Hsel = models[dsel]; Lb, Ab = Lall[:Rsel], Aall[:Rsel]
    w_used = Hsel["w_used"][:Rsel]; top10 = np.sort(w_used, axis=1)[:, -10:].sum(1)
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.4)); rr = np.arange(1, Rsel + 1)
    for x, v, nm in zip(ax, (Hsel["err"][:Rsel], Hsel["alpha"][:Rsel], w_used.max(1), top10),
                        ("weighted training error", "learner weight alpha", "max sample weight", "total weight of 10 heaviest")):
        x.plot(rr, v); x.set_xlabel("round"); x.set_title(nm, fontsize=9); x.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_q2b.png"), dpi=200); plt.close()

    log("\n===== Q2(c) boosting vs bagging, 25/50/100 learners =====")
    rows = []
    for T in [25, 50, 100]:
        t0 = time.perf_counter(); bt, _ = bag_fit(Xtr, ytr, frozen, T, np.random.default_rng([seed, 2, T])); tb = time.perf_counter() - t0
        t0 = time.perf_counter(); L, A, _ = samme_fit(Xtr, ytr, NC, dsel, T, seed); tg = time.perf_counter() - t0
        rows.append(dict(model="bagging", T=T, train=bag_eval(bt, Xtr)[0].__eq__(ytr).mean(),
                         val=(bag_eval(bt, Xva)[0] == yva).mean(), test=(bag_eval(bt, Xte)[0] == yte).mean(), train_time_s=tb))
        rows.append(dict(model="boosting", T=T, train=(boost_eval(L, A, Xtr)[0] == ytr).mean(),
                         val=(boost_eval(L, A, Xva)[0] == yva).mean(), test=(boost_eval(L, A, Xte)[0] == yte).mean(), train_time_s=tg))
    d2c = pd.DataFrame(rows); d2c.to_csv(os.path.join(OUT, "q2c_compare.csv"), index=False); log(d2c.round(4).to_string())

    wfin = Hsel["w_next"][Rsel - 1]; hv = int(np.argmax(wfin))
    wh = np.append(w_used[:, hv], wfin[hv]); predh = Hsel["pred"][:Rsel, hv]; wrong = predh != ytr[hv]
    vals, cnts = np.unique(predh[wrong], return_counts=True)
    confuse = int(vals[np.argmax(cnts)]) if len(vals) else int(ytr[hv])
    info = dict(train_index=int(hv), true=int(ytr[hv]), final_weight=float(wfin[hv]), uniform_weight=1 / len(ytr),
                frac_rounds_misclassified=float(wrong.mean()), most_predicted_wrong_class=confuse,
                wrong_class_counts={int(k): int(c) for k, c in zip(vals, cnts)})
    log("heaviest training face:", info); jdump(info, "q2c_heavy_sample.json")
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.4), gridspec_kw=dict(width_ratios=[1, 1, 2.2]))
    ax[0].imshow(Xtr[hv].reshape(64, 64), cmap="gray"); ax[0].set_title(f"heaviest face (id {ytr[hv]})", fontsize=8); ax[0].axis("off")
    other = np.where(ytr == confuse)[0][0]
    ax[1].imshow(Xtr[other].reshape(64, 64), cmap="gray"); ax[1].set_title(f"most-confused id {confuse}", fontsize=8); ax[1].axis("off")
    ax[2].plot(np.arange(1, Rsel + 2), wh, "-"); ax[2].scatter(np.where(wrong)[0] + 1, wh[:-1][wrong], c="r", s=14, label="learner wrong")
    ax[2].set_xlabel("round"); ax[2].set_ylabel("sample weight"); ax[2].legend(); ax[2].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_q2c_heavy.png"), dpi=200); plt.close()

    log("\n===== Q2(d) tree vs bagging vs boosting, clean & masked =====")
    sets = {"clean": Xte, "masked": Xte_m}; res = {}
    for nm, Xs in sets.items():
        res[nm] = dict(tree=tree_pc(tree0, Xs), bag=bag_eval(bag_trees, Xs), boost=boost_eval(Lb, Ab, Xs))
    rows = []
    for nm in sets:
        for mdl in ("tree", "bag", "boost"):
            pr, cf = res[nm][mdl][0], res[nm][mdl][1]
            rows.append(dict(model=mdl, test=nm, acc=(pr == yte).mean(), mean_win_conf=cf.mean()))
    d2d = pd.DataFrame(rows); d2d.to_csv(os.path.join(OUT, "q2d_table.csv"), index=False); log(d2d.round(4).to_string())

    bag_ok = res["masked"]["bag"][0] == yte; bst_ok = res["masked"]["boost"][0] == yte
    cand = np.where(bag_ok != bst_ok)[0]
    if len(cand):
        i = int(cand[np.argmax(np.abs(res["masked"]["bag"][1][cand] - res["masked"]["boost"][1][cand]))])
    else:
        drop = (res["clean"]["bag"][1] - res["masked"]["bag"][1]) - (res["clean"]["boost"][1] - res["masked"]["boost"][1])
        i = int(np.argmax(np.abs(drop)))
    Vm = res["masked"]["bag"][2][i]; Bn = len(bag_trees)
    pc = tree_preds(bag_trees, Xte[i:i + 1])[:, 0]; pm = tree_preds(bag_trees, Xte_m[i:i + 1])[:, 0]
    bh_c = np.array([h.predict(Xte[i:i + 1])[0] for h in Lb]); bh_m = np.array([h.predict(Xte_m[i:i + 1])[0] for h in Lb])
    A_ = np.array(Ab); ident_w = float(wfin[ytr == yte[i]].sum())
    face = dict(test_index=i, true=int(yte[i]),
                bag_clean=(int(bp_c[i]), float(bc_c[i])), bag_masked=(int(res["masked"]["bag"][0][i]), float(res["masked"]["bag"][1][i])),
                boost_clean=(int(res["clean"]["boost"][0][i]), float(res["clean"]["boost"][1][i])),
                boost_masked=(int(res["masked"]["boost"][0][i]), float(res["masked"]["boost"][1][i])),
                bag_trees_changed_pred=float((pc != pm).mean()),
                bag_pairwise_disagreement_masked=float((Bn * Bn - (Vm ** 2).sum()) / (Bn * (Bn - 1))) if Bn > 1 else None,
                bag_masked_path_share=float(np.mean([masked_path_share(t, Xte[i]) for t in bag_trees])),
                boost_alpha_mass_changed_pred=float(A_[bh_c != bh_m].sum() / A_.sum()),
                boost_masked_path_share=float(np.sum([a * masked_path_share(h, Xte[i]) for h, a in zip(Lb, Ab)]) / A_.sum()),
                boost_final_weight_on_true_identity=ident_w, uniform_share=6 / len(ytr),
                bag_overall_masked_split_share=float(np.mean([masked_split_share(t) for t in bag_trees])),
                boost_overall_masked_split_share=float(np.sum([a * masked_split_share(h) for h, a in zip(Lb, Ab)]) / A_.sum()))
    log("differing face:", face); jdump(face, "q2d_face.json")
    show_faces([(Xte[i], f"clean | true {yte[i]}\nbag {bp_c[i]} boost {res['clean']['boost'][0][i]}"),
                (Xte_m[i], f"masked\nbag {face['bag_masked'][0]} boost {face['boost_masked'][0]}")], "fig_q2d_face.png", ncols=2)


    log("\n===== Q3 bias / variance =====")
    R = a.reps; rng3 = np.random.default_rng([seed, 3]); resamples = []
    for _ in range(R):
        idx = []
        for c in range(NC):
            ids = tr[ytr == c]; idx += list(rng3.choice(ids, 6, replace=True))
        resamples.append(np.array(idx))                              # indices into the full X (not into X[tr])
    np.save(os.path.join(OUT, "q3_resamples.npy"), np.stack(resamples))
    Xtests = [Xte, Xte_m]

    log("-- Q3(a) selected tree, bagging(final B), boosting")
    st = tree_bag_study(X, y, resamples, frozen, bsel, Xtests, yte, seed)
    Gb = [[] for _ in Xtests]
    for r, idx in enumerate(resamples):
        L, A, _ = samme_fit(X[idx], y[idx], NC, dsel, Rsel, seed + 1000 * r)
        for j, Xs in enumerate(Xtests): Gb[j].append(boost_eval(L, A, Xs)[2])
    Gb = [np.stack(g) for g in Gb]
    Pm = {"single tree": [st["tree"][0], st["tree"][1]], "bagging": [st["bag"][0], st["bag"][1]], "boosting": Gb}
    rows = []
    for nm, (Pc, Pmk) in Pm.items():
        b2c, vc = bias_var(Pc, yte); b2m, vm = bias_var(Pmk, yte); mc, sc = acc_stats(Pc, yte); mm, sm = acc_stats(Pmk, yte)
        rows.append(dict(model=nm, clean_acc_mean=mc, clean_acc_std=sc, masked_acc_mean=mm, masked_acc_std=sm,
                         bias2_clean=b2c, var_clean=vc, bias2_masked=b2m, var_masked=vm,
                         mse_check_clean=float(((Pc - np.eye(NC)[yte]) ** 2).sum(2).mean())))
    d3 = pd.DataFrame(rows); d3.to_csv(os.path.join(OUT, "q3a_q3d_table.csv"), index=False); log(d3.round(4).to_string())

    log("-- Q3(b) depth study, 100-tree bagged ensembles")
    sel = frozen["max_depth"]
    if sel is None: dl = [("shallow", 2), ("intermediate", 8), ("selected=None (fully grown)", None)]
    else:
        dl = [("shallow", 2 if sel != 2 else 1)]
        dl.append(("selected", sel)); dl.append(("deep (None)", None))
    rows = []
    for nm, md in dl:
        p = dict(max_depth=md, min_samples_leaf=frozen["min_samples_leaf"])
        s = tree_bag_study(X, y, resamples, p, 100, Xtests, yte, seed)
        b2t, vt = bias_var(s["tree"][0], yte); b2g, vg = bias_var(s["bag"][0], yte)
        rows.append(dict(setting=nm, max_depth=md, mean_actual_depth=s["mean_depth"],
                         tree_acc=acc_stats(s["tree"][0], yte)[0], bag_acc=acc_stats(s["bag"][0], yte)[0],
                         acc_gain=acc_stats(s["bag"][0], yte)[0] - acc_stats(s["tree"][0], yte)[0],
                         tree_bias2=b2t, bag_bias2=b2g, tree_var=vt, bag_var=vg, var_removed=vt - vg,
                         var_removed_pct=100 * (vt - vg) / vt if vt > 0 else np.nan,
                         tree_mean_samples_per_leaf=s["tree_leaf"]["mean_samples_per_leaf"],
                         tree_frac_singleton=s["tree_leaf"]["frac_singleton_leaves"],
                         bag_mean_samples_per_leaf=s["bag_leaf"]["mean_samples_per_leaf"],
                         bag_pairwise_disagreement=s["disagreement"]))
    d3b = pd.DataFrame(rows); d3b.to_csv(os.path.join(OUT, "q3b_depth.csv"), index=False); log(d3b.round(4).T.to_string())

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6)); w = 0.25; xs = np.arange(len(d3))
    ax[0].bar(xs - w / 2, d3.bias2_clean, w, label="bias$^2$"); ax[0].bar(xs + w / 2, d3.var_clean, w, label="variance")
    ax[0].set_xticks(xs); ax[0].set_xticklabels(d3.model); ax[0].legend(); ax[0].set_title("clean test", fontsize=9)
    ax[1].bar(xs - w / 2, d3.bias2_masked, w, label="bias$^2$"); ax[1].bar(xs + w / 2, d3.var_masked, w, label="variance")
    ax[1].set_xticks(xs); ax[1].set_xticklabels(d3.model); ax[1].set_title("masked test", fontsize=9)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_q3.png"), dpi=200); plt.close()
    log("\nDONE. Everything is in", OUT)


if __name__ == "__main__":
    main()
