import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

SEED = 42
SEEDS = [42, 123, 456, 789, 2024, 2025, 2026, 3407, 5678, 9999]
VIEWS = ["Frontal", "Left45", "Right45", "Left90", "Right90"]
FOLDS = ["F0", "F1", "F2", "F3", "F4"]
KINDS = {"raw": "Raw", "seg": "Seg"}
SETTINGS = {"Frontal, Left45, Right45, Left90, Right90": VIEWS,
            "Frontal, Laterals": ["Frontal", "Laterals"],
            "Frontal, Left, Right": ["Frontal", "Left", "Right"],
            "9 saídas": VIEWS + ["Laterals", "Left", "Right", "All"]}
LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def f1_at(y, s, thr=0.5):
    p = s >= thr
    tp = np.count_nonzero(p & (y == 1))
    return 2 * tp / (np.count_nonzero(p) + np.count_nonzero(y == 1)) if tp else 0.0


def fit_annealed(P, y, seed=SEED, t0=0.1, cooling=0.95, t_stop=1e-4, min_iters=200, window=100,
                 epsilon=0.001):
    rng = np.random.default_rng(seed)
    n = P.shape[1]

    def move(w):
        w = w.copy()
        i, j = rng.choice(n, 2, replace=False)
        step = rng.uniform(0, w[i])
        w[i] -= step
        w[j] += step
        return w / w.sum()

    cur = np.full(n, 1.0 / n)
    cur_f = f1_at(y, P @ cur)
    best, best_f, t = cur.copy(), cur_f, t0
    while t > t_stop:
        total, it, avgs, diff = 0.0, 0, [], 1.0
        while it < min_iters or diff > epsilon:
            it += 1
            cand = move(cur)
            cand_f = f1_at(y, P @ cand)
            delta = cand_f - cur_f
            if delta >= 0 or rng.random() < np.exp(delta / t):
                cur, cur_f = cand, cand_f
            if cur_f > best_f:
                best, best_f = cur.copy(), cur_f
            total += cur_f
            avgs.append(total / it)
            if it > window:
                diff = abs(avgs[-1] - avgs[-1 - window])
        t *= cooling
    return best


def anneal(P, y, seed, cut, t0=0.1, cooling=0.95, t_stop=1e-4, min_iters=200, window=100, epsilon=0.001,
           frozen_after=5, frozen_rate=0.02):
    rng = np.random.default_rng(seed)
    n = P.shape[1]

    def move(w):
        w = w.copy()
        i, j = rng.choice(n, 2, replace=False)
        step = rng.uniform(0, w[i])
        w[i] -= step
        w[j] += step
        return w

    cur = np.full(n, 1.0 / n)
    cur_f = f1_at(y, P @ cur, cut)
    best_f, plateau, t, frozen = cur_f, [cur.copy()], t0, 0
    while t > t_stop and frozen < frozen_after:
        total, it, avgs, diff, worse, worse_ok, improved = 0.0, 0, [], 1.0, 0, 0, False
        while it < min_iters or diff > epsilon:
            it += 1
            cand = move(cur)
            cand_f = f1_at(y, P @ cand, cut)
            delta = cand_f - cur_f
            if delta < 0:
                worse += 1
            if delta >= 0 or rng.random() < np.exp(delta / t):
                worse_ok += delta < 0
                cur, cur_f = cand, cand_f
            if cur_f > best_f:
                best_f, plateau, improved = cur_f, [cur.copy()], True
            elif cur_f == best_f:
                plateau.append(cur.copy())
            total += cur_f
            avgs.append(total / it)
            if it > window:
                diff = abs(avgs[-1] - avgs[-1 - window])
        frozen = frozen + 1 if (not improved and worse and worse_ok / worse <= frozen_rate) else 0
        t *= cooling
    states = np.array(plateau)
    center = states.mean(axis=0)
    if f1_at(y, P @ center, cut) == best_f:
        return center / center.sum()
    return states[np.argmin(((states - center) ** 2).sum(axis=1))]


def fit_final(P, y, seed, cut):
    n = P.shape[1]
    score = {lam: [] for lam in LAMBDAS}
    held = {lam: np.zeros(len(y)) for lam in LAMBDAS}
    for i, (a, b) in enumerate(RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=seed).split(P, y)):
        w = anneal(P[a], y[a], seed, cut)
        for lam in LAMBDAS:
            held[lam][b] = P[b] @ (lam * w + (1 - lam) / n)
        if i % 5 == 4:
            for lam in LAMBDAS:
                score[lam].append(f1_at(y, held[lam], cut))
    lam = max(LAMBDAS, key=lambda l: (np.mean(score[l]), -l))
    return lam * anneal(P, y, seed, cut) + (1 - lam) / n, lam


def metrics(y, score, pred):
    tn, fp, fn, tp = (int(((pred == a) & (y == b)).sum()) for a, b in ((0, 0), (1, 0), (0, 1), (1, 1)))
    return {"acc": (tp + tn) / len(y), "f1": f1_score(y, pred, zero_division=0),
            "roc_auc": roc_auc_score(y, score), "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn), "tn": tn, "fp": fp, "fn": fn, "tp": tp, "total": len(y)}


def run(data, kind, seed):
    path = Path(data) / KINDS[kind] / f"{kind}_por_fold_{seed}.xlsx"
    train, val, test = (pd.read_excel(path, sheet_name=s) for s in ("train", "val", "test"))
    y = test.True_Label.to_numpy()
    rows, weights = [], []
    for setting, names in SETTINGS.items():
        cols = [f"{c}_Prob" for c in names]
        scores = {"SA": np.zeros(len(test)), "SA final": np.zeros(len(test))}
        cuts = {"SA": np.full(len(test), 0.5), "SA final": np.zeros(len(test))}
        for k in FOLDS:
            v, m = val[val.Fold == k], (test.Fold == k).to_numpy()
            P, yv = v[cols].to_numpy(), v.True_Label.to_numpy()
            cut = float(train[train.Fold == k].True_Label.mean())
            w_final, lam = fit_final(P, yv, seed, cut)
            cuts["SA final"][m] = cut
            for method, wm, l in (("SA", fit_annealed(P, yv, seed), np.nan), ("SA final", w_final, lam)):
                scores[method][m] = test.loc[m, cols].to_numpy() @ wm
                weights += [{"kind": kind, "seed": seed, "setting": setting, "method": method, "fold": k,
                             "cut": cuts[method][m][0], "lambda": l, "input": c, "weight": float(x)} for c, x in zip(names, wm)]
        for method, s in scores.items():
            rows.append({"kind": kind, "seed": seed, "setting": setting, "method": method,
                         **metrics(y, s, (s >= cuts[method]).astype(int))})
    return rows, weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_v2/Pasta final_SA")
    ap.add_argument("--out", default="results")
    ap.add_argument("--kind", action="append", choices=list(KINDS))
    ap.add_argument("--seed", action="append", type=int, choices=SEEDS)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, weights = [], []
    for kind in args.kind or list(KINDS):
        for seed in args.seed or SEEDS:
            r, w = run(args.data, kind, seed)
            rows += r
            weights += w
            pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
            pd.DataFrame(weights).to_csv(out / "weights.csv", index=False)
            print(kind, seed, flush=True)


if __name__ == "__main__":
    main()
