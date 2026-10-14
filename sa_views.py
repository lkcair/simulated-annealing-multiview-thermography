import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

SEED = 42
VIEWS = ["Frontal", "Left45", "Right45", "Left90", "Right90"]
FOLDS = ["F0", "F1", "F2", "F3", "F4"]
THREE = ["Frontal", "Left", "Right"]
TWO = ["Frontal", "Laterals"]
FILES = {"ResNet18 raw": "raw_por_fold resnet18.xlsx", "ResNet18 seg": "seg_por_fold resnet18.xlsx",
         "ResNet18 truncada raw": "raw_por_fold resnet truncada.xlsx",
         "ResNet18 truncada seg": "seg_por_fold resnet truncada.xlsx",
         "VGG16 raw": "raw_por_fold vgg16.xlsx", "VGG16 seg": "seg_por_fold vgg16.xlsx"}
SOURCES = {**{name: (f, VIEWS) for name, f in FILES.items()},
           **{f"ResNet18 truncada {kind} {len(cols)}attr": (FILES[f"ResNet18 truncada {kind}"], cols)
              for kind in ("raw", "seg") for cols in (THREE, TWO)}}
GRID = [(t0, cooling) for t0 in (0.1, 0.01, 0.001) for cooling in (0.90, 0.95)]
BASE = (0.1, 0.95)
INNER = 5
SA_NAMES = ["SA base", "SA tuned", "SA bagged", "SA pruned", "SA nested"]
THRESHOLDED = "SA base, threshold on val"
COLUMNS = ["acc", "f1", "roc_auc", "precision", "recall", "tn", "fp", "fn", "tp", "f1 mean over seeds",
           "f1 sd over seeds"]


def read(data_dir, name, split, prefix=""):
    path, inputs = SOURCES[name]
    t = pd.read_excel(Path(data_dir) / path, sheet_name=split)
    cols = {f"{v}_Prob": f"{prefix}{v}" for v in inputs}
    return t[["ID", "Fold", "True_Label", *cols]].rename(columns=cols)


def load(data_dir, name):
    if name != "All sources":
        return {s: read(data_dir, name, s) for s in ("val", "test")}
    out = {}
    for s in ("val", "test"):
        parts = [read(data_dir, n, s, f"{n} | ") for n in FILES]
        merged = parts[0]
        for p in parts[1:]:
            merged = merged.merge(p, on=["ID", "Fold", "True_Label"])
        out[s] = merged
    return out


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
        total, it, avgs, diff = cur_f, 0, [], 1.0
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


def fit_bagged(P, y, seed, schedule, n_bags=5):
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    draws = [np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))] for _ in range(n_bags)]
    return np.mean([fit_annealed(P[i], y[i], seed, *schedule) for i in draws], axis=0)


def top_views(P, y, top=2):
    return np.argsort([-f1_at(y, P[:, j]) for j in range(P.shape[1])], kind="stable")[:top]


def fit_pruned(P, y, seed, schedule):
    keep = top_views(P, y)
    w = np.zeros(P.shape[1])
    w[keep] = fit_annealed(P[:, keep], y, seed, *schedule)
    return w


def held_out_f1(fit, P, y):
    held = np.zeros(len(y))
    for a, b in StratifiedKFold(INNER, shuffle=True, random_state=SEED).split(P, y):
        held[b] = P[b] @ fit(P[a], y[a])
    return f1_at(y, held)


def best_threshold(y, s):
    grid = np.round(np.arange(0.05, 0.951, 0.05), 2)
    return float(grid[np.argmax([f1_at(y, s, t) for t in grid])])


def fold_variants(P, y, seed):
    grid = {g: held_out_f1(lambda A, b, g=g: fit_annealed(A, b, seed, *g), P, y) for g in GRID}
    tuned = max(GRID, key=grid.get)
    fits = {"SA base": lambda A, b: fit_annealed(A, b, seed, *BASE),
            "SA tuned": lambda A, b: fit_annealed(A, b, seed, *tuned),
            "SA bagged": lambda A, b: fit_bagged(A, b, seed, tuned),
            "SA pruned": lambda A, b: fit_pruned(A, b, seed, tuned)}
    weights = {k: f(P, y) for k, f in fits.items()}
    inner = {"SA tuned": grid[tuned], **{k: held_out_f1(fits[k], P, y) for k in ("SA bagged", "SA pruned")}}
    pick = max(inner, key=inner.get)
    weights["SA nested"] = weights[pick]
    return weights, tuned, pick


def metrics(y, score, pred):
    tn, fp, fn, tp = (int(((pred == a) & (y == b)).sum()) for a, b in ((0, 0), (1, 0), (0, 1), (1, 1)))
    return {"acc": (tp + tn) / len(y), "f1": f1_score(y, pred, zero_division=0),
            "roc_auc": roc_auc_score(y, score), "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn), "tn": tn, "fp": fp, "fn": fn, "tp": tp, "total": len(y)}


def fold_record(fold, tuned, pick, thr, cols, w, pruned):
    return {"fold": fold, "tuned schedule": f"t0={tuned[0]}, cooling={tuned[1]}", "nested pick": pick,
            "threshold": thr, "pruned views": " + ".join(cols[j] for j in pruned),
            **{f"w {c}": round(x, 4) for c, x in zip(cols, w)}}


def run_source(data, seeds):
    val, test = data["val"], data["test"]
    cols = [c for c in val.columns if c not in ("ID", "Fold", "True_Label")]
    y_test = test.True_Label.to_numpy()
    rows, per_seed, chosen = {}, {}, []
    for c in cols:
        rows[c] = metrics(y_test, test[c].to_numpy(), (test[c].to_numpy() >= 0.5).astype(int))
    mean = test[cols].to_numpy().mean(axis=1)
    rows["Mean of views"] = metrics(y_test, mean, (mean >= 0.5).astype(int))
    for seed in seeds:
        scores = {k: np.zeros(len(test)) for k in ["Best view on val", *SA_NAMES, THRESHOLDED]}
        thr_pred = np.zeros(len(test), dtype=int)
        for k in FOLDS:
            v, t, mask = val[val.Fold == k], test[test.Fold == k], (test.Fold == k).to_numpy()
            Pv, yv, Pt = v[cols].to_numpy(), v.True_Label.to_numpy(), t[cols].to_numpy()
            lead = int(np.argmax([f1_at(yv, Pv[:, j]) for j in range(len(cols))]))
            scores["Best view on val"][mask] = Pt[:, lead]
            weights, tuned, pick = fold_variants(Pv, yv, seed)
            for name, w in weights.items():
                scores[name][mask] = Pt @ w
            thr = best_threshold(yv, Pv @ weights["SA base"])
            scores[THRESHOLDED][mask] = Pt @ weights["SA base"]
            thr_pred[mask] = scores[THRESHOLDED][mask] >= thr
            if seed == seeds[0]:
                chosen.append(fold_record(k, tuned, pick, thr, cols, weights["SA base"], top_views(Pv, yv)))
        for name, s in scores.items():
            pred = thr_pred if name == THRESHOLDED else (s >= 0.5).astype(int)
            per_seed.setdefault(name, []).append(metrics(y_test, s, pred))
    for name, ms in per_seed.items():
        rows[name] = ms[0]
        rows[name]["f1 mean over seeds"] = float(np.mean([m["f1"] for m in ms]))
        rows[name]["f1 sd over seeds"] = float(np.std([m["f1"] for m in ms]))
    return pd.DataFrame(rows).T, pd.DataFrame(chosen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--source", action="append", choices=[*SOURCES, "All sources"])
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = [SEED + i for i in range(args.seeds)]
    pd.set_option("display.width", 250)
    with pd.ExcelWriter(out / "sa_resultados.xlsx") as xl:
        for name in args.source or [*SOURCES, "All sources"]:
            table, chosen = run_source(load(args.data, name), seeds)
            table.to_excel(xl, sheet_name=name[:31])
            chosen.to_excel(xl, sheet_name=f"{name[:24]} folds")
            print(f"\n=== {name}", flush=True)
            print(table[COLUMNS].to_string(float_format=lambda v: f"{v:.4f}"), flush=True)
            print(chosen.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
