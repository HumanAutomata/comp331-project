"""
predict_g3.py generalised to a survey with no single target column.

predict_g3 predicts ONE target column (G3) from every other column, in two scenarios:
  1. all features, including G1 and G2 (the two strongest predictors)
  2. without G1 and G2
A survey has no single "grade" column, so every question takes a turn as the target:
  Scenario 1: predict the question from all 134 other questions.
  Scenario 2: same, but drop the 2 questions most correlated with the target (the G1/G2 analogue).
              The 2 are chosen on the TRAINING rows of each split, so the test rows play no part in it.
The baseline "copy most-correlated question" plays the role of "copy G2".

The team's user_cf, item_cf, minmax_fit_transform, rmse and mae are used unchanged. Two changes were
needed for this data:
  - The survey has real missing answers. Rows with a missing target are dropped. Missing feature values are
    filled with the training-column mean (cosine similarity cannot handle NaN).
  - Predictions are clipped to [1, 5] instead of [0, 20].

Imported by run_bigfive.py, which supplies the data loader.
"""
import sys
import types

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.modules.setdefault("ucimlrepo", types.SimpleNamespace(fetch_ucirepo=None))
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "student_performance"))
import predict_g3 as pg  # noqa: E402


def load_data(path):
    df = pd.read_csv(path)
    num = df.select_dtypes("number")
    likert = [c for c in num.columns if num[c].min() >= 1 and num[c].max() <= 5]
    return num[likert].astype(float)


def top_correlated(train_feats, train_target, n):
    """Indices of the n feature columns most correlated (|Pearson|) with the target, on training rows."""
    f = np.where(np.isnan(train_feats), np.nanmean(train_feats, 0), train_feats)
    fz = (f - f.mean(0)) / np.where(f.std(0) > 0, f.std(0), 1)
    tz = (train_target - train_target.mean()) / train_target.std()
    corr = (fz * tz[:, None]).mean(0)
    return np.argsort(-np.abs(corr))[:n], corr


def run_target(X, cols, t, seeds):
    y_all = X[:, t]
    keep = ~np.isnan(y_all)
    feats_all = np.delete(X, t, axis=1)[keep]
    y = y_all[keep]
    rows = []
    for seed in range(seeds):
        tr, te = train_test_split(np.arange(len(y)), test_size=pg.TEST_SIZE, random_state=seed)
        top2, corr = top_correlated(feats_all[tr], y[tr], 2)
        for scen, drop in (("1: all other questions", []), ("2: without top-2 correlated", list(top2))):
            F = np.delete(feats_all, drop, axis=1) if drop else feats_all
            Ftr, Fte = F[tr], F[te]
            mu = np.nanmean(Ftr, 0)
            Ftr = np.where(np.isnan(Ftr), mu, Ftr)
            Fte = np.where(np.isnan(Fte), mu, Fte)
            Str, Ste = pg.minmax_fit_transform(Ftr, Fte)
            preds = {
                "user-CF (cosine)": pg.user_cf(Str, y[tr], Ste),
                "item-CF (PCC)": pg.item_cf(Str, y[tr], Ste),
                "global mean": np.full(len(te), y[tr].mean()),
            }
            if not drop:  # analogue of "copy G2": copy the single most-correlated question
                best = top2[0]
                c = feats_all[te, best]
                c = np.where(np.isnan(c), np.nanmean(feats_all[tr, best]), c)
                # a negatively correlated question is reversed on the 1-5 scale before copying
                preds["copy most-correlated question"] = c if corr[best] >= 0 else 6 - c
            for m, p in preds.items():
                p = np.clip(p, 1, 5)
                rows.append(dict(target=cols[t], scenario=scen, seed=seed, method=m,
                                 RMSE=pg.rmse(p, y[te]), MAE=pg.mae(p, y[te])))
    return rows


def main():
    Xdf = load_data(sys.argv[1])
    X, cols = Xdf.to_numpy(), list(Xdf.columns)
    rows = []
    for t in range(len(cols)):
        rows += run_target(X, cols, t, pg.N_SEEDS)
    res = pd.DataFrame(rows)
    

    # per target: mean over seeds; then summarise across the 135 targets
    per_target = res.groupby(["scenario", "target", "method"])[["RMSE", "MAE"]].mean().reset_index()
    summ = per_target.groupby(["scenario", "method"])[["RMSE", "MAE"]].agg(["mean", "std"]).round(3)
    print(f"\n=== Each of {len(cols)} questions predicted from the others "
          f"({pg.N_SEEDS} random 80/20 splits per question; mean and std ACROSS questions) ===")
    print(summ.sort_values([("MAE", "mean")]).sort_index(level=0, sort_remaining=False).to_string())

    # how often does user-CF beat the global mean, per question?
    pv = per_target.pivot_table(index=["scenario", "target"], columns="method", values="MAE")
    for scen, g in pv.groupby(level=0):
        for m in ("user-CF (cosine)", "item-CF (PCC)"):
            wins = int((g[m] < g["global mean"]).sum())
            print(f"{scen} | {m} beats global mean on {wins}/{len(g)} questions")

    # the questions CF predicts best and worst (scenario 1, user-CF, as MAE gain over the mean)
    s1 = pv.loc["1: all other questions"].copy()
    s1["gain vs mean"] = s1["global mean"] - s1["user-CF (cosine)"]
    print("\nScenario 1, best 5 questions for user-CF (MAE gain over global mean):")
    print(s1.sort_values("gain vs mean", ascending=False).head(5).round(3).to_string())
    print("\nScenario 1, worst 5:")
    print(s1.sort_values("gain vs mean").head(5).round(3).to_string())


if __name__ == "__main__":
    main()
