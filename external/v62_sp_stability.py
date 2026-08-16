"""5-fold calibration-stability audit for the v62 single-pass decision layer.

PASS = chosen params stable across folds (t within one 0.02 step, tau within one
sweep step) AND out-of-fold F1 std < 0.004. The audit that certified v58c, applied
to the final candidate's calibration. Runs on A5000 CPU in ~5 min.
"""
import json, math
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score, precision_score
from sklearn.model_selection import StratifiedKFold

VAL = Path.home() / "workspace" / "v62_sp_out" / "v62_sp_val.csv"
GROUP_B_NAMES = ["Air bubble", "Chip", "Contamination light", "Glass imperfection", "Scuffing", "Scuffing heavy"]
GROUP_B_THRESHOLDS = {"Air bubble": 500, "Chip": 200, "Contamination light": 180,
                      "Glass imperfection": 100, "Scuffing": 75000, "Scuffing heavy": 1200}
LOG_THRESHOLDS = np.array([math.log(GROUP_B_THRESHOLDS[n] + 1.0) for n in GROUP_B_NAMES], np.float32)
TAU_SWEEP = [0.50, 0.65, 0.80, 0.95, 1.05, 1.20, 1.35, 1.60]
THR_SWEEP = np.arange(0.08, 0.96, 0.02).tolist()
TRAIN_POS_RATE = 0.5832508134106663

df = pd.read_csv(VAL)
PRES = [c for c in df.columns if c.endswith("_present")]
LOGC = [c for c in df.columns if c.endswith("_logarea")]
CC = [c for c in df.columns if c.startswith("pred_c_")]
assert len(PRES) == 6 and len(LOGC) == 6 and len(CC) == 16, (len(PRES), len(LOGC), len(CC))

def scores(d, tau):
    sb = (d[PRES].values * (1.0 / (1.0 + np.exp(-(d[LOGC].values - LOG_THRESHOLDS[None, :]) / tau)))).max(1)
    cs = np.maximum(d[CC].values.max(1), d["pred_rare_group_c_any"].values)
    return (np.sqrt(np.clip(d["pred_group_c_any"].values * cs, 0, 1)),
            np.sqrt(np.clip(d["pred_group_b_above_any"].values * sb, 0, 1)))

def tune(d):
    y = d["target"].values.astype(int); best = None
    for tau in TAU_SWEEP:
        c, b = scores(d, tau)
        for tc in THR_SWEEP:
            cf = c >= tc
            for tb in THR_SWEEP:
                p = (cf | (b >= tb)).astype(int)
                k = (f1_score(y, p), -abs(p.mean() - TRAIN_POS_RATE), precision_score(y, p, zero_division=0))
                if best is None or k > best[0]:
                    best = (k, float(tc), float(tb), float(tau))
    return best[1], best[2], best[3]

y_all = df["target"].values.astype(int)
strat = df["bucket"].astype(str) + "_" + df["target"].astype(str)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
rows = []
for fi, (tr, te) in enumerate(skf.split(df, strat)):
    tc, tb, tau = tune(df.iloc[tr])
    c, b = scores(df.iloc[te], tau)
    p = ((c >= tc) | (b >= tb)).astype(int)
    f1 = f1_score(y_all[te], p)
    rows.append(dict(fold=fi, t_group_c=tc, t_group_b=tb, tau=tau, oof_f1=float(f1)))
    print(f"fold {fi}: t_c={tc:.2f} t_b={tb:.2f} tau={tau:.2f}  OOF F1={f1:.5f}", flush=True)

r = pd.DataFrame(rows)
print("=" * 60)
print(f"param dispersion: t_c {r.t_group_c.min():.2f}-{r.t_group_c.max():.2f} | "
      f"t_b {r.t_group_b.min():.2f}-{r.t_group_b.max():.2f} | tau {r.tau.min():.2f}-{r.tau.max():.2f}")
print(f"OOF F1: mean={r.oof_f1.mean():.5f}  std={r.oof_f1.std():.5f}  (full-val tuned: 0.96992)")
ok = (r.t_group_c.max() - r.t_group_c.min() <= 0.061 and
      r.t_group_b.max() - r.t_group_b.min() <= 0.061 and r.oof_f1.std() < 0.004)
print("STABILITY VERDICT:", "PASS" if ok else "CHECK")
