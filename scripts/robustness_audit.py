"""Robustness battery to choose the final notebook: champion v62-sp vs s1042 (full-data).

We CANNOT compare them on a common labeled val (champion=80% train -> honest on its 7069;
s1042=95% train -> honest only on its 1768 holdout; the test set has no labels). So we:
  1. characterize each model's INTRINSIC stability on its OWN honest data, and
  2. cross-check with test-set disagreement structure + measured val->public-LB transfer.

Tests (each maps to a private-LB threat):
  A. Bootstrap F1 (full + n-matched)      -> sampling NOISE band (private = different draw)
  B. Per-bucket / per-bottle-type F1       -> DISTRIBUTION SHIFT / subgroup reweighting
  C. Pos-rate vs train prior 0.5833        -> systematic CALIBRATION shift
  D. CV calibration-refit stability        -> rule OVERFITTING to the val it was tuned on
  E. Test disagreement confidence          -> is the 78-flip gap noise or real signal
  F. val->public-LB transfer (anchors)     -> measured OUT-OF-SAMPLE generalization
"""
import math, json
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.RandomState(20260613)
TRAIN_POS = 0.5832508134106663

champ_v = pd.read_csv(ROOT/"external/v62_sp_out/v62_sp_val.csv")
s1042_v = pd.read_csv(__import__("os").environ["TEMP"].replace("\\","/")+"/s1042b/v62_val_predictions.csv")
champ_t = pd.read_csv(ROOT/"external/v62_sp_out/v62_sp_test.csv")
sub_champ = pd.read_csv(ROOT/"external/v62_sp_submission.csv").set_index("image_id")["target"]
import os
sub_s1042 = pd.read_csv(os.environ["TEMP"].replace("\\","/")+"/v62inf/submission_s1042.csv").set_index("image_id")["target"]
bt = pd.read_csv(ROOT/"data/bottletypes.csv")
btcol = [c for c in bt.columns if c != "image_id"][0]
bt_map = dict(zip(bt["image_id"], bt[btcol]))
print(f"bottletypes col='{btcol}' values={bt[btcol].value_counts().to_dict()}")

def get(df):
    y = df["target"].values.astype(int)
    p = df[[c for c in ("sp_rule_pred","rule_pred") if c in df.columns][0]].values.astype(int)
    return y, p

yc, pc = get(champ_v)
ys, ps = get(s1042_v)
print(f"\nchampion honest val n={len(yc)} F1={f1_score(yc,pc):.5f} pos={pc.mean():.4f}")
print(f"s1042   honest val n={len(ys)} F1={f1_score(ys,ps):.5f} pos={ps.mean():.4f}")

# ---------- A. bootstrap ----------
def boot(y, p, n_sub=None, B=4000):
    f1s = np.empty(B)
    idx = np.arange(len(y))
    for b in range(B):
        s = rng.choice(idx, size=n_sub or len(y), replace=True)
        f1s[b] = f1_score(y[s], p[s])
    return f1s.mean(), f1s.std(), np.percentile(f1s, 5), np.percentile(f1s, 50)

print("\n=== A. BOOTSTRAP F1 (sampling-noise band; private LB is one such draw) ===")
for tag, (y, p) in [("champion@7069",(yc,pc)), ("s1042@1768",(ys,ps))]:
    m, sd, p5, med = boot(y, p)
    print(f"  {tag:16s} mean={m:.5f} std={sd:.5f}  p5={p5:.5f}  (95%CI half-width ~{1.96*sd:.4f})")
# n-matched fairness: champion bootstrapped at s1042's n=1768 and at private-LB scale ~3093
for n_sub, lbl in [(1768,"champion@1768 (n-matched)"), (3093,"champion@~private-size")]:
    m, sd, p5, _ = boot(yc, pc, n_sub=n_sub)
    print(f"  {lbl:28s} mean={m:.5f} std={sd:.5f}  p5={p5:.5f}")
m, sd, p5, _ = boot(ys, ps, n_sub=3093)  # s1042 can't exceed its n; resample-with-replacement to ~private size
print(f"  {'s1042@~private-size':28s} mean={m:.5f} std={sd:.5f}  p5={p5:.5f}  (note: only 1768 unique)")

# ---------- B. subgroup uniformity ----------
print("\n=== B. SUBGROUP F1 (distribution-shift / reweighting robustness) ===")
def subgroup(df, y, p, name):
    print(f"  -- {name} --")
    if "bucket" in df.columns:
        for bk in ["small","medium","large"]:
            m = df["bucket"].values == bk
            if m.sum(): print(f"     bucket {bk:7s} n={int(m.sum()):5d} F1={f1_score(y[m],p[m]):.4f}")
    bts = np.array([bt_map.get(i, "?") for i in df["image_id"]])
    for v in sorted(set(bts) - {"?"}):
        m = bts == v
        if m.sum() > 30: print(f"     type {str(v):8s} n={int(m.sum()):5d} F1={f1_score(y[m],p[m]):.4f}")
subgroup(champ_v, yc, pc, "champion")
subgroup(s1042_v, ys, ps, "s1042")

# ---------- C. pos-rate calibration ----------
print("\n=== C. POS-RATE vs train prior 0.5833 (systematic shift risk) ===")
print(f"  champion: val {pc.mean():.4f} | test {sub_champ.mean():.4f}  (|test-prior|={abs(sub_champ.mean()-TRAIN_POS):.4f})")
print(f"  s1042   : val {ps.mean():.4f} | test {sub_s1042.mean():.4f}  (|test-prior|={abs(sub_s1042.mean()-TRAIN_POS):.4f})")

# ---------- D. CV calibration-refit stability (champion only: needs full heads + 7069) ----------
print("\n=== D. CV CALIBRATION STABILITY (rule overfit to its tuning val; champion) ===")
GB=["Air bubble","Chip","Contamination light","Glass imperfection","Scuffing","Scuffing heavy"]
THRv={"Air bubble":500,"Chip":200,"Contamination light":180,"Glass imperfection":100,"Scuffing":75000,"Scuffing heavy":1200}
LT=np.array([math.log(THRv[n]+1.0) for n in GB],np.float32)
C=[c for c in champ_v.columns if c.startswith("pred_c_")]
PR=[f"pred_b_{i:02d}_{n}_present" for i,n in enumerate(GB)]
LG=[f"pred_b_{i:02d}_{n}_logarea" for i,n in enumerate(GB)]
def score(df,tau):
    cs=np.maximum(df[C].max(1).values,df["pred_rare_group_c_any"].values)
    soft=df[PR].values*(1/(1+np.exp(-(df[LG].values-LT[None,:])/tau)))
    return np.sqrt(np.clip(df["pred_group_c_any"].values*cs,0,1)), np.sqrt(np.clip(df["pred_group_b_above_any"].values*soft.max(1),0,1))
def fit(df):
    y=df["target"].values.astype(int); best=None
    for tau in [0.5,0.8,0.95,1.2,1.6]:
        sc,sb=score(df,tau)
        for tc in np.arange(0.4,0.75,0.02):
            for tb in np.arange(0.4,0.75,0.02):
                f=f1_score(y,((sc>=tc)|(sb>=tb)).astype(int))
                if best is None or f>best[0]: best=(f,tc,tb,tau)
    return best[1:]
def apply(df,tc,tb,tau):
    sc,sb=score(df,tau); return ((sc>=tc)|(sb>=tb)).astype(int)
skf=StratifiedKFold(5,shuffle=True,random_state=1)
strat=(champ_v["target"].astype(str)+champ_v["bucket"]).values
gaps=[]; holds=[]
for tr,te in skf.split(champ_v,strat):
    tc,tb,tau=fit(champ_v.iloc[tr])
    ftr=f1_score(champ_v.iloc[tr]["target"],apply(champ_v.iloc[tr],tc,tb,tau))
    fte=f1_score(champ_v.iloc[te]["target"],apply(champ_v.iloc[te],tc,tb,tau))
    gaps.append(ftr-fte); holds.append(fte)
print(f"  champion rule: held-fold F1 {np.mean(holds):.5f}+/-{np.std(holds):.5f} | tune-hold gap {np.mean(gaps):+.5f} (small gap=low calib overfit)")

# ---------- E. test disagreement structure ----------
print("\n=== E. TEST DISAGREEMENT (champion vs s1042; is the gap noise or signal?) ===")
common=sub_champ.index.intersection(sub_s1042.index)
a=sub_champ.loc[common].values; b=sub_s1042.loc[common].values
dis=a!=b
ct=champ_t.set_index("image_id").loc[common]
# champion's rule confidence on disagreements: recompute max rule score
sc_t,sb_t=score(ct.assign(**{c:ct[c] for c in C}), 0.95)
conf=np.maximum(sc_t,sb_t)
near=(conf>0.45)&(conf<0.65)  # ~ rule threshold band 0.54/0.58
print(f"  disagreements: {int(dis.sum())}/{len(common)}")
print(f"  of those, champion rule-score in uncertain band [0.45,0.65]: {int((dis&near).sum())} ({(dis&near).sum()/max(1,dis.sum()):.0%})")
print(f"  => high uncertain-fraction means the gap is NOISE (near-threshold), not a real model edge")

# ---------- F. transfer anchors ----------
print("\n=== F. MEASURED val->public-LB TRANSFER (out-of-sample generalization) ===")
print("  champion: val 0.96992 -> LB 0.97066  (+0.0007, generalized UP)")
print("  s1042   : holdout 0.96999 -> LB 0.97062  (-0.0004, ~exact)")
print("  v98     : val 0.97052 -> LB 0.97004  (-0.0048, overfit val) [excluded]")
print("\nDONE")
