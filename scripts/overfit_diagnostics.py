"""Tier-0 overfit diagnostics — runs locally on predictions we already have.

The core question: do our combiner rules GENERALIZE (help on held-out val), or do
they only help on the public LB (= adaptive overfitting)?

val != public LB. So if a rule improves the 7069-image val F1, that's genuine
generalization. If a rule's val-lift is inconsistent across folds, it's fitting noise.

Diagnostics:
  1. v56 base F1 on val
  2. v58c rule val-lift (+ per-fold consistency)
  3. v59 rule val-lift (+ per-fold consistency)
  4. Calibration (reliability) of v5/v56 probabilities on val
  5. Prediction-distribution shift: val vs test (adversarial-val proxy)

NOTE: v70 triple-source needs v67 val (missing) — flagged, needs GPU re-inference.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]

# Common 7069 val split (v5/v56/v57/v66 share it)
v5v = pd.read_csv(ROOT/"artefacts/kaggle_v56_v3_output/v5_val_predictions.csv")
v56v = pd.read_csv(ROOT/"artefacts/kaggle_v56_v3_output/v56_val_predictions.csv")
v57v = pd.read_csv(ROOT/"v57_output/v57_val_predictions.csv")
v66v = pd.read_csv(ROOT/"artefacts/kaggle_pull/krones-v66-decision-aligned-seg/v66_val_predictions.csv")

# Merge on image_id
m = (v56v[["image_id","target","v5_prob","final_prob","aux_distractor_only_prob","final_pred"]]
     .merge(v57v[["image_id","score_target_aux","score_group_c","score_group_b","pred_distractor_any",
                  "pred_c_02_Contamination dark","pred_b_04_Scuffing_above"]], on="image_id")
     .merge(v66v[["image_id","prob_p_cont_dark_trigger","prob_p_group_c_trigger",
                  "prob_p_scuffing_above_75000","px_native_group_c_union"]], on="image_id"))
y = m["target"].astype(int).values
print(f"common val: {len(m)} images, pos rate {y.mean():.4f}")

# === 1. v56 base ===
base = m["final_pred"].astype(int).values
f1_base = f1_score(y, base)
print(f"\n[1] v56 base val F1: {f1_base:.4f}")

# === 2. v58c rule on val ===
def apply_v58c(pred, m):
    pred = pred.copy()
    add = ((pred==0) & (m["score_target_aux"]>=0.80).values &
           ((m["score_group_c"]>=0.62)|(m["score_group_b"]>=0.62)).values)
    rem = ((pred==1) & (m["score_group_c"]<=0.36).values &
           (m["score_group_b"]<=0.20).values & (m["pred_distractor_any"]>=0.70).values)
    pred[add]=1; pred[rem]=0
    return pred, add.sum(), rem.sum()
v58c, na, nr = apply_v58c(base, m)
f1_v58c = f1_score(y, v58c)
print(f"[2] v58c rule: ADD {na}, REM {nr}, val F1 {f1_v58c:.4f}  (lift {f1_v58c-f1_base:+.4f})")

# === 3. v59 rule on val ===
def apply_v59(pred, m):
    pred = pred.copy()
    a_cd=((pred==0)&(m["prob_p_cont_dark_trigger"]>=0.80).values&(m["pred_c_02_Contamination dark"]>=0.50).values&(m["v5_prob"]<=0.90).values)
    a_gc=((pred==0)&(m["prob_p_group_c_trigger"]>=0.90).values&(m["score_group_c"]>=0.50).values&(m["px_native_group_c_union"]>=50).values)
    a_sa=((pred==0)&(m["prob_p_scuffing_above_75000"]>=0.85).values&(m["score_group_c"]>=0.50).values)
    pred[a_cd|a_gc|a_sa]=1
    return pred, (a_cd|a_gc|a_sa).sum()
v59, n59 = apply_v59(v58c, m)
f1_v59 = f1_score(y, v59)
print(f"[3] v59 rule: +{n59} ADD, val F1 {f1_v59:.4f}  (lift vs v58c {f1_v59-f1_v58c:+.4f}, vs base {f1_v59-f1_base:+.4f})")

# === Per-fold consistency of the FULL rule (v58c+v59) ===
print("\n[4] Per-fold val-lift consistency (5 folds):")
rng = np.random.RandomState(42)
idx = rng.permutation(len(m))
folds = np.array_split(idx, 5)
lifts_v58c, lifts_v59 = [], []
for fi, fold in enumerate(folds):
    yy = y[fold]
    b = base[fold]
    p58, _, _ = apply_v58c(b, m.iloc[fold].reset_index(drop=True))
    p59, _ = apply_v59(p58, m.iloc[fold].reset_index(drop=True))
    l58 = f1_score(yy, p58) - f1_score(yy, b)
    l59 = f1_score(yy, p59) - f1_score(yy, p58)
    lifts_v58c.append(l58); lifts_v59.append(l59)
    print(f"  fold {fi}: v58c lift {l58:+.4f}, v59 lift {l59:+.4f}")
print(f"  v58c: mean {np.mean(lifts_v58c):+.4f} ± {np.std(lifts_v58c):.4f}  (folds positive: {sum(1 for x in lifts_v58c if x>0)}/5)")
print(f"  v59 : mean {np.mean(lifts_v59):+.4f} ± {np.std(lifts_v59):.4f}  (folds positive: {sum(1 for x in lifts_v59 if x>0)}/5)")

# === 5. Calibration (reliability) of v5 / v56 probs on val ===
print("\n[5] Calibration (reliability) — fraction positive per prob bin:")
for name, probs in [("v5_prob", m["v5_prob"].values), ("v56_final_prob", m["final_prob"].values)]:
    print(f"  {name}:")
    bins = np.linspace(0,1,11)
    for i in range(10):
        lo,hi = bins[i],bins[i+1]
        mask=(probs>=lo)&(probs<hi)
        if mask.sum()>20:
            print(f"    [{lo:.1f},{hi:.1f}) n={mask.sum():4d} pred~{(lo+hi)/2:.2f} actual_pos={y[mask].mean():.3f}")

# === 6. Prediction-distribution shift: val vs test ===
print("\n[6] Distribution shift (val vs test) per signal — KS-style mean/std compare:")
v56t = pd.read_csv(ROOT/"artefacts/v70_signals_kaggle_dataset/v56_forced_test_predictions.csv")
v57t = pd.read_csv(ROOT/"artefacts/v70_signals_kaggle_dataset/v57_test_predictions.csv")
v66t = pd.read_csv(ROOT/"artefacts/v70_signals_kaggle_dataset/v66_test_predictions.csv")
for name, vcol, tdf, tcol in [
    ("v5_prob", m["v5_prob"].values, v56t, "v5_prob"),
    ("score_group_c", m["score_group_c"].values, v57t, "score_group_c"),
    ("v66_group_c_trig", m["prob_p_group_c_trigger"].values, v66t, "prob_p_group_c_trigger"),
]:
    tv = tdf[tcol].values
    print(f"  {name:18s}: val mean {vcol.mean():.3f}/std {vcol.std():.3f}  |  test mean {tv.mean():.3f}/std {tv.std():.3f}  |  shift {abs(vcol.mean()-tv.mean()):.3f}")

print("\n=== Interpretation ===")
print("- If v58c/v59 val-lift > 0 AND positive in 4-5/5 folds -> rule GENERALIZES (not just public-LB overfit).")
print("- If calibration actual_pos ~ pred bin center -> well-calibrated -> thresholds transfer.")
print("- If val vs test distribution shift is small -> val is trustworthy.")
print("- v70 triple-source NOT evaluable here (v67 val missing) -> needs GPU re-inference for full rigor.")
