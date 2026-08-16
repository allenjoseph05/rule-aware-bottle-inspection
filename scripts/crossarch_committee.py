"""Cross-ARCHITECTURE committee: the unexplored axis. The 6-model committee saturated
because it was all ConvNeXt/rule family (correlated). Here we add the DECORRELATED
ViT models (v77, v92) and test on the 1396 images held out by BOTH splits.

Hypothesis: a flip confirmed by INDEPENDENT architectures (ViT + seg + ConvNeXt) is
higher-precision than within-family agreement -> could transfer where v72-78 didn't.
Honest caveat: v77/v92 overfit their val (val->LB -0.01), so overlap precision may
OVERESTIMATE test transfer; we require a high bar (>=0.8) before probing.
"""
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score
ROOT = Path(__file__).resolve().parents[1]
TRAIN_POS = 0.5833

def load(path, idcol="image_id"):
    return pd.read_csv(ROOT/path)

base = load('external/v62_sp_out/v62_sp_val.csv')
ids_base = base.set_index('image_id')
# models: (name, path, kind, col, arch)
M = [
 ("v77","artefacts/v77_vit_base/v77_val_predictions.csv","prob","v77_prob","ViT"),
 ("v92","external/v92_corrector_out/v92_corrector_val.csv","prob","bin_prob","ViT"),
 ("v66","artefacts/kaggle_pull/krones-v66-decision-aligned-seg/v66_val_predictions.csv","rule","seg_rule_pred","seg"),
 ("v95","external/v95_kernel/out/v95_val_predictions.csv","rule","seg_rule_pred","seg"),
 ("v57","artefacts/v57_recovered/v57_val_predictions.csv","prob","pred_target","cnxt"),
 ("v98","external/v98_out/v62_val_predictions.csv","prob","pred_target","cnxt"),
 ("v93","external/v93_kernel/out/v93_val_predictions.csv","prob","pred_target","cnxt"),
]
# prior-matched binary decision per model (threshold on full val, no label leakage)
dec = {}
for name,path,kind,col,arch in M:
    d = load(path).set_index('image_id')
    if kind=="rule":
        s = d[col].astype(int)
    else:
        v = d[col].astype(float)
        thr = v.quantile(1.0-TRAIN_POS)
        s = (v>=thr).astype(int)
    dec[name] = s
# overlap = ids present in ALL models + base
common = set(base.image_id)
for name in dec: common &= set(dec[name].index)
common = [i for i in base.image_id if i in common]
print(f"cross-arch common holdout: {len(common)} images")
y = base.set_index('image_id').loc[common,'target'].values.astype(int)
b = base.set_index('image_id').loc[common,'sp_rule_pred'].values.astype(int)
D = {n: dec[n].loc[common].values for n in dec}
arch = {name:a for name,_,_,_,a in M}
print(f"overlap base(v62sp) F1={f1_score(y,b):.5f} pos={b.mean():.4f}  true pos={y.mean():.4f}")
for n in D: print(f"  {n:5} ({arch[n]:4}) F1={f1_score(y,D[n]):.4f} agree_w_base={ (D[n]==b).mean():.3f}")

print("\n=== cross-architecture AGREEMENT flips on base v62sp ===")
def test_rule(mask_add, mask_rm, label):
    add = (b==0)&mask_add; rm = (b==1)&mask_rm
    ap = (y[add]==1).mean() if add.sum() else float('nan')
    rp = (y[rm]==0).mean() if rm.sum() else float('nan')
    p=b.copy(); p[add]=1; p[rm]=0
    print(f"{label:42} ADD {int(add.sum()):3d}@{ap:.2f}  RM {int(rm.sum()):3d}@{rp:.2f}  F1 {f1_score(y,p):.5f} (base {f1_score(y,b):.5f})")

vit_add = (D['v77']==1)&(D['v92']==1)         # both ViT say 1
vit_rm  = (D['v77']==0)&(D['v92']==0)
test_rule(vit_add, vit_rm, "2 ViT agree (v77+v92)")
# cross-arch: ViT AND seg AND cnxt all agree
ca_add = (D['v77']==1)&(D['v66']==1)&(D['v57']==1)
ca_rm  = (D['v77']==0)&(D['v66']==0)&(D['v57']==0)
test_rule(ca_add, ca_rm, "ViT+seg+cnxt all agree (3-arch)")
# stronger: ViT(both) + seg(either) + cnxt(any 2 of 3)
cnxt_ct = D['v57']+D['v98']+D['v93']; seg_ct=D['v66']+D['v95']
sa_add=(D['v77']==1)&(D['v92']==1)&(seg_ct>=1)&(cnxt_ct>=2)
sa_rm =(D['v77']==0)&(D['v92']==0)&(seg_ct<=1)&(cnxt_ct<=1)
test_rule(sa_add, sa_rm, "2ViT + >=1seg + >=2cnxt (super-majority)")
# ALL 7 unanimous
allv = np.stack([D[n] for n in D],1)
una_add=(allv.sum(1)==7); una_rm=(allv.sum(1)==0)
test_rule(una_add, una_rm, "ALL 7 models unanimous")

print("\nDECISION: a rule with ADD precision AND RM precision both >=0.8 on this honest")
print("overlap (discounted for v77/v92 overfit) is worth probing on the v70 test base.")
PY
