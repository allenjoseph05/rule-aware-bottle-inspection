"""v62 SINGLE-PASS measurement (no TTA) — the last open F1-at-v5-cost question.

v62's 0.97004 LB used 2x h-flip TTA (2 forwards). Stored prediction CSVs are
TTA-averaged, so the single-pass operating point was never measured. This script:
  1. loads best_model_v62.pth, runs ONE forward per image (val 7069 + test 4418)
  2. re-tunes the decision calibration (global + per-bucket tau) on single-pass val
  3. reports single-pass val F1 vs the gate (>= 0.9680 to justify an LB probe)
  4. writes the single-pass test submission + calib json

Runs on A5000 CPU only (GPU belongs to dll0402). ~25-45 min.
  cd ~/workspace && nohup ~/krones_venv/bin/python -u v62_singlepass.py > v62_sp.log 2>&1 &
"""
import os, json, math, time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import f1_score, precision_score, recall_score
import timm

torch.set_num_threads(os.cpu_count())
DATA = Path.home() / "workspace" / "data"
CKPT = Path.home() / "workspace" / "best_model_v62.pth"
OUT = Path.home() / "workspace" / "v62_sp_out"; OUT.mkdir(exist_ok=True)
VAL_CSV = Path.home() / "workspace" / "v62_val_predictions.csv"   # only for ids/target/bucket/roi

S = 640; PAD = 0.10
TRAIN_POS_RATE = 0.5832508134106663
GROUP_A = ["Embossing", "Foam residue", "No fault", "Water drop"]
GROUP_B_NAMES = ["Air bubble", "Chip", "Contamination light", "Glass imperfection", "Scuffing", "Scuffing heavy"]
GROUP_B_THRESHOLDS = {"Air bubble": 500, "Chip": 200, "Contamination light": 180,
                      "Glass imperfection": 100, "Scuffing": 75000, "Scuffing heavy": 1200}
LOG_AREA_PAD = 1.0
TAU_SWEEP = [0.50, 0.65, 0.80, 0.95, 1.05, 1.20, 1.35, 1.60]
THR_SWEEP = np.arange(0.08, 0.96, 0.02).tolist()
MEAN = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
STD = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]

# --- categories from COCO (GROUP_C order must match training: sorted) ---
with open(DATA / "train_annotations.json") as f:
    coco = json.load(f)
cat_name = {c["id"]: c["name"] for c in coco["categories"]}
GROUP_C_NAMES = sorted(set(cat_name.values()) - set(GROUP_A) - set(GROUP_B_NAMES) - {"Roi"})
LOG_THRESHOLDS = np.array([math.log(GROUP_B_THRESHOLDS[n] + LOG_AREA_PAD) for n in GROUP_B_NAMES], np.float32)
fname_to_imgid = {im["file_name"]: im["id"] for im in coco["images"]}
roi_by_img = {}
for a in coco["annotations"]:
    if cat_name.get(a["category_id"]) == "Roi":
        roi_by_img.setdefault(a["image_id"], a["bbox"])

def to_bucket(w):
    w = float(w)
    return "small" if w < 510 else ("large" if w > 590 else "medium")

# --- val set = exactly the rows of the saved val CSV (no split re-derivation) ---
val_ref = pd.read_csv(VAL_CSV, usecols=["image_id", "target", "bucket"])
val_roi = {f: roi_by_img[fname_to_imgid[f]] for f in val_ref["image_id"]}
print(f"val rows: {len(val_ref)}  buckets: {val_ref['bucket'].value_counts().to_dict()}")

with open(DATA / "test_annotations_roi_only.json") as f:
    troi = json.load(f)
tcat = {c["id"]: c["name"] for c in troi["categories"]}
tby = {}
for a in troi["annotations"]:
    if tcat.get(a["category_id"]) == "Roi":
        tby.setdefault(a["image_id"], a["bbox"])
test_roi = {im["file_name"]: tby[im["id"]] for im in troi["images"]}
sample_sub = pd.read_csv(DATA / "sample_submission.csv")
print(f"test rows: {len(sample_sub)}  roi coverage: {len(test_roi)}")

def roi_crop_box(W, H, roi, pad=PAD):
    x, y, w, h = [float(v) for v in roi[:4]]
    side = int(round(max(w, h) * (1 + 2 * pad))); side = max(1, min(side, W, H))
    cx, cy = x + w / 2, y + h / 2
    l = max(0, min(int(round(cx - side / 2)), W - side))
    t = max(0, min(int(round(cy - side / 2)), H - side))
    return l, t, side, side

class SPDataset(Dataset):
    def __init__(self, ids, roi_lookup, img_dir):
        self.ids = list(ids); self.roi = roi_lookup; self.dir = Path(img_dir)
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        f = self.ids[i]; roi = self.roi[f]
        img = Image.open(self.dir / f).convert("L"); W, H = img.size
        l, t, sw, sh = roi_crop_box(W, H, roi)
        crop = img.crop((l, t, l + sw, t + sh))
        if crop.size != (S, S): crop = crop.resize((S, S), Image.BILINEAR)
        arr = np.asarray(crop, np.float32)
        x = (np.stack([arr, arr, arr], 0) / 255.0 - MEAN) / STD
        b = to_bucket(roi[2])
        bv = np.array([b == "small", b == "medium", b == "large"], np.float32)
        rg = np.array([roi[2] / S, roi[3] / S], np.float32)
        return torch.from_numpy(np.ascontiguousarray(x)).float(), torch.from_numpy(bv), torch.from_numpy(rg), f

class AugSmartModel(nn.Module):
    def __init__(self, backbone_name, n_group=4, topk_frac=0.08):
        super().__init__()
        self.encoder = timm.create_model(backbone_name, pretrained=False, num_classes=0, global_pool="")
        self.topk_frac = topk_frac
        feat = int(self.encoder.num_features)
        self.trigger_map = nn.Conv2d(feat, 1, 1)
        self.distractor_map = nn.Conv2d(feat, 1, 1)
        state_dim = feat + 2 + 3 + 2
        self.norm = nn.LayerNorm(state_dim)
        self.shared = nn.Sequential(nn.Linear(state_dim, 256), nn.GELU(), nn.Dropout(0.08))
        self.target_head = nn.Linear(256, 1)
        self.group_head = nn.Linear(256, n_group)
        self.group_c_head = nn.Linear(256, len(GROUP_C_NAMES))
        self.group_b_presence_head = nn.Linear(256, len(GROUP_B_NAMES))
        self.group_b_logarea_head = nn.Linear(256, len(GROUP_B_NAMES))
    def topk_pool(self, l):
        flat = l.flatten(1); k = max(1, int(round(flat.shape[1] * self.topk_frac)))
        return torch.topk(flat, k=k, dim=1).values.mean(dim=1, keepdim=True)
    def forward(self, x, bv, rg):
        f = self.encoder(x)
        pooled = F.adaptive_avg_pool2d(f, 1).flatten(1)
        tt = self.topk_pool(self.trigger_map(f)); dt = self.topk_pool(self.distractor_map(f))
        z = self.shared(self.norm(torch.cat([pooled, tt, dt, bv, rg], 1)))
        return dict(target_logit=self.target_head(z).squeeze(1), group_logits=self.group_head(z),
                    group_c_logits=self.group_c_head(z), group_b_presence=self.group_b_presence_head(z),
                    group_b_logarea=self.group_b_logarea_head(z))

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
sd = ck["model_state_dict"]
n_group = sd["group_head.weight"].shape[0]
model = AugSmartModel(ck["config"]["backbone"], n_group=n_group)
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f"ckpt load: missing={len(missing)} unexpected={len(unexpected)}")
assert len(missing) == 0, missing[:5]
model.eval()

PRES_COLS = [f"pred_b_{i:02d}_{n}_present" for i, n in enumerate(GROUP_B_NAMES)]
LOG_COLS = [f"pred_b_{i:02d}_{n}_logarea" for i, n in enumerate(GROUP_B_NAMES)]
C_COLS = [f"pred_c_{i:02d}_{n}" for i, n in enumerate(GROUP_C_NAMES)]

@torch.no_grad()
def run(ids, roi_lookup, img_dir, tag):
    ds = SPDataset(ids, roi_lookup, img_dir)
    dl = DataLoader(ds, batch_size=8, num_workers=4, shuffle=False)
    rows = []; t0 = time.time()
    for bi, (x, bv, rg, names) in enumerate(dl):
        out = model(x, bv.float(), rg.float())            # SINGLE PASS — no flip
        g = torch.sigmoid(out["group_logits"]).numpy()
        c = torch.sigmoid(out["group_c_logits"]).numpy()
        bp = torch.sigmoid(out["group_b_presence"]).numpy()
        bl = out["group_b_logarea"].numpy()
        tgt = torch.sigmoid(out["target_logit"]).numpy()
        for j, f in enumerate(names):
            r = {"image_id": f, "pred_target": float(tgt[j]),
                 "pred_group_c_any": float(g[j, 0]), "pred_group_b_above_any": float(g[j, 1]),
                 "pred_distractor_any": float(g[j, 2]), "pred_rare_group_c_any": float(g[j, 3])}
            for i2, n in enumerate(GROUP_C_NAMES): r[C_COLS[i2]] = float(c[j, i2])
            for i2, n in enumerate(GROUP_B_NAMES):
                r[PRES_COLS[i2]] = float(bp[j, i2]); r[LOG_COLS[i2]] = float(bl[j, i2])
            rows.append(r)
        if (bi + 1) % 50 == 0:
            done = (bi + 1) * 8
            print(f"  {tag} {done}/{len(ds)}  ({(time.time()-t0)/done*1000:.0f} ms/img)", flush=True)
    return pd.DataFrame(rows)

def soft_b_score(df, tau):
    pres = df[PRES_COLS].values; log_area = df[LOG_COLS].values
    return pres * (1.0 / (1.0 + np.exp(-(log_area - LOG_THRESHOLDS[None, :]) / max(1e-3, tau))))

def add_scores(df, tau):
    c_sup = np.maximum(df[C_COLS].max(1).values, df["pred_rare_group_c_any"].values)
    b_sup = soft_b_score(df, tau).max(1)
    df = df.copy()
    df["score_group_c"] = np.sqrt(np.clip(df["pred_group_c_any"].values * c_sup, 0, 1))
    df["score_group_b"] = np.sqrt(np.clip(df["pred_group_b_above_any"].values * b_sup, 0, 1))
    return df

def tune_global(df):
    y = df["target"].values.astype(int); best = None
    for tau in TAU_SWEEP:
        dfs = add_scores(df, tau)
        cv = dfs["score_group_c"].values; bvv = dfs["score_group_b"].values
        for tc in THR_SWEEP:
            cf = cv >= tc
            for tb in THR_SWEEP:
                p = (cf | (bvv >= tb)).astype(int)
                f1 = f1_score(y, p); pos = float(p.mean())
                k = (f1, -abs(pos - TRAIN_POS_RATE), precision_score(y, p, zero_division=0))
                if best is None or k > best["k"]:
                    best = dict(k=k, t_group_c=float(tc), t_group_b=float(tb), tau=float(tau), f1=float(f1), positive_rate=pos)
    best.pop("k"); return best

def tune_per_bucket(df):
    y = df["target"].values.astype(int)
    g = tune_global(df); tau_pb = {}
    for b in sorted(df["bucket"].unique()):
        sub = df[df["bucket"] == b]
        if len(sub) < 50: tau_pb[b] = g["tau"]; continue
        best_t, best_f1 = g["tau"], -1.0
        for tau in TAU_SWEEP:
            dfs = add_scores(sub, tau)
            p = ((dfs["score_group_c"].values >= g["t_group_c"]) | (dfs["score_group_b"].values >= g["t_group_b"])).astype(int)
            f1 = f1_score(sub["target"].values.astype(int), p)
            if f1 > best_f1: best_f1, best_t = f1, tau
        tau_pb[b] = best_t
    pred = np.zeros(len(df), dtype=int)
    for b, tau in tau_pb.items():
        sub = df[df["bucket"] == b]; dfs = add_scores(sub, tau)
        p = ((dfs["score_group_c"].values >= g["t_group_c"]) | (dfs["score_group_b"].values >= g["t_group_b"])).astype(int)
        pred[sub.index] = p
    return dict(t_group_c=g["t_group_c"], t_group_b=g["t_group_b"], tau_per_bucket=tau_pb, tau=g["tau"],
                f1=float(f1_score(y, pred)), positive_rate=float(pred.mean())), pred

def apply_calib(df, calib):
    pred = np.zeros(len(df), dtype=int)
    for b, tau in calib["tau_per_bucket"].items():
        m = df["bucket"].values == b
        if not m.any(): continue
        dfs = add_scores(df[m], tau)
        pred[np.where(m)[0]] = ((dfs["score_group_c"].values >= calib["t_group_c"]) |
                                (dfs["score_group_b"].values >= calib["t_group_b"])).astype(int)
    return pred

# ===== VAL =====
val_df = run(val_ref["image_id"].tolist(), val_roi, DATA / "train_images", "val")
val_df = val_df.merge(val_ref, on="image_id", how="left")
calib, val_pred = tune_per_bucket(val_df.reset_index(drop=True))
y = val_df["target"].values.astype(int)
gl = tune_global(val_df)
print("=" * 60)
print(f"SINGLE-PASS val F1 (global calib):     {gl['f1']:.5f}  pos={gl['positive_rate']:.4f}")
print(f"SINGLE-PASS val F1 (per-bucket calib): {calib['f1']:.5f}  pos={calib['positive_rate']:.4f}")
for b in ["small", "medium", "large"]:
    m = val_df["bucket"].values == b
    print(f"  bucket {b:8s} F1={f1_score(y[m], val_pred[m]):.4f}")
print(f"refs: v62-TTA val 0.96969 | v5 val 0.9635 (LB 0.9684) | GATE: >= 0.9680 to probe LB")
# Also: plain binary-head threshold (v5-style decision on pred_target)
best_bin = max(((f1_score(y, (val_df['pred_target'].values >= t).astype(int)), t)
                for t in np.arange(0.30, 0.96, 0.01)), key=lambda z: z[0])
print(f"plain target-head best: F1={best_bin[0]:.5f} @ thr={best_bin[1]:.2f}")
val_df["sp_rule_pred"] = val_pred
val_df.to_csv(OUT / "v62_sp_val.csv", index=False)
with open(OUT / "v62_sp_calib.json", "w") as f:
    json.dump(dict(calib=calib, global_=gl, bin_thr=float(best_bin[1]), bin_f1=float(best_bin[0])), f, indent=2, default=float)

# ===== TEST =====
test_df = run(sample_sub["image_id"].tolist(), test_roi, DATA / "test_images", "test")
test_df["bucket"] = [to_bucket(test_roi[f][2]) for f in test_df["image_id"]]
tpred = apply_calib(test_df.reset_index(drop=True), calib)
print(f"TEST pos_rate (single-pass per-bucket rule): {tpred.mean():.4f}  (band 0.55-0.62)")
test_df["sp_rule_pred"] = tpred
test_df.to_csv(OUT / "v62_sp_test.csv", index=False)
sub = sample_sub.copy(); sub["target"] = tpred.astype(int)
sub.to_csv(OUT / "v62_sp_submission.csv", index=False)
print("DONE — outputs in", OUT)
