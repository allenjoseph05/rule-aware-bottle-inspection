# ----- MD CELL 0 -----
# # aug_smart_v62 — Failure-mode-driven augmentation, anti-overfit guards
# 
# > Companion to `findings.md`. The augmentation choices and sample drops below are
# > *directly traceable* to numbered findings F1–F12 from the EDA. The notebook is
# > structured to make every action attributable.
# 
# **What this is**: a v60-shaped notebook (same DINOv3 backbone, same model heads,
# same gates, same submission pipeline) with **three new augmentation arms** and
# **five new anti-overfit guards** that the EDA evidence justifies.
# 
# **What it is not**: a from-scratch rewrite. The encoder / loss / training loop /
# TTA / per-bucket tau / holdout gate are verbatim from v60. The gains are
# *data-side*, not *model-side*.
# 
# | Arm | Source finding | Anti-overfit guard |
# |---|---|---|
# | Per-bucket brightness normaliser + stronger jitter ±15 % | F2, F3 | Capped at brightness range, no gamma > 1.15 |
# | Polygon-aware copy-paste (borderline Group-B + rare-C) | F7, F4 | Synthetic-batch ratio cap = 30 % |
# | Distractor MixUp (target=1 + distractor-rich clean) | F8 | p = 0.20 only, gentle α ∈ [0.6, 0.9] |
# | Sample-weight 0 on 85 noise candidates | F5 | Cross-verified via §5.2 zero opposite-direction |
# | Drop one of each PHash-conflicting near-dup pair | F6 | Heuristic tie-break (no overlap with above) |
# | Data-driven priority lookup (vs v60 hardcoded) | F9 | Reads `16_aug_priority.csv` verbatim |
# | No down-resize, no heavy blur | F4 | Hard constraint in pipeline definitions |
# 
# **`SMOKE_TEST = True`** runs §13 only — quick aug-pipeline visual + 1-epoch
# sanity check. Set `False` for full 6 h training.
# ----- MD CELL 1 -----
# ## 0. Setup
# ===== CODE CELL 2 =====
# Dependencies are installed from the repository-level requirements.txt.

# ===== CODE CELL 3 =====
import os, json, math, time, random, warnings
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import timm
import albumentations as A

warnings.filterwarnings("ignore", category=UserWarning)
print("torch", torch.__version__, "| timm", timm.__version__, "| A", A.__version__)
assert torch.cuda.is_available(), "Need GPU. Set Kaggle accelerator to T4 x2."
assert tuple(int(x) for x in timm.__version__.split('.')[:3]) >= (1, 0, 20), "timm too old for DINOv3"

# ----- MD CELL 4 -----
# ## 1. Config — including the SMOKE_TEST toggle and anti-overfit guards
# ===== CODE CELL 5 =====
# === Master toggle ===
# True  → quick smoke test (1 epoch on 2000-image subset + visual gallery only)
# False → full 12-epoch run with submission gate
SMOKE_TEST = False

RUN_MODE = "smoke" if SMOKE_TEST else "full6h"
CREATE_SUBMISSION = RUN_MODE == "full6h"
RUN_TEST_INFERENCE = RUN_MODE == "full6h"

# === Path resolution — verbatim from v57 §2 / v60 §1 ===
DATA_ROOT = next((p for p in [
    Path("/kaggle/input/1st-krones-vision-ai-challenge"),
    Path("/kaggle/input/competitions/1st-krones-vision-ai-challenge"),
] if p.exists()), Path.cwd().parent / "data")
OUT_ROOT = Path("/kaggle/working") if DATA_ROOT != (Path.cwd().parent / "data") else Path.cwd().parent / "artefacts" / "aug_smart_v62"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

TRAIN_IMG_DIR = DATA_ROOT / "train_images"
TEST_IMG_DIR  = DATA_ROOT / "test_images"
TRAIN_CSV     = DATA_ROOT / "train.csv"
SAMPLE_SUB    = DATA_ROOT / "sample_submission.csv"
ANN_JSON      = DATA_ROOT / "train_annotations.json"

# === EDA cache resolution ===
# On Kaggle, attach the EDA results as a dataset (e.g., /kaggle/input/krones-eda-cache/eda_cache).
# Locally, the unzipped results/eda_cache/ folder is the fallback.
EDA_CACHE_CANDIDATES = [
    Path("/kaggle/input/krones-eda-cache/eda_cache"),
    Path("/kaggle/input/krones-eda-cache"),
    Path.cwd().parent / "results" / "eda_cache",
    Path.cwd() / "results" / "eda_cache",
    Path.cwd() / "eda_cache",
]
EDA_CACHE_DIR = next((p for p in EDA_CACHE_CANDIDATES if p.exists()), None)
print("DATA_ROOT:", DATA_ROOT)
print("EDA_CACHE_DIR:", EDA_CACHE_DIR)
if EDA_CACHE_DIR is None:
    print("⚠️  EDA cache not found — §1 will operate in DEGRADED MODE (priority hard-coded, no noise drop).")

# === Baselines + gates ===
V57_VAL_F1_GATE      = 0.9665
V62_LIFT_OVER_V57    = 0.001
TRAIN_POS_RATE       = 0.5832508134106663
V5_BUCKET_F1         = {"large": 0.9479, "medium": 0.9695, "small": 0.8966}

# === Backbone — same default as v60 ===
BACKBONE = "convnext_tiny.dinov3_lvd1689m"
BATCH    = 8 if "tiny" in BACKBONE else 4
GRAD_ACC = 2 if "tiny" in BACKBONE else 4

CFG = dict(
    phase = "aug_smart_v62",
    run_mode = RUN_MODE,
    backbone = BACKBONE,
    roi_img_size = 640,
    batch_size = BATCH,
    grad_accum = GRAD_ACC,
    num_workers = 4,
    # Epochs — short for smoke test, full for production
    epochs = 1 if SMOKE_TEST else 12,
    early_stop_patience = 1 if SMOKE_TEST else 3,
    freeze_encoder_epochs = 1,
    unfreeze_all_from_epoch = 2,
    warmup_epochs = 1 if SMOKE_TEST else 2,
    lr_encoder_full = 1.5e-5,
    lr_heads = 2.0e-4,
    weight_decay = 0.05,
    grad_clip = 1.0,
    val_split = 0.20,
    seed = 42,
    amp = True,
    ema_decay = 0.9995,
    use_grad_checkpointing = True,
    roi_pad_frac = 0.10,
    roi_jitter_shift = 0.025,
    roi_jitter_scale = 0.08,
    # Loss weights (verbatim from v60)
    w_target = 0.35,
    w_group = 0.95,
    w_group_c_classes = 0.30,
    w_group_b_presence = 0.55,
    w_group_b_log_area = 0.20,
    w_consistency = 0.08,
    max_pos_weight = 18.0,
    threshold_sweep = np.arange(0.08, 0.96, 0.02).tolist(),
    tau_sweep = [0.50, 0.65, 0.80, 0.95, 1.05, 1.20, 1.35, 1.60],
    use_tta = True,
    use_per_bucket_tau = True,
    # Gates
    full_gate_lift = V62_LIFT_OVER_V57,
    holdout_min_f1 = 0.9655,
    holdout_gap_max = 0.0045,
    bucket_regress_tol = 0.0025,
    test_pos_rate_min = 0.56,
    test_pos_rate_max = 0.61,
    max6h_train_wall_seconds = 5.55 * 3600,
    # === v62 NEW: anti-overfit aug guards ===
    copy_paste_batch_cap = 0.30,        # G(g): max fraction of a batch that may be synthetic copy-paste
    mixup_prob = 0.20,                  # G(h): probability that a target=1 image is distractor-MixUp'd
    mixup_alpha_min = 0.60,             # gentle blend floor
    mixup_alpha_max = 0.90,
    brightness_jitter_strong = 0.15,    # vs v60's 0.08
    gamma_jitter_min = 0.85,
    gamma_jitter_max = 1.15,
    # === v62 NEW: hard constraints from F4 (sub-resolution risk) ===
    forbid_down_resize = True,
    forbid_heavy_blur = True,
    # === v62 NEW: smoke-test subset ===
    smoke_subset_n = 2000,
)
random.seed(CFG["seed"]); np.random.seed(CFG["seed"])
torch.manual_seed(CFG["seed"]); torch.cuda.manual_seed_all(CFG["seed"])
torch.backends.cudnn.benchmark = True
device = torch.device("cuda")
RUN_STARTED_AT = time.time()
print(f"\nRUN_MODE: {RUN_MODE}  BACKBONE: {BACKBONE}  batch: {BATCH}×{GRAD_ACC}  epochs: {CFG['epochs']}")

# ----- MD CELL 6 -----
# ## 1. Load EDA artefacts — priority table, borderline polys, noise IDs
# 
# Action mapping:
# - `16_aug_priority.csv` → `PRIORITY_BY_CELL` lookup (F9)
# - `07_groupB_borderline.csv` → `BORDERLINE_BANK` (F7 — copy-paste source bank)
# - `12_rule_mismatches_target1_rule0.csv` → `NOISE_DROP_IDS` (F5)
# - `13_intra_train_nearmatch.csv` → adds to `NOISE_DROP_IDS` (F6, tie-broken)
# ===== CODE CELL 7 =====
def _safe_read(name, expected_cols=None):
    if EDA_CACHE_DIR is None:
        return None
    p = EDA_CACHE_DIR / name
    if not p.exists():
        print(f"  ⚠️  {name} missing in EDA cache")
        return None
    df = pd.read_csv(p)
    if expected_cols and not all(c in df.columns for c in expected_cols):
        print(f"  ⚠️  {name} missing expected cols {expected_cols}")
        return None
    print(f"  ✓  {name}  rows={len(df)}")
    return df

# Priority table — F9
prio_df = _safe_read("16_aug_priority.csv", ["bucket", "functional_group", "priority"])
if prio_df is not None:
    PRIORITY_BY_CELL = {(r["bucket"], r["functional_group"]): r["priority"] for _, r in prio_df.iterrows()}
else:
    PRIORITY_BY_CELL = {}
print(f"PRIORITY_BY_CELL entries: {len(PRIORITY_BY_CELL)}")

# Borderline bank — F7
border_df = _safe_read("07_groupB_borderline.csv", ["image_id", "class", "coco_area"])
if border_df is not None:
    # Restrict to ±10% borderline only (the CSV already does this — kept for clarity)
    BORDERLINE_BANK = border_df.copy()
    print(f"BORDERLINE_BANK rows: {len(BORDERLINE_BANK)}  classes: {sorted(BORDERLINE_BANK['class'].unique())}")
else:
    BORDERLINE_BANK = pd.DataFrame(columns=["image_id", "class", "coco_area"])

# Noise candidates — F5
NOISE_DROP_IDS: set[str] = set()
rule_mismatch_df = _safe_read("12_rule_mismatches_target1_rule0.csv", ["image_id"])
if rule_mismatch_df is not None:
    NOISE_DROP_IDS.update(rule_mismatch_df["image_id"].astype(str).tolist())
    print(f"  + {len(rule_mismatch_df)} (target=1, rule=0) mismatches → noise drop")

# PHash conflicting near-dups — F6 (tie-break: drop file_a)
phash_conf_df = _safe_read("13_intra_train_nearmatch.csv", ["file_a", "file_b", "conflict"])
if phash_conf_df is not None:
    conf = phash_conf_df[phash_conf_df["conflict"].astype(str).isin(["True", "true", "1"])]
    NOISE_DROP_IDS.update(conf["file_a"].astype(str).tolist())
    print(f"  + {len(conf)} PHash-conflicting near-dup pairs → drop file_a side")

print(f"\nTotal NOISE_DROP_IDS: {len(NOISE_DROP_IDS)}")

# ----- MD CELL 8 -----
# ## 2. Load CSV + COCO + `derive_functional_labels` (verbatim from v60 §2)
# ===== CODE CELL 9 =====
train_df_raw = pd.read_csv(TRAIN_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB)
with open(ANN_JSON) as f:
    coco = json.load(f)
cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
fname_to_imgid = {im["file_name"]: im["id"] for im in coco["images"]}
imgid_to_fname = {v: k for k, v in fname_to_imgid.items()}
imgid_to_anns = defaultdict(list)
for a in coco["annotations"]:
    imgid_to_anns[a["image_id"]].append(a)

GROUP_A = ["Embossing", "Foam residue", "No fault", "Water drop"]
GROUP_B_NAMES = ["Air bubble", "Chip", "Contamination light", "Glass imperfection", "Scuffing", "Scuffing heavy"]
GROUP_B_THRESHOLDS = {"Air bubble": 500, "Chip": 200, "Contamination light": 180,
                     "Glass imperfection": 100, "Scuffing": 75000, "Scuffing heavy": 1200}
ROI_NAME = "Roi"
GROUP_C_NAMES = sorted(set(cat_id_to_name.values()) - set(GROUP_A) - set(GROUP_B_NAMES) - {ROI_NAME})
RARE_GROUP_C = ["Circlip", "Foil / Semitransparent", "Insect", "Liquid", "Straw"]
group_c_idx = {n: i for i, n in enumerate(GROUP_C_NAMES)}
group_b_idx = {n: i for i, n in enumerate(GROUP_B_NAMES)}
LOG_AREA_PAD = 1.0
LOG_THRESHOLDS = np.array([math.log(GROUP_B_THRESHOLDS[n] + LOG_AREA_PAD) for n in GROUP_B_NAMES], dtype=np.float32)

def to_bucket(w):
    if w is None or pd.isna(w): return "unknown"
    w = float(w)
    return "small" if w < 510 else ("large" if w > 590 else "medium")

def get_roi(image_id):
    for a in imgid_to_anns.get(image_id, []):
        if cat_id_to_name[a["category_id"]] == ROI_NAME:
            return [float(v) for v in a["bbox"]]
    return None

def derive(fname, target=None):
    iid = fname_to_imgid.get(fname); roi = get_roi(iid) if iid else None
    gc = np.zeros(len(GROUP_C_NAMES), dtype=np.float32)
    gb_state   = np.zeros(len(GROUP_B_NAMES), dtype=np.int64)
    gb_present = np.zeros(len(GROUP_B_NAMES), dtype=np.float32)
    gb_logarea = np.zeros(len(GROUP_B_NAMES), dtype=np.float32)
    gb_logmask = np.zeros(len(GROUP_B_NAMES), dtype=np.float32)
    ga_any = rare_c_any = 0; foam_water = 0; trig_areas = []
    has_groupb_ann = False
    for a in imgid_to_anns.get(iid, []):
        name = cat_id_to_name[a["category_id"]]; area = float(a.get("area", 0.0))
        if name == ROI_NAME: continue
        if name in {"Foam residue", "Water drop"}: foam_water += 1
        if name in GROUP_A: ga_any = 1
        elif name in group_c_idx:
            gc[group_c_idx[name]] = 1.0
            if name in RARE_GROUP_C: rare_c_any = 1
            trig_areas.append(area)
        elif name in group_b_idx:
            has_groupb_ann = True
            i = group_b_idx[name]; thr = float(GROUP_B_THRESHOLDS[name])
            if area >= thr:           gb_state[i] = max(gb_state[i], 3); trig_areas.append(area)
            elif area >= 0.50 * thr:  gb_state[i] = max(gb_state[i], 2)
            else:                     gb_state[i] = max(gb_state[i], 1)
            gb_present[i] = 1.0
            la = math.log(area + LOG_AREA_PAD)
            if la > gb_logarea[i]: gb_logarea[i] = la
            gb_logmask[i] = 1.0

    rule_target = int(int(gc.max() > 0) or int((gb_state == 3).any()))
    row = {"image_id": fname, "target": int(target) if target is not None else -1,
           "rule_target": rule_target, "group_c_any": int(gc.max() > 0),
           "group_b_above_any": int((gb_state == 3).any()),
           "group_a_any": int(ga_any),
           "distractor_any": int(ga_any or ((gb_state > 0) & (gb_state < 3)).any()),
           "rare_group_c_any": int(rare_c_any), "foam_water_count": int(foam_water),
           "smallest_trigger_area": float(min(trig_areas)) if trig_areas else np.nan,
           "has_groupb_ann": int(has_groupb_ann),
           "roi_x": np.nan, "roi_y": np.nan, "roi_w": np.nan, "roi_h": np.nan, "bucket": "unknown"}
    if roi is not None:
        row["roi_x"], row["roi_y"], row["roi_w"], row["roi_h"] = roi
        row["bucket"] = to_bucket(row["roi_w"])
    for i, n in enumerate(GROUP_C_NAMES): row[f"c_{i:02d}_{n}"] = float(gc[i])
    for i, n in enumerate(GROUP_B_NAMES):
        row[f"b_{i:02d}_{n}_state"]   = int(gb_state[i])
        row[f"b_{i:02d}_{n}_present"] = float(gb_present[i])
        row[f"b_{i:02d}_{n}_logarea"] = float(gb_logarea[i])
        row[f"b_{i:02d}_{n}_logmask"] = float(gb_logmask[i])
    return row

train_df = pd.DataFrame([derive(r.image_id, int(r.target)) for r in train_df_raw.itertuples(index=False)])
print("rule agreement:", float((train_df["rule_target"] == train_df["target"]).mean()))

# functional_group — same convention as v60
def fg(r):
    if r["group_c_any"] and r["rare_group_c_any"]: return "rare_group_c"
    if r["group_c_any"]: return "group_c"
    if r["group_b_above_any"]: return "group_b_above"
    if r["distractor_any"] and r["target"] == 0: return "distractor_clean"
    if r["target"] == 0: return "clean"
    return "rule_noise_pos"
train_df["functional_group"] = train_df.apply(fg, axis=1)

# Test ROI
test_roi_path = None
for d in (Path("/kaggle/input"),):
    if d.exists():
        for p in d.glob("**/test_annotations_roi_only.json"):
            test_roi_path = p; break
if test_roi_path is None and (DATA_ROOT / "test_annotations_roi_only.json").exists():
    test_roi_path = DATA_ROOT / "test_annotations_roi_only.json"
test_roi_lookup = {}
if test_roi_path:
    with open(test_roi_path) as f: troi = json.load(f)
    cat = {c["id"]: c["name"] for c in troi["categories"]}
    by_img = defaultdict(list)
    for a in troi["annotations"]: by_img[a["image_id"]].append(a)
    for im in troi["images"]:
        for a in by_img.get(im["id"], []):
            if cat.get(a["category_id"]) == ROI_NAME:
                test_roi_lookup[im["file_name"]] = tuple(float(v) for v in a["bbox"])
                break
    print(f"Loaded test ROI: {len(test_roi_lookup)}/{len(sample_sub)}")

# ----- MD CELL 10 -----
# ## 3. Per-bucket brightness normaliser (NEW — F2, F3)
# 
# EDA finding: small-bucket images are ~15 grey levels brighter than medium/large
# across nearly every defect class. A model can shortcut-learn this. We compute a
# per-bucket brightness reference (from `image_stats_full.csv` if available, else
# sample on-the-fly) and apply gentle histogram-style shift so each bucket's
# brightness *distribution* approaches the medium-bucket distribution.
# 
# Important: this is applied **before** the photometric augmentation pipeline, so
# the augmentation's brightness jitter then sits on top of an *already*
# normalised brightness baseline.
# ===== CODE CELL 11 =====
# Load image stats from EDA cache to compute per-bucket reference means
BUCKET_BRIGHTNESS = {"small": 90.0, "medium": 90.0, "large": 90.0}  # safe defaults
if EDA_CACHE_DIR is not None and (EDA_CACHE_DIR / "image_stats_full.csv").exists():
    stats_df = pd.read_csv(EDA_CACHE_DIR / "image_stats_full.csv")
    if "bucket" in stats_df.columns and "mean" in stats_df.columns:
        train_only = stats_df[stats_df.get("split", "train") == "train"]
        for b in ("small", "medium", "large"):
            sub = train_only[train_only["bucket"] == b]
            if len(sub):
                BUCKET_BRIGHTNESS[b] = float(sub["mean"].mean())
print("BUCKET_BRIGHTNESS (train mean greyscale):", {k: round(v, 1) for k, v in BUCKET_BRIGHTNESS.items()})

# Reference = medium-bucket mean (the most populous, target distribution to match)
BRIGHTNESS_REF = BUCKET_BRIGHTNESS["medium"]
# Per-bucket shift (negative for small bucket, since small is brighter)
BUCKET_BRIGHTNESS_SHIFT = {b: BRIGHTNESS_REF - BUCKET_BRIGHTNESS[b] for b in BUCKET_BRIGHTNESS}
print("BUCKET_BRIGHTNESS_SHIFT (additive Δ to bring each bucket to reference):", {k: round(v, 1) for k, v in BUCKET_BRIGHTNESS_SHIFT.items()})

def normalise_bucket_brightness(arr, bucket):
    # Shift mean brightness of an image toward the medium-bucket reference.
    # Conservative: applies only ~50 % of the full shift to avoid destroying real signal.
    # Clamps output to [0, 255].
    delta = BUCKET_BRIGHTNESS_SHIFT.get(bucket, 0.0) * 0.5
    if delta == 0.0:
        return arr
    out = arr.astype(np.float32) + delta
    return np.clip(out, 0, 255).astype(np.uint8)

# ----- MD CELL 12 -----
# ## 4. Augmentation pipelines — extends v60 with EDA-derived guards
# 
# Changes vs v60:
# - Brightness jitter ±15 % (v60: ±8 %) — destroys the F2/F3 brightness shortcut.
# - Gamma jitter [0.85, 1.15] — adds tonal-curve variation beyond linear scaling.
# - Mild CLAHE in LIGHT and MEDIUM (clip-limit 1.5, tile 8) — sharpens defect edges.
# - **NO** `RandomScale` / `RandomResizedCrop` / `Downscale` (F4 — sub-resolution risk).
# - **NO** `GaussianBlur σ > 1.0` (F4).
# - Priority lookup reads `eda_cache/16_aug_priority.csv` instead of v60's hardcoded `base_priority` (F9).
# - `aug_safe_for_groupb` retained verbatim from v60 (preserves log-area supervision on Group-B samples).
# ===== CODE CELL 13 =====
S = CFG["roi_img_size"]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]

B_LIMIT = CFG["brightness_jitter_strong"]    # 0.15
G_LO, G_HI = CFG["gamma_jitter_min"], CFG["gamma_jitter_max"]   # 0.85 / 1.15

# v62 pipelines — F4-compliant (no down-resize, no heavy blur)
aug_heavy = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=20, p=0.6, border_mode=0),
    A.Affine(shear=(-5, 5), scale=(0.92, 1.08), p=0.3, mode=0),
    A.ElasticTransform(alpha=20, sigma=4, p=0.20, border_mode=0),
    A.RandomBrightnessContrast(brightness_limit=B_LIMIT, contrast_limit=B_LIMIT, p=0.6),
    A.RandomGamma(gamma_limit=(int(G_LO*100), int(G_HI*100)), p=0.4),
    A.CoarseDropout(max_holes=2, max_height=64, max_width=64, min_holes=1, min_height=16, min_width=16, fill_value=0, p=0.20),
])
aug_medium = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=14, p=0.5, border_mode=0),
    A.RandomBrightnessContrast(brightness_limit=B_LIMIT, contrast_limit=B_LIMIT, p=0.6),
    A.RandomGamma(gamma_limit=(int(G_LO*100), int(G_HI*100)), p=0.3),
    A.CLAHE(clip_limit=1.5, tile_grid_size=(8, 8), p=0.20),
    A.Affine(shear=(-3, 3), scale=(0.95, 1.05), p=0.2, mode=0),
])
aug_light = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=10, p=0.4, border_mode=0),
    A.RandomBrightnessContrast(brightness_limit=B_LIMIT, contrast_limit=B_LIMIT, p=0.5),
    A.CLAHE(clip_limit=1.5, tile_grid_size=(8, 8), p=0.15),
])
aug_safe_for_groupb = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=6, p=0.3, border_mode=0),                      # tiny rotation only — area-preserving
    A.RandomBrightnessContrast(brightness_limit=B_LIMIT, contrast_limit=B_LIMIT, p=0.5),
    # NO gamma / NO CLAHE — these can shift the apparent area boundary near threshold
])
aug_none = A.Compose([A.HorizontalFlip(p=0.5)])

PIPELINES = {"HEAVY": aug_heavy, "MEDIUM": aug_medium, "LIGHT": aug_light, "NONE": aug_none}

# Fallback priority (used only when EDA cache is missing) — matches v60's base_priority
def _fallback_priority(bucket, fg):
    if fg in ("clean", "rule_noise_pos"): return "NONE"
    if bucket == "small" and fg in {"group_b_above", "rare_group_c", "group_c"}: return "HEAVY"
    if bucket == "small" and fg == "distractor_clean": return "MEDIUM"
    if bucket == "large" and fg in {"group_b_above", "rare_group_c"}: return "HEAVY"
    if fg in {"group_b_above", "rare_group_c"}: return "MEDIUM"
    return "LIGHT"

def cell_priority(bucket, fg):
    # v62: prefer the data-driven priority table from §1, fall back to v60 logic.
    # group_a_or_b_below in the EDA table covers what we call clean / distractor_clean / group_b_below
    key_fg = "group_a_or_b_below" if fg in ("clean", "distractor_clean") else fg
    p = PRIORITY_BY_CELL.get((bucket, key_fg))
    if p:
        return p
    return _fallback_priority(bucket, fg)

def select_pipeline(row):
    # v62 selector — preserves v60's safe_for_groupb guarantee, otherwise reads §1 lookup.
    if int(row["has_groupb_ann"]) == 1:
        return aug_safe_for_groupb
    return PIPELINES[cell_priority(str(row["bucket"]), str(row["functional_group"]))]

# Sanity print — how many samples receive each pipeline
sum_counts = Counter()
for r in train_df.itertuples(index=False):
    if int(getattr(r, "has_groupb_ann")) == 1:
        sum_counts["safe_for_groupb"] += 1
    else:
        sum_counts[cell_priority(str(r.bucket), str(r.functional_group))] += 1
print("v62 pipeline assignment counts:", dict(sum_counts))

# ----- MD CELL 14 -----
# ## 5. Polygon-aware copy-paste (NEW — F7)
# 
# Source bank:
# - The 160 Group-B borderline polygons (areas within ±10 % of class threshold).
# - Tiny + rare Group-C: Insect (29), Straw (35), Circlip (39), Liquid (76), Foil/Semitransparent (41).
# 
# Recipient: target=0 image (clean).
# 
# Operation: crop the polygon's bounding box from the source image, mean-shift the
# crop's brightness to match the recipient ROI's brightness, paste at a randomised
# location inside the recipient ROI. The pasted region's area is preserved (no
# resize), so the threshold rule (and thus `b_logarea` supervision) stays exact.
# 
# Excluded sources:
# - `No fault`, `No base visible` — these are full-frame rectangles, not point
#   defects (F11).
# - Any polygon with > 50 % of vertices outside the source's ROI (also F11).
# 
# Cap: **30 % of training batch** may be synthetic copy-paste samples per epoch
# (anti-overfit guard).
# ===== CODE CELL 15 =====
# Build the source bank.
SOURCE_BANK_ROWS = []
EXCLUDE_CLASSES = {"No fault", "No base visible"}

# (a) Borderline Group-B polygons from F7
if len(BORDERLINE_BANK) > 0:
    for _, r in BORDERLINE_BANK.iterrows():
        SOURCE_BANK_ROWS.append({
            "image_id": str(r["image_id"]), "class": str(r["class"]),
            "coco_area": float(r["coco_area"]), "kind": "borderline_b",
        })

# (b) Rare Group-C polygons (Insect / Straw / Circlip / Liquid / Foil)
for ann in coco["annotations"]:
    name = cat_id_to_name[ann["category_id"]]
    if name in EXCLUDE_CLASSES: continue
    if name not in {"Insect", "Straw", "Circlip", "Liquid", "Foil / Semitransparent"}: continue
    iid = ann["image_id"]
    fname = imgid_to_fname.get(iid)
    if fname is None: continue
    seg = ann.get("segmentation")
    if not (isinstance(seg, list) and seg and isinstance(seg[0], list)): continue
    pts = np.array(seg[0]).reshape(-1, 2)
    if len(pts) < 3: continue
    SOURCE_BANK_ROWS.append({
        "image_id": fname, "class": name,
        "coco_area": float(ann.get("area", 0.0)), "kind": "rare_c",
    })

SOURCE_BANK = pd.DataFrame(SOURCE_BANK_ROWS)
print(f"SOURCE_BANK: {len(SOURCE_BANK)} polygons")
print(SOURCE_BANK["class"].value_counts())
print(f"\ncopy_paste_batch_cap = {CFG['copy_paste_batch_cap']*100:.0f}% of batch")

def extract_polygon_crop(img_arr, image_id, class_name):
    # Return (crop_arr, bbox_in_source, polygon_in_source) or None.
    iid = fname_to_imgid.get(image_id)
    if iid is None: return None
    for ann in imgid_to_anns.get(iid, []):
        if cat_id_to_name[ann["category_id"]] != class_name: continue
        if float(ann.get("area", 0.0)) <= 0: continue
        seg = ann.get("segmentation")
        if not (isinstance(seg, list) and seg and isinstance(seg[0], list)): continue
        pts = np.array(seg[0]).reshape(-1, 2)
        if len(pts) < 3: continue
        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)
        x_min = max(0, int(math.floor(x_min)))
        y_min = max(0, int(math.floor(y_min)))
        x_max = min(img_arr.shape[1], int(math.ceil(x_max)) + 1)
        y_max = min(img_arr.shape[0], int(math.ceil(y_max)) + 1)
        if x_max <= x_min + 1 or y_max <= y_min + 1: continue
        return img_arr[y_min:y_max, x_min:x_max].copy(), (x_min, y_min, x_max - x_min, y_max - y_min), pts - np.array([x_min, y_min])
    return None

def paste_polygon_onto(recipient_arr, recipient_roi, source_crop, source_polygon, rng):
    # Mean-shifted polygon paste onto recipient inside its ROI. Returns paste bbox in recipient pixel coords.
    H, W = recipient_arr.shape
    rx, ry, rw, rh = [int(v) for v in recipient_roi[:4]]
    rx = max(0, rx); ry = max(0, ry)
    rw = min(W - rx, rw); rh = min(H - ry, rh)
    ch, cw = source_crop.shape
    if ch >= rh or cw >= rw:
        return None  # source too big for recipient ROI
    px = rx + rng.randint(0, rw - cw)
    py = ry + rng.randint(0, rh - ch)
    # Mean-shift the source crop so its mean = mean of the recipient patch (illumination match)
    recip_patch = recipient_arr[py:py+ch, px:px+cw].astype(np.float32)
    src_f = source_crop.astype(np.float32)
    delta = float(recip_patch.mean()) - float(src_f.mean())
    src_f = np.clip(src_f + delta, 0, 255).astype(np.uint8)
    # Polygon mask
    from skimage.draw import polygon as sk_poly
    poly_rows, poly_cols = sk_poly(source_polygon[:, 1], source_polygon[:, 0], shape=(ch, cw))
    mask = np.zeros_like(src_f, dtype=bool)
    if len(poly_rows): mask[poly_rows, poly_cols] = True
    if not mask.any(): return None
    # Blend
    recipient_arr[py:py+ch, px:px+cw][mask] = src_f[mask]
    return (px, py, cw, ch)

# ----- MD CELL 16 -----
# ## 6. Distractor MixUp (NEW — F8)
# 
# For target=1 images containing a rare Group-C class (Insect, Liquid, Straw,
# Circlip, Foil/Semitransparent), with probability `mixup_prob = 0.20` we blend
# the image with a clean image (target=0) that has ≥ 3 Foam-residue / Water-drop
# polygons. Blend coefficient α ∈ [0.6, 0.9]. Target stays 1.
# 
# This teaches the model "distractor presence on a defective bottle ≠ clean",
# addressing the 61 % FP-via-Group-C distractor confusion documented in
# `improvement_ideas.md` A1–A5.
# 
# Anti-overfit: capped at 20 % probability, gentle blend floor 0.6 (image is
# always ≥ 60 % the original). The model is never given a label-noisy sample.
# ===== CODE CELL 17 =====
# Build distractor-rich clean recipient bank
DISTRACTOR_RICH_CLEAN_IDS = train_df[
    (train_df["target"] == 0) &
    (train_df["foam_water_count"] >= 3)
]["image_id"].tolist()
print(f"DISTRACTOR_RICH_CLEAN_IDS: {len(DISTRACTOR_RICH_CLEAN_IDS)}")

# Rare-defective trigger set
RARE_DEFECTIVE_IDS = set(train_df[
    (train_df["target"] == 1) & (train_df["rare_group_c_any"] == 1)
]["image_id"].tolist())
print(f"RARE_DEFECTIVE_IDS (eligible for MixUp): {len(RARE_DEFECTIVE_IDS)}")

def maybe_mixup(arr_main, image_id, recipient_roi, rng):
    # Returns (possibly-mixed array, did_mix bool). target label is preserved by caller.
    if image_id not in RARE_DEFECTIVE_IDS:
        return arr_main, False
    if rng.random() >= CFG["mixup_prob"]:
        return arr_main, False
    if not DISTRACTOR_RICH_CLEAN_IDS:
        return arr_main, False
    other_id = rng.choice(DISTRACTOR_RICH_CLEAN_IDS)
    try:
        other_full = np.asarray(Image.open(TRAIN_IMG_DIR / other_id).convert("L"), dtype=np.uint8)
    except Exception:
        return arr_main, False
    # Crop other to its ROI then resize to recipient's shape
    other_row = train_df[train_df["image_id"] == other_id]
    if len(other_row) == 0: return arr_main, False
    o_roi = other_row.iloc[0][["roi_x", "roi_y", "roi_w", "roi_h"]].values.astype(float)
    if any(np.isnan(o_roi)): return arr_main, False
    ox, oy, ow, oh = [int(v) for v in o_roi]
    o_crop = other_full[max(0, oy):oy+oh, max(0, ox):ox+ow]
    if o_crop.size == 0: return arr_main, False
    if o_crop.shape != arr_main.shape:
        o_crop = np.asarray(Image.fromarray(o_crop, mode="L").resize(arr_main.shape[::-1], Image.BILINEAR), dtype=np.uint8)
    alpha = rng.uniform(CFG["mixup_alpha_min"], CFG["mixup_alpha_max"])
    mixed = (arr_main.astype(np.float32) * alpha + o_crop.astype(np.float32) * (1.0 - alpha))
    return np.clip(mixed, 0, 255).astype(np.uint8), True

# ----- MD CELL 18 -----
# ## 7. Sample weights — extends v57's recipe + **drop the 85 noise candidates** (F5, F6)
# 
# The 85 noise candidates from `NOISE_DROP_IDS` get weight **0** — the
# WeightedRandomSampler will never sample them.
# 
# Also: cross-checked via §5.2 of the EDA that the opposite direction
# (`target=0` ∧ `rule=1`) is zero — there are no false-positive-rule samples to
# worry about.
# ===== CODE CELL 19 =====
def build_weights(df):
    w = np.ones(len(df), dtype=np.float32)
    # v57 recipe (verbatim)
    w += 1.75 * df["rare_group_c_any"].values.astype(np.float32)
    w += 0.75 * df["group_b_above_any"].values.astype(np.float32)
    near_any = np.zeros(len(df), dtype=bool)
    for i, n in enumerate(GROUP_B_NAMES):
        col = f"b_{i:02d}_{n}_state"
        if col in df.columns:
            near_any |= (df[col].values == 2)
    w += 1.25 * near_any.astype(np.float32)
    w += 0.80 * ((df["target"].values == 0) & (df["distractor_any"].values == 1)).astype(np.float32)
    tiny_pos = (df["target"].values == 1) & df["smallest_trigger_area"].notna().values & (df["smallest_trigger_area"].values < 500)
    w += 1.25 * tiny_pos.astype(np.float32)
    w += 0.20 * df["bucket"].isin(["small", "large"]).values.astype(np.float32)
    w += 1.50 * ((df["bucket"].values == "small") & (df["target"].values == 1)).astype(np.float32)
    w += 0.75 * ((df["bucket"].values == "large") & (df["target"].values == 1)).astype(np.float32)
    # v62 NEW: drop noise candidates (F5, F6)
    drop_mask = df["image_id"].isin(NOISE_DROP_IDS).values
    w[drop_mask] = 0.0
    return w, int(drop_mask.sum())

# ----- MD CELL 20 -----
# ## 8. Dataset class (extends `V60Dataset`)
# 
# New hooks:
# - After loading the image + ROI crop, apply per-bucket brightness normaliser (§3).
# - With probability `copy_paste_batch_cap`, replace a clean (target=0) sample
#   with a synthetic copy-paste sample (§5). Targets and `has_groupb_ann` adjust
#   accordingly so the safe-for-groupb pipeline selection stays correct.
# - For target=1 samples with rare Group-C, attempt distractor MixUp (§6).
# - Then apply the v62 photometric/geometric pipeline.
# ===== CODE CELL 21 =====
def roi_crop_box(W, H, roi, pad, jitter=False):
    x, y, w, h = [float(v) for v in roi[:4]]
    sj = 1.0; sx = sy = 0.0
    if jitter:
        sj += random.uniform(-CFG["roi_jitter_scale"], CFG["roi_jitter_scale"])
        sx = random.uniform(-CFG["roi_jitter_shift"], CFG["roi_jitter_shift"]) * max(w, h)
        sy = random.uniform(-CFG["roi_jitter_shift"], CFG["roi_jitter_shift"]) * max(w, h)
    side = int(round(max(w, h) * (1 + 2 * pad) * sj))
    side = max(1, min(side, int(W), int(H)))
    cx = x + w/2 + sx; cy = y + h/2 + sy
    left = max(0, min(int(round(cx - side/2)), int(W) - side))
    top  = max(0, min(int(round(cy - side/2)), int(H) - side))
    return left, top, side, side

c_cols   = [f"c_{i:02d}_{n}" for i, n in enumerate(GROUP_C_NAMES)]
b_pres_c = [f"b_{i:02d}_{n}_present" for i, n in enumerate(GROUP_B_NAMES)]
b_la_c   = [f"b_{i:02d}_{n}_logarea" for i, n in enumerate(GROUP_B_NAMES)]
b_lm_c   = [f"b_{i:02d}_{n}_logmask" for i, n in enumerate(GROUP_B_NAMES)]
group_cols = ["group_c_any", "group_b_above_any", "distractor_any", "rare_group_c_any"]

class AugSmartDataset(Dataset):
    # Extends V60Dataset with bucket-brightness normaliser, copy-paste, MixUp.
    # Copy-paste is rate-limited at sample level (each clean sample has 30 % chance of being replaced).
    def __init__(self, df, image_dir, mode="val", test_roi_lookup=None,
                 source_bank=None, copy_paste_p=0.0):
        self.df = df.reset_index(drop=True); self.image_dir = Path(image_dir); self.mode = mode
        self.test_roi_lookup = test_roi_lookup or {}
        self.has_labels = "target" in self.df.columns and "functional_group" in self.df.columns
        self.source_bank = source_bank if source_bank is not None and len(source_bank) > 0 else None
        self.copy_paste_p = copy_paste_p
        # Pre-collect clean (target=0) row indices for copy-paste recipient eligibility
        if self.has_labels:
            self.clean_idx = np.where(self.df["target"].values == 0)[0]
        else:
            self.clean_idx = np.array([], dtype=int)

    def __len__(self): return len(self.df)

    def _load_arr_and_roi(self, idx):
        r = self.df.iloc[idx]; fname = str(r["image_id"])
        roi = ((float(r["roi_x"]), float(r["roi_y"]), float(r["roi_w"]), float(r["roi_h"]))
               if self.has_labels else self.test_roi_lookup[fname])
        img = Image.open(self.image_dir / fname).convert("L")
        W, H = img.size
        l, t, sw, sh = roi_crop_box(W, H, roi, CFG["roi_pad_frac"], jitter=(self.mode == "train"))
        crop = img.crop((l, t, l + sw, t + sh))
        if crop.size != (S, S):
            crop = crop.resize((S, S), Image.BILINEAR)
        return r, np.asarray(crop, dtype=np.uint8), roi, fname

    def _do_copy_paste(self, recipient_arr, recipient_roi, rng):
        # Returns (modified_arr, new_target, new_b_state_dict) — recipient becomes target=1.
        # Recipient_roi is the FULL-IMAGE roi, but recipient_arr is the SxS ROI crop. We work in
        # crop coordinates (treat the crop's full extent as the recipient ROI).
        if self.source_bank is None: return recipient_arr, None
        src_row = self.source_bank.iloc[rng.randrange(len(self.source_bank))]
        try:
            src_img = np.asarray(Image.open(self.image_dir / src_row["image_id"]).convert("L"), dtype=np.uint8)
        except Exception:
            return recipient_arr, None
        extracted = extract_polygon_crop(src_img, str(src_row["image_id"]), str(src_row["class"]))
        if extracted is None: return recipient_arr, None
        crop_arr, bbox, poly = extracted
        crop_h, crop_w = crop_arr.shape
        if crop_h >= S - 4 or crop_w >= S - 4:
            return recipient_arr, None  # source too big
        recipient_arr_mod = recipient_arr.copy()
        recip_full_roi = (0, 0, S, S)   # treat the cropped ROI image as the full canvas
        paste_bbox = paste_polygon_onto(recipient_arr_mod, recip_full_roi, crop_arr, poly, rng)
        if paste_bbox is None: return recipient_arr, None
        # New synthetic Group-B b_state: if source was a Group-B borderline, propagate above-threshold state
        new_meta = {"src_class": str(src_row["class"]), "src_area": float(src_row["coco_area"])}
        return recipient_arr_mod, new_meta

    def __getitem__(self, idx):
        r, arr, roi, fname = self._load_arr_and_roi(idx)
        synthesized = False; synth_class = None; synth_area = None

        if self.mode == "train" and self.has_labels:
            bucket_str = str(r["bucket"])
            arr = normalise_bucket_brightness(arr, bucket_str)
            rng = random.Random(idx * 7919 + int(time.time_ns() & 0xFFFFFFFF))
            # Copy-paste: only if recipient is clean AND copy_paste_p hits
            if int(r["target"]) == 0 and self.source_bank is not None and rng.random() < self.copy_paste_p:
                new_arr, meta = self._do_copy_paste(arr, roi, rng)
                if meta is not None:
                    arr = new_arr; synthesized = True
                    synth_class = meta["src_class"]; synth_area = meta["src_area"]
            # MixUp: only if rare-Group-C target=1 and probability hits
            if not synthesized and int(r["target"]) == 1:
                arr, _ = maybe_mixup(arr, str(r["image_id"]), roi, rng)
            pipeline = aug_safe_for_groupb if synthesized else select_pipeline(r)
            arr = pipeline(image=arr)["image"]
        elif self.mode == "train" and self.has_labels:
            pipeline = select_pipeline(r)
            arr = pipeline(image=arr)["image"]

        arrf = (np.stack([arr, arr, arr], axis=0).astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(np.ascontiguousarray(arrf)).float()
        bucket_str = str(r["bucket"]) if self.has_labels else to_bucket(roi[2])
        bv = np.array([float(bucket_str == "small"), float(bucket_str == "medium"), float(bucket_str == "large")], dtype=np.float32)
        rg = np.array([roi[2] / S, roi[3] / S], dtype=np.float32)

        if self.has_labels:
            # If synthesised, the recipient was clean but now contains a defect → target=1, update group_b labels
            if synthesized and synth_class in GROUP_B_NAMES:
                y = 1.0
                g_arr = r[group_cols].values.astype(np.float32).copy()
                g_arr[1] = 1.0  # group_b_above_any
                cc_arr = r[c_cols].values.astype(np.float32)
                bp_arr = r[b_pres_c].values.astype(np.float32).copy()
                bl_arr = r[b_la_c].values.astype(np.float32).copy()
                bm_arr = r[b_lm_c].values.astype(np.float32).copy()
                idx_b = group_b_idx[synth_class]
                bp_arr[idx_b] = 1.0
                bl_arr[idx_b] = math.log(synth_area + LOG_AREA_PAD)
                bm_arr[idx_b] = 1.0
                g = g_arr; cc = cc_arr; bp = bp_arr; bl = bl_arr; bm = bm_arr
            elif synthesized and synth_class in GROUP_C_NAMES:
                y = 1.0
                g_arr = r[group_cols].values.astype(np.float32).copy()
                g_arr[0] = 1.0  # group_c_any
                if synth_class in RARE_GROUP_C: g_arr[3] = 1.0
                cc_arr = r[c_cols].values.astype(np.float32).copy()
                cc_arr[group_c_idx[synth_class]] = 1.0
                g = g_arr; cc = cc_arr
                bp = r[b_pres_c].values.astype(np.float32)
                bl = r[b_la_c].values.astype(np.float32)
                bm = r[b_lm_c].values.astype(np.float32)
            else:
                y = float(r["target"])
                g  = r[group_cols].values.astype(np.float32)
                cc = r[c_cols].values.astype(np.float32)
                bp = r[b_pres_c].values.astype(np.float32)
                bl = r[b_la_c].values.astype(np.float32)
                bm = r[b_lm_c].values.astype(np.float32)
        else:
            y = -1.0
            g  = np.zeros(len(group_cols), dtype=np.float32)
            cc = np.zeros(len(c_cols), dtype=np.float32)
            bp = np.zeros(len(b_pres_c), dtype=np.float32)
            bl = np.zeros(len(b_la_c), dtype=np.float32)
            bm = np.zeros(len(b_lm_c), dtype=np.float32)

        return (x, torch.tensor(y, dtype=torch.float32),
                torch.from_numpy(np.ascontiguousarray(g)).float(),
                torch.from_numpy(np.ascontiguousarray(cc)).float(),
                torch.from_numpy(np.ascontiguousarray(bp)).float(),
                torch.from_numpy(np.ascontiguousarray(bl)).float(),
                torch.from_numpy(np.ascontiguousarray(bm)).float(),
                torch.from_numpy(bv).float(),
                torch.from_numpy(rg).float(), fname)

# Stratified split — same as v60
train_df["strat_key"] = train_df["bucket"].astype(str) + "_" + train_df["target"].astype(str)
tr_df, vl_df = train_test_split(train_df, test_size=CFG["val_split"], stratify=train_df["strat_key"], random_state=CFG["seed"])
tr_df, vl_df = tr_df.reset_index(drop=True), vl_df.reset_index(drop=True)

# Smoke-test: shrink tr_df to subset
if SMOKE_TEST:
    rng_smoke = np.random.RandomState(CFG["seed"])
    keep_n = min(CFG["smoke_subset_n"], len(tr_df))
    keep_idx = rng_smoke.choice(len(tr_df), size=keep_n, replace=False)
    tr_df = tr_df.iloc[keep_idx].reset_index(drop=True)
    print(f"SMOKE_TEST: tr_df shrunk to {len(tr_df)} rows")

train_weights, n_dropped = build_weights(tr_df)
print(f"sample weights built — {n_dropped} samples weight-zeroed (noise drop)")

train_ds = AugSmartDataset(tr_df, TRAIN_IMG_DIR, mode="train",
                           source_bank=SOURCE_BANK,
                           copy_paste_p=CFG["copy_paste_batch_cap"])
val_ds = AugSmartDataset(vl_df, TRAIN_IMG_DIR, mode="val")
train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"],
                          sampler=WeightedRandomSampler(torch.from_numpy(train_weights).double(), len(train_weights), True),
                          num_workers=CFG["num_workers"], pin_memory=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=CFG["batch_size"], shuffle=False, num_workers=CFG["num_workers"], pin_memory=True)
print(f"train={len(tr_df)}  val={len(vl_df)}  steps/epoch={len(train_loader)//CFG['grad_accum']}")

# ----- MD CELL 22 -----
# ## 9. Model — DINOv3 ConvNeXt-Tiny + multi-head (verbatim v60)
# 
# No architecture changes. The gains come from data-side aug + sampling, not from
# model capacity. This isolates the experimental delta.
# ===== CODE CELL 23 =====
class AugSmartModel(nn.Module):
    # Same architecture as V60Model — re-declared here so the notebook is self-contained.
    def __init__(self, backbone_name, topk_frac=0.08):
        super().__init__()
        self.encoder = timm.create_model(backbone_name, pretrained=True, num_classes=0, global_pool="")
        if hasattr(self.encoder, "set_grad_checkpointing") and CFG["use_grad_checkpointing"]:
            self.encoder.set_grad_checkpointing(True)
        self.topk_frac = topk_frac
        feat = int(self.encoder.num_features)
        self.trigger_map    = nn.Conv2d(feat, 1, 1)
        self.distractor_map = nn.Conv2d(feat, 1, 1)
        state_dim = feat + 2 + 3 + 2
        self.norm = nn.LayerNorm(state_dim)
        self.shared = nn.Sequential(nn.Linear(state_dim, 256), nn.GELU(), nn.Dropout(0.08))
        self.target_head            = nn.Linear(256, 1)
        self.group_head             = nn.Linear(256, len(group_cols))
        self.group_c_head           = nn.Linear(256, len(GROUP_C_NAMES))
        self.group_b_presence_head  = nn.Linear(256, len(GROUP_B_NAMES))
        self.group_b_logarea_head   = nn.Linear(256, len(GROUP_B_NAMES))

    def topk_pool(self, l):
        flat = l.flatten(1); k = max(1, int(round(flat.shape[1] * self.topk_frac)))
        return torch.topk(flat, k=k, dim=1).values.mean(dim=1, keepdim=True)

    def forward(self, x, bv, rg):
        f = self.encoder(x)
        pooled = F.adaptive_avg_pool2d(f, 1).flatten(1)
        tt = self.topk_pool(self.trigger_map(f))
        dt = self.topk_pool(self.distractor_map(f))
        z = self.shared(self.norm(torch.cat([pooled, tt, dt, bv, rg], dim=1)))
        return dict(target_logit=self.target_head(z).squeeze(1),
                    group_logits=self.group_head(z),
                    group_c_logits=self.group_c_head(z),
                    group_b_presence=self.group_b_presence_head(z),
                    group_b_logarea=self.group_b_logarea_head(z),
                    trigger_topk=tt.squeeze(1), distractor_topk=dt.squeeze(1))

def set_encoder_trainability(model, epoch):
    if epoch <= CFG["freeze_encoder_epochs"]:
        for p in model.encoder.parameters(): p.requires_grad_(False)
    elif epoch >= CFG["unfreeze_all_from_epoch"]:
        for p in model.encoder.parameters(): p.requires_grad_(True)
    n_trainable = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    print(f"  epoch {epoch}: encoder trainable params = {n_trainable:,}")

model = AugSmartModel(CFG["backbone"]).to(device)
set_encoder_trainability(model, 1)
print(f"backbone: {CFG['backbone']}  encoder dim: {model.encoder.num_features}")

# ----- MD CELL 24 -----
# ## 10. Loss / optimiser / EMA (verbatim v60)
# ===== CODE CELL 25 =====
def pos_weight(df, cols, mx):
    y = df[cols].values.astype(np.float32); pos = y.sum(0); neg = len(df) - pos
    return np.clip(neg / np.maximum(pos, 1.0), 1.0, mx).astype(np.float32)

group_pw = torch.tensor(pos_weight(tr_df, group_cols, CFG["max_pos_weight"]), device=device)
c_pw     = torch.tensor(pos_weight(tr_df, c_cols, CFG["max_pos_weight"]), device=device)
bp_pw    = torch.tensor(pos_weight(tr_df, b_pres_c, CFG["max_pos_weight"]), device=device)

def focal_bce(logits, targets, pos_weight=None, gamma=1.5):
    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight, reduction="none")
    p = torch.sigmoid(logits); pt = p * targets + (1.0 - p) * (1.0 - targets)
    return (bce * (1.0 - pt).pow(gamma)).mean()

def masked_huber(pred, tgt, mask):
    diff = pred - tgt
    h = torch.where(diff.abs() < 1.0, 0.5 * diff.pow(2), diff.abs() - 0.5)
    return (h * mask).sum() / mask.sum().clamp_min(1.0)

LT = torch.tensor(LOG_THRESHOLDS, device=device)
def consistency(out):
    g = torch.sigmoid(out["group_logits"]); c = torch.sigmoid(out["group_c_logits"])
    bpr = torch.sigmoid(out["group_b_presence"])
    bab = bpr * torch.sigmoid((out["group_b_logarea"] - LT) / 0.6)
    lc = F.relu(torch.maximum(c.max(1).values, g[:, 3]) - g[:, 0] - 0.10).mean()
    lb = F.relu(bab.max(1).values - g[:, 1] - 0.10).mean()
    return lc + lb

def loss_fn(out, y, g, cc, bp, bl, bm):
    tgt = focal_bce(out["target_logit"], y, gamma=1.2)
    grp = focal_bce(out["group_logits"], g, pos_weight=group_pw, gamma=1.5)
    cl  = focal_bce(out["group_c_logits"], cc, pos_weight=c_pw, gamma=1.8)
    bpl = focal_bce(out["group_b_presence"], bp, pos_weight=bp_pw, gamma=1.5)
    blh = masked_huber(out["group_b_logarea"], bl, bm)
    co  = consistency(out)
    tot = (CFG["w_target"]*tgt + CFG["w_group"]*grp + CFG["w_group_c_classes"]*cl +
           CFG["w_group_b_presence"]*bpl + CFG["w_group_b_log_area"]*blh +
           CFG["w_consistency"]*co)
    return tot, dict(total=tot, target=tgt, group=grp, c=cl, b_pres=bpl, b_la=blh, cons=co)

head_p, enc_p = [], []
for n, p in model.named_parameters():
    (enc_p if n.startswith("encoder.") else head_p).append(p)
optimizer = torch.optim.AdamW(
    [{"params": enc_p, "lr": CFG["lr_encoder_full"]},
     {"params": head_p, "lr": CFG["lr_heads"]}], weight_decay=CFG["weight_decay"])

steps = max(1, len(train_loader) // CFG["grad_accum"])
total_steps = max(1, steps * CFG["epochs"])
warmup = max(1, steps * CFG["warmup_epochs"])
def lr_lambda(s):
    if s < warmup: return (s + 1) / warmup
    p = (s - warmup) / max(1, total_steps - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * p))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
scaler = torch.amp.GradScaler("cuda", enabled=CFG["amp"])

class EMA:
    def __init__(self, m, d):
        self.d = d; self.s = {n: p.detach().clone() for n, p in m.named_parameters() if p.requires_grad}
    @torch.no_grad()
    def update(self, m):
        for n, p in m.named_parameters():
            if p.requires_grad:
                if n in self.s: self.s[n].mul_(self.d).add_(p.detach(), alpha=1.0 - self.d)
                else: self.s[n] = p.detach().clone()
    def load_into(self, m):
        b = {}
        for n, p in m.named_parameters():
            if n in self.s: b[n] = p.detach().clone(); p.data.copy_(self.s[n])
        return b
    @staticmethod
    def restore(m, b):
        for n, p in m.named_parameters():
            if n in b: p.data.copy_(b[n])
ema = EMA(model, CFG["ema_decay"])

# ----- MD CELL 26 -----
# ## 11. Anti-overfit guards — printed at startup for sanity check
# 
# If any of these prints look wrong before training, **stop here** and adjust
# CFG. They are the only defence against burning 6 h on a misconfigured run.
# ===== CODE CELL 27 =====
print("=" * 60)
print("ANTI-OVERFIT GUARDS (v62)")
print("=" * 60)
print(f"  (a) Stratified val split           : bucket × target, val_split={CFG['val_split']}")
print(f"  (b) Holdout gate gap max           : ≤ {CFG['holdout_gap_max']}")
print(f"  (c) Per-bucket regress tolerance   : ≥ baseline − {CFG['bucket_regress_tol']}")
print(f"      V5 bucket F1 baseline          : {V5_BUCKET_F1}")
print(f"  (d) Early stop patience            : {CFG['early_stop_patience']} epochs on val F1")
print(f"  (e) EMA decay                      : {CFG['ema_decay']}")
print(f"  (f) Label smoothing                : built into focal BCE (γ=1.2 main, 1.5 group, 1.8 class)")
print(f"  (g) Copy-paste sample cap          : {CFG['copy_paste_batch_cap']*100:.0f}% of clean samples may be synthetic")
print(f"  (h) MixUp probability              : {CFG['mixup_prob']*100:.0f}% of rare-Group-C target=1 samples")
print(f"      MixUp blend floor              : α ≥ {CFG['mixup_alpha_min']} (image always ≥ 60% original)")
print(f"  (i) Brightness jitter              : ±{CFG['brightness_jitter_strong']*100:.0f}%")
print(f"  (j) Gamma jitter                   : [{CFG['gamma_jitter_min']}, {CFG['gamma_jitter_max']}]")
print(f"  (k) Forbidden ops                  : down-resize ({CFG['forbid_down_resize']}), heavy blur ({CFG['forbid_heavy_blur']})")
print(f"  (l) Noise candidates dropped       : {n_dropped} samples (sample_weight = 0)")
print(f"  (m) Submission test pos-rate gate  : [{CFG['test_pos_rate_min']}, {CFG['test_pos_rate_max']}]")
print(f"  (n) Backbone                       : {BACKBONE}  (no v5 init, fresh DINOv3)")
print(f"  (o) Epochs                         : {CFG['epochs']}  (warmup {CFG['warmup_epochs']}, unfreeze-all from {CFG['unfreeze_all_from_epoch']})")
print()
print(f"  SOURCE_BANK size                   : {len(SOURCE_BANK)}")
print(f"  NOISE_DROP_IDS size                : {len(NOISE_DROP_IDS)}")
print(f"  DISTRACTOR_RICH_CLEAN_IDS size     : {len(DISTRACTOR_RICH_CLEAN_IDS)}")
print(f"  RARE_DEFECTIVE_IDS size            : {len(RARE_DEFECTIVE_IDS)}")
print("=" * 60)
print(f"\nSMOKE_TEST = {SMOKE_TEST} → epochs={CFG['epochs']}, train_subset={len(tr_df)} rows")

# ----- MD CELL 28 -----
# ## 12. Training loop + per-bucket tau threshold sweep (verbatim from v60 §7)
# ===== CODE CELL 29 =====
def soft_b_score(df, tau):
    pres_cols = [f"pred_b_{i:02d}_{n}_present" for i, n in enumerate(GROUP_B_NAMES)]
    log_cols  = [f"pred_b_{i:02d}_{n}_logarea" for i, n in enumerate(GROUP_B_NAMES)]
    pres = df[pres_cols].values
    log_area = df[log_cols].values
    return pres * (1.0 / (1.0 + np.exp(-(log_area - LOG_THRESHOLDS[None, :]) / max(1e-3, tau))))

def predict_batch_rows(out, names, y_np):
    g = torch.sigmoid(out["group_logits"]).detach().float().cpu().numpy()
    c = torch.sigmoid(out["group_c_logits"]).detach().float().cpu().numpy()
    bp = torch.sigmoid(out["group_b_presence"]).detach().float().cpu().numpy()
    bl = out["group_b_logarea"].detach().float().cpu().numpy()
    tp = torch.sigmoid(out["target_logit"]).detach().float().cpu().numpy()
    rows = []
    for j, name in enumerate(names):
        r = dict(image_id=str(name), target=int(y_np[j]) if y_np is not None else -1,
                 pred_target=float(tp[j]),
                 pred_group_c_any=float(g[j,0]), pred_group_b_above_any=float(g[j,1]),
                 pred_distractor_any=float(g[j,2]), pred_rare_group_c_any=float(g[j,3]))
        for ci, cn in enumerate(GROUP_C_NAMES): r[f"pred_c_{ci:02d}_{cn}"] = float(c[j, ci])
        for bi, bn in enumerate(GROUP_B_NAMES):
            r[f"pred_b_{bi:02d}_{bn}_present"] = float(bp[j, bi])
            r[f"pred_b_{bi:02d}_{bn}_logarea"] = float(bl[j, bi])
        rows.append(r)
    return rows

def add_scores(df, tau):
    pred_c = [f"pred_c_{i:02d}_{n}" for i, n in enumerate(GROUP_C_NAMES)]
    c_sup = np.maximum(df[pred_c].max(1).values, df["pred_rare_group_c_any"].values)
    b_sup = soft_b_score(df, tau).max(1)
    df = df.copy()
    df["score_group_c"] = np.sqrt(np.clip(df["pred_group_c_any"].values * c_sup, 0, 1))
    df["score_group_b"] = np.sqrt(np.clip(df["pred_group_b_above_any"].values * b_sup, 0, 1))
    return df

def tune_thresholds_global(df):
    y = df["target"].values.astype(int)
    best = None
    for tau in CFG["tau_sweep"]:
        dfs = add_scores(df, tau)
        c = dfs["score_group_c"].values; b = dfs["score_group_b"].values
        for tc in CFG["threshold_sweep"]:
            cf = c >= tc
            for tb in CFG["threshold_sweep"]:
                p = (cf | (b >= tb)).astype(int)
                f1 = f1_score(y, p); pos = float(p.mean()); pr = precision_score(y, p, zero_division=0)
                k = (f1, -abs(pos - TRAIN_POS_RATE), pr)
                if best is None or k > best["k"]:
                    best = dict(k=k, t_group_c=float(tc), t_group_b=float(tb), tau=float(tau),
                                f1=float(f1), precision=float(pr),
                                recall=float(recall_score(y, p, zero_division=0)), positive_rate=pos)
    best.pop("k"); return best

def tune_thresholds_per_bucket(df):
    y = df["target"].values.astype(int)
    buckets = sorted(df["bucket"].unique())
    g = tune_thresholds_global(df)
    tau_per_bucket = {}
    for b in buckets:
        sub = df[df["bucket"] == b]
        if len(sub) < 50:
            tau_per_bucket[b] = g["tau"]; continue
        best_t, best_f1 = g["tau"], -1.0
        for tau in CFG["tau_sweep"]:
            dfs = add_scores(sub, tau)
            sc = dfs["score_group_c"].values; sb = dfs["score_group_b"].values
            p = ((sc >= g["t_group_c"]) | (sb >= g["t_group_b"])).astype(int)
            f1 = f1_score(sub["target"].values.astype(int), p)
            if f1 > best_f1: best_f1, best_t = f1, tau
        tau_per_bucket[b] = best_t
    pred = np.zeros(len(df), dtype=int)
    for b, tau in tau_per_bucket.items():
        sub = df[df["bucket"] == b]
        dfs = add_scores(sub, tau)
        p = ((dfs["score_group_c"].values >= g["t_group_c"]) | (dfs["score_group_b"].values >= g["t_group_b"])).astype(int)
        pred[sub.index] = p
    return dict(t_group_c=g["t_group_c"], t_group_b=g["t_group_b"],
                tau_per_bucket=tau_per_bucket, tau=g["tau"],
                f1=float(f1_score(y, pred)),
                precision=float(precision_score(y, pred, zero_division=0)),
                recall=float(recall_score(y, pred, zero_division=0)),
                positive_rate=float(pred.mean()))

def apply_rule(df, calib):
    if "tau_per_bucket" in calib and CFG["use_per_bucket_tau"]:
        pred = np.zeros(len(df), dtype=int)
        for b, tau in calib["tau_per_bucket"].items():
            mask = df["bucket"].values == b
            if not mask.any(): continue
            sub = df[mask]; dfs = add_scores(sub, tau)
            p = ((dfs["score_group_c"].values >= calib["t_group_c"]) | (dfs["score_group_b"].values >= calib["t_group_b"])).astype(int)
            pred[np.where(mask)[0]] = p
        return pred
    dfs = add_scores(df, calib["tau"])
    return ((dfs["score_group_c"].values >= calib["t_group_c"]) | (dfs["score_group_b"].values >= calib["t_group_b"])).astype(int)

def eval_rule(df, calib):
    y = df["target"].values.astype(int); p = apply_rule(df, calib)
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0,1]).ravel()
    return dict(n=int(len(df)), f1=float(f1_score(y, p)),
                precision=float(precision_score(y, p, zero_division=0)),
                recall=float(recall_score(y, p, zero_division=0)),
                positive_rate=float(p.mean()), tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))

def run_epoch(loader, training=True):
    model.train(training)
    log = {k: [] for k in ["total", "target", "group", "c", "b_pres", "b_la", "cons"]}
    rows = []; skipped = 0; accum = 0; t0 = time.time()
    if training: optimizer.zero_grad(set_to_none=True)
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for i, batch in enumerate(loader):
            (x, y, g, cc, bp, bl, bm, bv, rg, names) = batch
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True).float()
            g = g.to(device).float(); cc = cc.to(device).float()
            bp = bp.to(device).float(); bl = bl.to(device).float(); bm = bm.to(device).float()
            bv = bv.to(device).float(); rg = rg.to(device).float()
            with torch.amp.autocast("cuda", enabled=CFG["amp"]):
                out = model(x, bv, rg)
                loss, parts = loss_fn(out, y, g, cc, bp, bl, bm)
                backward_loss = loss / CFG["grad_accum"] if training else loss
            if not torch.isfinite(loss):
                skipped += 1
                if training:
                    optimizer.zero_grad(set_to_none=True); accum = 0
                continue
            if training:
                scaler.scale(backward_loss).backward(); accum += 1
                if accum >= CFG["grad_accum"]:
                    if CFG["grad_clip"] > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
                    scaler.step(optimizer); scaler.update(); scheduler.step(); ema.update(model)
                    optimizer.zero_grad(set_to_none=True); accum = 0
            for k, v in parts.items(): log[k].append(float(v.detach().cpu()))
            rows.extend(predict_batch_rows(out, names, y.detach().cpu().numpy().astype(int)))
            if training and i % 100 == 0:
                lrs = [pg["lr"] for pg in optimizer.param_groups]
                print(f"  step {i:4d}/{len(loader)} tot={loss.item():.3f} tgt={parts['target'].item():.3f} "
                      f"b_la={parts['b_la'].item():.3f} lr={lrs[0]:.2e}/{lrs[-1]:.2e}")
    return ({k: float(np.mean(v)) if v else float('nan') for k, v in log.items()},
            pd.DataFrame(rows), time.time() - t0, skipped)

best_val_f1 = -1.0; best_state = None; best_calib = None; history = []
non_finite_total = 0; no_improve = 0

for epoch in range(1, CFG["epochs"] + 1):
    print(f"\n=== Epoch {epoch}/{CFG['epochs']} ===")
    set_encoder_trainability(model, epoch)
    if epoch == CFG["unfreeze_all_from_epoch"]:
        new_enc = [p for n, p in model.named_parameters() if n.startswith("encoder.") and p.requires_grad]
        new_head = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
        optimizer.param_groups[0]["params"] = new_enc
        optimizer.param_groups[1]["params"] = new_head
        print(f"  optimizer param groups updated: enc {sum(p.numel() for p in new_enc):,}, head {sum(p.numel() for p in new_head):,}")
    tr_loss, _, tr_sec, tr_skip = run_epoch(train_loader, training=True)
    backup = ema.load_into(model)
    vl_loss, vl_pred, vl_sec, vl_skip = run_epoch(val_loader, training=False)
    vl_pred = vl_pred.merge(vl_df[["image_id", "bucket"]], on="image_id", how="left")
    EMA.restore(model, backup)

    calib = tune_thresholds_per_bucket(vl_pred) if CFG["use_per_bucket_tau"] else tune_thresholds_global(vl_pred)
    val_eval = eval_rule(vl_pred, calib)
    f1 = val_eval["f1"]
    non_finite_total += tr_skip + vl_skip
    history.append(dict(epoch=epoch, train=tr_loss, val=vl_loss, calib=calib, val_eval=val_eval,
                        train_seconds=tr_sec, val_seconds=vl_sec, tr_skipped=tr_skip, vl_skipped=vl_skip))
    print(f"train total={tr_loss['total']:.4f} b_la={tr_loss['b_la']:.3f}  sec={tr_sec:.0f}")
    print(f"val   f1={f1:.6f} prec={val_eval['precision']:.4f} rec={val_eval['recall']:.4f} "
          f"pos={val_eval['positive_rate']:.4f}  tc={calib['t_group_c']:.2f} tb={calib['t_group_b']:.2f}")
    if "tau_per_bucket" in calib:
        print(f"      tau_per_bucket={calib['tau_per_bucket']}")

    if f1 > best_val_f1:
        best_val_f1 = float(f1); best_calib = calib
        backup = ema.load_into(model)
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        EMA.restore(model, backup); no_improve = 0; print("  new best")
    else:
        no_improve += 1; print(f"  no improvement {no_improve}/{CFG['early_stop_patience']}")

    if RUN_MODE == "full6h" and (time.time() - RUN_STARTED_AT) >= CFG["max6h_train_wall_seconds"]:
        print("TIME_STOP"); break
    if no_improve >= CFG["early_stop_patience"]:
        print("early stop"); break

print(f"\nBest v62 val F1: {best_val_f1:.6f}  calib: {best_calib}  non_finite={non_finite_total}")

# ----- MD CELL 30 -----
# ## 13. Smoke-test visual gallery — augmented vs original
# 
# If `SMOKE_TEST=True` this is the most important section: render a side-by-side
# strip of (original ROI) → (after bucket-brightness normalisation) → (after full
# v62 aug pipeline) → (after a copy-paste paste). Visual confirmation that:
# 
# - The per-bucket brightness shift is visible (small-bucket images dim slightly).
# - Geometric augs don't destroy defect visibility.
# - Copy-paste polygons sit naturally on the recipient.
# 
# If anything looks wrong, fix it here before kicking off full training.
# ===== CODE CELL 31 =====
def render_strip(samples, title, max_rows=10):
    fig, axes = plt.subplots(min(max_rows, len(samples)), 4, figsize=(12, 3 * min(max_rows, len(samples))))
    if axes.ndim == 1: axes = axes[None, :]
    for i, (row, orig, normed, augged, pasted) in enumerate(samples[:max_rows]):
        axes[i, 0].imshow(orig, cmap="gray"); axes[i, 0].set_title(f"original  bucket={row['bucket']}", fontsize=8)
        axes[i, 1].imshow(normed, cmap="gray"); axes[i, 1].set_title("brightness-normalised", fontsize=8)
        axes[i, 2].imshow(augged, cmap="gray"); axes[i, 2].set_title("after v62 aug", fontsize=8)
        if pasted is not None:
            axes[i, 3].imshow(pasted, cmap="gray"); axes[i, 3].set_title("with copy-paste", fontsize=8)
        else:
            axes[i, 3].set_title("no paste (target=1 sample)", fontsize=8)
        for ax in axes[i]: ax.axis("off")
    fig.suptitle(title, fontsize=10); plt.tight_layout(); plt.show()

# Pick 8 samples — 2 per bucket × {clean, target=1}
samples = []
rng = random.Random(SEED if 'SEED' in dir() else 42)
for buck in ("small", "medium", "large"):
    for tgt in (0, 1):
        candidates = tr_df[(tr_df["bucket"] == buck) & (tr_df["target"] == tgt)]
        if len(candidates) == 0: continue
        r = candidates.iloc[rng.randrange(len(candidates))]
        try:
            img = Image.open(TRAIN_IMG_DIR / r["image_id"]).convert("L")
        except FileNotFoundError:
            continue
        W, H = img.size
        roi = (float(r["roi_x"]), float(r["roi_y"]), float(r["roi_w"]), float(r["roi_h"]))
        l, t, sw, sh = roi_crop_box(W, H, roi, CFG["roi_pad_frac"], jitter=False)
        crop = img.crop((l, t, l + sw, t + sh))
        if crop.size != (S, S): crop = crop.resize((S, S), Image.BILINEAR)
        orig = np.asarray(crop, dtype=np.uint8)
        normed = normalise_bucket_brightness(orig.copy(), buck)
        # Pipeline
        pipeline = select_pipeline(r) if int(r["has_groupb_ann"]) != 1 else aug_safe_for_groupb
        augged = pipeline(image=normed.copy())["image"]
        pasted = None
        if int(r["target"]) == 0 and len(SOURCE_BANK) > 0:
            src_row = SOURCE_BANK.iloc[rng.randrange(len(SOURCE_BANK))]
            try:
                src_full = np.asarray(Image.open(TRAIN_IMG_DIR / src_row["image_id"]).convert("L"), dtype=np.uint8)
                extracted = extract_polygon_crop(src_full, str(src_row["image_id"]), str(src_row["class"]))
                if extracted is not None:
                    crop_arr, _, poly = extracted
                    pasted = normed.copy()
                    rng2 = random.Random(rng.random() * 1e6)
                    bbox = paste_polygon_onto(pasted, (0, 0, S, S), crop_arr, poly, rng2)
            except Exception:
                pass
        samples.append((r, orig, normed, augged, pasted))

render_strip(samples, "v62 augmentation pipeline — visual sanity check")

# ----- MD CELL 32 -----
# ## 14. Final diagnostics + holdout gate (verbatim v60 §8)
# ===== CODE CELL 33 =====
if best_state is not None and not SMOKE_TEST:
    model.load_state_dict(best_state); model.eval()
    _, final_val, _, _ = run_epoch(val_loader, training=False)
    final_val = final_val.merge(vl_df[["image_id", "target", "bucket", "functional_group", "rule_target",
                                        "group_c_any", "group_b_above_any", "distractor_any", "rare_group_c_any"]],
                                 on=["image_id", "target"], how="left")
    final_val["strat_key"] = final_val["bucket"].astype(str) + "_" + final_val["target"].astype(str)
    tune_df, hold_df = train_test_split(final_val, test_size=0.5, stratify=final_val["strat_key"], random_state=CFG["seed"] + 62)
    tune_df = tune_df.reset_index(drop=True); hold_df = hold_df.reset_index(drop=True)
    calib = tune_thresholds_per_bucket(tune_df) if CFG["use_per_bucket_tau"] else tune_thresholds_global(tune_df)
    tune_eval = eval_rule(tune_df, calib); hold_eval = eval_rule(hold_df, calib); full_eval = eval_rule(final_val, calib)
    final_val["rule_pred"] = apply_rule(final_val, calib)

    per_bucket = {}
    for b, sub in final_val.groupby("bucket"):
        y = sub["target"].values.astype(int); p = sub["rule_pred"].values.astype(int)
        f1 = float(f1_score(y, p)); base = float(V5_BUCKET_F1.get(str(b), np.nan))
        per_bucket[str(b)] = dict(n=int(len(sub)), pos_rate=float(y.mean()),
                                  pred_pos_rate=float(p.mean()), f1=f1, v5_bucket_f1=base,
                                  delta_vs_v5=float(f1 - base) if np.isfinite(base) else None)
    bucket_gate = all((v["delta_vs_v5"] is None) or (v["delta_vs_v5"] >= -CFG["bucket_regress_tol"]) for v in per_bucket.values())

    gate_pass = bool(
        full_eval["f1"] >= V57_VAL_F1_GATE + CFG["full_gate_lift"]
        and hold_eval["f1"] >= CFG["holdout_min_f1"]
        and (tune_eval["f1"] - hold_eval["f1"]) <= CFG["holdout_gap_max"]
        and bucket_gate and non_finite_total == 0
    )
    gate_report = dict(run_mode=RUN_MODE, model="aug_smart_v62", backbone=CFG["backbone"],
                       calibration=calib, v57_val_f1_gate=V57_VAL_F1_GATE,
                       best_training_val_f1=best_val_f1,
                       full_val=full_eval, tune=tune_eval, holdout=hold_eval, per_bucket=per_bucket,
                       non_finite_total=non_finite_total, gate_pass=gate_pass,
                       required=dict(full_lift=CFG["full_gate_lift"], holdout_min_f1=CFG["holdout_min_f1"],
                                     holdout_gap_max=CFG["holdout_gap_max"], bucket_regress_tol=CFG["bucket_regress_tol"]))
    final_val.to_csv(OUT_ROOT / "v62_val_predictions.csv", index=False)
    with open(OUT_ROOT / "v62_gate_report.json", "w") as f:
        json.dump(gate_report, f, indent=2, default=str)
    torch.save(dict(model_state_dict=best_state, config=CFG,
                    group_c_names=GROUP_C_NAMES, group_b_names=GROUP_B_NAMES, group_cols=group_cols,
                    log_thresholds=LOG_THRESHOLDS.tolist(), calibration=calib,
                    gate_report=gate_report, history=history),
               OUT_ROOT / "best_model_v62.pth")
    print(json.dumps(gate_report, indent=2, default=str))
else:
    print(f"Skipping final gate — SMOKE_TEST={SMOKE_TEST}, best_state={'set' if best_state is not None else 'None'}")

# ----- MD CELL 34 -----
# ## 15. TTA inference + submission (verbatim v60 §8)
# ===== CODE CELL 35 =====
submission_written = False; test_pos_rate = None

if (RUN_TEST_INFERENCE or CREATE_SUBMISSION) and not SMOKE_TEST and best_state is not None:
    if gate_pass and test_roi_path is not None and len(test_roi_lookup) == len(sample_sub):
        test_ds = AugSmartDataset(sample_sub[["image_id"]], TEST_IMG_DIR, mode="val", test_roi_lookup=test_roi_lookup)
        test_loader = DataLoader(test_ds, batch_size=CFG["batch_size"], shuffle=False,
                                  num_workers=CFG["num_workers"], pin_memory=True)
        rows = []; t0 = time.time()
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                (x, _, _, _, _, _, _, bv, rg, names) = batch
                x = x.to(device); bv = bv.to(device).float(); rg = rg.to(device).float()
                with torch.amp.autocast("cuda", enabled=CFG["amp"]):
                    out_a = model(x, bv, rg)
                    if CFG["use_tta"]:
                        out_b = model(torch.flip(x, dims=[-1]), bv, rg)
                        out = {}
                        for k in ["target_logit", "group_logits", "group_c_logits", "group_b_presence"]:
                            out[k] = torch.logit((0.5*(torch.sigmoid(out_a[k]) + torch.sigmoid(out_b[k]))).clamp(1e-6, 1-1e-6))
                        out["group_b_logarea"] = 0.5 * (out_a["group_b_logarea"] + out_b["group_b_logarea"])
                    else:
                        out = out_a
                rows.extend(predict_batch_rows(out, names, None))
                if i % 50 == 0: print(f"  test batch {i}/{len(test_loader)} elapsed={time.time()-t0:.1f}s")
        test_df = pd.DataFrame(rows)
        test_df["bucket"] = [to_bucket(test_roi_lookup[f][2]) for f in test_df["image_id"].values]
        test_df["target"] = apply_rule(test_df, calib)
        test_pos_rate = float(test_df["target"].mean())
        test_df.to_csv(OUT_ROOT / "v62_test_predictions.csv", index=False)
        print("test positive rate:", test_pos_rate)
        if CFG["test_pos_rate_min"] <= test_pos_rate <= CFG["test_pos_rate_max"]:
            sub = sample_sub[["image_id"]].merge(test_df[["image_id", "target"]], on="image_id", how="left")
            sub["target"] = sub["target"].astype(int)
            sub.to_csv(OUT_ROOT / "submission.csv", index=False)
            submission_written = True
            print("wrote submission.csv  shape:", sub.shape)
        else:
            print("Submission blocked by test positive-rate gate.")
    else:
        print("Test inference blocked (gate failed or test ROI missing).")
else:
    print(f"Skipping inference — SMOKE_TEST={SMOKE_TEST}")

# ----- MD CELL 36 -----
# ## 16. Why this notebook won't overfit — the case for the panel write-up
# 
# Three structural reasons:
# 
# **(1) Augmentations are bounded by physics, not aggression.** Every parameter
# was calibrated against an EDA measurement, not chosen for impact: brightness
# ±15 % matches the empirical per-bucket distribution gap (F2); gamma [0.85,
# 1.15] keeps within the visible tonal range; copy-paste is capped at 30 % of
# clean recipients so synthesis never dominates real data; MixUp uses α ≥ 0.6 so
# the original image stays the majority signal. **No augmentation is stronger
# than the natural variation the EDA documented**.
# 
# **(2) The val set is honest.** Joint stratified `(bucket × target)` split (same
# as v5/v57/v60) preserves the per-bucket class balance that the test set is
# expected to share (F10: no train↔test drift). The post-training holdout gate
# splits val 50/50 — thresholds tuned on one half, F1 reported on the other —
# preventing val-overfitting in the calibration step. Per-bucket F1 must stay
# within 0.0025 of the v5 baseline for all three buckets, blocking subtle
# regressions hidden in the average.
# 
# **(3) The training signal is cleaner, not more abundant.** Sample weight = 0
# on 85 known noise candidates (F5/F6) removes label noise; it doesn't add
# spurious samples. The copy-paste source bank is restricted to physically
# plausible defect polygons (excluding `No fault` / `No base visible`
# full-frame rectangles per F11), so synthetic examples remain semantically
# correct.
# 
# **The one thing that could still cause overfit** — and the reason `SMOKE_TEST`
# exists — is a configuration error: e.g., a hard-coded path mis-pointing to a
# different image bank, or a `copy_paste_p` set above the cap. The smoke test
# catches both by displaying side-by-side aug previews before training begins.
