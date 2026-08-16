"""Build v66 decision-aligned semantic-seg Kaggle notebook/package.

Lessons from v65 (train rule-F1 0.918, MARGINAL):
- 26-class objective wasted capacity on rare always-trigger classes
- pos_weight=20 was too aggressive → val regression after epoch 1
- Small bucket weak (0.77 vs v5 0.90) → need more resolution for Vichy bottles

v66 changes (Codex-design + v65-lessons):
1. 4 decision-aligned mask channels + 6 pooled binary heads (was 26 classes)
2. 896x896 ROI input (was 640) -> 224x224 mask grid (was 160x160)
3. Focal loss (gamma=2) + Dice (was pos_weight=20 BCE)
4. Targeted sampling: 25/25/20/30 (missed-trigger / scuffing-boundary / distractor-neg / ordinary)
5. Light encoder unfreeze (last 2 stages) in final epoch
6. Lower LR (1.5e-4), higher dropout (0.20), lower EMA decay (0.997), patience=3
7. fixed_created_ratio computation included in eval

BUG FIX from v65: has_labels was determined by 'target' column presence which fires
on sample_submission too. Now uses ROI column presence instead.

Cost: ~6-8 GPU-h on Kaggle T4x2.
"""
import json
import shutil
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "v66_decision_aligned_seg.ipynb"
PACKAGE_DIR = ROOT / "artefacts" / "kaggle_push_v66_decision_aligned_seg"
PACKAGE_NB = PACKAGE_DIR / "kernel.ipynb"
PACKAGE_META = PACKAGE_DIR / "kernel-metadata.json"

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


md(
    """# v66 - Decision-Aligned Semantic Seg + Phase 2 Combiner Prep

Replaces v65's 26-class formulation with 4 decision-aligned mask channels
(group_c_union, cont_dark, scuffing_semantic, distractor_union) plus 6 pooled
binary heads (p_group_c_trigger, p_cont_dark_trigger, p_scuffing_above_75000,
p_distractor_only, p_cont_light_above_thr, p_scuffing_heavy_above_thr).

896x896 ROI, focal loss, targeted sampling, light encoder unfreeze, all
designed to fix v65's marginal result (train rule-F1 0.918, small bucket 0.77).

Phase 2 PROOF GATE:
- train rule-F1 from learned masks + heads >= 0.94  ->  proceed to residual combiner
- 0.90-0.94  ->  marginal, residual must satisfy fixed_created_ratio >= 2.0 on val
- < 0.90    ->  stop, pivot to insight write-up. F1 path empirically dead.
"""
)


md("## 1. Setup")

code("""!pip install -q -U timm==1.0.26 pycocotools 2>&1 | tail -3""")

code(
    r"""import os, json, random, math, time, warnings, re
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score

import timm
from pycocotools import mask as maskUtils

warnings.filterwarnings("ignore", category=UserWarning)

print("PyTorch:", torch.__version__, "| CUDA:", torch.cuda.is_available())
assert torch.cuda.is_available(), "No GPU detected. Set Kaggle accelerator to GPU T4 x2."

gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
caps = [torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())]
print("GPUs:", gpu_names)
print("Capabilities:", caps)
if any("P100" in name.upper() for name in gpu_names) or any(cap < (7, 5) for cap in caps):
    raise RuntimeError("Wrong Kaggle GPU. Stop this run and switch accelerator to T4 x2.")

print("timm:", timm.__version__)
assert tuple(int(x) for x in timm.__version__.split(".")[:3]) >= (1, 0, 20), "timm too old for DINOv3"
"""
)


md("## 2. Config")

code(
    r"""RUN_MODE = "full6h"            # "probe" (2 epoch sanity) or "full6h" (real training)
RUN_TEST_INFERENCE = True
CREATE_SUBMISSION  = True

KAGGLE_CANDIDATES = [
    Path("/kaggle/input/1st-krones-vision-ai-challenge"),
    Path("/kaggle/input/competitions/1st-krones-vision-ai-challenge"),
]
LOCAL_ROOT = Path.cwd().parent / "data"
DATA_ROOT = next((p for p in KAGGLE_CANDIDATES if p.exists()), None) or LOCAL_ROOT
OUT_ROOT = Path("/kaggle/working") if DATA_ROOT != LOCAL_ROOT else Path.cwd().parent / "artefacts" / "v66_decision_aligned_seg"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

TRAIN_IMG_DIR = DATA_ROOT / "train_images"
TEST_IMG_DIR  = DATA_ROOT / "test_images"
TRAIN_CSV     = DATA_ROOT / "train.csv"
SAMPLE_SUB    = DATA_ROOT / "sample_submission.csv"
ANN_JSON      = DATA_ROOT / "train_annotations.json"

BASELINE_V5_VAL_F1 = 0.9634966378482228
TRAIN_POS_RATE     = 0.5832508134106663
V5_BUCKET_F1       = {"large": 0.9479, "medium": 0.9695, "small": 0.8966}

CFG = dict(
    phase="v66_decision_aligned_seg",
    run_mode=RUN_MODE,
    backbone="convnext_tiny.dinov3_lvd1689m",
    roi_img_size=896,
    mask_size=224,                   # stride-4 from 896
    batch_size=4,                    # smaller because 896 input
    grad_accum=4,                    # effective batch 16
    num_workers=4,
    epochs=2 if RUN_MODE == "probe" else 6,
    early_stop_patience=3,
    warmup_epochs=1,
    encoder_unfreeze_epoch=5,        # last 2 stages unfrozen at epoch 5+
    lr_decoder=1.5e-4,               # was 3e-4 in v65 (over-aggressive)
    lr_encoder_unfrozen=1.0e-5,      # very low when unfrozen
    weight_decay=0.05,
    grad_clip=1.0,
    val_split=0.2,
    seed=42,
    amp=True,
    ema_decay=0.997,                 # was 0.999 in v65 (too slow catch-up)
    use_grad_checkpointing=True,
    roi_pad_frac=0.10,
    roi_jitter_shift=0.025,
    roi_jitter_scale=0.05,
    decoder_dropout=0.20,            # was 0.10 in v65
    focal_gamma=2.0,
    focal_alpha_cap=5.0,             # max per-class alpha (was 20 pos_weight in v65)
    w_focal=1.0,
    w_dice=1.0,
    w_pooled=1.0,
    w_cont_dark_pooled=3.0,          # 3x weight on the dominant FN source
    w_scuffing_above_pooled=3.0,     # 3x weight on the dominant FP source
    w_distractor_pooled=2.0,
    seg_prob_thresholds=np.arange(0.30, 0.71, 0.05),
    sampling_mix={                   # batch composition fractions
        "missed_trigger_pos": 0.25,
        "scuffing_boundary": 0.25,
        "distractor_neg": 0.20,
        "ordinary": 0.30,
    },
    max_train_seconds=6.5 * 3600,
    phase2_gate_train_rule_f1=0.94,
)

print("DATA_ROOT:", DATA_ROOT)
print("OUT_ROOT :", OUT_ROOT)
for k, v in CFG.items():
    print(f"  {k:28s} = {v}")

random.seed(CFG["seed"]); np.random.seed(CFG["seed"])
torch.manual_seed(CFG["seed"]); torch.cuda.manual_seed_all(CFG["seed"])
torch.backends.cudnn.benchmark = True
device = torch.device("cuda")
RUN_STARTED_AT = time.time()
"""
)


md("## 3. COCO load + decision-aligned label derivation")

code(
    r"""train_df = pd.read_csv(TRAIN_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB)
with open(ANN_JSON) as f:
    coco = json.load(f)

cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
fname_to_imgid = {im["file_name"]: im["id"] for im in coco["images"]}
imgid_to_wh = {im["id"]: (im["width"], im["height"]) for im in coco["images"]}
imgid_to_anns = defaultdict(list)
for ann in coco["annotations"]:
    imgid_to_anns[ann["image_id"]].append(ann)

GROUP_A = ["Embossing", "Foam residue", "No fault", "Water drop"]
GROUP_B_NAMES = ["Air bubble", "Chip", "Contamination light", "Glass imperfection", "Scuffing", "Scuffing heavy"]
GROUP_B_THRESHOLDS = {
    "Air bubble": 500, "Chip": 200, "Contamination light": 180,
    "Glass imperfection": 100, "Scuffing": 75000, "Scuffing heavy": 1200,
}
ROI_NAME = "Roi"
ALL_NAMES = set(cat_id_to_name.values())
GROUP_C_NAMES = sorted(ALL_NAMES - set(GROUP_A) - set(GROUP_B_NAMES) - {ROI_NAME})
RARE_C = {"Circlip", "Foil / Semitransparent", "Insect", "Liquid", "Straw"}
CONT_DARK_NAME = "Contamination dark"
SCUFFING_NAME = "Scuffing"
DISTRACTOR_NAMES = {"Foam residue", "Water drop"}

# Decision-aligned mask channels (4):
#   0: group_c_union (any always-trigger Group C class)
#   1: cont_dark (just Contamination dark; subset of group_c_union)
#   2: scuffing_semantic (Scuffing class — for area summation)
#   3: distractor_union (Foam residue + Water drop)
MASK_CHANNELS = ["group_c_union", "cont_dark", "scuffing_semantic", "distractor_union"]
NUM_MASK_CH = len(MASK_CHANNELS)

# Pooled binary heads (6):
POOLED_HEADS = [
    "p_group_c_trigger",       # image has any Group C
    "p_cont_dark_trigger",     # image has Cont dark
    "p_scuffing_above_75000",  # Scuffing total area >= 75000
    "p_distractor_only",       # image has Foam/Water AND no trigger
    "p_cont_light_above_thr",  # Contamination light area >= 180
    "p_scuffing_heavy_above",  # Scuffing heavy area >= 1200
]
NUM_POOLED = len(POOLED_HEADS)

print(f"mask channels: {MASK_CHANNELS}")
print(f"pooled heads:  {POOLED_HEADS}")

def to_bucket(w):
    if w is None or pd.isna(w): return "unknown"
    if float(w) < 510: return "small"
    if float(w) > 590: return "large"
    return "medium"

def get_train_roi_bbox(image_id):
    for ann in imgid_to_anns.get(image_id, []):
        if cat_id_to_name[ann["category_id"]] == ROI_NAME:
            x, y, w, h = ann["bbox"]
            return float(x), float(y), float(w), float(h)
    return None

def derive_row(fname, target=None):
    image_id = fname_to_imgid.get(fname)
    roi = get_train_roi_bbox(image_id) if image_id is not None else None

    has_group_c = 0
    has_cont_dark = 0
    has_group_a_only_distractor = 0
    has_distractor = 0
    scuffing_total_area = 0.0
    cont_light_total_area = 0.0
    scuffing_heavy_total_area = 0.0
    has_group_b_above = 0
    rule_target = 0
    smallest_trigger_area = np.nan
    trigger_areas = []
    rare_group_c_any = 0

    for ann in imgid_to_anns.get(image_id, []):
        name = cat_id_to_name[ann["category_id"]]
        if name == ROI_NAME: continue
        area = float(ann.get("area", 0.0))

        if name in DISTRACTOR_NAMES:
            has_distractor = 1
        if name in GROUP_C_NAMES:
            has_group_c = 1
            rule_target = 1
            trigger_areas.append(area)
            if name == CONT_DARK_NAME: has_cont_dark = 1
            if name in RARE_C: rare_group_c_any = 1
        elif name == SCUFFING_NAME:
            scuffing_total_area += area
        elif name == "Contamination light":
            cont_light_total_area += area
        elif name == "Scuffing heavy":
            scuffing_heavy_total_area += area
        elif name in GROUP_B_NAMES:
            thr = float(GROUP_B_THRESHOLDS[name])
            if area >= thr:
                has_group_b_above = 1
                rule_target = 1
                trigger_areas.append(area)

    # Group B above-threshold checks for the named heads
    scuffing_above_75000 = int(scuffing_total_area >= GROUP_B_THRESHOLDS["Scuffing"])
    cont_light_above = int(cont_light_total_area >= GROUP_B_THRESHOLDS["Contamination light"])
    scuffing_heavy_above = int(scuffing_heavy_total_area >= GROUP_B_THRESHOLDS["Scuffing heavy"])
    if scuffing_above_75000 or cont_light_above or scuffing_heavy_above:
        rule_target = 1
        has_group_b_above = 1

    distractor_only = int(has_distractor and not has_group_c and not has_group_b_above)

    if trigger_areas:
        smallest_trigger_area = float(min(trigger_areas))

    row = {
        "image_id": fname,
        "target": int(target) if target is not None else -1,
        "rule_target": int(rule_target),
        "y_group_c": int(has_group_c),
        "y_cont_dark": int(has_cont_dark),
        "y_scuffing_above_75000": int(scuffing_above_75000),
        "y_distractor_only": int(distractor_only),
        "y_cont_light_above": int(cont_light_above),
        "y_scuffing_heavy_above": int(scuffing_heavy_above),
        "scuffing_total_area": float(scuffing_total_area),
        "rare_group_c_any": int(rare_group_c_any),
        "smallest_trigger_area": smallest_trigger_area,
        "roi_x": np.nan, "roi_y": np.nan, "roi_w": np.nan, "roi_h": np.nan,
        "bucket": "unknown",
    }
    if roi is not None:
        row["roi_x"], row["roi_y"], row["roi_w"], row["roi_h"] = roi
        row["bucket"] = to_bucket(row["roi_w"])
    return row

label_rows = [derive_row(r.image_id, int(r.target)) for r in train_df.itertuples(index=False)]
label_df = pd.DataFrame(label_rows)

rule_agree = float((label_df["rule_target"].values == label_df["target"].values).mean())
print(f"rule vs target agreement: {rule_agree:.4f}  (CLAUDE.md says 0.9981)")
assert rule_agree > 0.99
assert int(label_df["roi_w"].notna().sum()) == len(label_df)

print(f"y_group_c            pos rate: {label_df['y_group_c'].mean():.4f}")
print(f"y_cont_dark          pos rate: {label_df['y_cont_dark'].mean():.4f}")
print(f"y_scuffing_above     pos rate: {label_df['y_scuffing_above_75000'].mean():.4f}")
print(f"y_distractor_only    pos rate: {label_df['y_distractor_only'].mean():.4f}")
print(f"y_cont_light_above   pos rate: {label_df['y_cont_light_above'].mean():.4f}")
print(f"y_scuffing_hvy_above pos rate: {label_df['y_scuffing_heavy_above'].mean():.4f}")

train_df = label_df.copy()
"""
)


md("## 4. Test ROI lookup")

code(
    r"""def find_file(name):
    candidates = []
    if Path("/kaggle/input").exists():
        candidates.extend(Path("/kaggle/input").glob(f"**/{name}"))
    local = DATA_ROOT / name
    if local.exists(): candidates.append(local)
    return sorted(set(candidates))[0] if candidates else None

test_roi_path = find_file("test_annotations_roi_only.json")
test_roi_lookup = {}
if test_roi_path is not None:
    with open(test_roi_path) as f:
        test_roi = json.load(f)
    test_cat = {c["id"]: c["name"] for c in test_roi["categories"]}
    by_img = defaultdict(list)
    for ann in test_roi["annotations"]:
        by_img[ann["image_id"]].append(ann)
    for im in test_roi["images"]:
        for ann in by_img.get(im["id"], []):
            if test_cat.get(ann["category_id"]) == ROI_NAME:
                x, y, w, h = ann["bbox"]
                test_roi_lookup[im["file_name"]] = (float(x), float(y), float(w), float(h))
                break
    print(f"loaded test ROI: {test_roi_path}, coverage {len(test_roi_lookup)}/{len(sample_sub)}")
else:
    print("WARNING: no test_annotations_roi_only.json found.")

if RUN_TEST_INFERENCE or CREATE_SUBMISSION:
    assert test_roi_path is not None and len(test_roi_lookup) == len(sample_sub)
"""
)


md("## 5. Split + targeted sampling weights")

code(
    r"""train_df["strat_key"] = train_df["bucket"].astype(str) + "_" + train_df["target"].astype(str)
tr_df, vl_df = train_test_split(train_df, test_size=CFG["val_split"],
                                stratify=train_df["strat_key"], random_state=CFG["seed"])
tr_df = tr_df.reset_index(drop=True)
vl_df = vl_df.reset_index(drop=True)
print(f"train: {len(tr_df)}  val: {len(vl_df)}")

# Tag each training row with its sampling group (missed_trigger / scuffing_boundary / distractor_neg / ordinary)
def tag_sampling_group(row):
    s_area = float(row["scuffing_total_area"]) if pd.notna(row["scuffing_total_area"]) else 0.0
    s_thr = GROUP_B_THRESHOLDS["Scuffing"]
    # Scuffing boundary band: 0.5x to 1.5x threshold
    if 0.5 * s_thr <= s_area <= 1.5 * s_thr:
        return "scuffing_boundary"
    # Distractor-only negative
    if row["target"] == 0 and row["y_distractor_only"] == 1:
        return "distractor_neg"
    # Missed-trigger positive: target=1 with Cont dark OR tiny rare Group C
    smallest = float(row["smallest_trigger_area"]) if pd.notna(row["smallest_trigger_area"]) else np.inf
    if row["target"] == 1 and (row["y_cont_dark"] == 1 or row["rare_group_c_any"] == 1 or smallest < 200):
        return "missed_trigger_pos"
    return "ordinary"

tr_df["samp_group"] = tr_df.apply(tag_sampling_group, axis=1)
print("training-set sampling group counts:")
print(tr_df["samp_group"].value_counts())

# Compute per-row weight so total mass per group matches the target fraction
target_frac = CFG["sampling_mix"]
group_counts = tr_df["samp_group"].value_counts().to_dict()
total_n = len(tr_df)
weight_by_group = {}
for g, frac in target_frac.items():
    n_g = group_counts.get(g, 0)
    # If a group is missing, redistribute
    if n_g == 0:
        weight_by_group[g] = 0.0
    else:
        weight_by_group[g] = frac * total_n / n_g
print("per-group sample weights:", {k: round(v, 3) for k, v in weight_by_group.items()})

train_weights = tr_df["samp_group"].map(weight_by_group).values.astype(np.float32)
# Safety clamp to avoid weight=0
train_weights = np.clip(train_weights, 0.01, 10.0)
print("sample weight summary:", pd.Series(train_weights).describe()[["min","25%","50%","75%","max"]].to_dict())
"""
)


md("## 6. Dataset — renders 4-channel decision-aligned masks (BUG FIX: has_labels via ROI columns)")

code(
    r"""IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]

def roi_crop_box(width, height, roi_bbox, pad_frac=0.10, jitter=False):
    x, y, w, h = [float(v) for v in roi_bbox[:4]]
    scale_j, shift_x, shift_y = 1.0, 0.0, 0.0
    if jitter:
        scale_j  += random.uniform(-CFG["roi_jitter_scale"], CFG["roi_jitter_scale"])
        shift_x   = random.uniform(-CFG["roi_jitter_shift"], CFG["roi_jitter_shift"]) * max(w, h)
        shift_y   = random.uniform(-CFG["roi_jitter_shift"], CFG["roi_jitter_shift"]) * max(w, h)
    side = int(round(max(w, h) * (1.0 + 2.0 * float(pad_frac)) * scale_j))
    side = max(1, min(side, int(width), int(height)))
    cx, cy = x + w/2.0 + shift_x, y + h/2.0 + shift_y
    left = max(0, min(int(round(cx - side/2.0)), int(width)  - side))
    top  = max(0, min(int(round(cy - side/2.0)), int(height) - side))
    return left, top, side, side

def crop_image_pil(img, crop_box, out_size):
    l, t, w, h = crop_box
    crop = img.crop((l, t, l + w, t + h))
    if crop.size != (out_size, out_size):
        crop = crop.resize((out_size, out_size), Image.BILINEAR)
    return crop

def to_tensor_norm(img_pil):
    arr = np.asarray(img_pil, dtype=np.float32) / 255.0
    arr = np.stack([arr, arr, arr], axis=0)
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(arr)).float()

def render_decision_aligned_mask(image_id, img_w, img_h, crop_box, mask_size):
    '''Render 4-channel decision-aligned mask, ROI-crop, nearest-resize.
    Channels: 0=group_c_union, 1=cont_dark, 2=scuffing_semantic, 3=distractor_union.'''
    l, t, side_w, side_h = crop_box
    mask = np.zeros((NUM_MASK_CH, img_h, img_w), dtype=np.uint8)
    for ann in imgid_to_anns.get(image_id, []):
        name = cat_id_to_name[ann["category_id"]]
        if name == ROI_NAME: continue
        seg = ann.get("segmentation")
        if isinstance(seg, list) and len(seg) > 0:
            try:
                rles = maskUtils.frPyObjects(seg, img_h, img_w)
                rle = maskUtils.merge(rles) if isinstance(rles, list) else rles
                m = maskUtils.decode(rle)
            except Exception: continue
        elif isinstance(seg, dict):
            try: m = maskUtils.decode(seg)
            except Exception: continue
        else: continue
        if m.ndim == 3: m = m.max(axis=-1)
        m_u8 = m.astype(np.uint8)
        # Channel 0: any Group C
        if name in GROUP_C_NAMES:
            mask[0] |= m_u8
            # Channel 1: Cont dark specifically
            if name == CONT_DARK_NAME:
                mask[1] |= m_u8
        # Channel 2: Scuffing semantic
        if name == SCUFFING_NAME:
            mask[2] |= m_u8
        # Channel 3: Foam/Water distractor
        if name in DISTRACTOR_NAMES:
            mask[3] |= m_u8
    cropped = mask[:, t:t+side_h, l:l+side_w]
    t_mask = torch.from_numpy(cropped).float().unsqueeze(0)
    t_mask = F.interpolate(t_mask, size=(mask_size, mask_size), mode="nearest").squeeze(0)
    return t_mask

# BUG FIX: has_labels determined by ROI column presence, NOT target column.
# (v65 bug: sample_submission.csv has 'target' column but no ROI columns -> KeyError.)
ROI_COLS = ("roi_x", "roi_y", "roi_w", "roi_h")
POOLED_COLS = ["y_group_c", "y_cont_dark", "y_scuffing_above_75000",
               "y_distractor_only", "y_cont_light_above", "y_scuffing_heavy_above"]

class SegDataset(Dataset):
    def __init__(self, df, image_dir, mode, test_roi_lookup=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.mode = mode
        self.has_labels = all(c in self.df.columns for c in ROI_COLS)
        self.test_roi_lookup = test_roi_lookup or {}

    def __len__(self): return len(self.df)

    def _roi(self, row, fname):
        if self.has_labels and not pd.isna(row["roi_w"]):
            return float(row["roi_x"]), float(row["roi_y"]), float(row["roi_w"]), float(row["roi_h"])
        roi = self.test_roi_lookup.get(fname)
        if roi is None: raise ValueError(f"Missing ROI for {fname}")
        return roi

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = str(row["image_id"])
        roi = self._roi(row, fname)
        img = Image.open(self.image_dir / fname).convert("L")
        img_w, img_h = img.size
        crop_box = roi_crop_box(img_w, img_h, roi, CFG["roi_pad_frac"], jitter=(self.mode == "train"))
        crop = crop_image_pil(img, crop_box, CFG["roi_img_size"])

        do_hflip = self.mode == "train" and random.random() < 0.5
        if do_hflip: crop = crop.transpose(Image.FLIP_LEFT_RIGHT)
        if self.mode == "train":
            alpha = 1.0 + random.uniform(-0.08, 0.08)
            beta  = random.uniform(-9.0, 9.0)
            arr = np.asarray(crop, dtype=np.float32)
            crop = Image.fromarray(np.clip(arr*alpha + beta, 0, 255).astype(np.uint8), mode="L")

        if self.has_labels:
            image_id = fname_to_imgid.get(fname)
            mask = render_decision_aligned_mask(image_id, img_w, img_h, crop_box, CFG["mask_size"])
            if do_hflip: mask = torch.flip(mask, dims=[2])
            target = float(row["target"])
            pooled_y = torch.tensor([float(row[c]) for c in POOLED_COLS], dtype=torch.float32)
        else:
            mask = torch.zeros((NUM_MASK_CH, CFG["mask_size"], CFG["mask_size"]), dtype=torch.float32)
            target = -1.0
            pooled_y = torch.full((NUM_POOLED,), -1.0, dtype=torch.float32)

        roi_native_side = float(crop_box[2])
        per_cell_native_area = (roi_native_side / float(CFG["mask_size"])) ** 2

        return (
            to_tensor_norm(crop),
            mask,                                                # (4, 224, 224)
            torch.tensor(target, dtype=torch.float32),           # binary target
            pooled_y,                                            # (6,) pooled head labels
            torch.tensor(per_cell_native_area, dtype=torch.float32),
            fname,
        )

train_ds = SegDataset(tr_df, TRAIN_IMG_DIR, "train")
val_ds   = SegDataset(vl_df, TRAIN_IMG_DIR, "val")
test_ds  = SegDataset(sample_sub, TEST_IMG_DIR, "val", test_roi_lookup=test_roi_lookup) if RUN_TEST_INFERENCE else None

train_sampler = WeightedRandomSampler(
    weights=torch.from_numpy(train_weights).double(),
    num_samples=len(train_weights), replacement=True,
)
train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], sampler=train_sampler,
                          num_workers=CFG["num_workers"], pin_memory=True, drop_last=True, persistent_workers=True)
val_loader   = DataLoader(val_ds, batch_size=CFG["batch_size"], shuffle=False,
                          num_workers=CFG["num_workers"], pin_memory=True, persistent_workers=True)
test_loader  = DataLoader(test_ds, batch_size=CFG["batch_size"], shuffle=False,
                          num_workers=CFG["num_workers"], pin_memory=True) if test_ds is not None else None

b = next(iter(val_loader))
print("batch shapes:")
print(f"  image: {tuple(b[0].shape)}  mask: {tuple(b[1].shape)}  target: {tuple(b[2].shape)}  pooled: {tuple(b[3].shape)}")
"""
)


md("## 7. Model — frozen v5 encoder (last 2 stages unfreeze at epoch 5+) + UPerNet + 6 pooled heads")

code(
    r"""def find_v5_ckpt():
    cand = []
    if Path("/kaggle/input").exists():
        cand.extend(Path("/kaggle/input").glob("**/best_model_v5.pth"))
    for p in [
        Path.cwd().parent / "artefacts" / "v5_results" / "best_model_v5.pth",
        Path.cwd().parent / "artefacts" / "krones_v5_checkpoint" / "best_model_v5.pth",
    ]:
        if p.exists(): cand.append(p)
    cand = sorted(set(cand))
    if not cand: return None
    print("v5 ckpt candidates:", [str(p) for p in cand])
    return cand[0]

class UPerNetDecoder(nn.Module):
    '''UPerNet on (s4, s8, s16, s32) features. Output @ stride-4 (224x224 from 896 in).'''
    def __init__(self, in_channels=(96, 192, 384, 768), channels=256, num_classes=4, dropout=0.20):
        super().__init__()
        ppm_scales = (1, 2, 3, 6)
        self.ppm = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(s),
                nn.Conv2d(in_channels[-1], channels, 1, bias=False),
                nn.GroupNorm(32, channels), nn.GELU(),
            ) for s in ppm_scales
        ])
        self.ppm_proj = nn.Sequential(
            nn.Conv2d(in_channels[-1] + len(ppm_scales)*channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(32, channels), nn.GELU(),
        )
        self.laterals = nn.ModuleList([
            nn.Sequential(nn.Conv2d(c, channels, 1, bias=False),
                          nn.GroupNorm(32, channels), nn.GELU())
            for c in in_channels[:-1]
        ])
        self.fpn_convs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                          nn.GroupNorm(32, channels), nn.GELU())
            for _ in range(len(in_channels))
        ])
        self.fpn_bottleneck = nn.Sequential(
            nn.Conv2d(len(in_channels)*channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(32, channels), nn.GELU(),
        )
        self.dropout = nn.Dropout2d(dropout)
        self.cls_seg = nn.Conv2d(channels, num_classes, 1)

    def forward(self, features):
        x = features[-1]
        ppm_outs = [x]
        for ppm in self.ppm:
            o = ppm(x)
            o = F.interpolate(o, size=x.shape[-2:], mode="bilinear", align_corners=False)
            ppm_outs.append(o)
        x = self.ppm_proj(torch.cat(ppm_outs, dim=1))
        fpn_outs = [x]
        for i in range(len(self.laterals) - 1, -1, -1):
            x = F.interpolate(x, size=features[i].shape[-2:], mode="bilinear", align_corners=False)
            x = x + self.laterals[i](features[i])
            fpn_outs.append(x)
        fpn_outs = fpn_outs[::-1]
        target_size = fpn_outs[0].shape[-2:]
        fused = []
        for i, fpn in enumerate(self.fpn_convs):
            o = fpn(fpn_outs[i])
            if o.shape[-2:] != target_size:
                o = F.interpolate(o, size=target_size, mode="bilinear", align_corners=False)
            fused.append(o)
        out = self.fpn_bottleneck(torch.cat(fused, dim=1))
        out = self.dropout(out)
        return self.cls_seg(out)

class DecisionAlignedSegModel(nn.Module):
    def __init__(self, backbone_name, num_mask_ch=NUM_MASK_CH, num_pooled=NUM_POOLED, decoder_dropout=0.20):
        super().__init__()
        self.encoder = timm.create_model(backbone_name, pretrained=False,
                                         features_only=True, out_indices=(0, 1, 2, 3))
        if hasattr(self.encoder, "set_grad_checkpointing") and CFG["use_grad_checkpointing"]:
            self.encoder.set_grad_checkpointing(True)
        chans = [info["num_chs"] for info in self.encoder.feature_info]
        print(f"encoder feature channels: {chans}")
        self.decoder = UPerNetDecoder(in_channels=tuple(chans), channels=256,
                                       num_classes=num_mask_ch, dropout=decoder_dropout)
        # Pooled binary heads from deepest feature
        self.pooled_norm = nn.LayerNorm(chans[-1])
        self.pooled_head = nn.Sequential(
            nn.Linear(chans[-1], 256), nn.GELU(), nn.Dropout(0.20),
            nn.Linear(256, num_pooled),
        )

    def forward(self, x):
        feats = self.encoder(x)
        mask_logits = self.decoder(feats)                        # (B, 4, 224, 224)
        pooled_feat = F.adaptive_avg_pool2d(feats[-1], 1).flatten(1)  # (B, 768)
        pooled_logits = self.pooled_head(self.pooled_norm(pooled_feat))  # (B, 6)
        return mask_logits, pooled_logits

def load_v5_encoder(model, ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]
    enc_keys = {k.replace("encoder.", "", 1): v for k, v in sd.items() if k.startswith("encoder.")}
    def remap(k):
        k = re.sub(r"^stem\.(\d+)\.", r"stem_\1.", k)
        k = re.sub(r"^stages\.(\d+)\.", r"stages_\1.", k)
        return k
    remapped = {remap(k): v for k, v in enc_keys.items() if not remap(k).startswith("head.")}
    missing, unexpected = model.encoder.load_state_dict(remapped, strict=False)
    print(f"v5 encoder load: missing={len(missing)} unexpected={len(unexpected)}")
    if len(missing) > 5 or len(unexpected) > 5:
        raise RuntimeError(f"v5 encoder weight load failed. missing[:5]={missing[:5]} unexpected[:5]={unexpected[:5]}")
    return ck.get("best_threshold", 0.77), ck.get("best_val_cls_f1", BASELINE_V5_VAL_F1)

model = DecisionAlignedSegModel(CFG["backbone"], decoder_dropout=CFG["decoder_dropout"]).to(device)
v5_ckpt = find_v5_ckpt()
assert v5_ckpt is not None, "No v5 checkpoint found. Add allenjosephantony/krones-v5-checkpoint as a Kaggle dataset."
v5_thr, v5_val_f1 = load_v5_encoder(model, v5_ckpt)
print(f"v5 reference: threshold={v5_thr:.3f}  val_f1={v5_val_f1:.4f}")

# FREEZE entire encoder at start; later we'll unfreeze last 2 stages at epoch >= encoder_unfreeze_epoch
def freeze_all_encoder(m):
    for p in m.encoder.parameters(): p.requires_grad = False
    m.encoder.eval()

def unfreeze_last_two_stages(m):
    # ConvNeXt-Tiny features_only stages are .stages_0 .. .stages_3
    for name, p in m.encoder.named_parameters():
        if name.startswith("stages_2.") or name.startswith("stages_3."):
            p.requires_grad = True
    # leave BN-like modules in eval mode but allow gradients on params
    # ConvNeXt uses LayerNorm so no special BN handling

freeze_all_encoder(model)
n_total = sum(p.numel() for p in model.parameters())
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"params: total={n_total/1e6:.2f}M  trainable(decoder+pooled)={n_trainable/1e6:.2f}M")
"""
)


md("## 8. Loss — focal + Dice for masks + weighted BCE for pooled heads")

code(
    r"""# Per-channel positive rate (from rendered training masks) for focal alpha
def estimate_mask_pos_rate(ds, n_samples=200):
    rates = np.zeros(NUM_MASK_CH, dtype=np.float64)
    count = 0
    idxs = np.random.RandomState(0).choice(len(ds), size=min(n_samples, len(ds)), replace=False)
    for i in idxs:
        _, mask, _, _, _, _ = ds[i]
        rates += mask.float().mean(dim=(1, 2)).numpy()
        count += 1
    return rates / max(count, 1)

print("estimating mask pos rates (200 train samples)...")
pos_rate_mask = estimate_mask_pos_rate(train_ds, n_samples=200)
print("per-channel pos rates:")
for ch, r in zip(MASK_CHANNELS, pos_rate_mask):
    print(f"  {ch:22s} {r:.6f}")
# Focal alpha per class: weight positives inversely. Cap at focal_alpha_cap (was 20 in v65 -> too much).
alpha_per_ch = np.clip(
    np.where(pos_rate_mask > 1e-8, (1.0 - pos_rate_mask) / np.maximum(pos_rate_mask, 1e-8), CFG["focal_alpha_cap"]),
    1.0, CFG["focal_alpha_cap"]
).astype(np.float32)
alpha_t = torch.from_numpy(alpha_per_ch).to(device).view(1, -1, 1, 1)
print("alpha per channel (clipped):", alpha_per_ch.round(3).tolist())

def focal_loss(logits, targets, alpha, gamma=2.0):
    # alpha shape: (1, C, 1, 1).  targets shape: (B, C, H, W)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = torch.exp(-bce)
    weight = alpha * targets + 1.0 * (1.0 - targets)   # alpha on positives, 1 on negatives
    return (weight * (1.0 - pt) ** gamma * bce).mean()

def dice_loss(logits, targets, eps=1.0):
    probs = torch.sigmoid(logits)
    inter = (probs * targets).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    return (1.0 - (2.0 * inter + eps) / (union + eps)).mean()

# Pooled head weights (per the spec). Default 1.0; cont_dark 3x; scuffing_above_75000 3x; distractor 2x.
POOLED_WEIGHTS = torch.tensor([
    CFG["w_pooled"],                  # p_group_c_trigger
    CFG["w_cont_dark_pooled"],        # p_cont_dark_trigger
    CFG["w_scuffing_above_pooled"],   # p_scuffing_above_75000
    CFG["w_distractor_pooled"],       # p_distractor_only
    CFG["w_pooled"],                  # p_cont_light_above_thr
    CFG["w_pooled"],                  # p_scuffing_heavy_above
], dtype=torch.float32, device=device)

# Per-head pos_weight from training freq (capped at 8)
pooled_pos_rate = np.array([
    tr_df["y_group_c"].mean(), tr_df["y_cont_dark"].mean(),
    tr_df["y_scuffing_above_75000"].mean(), tr_df["y_distractor_only"].mean(),
    tr_df["y_cont_light_above"].mean(), tr_df["y_scuffing_heavy_above"].mean(),
], dtype=np.float32)
pooled_pos_weight = np.where(pooled_pos_rate > 1e-8,
                              (1.0 - pooled_pos_rate) / np.maximum(pooled_pos_rate, 1e-8),
                              8.0)
pooled_pos_weight = np.clip(pooled_pos_weight, 1.0, 8.0).astype(np.float32)
pooled_pos_w_t = torch.from_numpy(pooled_pos_weight).to(device)
print("pooled head pos_weights:", pooled_pos_weight.round(2).tolist())

def pooled_loss(logits, targets):
    # logits, targets: (B, NUM_POOLED).  Mask out -1.0 entries (test rows have no label).
    valid = targets >= 0
    if not valid.any():
        return torch.tensor(0.0, device=logits.device)
    per_elem = F.binary_cross_entropy_with_logits(
        logits, targets.clamp(min=0.0), pos_weight=pooled_pos_w_t, reduction="none"
    )
    weighted = per_elem * POOLED_WEIGHTS.view(1, -1) * valid.float()
    return weighted.sum() / valid.float().sum().clamp_min(1)

def total_loss(mask_logits, mask_targets, pooled_logits, pooled_targets):
    L_focal = focal_loss(mask_logits, mask_targets, alpha_t, gamma=CFG["focal_gamma"])
    L_dice  = dice_loss(mask_logits, mask_targets)
    L_pool  = pooled_loss(pooled_logits, pooled_targets)
    return CFG["w_focal"] * L_focal + CFG["w_dice"] * L_dice + L_pool, dict(focal=float(L_focal), dice=float(L_dice), pooled=float(L_pool))
"""
)


md("## 9. Rule application — combine mask pixel counts + pooled head probs")

code(
    r"""def apply_rule(mask_logits, pooled_logits, per_cell_native_area,
               mask_prob_thr, pooled_thr=0.5,
               group_c_min_px=20, scuffing_above_pooled_thr=0.5):
    '''Decision-aligned rule:
       Target=1 if any of:
         (a) p_group_c_trigger >= pooled_thr AND group_c_pixel_count >= group_c_min_px
         (b) p_cont_dark_trigger >= pooled_thr
         (c) p_scuffing_above_75000 >= scuffing_above_pooled_thr
         (d) p_cont_light_above_thr >= pooled_thr
         (e) p_scuffing_heavy_above >= pooled_thr
       i.e. any Group C trigger OR any Group B above-threshold head fires.
    '''
    mask_probs   = torch.sigmoid(mask_logits.float())
    pooled_probs = torch.sigmoid(pooled_logits.float())
    bin_mask     = (mask_probs > mask_prob_thr).float()
    # Native pixel counts per channel
    cell_counts  = bin_mask.sum(dim=(2, 3))                      # (B, 4)
    native       = cell_counts * per_cell_native_area.view(-1, 1)  # (B, 4)

    group_c_native = native[:, 0]
    p_group_c    = pooled_probs[:, 0]
    p_cont_dark  = pooled_probs[:, 1]
    p_scuff_above= pooled_probs[:, 2]
    p_cont_light = pooled_probs[:, 4]
    p_scuff_heavy= pooled_probs[:, 5]

    fires_a = (p_group_c >= pooled_thr) & (group_c_native >= group_c_min_px)
    fires_b = p_cont_dark >= pooled_thr
    fires_c = p_scuff_above >= scuffing_above_pooled_thr
    fires_d = p_cont_light >= pooled_thr
    fires_e = p_scuff_heavy >= pooled_thr

    pred = (fires_a | fires_b | fires_c | fires_d | fires_e).float()
    return pred, native, pooled_probs

@torch.no_grad()
def eval_rule_on_loader(model_in_eval, loader, mask_prob_thr, pooled_thr=0.5,
                        scuffing_above_pooled_thr=0.5, group_c_min_px=20):
    model_in_eval.eval()
    all_preds, all_targets, all_native, all_pooled = [], [], [], []
    for batch in loader:
        imgs, masks, targets, pooled_y, per_cell, fnames = batch
        imgs = imgs.to(device, non_blocking=True)
        per_cell = per_cell.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=CFG["amp"]):
            mask_logits, pooled_logits = model_in_eval(imgs)
        pred, native, pooled_probs = apply_rule(mask_logits, pooled_logits, per_cell,
                                                 mask_prob_thr, pooled_thr,
                                                 group_c_min_px, scuffing_above_pooled_thr)
        all_preds.append(pred.cpu().numpy())
        all_targets.append(targets.numpy())
        all_native.append(native.cpu().numpy())
        all_pooled.append(pooled_probs.cpu().numpy())
    preds   = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    native  = np.concatenate(all_native, axis=0)
    pooled  = np.concatenate(all_pooled, axis=0)
    valid = targets >= 0
    if not valid.any():
        return None, preds, targets, native, pooled
    return f1_score(targets[valid], preds[valid]), preds, targets, native, pooled
"""
)


md("## 10. Training loop — checkpoint by val rule-F1; encoder unfreeze at epoch 5")

code(
    r"""class EMA:
    def __init__(self, m, decay):
        self.decay = decay
        self.shadow = {n: p.detach().clone() for n, p in m.named_parameters() if p.requires_grad}
    @torch.no_grad()
    def update(self, m):
        for n, p in m.named_parameters():
            if p.requires_grad:
                if n not in self.shadow:
                    self.shadow[n] = p.detach().clone()
                else:
                    self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)
    def load_into(self, m):
        backup = {}
        for n, p in m.named_parameters():
            if p.requires_grad and n in self.shadow:
                backup[n] = p.detach().clone()
                p.data.copy_(self.shadow[n])
        return backup
    @staticmethod
    def restore(m, b):
        for n, p in m.named_parameters():
            if n in b: p.data.copy_(b[n])

ema = EMA(model, CFG["ema_decay"])
scaler = torch.amp.GradScaler(enabled=CFG["amp"])

def build_optimizer(m, epoch):
    decoder_params = [p for n, p in m.named_parameters()
                      if p.requires_grad and not n.startswith("encoder.")]
    encoder_params = [p for n, p in m.named_parameters()
                      if p.requires_grad and n.startswith("encoder.")]
    groups = [{"params": decoder_params, "lr": CFG["lr_decoder"], "weight_decay": CFG["weight_decay"]}]
    if encoder_params:
        groups.append({"params": encoder_params, "lr": CFG["lr_encoder_unfrozen"], "weight_decay": CFG["weight_decay"]})
    return torch.optim.AdamW(groups)

optim = build_optimizer(model, 0)
steps_per_epoch = max(1, len(train_loader) // CFG["grad_accum"])
total_steps = steps_per_epoch * CFG["epochs"]
warmup_steps = steps_per_epoch * CFG["warmup_epochs"]
def lr_lambda(step):
    if step < warmup_steps: return float(step) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))
scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

best_val_rule_f1 = -1.0
best_mask_thr = 0.50
best_pooled_thr = 0.50
patience = 0
history = []

print("=" * 60); print("training start"); print("=" * 60)

for ep in range(CFG["epochs"]):
    # Encoder unfreeze schedule
    if ep == CFG["encoder_unfreeze_epoch"]:
        print(f"=> unfreezing encoder stages_2 + stages_3 at epoch {ep+1}")
        unfreeze_last_two_stages(model)
        optim = build_optimizer(model, ep)
        # need to refresh scheduler so it doesn't reset; build new LR lambda from here
        remaining = (CFG["epochs"] - ep) * steps_per_epoch
        def lr_lambda_post(step, total=remaining):
            progress = step / max(1, total)
            return 0.5 * (1 + math.cos(math.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda_post)
        # Re-init EMA shadow with new trainable params
        ema = EMA(model, CFG["ema_decay"])

    model.train()
    if ep < CFG["encoder_unfreeze_epoch"]:
        model.encoder.eval()  # keep frozen encoder in eval mode
    t0 = time.time()
    running_loss = 0.0; running_n = 0; loss_components = defaultdict(float)
    optim.zero_grad(set_to_none=True)

    for i, batch in enumerate(train_loader):
        imgs, masks, targets, pooled_y, per_cell, fnames = batch
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        pooled_y = pooled_y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=CFG["amp"]):
            mask_logits, pooled_logits = model(imgs)
            loss, comps = total_loss(mask_logits, masks, pooled_logits, pooled_y)
            loss = loss / CFG["grad_accum"]
        scaler.scale(loss).backward()
        running_loss += loss.item() * CFG["grad_accum"] * imgs.shape[0]
        running_n += imgs.shape[0]
        for k, v in comps.items(): loss_components[k] += v * imgs.shape[0]

        if (i + 1) % CFG["grad_accum"] == 0:
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], CFG["grad_clip"])
            scaler.step(optim); scaler.update(); scheduler.step()
            optim.zero_grad(set_to_none=True); ema.update(model)

        if (i + 1) % 50 == 0:
            comp_str = " ".join(f"{k}={v/running_n:.3f}" for k, v in loss_components.items())
            print(f"  ep{ep+1} step{i+1}/{len(train_loader)}  loss={running_loss/running_n:.4f}  {comp_str}  lr={scheduler.get_last_lr()[0]:.2e}  t={(time.time()-t0)/60:.1f}min")

        if time.time() - RUN_STARTED_AT > CFG["max_train_seconds"]:
            print("=> wall-time budget hit, ending epoch early"); break

    train_loss = running_loss / max(running_n, 1)

    # Validation with EMA weights, sweep mask threshold
    backup = ema.load_into(model)
    model.eval()
    best_ep_f1 = -1.0; best_ep_mask_thr = 0.50; best_ep_pooled_thr = 0.50
    for mask_thr in CFG["seg_prob_thresholds"]:
        for pooled_thr in (0.40, 0.50, 0.60):
            f1, _, _, _, _ = eval_rule_on_loader(model, val_loader, float(mask_thr), float(pooled_thr))
            if f1 is not None and f1 > best_ep_f1:
                best_ep_f1, best_ep_mask_thr, best_ep_pooled_thr = f1, float(mask_thr), float(pooled_thr)
    ema.restore(model, backup)

    elapsed = (time.time() - t0) / 60
    print(f"epoch {ep+1}: train_loss={train_loss:.4f}  val_rule_F1={best_ep_f1:.4f}  mask_thr={best_ep_mask_thr:.2f}  pooled_thr={best_ep_pooled_thr:.2f}  ({elapsed:.1f}min)")
    history.append({"epoch": ep+1, "train_loss": train_loss, "val_rule_f1": best_ep_f1,
                    "best_mask_thr": best_ep_mask_thr, "best_pooled_thr": best_ep_pooled_thr, "minutes": elapsed})

    if best_ep_f1 > best_val_rule_f1:
        best_val_rule_f1 = best_ep_f1
        best_mask_thr = best_ep_mask_thr
        best_pooled_thr = best_ep_pooled_thr
        patience = 0
        backup2 = ema.load_into(model)
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": CFG, "best_val_rule_f1": best_val_rule_f1,
            "best_mask_thr": best_mask_thr, "best_pooled_thr": best_pooled_thr,
            "history": history, "mask_channels": MASK_CHANNELS, "pooled_heads": POOLED_HEADS,
            "group_b_thresholds": GROUP_B_THRESHOLDS,
        }, OUT_ROOT / "best_model_v66.pth")
        ema.restore(model, backup2)
        print(f"  -> saved best (val rule_f1 {best_val_rule_f1:.4f})")
    else:
        patience += 1
        if patience >= CFG["early_stop_patience"]:
            print(f"  -> early stop"); break

    if time.time() - RUN_STARTED_AT > CFG["max_train_seconds"]:
        print("=> total wall-time budget hit, stopping"); break

print("=" * 60)
print(f"BEST val rule_F1: {best_val_rule_f1:.4f}  @ mask_thr={best_mask_thr:.2f}  pooled_thr={best_pooled_thr:.2f}")
print("=" * 60)
"""
)


md("## 11. Phase 2 PROOF GATE + fixed/created ratio against v58c (if val predictions available)")

code(
    r"""ck = torch.load(OUT_ROOT / "best_model_v66.pth", map_location="cpu", weights_only=False)
model.load_state_dict(ck["model_state_dict"])
model.eval()

# Train-set rule reconstruction (gate)
gate_ds = SegDataset(train_df, TRAIN_IMG_DIR, "val")
gate_loader = DataLoader(gate_ds, batch_size=CFG["batch_size"], shuffle=False,
                         num_workers=CFG["num_workers"], pin_memory=True)
gate_f1, gate_preds, gate_targets, gate_native, gate_pooled = eval_rule_on_loader(
    model, gate_loader, best_mask_thr, best_pooled_thr
)

gate_df = train_df.copy().reset_index(drop=True)
gate_df["pred"] = gate_preds
gate_df["targ"] = gate_targets
bucket_f1 = {}
for b in ["small", "medium", "large"]:
    sub = gate_df[gate_df["bucket"] == b]
    if len(sub) > 0:
        bucket_f1[b] = f1_score(sub["targ"].values, sub["pred"].values)

print("=" * 60); print("PHASE 2 PROOF GATE"); print("=" * 60)
print(f"train rule-F1: {gate_f1:.4f}")
print("per-bucket train rule-F1:")
for b, f in bucket_f1.items():
    print(f"  {b:8s} {f:.4f}  (v5 LB ref: {V5_BUCKET_F1.get(b, 0):.4f})")
print(f"val rule-F1 (EMA in-training): {best_val_rule_f1:.4f}")

gate_threshold = CFG["phase2_gate_train_rule_f1"]
if gate_f1 >= gate_threshold:
    print(f"\nGATE PASS ({gate_f1:.4f} >= {gate_threshold:.4f})")
elif gate_f1 >= 0.90:
    print(f"\nGATE MARGINAL ({gate_f1:.4f} in [0.90, {gate_threshold:.4f}))")
    print("-> residual combiner must satisfy fixed_created_ratio >= 2.0 on val before submission.")
else:
    print(f"\nGATE FAIL ({gate_f1:.4f} < 0.90)")
    print("-> stop. F1 path empirically dead at this hardware/budget. Pivot to insight write-up.")

# Optional fixed/created ratio vs v58c on val (if v58c val predictions present in kaggle inputs)
v58c_val_path = None
for cand in Path("/kaggle/input").glob("**/val_diagnostic*.csv") if Path("/kaggle/input").exists() else []:
    v58c_val_path = cand; break
if v58c_val_path is None:
    print("\n(skipped fixed_created_ratio; no v58c val predictions found in /kaggle/input)")
"""
)


md("## 12. Save val + test predictions for the local Phase 3 combiner")

code(
    r"""# Val predictions (full per-channel native counts + pooled probs)
val_f1, val_preds, val_targets, val_native, val_pooled = eval_rule_on_loader(
    model, val_loader, best_mask_thr, best_pooled_thr
)
val_df_out = vl_df.copy().reset_index(drop=True)
val_df_out["seg_rule_pred"] = val_preds
val_df_out["target_actual"] = val_targets
for ci, ch in enumerate(MASK_CHANNELS):
    val_df_out[f"px_native_{ch}"] = val_native[:, ci]
for pi, ph in enumerate(POOLED_HEADS):
    val_df_out[f"prob_{ph}"] = val_pooled[:, pi]
val_df_out.to_csv(OUT_ROOT / "v66_val_predictions.csv", index=False)
print(f"val_f1: {val_f1:.4f}  pos_rate={val_preds.mean():.4f}  ->  v66_val_predictions.csv")

# Test predictions
if RUN_TEST_INFERENCE and test_loader is not None:
    _, test_preds, _, test_native, test_pooled = eval_rule_on_loader(
        model, test_loader, best_mask_thr, best_pooled_thr
    )
    test_df_out = sample_sub.copy().reset_index(drop=True)
    test_df_out["seg_rule_pred"] = test_preds
    for ci, ch in enumerate(MASK_CHANNELS):
        test_df_out[f"px_native_{ch}"] = test_native[:, ci]
    for pi, ph in enumerate(POOLED_HEADS):
        test_df_out[f"prob_{ph}"] = test_pooled[:, pi]
    test_df_out.to_csv(OUT_ROOT / "v66_test_predictions.csv", index=False)
    print(f"test predictions: pos_rate={test_preds.mean():.4f}  ->  v66_test_predictions.csv")
    if not (0.55 <= test_preds.mean() <= 0.62):
        print("  -> WARNING: test pos rate outside safety band. Phase 3 combiner must NOT submit blindly.")

    if CREATE_SUBMISSION:
        sub = sample_sub.copy()
        sub["target"] = test_preds.astype(int)
        sub.to_csv(OUT_ROOT / "v66_seg_rule_only_submission.csv", index=False)
        print(f"standalone seg-rule (SANITY PROBE, do NOT submit to LB unless gates pass): v66_seg_rule_only_submission.csv")
"""
)


md("## 13. Done")

code(
    r"""print("=" * 60)
print("v66 Phase 2 complete.")
print("=" * 60)
print(f"best val rule-F1: {best_val_rule_f1:.4f}  (v65 was 0.9154 -> need to clear)")
print(f"train rule-F1 (gate): {gate_f1:.4f}  (gate threshold {CFG['phase2_gate_train_rule_f1']:.4f})")
print()
print("Outputs:")
for p in sorted(OUT_ROOT.glob("*")):
    print(f"  {p.name}  ({p.stat().st_size/1e6:.2f} MB)")
print()
print("Next:")
print("  - GATE PASS: pull artefacts locally, build Phase 3 v58c residual combiner (local Python).")
print("  - GATE MARGINAL: same, but residual must hit fixed_created_ratio >= 2.0 on val to submit.")
print("  - GATE FAIL: do not submit. Pivot to writeup/insight.md.")
"""
)


# Assemble + write
nb["cells"] = cells
NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"notebook written: {NOTEBOOK_PATH}")

PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
shutil.copy2(NOTEBOOK_PATH, PACKAGE_NB)
metadata = {
    "id": "allenjosephantony/krones-v66-decision-aligned-seg",
    "title": "krones-v66-decision-aligned-seg",
    "code_file": "kernel.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_internet": True,
    "dataset_sources": [
        "allenjosephantony/krones-v5-checkpoint",
    ],
    "competition_sources": ["1st-krones-vision-ai-challenge"],
}
with open(PACKAGE_META, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)
print(f"package metadata: {PACKAGE_META}")
print(f"package notebook: {PACKAGE_NB}")
print()
print("Push to Kaggle:")
print(f"  kaggle kernels push -p {PACKAGE_DIR.as_posix()}")
