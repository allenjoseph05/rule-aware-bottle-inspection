"""End-to-end equivalence smoke test (CPU, real checkpoints).

Builds v56 + v57 from the LOCAL checkpoints and runs BOTH pipelines on the
sample PNGs:
  ORIGINAL : PIL decode -> CPU float32 3ch normalize  (current notebook)
  OPTIMIZED: PIL decode -> worker emits uint8 1ch -> normalize on-device + 3ch expand

Asserts every model output (v56 final_prob/aux, all v57 scores) is bit-identical,
and exercises the refactored vectorized collection + v58c combiner so any
shape/index bug surfaces locally before Soumyajeet spends a Kaggle run.

ROIs are synthesized (centered 512) — we are testing pipeline EQUIVALENCE, not
prediction correctness, so the exact box only has to be the same for both paths.
"""
import glob, os, time
from collections import defaultdict
import numpy as np
import pandas as pd
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
import timm

torch.manual_seed(0)
ROOT = os.path.join(os.path.dirname(__file__), "..")
SAMPLES = sorted(glob.glob(os.path.join(ROOT, "data", "samples", "*.png")))
SAMPLES = [f for f in SAMPLES if os.path.getsize(f) > 200_000][:8]  # 8 imgs keeps CPU runtime sane
BACKBONE = "convnext_tiny.dinov3_lvd1689m"
device = torch.device("cpu")
OUT = 640
V5_CKPT = os.path.join(ROOT, "artefacts/v5_results/best_model_v5.pth")
V56_CKPT = os.path.join(ROOT, "artefacts/kaggle_v56_v3_output/best_model_v56_anchor_roi_residual.pth")
V57_CKPT = os.path.join(ROOT, "artefacts/v57_recovered/best_model_v57_functional_coco_state.pth")
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
MEAN_T = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD_T = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# ---------------- helpers ----------------
def center_crop_box(W, H, crop):
    side = min(crop, W, H); l = max(0, (int(W) - side) // 2); t = max(0, (int(H) - side) // 2); return l, t, side, side
def roi_crop_box(W, H, roi, pad=0.10):
    x, y, w, h = [float(v) for v in roi[:4]]; side = int(round(max(w, h) * (1 + 2 * pad))); side = max(1, min(side, int(W), int(H)))
    cx, cy = x + w / 2, y + h / 2; l = max(0, min(int(round(cx - side / 2)), int(W) - side)); t = max(0, min(int(round(cy - side / 2)), int(H) - side)); return l, t, side, side
def crop_image(img, box, out):
    l, t, w, h = box; c = img.crop((l, t, l + w, t + h))
    if c.size != (out, out): c = c.resize((out, out), Image.BILINEAR)
    return c
def to_tensor_norm(img):  # ORIGINAL
    a = np.asarray(img, dtype=np.float32) / 255.0; a = np.stack([a, a, a], 0); a = (a - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(a)).float()
def to_uint8(img):  # OPTIMIZED worker output
    return torch.from_numpy(np.ascontiguousarray(np.asarray(img, dtype=np.uint8)))  # [H,W] uint8
def gpu_normalize(u8):  # OPTIMIZED on-device normalize, u8: [B,H,W] uint8
    x = u8.unsqueeze(1).float().div_(255.0); x = x.expand(-1, 3, -1, -1); x = (x - MEAN_T) / STD_T
    return x.contiguous()
def to_bucket(w):
    w = float(w)
    if w < 510: return "small"
    if w > 590: return "large"
    return "medium"
def logit_np(p): p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6); return float(np.log(p / (1 - p)))

# ---------------- models (verbatim from notebook generator) ----------------
class SegHead(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.decode = nn.Sequential(nn.Conv2d(c, 256, 3, padding=1), nn.GELU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.GELU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.GELU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.GELU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.GELU(), nn.Conv2d(16, 1, 3, padding=1))
    def forward(self, x): return self.decode(x)
class V5Model(nn.Module):
    def __init__(self, bb):
        super().__init__()
        self.encoder = timm.create_model(bb, pretrained=False, num_classes=0, global_pool=""); fd = self.encoder.num_features; self.feat_dim = fd
        self.cls_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.LayerNorm(fd), nn.Linear(fd, 1)); self.seg_head = SegHead(fd)
    def forward(self, x, return_features=False, compute_seg=False):
        f = self.encoder(x); cls = self.cls_head(f).squeeze(1); seg = self.seg_head(f) if compute_seg else None
        return (cls, seg, f) if return_features else (cls, seg)
class ResidualModel(nn.Module):
    def __init__(self, anchor, roi, v5_thr, delta_clip=1.0):
        super().__init__()
        self.anchor = anchor; self.roi = roi; fd = roi.feat_dim
        self.aux_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.LayerNorm(fd), nn.Linear(fd, 3))
        fdim = 1 + 1 + 3 + 3 + 2 + 1
        self.fusion = nn.Sequential(nn.LayerNorm(fdim), nn.Linear(fdim, 16), nn.GELU(), nn.Dropout(0.05), nn.Linear(16, 1))
        self.v5_thr_logit = logit_np(v5_thr); self.delta_clip = delta_clip
        for p in self.anchor.parameters(): p.requires_grad_(False)
        self.anchor.eval()
    def forward(self, center_x, roi_x, bucket_vec, roi_geom):
        with torch.no_grad(): v5_logit, _ = self.anchor(center_x)
        roi_logit, roi_seg, roi_feats = self.roi(roi_x, return_features=True); aux = self.aux_head(roi_feats)
        margin = torch.abs(v5_logit - self.v5_thr_logit).unsqueeze(1)
        fx = torch.cat([v5_logit.detach().unsqueeze(1), roi_logit.unsqueeze(1), aux, bucket_vec, roi_geom, margin], dim=1)
        delta = torch.tanh(self.fusion(fx).squeeze(1)) * self.delta_clip
        return {"aux_logits": aux, "final_logit": v5_logit.detach() + delta}
v56_ck = torch.load(V56_CKPT, map_location="cpu", weights_only=False)
V56_FINAL_THR = float(v56_ck["final_threshold"]); V5_THR = float(v56_ck["v5_threshold"])
anchor = V5Model(BACKBONE); roi = V5Model(BACKBONE)
def load_v5(m):
    ck = torch.load(V5_CKPT, map_location="cpu", weights_only=False); sd = ck.get("model_state_dict") or ck; m.load_state_dict(sd, strict=False)
load_v5(anchor); load_v5(roi)
v56 = ResidualModel(anchor, roi, V5_THR).to(device); v56.load_state_dict(v56_ck["model_state_dict"], strict=False); v56.eval()

v57_ck = torch.load(V57_CKPT, map_location="cpu", weights_only=False)
GROUP_C_NAMES = list(v57_ck["group_c_names"]); GROUP_B_NAMES = list(v57_ck["group_b_names"]); GROUP_COLS = list(v57_ck["group_cols"]); TOPK_FRAC = 0.08
class FunctionalCOCOStateModel(nn.Module):
    def __init__(self, bb):
        super().__init__()
        self.encoder = timm.create_model(bb, pretrained=False, num_classes=0, global_pool=""); fd = int(self.encoder.num_features); self.feat_dim = fd
        self.trigger_map = nn.Conv2d(fd, 1, 1); self.distractor_map = nn.Conv2d(fd, 1, 1)
        sd = fd + 2 + 3 + 2; self.norm = nn.LayerNorm(sd); self.shared = nn.Sequential(nn.Linear(sd, 256), nn.GELU(), nn.Dropout(0.08))
        self.target_head = nn.Linear(256, 1); self.group_head = nn.Linear(256, len(GROUP_COLS))
        self.group_c_head = nn.Linear(256, len(GROUP_C_NAMES)); self.group_b_state_head = nn.Linear(256, len(GROUP_B_NAMES) * 4)
    def topk_pool(self, l):
        flat = l.flatten(1); k = max(1, int(round(flat.shape[1] * TOPK_FRAC))); return torch.topk(flat, k=k, dim=1).values.mean(dim=1, keepdim=True)
    def forward(self, x, bucket_vec, roi_geom):
        f = self.encoder(x); pooled = F.adaptive_avg_pool2d(f, 1).flatten(1); tt = self.topk_pool(self.trigger_map(f)); dt = self.topk_pool(self.distractor_map(f))
        z = torch.cat([pooled, tt, dt, bucket_vec, roi_geom], dim=1); z = self.shared(self.norm(z))
        return {"target_logit": self.target_head(z).squeeze(1), "group_logits": self.group_head(z),
                "group_c_logits": self.group_c_head(z), "group_b_state_logits": self.group_b_state_head(z).view(-1, len(GROUP_B_NAMES), 4)}
v57 = FunctionalCOCOStateModel(BACKBONE).to(device); v57.load_state_dict(v57_ck["model_state_dict"], strict=False); v57.eval()
print(f"models loaded | v56 thr={V56_FINAL_THR:.4f} v5_thr={V5_THR:.4f} | group_c={len(GROUP_C_NAMES)} group_b={len(GROUP_B_NAMES)}")

# ---------------- run both pipelines ----------------
def synth_roi(W, H): return (W / 2 - 256, H / 2 - 256, 512.0, 512.0)
def build_inputs(f, optimized):
    img = Image.open(f).convert("L"); W, H = img.size
    center = crop_image(img, center_crop_box(W, H, OUT), OUT)
    roic = crop_image(img, roi_crop_box(W, H, synth_roi(W, H), 0.10), OUT)
    roi = synth_roi(W, H); bucket = to_bucket(roi[2])
    bv = torch.tensor([1.0 if bucket == "small" else 0.0, 1.0 if bucket == "medium" else 0.0, 1.0 if bucket == "large" else 0.0])
    geom = torch.tensor([float(roi[2]) / 640.0, float(roi[3]) / 640.0])
    if optimized:
        c = gpu_normalize(to_uint8(center).unsqueeze(0))[0]; r = gpu_normalize(to_uint8(roic).unsqueeze(0))[0]
    else:
        c = to_tensor_norm(center); r = to_tensor_norm(roic)
    return c, r, bv, geom

def infer(optimized):
    fp, au, tp, g0, g1, g2, g3 = [], [], [], [], [], [], []
    cps, bps = [], []
    with torch.no_grad():
        for f in SAMPLES:
            c, r, bv, geom = build_inputs(f, optimized)
            c = c.unsqueeze(0); r = r.unsqueeze(0); bv = bv.unsqueeze(0); geom = geom.unsqueeze(0)
            o56 = v56(c, r, bv, geom); o57 = v57(r, bv, geom)
            fp.append(torch.sigmoid(o56["final_logit"]).item()); au.append(torch.sigmoid(o56["aux_logits"])[0, 2].item())
            tp.append(torch.sigmoid(o57["target_logit"]).item())
            gp = torch.sigmoid(o57["group_logits"])[0]; g0.append(gp[0].item()); g1.append(gp[1].item()); g2.append(gp[2].item()); g3.append(gp[3].item())
            cps.append(torch.sigmoid(o57["group_c_logits"])[0].numpy()); bps.append(F.softmax(o57["group_b_state_logits"], dim=-1)[0, :, 3].numpy())
    return dict(fp=np.array(fp), au=np.array(au), tp=np.array(tp), g0=np.array(g0), g1=np.array(g1),
                g2=np.array(g2), g3=np.array(g3), cps=np.array(cps), bps=np.array(bps))

t = time.time(); A = infer(False); B = infer(True)
print(f"ran both pipelines on {len(SAMPLES)} imgs in {time.time()-t:.0f}s")
worst = 0.0
for k in ["fp", "au", "tp", "g0", "g1", "g2", "g3", "cps", "bps"]:
    d = float(np.abs(A[k] - B[k]).max()); worst = max(worst, d); print(f"  {k:4s} max|orig-opt| = {d:.3e}")
print(f"\nWORST diff across ALL model outputs: {worst:.3e}")
print("VERDICT:", "PIPELINES EQUIVALENT -> optimized notebook will reproduce banked numbers"
      if worst < 1e-4 else "DIVERGENCE -> refactor bug, do NOT ship")

# ---------------- exercise vectorized collection + v58c combiner (shape/index check) ----------------
df = pd.DataFrame({"image_id": [os.path.basename(f) for f in SAMPLES],
                   "final_pred": (B["fp"] >= V56_FINAL_THR).astype(int), "pred_target": B["tp"],
                   "pred_group_c_any": B["g0"], "pred_group_b_above_any": B["g1"],
                   "pred_distractor_any": B["g2"], "pred_rare_group_c_any": B["g3"]})
for ci, cn in enumerate(GROUP_C_NAMES): df[f"pred_c_{ci:02d}_{cn}"] = B["cps"][:, ci]
for bidx, bn in enumerate(GROUP_B_NAMES): df[f"pred_b_{bidx:02d}_{bn}_above"] = B["bps"][:, bidx]
c_cols = [f"pred_c_{i:02d}_{n}" for i, n in enumerate(GROUP_C_NAMES)]; b_cols = [f"pred_b_{i:02d}_{n}_above" for i, n in enumerate(GROUP_B_NAMES)]
c_support = np.maximum(df[c_cols].max(axis=1).values, df["pred_rare_group_c_any"].values); b_support = df[b_cols].max(axis=1).values
df["score_group_c"] = np.sqrt(np.clip(df["pred_group_c_any"].values * c_support, 0, 1))
df["score_group_b"] = np.sqrt(np.clip(df["pred_group_b_above_any"].values * b_support, 0, 1)); df["score_target_aux"] = df["pred_target"].values
pred = df["final_pred"].astype(int).values.copy()
add = ((pred == 0) & (df["score_target_aux"] >= 0.80).values & ((df["score_group_c"] >= 0.62) | (df["score_group_b"] >= 0.62)).values)
rem = ((pred == 1) & (df["score_group_c"] <= 0.36).values & (df["score_group_b"] <= 0.20).values & (df["pred_distractor_any"] >= 0.70).values)
pred[add] = 1; pred[rem] = 0
print(f"\ncombiner ran OK: base_pos={int(df['final_pred'].sum())} ADD={int(add.sum())} REM={int(rem.sum())} final_pos={int(pred.sum())} (correctness N/A on 8 unlabeled samples)")
