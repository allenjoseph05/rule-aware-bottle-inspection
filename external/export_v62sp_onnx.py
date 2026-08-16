"""Export v62-sp champion (LB 0.97066) to ONNX, v5-style baked preprocessing.

Inputs : image_u8 (1,1,640,640) uint8 ROI crop, bv (1,3) float32 bucket one-hot,
         rg (1,2) float32 = (roi_w/640, roi_h/640)
Outputs: target_prob(1,1), group_probs(1,4), c_probs(1,16), bp_probs(1,6) [sigmoid],
         b_logarea(1,6) [RAW — the rule applies logistic((la-LT)/tau) itself]
Run: ~/krones_venv/bin/python export_v62sp_onnx.py   (A5000 CPU, ~5 min)
"""
import os
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import timm

CKPT = Path.home() / "workspace" / "best_model_v62_80pct.pth"
OUT = Path.home() / "workspace" / "v62_onnx"; OUT.mkdir(exist_ok=True)

class AugSmartModel(nn.Module):
    def __init__(self, backbone_name, n_group=4, n_c=16, n_b=6, topk_frac=0.08):
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
        self.group_c_head = nn.Linear(256, n_c)
        self.group_b_presence_head = nn.Linear(256, n_b)
        self.group_b_logarea_head = nn.Linear(256, n_b)
    def topk_pool(self, l):
        flat = l.flatten(1)
        k = max(1, int(round(flat.shape[1] * self.topk_frac)))
        return torch.topk(flat, k=k, dim=1).values.mean(dim=1, keepdim=True)
    def forward(self, x, bv, rg):
        f = self.encoder(x)
        pooled = F.adaptive_avg_pool2d(f, 1).flatten(1)
        tt = self.topk_pool(self.trigger_map(f)); dt = self.topk_pool(self.distractor_map(f))
        z = self.shared(self.norm(torch.cat([pooled, tt, dt, bv, rg], 1)))
        return (self.target_head(z), self.group_head(z), self.group_c_head(z),
                self.group_b_presence_head(z), self.group_b_logarea_head(z))

class Wrap(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    def forward(self, image_u8, bv, rg):
        x = image_u8.float() / 255.0
        x = x.repeat(1, 3, 1, 1)
        x = (x - self.mean) / self.std
        t, g, c, bp, bl = self.m(x, bv, rg)
        return torch.sigmoid(t), torch.sigmoid(g), torch.sigmoid(c), torch.sigmoid(bp), bl

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
sd = ck["model_state_dict"]
core = AugSmartModel(ck["config"]["backbone"], n_group=sd["group_head.weight"].shape[0])
missing, unexpected = core.load_state_dict(sd, strict=False)
print(f"load: missing={len(missing)} unexpected={len(unexpected)}")
assert not missing
core.eval()
model = Wrap(core).eval()

u8 = torch.randint(0, 256, (1, 1, 640, 640), dtype=torch.uint8)
bv = torch.tensor([[0., 1., 0.]]); rg = torch.tensor([[0.84, 0.84]])
with torch.no_grad():
    outs = model(u8, bv, rg)
print("fwd:", [tuple(o.shape) for o in outs])

p = OUT / "v62sp.onnx"
torch.onnx.export(model, (u8, bv, rg), str(p),
                  input_names=["image_u8", "bv", "rg"],
                  output_names=["target_prob", "group_probs", "c_probs", "bp_probs", "b_logarea"],
                  opset_version=17, do_constant_folding=True)
print(f"saved {p} ({p.stat().st_size/1e6:.1f} MB)")

import onnxruntime as ort
s = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
maxd = 0.0
for _ in range(3):
    u = torch.randint(0, 256, (1, 1, 640, 640), dtype=torch.uint8)
    b = torch.eye(3)[torch.randint(0, 3, (1,))].float(); r = torch.rand(1, 2) * 0.3 + 0.7
    with torch.no_grad():
        t_outs = model(u, b, r)
    o_outs = s.run(None, {"image_u8": u.numpy(), "bv": b.numpy(), "rg": r.numpy()})
    for a, o in zip(t_outs, o_outs):
        maxd = max(maxd, float(np.abs(a.numpy() - o).max()))
print(f"torch-vs-ort max diff: {maxd:.6f}")
assert maxd < 2e-3

import onnx
from onnxconverter_common import float16
m16 = float16.convert_float_to_float16(onnx.load(str(p)), keep_io_types=True)
p16 = OUT / "v62sp_fp16.onnx"
onnx.save(m16, str(p16))
s16 = ort.InferenceSession(str(p16), providers=["CPUExecutionProvider"])
maxd16 = 0.0
for _ in range(3):
    u = np.random.randint(0, 256, (1, 1, 640, 640), dtype=np.uint8)
    b = np.eye(3, dtype=np.float32)[[1]]; r = np.array([[0.84, 0.84]], np.float32)
    o32 = s.run(None, {"image_u8": u, "bv": b, "rg": r})
    o16 = s16.run(None, {"image_u8": u, "bv": b, "rg": r})
    for a, o in zip(o32, o16):
        maxd16 = max(maxd16, float(np.abs(a - o).max()))
print(f"fp32-vs-fp16 max diff: {maxd16:.6f}  ({p16.stat().st_size/1e6:.1f} MB)")
print("EXPORT DONE")
