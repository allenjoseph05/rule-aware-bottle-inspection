"""Properly convert v5.onnx (fp32, uint8 input) -> fp16 and validate numerics on CPU.

The dataset's existing v5_fp16.onnx is broken (Cast output type mismatch, ORT can't
load it). This redoes the conversion with onnxconverter_common keep_io_types=True
and validates fp32-vs-fp16 logits on real train images with v5's exact preprocessing.
Runs entirely on CPU (A5000 GPU belongs to dll0402's job — untouched).
"""
import os, glob, time
import numpy as np
import onnx
from onnxconverter_common import float16

SRC = os.path.expanduser("~/workspace/persist/v5.onnx")
DST = os.path.expanduser("~/workspace/persist/v5_fp16_fixed.onnx")

m = onnx.load(SRC)

ops = {}
def scan(graph):
    for n in graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
        for a in n.attribute:
            if a.type == onnx.AttributeProto.GRAPH:
                scan(a.g)
            elif a.type == onnx.AttributeProto.GRAPHS:
                for g in a.graphs:
                    scan(g)
scan(m.graph)
print("graph ops:", sorted(ops.items()))
has_controlflow = any(k in ops for k in ("Loop", "If", "Scan"))
print("control flow present:", has_controlflow)

m16 = float16.convert_float_to_float16(m, keep_io_types=True)
onnx.save(m16, DST)
print(f"saved {DST} ({os.path.getsize(DST)/1e6:.1f} MB)")

import onnxruntime as ort
s32 = ort.InferenceSession(SRC, providers=["CPUExecutionProvider"])
s16 = ort.InferenceSession(DST, providers=["CPUExecutionProvider"])
i32, i16 = s32.get_inputs()[0], s16.get_inputs()[0]
print("in32:", i32.type, i32.shape, "| in16:", i16.type, i16.shape)
o32, o16 = s32.get_outputs()[0], s16.get_outputs()[0]
print("out32:", o32.type, "| out16:", o16.type)

from PIL import Image
files = sorted(glob.glob(os.path.expanduser("~/workspace/data/train_images/*.png")))[:100]
print("validation images:", len(files))
CROP = 640
def prep(p):
    g = np.asarray(Image.open(p).convert("L"))
    H, W = g.shape
    s = min(CROP, W, H); l = (W - s) // 2; t = (H - s) // 2
    return np.ascontiguousarray(g[t:t + s, l:l + s])[None, None]

lg32, lg16 = [], []
t0 = time.time()
for k, f in enumerate(files):
    x = prep(f)
    lg32.append(float(np.asarray(s32.run(None, {i32.name: x})[0]).reshape(-1)[0]))
    lg16.append(float(np.asarray(s16.run(None, {i16.name: x})[0]).reshape(-1)[0]))
    if (k + 1) % 20 == 0:
        print(f"  {k+1}/{len(files)}  ({time.time()-t0:.0f}s)", flush=True)

lg32 = np.array(lg32); lg16 = np.array(lg16)
p32 = 1 / (1 + np.exp(-lg32)); p16 = 1 / (1 + np.exp(-lg16))
flips = int(((p32 >= 0.77) != (p16 >= 0.77)).sum())
dp = np.abs(p32 - p16)
print(f"n={len(files)}  flips={flips}  max|dp|={dp.max():.6f}  mean|dp|={dp.mean():.8f}  t={time.time()-t0:.0f}s")
print("VERDICT:", "GOOD" if flips == 0 and dp.max() < 0.02 else "CHECK")
