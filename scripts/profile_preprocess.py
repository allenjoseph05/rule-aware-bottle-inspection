"""Line-by-line CPU-side profiler for the v58c live-inference preprocessing.

Reproduces EXACTLY the per-image work in notebooks/_build_v58c_live_inference.py
(TestDS.__getitem__ + helpers) and times every stage on the real sample PNGs.

Goal: find which CPU stage dominates (file-read / decode / crop / resize /
to_tensor_norm), prove cv2-decode == PIL-decode pixel-exact (lossless PNG), and
quantify the wasted work in the 3-channel-replicate + float32 normalize + H2D.

Runs locally, no GPU needed — the bottleneck is CPU/IO by hypothesis.
"""
import glob, os, time, io
import numpy as np
from PIL import Image
import cv2
import torch

SAMPLES = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "data", "samples", "*.png")))
SAMPLES = [f for f in SAMPLES if os.path.getsize(f) > 200_000]  # drop the few tiny/odd ones; keep realistic ~580KB PNGs
OUT = 640
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]

print(f"profiling on {len(SAMPLES)} real PNGs (>200KB), OUT={OUT}\n")

# ---- helpers copied verbatim from the notebook generator ----
def center_crop_box(W, H, crop):
    side = min(crop, W, H); l = max(0, (int(W) - side) // 2); t = max(0, (int(H) - side) // 2)
    return l, t, side, side

def roi_crop_box(W, H, roi, pad=0.10):
    x, y, w, h = [float(v) for v in roi[:4]]
    side = int(round(max(w, h) * (1 + 2 * pad))); side = max(1, min(side, int(W), int(H)))
    cx, cy = x + w / 2, y + h / 2
    l = max(0, min(int(round(cx - side / 2)), int(W) - side)); t = max(0, min(int(round(cy - side / 2)), int(H) - side))
    return l, t, side, side

def crop_image(img, box, out):
    l, t, w, h = box; c = img.crop((l, t, l + w, t + h))
    if c.size != (out, out): c = c.resize((out, out), Image.BILINEAR)
    return c

def to_tensor_norm(img):
    a = np.asarray(img, dtype=np.float32) / 255.0; a = np.stack([a, a, a], 0); a = (a - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(a)).float()

# synthetic ROI: a centered 512x512 box (typical). Cost of crop/resize is box-size driven, not position.
def synth_roi(W, H):
    return (W / 2 - 256, H / 2 - 256, 512.0, 512.0)

def timeit(fn, n_rep=5):
    # warm up
    for f in SAMPLES[:3]:
        fn(f)
    best_total = None
    for _ in range(n_rep):
        t0 = time.perf_counter()
        for f in SAMPLES:
            fn(f)
        dt = time.perf_counter() - t0
        best_total = dt if best_total is None else min(best_total, dt)
    return best_total / len(SAMPLES) * 1000.0  # ms per image (best of n_rep)

# =====================================================================
# STAGE 0 — pure file read (bytes off disk, no decode)
# =====================================================================
def s_fileread(f):
    with open(f, "rb") as fh:
        return fh.read()
print(f"[0] file read (bytes only)              : {timeit(s_fileread):7.2f} ms/img")

# =====================================================================
# STAGE 1 — DECODE: PIL vs cv2 (file path) vs cv2 (from bytes)
# =====================================================================
def s_pil_decode(f):
    img = Image.open(f).convert("L"); img.load(); return img  # .load() forces actual decode
def s_cv2_decode_path(f):
    return cv2.imread(f, cv2.IMREAD_GRAYSCALE)
_bcache = {f: open(f, "rb").read() for f in SAMPLES}
def s_cv2_decode_bytes(f):
    return cv2.imdecode(np.frombuffer(_bcache[f], np.uint8), cv2.IMREAD_GRAYSCALE)
def s_pil_decode_bytes(f):
    img = Image.open(io.BytesIO(_bcache[f])).convert("L"); img.load(); return img

pil_ms = timeit(s_pil_decode)
cv2p_ms = timeit(s_cv2_decode_path)
cv2b_ms = timeit(s_cv2_decode_bytes)
pilb_ms = timeit(s_pil_decode_bytes)
print(f"[1] PIL  decode (open+convert L+load)   : {pil_ms:7.2f} ms/img")
print(f"[1] PIL  decode (from cached bytes)     : {pilb_ms:7.2f} ms/img")
print(f"[1] cv2  decode (imread path)           : {cv2p_ms:7.2f} ms/img   speedup x{pil_ms/cv2p_ms:.2f}")
print(f"[1] cv2  decode (imdecode bytes)        : {cv2b_ms:7.2f} ms/img   speedup x{pil_ms/cv2b_ms:.2f}")

# =====================================================================
# STAGE 1b — PIXEL EXACTNESS: is cv2 grayscale == PIL convert('L')?
# (PNG is lossless; if equal we can swap decode with ZERO output change)
# =====================================================================
max_abs = 0; n_exact = 0
for f in SAMPLES:
    a = np.asarray(Image.open(f).convert("L"))
    b = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    d = int(np.abs(a.astype(int) - b.astype(int)).max())
    max_abs = max(max_abs, d); n_exact += (d == 0)
print(f"[1b] cv2 vs PIL grayscale decode: {n_exact}/{len(SAMPLES)} pixel-exact, max abs diff = {max_abs}")

# =====================================================================
# STAGE 2 — crop + resize (PIL) ; both center(640) and roi crops
# =====================================================================
_pilcache = {f: Image.open(f).convert("L") for f in SAMPLES}
for f in _pilcache.values():
    f.load()
def s_crop_center(f):
    img = _pilcache[f]; W, H = img.size
    return crop_image(img, center_crop_box(W, H, OUT), OUT)
def s_crop_roi(f):
    img = _pilcache[f]; W, H = img.size
    return crop_image(img, roi_crop_box(W, H, synth_roi(W, H), 0.10), OUT)
cc_ms = timeit(s_crop_center); cr_ms = timeit(s_crop_roi)
print(f"[2] PIL center crop(640) [no resize]    : {cc_ms:7.2f} ms/img")
print(f"[2] PIL roi crop+resize(640)            : {cr_ms:7.2f} ms/img")

# =====================================================================
# STAGE 3 — to_tensor_norm: the /255 + stack3 + normalize + contiguous
# =====================================================================
_center_cache = {f: s_crop_center(f) for f in SAMPLES}
def s_to_tensor(f):
    return to_tensor_norm(_center_cache[f])
tt_ms = timeit(s_to_tensor)
print(f"[3] to_tensor_norm (per crop)           : {tt_ms:7.2f} ms/img  (x2 crops = {2*tt_ms:.2f})")

# break to_tensor_norm into sub-steps
arr_u8 = np.asarray(_center_cache[SAMPLES[0]])
def s_tt_astype(_):
    a = np.asarray(arr_u8, dtype=np.float32) / 255.0
def s_tt_stack(_):
    a = np.asarray(arr_u8, dtype=np.float32) / 255.0; a = np.stack([a, a, a], 0)
def s_tt_norm(_):
    a = np.asarray(arr_u8, dtype=np.float32) / 255.0; a = np.stack([a, a, a], 0); a = (a - IMAGENET_MEAN) / IMAGENET_STD
def s_tt_full(_):
    a = np.asarray(arr_u8, dtype=np.float32) / 255.0; a = np.stack([a, a, a], 0); a = (a - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(a)).float()
print(f"     - astype/255 only                  : {timeit(s_tt_astype):7.2f} ms")
print(f"     - + stack3                         : {timeit(s_tt_stack):7.2f} ms")
print(f"     - + normalize                      : {timeit(s_tt_norm):7.2f} ms")
print(f"     - + contiguous+torch (full)        : {timeit(s_tt_full):7.2f} ms")

# =====================================================================
# STAGE 4 — OPTIMIZED to_tensor: keep uint8 1-channel on CPU (transfer that),
# do /255 + 3ch replicate + normalize on GPU. Here we just measure the CPU half.
# =====================================================================
def s_opt_cpu(f):
    # what the worker would emit: a contiguous uint8 1xHxW tensor (tiny, fast)
    a = np.asarray(_center_cache[f])  # HxW uint8
    return torch.from_numpy(np.ascontiguousarray(a))  # uint8, 1 plane
opt_ms = timeit(s_opt_cpu)
print(f"[4] OPT cpu-side (uint8 1ch, no norm)   : {opt_ms:7.2f} ms/img  (vs {tt_ms:.2f} current)")

# =====================================================================
# STAGE 5 — data volume to transfer over PCIe per image (H2D)
# =====================================================================
cur_bytes = 2 * 3 * OUT * OUT * 4   # 2 crops x 3ch x 640^2 x float32
opt_bytes = 2 * 1 * OUT * OUT * 1   # 2 crops x 1ch x 640^2 x uint8
print(f"\n[5] H2D bytes/img  current={cur_bytes/1e6:.2f} MB   optimized={opt_bytes/1e6:.3f} MB   "
      f"reduction x{cur_bytes/opt_bytes:.0f}")

# =====================================================================
# SUMMARY — current full per-image CPU cost (decode + 2 crops + 2 tensor)
# =====================================================================
def s_full_current(f):
    img = Image.open(f).convert("L"); W, H = img.size
    center = crop_image(img, center_crop_box(W, H, OUT), OUT)
    roic = crop_image(img, roi_crop_box(W, H, synth_roi(W, H), 0.10), OUT)
    return to_tensor_norm(center), to_tensor_norm(roic)
def s_full_opt(f):
    a = cv2.imread(f, cv2.IMREAD_GRAYSCALE); H, W = a.shape
    img = Image.fromarray(a)  # hand to PIL only for the (exact) bilinear resize
    center = crop_image(img, center_crop_box(W, H, OUT), OUT)
    roic = crop_image(img, roi_crop_box(W, H, synth_roi(W, H), 0.10), OUT)
    c = torch.from_numpy(np.ascontiguousarray(np.asarray(center)))
    r = torch.from_numpy(np.ascontiguousarray(np.asarray(roic)))
    return c, r
fc_ms = timeit(s_full_current); fo_ms = timeit(s_full_opt)
print(f"\n[SUMMARY] full per-image CPU pipeline")
print(f"   current (PIL decode + float norm + 3ch): {fc_ms:7.2f} ms/img")
print(f"   optimized (cv2 decode + uint8 1ch)     : {fo_ms:7.2f} ms/img   speedup x{fc_ms/fo_ms:.2f}")
print(f"\n   @ 4418 imgs, 4 effective cores (Kaggle T4x2):")
print(f"     current   ~ {fc_ms*4418/4/1000:6.1f}s CPU-bound floor")
print(f"     optimized ~ {fo_ms*4418/4/1000:6.1f}s CPU-bound floor")
