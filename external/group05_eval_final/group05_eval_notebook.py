# %% [code]
"""
Group 05 - Evaluation Notebook (v62 single-pass LB 0.97066)

ONE model, ONE forward per image, coupled submission:
  submission.csv -> raw F1 (50%) ; the timed sequential loop -> efficiency (30%).

Model: DINOv3 ConvNeXt-Tiny @640 ROI with 5 heads, exported to ONNX with
preprocessing baked in (uint8 in; /255 + 1->3ch + ImageNet-norm inside the graph).
Decision = deterministic rule on the heads (fixed thresholds + bucket tau, validated on LB):
  score_c = sqrt(p_group_c * max(max_i p_class_i, p_rare))        >= 0.54
  score_b = sqrt(p_groupB_above * max_i [p_pres_i * sigm((logarea_i - logthr_i)/tau_bucket)]) >= 0.58
  positive if either fires.

Rules-faithful: model load + warmup UNTIMED; timed loop is SEQUENTIAL batch-1
(read + decode + ROI crop + resize + forward + rule); no batching/multiprocessing/
caching. Hidden-test ROI file (test_annotations_roi_only.json) is provided by the
organizers; a centre-crop fallback exists if any ROI is missing.

Hardening for the fresh-environment re-run:
  * offline wheels (onnxruntime-gpu, pyspng) with PIL fallback runs internet-OFF
  * decode guards: 16-bit PNGs rescaled, RGB collapsed to grey, any size handled
  * pos-rate circuit breaker: if the fixed rule yields a positive rate outside
    [0.45, 0.75] on the hidden data (catastrophic shift), fall back to quantile
    thresholding at the train rate 0.5833 on the continuous score.

INPUTS (no internet required):
  * Competition data: test_images/, sample_submission.csv, test_annotations_roi_only.json
  * Attached dataset 'krones-group05-model': the ONNX model (v62sp_fp16.onnx) +
    offline pip wheels (onnxruntime-gpu, pyspng, deps). The notebook installs these
    from the dataset nothing is downloaded.

HOW TO RUN: Accelerator = GPU (Tesla T4); Internet = OFF; Run All (top to bottom).

EXPECTED OUTPUT (sanity check for the re-run):
  * Writes submission.csv with one row per test image (image_id, target in {0,1}).
  * Prints the decoder used, the positive rate (~0.584 expected), and the timed-loop
    seconds. A positive rate far from ~0.58 would indicate an input/setup problem.
"""
import os, sys, glob, time, json, math, subprocess
import numpy as np, pandas as pd

# ===================== CONFIG (validated operating point) =====================
S = 640; ROI_PAD = 0.10
T_GROUP_C = 0.54; T_GROUP_B = 0.58
TAU_PER_BUCKET = {"small": 1.35, "medium": 1.6, "large": 0.95}
TRAIN_POS_RATE = 0.5832508134106663
GROUP_B_LOG_THR = np.array([math.log(t + 1.0) for t in (500, 200, 180, 100, 75000, 1200)], np.float32)

def _find(pat):
    m = glob.glob(pat, recursive=True); return m[0] if m else None

def _find_onnx():
    # robust against lazy /kaggle/input mounts: walk + retry
    for attempt in range(5):
        for pat in ("/kaggle/input/**/v62sp_fp16.onnx", "/kaggle/input/**/v62sp.onnx"):
            m = glob.glob(pat, recursive=True)
            if m: return m[0]
        for root, _, files in os.walk("/kaggle/input"):
            for f in files:
                if f in ("v62sp_fp16.onnx", "v62sp.onnx"):
                    return os.path.join(root, f)
        time.sleep(8)
    return None

ONNX_PATH = _find_onnx()
SUB_PATH  = _find("/kaggle/input/**/sample_submission.csv")
ROI_JSON  = _find("/kaggle/input/**/test_annotations_roi_only.json")
_png      = _find("/kaggle/input/**/test_images/*.png") or _find("/kaggle/input/**/test*/**/*.png")
TEST_DIR  = os.path.dirname(_png) if _png else None
WHEEL_DIR = os.path.dirname(_find("/kaggle/input/**/pyspng*.whl") or "") or None
if not ONNX_PATH:
    print("DIAGNOSTIC /kaggle/input tree (model dataset missing attach 'krones-group05-model'):")
    for d in sorted(glob.glob("/kaggle/input/*") + glob.glob("/kaggle/input/*/*")):
        print(" ", d)
assert ONNX_PATH and SUB_PATH and TEST_DIR, f"paths: onnx={ONNX_PATH} sub={SUB_PATH} test={TEST_DIR}"
print(f"onnx={ONNX_PATH}\nroi={ROI_JSON}\ntest={TEST_DIR}\nwheels={WHEEL_DIR}")

# ===================== deps: pre-installed first, offline wheels fallback =====================
def _pip_offline(pat):
    w = glob.glob(os.path.join(WHEEL_DIR, pat)) if WHEEL_DIR else []
    if w: subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "-q", w[0]], check=False)
try:
    import onnxruntime as ort
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        _pip_offline("onnxruntime_gpu*.whl"); import importlib; importlib.reload(ort)
except Exception:
    _pip_offline("onnxruntime_gpu*.whl"); import onnxruntime as ort
DECODER = "PIL"
try:
    import pyspng; DECODER = "pyspng"
except Exception:
    _pip_offline("pyspng*.whl")
    try: import pyspng; DECODER = "pyspng"
    except Exception: DECODER = "PIL"
from PIL import Image
print(f"onnxruntime {ort.__version__} | decoder {DECODER}")

# ===================== ONNX session + warmup (UNTIMED; conv search paid here) =====================
_cuda = "CUDAExecutionProvider" in ort.get_available_providers()
_provs = ([("CUDAExecutionProvider", {"cudnn_conv_algo_search": "EXHAUSTIVE"}), "CPUExecutionProvider"]
          if _cuda else ["CPUExecutionProvider"])
_so = ort.SessionOptions(); _so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess = ort.InferenceSession(ONNX_PATH, sess_options=_so, providers=_provs)
print("providers:", sess.get_providers())
import subprocess as _sp
print("GPU:", _sp.run("nvidia-smi --query-gpu=name --format=csv,noheader", shell=True, capture_output=True, text=True).stdout.strip())
_wu = np.random.randint(0, 256, (1, 1, S, S), dtype=np.uint8)
_wb = np.array([[0., 1., 0.]], np.float32); _wr = np.array([[0.84, 0.84]], np.float32)
for _ in range(15):
    sess.run(None, {"image_u8": _wu, "bv": _wb, "rg": _wr})

# ===================== test ROI lookup (centre-crop fallback if absent) =====================
ROI = {}
if ROI_JSON:
    coco = json.load(open(ROI_JSON))
    _cat = {c["id"]: c["name"] for c in coco["categories"]}
    _by = {}
    for a in coco["annotations"]:
        if _cat.get(a["category_id"]) == "Roi":
            _by.setdefault(a["image_id"], a["bbox"])
    ROI = {im["file_name"]: _by.get(im["id"]) for im in coco["images"]}
print(f"ROI entries: {sum(v is not None for v in ROI.values())}/{len(ROI) or 0}")

def _decode_gray(path):
    if DECODER == "pyspng":
        with open(path, "rb") as f: a = pyspng.load(f.read())
    else:
        a = np.asarray(Image.open(path))
    if a.ndim == 3: a = a[..., 0]                       # RGB(A) -> first channel
    if a.dtype == np.uint16: a = (a >> 8).astype(np.uint8)
    elif a.dtype != np.uint8: a = np.clip(a, 0, 255).astype(np.uint8)
    return a

def preprocess(path, roi):
    g = _decode_gray(path); H, W = g.shape[:2]
    if roi is None:                                      # fallback: centre square
        side = min(W, H); l = (W - side) // 2; t = (H - side) // 2
        rw = rh = float(side) / (1 + 2 * ROI_PAD)
    else:
        x, y, rw, rh = [float(v) for v in roi[:4]]
        side = int(round(max(rw, rh) * (1 + 2 * ROI_PAD))); side = max(1, min(side, W, H))
        cx, cy = x + rw / 2, y + rh / 2
        l = max(0, min(int(round(cx - side / 2)), W - side))
        t = max(0, min(int(round(cy - side / 2)), H - side))
    crop = g[t:t + side, l:l + side]
    if crop.shape != (S, S):
        crop = np.asarray(Image.fromarray(crop, mode="L").resize((S, S), Image.BILINEAR))
    b = "small" if rw < 510 else ("large" if rw > 590 else "medium")
    bv = np.array([[b == "small", b == "medium", b == "large"]], np.float32)
    rg = np.array([[rw / S, rh / S]], np.float32)
    return np.ascontiguousarray(crop)[None, None], bv, rg, b

def decide(g, c, bp, bl, bucket):
    if not (np.all(np.isfinite(g)) and np.all(np.isfinite(c)) and np.all(np.isfinite(bp)) and np.all(np.isfinite(bl))):
        return 1, 1.0                                      # HARDENING: non-finite output -> safe reject (no-op when finite)
    tau = TAU_PER_BUCKET.get(bucket, TAU_PER_BUCKET["medium"])
    b_sup = float((bp * (1.0 / (1.0 + np.exp(-(bl - GROUP_B_LOG_THR) / tau)))).max())
    c_sup = max(float(c.max()), float(g[3]))
    sc = math.sqrt(max(0.0, min(1.0, float(g[0]) * c_sup)))
    sb = math.sqrt(max(0.0, min(1.0, float(g[1]) * b_sup)))
    return int(sc >= T_GROUP_C or sb >= T_GROUP_B), max(sc, sb)

# ===================== TIMED LOOP (sequential batch-1) =====================
sub = pd.read_csv(SUB_PATH); IDC = sub.columns[0]
t0 = time.perf_counter(); preds = []; scores = []; _n_fail = 0
for img_id in sub[IDC].tolist():
    try:                                                   # HARDENING: one bad image must NOT crash the run (=score 0)
        s = str(img_id); key = s if s.lower().endswith(".png") else s + ".png"
        x, bv, rg, bucket = preprocess(os.path.join(TEST_DIR, key), ROI.get(key))
        _, g, c, bp, bl = sess.run(None, {"image_u8": x, "bv": bv, "rg": rg})
        p, sc = decide(g[0], c[0], bp[0], bl[0], bucket)
    except Exception:
        p, sc = 1, 1.0; _n_fail += 1                       # unreadable/odd image -> safe reject (majority class)
    preds.append(p); scores.append(sc)
elapsed = time.perf_counter() - t0
# =============================================================================

if len(preds) != len(sub):                             # HARDENING: guarantee complete, row-aligned submission
    preds = (preds + [1] * len(sub))[:len(sub)]; scores = (scores + [1.0] * len(sub))[:len(sub)]
if _n_fail:
    print(f"HARDENING: {_n_fail} image(s) failed and were defaulted to target=1")
pos_rate = float(np.mean(preds))
if not (0.45 <= pos_rate <= 0.75):                      # circuit breaker (never fires normally)
    print(f"CIRCUIT BREAKER: pos_rate {pos_rate:.4f} out of range -> quantile fallback @ {TRAIN_POS_RATE:.4f}")
    thr = float(np.quantile(np.asarray(scores), 1.0 - TRAIN_POS_RATE))
    preds = [int(s >= thr) for s in scores]
    pos_rate = float(np.mean(preds))

sub["target" if "target" in sub.columns else sub.columns[-1]] = preds
sub.to_csv("submission.csv", index=False)
print(f"decoder={DECODER}  positives={int(np.sum(preds))}/{len(preds)}  pos_rate={pos_rate:.4f}")
print(f"timed loop: {elapsed:.1f}s  ({elapsed/len(preds)*1000:.2f} ms/img)")
print("EXPECTED: pos_rate ~0.5844 ; exact match to v62_sp_submission.csv (LB 0.97066)")
