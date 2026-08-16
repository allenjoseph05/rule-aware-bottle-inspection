"""Generate notebooks/fast_benchmark.ipynb — find the FASTEST batch-1 forward on T4.

We've been timing ConvNeXt-Tiny@640 (a big model at huge res = 12.9ms/forward).
This benchmarks small/low-res architectures + torch.compile(reduce-overhead) CUDA graphs
(which help when launch-bound, i.e. small models) to find the true per-image floor under
the organizer's batch-1 sequential rules. No TRT (env-blocked), no ORT (slower) — just
the reliable eager-FP16 + CUDA-graph path. Self-contained (random init + synthetic PNG).

Reports: pyspng decode ms, and per-model eager-FP16 vs compiled(reduce-overhead) forward ms,
+ projected full-set total = decode + forward. Compile/warmup are UNTIMED.
Attach NOTHING. GPU T4 x1, Internet ON (pip pyspng only).
"""
import json
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "fast_benchmark.ipynb"
cells = []
def md(t): cells.append(nbf.v4.new_markdown_cell(t.strip()))
def code(t): cells.append(nbf.v4.new_code_cell(t.strip()))

md("""
# Fastest-forward benchmark — small models + CUDA graphs (single T4, batch-1)

Finds the true per-image floor: tiny CNNs at low res + `torch.compile(reduce-overhead)`
(CUDA graphs) vs eager-FP16. No TRT/ORT (one's blocked, the other's slower). Self-contained.
GPU **T4 x1**, Internet ON.
""")

md("## 0. Install pyspng (untimed)")
code(r"""
import sys, subprocess
subprocess.run([sys.executable,"-m","pip","install","-q","pyspng"], check=False)
""")

md("## 1. Setup + synthetic PNG + decode benchmark")
code(r"""
import os, time, importlib
import numpy as np, torch, torch.nn as nn
from PIL import Image
import timm
torch.backends.cudnn.benchmark=True
dev=torch.device("cuda"); assert torch.cuda.is_available()
print("torch",torch.__version__,"| timm",timm.__version__,"| GPU",torch.cuda.get_device_name(0))

# synthetic 1280x1024 grayscale PNG (~real size)
yy,xx=np.mgrid[0:1024,0:1280]
img=np.clip(127+80*np.sin(xx/40.0)*np.cos(yy/55.0)+np.random.randn(1024,1280)*8,0,255).astype(np.uint8)
SP="/kaggle/working/s.png"; Image.fromarray(img,mode="L").save(SP)
print("synthetic PNG",round(os.path.getsize(SP)/1e6,2),"MB")

def have(m):
    try: importlib.import_module(m); return True
    except Exception: return False
def bench_d(fn,n=120):
    for _ in range(3): fn()
    t=time.perf_counter()
    for _ in range(n): fn()
    return (time.perf_counter()-t)/n*1000
dres={"PIL":bench_d(lambda: np.asarray(Image.open(SP).convert("L")))}
if have("cv2"):
    import cv2; dres["cv2"]=bench_d(lambda: cv2.imread(SP,cv2.IMREAD_GRAYSCALE))
if have("pyspng"):
    import pyspng; b=open(SP,"rb").read(); dres["pyspng"]=bench_d(lambda: pyspng.load(b))
print("\nDECODE ms/img:")
for k,v in dres.items(): print(f"  {k:8s}: {v:5.2f} ms -> {v*4418/1000:4.0f}s")
DECODE_MS=min(dres.values())
""")

md("## 2. Forward benchmark — tiny models @ low res, eager-FP16 vs CUDA-graph (reduce-overhead)")
code(r"""
class Net(nn.Module):
    def __init__(s,bb,res):
        super().__init__(); s.res=res
        s.m=timm.create_model(bb,pretrained=False,num_classes=1,global_pool="avg")
    def forward(s,x): return s.m(x).squeeze(1)

# (name, timm backbone, resolution). Small+low-res = launch-bound = CUDA graphs help.
CONFIGS=[
    ("convnext_tiny@640 (current)","convnext_tiny.dinov3_lvd1689m",640),
    ("convnext_tiny@320","convnext_tiny.dinov3_lvd1689m",320),
    ("convnext_atto@224","convnext_atto",224),
    ("resnet18@224","resnet18",224),
    ("mobilenetv4_small@224","mobilenetv4_conv_small.e2400_r224_in1k",224),
    ("mobilenetv4_small@160","mobilenetv4_conv_small.e2400_r224_in1k",160),
    ("efficientnet_lite0@224","efficientnet_lite0",224),
]
def time_fwd(fn,reps):
    fn(); torch.cuda.synchronize()
    t=time.perf_counter()
    for _ in range(reps): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/reps*1000

print(f"{'config':30s} {'eager-FP16':>11} {'compiled(CUDAgraph)':>20} {'best fwd':>9} {'+decode -> full-set':>22}")
rows=[]
for name,bb,res in CONFIGS:
    try:
        net=Net(bb,res).to(dev).eval().half()
        x=torch.randn(1,3,res,res,device=dev,dtype=torch.float16)
        def run_e():
            with torch.no_grad(): return float(net(x).item())
        for _ in range(10): run_e()
        te=time_fwd(run_e,200)
        # CUDA-graph via torch.compile reduce-overhead (warmup = untimed)
        tc=float('nan')
        try:
            netc=torch.compile(net,mode="reduce-overhead")
            def run_c():
                with torch.no_grad(): return float(netc(x).item())
            for _ in range(12): run_c()   # warmup (compile + graph capture)
            tc=time_fwd(run_c,200)
        except Exception as e:
            print("  compile failed for",name,str(e)[:80])
        best=min(te, tc if tc==tc else te)
        total=(DECODE_MS+best)*4418/1000
        print(f"{name:30s} {te:8.2f}ms {tc:17.2f}ms {best:6.2f}ms {total:16.0f}s")
        rows.append((name,te,tc,best,total))
        del net; torch.cuda.empty_cache()
    except Exception as e:
        print(f"{name:30s}  FAILED: {str(e)[:80]}")
print(f"\ndecode floor: {DECODE_MS:.2f} ms/img ({DECODE_MS*4418/1000:.0f}s) — the irreducible CPU cost IF we decode.")
print("If organizer pre-decodes in their loop, only the forward counts (subtract decode).")
best_row=min(rows,key=lambda r:r[4]) if rows else None
if best_row: print(f"\n==> FASTEST: {best_row[0]} = {best_row[3]:.2f}ms fwd -> ~{best_row[4]:.0f}s full-set (decode+fwd)")
""")

nb=nbf.v4.new_notebook(); nb["cells"]=cells
nb["metadata"]={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}}
NB.write_text(json.dumps(nb,indent=1),encoding="utf-8"); print("wrote",NB)
