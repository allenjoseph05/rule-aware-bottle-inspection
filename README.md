# Rule-Aware Bottle Inspection

An end-to-end computer-vision system for binary quality inspection of returnable
glass bottles, developed for the Krones AG / TH Deggendorf Vision AI Challenge.
The task combines predictive F1, inference efficiency, and technical insight.

The system was developed on 35,342 grayscale images at 1280 x 1024 resolution
with 134,471 COCO annotations across 27 categories. Its best public-leaderboard
result was **0.97321 F1 (#2 of 41 teams, 0.00003 behind #1)**.

## Key ideas

- Reconstructed the deterministic pass/fail target from per-class COCO polygon
  areas, matching all 35,342 training labels.
- Fine-tuned a DINOv3 ConvNeXt-Tiny backbone with decision-aligned auxiliary
  heads, hard-case sampling, and illumination-matched defect copy-paste.
- Combined classifier, segmentation, and rule-state evidence through strict
  cross-bias agreement rather than unconstrained score averaging.
- Audited threshold stability, bottle-size subgroup F1, prediction flips,
  public/hidden generalization, and preprocessing equivalence.
- Exported the deployed single-pass model to FP16 ONNX Runtime with hardened
  offline inference and 4,416 / 4,418 decision parity against FP32.

## Results

| System | Public F1 | Approximate runtime for 4,418 images | Purpose |
|---|---:|---:|---|
| DINOv3 ConvNeXt-Tiny anchor | 0.96839 | 175-222 s | Transfer-learning baseline |
| Single-pass deployed model | 0.97066 | 187 s | Accuracy/efficiency submission |
| Cross-bias agreement ensemble | 0.97321 | 650 s | Highest public F1 |
| Decision-aligned segmentation rule | 0.97120 | 862 s | High-resolution verifier |

Runtime values are hardware- and evaluation-protocol-dependent. The competition
timed sequential inference in a fresh offline environment, so the shipped model
was selected on the coupled accuracy-efficiency objective rather than F1 alone.

## Repository layout

```text
external/group05_train/       Final training notebook
external/group05_eval_final/  Hardened offline evaluation pipeline
external/v62_pull/            Searchable training-script export
notebooks/                     Selected segmentation and latency experiments
scripts/                       Robustness, equivalence, profiling, and ensembles
src/                           Sparse rule-composition utility
```

Only representative, reproducible source files are published. Internal planning
documents, submissions, logs, cached predictions, model weights, vendored wheels,
and generated reports are intentionally excluded.

## Setup

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

The original competition dataset and trained checkpoints are not included. Place
the competition files in `data/` using the paths expected by the notebooks. That
directory is ignored to comply with the competition's data-distribution rules.

## Main entry points

- Training: `external/group05_train/group05_training_notebook.ipynb`
- Searchable training export: `external/v62_pull/v62_flat.py`
- Final evaluation: `external/group05_eval_final/group05_eval_notebook.py`
- ONNX export: `external/export_v62sp_onnx.py`
- FP16 conversion/parity: `external/fp16_fix.py`
- Robustness audit: `scripts/robustness_audit.py`
- Pipeline equivalence: `scripts/verify_pipeline_equivalence.py`

## Technology

Python, PyTorch, timm, DINOv3, OpenCV, Albumentations, scikit-learn,
pycocotools, ONNX, ONNX Runtime CUDA, and mixed-precision inference.

## License

Apache License 2.0. See `LICENSE`.
