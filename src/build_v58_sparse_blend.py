"""Build v58 sparse-blend submission candidates from v56 and v57 outputs.

The goal is not to replace v56. v56 forced is the current public-LB champion.
v57's broad COCO-state rule added too many positives, so here we only use v57
as high-confidence correction evidence.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artefacts" / "v58_sparse_blend"
OUT.mkdir(parents=True, exist_ok=True)

V56_TEST = ROOT / "artefacts" / "kaggle_v56_forced_submit_cpu_output" / "v56_forced_test_predictions.csv"
V56_SUB = ROOT / "artefacts" / "kaggle_v56_forced_submit_cpu_output" / "submission.csv"
V56_VAL = ROOT / "artefacts" / "kaggle_v56_v3_output" / "v56_val_predictions.csv"
V57_TEST = ROOT / "v57_output" / "v57_test_predictions.csv"
V57_SUB = ROOT / "v57_output" / "submission.csv"
V57_VAL = ROOT / "v57_output" / "v57_val_predictions.csv"


def metric_report(y_true: pd.Series, pred: pd.Series) -> dict:
    return {
        "f1": float(f1_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred)),
        "recall": float(recall_score(y_true, pred)),
        "positive_rate": float(pred.mean()),
        "tp": int(((pred == 1) & (y_true == 1)).sum()),
        "fp": int(((pred == 1) & (y_true == 0)).sum()),
        "fn": int(((pred == 0) & (y_true == 1)).sum()),
        "tn": int(((pred == 0) & (y_true == 0)).sum()),
    }


def add_v57_scores(base: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "image_id",
        "score_group_c",
        "score_group_b",
        "score_target_aux",
        "pred_group_c_any",
        "pred_group_b_above_any",
        "pred_distractor_any",
    ]
    return base.merge(scores[cols], on="image_id", how="left")


def build_candidate(df: pd.DataFrame, add_aux: float | None, add_score: float | None, rem_sc: float | None,
                    rem_sb: float | None, rem_dist: float | None) -> tuple[pd.Series, pd.Series, pd.Series]:
    pred = df["v56_pred"].astype(int).copy()

    if add_aux is None:
        add_mask = pd.Series(False, index=df.index)
    else:
        add_mask = (
            (df["v56_pred"] == 0)
            & (df["score_target_aux"] >= float(add_aux))
            & ((df["score_group_c"] >= float(add_score)) | (df["score_group_b"] >= 0.62))
        )
        pred.loc[add_mask] = 1

    if rem_sc is None:
        rem_mask = pd.Series(False, index=df.index)
    else:
        rem_mask = (
            (df["v56_pred"] == 1)
            & (df["score_group_c"] <= float(rem_sc))
            & (df["score_group_b"] <= float(rem_sb))
            & (df["pred_distractor_any"] >= float(rem_dist))
        )
        pred.loc[rem_mask] = 0

    return pred.astype(int), add_mask, rem_mask


def main() -> None:
    v56_sub = pd.read_csv(V56_SUB).rename(columns={"target": "v56_pred"})
    v56_test = pd.read_csv(V56_TEST).rename(columns={"target": "v56_pred_from_scores"})
    v57_sub = pd.read_csv(V57_SUB).rename(columns={"target": "v57_pred"})
    v57_test = pd.read_csv(V57_TEST)
    v56_val = pd.read_csv(V56_VAL).rename(columns={"final_pred": "v56_pred"})
    v57_val = pd.read_csv(V57_VAL).rename(columns={"rule_pred": "v57_pred"})

    test = v56_sub.merge(v56_test.drop(columns=["v56_pred_from_scores"]), on="image_id", how="left")
    test = test.merge(v57_sub, on="image_id", how="left")
    test = add_v57_scores(test, v57_test)

    val = v56_val[
        [
            "image_id",
            "target",
            "bucket",
            "v56_pred",
            "v5_prob",
            "roi_prob",
            "final_prob",
            "aux_distractor_only_prob",
        ]
    ].merge(v57_val[["image_id", "v57_pred"]], on="image_id", how="left")
    val = add_v57_scores(val, v57_val)

    configs = [
        {
            "name": "v58a_removal_tiny_high_precision",
            "description": "Only remove v56 positives with extremely weak v57 class-state evidence and very strong distractor evidence.",
            "add_aux": None,
            "add_score": None,
            "rem_sc": 0.28,
            "rem_sb": 0.24,
            "rem_dist": 0.90,
        },
        {
            "name": "v58b_removal_medium_precision",
            "description": "Remove a medium-size set of likely v56 false positives; no v57 additions.",
            "add_aux": None,
            "add_score": None,
            "rem_sc": 0.36,
            "rem_sb": 0.20,
            "rem_dist": 0.70,
        },
        {
            "name": "v58c_sparse_balanced_best_val",
            "description": "Best local sparse family: tiny high-confidence v57 additions plus medium high-precision removals.",
            "add_aux": 0.80,
            "add_score": 0.62,
            "rem_sc": 0.36,
            "rem_sb": 0.20,
            "rem_dist": 0.70,
        },
        {
            "name": "v58d_sparse_balanced_more_conservative",
            "description": "Balanced, but stricter additions and same medium removals.",
            "add_aux": 0.85,
            "add_score": 0.76,
            "rem_sc": 0.36,
            "rem_sb": 0.20,
            "rem_dist": 0.70,
        },
    ]

    rows = []
    baseline = metric_report(val["target"].astype(int), val["v56_pred"].astype(int))
    v57_full = metric_report(val["target"].astype(int), val["v57_pred"].astype(int))

    for cfg in configs:
        val_pred, val_add, val_rem = build_candidate(val, cfg["add_aux"], cfg["add_score"], cfg["rem_sc"], cfg["rem_sb"], cfg["rem_dist"])
        test_pred, test_add, test_rem = build_candidate(test, cfg["add_aux"], cfg["add_score"], cfg["rem_sc"], cfg["rem_sb"], cfg["rem_dist"])

        sub = test[["image_id"]].copy()
        sub["target"] = test_pred.astype(int).values
        out_csv = OUT / f"{cfg['name']}_submission.csv"
        sub.to_csv(out_csv, index=False)

        change_df = test[["image_id", "v56_pred", "v57_pred", "bucket", "v5_prob", "roi_prob", "final_prob",
                          "score_group_c", "score_group_b", "score_target_aux", "pred_distractor_any"]].copy()
        change_df["v58_pred"] = test_pred.values
        change_df["added_by_v58"] = test_add.values
        change_df["removed_by_v58"] = test_rem.values
        change_df = change_df[change_df["v56_pred"] != change_df["v58_pred"]]
        change_df.to_csv(OUT / f"{cfg['name']}_changed_rows.csv", index=False)

        rep = metric_report(val["target"].astype(int), val_pred)
        rows.append({
            "name": cfg["name"],
            "description": cfg["description"],
            "submission": str(out_csv),
            "val_f1": rep["f1"],
            "val_precision": rep["precision"],
            "val_recall": rep["recall"],
            "val_pos_rate": rep["positive_rate"],
            "test_pos_rate": float(test_pred.mean()),
            "test_positive_count": int(test_pred.sum()),
            "test_changes_vs_v56": int((test_pred != test["v56_pred"]).sum()),
            "test_0_to_1": int(((test["v56_pred"] == 0) & (test_pred == 1)).sum()),
            "test_1_to_0": int(((test["v56_pred"] == 1) & (test_pred == 0)).sum()),
            "val_add_count": int(val_add.sum()),
            "val_remove_count": int(val_rem.sum()),
            "test_add_count": int(test_add.sum()),
            "test_remove_count": int(test_rem.sum()),
        })

    report = {
        "baseline_v56_val": baseline,
        "v57_full_val": v57_full,
        "candidates": rows,
        "note": "v58 candidates are no-GPU sparse blends. Submit at most 1-2; repeated LB probing can overfit public.",
    }
    pd.DataFrame(rows).to_csv(OUT / "v58_candidate_summary.csv", index=False)
    (OUT / "v58_candidate_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
