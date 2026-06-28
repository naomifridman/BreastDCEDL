#!/usr/bin/env python3
"""Standalone smoke test for pCR prediction outputs.

Reads original repo metadata + sample NIfTI files, trains a Random Forest,
and writes PNG/CSV/HTML artifacts to a separate output directory.

Does not modify any source files in the repository.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils import data_utils as ds  # noqa: E402

TARGET = "pCR"
SPLIT_COL = "test"
DROP_COLS = {
    "pid",
    "pCR",
    "HR",
    "HER2",
    "HR_HER2_STATUS",
    "TripleNeg",
    "HER2pos",
    "HRposHER2neg",
    "test",
    "dataset",
}


def load_split(metadata: Path, limit_per_split: int):
    df = pd.read_csv(metadata)
    df = df[df[TARGET].notna()].copy()
    split = df[SPLIT_COL].fillna(0).astype(int)
    train_df = df[split == 0].copy()
    test_df = df[split == 1].copy()
    if limit_per_split > 0:
        train_df = train_df.head(limit_per_split)
        test_df = test_df.head(limit_per_split)
    return train_df, test_df


def make_xy(df: pd.DataFrame):
    y = df[TARGET].astype(int)
    X = df.drop(columns=[c for c in df.columns if c in DROP_COLS], errors="ignore")
    return X, y


def build_preprocessor(X: pd.DataFrame):
    categorical = [c for c in X.columns if X[c].dtype == "object"]
    numeric = [c for c in X.columns if c not in categorical]
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )


def save_confusion_matrix(cm, labels, out_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_roc_curve(y_true, y_score, out_path: Path, title: str):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_val = roc_auc_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(fpr, tpr, label=f"AUC={auc_val:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(title)
    ax.legend()
    ax.set_aspect("equal", "box")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return float(auc_val)


def save_feature_importance(feature_names, importances, out_path: Path, top_n: int = 15):
    order = np.argsort(importances)[::-1][:top_n]
    names = [feature_names[i] for i in order]
    values = importances[order]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(names))))
    ax.barh(range(len(names)), values[::-1], color="#4C72B0")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1])
    ax.set_xlabel("Importance")
    ax.set_title(f"Random Forest feature importance (top {len(names)})")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_sample_mri_preview(repo_root: Path, out_path: Path):
    dce = {
        "spy2": str(repo_root / "ISPY2" / "data_samples" / "dce"),
        "spy1": str(repo_root / "ISPY1" / "data_samples" / "dce"),
        "duke": str(repo_root / "DUKE" / "data_samples" / "dce"),
    }
    mask = {
        "spy2": str(repo_root / "ISPY2" / "data_samples" / "mask"),
        "spy1": str(repo_root / "ISPY1" / "data_samples" / "mask"),
        "duke": str(repo_root / "DUKE" / "data_samples" / "mask"),
    }
    ds.setup_paths(str(repo_root), dce, mask)

    samples = [
        ("ISPY1_1072", "I-SPY1"),
        ("ISPY2-550421", "I-SPY2"),
        ("Breast_MRI_001", "Duke"),
    ]
    fig, axes = plt.subplots(1, len(samples), figsize=(4 * len(samples), 4))
    if len(samples) == 1:
        axes = [axes]

    for ax, (pid, label) in zip(axes, samples):
        acqs = ds.get_all_nifti_acquisitions(pid)
        if not acqs:
            ax.text(0.5, 0.5, f"No data\n{pid}", ha="center", va="center")
            ax.set_title(label)
            ax.axis("off")
            continue
        mid = acqs[0].shape[0] // 2
        img = acqs[0][mid]
        img = ds.cont_br(img)
        ax.imshow(img, cmap="gray", origin="lower")
        ax.set_title(f"{label}\n{pid}")
        ax.axis("off")

    fig.suptitle("Sample DCE-MRI (middle slice, pre-contrast)", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_html_report(out_dir: Path, metrics_df: pd.DataFrame):
    rows = metrics_df.to_html(index=False, float_format=lambda x: f"{x:.4f}")
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>BreastDCEDL pCR smoke test</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; max-width: 1100px; }}
    h1, h2 {{ color: #222; }}
    img {{ max-width: 100%; border: 1px solid #ddd; margin: 8px 0 16px; }}
    table {{ border-collapse: collapse; margin-bottom: 24px; }}
    th, td {{ border: 1px solid #ccc; padding: 6px 10px; }}
  </style>
</head>
<body>
  <h1>pCR prediction smoke test</h1>
  <p>Random Forest on metadata features. Outputs saved next to this file.</p>

  <h2>Evaluation table</h2>
  {rows}

  <h2>Sample MRI preview</h2>
  <img src="sample_mri_preview.png" alt="Sample MRI preview">

  <h2>Confusion matrix</h2>
  <img src="confusion_matrix.png" alt="Confusion matrix">

  <h2>ROC curve</h2>
  <img src="roc_curve.png" alt="ROC curve">

  <h2>Feature importance</h2>
  <img src="feature_importance.png" alt="Feature importance">
</body>
</html>
"""
    (out_dir / "report.html").write_text(html, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Smoke test pCR Random Forest + plots")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=REPO_ROOT / "BreastDCEDL_metadata.csv",
        help="Combined metadata CSV with pCR labels and test split",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/breastdcedl_pcr_smoke"),
        help="Separate output directory (not inside repo source tree by default)",
    )
    parser.add_argument(
        "--limit-per-split",
        type=int,
        default=40,
        help="Use only N train and N test rows for a quick smoke test (0 = full data)",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Repo root : {repo_root}")
    print(f"Metadata  : {args.metadata}")
    print(f"Output dir: {out_dir}")
    print(f"Limit/split: {args.limit_per_split or 'full'}")

    train_df, test_df = load_split(args.metadata, args.limit_per_split)
    X_train, y_train = make_xy(train_df)
    X_test, y_test = make_xy(test_df)

    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise SystemExit(
            "Need both pCR classes in train and test. Increase --limit-per-split "
            f"(train classes={sorted(y_train.unique())}, test classes={sorted(y_test.unique())})."
        )

    pipe = Pipeline(
        [
            ("prep", build_preprocessor(X_train)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pipe.fit(X_train, y_train)
    y_score = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_score >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_score)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=["pCR-", "pCR+"], output_dict=True
    )

    metrics_df = pd.DataFrame(
        [
            {
                "model": "RandomForest",
                "target": TARGET,
                "n_train": len(y_train),
                "n_test": len(y_test),
                "accuracy": acc,
                "precision": report["pCR+"]["precision"],
                "recall": report["pCR+"]["recall"],
                "f1": report["pCR+"]["f1-score"],
                "auc": auc,
            }
        ]
    )
    metrics_df.to_csv(out_dir / "pcr_metrics_table.csv", index=False)
    pd.DataFrame(cm, index=["pCR-", "pCR+"], columns=["pred_pCR-", "pred_pCR+"]).to_csv(
        out_dir / "confusion_matrix.csv"
    )

    with open(out_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(f"pCR Random Forest  --  Acc: {acc:.3f}  AUC: {auc:.3f}\n\n")
        f.write(classification_report(y_test, y_pred, target_names=["pCR-", "pCR+"]))
        f.write("\n")
        f.write(str(pd.DataFrame(cm, index=["pCR-", "pCR+"], columns=["pred_pCR-", "pred_pCR+"])))

    save_confusion_matrix(cm, ["pCR-", "pCR+"], out_dir / "confusion_matrix.png", "pCR confusion matrix")
    save_roc_curve(y_test, y_score, out_dir / "roc_curve.png", "pCR ROC (Random Forest)")

    prep = pipe.named_steps["prep"]
    model = pipe.named_steps["model"]
    feature_names = prep.get_feature_names_out()
    save_feature_importance(feature_names, model.feature_importances_, out_dir / "feature_importance.png")

    save_sample_mri_preview(repo_root, out_dir / "sample_mri_preview.png")
    write_html_report(out_dir, metrics_df)

    print("\nSaved outputs:")
    for name in [
        "report.html",
        "pcr_metrics_table.csv",
        "confusion_matrix.csv",
        "classification_report.txt",
        "confusion_matrix.png",
        "roc_curve.png",
        "feature_importance.png",
        "sample_mri_preview.png",
    ]:
        print(f"  - {out_dir / name}")

    print(f"\nOpen in browser: file://{out_dir / 'report.html'}")


if __name__ == "__main__":
    main()
