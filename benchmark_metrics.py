#!/usr/bin/env python3
"""Run a local benchmark on BreastDCEDL metadata.

This does not alter the repository's core algorithm. It only measures:
- Accuracy / Precision / Recall / F1 / AUC
- Latency / CPU time / peak memory
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils.performance_utils import classification_metrics, profile_inference


TARGETS = ("pCR", "HR", "HER2")
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
}


def load_split(df: pd.DataFrame):
    df = df[df["pCR"].notna()].copy()
    split = df[SPLIT_COL].fillna(0).astype(int)
    return df[split == 0].copy(), df[split == 1].copy()


def make_xy(df: pd.DataFrame, target: str):
    df = df[df[target].notna()].copy()
    y = df[target].astype(int)
    X = df.drop(columns=[c for c in df.columns if c in DROP_COLS], errors="ignore")
    return X, y


def build_preprocessor(X: pd.DataFrame):
    categorical = [c for c in X.columns if X[c].dtype == "object"]
    numeric = [c for c in X.columns if c not in categorical]
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ]
    )


def model_specs():
    return {
        "logreg": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1),
        "extra_trees": ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1),
        "hist_gbdt": HistGradientBoostingClassifier(learning_rate=0.05, max_depth=4, max_iter=200, random_state=42),
    }


def fit_eval(name, estimator, X_train, y_train, X_test, y_test):
    pipe = Pipeline([("prep", build_preprocessor(X_train)), ("model", estimator)])
    pipe.fit(X_train, y_train)
    if hasattr(pipe.named_steps["model"], "predict_proba"):
        score = pipe.predict_proba(X_test)[:, 1]
    else:
        score = pipe.decision_function(X_test)
    pred = pipe.predict(X_test)
    metrics = classification_metrics(y_test, score)
    profile = profile_inference(lambda row: pipe.predict_proba(row.to_frame().T)[0, 1] if hasattr(pipe.named_steps["model"], "predict_proba") else pipe.decision_function(row.to_frame().T)[0], [X_test.iloc[i] for i in range(min(3, len(X_test)))])
    return name, metrics, profile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--limit-per-split", type=int, default=0, help="Limit rows per split for a quick smoke test; 0 = full set")
    args = parser.parse_args()

    df = pd.read_csv(args.metadata)
    if args.limit_per_split > 0:
        train_df, test_df = load_split(df)
        train_df = train_df.head(args.limit_per_split)
        test_df = test_df.head(args.limit_per_split)
    else:
        train_df, test_df = load_split(df)

    for target in TARGETS:
        X_train, y_train = make_xy(train_df, target)
        X_test, y_test = make_xy(test_df, target)
        print(f"\n== {target} ==")
        print("name,accuracy,precision,recall,f1,auc,latency_ms,cpu_ms,peak_mb")
        rows = []
        for name, estimator in model_specs().items():
            try:
                model_name, metrics, profile = fit_eval(name, estimator, X_train, y_train, X_test, y_test)
                rows.append((model_name, metrics, profile))
                print(
                    f"{model_name},{metrics.accuracy:.4f},{metrics.precision:.4f},{metrics.recall:.4f},"
                    f"{metrics.f1:.4f},{metrics.auc:.4f},{profile.latency_ms:.3f},{profile.process_time_ms:.3f},{profile.peak_memory_mb:.2f}"
                )
            except Exception as e:
                print(f"{name},ERROR,{e}")


if __name__ == "__main__":
    main()

