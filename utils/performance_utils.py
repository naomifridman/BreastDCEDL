#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Performance helpers for BreastDCEDL benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from time import perf_counter, process_time
from typing import Callable, Iterable, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


@dataclass
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: float
    confusion: list[list[int]]


@dataclass
class InferenceProfile:
    latency_ms: float
    process_time_ms: float
    peak_memory_mb: float


def classification_metrics(y_true, y_score, threshold: float = 0.5) -> ClassificationMetrics:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= threshold).astype(int)
    return ClassificationMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        auc=float(roc_auc_score(y_true, y_score)),
        confusion=confusion_matrix(y_true, y_pred).tolist(),
    )


def profile_inference(predict_fn: Callable[[object], object], inputs: Sequence[object], repeats: int = 1) -> InferenceProfile:
    """Profile a prediction callable on a small input sequence.

    `predict_fn` should accept one input item and return a prediction score or logits.
    """
    import tracemalloc

    if repeats < 1:
        raise ValueError("repeats must be >= 1")

    total_items = len(inputs) * repeats
    if total_items == 0:
        raise ValueError("inputs must not be empty")

    tracemalloc.start()
    wall_start = perf_counter()
    cpu_start = process_time()
    for _ in range(repeats):
        for item in inputs:
            predict_fn(item)
    wall = perf_counter() - wall_start
    cpu = process_time() - cpu_start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return InferenceProfile(
        latency_ms=float((wall * 1000.0) / total_items),
        process_time_ms=float((cpu * 1000.0) / total_items),
        peak_memory_mb=float(peak / (1024 * 1024)),
    )

