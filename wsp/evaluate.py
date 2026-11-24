"""Metrics and error analysis."""
from __future__ import annotations

import numpy as np


def compute_metrics(task: str, y, pred, prob=None, k: int | None = None) -> dict:
    if task == "classification":
        y = np.asarray(y).astype(int)
        pred = np.asarray(pred).astype(int)
        k = k or int(max(y.max(), pred.max()) + 1)
        f1s = []
        for c in range(k):
            tp = np.sum((pred == c) & (y == c))
            fp = np.sum((pred == c) & (y != c))
            fn = np.sum((pred != c) & (y == c))
            if tp + fp + fn == 0:
                continue
            f1s.append(2 * tp / (2 * tp + fp + fn))
        m = {"accuracy": float(np.mean(pred == y)), "macro_f1": float(np.mean(f1s)) if f1s else 0.0}
        if prob is not None:
            eps = 1e-12
            if prob.ndim == 1:
                p = np.clip(prob, eps, 1 - eps)
                m["log_loss"] = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
            else:
                m["log_loss"] = float(-np.mean(np.log(np.clip(prob[np.arange(len(y)), y], eps, 1))))
        return m
    y2 = np.asarray(y, dtype=float).reshape(len(y), -1)
    p2 = np.asarray(pred, dtype=float).reshape(len(y), -1)
    err = p2 - y2
    sst = ((y2 - y2.mean(axis=0)) ** 2).sum(axis=0)
    sse = (err ** 2).sum(axis=0)
    r2 = np.where(sst > 0, 1 - sse / np.where(sst > 0, sst, 1), 0.0)
    return {"mse": float(np.mean(err ** 2)), "mae": float(np.mean(np.abs(err))), "r2": float(np.mean(r2))}


def primary_metric(task: str) -> str:
    return "accuracy" if task == "classification" else "r2"


def confusion_matrix(y, pred, k: int) -> np.ndarray:
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(np.asarray(y).astype(int), np.asarray(pred).astype(int)):
        if 0 <= t < k and 0 <= p < k:
            cm[t, p] += 1
    return cm


def worst_examples(task: str, y, pred, prob, n: int) -> np.ndarray:
    """Indices of the n worst predictions."""
    if task == "classification":
        wrong = np.where(np.asarray(pred) != np.asarray(y))[0]
        if prob is not None and len(wrong):
            if prob.ndim == 1:
                conf = np.abs(prob[wrong] - 0.5)
            else:
                conf = prob[wrong, np.asarray(pred)[wrong]]
            wrong = wrong[np.argsort(-conf)]
        return wrong[:n]
    err = np.abs(np.asarray(pred, float).reshape(len(y), -1) - np.asarray(y, float).reshape(len(y), -1)).sum(axis=1)
    return np.argsort(-err)[:n]
