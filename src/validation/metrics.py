"""Regression and classification metrics from the assignment."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def smape(y_true, y_pred, eps: float = 1e-8) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0 + eps
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)


def mase(y_true, y_pred, y_naive) -> float:
    """MAE / MAE(naive) on the same window. Naive = 1, perfect forecast = 0."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    y_naive = np.asarray(y_naive, float)
    scale = float(np.mean(np.abs(y_true - y_naive)))
    if not scale:
        return 0.0 if float(np.mean(np.abs(y_true - y_pred))) == 0 else float("inf")
    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def regression_report(y_true, y_pred, y_naive) -> dict:
    return {
        "MASE": mase(y_true, y_pred, y_naive),
        "sMAPE": smape(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAE": float(np.mean(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float)))),
        "n": int(len(y_true)),
    }


def classification_report_full(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true, int)
    y_prob = np.asarray(y_prob, float)
    y_pred = (y_prob >= threshold).astype(int)
    out = {
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "Brier": float(brier_score_loss(y_true, y_prob)),
        "n": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "ROC_AUC": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
    }
    return out
