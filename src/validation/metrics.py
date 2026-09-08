"""Forecast accuracy metrics for the regression (Variant A) and classification
(Variant B) targets."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0 + eps
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)


def mase(y_true: np.ndarray, y_pred: np.ndarray, y_train_history: np.ndarray, seasonal_period: int = 1) -> float:
    """Mean Absolute Scaled Error, scaled by the in-sample naive (persistence /
    seasonal-naive) MAE computed on the TRAINING history only (never on the
    test fold), matching the assignment's 'vs Baseline' MASE=1.00 convention.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_train_history = np.asarray(y_train_history, dtype=float)
    if len(y_train_history) <= seasonal_period:
        scale = np.nan
    else:
        naive_errors = np.abs(y_train_history[seasonal_period:] - y_train_history[:-seasonal_period])
        scale = np.mean(naive_errors)
    if not scale or np.isnan(scale) or scale == 0:
        scale = 1e-8
    mae = np.mean(np.abs(y_true - y_pred))
    return float(mae / scale)


def regression_report(y_true, y_pred, y_train_history, seasonal_period: int = 1) -> dict:
    return {
        "MASE": mase(y_true, y_pred, y_train_history, seasonal_period),
        "sMAPE": smape(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAE": float(np.mean(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)))),
        "n": int(len(y_true)),
    }


def classification_report_full(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    out = {
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "Brier": float(brier_score_loss(y_true, y_prob)),
        "n": int(len(y_true)),
        "n_positive": int(y_true.sum()),
    }
    if len(np.unique(y_true)) > 1:
        out["ROC_AUC"] = float(roc_auc_score(y_true, y_prob))
    else:
        out["ROC_AUC"] = float("nan")
    return out
