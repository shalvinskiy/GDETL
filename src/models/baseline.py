"""Naive persistence and rolling-mean baselines."""
from __future__ import annotations

import numpy as np
import pandas as pd


def predict_naive(df: pd.DataFrame, horizon: int) -> np.ndarray:
    return df["conflict_count_t0"].to_numpy(dtype=float)


def predict_rolling_mean(df: pd.DataFrame, horizon: int, window_col: str = "rolling_mean_4w") -> np.ndarray:
    vals = df[window_col].to_numpy(dtype=float)
    fallback = df["conflict_count_t0"].to_numpy(dtype=float)
    return np.where(np.isnan(vals), fallback, vals)


def predict_escalation_naive(df: pd.DataFrame, horizon: int) -> np.ndarray:
    return df["escalation_event"].astype(float).to_numpy()
