"""Baseline forecasters -- the mandatory 'floor' every other model must beat.

Two variants implemented (assignment allows either "seasonal naive or rolling
mean"); both are reported, the better one (lower CV MASE) is used as the
official baseline row in the results table.

Both are pure feature look-ups: they need no fitting at all because the
relevant quantity (last known value / trailing rolling mean) is already a
leakage-safe feature computed at the forecast origin week `t`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def predict_naive(df: pd.DataFrame, horizon: int) -> np.ndarray:
    """Persistence naive: predict the last fully-observed count (conflict_count_t0)
    unchanged for every horizon."""
    return df["conflict_count_t0"].to_numpy(dtype=float)


def predict_rolling_mean(df: pd.DataFrame, horizon: int, window_col: str = "rolling_mean_4w") -> np.ndarray:
    """Rolling-mean baseline: predict the trailing 4-week mean conflict count."""
    vals = df[window_col].to_numpy(dtype=float)
    fallback = df["conflict_count_t0"].to_numpy(dtype=float)
    return np.where(np.isnan(vals), fallback, vals)


def predict_escalation_naive(df: pd.DataFrame, horizon: int) -> np.ndarray:
    """Variant B baseline: probability of escalation = empirical escalation
    rate observed so far up to t (a fold-agnostic, leakage-safe running rate),
    approximated here by whether the ORIGIN week itself was already flagged
    as an escalation (persistence for the binary label)."""
    return df["escalation_event"].astype(float).to_numpy()
