"""Holt-Winters ETS. Falls back to simpler specs on short folds."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def fit_and_forecast(train_series: pd.Series, steps: int) -> np.ndarray:
    y = np.clip(train_series.astype(float).to_numpy(), 0.1, None)

    specs = []
    if len(y) >= 104:
        specs.append(dict(trend="add", damped_trend=True, seasonal="add", seasonal_periods=52))
    if len(y) >= 10:
        specs.append(dict(trend="add", damped_trend=True, seasonal=None))
    specs.append(dict(trend=None, seasonal=None))

    for spec in specs:
        try:
            fitted = ExponentialSmoothing(y, initialization_method="estimated", **spec).fit(optimized=True)
            return np.maximum(np.asarray(fitted.forecast(steps), dtype=float), 0.0)
        except Exception:
            continue
    return np.full(steps, float(np.mean(y)))
