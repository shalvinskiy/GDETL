"""Statistical benchmark: Holt-Winters Exponential Smoothing (ETS) via
statsmodels, fit on the univariate weekly conflict_count series.

We progressively fall back to simpler specifications so that the model can
always be fit, even on the short training windows seen in the earliest CV
folds:
  1. ETS with additive trend + additive 52-week seasonality (needs >= 2 full
     seasonal cycles, i.e. >= 104 weeks of history).
  2. ETS with additive trend, no seasonality (needs >= ~10 weeks).
  3. Simple Exponential Smoothing (no trend/seasonality) as the last resort.
All specifications use `damped_trend=True` where a trend is present, since an
undamped linear trend is not a sensible long-run assumption for a bounded
conflict-event count.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def fit_and_forecast(train_series: pd.Series, steps: int) -> np.ndarray:
    y = train_series.astype(float).to_numpy()
    y = np.clip(y, a_min=0.1, a_max=None)  # ETS with mult. components needs > 0; counts are >= 0

    specs = []
    if len(y) >= 104:
        specs.append(dict(trend="add", damped_trend=True, seasonal="add", seasonal_periods=52))
    if len(y) >= 10:
        specs.append(dict(trend="add", damped_trend=True, seasonal=None))
    specs.append(dict(trend=None, seasonal=None))

    last_exc = None
    for spec in specs:
        try:
            model = ExponentialSmoothing(y, initialization_method="estimated", **spec)
            fitted = model.fit(optimized=True)
            fc = fitted.forecast(steps)
            fc = np.maximum(fc, 0.0)
            return np.asarray(fc, dtype=float)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            continue
    # absolute last resort: flat forecast at the training mean
    return np.full(steps, float(np.mean(y)))
