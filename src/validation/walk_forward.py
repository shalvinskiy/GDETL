"""Walk-forward split: hold-out last N weeks, expanding-window CV on the rest."""
from __future__ import annotations

import pandas as pd


def split_holdout(feature_df: pd.DataFrame, holdout_weeks: int, max_horizon: int):
    """cv_pool never includes a row whose target (up to max_horizon) falls in the hold-out."""
    df = feature_df.sort_values("week_start").reset_index(drop=True)
    weeks = pd.to_datetime(df["week_start"])
    last_week = weeks.max()
    holdout_start = (last_week - pd.Timedelta(weeks=holdout_weeks - 1)).date()
    holdout_target_weeks = {
        (pd.Timestamp(holdout_start) + pd.Timedelta(weeks=i)).date() for i in range(holdout_weeks)
    }
    cv_cutoff = pd.Timestamp(holdout_start) - pd.Timedelta(weeks=max_horizon)
    cv_pool = df[weeks <= cv_cutoff].reset_index(drop=True)
    return cv_pool, df, holdout_start, holdout_target_weeks


def generate_folds(n_rows: int, initial_train_weeks: int, step_weeks: int, min_test_weeks: int):
    folds = []
    train_end = initial_train_weeks
    while True:
        test_start = train_end
        test_end = min(test_start + step_weeks, n_rows)
        if test_end - test_start < min_test_weeks:
            break
        folds.append((range(0, train_end), range(test_start, test_end)))
        train_end = test_end
        if train_end >= n_rows:
            break
    return folds
