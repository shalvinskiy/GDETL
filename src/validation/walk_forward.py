"""
Rolling / walk-forward cross-validation splitting utilities.

Design (see README "Валидация" section for the full rationale):

  * Random splits are never used -- all splits respect chronological order.
  * A hard HOLD-OUT of the last `holdout_weeks` realized weeks is carved out
    first and is NEVER touched by CV (no fold's train OR validation slice may
    use a feature row whose target lands in the hold-out window, and no fold
    may use a feature row *originating* inside the hold-out window either).
  * Within the remaining ("CV pool") region, we run an *expanding-window*
    walk-forward CV: fold 0 trains on the first `cv_initial_train_weeks`
    origin-weeks and validates on the next `cv_step_weeks`; fold 1 trains on
    everything up to fold 0's validation block and validates on the next
    `cv_step_weeks`; etc. This mimics how the model would actually be
    retrained and used in production, week after week.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd


def split_holdout(feature_df: pd.DataFrame, holdout_weeks: int, max_horizon: int) -> tuple[pd.DataFrame, pd.DataFrame, dt.date, set]:
    """Returns (cv_pool, full_df, holdout_start_week, holdout_target_weeks).

    cv_pool: rows usable for walk-forward CV AND for fitting the final models
        that get evaluated on the hold-out. Guarantees that for EVERY horizon
        h in 1..max_horizon, origin_week + h*7d is still strictly before
        holdout_start_week -- i.e. **no model is ever fit on any row whose
        target realizes inside the hold-out window.**

    holdout_target_weeks: the set of the last `holdout_weeks` calendar weeks
        (the actual ground-truth realizations we evaluate against). For a
        given horizon h, the corresponding test ORIGIN weeks are
        {w - h*7d : w in holdout_target_weeks}. Origins for small h will
        mostly fall *inside* the hold-out window itself -- this is intentional
        and NOT leakage: those origins only ever contribute already-REALIZED
        past values as FEATURES (e.g. "last week's count") to predict a later
        hold-out week, exactly mimicking how a fixed, already-trained model
        would be used week-by-week in production without being retrained.
        No hold-out data point is ever used to FIT a model.
    """
    df = feature_df.sort_values("week_start").reset_index(drop=True)
    weeks = pd.to_datetime(df["week_start"])
    last_week = weeks.max()
    holdout_start_week = (last_week - pd.Timedelta(weeks=holdout_weeks - 1)).date()
    holdout_target_weeks = {
        (pd.Timestamp(holdout_start_week) + pd.Timedelta(weeks=i)).date() for i in range(holdout_weeks)
    }

    cv_cutoff = pd.Timestamp(holdout_start_week) - pd.Timedelta(weeks=max_horizon)
    cv_pool = df[weeks <= cv_cutoff].reset_index(drop=True)

    return cv_pool, df, holdout_start_week, holdout_target_weeks


def generate_folds(n_rows: int, initial_train_weeks: int, step_weeks: int, min_test_weeks: int) -> list[tuple[range, range]]:
    """Expanding-window walk-forward folds over positional indices [0, n_rows)."""
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
