"""Lag / rolling / calendar features. All values on row t are known at the end of week t.

Targets are forward shifts of conflict_count / escalation_event — they never go into X.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils import load_config, repo_path

LAGS = [1, 2, 4, 12]
HORIZONS_DEFAULT = [1, 2, 3, 4]

FEATURE_COLUMNS = [
    "conflict_count_t0",
    "conflict_count_lag1", "conflict_count_lag2", "conflict_count_lag4", "conflict_count_lag12",
    "rolling_mean_4w", "rolling_std_12w", "rolling_max_8w", "rolling_mean_12w",
    "avg_goldstein", "avg_tone", "goldstein_volatility",
    "total_events_country", "news_volume_change_wow",
    "week_of_year", "month", "is_holiday",
    "google_trends_score", "google_trends_score_lag1", "google_trends_change_wow",
    "weeks_since_major_spike", "cumulative_conflict_12w",
]
BASE_FEATURE_COLUMNS = list(FEATURE_COLUMNS)


def get_feature_columns(df: pd.DataFrame | None = None) -> list[str]:
    return list(FEATURE_COLUMNS)


def _weeks_since_last_true(flags: pd.Series) -> pd.Series:
    out = np.full(len(flags), np.nan)
    last = None
    vals = flags.to_numpy()
    for i, v in enumerate(vals):
        out[i] = i + 1 if last is None else i - last
        if v:
            last = i
    return pd.Series(out, index=flags.index)


def add_escalation_label(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """escalation = conflict_count > rolling_mean_52w + k * rolling_std_52w, window strictly before t."""
    df = df.copy()
    w = config["target"]["escalation_rolling_window_weeks"]
    k = config["target"]["escalation_zscore_k"]
    shifted = df["conflict_count"].shift(1)
    roll_mean = shifted.rolling(w, min_periods=max(8, w // 4)).mean()
    roll_std = shifted.rolling(w, min_periods=max(8, w // 4)).std()
    df["escalation_threshold"] = roll_mean + k * roll_std
    df["escalation_event"] = (df["conflict_count"] > df["escalation_threshold"]).astype("Int64")
    df.loc[df["escalation_threshold"].isna(), "escalation_event"] = pd.NA
    return df


def build_features(weekly: pd.DataFrame, config: dict, horizons: list[int] | None = None) -> pd.DataFrame:
    horizons = horizons or config["target"]["horizons"]
    df = weekly.sort_values("week_start").reset_index(drop=True).copy()

    if "days_covered" not in df.columns:
        df["days_covered"] = 7
    incomplete = df["days_covered"].fillna(7) < 7

    for h in horizons:
        tgt = df["conflict_count"].shift(-h).astype(float)
        tgt[incomplete.shift(-h).fillna(False)] = float("nan")
        df[f"target_count_h{h}"] = tgt

    df = add_escalation_label(df, config)
    for h in horizons:
        evt = df["escalation_event"].shift(-h)
        evt[incomplete.shift(-h).fillna(False)] = pd.NA
        df[f"target_event_h{h}"] = evt

    df["conflict_count_t0"] = df["conflict_count"]
    for k in LAGS:
        df[f"conflict_count_lag{k}"] = df["conflict_count"].shift(k)

    df["rolling_mean_4w"] = df["conflict_count"].rolling(4, min_periods=2).mean()
    df["rolling_std_12w"] = df["conflict_count"].rolling(12, min_periods=4).std()
    df["rolling_max_8w"] = df["conflict_count"].rolling(8, min_periods=2).max()
    df["rolling_mean_12w"] = df["conflict_count"].rolling(12, min_periods=4).mean()

    for c in ["avg_goldstein", "avg_tone", "goldstein_volatility"]:
        df[c] = df[c].ffill()

    df["news_volume_change_wow"] = df["total_events_country"].pct_change().replace([np.inf, -np.inf], np.nan)

    week_start_dt = pd.to_datetime(df["week_start"])
    df["week_of_year"] = week_start_dt.dt.isocalendar().week.astype(int)
    df["month"] = week_start_dt.dt.month
    try:
        import holidays as pyholidays

        cal = pyholidays.country_holidays(config["country"]["iso2"])
        df["is_holiday"] = week_start_dt.apply(
            lambda d: int(any((d + pd.Timedelta(days=i)) in cal for i in range(7)))
        )
    except Exception:
        df["is_holiday"] = 0

    if "google_trends_score" in df.columns:
        df["google_trends_score_lag1"] = df["google_trends_score"].shift(1)
        df["google_trends_change_wow"] = df["google_trends_score"].pct_change().replace([np.inf, -np.inf], np.nan)
    else:
        df["google_trends_score"] = np.nan
        df["google_trends_score_lag1"] = np.nan
        df["google_trends_change_wow"] = np.nan

    is_spike = (df["escalation_event"] == 1).fillna(False)
    df["weeks_since_major_spike"] = _weeks_since_last_true(is_spike)
    df["cumulative_conflict_12w"] = df["conflict_count"].rolling(12, min_periods=1).sum().shift(1)

    out_path = repo_path(config["data"]["weekly_panel_path"]).with_name("feature_panel.parquet")
    df.to_parquet(out_path, index=False)
    df.to_csv(out_path.with_suffix(".csv"), index=False)
    print(f"[features] {df.shape[0]} rows x {df.shape[1]} cols")
    return df


if __name__ == "__main__":
    cfg = load_config()
    weekly = pd.read_parquet(repo_path(cfg["data"]["weekly_panel_path"]))
    build_features(weekly, cfg)
