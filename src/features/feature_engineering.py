"""
Feature engineering for the weekly conflict-escalation panel.

Conventions (documented explicitly to make the leakage story auditable):

  * Row `t` = the forecast ORIGIN week, i.e. the most recently fully-observed
    week. All feature values on row `t` must be computable using information
    available by the end of week `t` only.
  * Targets are created by SHIFTING FORWARD: `target_h = conflict_count.shift(-h)`
    for h in {1,2,3,4}. This is the only place the future appears, and only in
    the target column, never in a feature column.
  * Lag features: `conflict_count_lag{k} = conflict_count.shift(k)`, i.e. the
    count observed k weeks BEFORE the origin week t. The origin week's own,
    fully-observed count is also kept as a feature (`conflict_count_t0`)
    because it is legitimately known at prediction time for every horizon
    h >= 1.
  * Rolling stats (`rolling_mean_4w`, `rolling_std_12w`, `rolling_max_8w`, ...)
    use pandas `.rolling(window)` directly on the (unshifted) series -- by
    pandas convention the window for row `t` covers `[t-window+1, t]`, i.e.
    only current + past values. No future leakage.
  * The escalation threshold/label (Variant B) is intentionally computed on
    values shifted by 1 (`rolling_mean_52w`/`rolling_std_52w` over the 52
    weeks strictly BEFORE `t`), so the label at `t` never uses `t`'s own count
    to define its own anomaly threshold (which would dampen genuine spikes).
  * External regressor (Google Trends) is used contemporaneously at `t`
    (Trends has ~zero publication lag) plus a lag-1 version for a
    conservative variant; both are computed from the already-lag-safe weekly
    panel, no future values used.
  * Scaling / normalization is deliberately NOT done in this module -- it is
    fit inside each walk-forward CV fold on the TRAIN slice only (see
    src/validation/walk_forward.py) to avoid any cross-fold leakage.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils import load_config, repo_path  # noqa: E402

LAGS = [1, 2, 4, 12]
HORIZONS_DEFAULT = [1, 2, 3, 4]


def _weeks_since_last_true(flags: pd.Series) -> pd.Series:
    """For each position, count weeks since the last True (excluding current),
    using only past information. NaN / large number if none yet seen."""
    out = np.full(len(flags), np.nan)
    last_true_idx = None
    vals = flags.to_numpy()
    for i in range(len(vals)):
        if last_true_idx is None:
            out[i] = i + 1  # no spike observed yet in history -> "since start"
        else:
            out[i] = i - last_true_idx
        if vals[i]:
            last_true_idx = i
    return pd.Series(out, index=flags.index)


def add_escalation_label(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Variant B target: escalation_event_t = 1 if conflict_count_t > threshold_t,
    threshold_t = rolling_mean_52w + k * rolling_std_52w computed over the 52
    weeks strictly BEFORE t (shift(1) window) -- see module docstring."""
    df = df.copy()
    w = config["target"]["escalation_rolling_window_weeks"]
    k = config["target"]["escalation_zscore_k"]
    shifted = df["conflict_count"].shift(1)
    roll_mean = shifted.rolling(w, min_periods=max(8, w // 4)).mean()
    roll_std = shifted.rolling(w, min_periods=max(8, w // 4)).std()
    df["escalation_threshold"] = roll_mean + k * roll_std
    df["escalation_event"] = (df["conflict_count"] > df["escalation_threshold"]).astype("Int64")
    # Undefined while history < min_periods
    df.loc[df["escalation_threshold"].isna(), "escalation_event"] = pd.NA
    return df


def build_features(weekly: pd.DataFrame, config: dict, horizons: list[int] | None = None) -> pd.DataFrame:
    horizons = horizons or config["target"]["horizons"]
    df = weekly.sort_values("week_start").reset_index(drop=True).copy()

    # ---- data-quality mask: weeks with incomplete GDELT coverage (see
    # build_weekly_panel._compute_days_covered -- a real ~18-day GDELT-wide
    # outage, 2025-06-14..2025-07-01, was hit while building this dataset)
    # have an artificially LOW conflict_count and must never be used as a
    # training/evaluation TARGET, even though they can still serve as (
    # caveated) historical feature inputs. ----
    if "days_covered" not in df.columns:
        df["days_covered"] = 7
    incomplete_week = df["days_covered"].fillna(7) < 7

    # ---- target(s): Variant A regression, direct multi-horizon ----
    for h in horizons:
        tgt = df["conflict_count"].shift(-h).astype(float)
        tgt[incomplete_week.shift(-h).fillna(False)] = float("nan")
        df[f"target_count_h{h}"] = tgt

    # ---- escalation label (Variant B), computed BEFORE lag features so it
    # can also be shifted forward into future-facing target columns ----
    df = add_escalation_label(df, config)
    for h in horizons:
        evt = df["escalation_event"].shift(-h)
        evt[incomplete_week.shift(-h).fillna(False)] = pd.NA
        df[f"target_event_h{h}"] = evt

    # ---- lag features on target quantity ----
    df["conflict_count_t0"] = df["conflict_count"]
    for k in LAGS:
        df[f"conflict_count_lag{k}"] = df["conflict_count"].shift(k)

    # ---- rolling stats (window ends at t, current value included) ----
    df["rolling_mean_4w"] = df["conflict_count"].rolling(4, min_periods=2).mean()
    df["rolling_std_12w"] = df["conflict_count"].rolling(12, min_periods=4).std()
    df["rolling_max_8w"] = df["conflict_count"].rolling(8, min_periods=2).max()
    df["rolling_mean_12w"] = df["conflict_count"].rolling(12, min_periods=4).mean()

    # ---- tone / sentiment (already contemporaneous-safe, computed in the
    # weekly panel strictly from week t's own events) ----
    df["avg_goldstein"] = df["avg_goldstein"]
    df["avg_tone"] = df["avg_tone"]
    df["goldstein_volatility"] = df["goldstein_volatility"]
    # forward-fill tone features on weeks with zero conflict events (no
    # Goldstein/tone observation that week) using only PAST values.
    for c in ["avg_goldstein", "avg_tone", "goldstein_volatility"]:
        df[c] = df[c].ffill()

    # ---- news volume ----
    df["total_events_country"] = df["total_events_country"]
    df["news_volume_change_wow"] = df["total_events_country"].pct_change().replace([np.inf, -np.inf], np.nan)

    # ---- calendar ----
    week_start_dt = pd.to_datetime(df["week_start"])
    df["week_of_year"] = week_start_dt.dt.isocalendar().week.astype(int)
    df["month"] = week_start_dt.dt.month
    try:
        import holidays as pyholidays

        ua_holidays = pyholidays.country_holidays("UA")
        df["is_holiday"] = week_start_dt.apply(
            lambda d: int(any((d + pd.Timedelta(days=i)) in ua_holidays for i in range(7)))
        )
    except Exception:  # noqa: BLE001
        df["is_holiday"] = 0

    # ---- external regressor: Google Trends ----
    if "google_trends_score" in df.columns:
        df["google_trends_score_lag1"] = df["google_trends_score"].shift(1)
        df["google_trends_change_wow"] = df["google_trends_score"].pct_change().replace([np.inf, -np.inf], np.nan)
    else:
        df["google_trends_score"] = np.nan
        df["google_trends_score_lag1"] = np.nan
        df["google_trends_change_wow"] = np.nan

    # ---- trend / regime features ----
    is_spike = (df["escalation_event"] == 1).fillna(False)
    df["weeks_since_major_spike"] = _weeks_since_last_true(is_spike)
    df["cumulative_conflict_12w"] = df["conflict_count"].rolling(12, min_periods=1).sum().shift(1)

    out_path = repo_path(config["data"]["weekly_panel_path"]).with_name("feature_panel.parquet")
    df.to_parquet(out_path, index=False)
    df.to_csv(out_path.with_suffix(".csv"), index=False)
    print(f"[feature_engineering] Built feature panel: {df.shape[0]} rows x {df.shape[1]} cols -> {out_path}")
    return df


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


if __name__ == "__main__":
    cfg = load_config()
    weekly_path = repo_path(cfg["data"]["weekly_panel_path"])
    weekly = pd.read_parquet(weekly_path)
    build_features(weekly, cfg)
