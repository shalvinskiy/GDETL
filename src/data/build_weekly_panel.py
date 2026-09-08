"""
Aggregate raw (daily-granularity) GDELT events into the weekly panel that the
rest of the pipeline consumes.

Aggregation follows the exact definition given in the assignment (section 4.2
example SQL): conflict_count / avg_goldstein / avg_tone / goldstein_volatility
are computed over the SAME filtered subset of conflict events
(EventRootCode IN ('18','19')), NOT over all country events. Total country
news volume is tracked separately (all events, any type) to power the
"news volume" feature group.

Week definition: ISO week, Monday start (`week_start`), matching
`DATE_TRUNC(..., WEEK(MONDAY))` in the assignment's BigQuery example.

Output columns (one row per week_start):
  week_start, week_label (e.g. 2024-W35),
  conflict_count            -- target quantity, conflict events that week
  avg_goldstein             -- mean GoldsteinScale of conflict events
  avg_tone                  -- mean AvgTone of conflict events
  goldstein_volatility      -- std GoldsteinScale of conflict events
  total_events_country      -- ALL events for the country that week (any type)
  total_mentions_conflict   -- sum NumMentions of conflict events (extra volume signal)
  google_trends_score       -- external regressor (may be NaN if fetch failed)
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.download_gdelt_bulk import combine_daily_files  # noqa: E402
from src.data.fetch_google_trends import fetch_google_trends_weekly  # noqa: E402
from src.utils import iso_week_label, load_config, read_json, repo_path, week_start_monday  # noqa: E402


def _compute_days_covered(config: dict, full_index) -> pd.Series:
    """Data-quality guard: GDELT occasionally has multi-day outages (see
    README Limitations -- a real ~18-day gap, 2025-06-14..2025-07-01, was hit
    and confirmed missing in BOTH the bulk-CSV and GDELT 2.0 15-min products
    while building this dataset). For each week, count how many of its 7
    calendar days were actually successfully downloaded (per the ingestion
    manifest). Weeks with `days_covered < 7` have an artificially LOW
    conflict_count that must not be trusted as a training/evaluation target
    -- see feature_engineering.py, which sets target_count_h{h} to NaN for
    any horizon landing on such a week."""
    manifest = read_json(repo_path(config["data"]["download_manifest_path"]))
    downloaded = set(manifest.get("downloaded_days", {}).keys())
    counts = []
    for week_start in full_index:
        n = sum(1 for i in range(7) if (week_start + dt.timedelta(days=i)).strftime("%Y%m%d") in downloaded)
        counts.append(n)
    return pd.Series(counts, index=full_index)


def build_weekly_panel(config: dict, raw_df: pd.DataFrame | None = None) -> pd.DataFrame:
    cfg_data = config["data"]
    if raw_df is None:
        combined_path = repo_path(cfg_data["raw_combined_path"])
        if combined_path.exists():
            raw_df = pd.read_parquet(combined_path)
        else:
            raw_df = combine_daily_files(config)

    df = raw_df.copy()
    df["week_start"] = df["event_date"].apply(lambda d: week_start_monday(d.date()))

    conflict = df[df["is_conflict"]]
    weekly_conflict = conflict.groupby("week_start").agg(
        conflict_count=("GlobalEventID", "count"),
        avg_goldstein=("GoldsteinScale", "mean"),
        avg_tone=("AvgTone", "mean"),
        goldstein_volatility=("GoldsteinScale", "std"),
        total_mentions_conflict=("NumMentions", "sum"),
    ).reset_index()

    weekly_total = df.groupby("week_start").agg(
        total_events_country=("GlobalEventID", "count"),
    ).reset_index()

    weekly = weekly_total.merge(weekly_conflict, on="week_start", how="left")
    weekly["conflict_count"] = weekly["conflict_count"].fillna(0).astype(int)

    # Reindex to a complete, gap-free weekly calendar over the configured range
    # so that weeks with literally zero matching events are represented as
    # conflict_count = 0 rather than silently missing (important for lags /
    # rolling stats to be computed on a regular time index).
    start = week_start_monday(cfg_data["start_date"])
    end = week_start_monday(cfg_data["end_date"])
    full_index = pd.date_range(start, end, freq="7D").date
    weekly = weekly.set_index("week_start").reindex(full_index)
    weekly.index.name = "week_start"
    weekly["conflict_count"] = weekly["conflict_count"].fillna(0).astype(int)
    weekly["total_events_country"] = weekly["total_events_country"].fillna(0).astype(int)
    days_covered = _compute_days_covered(config, full_index)
    weekly = weekly.reset_index()
    weekly["days_covered"] = weekly["week_start"].map(days_covered)
    weekly["week_label"] = weekly["week_start"].apply(iso_week_label)
    n_incomplete = int((weekly["days_covered"] < 7).sum())
    if n_incomplete:
        print(f"[build_weekly_panel] WARNING: {n_incomplete} week(s) have incomplete GDELT coverage "
              f"(<7/7 days downloaded) -- their conflict_count is a lower bound, not trustworthy as a "
              f"target. Flagged via `days_covered`; see feature_engineering.py target masking.")
        print(weekly.loc[weekly["days_covered"] < 7, ["week_start", "week_label", "days_covered", "conflict_count"]]
              .to_string(index=False))

    # Drop the last (possibly partial) week if end_date doesn't align to a
    # Sunday boundary -- avoids a truncated final observation biasing rolling
    # stats / targets.
    last_full_week_start = week_start_monday(dt.datetime.strptime(cfg_data["end_date"], "%Y-%m-%d").date() - dt.timedelta(days=6))
    weekly = weekly[weekly["week_start"] <= last_full_week_start].reset_index(drop=True)

    # Merge external regressor (Google Trends), lag-safe join on week_start.
    trends = fetch_google_trends_weekly(config)
    if not trends.empty:
        trends = trends.copy()
        trends["week_start"] = pd.to_datetime(trends["week_start"]).dt.date
        weekly["week_start_dt"] = pd.to_datetime(weekly["week_start"])
        trends_small = trends[["week_start", "google_trends_score"]]
        weekly = weekly.merge(trends_small, on="week_start", how="left")
        weekly = weekly.drop(columns=["week_start_dt"])
    else:
        weekly["google_trends_score"] = float("nan")

    weekly = weekly.sort_values("week_start").reset_index(drop=True)

    out_path = repo_path(cfg_data["weekly_panel_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    weekly.to_parquet(out_path, index=False)
    csv_path = out_path.with_suffix(".csv")
    weekly.to_csv(csv_path, index=False)
    print(f"[build_weekly_panel] Built weekly panel: {len(weekly)} weeks "
          f"({weekly['week_start'].min()} .. {weekly['week_start'].max()}) -> {out_path}")
    print(weekly[["week_start", "week_label", "conflict_count", "total_events_country",
                   "avg_goldstein", "avg_tone", "google_trends_score"]].tail(8).to_string(index=False))
    return weekly


if __name__ == "__main__":
    cfg = load_config()
    build_weekly_panel(cfg)
