"""Weekly aggregation of GDELT events (ISO week, Monday start)."""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.download_gdelt_bulk import combine_daily_files
from src.data.fetch_google_trends import fetch_google_trends_weekly
from src.utils import iso_week_label, load_config, read_json, repo_path, week_start_monday


def _days_covered(config: dict, full_index) -> pd.Series:
    """How many of the 7 calendar days in each week were actually downloaded."""
    manifest = read_json(repo_path(config["data"]["download_manifest_path"]))
    downloaded = set(manifest.get("downloaded_days", {}))
    counts = []
    for week_start in full_index:
        n = sum(1 for i in range(7) if (week_start + dt.timedelta(days=i)).strftime("%Y%m%d") in downloaded)
        counts.append(n)
    return pd.Series(counts, index=full_index)


def build_weekly_panel(config: dict, raw_df: pd.DataFrame | None = None) -> pd.DataFrame:
    cfg_data = config["data"]
    if raw_df is None:
        combined_path = repo_path(cfg_data["raw_combined_path"])
        raw_df = pd.read_parquet(combined_path) if combined_path.exists() else combine_daily_files(config)

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

    start = week_start_monday(cfg_data["start_date"])
    end = week_start_monday(cfg_data["end_date"])
    full_index = pd.date_range(start, end, freq="7D").date
    weekly = weekly.set_index("week_start").reindex(full_index)
    weekly.index.name = "week_start"
    weekly["conflict_count"] = weekly["conflict_count"].fillna(0).astype(int)
    weekly["total_events_country"] = weekly["total_events_country"].fillna(0).astype(int)
    days_covered = _days_covered(config, full_index)
    weekly = weekly.reset_index()
    weekly["days_covered"] = weekly["week_start"].map(days_covered)
    weekly["week_label"] = weekly["week_start"].apply(iso_week_label)

    n_incomplete = int((weekly["days_covered"] < 7).sum())
    if n_incomplete:
        print(f"[weekly] {n_incomplete} week(s) with incomplete GDELT coverage")
        print(weekly.loc[weekly["days_covered"] < 7,
                         ["week_start", "week_label", "days_covered", "conflict_count"]].to_string(index=False))

    last_full = week_start_monday(
        dt.datetime.strptime(cfg_data["end_date"], "%Y-%m-%d").date() - dt.timedelta(days=6)
    )
    weekly = weekly[weekly["week_start"] <= last_full].reset_index(drop=True)

    trends = fetch_google_trends_weekly(config)
    if not trends.empty:
        trends = trends.copy()
        trends["week_start"] = pd.to_datetime(trends["week_start"]).dt.date
        weekly = weekly.merge(trends[["week_start", "google_trends_score"]], on="week_start", how="left")
    else:
        weekly["google_trends_score"] = float("nan")

    weekly = weekly.sort_values("week_start").reset_index(drop=True)
    out_path = repo_path(cfg_data["weekly_panel_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    weekly.to_parquet(out_path, index=False)
    weekly.to_csv(out_path.with_suffix(".csv"), index=False)
    print(f"[weekly] {len(weekly)} weeks ({weekly['week_start'].min()} .. {weekly['week_start'].max()})")
    return weekly


if __name__ == "__main__":
    build_weekly_panel(load_config())
