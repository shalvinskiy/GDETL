"""
External regressor #1 (required, minimum 1 additional open source): Google Trends
search interest for the keyword configured in config.yaml (`external_regressor`),
fetched via `pytrends` (unofficial Google Trends API wrapper, no auth required).

Notes / documented limitations (see README "Limitations"):
  - Google Trends returns *relative* search interest (0-100, scaled to the max
    in the requested window), not absolute volumes -- treat as a sentiment /
    salience proxy, not a literal counts.
  - For windows longer than ~9 months, Google returns WEEKLY points anchored
    on Sunday; we re-anchor to the Monday-start ISO weeks used everywhere else
    in this pipeline (label each Sunday-start week's value onto the Monday
    that falls inside the same ISO week, i.e. shift +1 day).
  - Google Trends data is essentially available in near real time (no material
    publication lag), but we still lag it by 1 week when used as a model
    feature (see feature_engineering.py) purely as a conservative safety
    margin against any provider-side backfill/revision.
  - The endpoint can rate-limit / fail intermittently in sandboxed or shared-IP
    environments. We cache results to disk and fail soft: if the fetch fails,
    the pipeline continues without this feature and logs a warning instead of
    crashing (documented in README limitations).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils import load_config, repo_path, week_start_monday  # noqa: E402


def fetch_google_trends_weekly(config: dict, force: bool = False) -> pd.DataFrame:
    cfg_data = config["data"]
    cache_path = repo_path(cfg_data["trends_cache_path"])
    if cache_path.exists() and not force:
        print(f"[fetch_google_trends] Using cached file {cache_path}")
        return pd.read_csv(cache_path, parse_dates=["week_start"])

    keyword = config["external_regressor"]["keyword"]
    geo = config["external_regressor"].get("geo", "")
    start = cfg_data["start_date"]
    end = cfg_data["end_date"]

    try:
        from pytrends.request import TrendReq

        pt = TrendReq(hl="en-US", tz=0)
        timeframe = f"{start} {end}"
        pt.build_payload([keyword], timeframe=timeframe, geo=geo)
        raw = pt.interest_over_time()
        if raw.empty:
            raise RuntimeError("pytrends returned an empty dataframe")
        raw = raw.reset_index().rename(columns={"date": "trend_date", keyword: "google_trends_score"})
        raw["week_start"] = raw["trend_date"].apply(lambda d: week_start_monday(d.date()))
        out = raw.groupby("week_start", as_index=False)["google_trends_score"].mean()
        out["source_keyword"] = keyword
        out["fetched_at"] = pd.Timestamp.utcnow().isoformat()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(cache_path, index=False)
        print(f"[fetch_google_trends] Fetched {len(out)} weekly points for '{keyword}' -> {cache_path}")
        return out
    except Exception as exc:  # noqa: BLE001 - external service, fail soft
        print(f"[fetch_google_trends] WARNING: could not fetch Google Trends ({exc}). "
              f"Pipeline will continue without this external regressor.")
        return pd.DataFrame(columns=["week_start", "google_trends_score", "source_keyword", "fetched_at"])


if __name__ == "__main__":
    cfg = load_config()
    df = fetch_google_trends_weekly(cfg)
    print(df.head())
    print(df.tail())
