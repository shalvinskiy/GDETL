"""Google Trends weekly interest for the keyword in config.yaml."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils import load_config, repo_path, week_start_monday


def fetch_google_trends_weekly(config: dict, force: bool = False) -> pd.DataFrame:
    cache_path = repo_path(config["data"]["trends_cache_path"])
    if cache_path.exists() and not force:
        print(f"[trends] cache {cache_path}")
        return pd.read_csv(cache_path, parse_dates=["week_start"])

    keyword = config["external_regressor"]["keyword"]
    geo = config["external_regressor"].get("geo", "")
    start = config["data"]["start_date"]
    end = config["data"]["end_date"]

    try:
        from pytrends.request import TrendReq

        pt = TrendReq(hl="en-US", tz=0)
        pt.build_payload([keyword], timeframe=f"{start} {end}", geo=geo)
        raw = pt.interest_over_time()
        if raw.empty:
            raise RuntimeError("empty response")
        raw = raw.reset_index().rename(columns={"date": "trend_date", keyword: "google_trends_score"})
        raw["week_start"] = raw["trend_date"].apply(lambda d: week_start_monday(d.date()))
        out = raw.groupby("week_start", as_index=False)["google_trends_score"].mean()
        out["source_keyword"] = keyword
        out["fetched_at"] = pd.Timestamp.utcnow().isoformat()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(cache_path, index=False)
        print(f"[trends] {len(out)} weeks for '{keyword}' -> {cache_path}")
        return out
    except Exception as exc:
        print(f"[trends] fetch failed ({exc}), continuing without it")
        return pd.DataFrame(columns=["week_start", "google_trends_score", "source_keyword", "fetched_at"])


if __name__ == "__main__":
    df = fetch_google_trends_weekly(load_config(), force=True)
    print(df.head())
    print(df.tail())
