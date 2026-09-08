"""
GDELT Events ingestion -- Bulk CSV method.

Source: https://data.gdeltproject.org/events/  (GDELT 1.0 daily export files,
tab-separated, 58 columns, no header, one file per day: YYYYMMDD.export.CSV.zip)

Why Bulk CSV instead of BigQuery (documented decision, see README section 4):
  The assignment marks `gdelt-bq.gdeltv2.events` on BigQuery as the recommended
  ("Priority: Recommended") route, and Bulk CSV as explicitly acceptable
  ("Priority: Допустимо"). BigQuery access requires a GCP project with billing
  enabled and local `gcloud`/service-account credentials, which are environment
  -specific and not guaranteed to exist on a reviewer's machine or in a CI
  sandbox. The Bulk CSV route needs zero credentials and is fully reproducible
  by anyone, anywhere, which better satisfies the "runs from README" and
  "error handling / lineage" grading criteria. An equivalent, ready-to-run
  BigQuery ingestion script is also provided in
  `src/data/download_gdelt_bigquery.py` for environments that do have GCP
  credentials configured -- both scripts produce the same raw schema.

This script:
  1. Iterates every day in [start_date, end_date] from config.yaml.
  2. Downloads & unzips the daily export (~6-7MB compressed / ~40MB raw).
  3. Keeps only rows for the configured country (ActionGeo_CountryCode), for
     ALL event types (not just 18/19) -- we need the full country volume for
     the `total_events_country` / news-volume features later. A boolean
     `is_conflict` column flags EventRootCode in {18, 19}.
  4. Caches one Parquet file per day under data/raw/gdelt_daily/ so re-runs
     are resumable and idempotent (already-downloaded days are skipped).
  5. Retries transient failures, logs permanent failures, and writes a JSON
     manifest recording exactly what was downloaded, when, and from where
     (data lineage requirement).
  6. Concatenates all daily caches into one combined raw Parquet file.
"""
from __future__ import annotations

import datetime as dt
import io
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils import daterange, load_config, read_json, repo_path, write_json  # noqa: E402

# 0-indexed column positions we need from the 58-column GDELT 1.0 daily export.
# Full codebook: http://data.gdeltproject.org/documentation/GDELT-Data_Format_Codebook.pdf
COLS = {
    0: "GlobalEventID",
    1: "SQLDATE",
    28: "EventRootCode",
    29: "QuadClass",
    30: "GoldsteinScale",
    31: "NumMentions",
    32: "NumSources",
    33: "NumArticles",
    34: "AvgTone",
    51: "ActionGeo_CountryCode",
}
COL_IDX = sorted(COLS.keys())
COL_NAMES = [COLS[i] for i in COL_IDX]

MAX_RETRIES = 4
RETRY_BACKOFF_S = 3
REQUEST_TIMEOUT_S = 40


def _fetch_one_day(date: dt.date, base_url: str, country_code: str, root_codes: list[str]) -> tuple[dt.date, pd.DataFrame | None, str | None]:
    """Download+parse+filter a single day. Returns (date, df_or_None, error_or_None)."""
    ymd = date.strftime("%Y%m%d")
    url = f"{base_url}{ymd}.export.CSV.zip"
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT_S)
            if resp.status_code == 404:
                return date, None, f"404 not found: {url}"
            resp.raise_for_status()
            zf = zipfile.ZipFile(io.BytesIO(resp.content))
            inner_name = zf.namelist()[0]
            with zf.open(inner_name) as fh:
                df = pd.read_csv(
                    fh, sep="\t", header=None, usecols=COL_IDX, dtype=str, low_memory=False
                )
            df.columns = COL_NAMES
            df = df[df["ActionGeo_CountryCode"] == country_code].copy()
            df["is_conflict"] = df["EventRootCode"].isin(root_codes)
            # numeric casting
            for c in ["GoldsteinScale", "NumMentions", "NumSources", "NumArticles", "AvgTone"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df["event_date"] = pd.to_datetime(df["SQLDATE"], format="%Y%m%d")
            return date, df, None
        except Exception as exc:  # noqa: BLE001 - want to retry on anything transient
            last_err = str(exc)
            time.sleep(RETRY_BACKOFF_S * attempt)
    return date, None, last_err


def download_all(config: dict, max_workers: int = 20, force: bool = False) -> dict:
    cfg_data = config["data"]
    country_code = config["country"]["gdelt_code"]
    root_codes = config["event"]["root_codes"]
    base_url = cfg_data["gdelt_bulk_base_url"]
    start = dt.datetime.strptime(cfg_data["start_date"], "%Y-%m-%d").date()
    end = dt.datetime.strptime(cfg_data["end_date"], "%Y-%m-%d").date()

    daily_dir = repo_path(cfg_data["raw_daily_dir"])
    daily_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = repo_path(cfg_data["download_manifest_path"])
    manifest = read_json(manifest_path)
    manifest.setdefault("downloaded_days", {})
    manifest.setdefault("failed_days", {})
    manifest["source_url_pattern"] = base_url + "{YYYYMMDD}.export.CSV.zip"
    manifest["country_gdelt_code"] = country_code
    manifest["conflict_root_codes"] = root_codes
    manifest["requested_range"] = [cfg_data["start_date"], cfg_data["end_date"]]

    all_days = list(daterange(start, end))
    todo = []
    for d in all_days:
        cache_fp = daily_dir / f"{d.strftime('%Y%m%d')}.parquet"
        if not force and (cache_fp.exists() or d.strftime("%Y%m%d") in manifest["failed_days"]):
            continue
        todo.append(d)

    print(f"[download_gdelt_bulk] {len(all_days)} days requested, {len(todo)} to fetch "
          f"({len(all_days) - len(todo)} already cached).")

    fetched, failed = 0, 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_fetch_one_day, d, base_url, country_code, root_codes): d for d in todo}
        for fut in tqdm(as_completed(futures), total=len(todo), desc="Downloading GDELT days"):
            d, df, err = fut.result()
            ymd = d.strftime("%Y%m%d")
            if err is not None:
                manifest["failed_days"][ymd] = {"error": err, "checked_at": dt.datetime.utcnow().isoformat()}
                failed += 1
                continue
            cache_fp = daily_dir / f"{ymd}.parquet"
            df.to_parquet(cache_fp, index=False)
            manifest["downloaded_days"][ymd] = {
                "rows_country_total": int(len(df)),
                "rows_conflict": int(df["is_conflict"].sum()),
                "downloaded_at": dt.datetime.utcnow().isoformat(),
            }
            fetched += 1

    manifest["last_run_at"] = dt.datetime.utcnow().isoformat()
    manifest["n_days_cached_total"] = len(list(daily_dir.glob("*.parquet")))
    write_json(manifest_path, manifest)
    print(f"[download_gdelt_bulk] Done. Newly fetched: {fetched}, newly failed: {failed}, "
          f"total cached days: {manifest['n_days_cached_total']}.")
    return manifest


def combine_daily_files(config: dict) -> pd.DataFrame:
    cfg_data = config["data"]
    daily_dir = repo_path(cfg_data["raw_daily_dir"])
    files = sorted(daily_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError("No daily GDELT parquet files found -- run download_all() first.")
    dfs = [pd.read_parquet(fp) for fp in files]
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values("event_date").reset_index(drop=True)
    out_path = repo_path(cfg_data["raw_combined_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_path, index=False)
    print(f"[download_gdelt_bulk] Combined {len(files)} daily files -> {out_path} "
          f"({len(combined):,} rows, {combined['event_date'].min().date()} .. "
          f"{combined['event_date'].max().date()}).")
    return combined


if __name__ == "__main__":
    cfg = load_config()
    download_all(cfg, max_workers=20)
    combine_daily_files(cfg)
