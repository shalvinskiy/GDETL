"""Download GDELT 1.0 daily export CSVs and keep rows for the configured country.

Bulk CSV is used instead of BigQuery because it needs no GCP credentials.
See download_gdelt_bigquery.py if you have a billing-enabled project.
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
from src.utils import daterange, load_config, read_json, repo_path, write_json

# GDELT 1.0 export: 58 columns, no header.
# Codebook: http://data.gdeltproject.org/documentation/GDELT-Data_Format_Codebook.pdf
COLS = {
    0: "GlobalEventID",
    1: "SQLDATE",
    28: "EventRootCode",
    30: "GoldsteinScale",
    31: "NumMentions",
    34: "AvgTone",
    51: "ActionGeo_CountryCode",
}
COL_IDX = sorted(COLS)
COL_NAMES = [COLS[i] for i in COL_IDX]

MAX_RETRIES = 4
RETRY_BACKOFF_S = 3
REQUEST_TIMEOUT_S = 40


def _fetch_one_day(date: dt.date, base_url: str, country_code: str, root_codes: list[str]):
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
            with zf.open(zf.namelist()[0]) as fh:
                df = pd.read_csv(fh, sep="\t", header=None, usecols=COL_IDX, dtype=str, low_memory=False)
            df.columns = COL_NAMES
            df = df[df["ActionGeo_CountryCode"] == country_code].copy()
            df["is_conflict"] = df["EventRootCode"].isin(root_codes)
            for c in ["GoldsteinScale", "NumMentions", "AvgTone"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df["event_date"] = pd.to_datetime(df["SQLDATE"], format="%Y%m%d")
            return date, df, None
        except Exception as exc:
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

    print(f"[download] {len(all_days)} days requested, {len(todo)} to fetch")

    fetched, failed = 0, 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_fetch_one_day, d, base_url, country_code, root_codes): d for d in todo}
        for fut in tqdm(as_completed(futures), total=len(todo), desc="GDELT"):
            d, df, err = fut.result()
            ymd = d.strftime("%Y%m%d")
            if err is not None:
                manifest["failed_days"][ymd] = {"error": err, "checked_at": dt.datetime.now(dt.timezone.utc).isoformat()}
                failed += 1
                continue
            df.to_parquet(daily_dir / f"{ymd}.parquet", index=False)
            manifest["downloaded_days"][ymd] = {
                "rows_country_total": int(len(df)),
                "rows_conflict": int(df["is_conflict"].sum()),
                "downloaded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            }
            fetched += 1

    manifest["last_run_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    manifest["n_days_cached_total"] = len(list(daily_dir.glob("*.parquet")))
    write_json(manifest_path, manifest)
    print(f"[download] fetched={fetched}, failed={failed}, cached={manifest['n_days_cached_total']}")
    return manifest


def combine_daily_files(config: dict) -> pd.DataFrame:
    cfg_data = config["data"]
    files = sorted(repo_path(cfg_data["raw_daily_dir"]).glob("*.parquet"))
    if not files:
        raise FileNotFoundError("No daily GDELT parquet files — run download_all() first.")
    combined = pd.concat([pd.read_parquet(fp) for fp in files], ignore_index=True)
    combined = combined.sort_values("event_date").reset_index(drop=True)
    out_path = repo_path(cfg_data["raw_combined_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_path, index=False)
    print(f"[download] combined {len(files)} days -> {out_path} ({len(combined):,} rows)")
    return combined


if __name__ == "__main__":
    cfg = load_config()
    download_all(cfg)
    combine_daily_files(cfg)
