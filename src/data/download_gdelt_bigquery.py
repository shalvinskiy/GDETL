"""
GDELT Events ingestion -- BigQuery method (the assignment's PRIORITY/recommended
route: `gdelt-bq.gdeltv2.events`).

This script is fully implemented and ready to run, but is NOT the path used to
produce the committed results in this repo -- see README section "4. Данные и
источники" for why the Bulk CSV route (`download_gdelt_bulk.py`) was used
instead (no GCP credentials / billing project available in the dev/CI
environment; BigQuery access is credential- and environment-specific).

To use this script instead:
  1. `pip install google-cloud-bigquery`
  2. Authenticate: `gcloud auth application-default login`
     (or set GOOGLE_APPLICATION_CREDENTIALS to a service-account JSON key)
  3. Set a billing-enabled GCP project: `export GCP_PROJECT_ID=your-project`
     (querying `gdelt-bq.gdeltv2.events` is free up to BigQuery's 1TB/month
     on-demand query free tier; a single full-history query over this table
     for one country/event-code filter comfortably fits within that).
  4. `python -m src.data.download_gdelt_bigquery`

The output schema matches `build_weekly_panel.py`'s expectations exactly, so
either ingestion script can feed the rest of the pipeline unchanged.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils import load_config, repo_path  # noqa: E402

QUERY_TEMPLATE = """
SELECT
  DATE_TRUNC(
    DATE(PARSE_TIMESTAMP('%Y%m%d', CAST(SQLDATE AS STRING))),
    WEEK(MONDAY)
  ) AS week_start,
  COUNTIF(EventRootCode IN UNNEST(@root_codes)) AS conflict_count,
  COUNT(*) AS total_events_country,
  AVG(IF(EventRootCode IN UNNEST(@root_codes), GoldsteinScale, NULL)) AS avg_goldstein,
  AVG(IF(EventRootCode IN UNNEST(@root_codes), AvgTone, NULL)) AS avg_tone,
  STDDEV(IF(EventRootCode IN UNNEST(@root_codes), GoldsteinScale, NULL)) AS goldstein_volatility,
  SUM(IF(EventRootCode IN UNNEST(@root_codes), NumMentions, 0)) AS total_mentions_conflict
FROM `gdelt-bq.gdeltv2.events`
WHERE ActionGeo_CountryCode = @country_code
  AND SQLDATE BETWEEN @start_date AND @end_date
GROUP BY week_start
ORDER BY week_start
"""


def download_via_bigquery(config: dict) -> pd.DataFrame:
    from google.cloud import bigquery  # local import: optional dependency

    project_id = os.environ.get("GCP_PROJECT_ID")
    if not project_id:
        raise RuntimeError("Set GCP_PROJECT_ID env var to a billing-enabled GCP project first.")

    client = bigquery.Client(project=project_id)
    cfg_data, cfg_event, cfg_country = config["data"], config["event"], config["country"]
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("root_codes", "STRING", cfg_event["root_codes"]),
            bigquery.ScalarQueryParameter("country_code", "STRING", cfg_country["gdelt_code"]),
            bigquery.ScalarQueryParameter("start_date", "INT64", int(cfg_data["start_date"].replace("-", ""))),
            bigquery.ScalarQueryParameter("end_date", "INT64", int(cfg_data["end_date"].replace("-", ""))),
        ]
    )
    weekly = client.query(QUERY_TEMPLATE, job_config=job_config).to_dataframe()
    out_path = repo_path(cfg_data["weekly_panel_path"]).with_name("weekly_panel_bigquery.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    weekly.to_parquet(out_path, index=False)
    print(f"[download_gdelt_bigquery] {len(weekly)} weekly rows -> {out_path}")
    return weekly


if __name__ == "__main__":
    cfg = load_config()
    download_via_bigquery(cfg)
