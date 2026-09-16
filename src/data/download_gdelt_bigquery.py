"""Optional GDELT load via BigQuery (`gdelt-bq.gdeltv2.events`).

Not used for the committed results — bulk CSV needs no GCP account.
To run: set GCP_PROJECT_ID, `gcloud auth application-default login`,
then `python -m src.data.download_gdelt_bigquery`.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils import load_config, repo_path

QUERY = """
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


def download_via_bigquery(config: dict):
    from google.cloud import bigquery

    project_id = os.environ.get("GCP_PROJECT_ID")
    if not project_id:
        raise RuntimeError("Set GCP_PROJECT_ID to a billing-enabled GCP project.")

    client = bigquery.Client(project=project_id)
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("root_codes", "STRING", config["event"]["root_codes"]),
            bigquery.ScalarQueryParameter("country_code", "STRING", config["country"]["gdelt_code"]),
            bigquery.ScalarQueryParameter("start_date", "INT64", int(config["data"]["start_date"].replace("-", ""))),
            bigquery.ScalarQueryParameter("end_date", "INT64", int(config["data"]["end_date"].replace("-", ""))),
        ]
    )
    weekly = client.query(QUERY, job_config=job_config).to_dataframe()
    out_path = repo_path(config["data"]["weekly_panel_path"]).with_name("weekly_panel_bigquery.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    weekly.to_parquet(out_path, index=False)
    print(f"[bigquery] {len(weekly)} weeks -> {out_path}")
    return weekly


if __name__ == "__main__":
    download_via_bigquery(load_config())
