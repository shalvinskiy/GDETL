"""Shared helpers: config loading, path resolution, ISO-week utilities."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path = None) -> dict[str, Any]:
    path = Path(path) if path else REPO_ROOT / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def repo_path(*parts: str) -> Path:
    """Resolve a path relative to the repo root, creating parent dirs as needed."""
    p = REPO_ROOT.joinpath(*parts)
    return p


def week_start_monday(date: dt.date | dt.datetime | str) -> dt.date:
    """Return the Monday that starts the ISO week containing `date`."""
    if isinstance(date, str):
        date = dt.datetime.strptime(date, "%Y%m%d" if len(date) == 8 else "%Y-%m-%d").date()
    elif isinstance(date, dt.datetime):
        date = date.date()
    return date - dt.timedelta(days=date.weekday())


def iso_week_label(monday: dt.date) -> str:
    iso_year, iso_week, _ = monday.isocalendar()
    return f"{iso_year}-W{iso_week:02d}"


def daterange(start: dt.date, end: dt.date):
    cur = start
    while cur <= end:
        yield cur
        cur += dt.timedelta(days=1)


def read_json(path: Path) -> dict:
    if not Path(path).exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str, ensure_ascii=False)
