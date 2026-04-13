from __future__ import annotations

from typing import Any

import pandas as pd


def transform_events_to_timeseries(events: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(events)
    if frame.empty:
        return frame

    if "timestamp" not in frame.columns:
        raise ValueError("Expected timestamp field in event rows")

    frame["date"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce").dt.date
    frame = frame.dropna(subset=["date"])

    aggregations: dict[str, str] = {}
    for metric in ["conversion", "price", "traffic"]:
        if metric in frame.columns:
            aggregations[metric] = "mean" if metric != "traffic" else "sum"

    if not aggregations:
        raise ValueError("No known metrics found. Provide conversion/price/traffic columns")

    grouped = frame.groupby("date", as_index=False).agg(aggregations)
    return grouped
