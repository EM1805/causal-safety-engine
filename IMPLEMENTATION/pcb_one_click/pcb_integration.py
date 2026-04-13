# FILE: pcb_integration.py
# Python 3.7 compatible
"""Integration-facing helpers for machine-readable CSE decisions.

This module provides a stable contract that orchestration systems can consume
without parsing human-readable logs.
"""

import os
import json
import datetime

import pandas as pd

OUT_DIR = "out"
DEFAULT_INSIGHTS_PATH = os.path.join(OUT_DIR, "insights_level2.csv")
DEFAULT_ALERTS_PATH = os.path.join(OUT_DIR, "alerts_today_level28.csv")
DEFAULT_CONTRACT_PATH = os.path.join(OUT_DIR, "integration_decision.json")


def _utc_now_iso():
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _row_count(csv_path):
    if not csv_path or (not os.path.exists(csv_path)):
        return 0
    df = pd.read_csv(csv_path)
    return int(len(df))


def _insights_kept_count(insights_path):
    if not insights_path or (not os.path.exists(insights_path)):
        return 0
    df = pd.read_csv(insights_path)
    if len(df) == 0:
        return 0
    if "kept" in df.columns:
        kept = pd.to_numeric(df["kept"], errors="coerce").fillna(0)
        return int((kept > 0).sum())
    return int(len(df))


def build_decision_contract(insights_path=DEFAULT_INSIGHTS_PATH, alerts_path=DEFAULT_ALERTS_PATH):
    """Return a machine-readable decision contract.

    Decision mapping:
      - SILENCE: no kept insights are available
      - BLOCK: kept insights exist and at least one alert exists
      - ALLOW: kept insights exist and zero alerts exist
    """
    kept_insights = _insights_kept_count(insights_path)
    alerts = _row_count(alerts_path)

    if kept_insights <= 0:
        state = "SILENCE"
        reason = "no_kept_insights"
    elif alerts > 0:
        state = "BLOCK"
        reason = "alerts_present"
    else:
        state = "ALLOW"
        reason = "no_active_alerts"

    return {
        "schema_version": "v1",
        "generated_at_utc": _utc_now_iso(),
        "decision_state": state,
        "decision_reason": reason,
        "metrics": {
            "kept_insights_count": int(kept_insights),
            "alerts_count": int(alerts),
        },
        "artifacts": {
            "insights_level2_csv": insights_path,
            "alerts_level28_csv": alerts_path,
        },
    }


def write_decision_contract(contract, out_path=DEFAULT_CONTRACT_PATH):
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(contract, f, indent=2, ensure_ascii=False)
    return out_path
