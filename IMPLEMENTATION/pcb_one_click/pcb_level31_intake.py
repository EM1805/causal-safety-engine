# FILE: pcb_level31_intake.py
# Python 3.7 compatible
#
# PCB — Level 3.1: Trial Intake + Normalization + Quality Gates (non-breaking, additive)
#
# Purpose:
# - Normalize logged trials into a stable canonical schema for Level 3.2
# - Apply conservative quality gates (completeness, basic sanity checks)
# - Produce additive artifacts only (does not modify existing outputs)
#
# Inputs:
# - out/experiment_results.csv  (logged trials)
#
# Outputs (additive):
# - out/experiment_intake_level31.csv
# - out/metrics_level31.json
# - out/warnings_level31.csv
#
# Notes:
# - This level does NOT add causal claims; it only prepares trials for validation.
# - Level 3.2 should prefer experiment_intake_level31.csv when present, falling back to experiment_results.csv.

import os
import json
import time
import argparse
import pandas as pd

OUT_DIR = "out"
EXP_RESULTS = os.path.join(OUT_DIR, "experiment_results.csv")
OUT_INTAKE = os.path.join(OUT_DIR, "experiment_intake_level31.csv")
OUT_WARN = os.path.join(OUT_DIR, "warnings_level31.csv")
OUT_METRICS = os.path.join(OUT_DIR, "metrics_level31.json")

SCHEMA_VERSION = "3.1-intake.1"

REQUIRED_COLS = ["trial_id", "date", "action_name"]  # minimal contract
OPTIONAL_COLS = ["action_type", "adherence", "window_days", "target_col", "notes"]

def _ensure_out():
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

def _safe_float(x, default=None):
    try:
        if x is None:
            return default
        v = float(x)
        if v == v:
            return v
        return default
    except Exception:
        return default

def _write_metrics(obj):
    with open(OUT_METRICS, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)

def main():
    _ensure_out()
    t0 = time.time()

    if not os.path.exists(EXP_RESULTS):
        raise FileNotFoundError("Missing %s (log trials first)." % EXP_RESULTS)

    df = pd.read_csv(EXP_RESULTS)
    n_in = int(df.shape[0])

    warnings = []
    keep_rows = []

    for idx, row in df.iterrows():
        reasons = []
        # required fields
        for c in REQUIRED_COLS:
            if c not in df.columns or pd.isna(row.get(c)):
                reasons.append("missing_%s" % c)

        # basic optional normalization
        adherence = row.get("adherence", None)
        adherence_v = _safe_float(adherence, default=None)
        if adherence_v is not None:
            # allow 0..1; if >1 assume percentage and scale
            if adherence_v > 1.0 and adherence_v <= 100.0:
                adherence_v = adherence_v / 100.0
            if adherence_v < 0.0 or adherence_v > 1.0:
                reasons.append("invalid_adherence")

        window_days = row.get("window_days", None)
        window_v = _safe_float(window_days, default=None)
        if window_v is not None and window_v <= 0:
            reasons.append("invalid_window_days")

        if reasons:
            warnings.append({
                "row_index": int(idx),
                "trial_id": str(row.get("trial_id", "")),
                "reasons": ";".join(reasons),
            })
            continue

        out_row = {
            "trial_id": str(row.get("trial_id")),
            "date": str(row.get("date")),
            "action_name": str(row.get("action_name")),
            "action_type": str(row.get("action_type", "")) if "action_type" in df.columns else "",
            "adherence": adherence_v if adherence_v is not None else "",
            "window_days": int(window_v) if window_v is not None else "",
            "target_col": str(row.get("target_col", "")) if "target_col" in df.columns else "",
            "notes": str(row.get("notes", "")) if "notes" in df.columns else "",
        }
        keep_rows.append(out_row)

    df_out = pd.DataFrame(keep_rows, columns=["trial_id","date","action_name","action_type","adherence","window_days","target_col","notes"])
    df_out.to_csv(OUT_INTAKE, index=False)

    df_warn = pd.DataFrame(warnings, columns=["row_index","trial_id","reasons"])
    df_warn.to_csv(OUT_WARN, index=False)

    metrics = {
        "schema_version": SCHEMA_VERSION,
        "level": "3.1",
        "generated_at": int(time.time()),
        "n_trials_in": n_in,
        "n_trials_kept": int(df_out.shape[0]),
        "n_trials_dropped": int(df_warn.shape[0]),
        "drop_reasons_counts": dict(df_warn["reasons"].value_counts()) if df_warn.shape[0] else {},
        "source_file": EXP_RESULTS,
        "out_intake_file": OUT_INTAKE,
        "elapsed_sec": round(time.time() - t0, 6),
    }
    _write_metrics(metrics)
    print("[pcb] Level 3.1 complete. Kept %d/%d trials." % (df_out.shape[0], n_in))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
