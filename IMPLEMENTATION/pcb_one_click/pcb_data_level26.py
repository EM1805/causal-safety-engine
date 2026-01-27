# FILE: pcb_data_level26.py
# Python 3.7 compatible
#
# PCB – Level 2.6: Data Quality + Schema Normalization (local-first)
#
# Goal:
# - Read data.csv
# - Normalize schema: ensure date + numeric columns coercion
# - Add optional derived features (safe, lightweight):
#     - weekday, is_weekend, is_workday
#     - mood_prev
#     - mood_ewma (optional)
#     - rolling mean/std (optional)
# - Produce:
#     - out/data_clean.csv        (canonical input for Level 2.5 + 3.x)
#     - out/data_profile.csv      (column profile)
#     - out/data_report.json      (warnings + summary)
#     - out/*.jsonl mirrors
#
# Dependencies: numpy, pandas only

import os
import json
import numpy as np
import pandas as pd

OUT_DIR = "out"

# -----------------------------
# Central config override (optional)
# -----------------------------
try:
    from pcb_config import load_config  # local file
    _CFG = load_config()
except Exception:
    _CFG = {}

# Allow overriding common columns/paths via pcb.json (keeps backward compatibility)
OUT_DIR = str(_CFG.get("out_dir", OUT_DIR))
DEFAULT_DATA_CSV = "data.csv"
FALLBACK_DATA_CSV = os.path.join(OUT_DIR, "demo_data.csv")

OUT_DATA_CLEAN = os.path.join(OUT_DIR, "data_clean.csv")
OUT_PROFILE_CSV = os.path.join(OUT_DIR, "data_profile.csv")
OUT_REPORT_JSON = os.path.join(OUT_DIR, "data_report.json")
OUT_PROFILE_JSONL = os.path.join(OUT_DIR, "data_profile.jsonl")

DATE_COL = "date"
TARGET_COL = "target"

# -----------------------------
# Knobs (safe defaults)
# -----------------------------
MIN_ROWS = 30
MIN_DATE_PARSE_FRAC = 0.20

MIN_NUMERIC_FRAC = 0.30
KEEP_ALWAYS = set([DATE_COL, TARGET_COL])

WARN_NAN_FRAC = 0.40
WARN_LOW_UNIQUE = 6
WARN_LOW_STD = 1e-6

ADD_WEEKDAY_FEATURES = True
ADD_MOOD_PREV = True

ADD_EWMA = False
EWMA_SPAN = 21
EWMA_MIN_PERIODS = 12

ADD_ROLLING = False
ROLL_WINDOW = 7
ROLL_MIN_PERIODS = 5

SORT_COLUMNS = True

# -----------------------------
# Utils
# -----------------------------
def _ensure_out():
    os.makedirs(OUT_DIR, exist_ok=True)

def _safe_float(x, default=np.nan):
    try:
        v = float(x)
        if np.isfinite(v):
            return v
        return float(default)
    except Exception:
        return float(default)

def _load_data_path(path=None):
    if path and os.path.exists(path):
        return path
    if os.path.exists(DEFAULT_DATA_CSV):
        return DEFAULT_DATA_CSV
    return FALLBACK_DATA_CSV

def _try_parse_date(df):
    if DATE_COL not in df.columns:
        return df, False
    dt = pd.to_datetime(df[DATE_COL], errors="coerce")
    frac = float(dt.notna().mean()) if len(df) > 0 else 0.0
    if frac >= float(MIN_DATE_PARSE_FRAC):
        out = df.copy()
        out[DATE_COL] = dt
        return out, True
    return df, False

def _profile_column(name, s, is_date_parsed=False):
    if is_date_parsed and name == DATE_COL and pd.api.types.is_datetime64_any_dtype(s):
        nn = int(s.notna().sum())
        nan_frac = float(1.0 - nn / float(len(s))) if len(s) else 1.0
        uniq = int(s.dropna().dt.normalize().nunique()) if nn > 0 else 0
        return {
            "col": name,
            "dtype": "datetime",
            "n": int(len(s)),
            "n_non_null": nn,
            "nan_frac": float(nan_frac),
            "numeric_frac": np.nan,
            "n_unique": uniq,
            "mean": np.nan,
            "std": np.nan,
            "min": str(s.min()) if nn > 0 else "",
            "max": str(s.max()) if nn > 0 else "",
            "notes": "",
        }

    num = pd.to_numeric(s, errors="coerce").astype(float)
    nn = int(num.notna().sum())
    nan_frac = float(1.0 - nn / float(len(num))) if len(num) else 1.0
    uniq = int(num.dropna().nunique()) if nn > 0 else 0
    std = float(np.nanstd(num.to_numpy(dtype=float))) if nn > 0 else np.nan
    mean = float(np.nanmean(num.to_numpy(dtype=float))) if nn > 0 else np.nan
    mn = float(np.nanmin(num.to_numpy(dtype=float))) if nn > 0 else np.nan
    mx = float(np.nanmax(num.to_numpy(dtype=float))) if nn > 0 else np.nan

    numeric_frac = float(nn / float(len(num))) if len(num) else 0.0
    dtype_guess = "numeric" if numeric_frac >= 0.01 else "non_numeric"

    notes = []
    if nan_frac >= float(WARN_NAN_FRAC):
        notes.append("high_missingness")
    if uniq > 0 and uniq < int(WARN_LOW_UNIQUE):
        notes.append("low_unique")
    if np.isfinite(std) and std < float(WARN_LOW_STD):
        notes.append("near_constant")

    return {
        "col": name,
        "dtype": dtype_guess,
        "n": int(len(num)),
        "n_non_null": nn,
        "nan_frac": float(nan_frac),
        "numeric_frac": float(numeric_frac),
        "n_unique": uniq,
        "mean": mean if np.isfinite(mean) else np.nan,
        "std": std if np.isfinite(std) else np.nan,
        "min": mn if np.isfinite(mn) else np.nan,
        "max": mx if np.isfinite(mx) else np.nan,
        "notes": "|".join(notes),
    }

def _save_csv(df, path):
    _ensure_out()
    df.to_csv(path, index=False)

def _save_json(path, obj):
    _ensure_out()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _save_jsonl(df, path):
    _ensure_out()
    with open(path, "w", encoding="utf-8") as f:
        for _, r in df.iterrows():
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

def _make_canonical(df, date_parsed):
    out = df.copy()

    if TARGET_COL not in out.columns:
        raise ValueError("Target column '%s' not found in data." % TARGET_COL)

    out[TARGET_COL] = pd.to_numeric(out[TARGET_COL], errors="coerce").astype(float)

    if ADD_WEEKDAY_FEATURES and date_parsed and pd.api.types.is_datetime64_any_dtype(out[DATE_COL]):
        out["weekday"] = out[DATE_COL].dt.weekday.astype(float)
        out["is_weekend"] = (out["weekday"] >= 5).astype(int)
        out["is_workday"] = (out["weekday"] <= 4).astype(int)

    if ADD_MOOD_PREV:
        # Backward compatible: keep historical name and also write neutral alias.
        out["mood_prev"] = out[TARGET_COL].shift(1)
        out["target_prev"] = out["mood_prev"]

    if ADD_EWMA:
        s = out[TARGET_COL].copy()
        out["mood_ewma"] = s.ewm(
            span=int(EWMA_SPAN),
            adjust=False,
            min_periods=int(EWMA_MIN_PERIODS),
        ).mean()

    if ADD_ROLLING:
        s = out[TARGET_COL].copy()
        out["mood_roll_mean"] = s.rolling(int(ROLL_WINDOW), min_periods=int(ROLL_MIN_PERIODS)).mean()
        out["mood_roll_std"] = s.rolling(int(ROLL_WINDOW), min_periods=int(ROLL_MIN_PERIODS)).std()

    cols = list(out.columns)
    keep_cols = []
    dropped_cols = []

    for c in cols:
        if c in KEEP_ALWAYS:
            keep_cols.append(c)
            continue
        if c == DATE_COL:
            keep_cols.append(c)
            continue

        num = pd.to_numeric(out[c], errors="coerce")
        numeric_frac = float(num.notna().mean()) if len(out) else 0.0

        if numeric_frac >= float(MIN_NUMERIC_FRAC):
            out[c] = num.astype(float)
            keep_cols.append(c)
        else:
            dropped_cols.append(c)

    out = out[keep_cols].copy()

    if SORT_COLUMNS:
        front = []
        if DATE_COL in out.columns:
            front.append(DATE_COL)
        if TARGET_COL in out.columns:
            front.append(TARGET_COL)
        rest = [c for c in out.columns if c not in front]
        out = out[front + sorted(rest)].copy()

    return out, dropped_cols

# -----------------------------
# Main
# -----------------------------
def main(data_csv_path=None):
    _ensure_out()
    data_csv_path = _load_data_path(data_csv_path)
    if not os.path.exists(data_csv_path):
        raise FileNotFoundError("Missing data.csv (or out/demo_data.csv).")

    df = pd.read_csv(data_csv_path)

    df, date_parsed = _try_parse_date(df)

    prof_rows = []
    for c in df.columns:
        prof_rows.append(_profile_column(c, df[c], is_date_parsed=date_parsed))
    df_profile = pd.DataFrame(prof_rows)

    df_clean, dropped_cols = _make_canonical(df, date_parsed=date_parsed)

    warnings = []
    tgt_non_null = int(pd.to_numeric(df_clean[TARGET_COL], errors="coerce").notna().sum())
    if tgt_non_null < int(max(15, 0.5 * len(df_clean))):
        warnings.append("target_mood_has_many_missing_values")

    if date_parsed:
        if DATE_COL in df_clean.columns and pd.api.types.is_datetime64_any_dtype(df_clean[DATE_COL]):
            if df_clean[DATE_COL].notna().sum() < int(max(10, 0.2 * len(df_clean))):
                warnings.append("date_parsing_low_coverage")
    else:
        if DATE_COL in df.columns:
            warnings.append("date_not_parsed_using_row_index_weekday_proxy_in_other_levels")

    for _, r in df_profile.iterrows():
        notes = str(r.get("notes", ""))
        if "high_missingness" in notes:
            warnings.append("high_missingness:" + str(r.get("col", "")))
        if "near_constant" in notes:
            warnings.append("near_constant:" + str(r.get("col", "")))

    warnings = sorted(list(dict.fromkeys(warnings)))

    report = {
        "level": "2.6",
        "input_path": data_csv_path,
        "date_parsed": bool(date_parsed),
        "rows": int(len(df)),
        "rows_clean": int(len(df_clean)),
        "cols_in": int(df.shape[1]),
        "cols_clean": int(df_clean.shape[1]),
        "dropped_cols": dropped_cols,
        "warnings": warnings,
        "derived_features": {
            "weekday_features": bool(ADD_WEEKDAY_FEATURES and date_parsed),
            "mood_prev": bool(ADD_MOOD_PREV),
            "target_prev": bool(ADD_MOOD_PREV),
            "ewma": bool(ADD_EWMA),
            "rolling": bool(ADD_ROLLING),
        },
        "recommended_next_step": "Run Level 2.5 on out/data_clean.csv (or copy it to data.csv).",
    }

    _save_csv(df_clean, OUT_DATA_CLEAN)
    _save_csv(df_profile, OUT_PROFILE_CSV)
    _save_json(OUT_REPORT_JSON, report)
    _save_jsonl(df_profile, OUT_PROFILE_JSONL)

    print("\n=== PCB LEVEL 2.6 (data quality + canonical clean) ===")
    print("Input:", data_csv_path)
    print("Date parsed:", bool(date_parsed))
    print("Saved:", OUT_DATA_CLEAN)
    print("Saved:", OUT_PROFILE_CSV)
    print("Saved:", OUT_REPORT_JSON)
    print("Dropped cols:", len(dropped_cols))
    if warnings:
        print("Warnings (top 10):")
        for w in warnings[:10]:
            print(" -", w)
    else:
        print("Warnings: none")

    return df_clean, df_profile, report

if __name__ == "__main__":
    main()
