# FILE: pcb_experiments_level29.py
# Python 3.7 compatible
#
# PCB – Level 2.9: Experiment Plan + Results Logging (local-first)
#
# Goal:
# - plan: build out/experiment_plan.csv from out/insights_level2.csv
# - log : append a daily "trial" row into out/experiment_results.csv
# - eval: compute quick per-insight/action summary using a simple past-only baseline
#
# Inputs:
# - out/insights_level2.csv (required for plan)
# - data.csv (or out/demo_data.csv) (required for eval baseline)
#
# Outputs:
# - out/experiment_plan.csv
# - out/experiment_results.csv
# - out/experiment_summary_level29.csv
# - out/*.jsonl mirrors
#
# Dependencies: numpy, pandas only

import os
import sys
import json
import argparse
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
    _CFG = {{}}

# Allow overriding common columns/paths via pcb.json (keeps backward compatibility)
OUT_DIR = str(_CFG.get("out_dir", OUT_DIR))

INSIGHTS_L2 = os.path.join(OUT_DIR, "insights_level2.csv")
EXP_PLAN = os.path.join(OUT_DIR, "experiment_plan.csv")
EXP_RESULTS = os.path.join(OUT_DIR, "experiment_results.csv")
EXP_SUMMARY = os.path.join(OUT_DIR, "experiment_summary_level29.csv")
EXP_SUMMARY_JSONL = os.path.join(OUT_DIR, "experiment_summary_level29.jsonl")

DEFAULT_DATA_CSV = "data.csv"
FALLBACK_DATA_CSV = os.path.join(OUT_DIR, "demo_data.csv")

DATE_COL = "date"
TARGET_COL = "target"

# plan defaults
DEFAULT_WINDOW_DAYS = 1
DEFAULT_COST = 1.0
DEFAULT_DOSE = ""
DEFAULT_NOTES = ""

# eval defaults
LOOKBACK_DAYS = 60
LOOKBACK_ROWS = 60
HARD_WEEKDAY_MATCH = True
MIN_BASELINE_N = 10
Z_CLIP = 6.0
Z_SUCCESS_THRESH = 0.2

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

def _as_str(x):
    try:
        return str(x)
    except Exception:
        return ""

def _save_csv(df, path):
    _ensure_out()
    df.to_csv(path, index=False)

def _save_jsonl(df, path):
    _ensure_out()
    with open(path, "w", encoding="utf-8") as f:
        for _, r in df.iterrows():
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

def _load_data_path(path=None):
    if path and os.path.exists(path):
        return path
    if os.path.exists(DEFAULT_DATA_CSV):
        return DEFAULT_DATA_CSV
    return FALLBACK_DATA_CSV

def _try_parse_date(df):
    if DATE_COL in df.columns:
        dt = pd.to_datetime(df[DATE_COL], errors="coerce")
        if dt.notna().sum() >= max(5, int(0.2 * len(df))):
            out = df.copy()
            out[DATE_COL] = dt
            return out
    return df

def _has_date(df):
    return (DATE_COL in df.columns) and pd.api.types.is_datetime64_any_dtype(df[DATE_COL])

def _weekday_series(df):
    if _has_date(df):
        return df[DATE_COL].dt.weekday
    return pd.Series(np.arange(len(df)) % 7, index=df.index)

def _past_window_indices(df, t_idx, lookback_days=LOOKBACK_DAYS, lookback_rows=LOOKBACK_ROWS):
    n = len(df)
    if t_idx <= 0 or t_idx >= n:
        return np.array([], dtype=int)

    if _has_date(df):
        d_t = df[DATE_COL].iloc[t_idx]
        if pd.notna(d_t):
            d0 = d_t.normalize() - pd.Timedelta(days=int(lookback_days))
            mask = (df[DATE_COL] < d_t) & (df[DATE_COL] >= d0)
            return df.index[mask].to_numpy(dtype=int)

    start = max(0, int(t_idx) - int(lookback_rows))
    return np.arange(start, int(t_idx), dtype=int)

def _z_against_baseline(df, t_idx, baseline_col, baseline_exclude_mask=None):
    if baseline_col not in df.columns:
        return np.nan, {"reason": "missing_col"}

    x = pd.to_numeric(df[baseline_col], errors="coerce").to_numpy(dtype=float)
    if t_idx < 0 or t_idx >= len(x) or not np.isfinite(x[t_idx]):
        return np.nan, {"reason": "missing_value_today"}

    idx = _past_window_indices(df, t_idx)
    if idx.size == 0:
        return np.nan, {"reason": "no_past_window"}

    wd = _weekday_series(df).to_numpy(dtype=int)
    cand = idx
    if HARD_WEEKDAY_MATCH:
        cand = cand[wd[cand] == int(wd[t_idx])]

    if baseline_exclude_mask is not None:
        ex = np.asarray(baseline_exclude_mask, dtype=bool)
        cand = cand[~ex[cand]]

    vals = x[cand]
    vals = vals[np.isfinite(vals)]
    if len(vals) < int(MIN_BASELINE_N):
        return np.nan, {"reason": "too_few_baseline", "baseline_n": int(len(vals))}

    mu = float(np.mean(vals))
    sd = float(np.std(vals))
    if not np.isfinite(sd) or sd < 1e-9:
        return np.nan, {"reason": "sd_zero", "baseline_n": int(len(vals)), "mu": mu}

    z = float((x[t_idx] - mu) / sd)
    z = float(np.clip(z, -Z_CLIP, Z_CLIP))
    meta = {"baseline_n": int(len(vals)), "mu": mu, "sd": sd, "method": "same_weekday" if HARD_WEEKDAY_MATCH else "window"}
    return z, meta

def _build_date_to_index_map(df_data):
    m = {"_len": int(len(df_data))}
    if _has_date(df_data):
        for i, d in enumerate(df_data[DATE_COL]):
            if pd.notna(d):
                key = d.normalize().date().isoformat()
                if key not in m:
                    m[key] = int(i)
    return m

def _normalize_date_key(x):
    try:
        d = pd.to_datetime(x, errors="coerce")
        if pd.isna(d):
            return ""
        return d.normalize().date().isoformat()
    except Exception:
        return ""

def _resolve_t_index(date_str, df_data):
    if not (DATE_COL in df_data.columns and _has_date(df_data)):
        return -1
    key = _normalize_date_key(date_str)
    if not key:
        return -1
    idx_map = _build_date_to_index_map(df_data)
    return int(idx_map.get(key, -1))

# -----------------------------
# Subcommand: plan
# -----------------------------
def cmd_plan(args):
    _ensure_out()
    if not os.path.exists(INSIGHTS_L2):
        raise FileNotFoundError("Missing %s (run Level 2.5 first)." % INSIGHTS_L2)

    df = pd.read_csv(INSIGHTS_L2)
    if len(df) == 0:
        out = pd.DataFrame(columns=[
            "insight_id", "action_name", "action_type",
            "dose", "window_days", "cost",
            # Backward-compatible name + neutral alias
            "expected_direction_on_mood",
            "expected_direction_on_target",
            "notes",
        ])
        _save_csv(out, EXP_PLAN)
        print("No insights. Saved empty:", EXP_PLAN)
        return out

    for c in ["insight_id", "source", "target", "lag", "delta_test", "strength", "recommendation", "human_statement"]:
        if c not in df.columns:
            df[c] = np.nan

    df = df[df["target"].astype(str) == TARGET_COL].copy()

    rows = []
    for _, r in df.iterrows():
        iid = _as_str(r.get("insight_id", ""))
        src = _as_str(r.get("source", "")).strip()
        lag = int(_safe_float(r.get("lag", 1), 1))
        delta = _safe_float(r.get("delta_test", np.nan), np.nan)

        if not iid or not src:
            continue

        exp_dir = np.nan
        if np.isfinite(delta) and delta != 0:
            exp_dir = float(np.sign(delta))

        if np.isfinite(delta) and delta > 0:
            action_type = "increase"
            action_name = "increase_%s" % src
        elif np.isfinite(delta) and delta < 0:
            action_type = "reduce"
            action_name = "reduce_%s" % src
        else:
            action_type = "adjust"
            action_name = "adjust_%s" % src

        rows.append({
            "insight_id": iid,
            "action_name": action_name,
            "action_type": action_type,
            "dose": DEFAULT_DOSE,
            "window_days": int(DEFAULT_WINDOW_DAYS),
            "cost": float(DEFAULT_COST),
            "expected_direction_on_mood": exp_dir,
            "expected_direction_on_target": exp_dir,
            "notes": "lag=%d | source=%s" % (lag, src),
        })

    out = pd.DataFrame(rows)
    if len(out) > 0:
        out = out.drop_duplicates(subset=["insight_id"], keep="first").reset_index(drop=True)

    _save_csv(out, EXP_PLAN)
    print("\n=== PCB LEVEL 2.9 (plan) ===")
    print("Saved:", EXP_PLAN)
    print("Rows:", len(out))
    return out

# -----------------------------
# Subcommand: log
# -----------------------------
def cmd_log(args):
    _ensure_out()

    insight_id = _as_str(args.insight_id).strip()
    if not insight_id:
        raise ValueError("--insight_id is required")

    action_name = _as_str(args.action_name).strip()
    dose = _as_str(args.dose).strip()
    notes = _as_str(args.notes).strip()

    adherence_flag = int(args.adherence_flag)
    if adherence_flag not in (0, 1):
        adherence_flag = 1

    data_path = _load_data_path(args.data)
    df_data = pd.read_csv(data_path)
    df_data = _try_parse_date(df_data)

    date_str = _as_str(args.date).strip() if args.date else ""
    t_index = int(args.t_index) if args.t_index is not None else -1
    if t_index < 0 and date_str:
        t_index = _resolve_t_index(date_str, df_data)

    if t_index < 0:
        t_index = int(len(df_data) - 1)

    if not date_str and _has_date(df_data) and 0 <= t_index < len(df_data):
        d = df_data[DATE_COL].iloc[t_index]
        if pd.notna(d):
            date_str = d.normalize().date().isoformat()

    if (not action_name) and os.path.exists(EXP_PLAN):
        try:
            df_plan = pd.read_csv(EXP_PLAN)
            if "insight_id" in df_plan.columns and "action_name" in df_plan.columns:
                m = dict(zip(df_plan["insight_id"].astype(str), df_plan["action_name"].astype(str)))
                action_name = _as_str(m.get(insight_id, "")).strip()
        except Exception:
            pass

    row = {
        "insight_id": insight_id,
        "action_name": action_name,
        "date": date_str,
        "t_index": int(t_index),
        "adherence_flag": int(adherence_flag),
        "dose": dose,
        "notes": notes,
    }

    if os.path.exists(EXP_RESULTS):
        df = pd.read_csv(EXP_RESULTS)
    else:
        df = pd.DataFrame(columns=list(row.keys()))

    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    _save_csv(df, EXP_RESULTS)

    print("\n=== PCB LEVEL 2.9 (log) ===")
    print("Saved:", EXP_RESULTS)
    print("Appended trial:", row)
    return df

# -----------------------------
# Subcommand: eval
# -----------------------------
def cmd_eval(args):
    _ensure_out()

    if not os.path.exists(EXP_RESULTS):
        out = pd.DataFrame(columns=[
            "insight_id", "action_name",
            "n_trials", "n_wins", "success_rate",
            "avg_z", "median_z",
            "baseline_method",
        ])
        _save_csv(out, EXP_SUMMARY)
        _save_jsonl(out, EXP_SUMMARY_JSONL)
        print("Missing experiment_results.csv. Saved empty summary:", EXP_SUMMARY)
        return out

    df_res = pd.read_csv(EXP_RESULTS)
    if len(df_res) == 0 or "insight_id" not in df_res.columns:
        out = pd.DataFrame(columns=[
            "insight_id", "action_name",
            "n_trials", "n_wins", "success_rate",
            "avg_z", "median_z",
            "baseline_method",
        ])
        _save_csv(out, EXP_SUMMARY)
        _save_jsonl(out, EXP_SUMMARY_JSONL)
        print("No trials. Saved empty summary:", EXP_SUMMARY)
        return out

    for c in ["insight_id", "action_name", "date", "t_index", "adherence_flag", "dose", "notes"]:
        if c not in df_res.columns:
            df_res[c] = np.nan

    df_res["insight_id"] = df_res["insight_id"].astype(str)
    df_res["action_name"] = df_res["action_name"].astype(str).replace({"nan": ""})
    df_res["t_index"] = pd.to_numeric(df_res["t_index"], errors="coerce")

    data_path = _load_data_path(args.data)
    df_data = pd.read_csv(data_path)
    df_data = _try_parse_date(df_data)
    if TARGET_COL not in df_data.columns:
        raise ValueError("Target column '%s' not found in data CSV." % TARGET_COL)

    zs = []
    succ = []
    bns = []
    bmu = []
    bsd = []
    bmethod = []
    reason = []

    for _, r in df_res.iterrows():
        t = _safe_float(r.get("t_index", np.nan), np.nan)
        if not np.isfinite(t):
            zs.append(np.nan); succ.append(np.nan)
            bns.append(0); bmu.append(np.nan); bsd.append(np.nan); bmethod.append(""); reason.append("missing_t_index")
            continue

        t_idx = int(t)
        z, meta = _z_against_baseline(df_data, t_idx, TARGET_COL, baseline_exclude_mask=None)
        if np.isfinite(z):
            zs.append(float(z))
            succ.append(int(float(z) >= float(Z_SUCCESS_THRESH)))
            bns.append(int(meta.get("baseline_n", 0)))
            bmu.append(float(meta.get("mu", np.nan)))
            bsd.append(float(meta.get("sd", np.nan)))
            bmethod.append(_as_str(meta.get("method", "")))
            reason.append("")
        else:
            zs.append(np.nan); succ.append(np.nan)
            bns.append(int(meta.get("baseline_n", 0)) if meta else 0)
            bmu.append(float(meta.get("mu", np.nan)) if meta else np.nan)
            bsd.append(float(meta.get("sd", np.nan)) if meta else np.nan)
            bmethod.append(_as_str(meta.get("method", "")) if meta else "")
            reason.append(_as_str(meta.get("reason", "baseline_failed")) if meta else "baseline_failed")

    df_res2 = df_res.copy()
    df_res2["z_vs_baseline"] = zs
    df_res2["success_flag"] = succ
    df_res2["baseline_n"] = bns
    df_res2["baseline_mu"] = bmu
    df_res2["baseline_sd"] = bsd
    df_res2["baseline_method"] = bmethod
    df_res2["debug_reason"] = reason

    rows = []
    grp_cols = ["insight_id", "action_name"]
    for (iid, act), g in df_res2.groupby(grp_cols):
        g2 = g[pd.to_numeric(g["z_vs_baseline"], errors="coerce").notna()].copy()
        n = int(len(g2))
        if n <= 0:
            continue
        wins = int(np.sum((pd.to_numeric(g2["success_flag"], errors="coerce").fillna(0) > 0).astype(int)))
        sr = float(wins / float(n)) if n > 0 else np.nan
        zarr = pd.to_numeric(g2["z_vs_baseline"], errors="coerce").to_numpy(dtype=float)
        avgz = float(np.nanmean(zarr)) if n > 0 else np.nan
        medz = float(np.nanmedian(zarr)) if n > 0 else np.nan
        method = _as_str(g2["baseline_method"].dropna().astype(str).iloc[-1]) if n > 0 else ""

        rows.append({
            "insight_id": _as_str(iid),
            "action_name": _as_str(act),
            "n_trials": int(n),
            "n_wins": int(wins),
            "success_rate": float(sr) if np.isfinite(sr) else np.nan,
            "avg_z": float(avgz) if np.isfinite(avgz) else np.nan,
            "median_z": float(medz) if np.isfinite(medz) else np.nan,
            "baseline_method": method,
        })

    out = pd.DataFrame(rows)
    if len(out) > 0:
        out = out.sort_values(["success_rate", "avg_z", "n_trials"], ascending=[False, False, False]).reset_index(drop=True)

    _save_csv(out, EXP_SUMMARY)
    _save_jsonl(out, EXP_SUMMARY_JSONL)

    print("\n=== PCB LEVEL 2.9 (eval) ===")
    print("Data:", data_path)
    print("Trials:", EXP_RESULTS)
    print("Saved:", EXP_SUMMARY)
    print("Saved:", EXP_SUMMARY_JSONL)
    print("Rows:", len(out))

    if len(out) > 0:
        show = ["insight_id", "action_name", "n_trials", "success_rate", "avg_z", "median_z"]
        print("\nTop actions:")
        print(out[show].head(10).to_string(index=False))

    return out

# -----------------------------
# CLI
# -----------------------------
def build_argparser():
    p = argparse.ArgumentParser(
        prog="pcb_experiments_level29.py",
        description="PCB Level 2.9 — Experiment plan + logging + quick eval (local-first)."
    )
    sub = p.add_subparsers(dest="cmd")

    p_plan = sub.add_parser("plan", help="Generate out/experiment_plan.csv from out/insights_level2.csv")
    p_plan.set_defaults(func=cmd_plan)

    p_log = sub.add_parser("log", help="Append one trial row to out/experiment_results.csv")
    p_log.add_argument("--insight_id", required=True, help="Insight ID to log (e.g., I2-00001)")
    p_log.add_argument("--action_name", default="", help="Optional; if empty, will try to load from experiment_plan.csv")
    p_log.add_argument("--date", default="", help="Optional ISO date (YYYY-MM-DD); used to resolve t_index if possible")
    p_log.add_argument("--t_index", type=int, default=None, help="Optional explicit t_index; overrides date mapping")
    p_log.add_argument("--adherence_flag", type=int, default=1, help="0/1; whether the action was actually performed")
    p_log.add_argument("--dose", default="", help="Optional; dose/intensity text")
    p_log.add_argument("--notes", default="", help="Optional notes")
    p_log.add_argument("--data", default=None, help="Optional data path (default: data.csv or out/demo_data.csv)")
    p_log.set_defaults(func=cmd_log)

    p_eval = sub.add_parser("eval", help="Compute out/experiment_summary_level29.csv from experiment_results.csv")
    p_eval.add_argument("--data", default=None, help="Optional data path (default: data.csv or out/demo_data.csv)")
    p_eval.set_defaults(func=cmd_eval)

    return p

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    parser = build_argparser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    args.func(args)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
