#!/usr/bin/env python3
# FILE: pcb_alerts_level28.py
# Python 3.7 compatible
#
# PCB – Level 2.8: Today Alerts (local-first)
#
# Change (v1.1 propagation):
# - Ignore insights with guardrail_flag==1 (even if they exist in insights_level2.csv)
# - Metrics + explainability (enterprise-friendly)
#
import os
import json
import numpy as np
import pandas as pd

# Safe import of invariants module
try:
    from pcb_invariants import check_invariants
    INVARIANTS_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    INVARIANTS_AVAILABLE = False
    def check_invariants(params):
        """Fallback when pcb_invariants is not available"""
        return True, ""

OUT_DIR = "out"

# Defaults (defined early so config override works)
DATE_COL = "date"
TARGET_COL = "target"
LOOKBACK_DAYS = 90
LOOKBACK_ROWS = 90
HARD_WEEKDAY_MATCH = True
MIN_BASELINE_N = 10
Z_TRIGGER = 0.80
Z_CLIP = 6.0
MAX_ALERTS = 15
MIN_STRENGTH = 0.45
PRIORITY_Z_WEIGHT = 0.25


# -----------------------------
# Central config override (optional)
# -----------------------------
try:
    from pcb_config import load_config  # local file
    _CFG = load_config()
except (ImportError, ModuleNotFoundError, Exception):
    _CFG = {}

# Allow overriding common columns/paths via pcb.json (keeps backward compatibility)
OUT_DIR = str(_CFG.get("out_dir", OUT_DIR))
DATE_COL = str(_CFG.get("date_col", DATE_COL))
TARGET_COL = str(_CFG.get("target_col", TARGET_COL))
LOOKBACK_DAYS = int(_CFG.get("level28", {}).get("lookback_days", LOOKBACK_DAYS))
HARD_WEEKDAY_MATCH = bool(_CFG.get("level28", {}).get("hard_weekday_match", HARD_WEEKDAY_MATCH))
MIN_BASELINE_N = int(_CFG.get("level28", {}).get("min_baseline_n", MIN_BASELINE_N))
Z_TRIGGER = float(_CFG.get("level28", {}).get("z_trigger", Z_TRIGGER))
MAX_ALERTS = int(_CFG.get("level28", {}).get("max_alerts", MAX_ALERTS))
MIN_STRENGTH = float(_CFG.get("level28", {}).get("min_strength", MIN_STRENGTH))

INSIGHTS_L2 = os.path.join(OUT_DIR, "insights_level2.csv")
DEFAULT_DATA_CSV = "data.csv"
FALLBACK_DATA_CSV = os.path.join(OUT_DIR, "demo_data.csv")

OUT_ALERTS_CSV = os.path.join(OUT_DIR, "alerts_today_level28.csv")
OUT_ALERTS_JSONL = os.path.join(OUT_DIR, "alerts_today_level28.jsonl")

# NEW: metrics file
OUT_METRICS_JSON = os.path.join(OUT_DIR, "metrics_level28.json")


# -----------------------------
# Guardrails propagation
# -----------------------------
IGNORE_FLAGGED_INSIGHTS = True


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


def _save_json(obj, path):
    _ensure_out()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _load_data_path():
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


def _past_window_indices(df, t_idx):
    n = len(df)
    if t_idx <= 0 or t_idx > n:
        return np.array([], dtype=int)

    if _has_date(df):
        d_t = df[DATE_COL].iloc[t_idx]
        if pd.notna(d_t):
            d0 = d_t.normalize() - pd.Timedelta(days=int(LOOKBACK_DAYS))
            mask = (df[DATE_COL] < d_t) & (df[DATE_COL] >= d0)
            return df.index[mask].to_numpy(dtype=int)

    start = max(0, int(t_idx) - int(LOOKBACK_ROWS))
    return np.arange(start, int(t_idx), dtype=int)


def _zscore_today_source(df, source_col, t_idx):
    if source_col not in df.columns:
        return np.nan, {"reason": "missing_source_col"}

    x = pd.to_numeric(df[source_col], errors="coerce").to_numpy(dtype=float)
    if t_idx < 0 or t_idx >= len(x) or not np.isfinite(x[t_idx]):
        return np.nan, {"reason": "missing_source_today"}

    past_idx = _past_window_indices(df, t_idx)
    if past_idx.size == 0:
        return np.nan, {"reason": "no_past_window"}

    wd = _weekday_series(df).to_numpy(dtype=int)
    cand = past_idx
    if HARD_WEEKDAY_MATCH:
        cand = cand[wd[cand] == int(wd[t_idx])]

    vals = x[cand]
    vals = vals[np.isfinite(vals)]
    if len(vals) < int(MIN_BASELINE_N):
        return np.nan, {"reason": "too_few_baseline", "baseline_n": int(len(vals))}

    mu = float(np.mean(vals))
    sd = float(np.std(vals))
    if sd < 1e-9:
        return np.nan, {"reason": "baseline_sd_zero", "baseline_n": int(len(vals)), "mu": mu}

    z = float((x[t_idx] - mu) / sd)
    z = float(np.clip(z, -Z_CLIP, Z_CLIP))
    meta = {
        "baseline_n": int(len(vals)),
        "mu": mu,
        "sd": sd,
        "method": "same_weekday" if HARD_WEEKDAY_MATCH else "window",
    }
    return z, meta


def _priority(strength, z_abs):
    s = float(np.clip(_safe_float(strength, 0.0), 0.0, 1.0))
    za = float(np.clip(_safe_float(z_abs, 0.0), 0.0, Z_CLIP))
    return float(s * (0.75 + float(PRIORITY_Z_WEIGHT) * (za / 1.0)))


def _make_card(source, lag, z, delta, strength, reason):
    src = source.replace("_", " ")
    d = _safe_float(delta, np.nan)
    s = _safe_float(strength, np.nan)
    zt = _safe_float(z, np.nan)

    if np.isfinite(d) and d > 0:
        dir_txt = "↑ target"
        action_txt = "source is LOW today → consider increasing it"
    elif np.isfinite(d) and d < 0:
        dir_txt = "↓ target"
        action_txt = "source is HIGH today → consider reducing it"
    else:
        dir_txt = "mood"
        action_txt = "watch source today"

    return (
        f"{src} (lag {int(lag)}) → {dir_txt} | "
        f"today z={zt:.2f} | strength={s:.2f} | {action_txt} | {reason}"
    )


# -----------------------------
# Main
# -----------------------------
def main(data_csv_path=None, insights_path=INSIGHTS_L2):
    _ensure_out()

    if not os.path.exists(insights_path):
        raise FileNotFoundError("Missing %s (run Level 2.5 first)." % insights_path)

    # resolve data path early so metrics is correct
    if data_csv_path is None:
        data_csv_path = _load_data_path()
    if not os.path.exists(data_csv_path):
        raise FileNotFoundError("Missing data.csv (or out/demo_data.csv).")

    df_i = pd.read_csv(insights_path)

    metrics = {
        "level": "2.8",
        "level_version": "ignore_flagged+metrics_v1.1",
        "data_path": str(data_csv_path),
        "insights_path": str(insights_path),
        "target_col": str(TARGET_COL),
        "ignore_flagged_insights": int(1 if IGNORE_FLAGGED_INSIGHTS else 0),

        "z_trigger": float(Z_TRIGGER),
        "min_strength": float(MIN_STRENGTH),
        "lookback_days": int(LOOKBACK_DAYS),
        "hard_weekday_match": int(1 if HARD_WEEKDAY_MATCH else 0),
        "min_baseline_n": int(MIN_BASELINE_N),
        "max_alerts_cap": int(MAX_ALERTS),

        "n_insights_total": int(len(df_i)),
        "n_insights_flagged": 0,

        "t_index_today": None,
        "today_date": "",

        "n_candidates_checked": 0,
        "n_skipped_flagged": 0,
        "n_skipped_wrong_target": 0,
        "n_skipped_low_strength": 0,
        "n_skipped_missing_source": 0,
        "n_skipped_z_failed": 0,
        "n_skipped_not_triggered": 0,
        "n_skipped_invariant": 0,
        "invariants_enabled": int(1 if INVARIANTS_AVAILABLE else 0),

        "n_after_basic_filters": 0,
        "n_after_zscore": 0,
        "n_alerts_saved": 0,
    }

    if len(df_i) == 0:
        out = pd.DataFrame(columns=[
            "date", "t_index", "insight_id", "source", "target", "lag",
            "strength", "delta_test",
            "z_source_today", "triggered", "priority", "reason",
            "baseline_n", "baseline_mu", "baseline_sd", "baseline_method",
            "recommended_action", "card",
            "guardrail_flag", "guardrail_reason",
            "shown"
        ])
        _save_csv(out, OUT_ALERTS_CSV)
        _save_jsonl(out, OUT_ALERTS_JSONL)
        metrics["n_alerts_saved"] = 0
        _save_json(metrics, OUT_METRICS_JSON)
        print("No insights in insights_level2.csv. Saved empty alerts.")
        print("Saved:", OUT_METRICS_JSON)
        return out

    # required cols
    for c in ["insight_id", "source", "target", "lag", "strength", "delta_test"]:
        if c not in df_i.columns:
            df_i[c] = np.nan

    # guardrail cols (may be missing on older runs)
    for c in ["guardrail_flag", "guardrail_reason"]:
        if c not in df_i.columns:
            df_i[c] = 0 if c == "guardrail_flag" else ""

    metrics["n_insights_flagged"] = int(
        pd.to_numeric(df_i["guardrail_flag"], errors="coerce").fillna(0).astype(int).sum()
    )

    df = pd.read_csv(data_csv_path)
    df = _try_parse_date(df)

    if TARGET_COL not in df.columns:
        raise ValueError("Target column '%s' not found in data CSV." % TARGET_COL)

    t_idx = int(len(df) - 1)
    today_date = ""
    if _has_date(df):
        d = df[DATE_COL].iloc[t_idx]
        today_date = d.normalize().date().isoformat() if pd.notna(d) else ""

    metrics["t_index_today"] = int(t_idx)
    metrics["today_date"] = str(today_date) if today_date else ""

    alerts = []

    for _, r in df_i.iterrows():
        metrics["n_candidates_checked"] += 1

        iid = _as_str(r.get("insight_id", ""))
        src = _as_str(r.get("source", "")).strip()
        tgt = _as_str(r.get("target", ""))
        lag = int(_safe_float(r.get("lag", 1), 1))
        strength = _safe_float(r.get("strength", 0.0), 0.0)
        delta = _safe_float(r.get("delta_test", np.nan), np.nan)

        gr_flag = int(_safe_float(r.get("guardrail_flag", 0), 0))
        gr_reason = _as_str(r.get("guardrail_reason", ""))

        if IGNORE_FLAGGED_INSIGHTS and gr_flag == 1:
            metrics["n_skipped_flagged"] += 1
            continue

        if tgt != TARGET_COL:
            metrics["n_skipped_wrong_target"] += 1
            continue
        if strength < float(MIN_STRENGTH):
            metrics["n_skipped_low_strength"] += 1
            continue
        if not src:
            metrics["n_skipped_missing_source"] += 1
            continue

        z, meta = _zscore_today_source(df, src, t_idx)
        if not np.isfinite(z):
            metrics["n_skipped_z_failed"] += 1
            continue

        triggered = 0
        reco = ""
        reason = ""

        if np.isfinite(delta) and delta > 0:
            if z <= -float(Z_TRIGGER):
                triggered = 1
                reco = "increase_source"
                reason = "source_low_and_positive_effect"
        elif np.isfinite(delta) and delta < 0:
            if z >= float(Z_TRIGGER):
                triggered = 1
                reco = "reduce_source"
                reason = "source_high_and_negative_effect"
        else:
            triggered = 0
            reason = "delta_unknown_conservative_no_trigger"

        if triggered != 1:
            metrics["n_skipped_not_triggered"] += 1
            continue

        # Invariants: hard safety rules (block even if triggered)
        ok, inv_reason = check_invariants({
            "level": "2.8",
            "source": src,
            "target": tgt,
            "lag": lag,
            "effect_size": float(strength),
            "guardrail_flag": int(gr_flag),
            "action": reco,
        })
        if not ok:
            metrics["n_skipped_invariant"] += 1
            continue

        pr = _priority(strength, abs(z))
        card = _make_card(src, lag, z, delta, strength, reason)

        alerts.append({
            "date": today_date,
            "t_index": int(t_idx),
            "insight_id": iid,
            "source": src,
            "target": tgt,
            "lag": int(lag),
            "strength": float(strength),
            "delta_test": float(delta) if np.isfinite(delta) else np.nan,
            "z_source_today": float(z),
            "triggered": int(triggered),
            "priority": float(pr),
            "reason": reason,
            "baseline_n": int(meta.get("baseline_n", 0)) if meta else 0,
            "baseline_mu": float(meta.get("mu", np.nan)) if meta else np.nan,
            "baseline_sd": float(meta.get("sd", np.nan)) if meta else np.nan,
            "baseline_method": _as_str(meta.get("method", "")) if meta else "",
            "recommended_action": reco,
            "card": card,
            "guardrail_flag": int(gr_flag),
            "guardrail_reason": gr_reason,
            "shown": 0,
        })

    df_a = pd.DataFrame(alerts)
    if len(df_a) == 0:
        df_a = pd.DataFrame(columns=[
            "date", "t_index", "insight_id", "source", "target", "lag",
            "strength", "delta_test",
            "z_source_today", "triggered", "priority", "reason",
            "baseline_n", "baseline_mu", "baseline_sd", "baseline_method",
            "recommended_action", "card",
            "guardrail_flag", "guardrail_reason",
            "shown"
        ])
    else:
        df_a = df_a.sort_values(["priority", "strength"], ascending=[False, False]).reset_index(drop=True)
        df_a = df_a.head(int(MAX_ALERTS)).copy()

    metrics["n_after_basic_filters"] = int(
        metrics["n_candidates_checked"]
        - metrics["n_skipped_flagged"]
        - metrics["n_skipped_wrong_target"]
        - metrics["n_skipped_low_strength"]
        - metrics["n_skipped_missing_source"]
    )
    metrics["n_after_zscore"] = int(
        metrics["n_candidates_checked"]
        - metrics["n_skipped_flagged"]
        - metrics["n_skipped_wrong_target"]
        - metrics["n_skipped_low_strength"]
        - metrics["n_skipped_missing_source"]
        - metrics["n_skipped_z_failed"]
    )
    metrics["n_alerts_saved"] = int(len(df_a))

    _save_csv(df_a, OUT_ALERTS_CSV)
    _save_jsonl(df_a, OUT_ALERTS_JSONL)
    _save_json(metrics, OUT_METRICS_JSON)

    print("\n=== PCB LEVEL 2.8 (alerts today) ===")
    print("Data:", data_csv_path)
    print("Insights:", insights_path)
    print("Today t_index:", t_idx, "date:", today_date if today_date else "(no date)")
    print("Saved:", OUT_ALERTS_CSV)
    print("Saved:", OUT_ALERTS_JSONL)
    print("Saved:", OUT_METRICS_JSON)
    print("Alerts:", len(df_a))
    print("Guardrail ignore policy:", "ON" if IGNORE_FLAGGED_INSIGHTS else "OFF")
    print("Invariants module:", "LOADED" if INVARIANTS_AVAILABLE else "NOT AVAILABLE (using fallback)")

    if len(df_a) > 0:
        show = ["insight_id", "source", "lag", "z_source_today", "strength", "priority", "reason"]
        print("\nTop alerts:")
        print(df_a[show].head(10).to_string(index=False))

    return df_a


if __name__ == "__main__":
    main()
