# FILE: pcb_level3_engine_32.py
# Python 3.7 compatible
#
# PCB – Level 3.2: Counterfactual Causal Validation Engine (local-first)
#
# Goal:
# - Validate Level 2.5 insights via counterfactual matching using past-only controls
# - No leakage: controls drawn only from times strictly before each trial t_index
# - Conservative success definition (lower bound)
# - Audit trail / ledger
#
# Inputs:
# - out/insights_level2.csv
# - out/experiment_results.csv   (from Level 2.9 log)
# - data.csv (or out/demo_data.csv, or out/data_clean.csv if you pass --data)
#
# Outputs:
# - out/insights_level3.csv
# - out/insights_level3_ledger.csv
# - out/experiment_trials_enriched_level32.csv
#
# Dependencies: numpy, pandas only

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd


from pcb_invariants import check_invariants
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

INSIGHTS_L2 = os.path.join(OUT_DIR, "insights_level2.csv")
EXP_RESULTS = os.path.join(OUT_DIR, "experiment_results.csv")
EXP_INTAKE_L31 = os.path.join(OUT_DIR, "experiment_intake_level31.csv")

def _pick_trials_path():
    return EXP_INTAKE_L31 if os.path.exists(EXP_INTAKE_L31) else EXP_RESULTS


OUT_L3 = os.path.join(OUT_DIR, "insights_level3.csv")
OUT_LEDGER = os.path.join(OUT_DIR, "insights_level3_ledger.csv")
OUT_TRIALS = os.path.join(OUT_DIR, "experiment_trials_enriched_level32.csv")

DEFAULT_DATA_CSV = "data.csv"
FALLBACK_DATA_CSV = os.path.join(OUT_DIR, "demo_data.csv")

TARGET_COL = str(_CFG.get("target_col", "target"))
DATE_COL = str(_CFG.get("date_col", "date"))

# Enterprise: negative control outcome (optional)
_L32_CFG = _CFG.get("level32", {}) if isinstance(_CFG, dict) else {}
NEGCTRL_ENABLE = bool(_L32_CFG.get("negative_control_enable", True))
NEGCTRL_OUTCOME_COL = str(_L32_CFG.get("negative_control_outcome_col", "negative_control_outcome"))
NEGCTRL_MAX_SUCCESS_LB = float(_L32_CFG.get("negative_control_max_success_lb", 0.55))

# Enterprise causal upgrades (optional)
ENABLE_PROPENSITY = bool(_L32_CFG.get("enable_propensity", True))
PROPENSITY_MAX_DIFF = float(_L32_CFG.get("propensity_max_diff", 0.20))
PROPENSITY_ACTION_COL = str(_L32_CFG.get("propensity_action_col", "action_active"))

ENABLE_PRETREND_CHECK = bool(_L32_CFG.get("enable_pretrend_check", True))
PRETREND_DAYS = int(_L32_CFG.get("pretrend_days", 7))
PRETREND_MAX_DIFF = float(_L32_CFG.get("pretrend_max_diff", 0.30))

# -----------------------------
# Knobs (conservative defaults)
# -----------------------------
LOOKBACK_DAYS = 90
LOOKBACK_ROWS = 90

K_CONTROLS = 10
MIN_MATCHED = 5
MATCH_DIST_MAX = 1.5  # mean absolute robust-scaled distance

Z_SUCCESS = 0.20
Z_CLIP = 6.0

MIN_TRIALS = 2
MIN_SUCCESS_LB = 0.55  # conservative lower bound

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

def _load_data_path(path=None):
    if path and os.path.exists(path):
        return path
    if os.path.exists(DEFAULT_DATA_CSV):
        return DEFAULT_DATA_CSV
    # prefer data_clean.csv if present
    dc = os.path.join(OUT_DIR, "data_clean.csv")
    if os.path.exists(dc):
        return dc
    return FALLBACK_DATA_CSV

def _try_parse_date(df):
    if DATE_COL in df.columns:
        d = pd.to_datetime(df[DATE_COL], errors="coerce")
        if d.notna().mean() > 0.2:
            df = df.copy()
            df[DATE_COL] = d
    return df

def _has_date(df):
    return (DATE_COL in df.columns) and pd.api.types.is_datetime64_any_dtype(df[DATE_COL])

def _past_indices(df, t_idx):
    if t_idx <= 0:
        return np.array([], dtype=int)

    if _has_date(df):
        d = df[DATE_COL].iloc[t_idx]
        if pd.notna(d):
            start = d.normalize() - pd.Timedelta(days=int(LOOKBACK_DAYS))
            mask = (df[DATE_COL] < d) & (df[DATE_COL] >= start)
            return df.index[mask].to_numpy(dtype=int)

    start = max(0, int(t_idx) - int(LOOKBACK_ROWS))
    return np.arange(start, int(t_idx), dtype=int)

def _robust_scale(x):
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    iqr = np.nanpercentile(x, 75) - np.nanpercentile(x, 25)
    if not np.isfinite(iqr) or iqr < 1e-6:
        iqr = np.nanstd(x)
    if not np.isfinite(iqr) or iqr < 1e-6:
        iqr = 1.0
    return (x - med) / iqr

def _sr_lower_bound(wins, n):
    # very conservative (one-success penalty)
    if n <= 0:
        return 0.0
    return max(0.0, (wins - 1) / float(n))

def _numeric_cols(df):
    cols = []
    for c in df.columns:
        if c in (DATE_COL,):
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() >= max(10, int(0.3 * len(df))):
            cols.append(c)
    return cols


def _ensure_calendar_covs(df):
    """Add conservative calendar/trend covariates for matching (enterprise-safe)."""
    out = df.copy()
    out["time_idx"] = np.arange(len(out), dtype=float)
    if _has_date(out):
        dow = out[DATE_COL].dt.dayofweek.astype(float)
        out["dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
        out["dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0)
    return out


def _sigmoid(z):
    z = np.asarray(z, dtype=float)
    z = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def _fit_logistic_proba(X, y, l2=1.0, steps=250, lr=0.1):
    """Lightweight logistic regression (no sklearn).

    X: (n,k) finite array
    y: (n,) in {0,1}
    Returns (w, mu, sig) such that p = sigmoid([1,(X-mu)/sig] @ w).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, k = X.shape

    mu = np.nanmean(X, axis=0)
    sig = np.nanstd(X, axis=0)
    sig = np.where(np.isfinite(sig) & (sig > 1e-6), sig, 1.0)
    Xs = (X - mu) / sig
    A = np.column_stack([np.ones(n), Xs])

    w = np.zeros(k + 1, dtype=float)
    l2 = float(max(0.0, l2))
    for _ in range(int(steps)):
        p = _sigmoid(A @ w)
        # gradient
        g = (A.T @ (p - y)) / float(n)
        # L2 on non-intercept
        g[1:] += (l2 / float(n)) * w[1:]
        w -= float(lr) * g
    return w, mu, sig


def _predict_logistic_proba(X, w, mu, sig):
    X = np.asarray(X, dtype=float)
    Xs = (X - mu) / sig
    A = np.column_stack([np.ones(len(Xs)), Xs])
    return _sigmoid(A @ w)


def _compute_propensity(df, action_col, covs):
    """Compute propensity p(treated|covs). Returns array (len(df),) with NaNs if not computable."""
    if action_col not in df.columns:
        return np.full(len(df), np.nan, dtype=float)

    cols = [c for c in covs if c in df.columns]
    if len(cols) == 0:
        return np.full(len(df), np.nan, dtype=float)

    X = []
    for c in cols:
        X.append(pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float))
    X = np.vstack(X).T
    y = pd.to_numeric(df[action_col], errors="coerce").fillna(0).to_numpy(dtype=float)

    m = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if int(np.sum(m)) < 25:
        return np.full(len(df), np.nan, dtype=float)

    yy = y[m]
    # must have both classes
    if (np.sum(yy > 0.5) < 5) or (np.sum(yy <= 0.5) < 5):
        return np.full(len(df), np.nan, dtype=float)

    Xm = X[m]
    w, mu, sig = _fit_logistic_proba(Xm, (yy > 0.5).astype(float), l2=1.0, steps=300, lr=0.15)

    p_all = np.full(len(df), np.nan, dtype=float)
    p_all[m] = _predict_logistic_proba(Xm, w, mu, sig)
    return p_all


def _trend_slope(y):
    y = np.asarray(y, dtype=float)
    m = np.isfinite(y)
    if int(np.sum(m)) < 3:
        return np.nan
    x = np.arange(len(y), dtype=float)[m]
    yy = y[m]
    try:
        a, b = np.polyfit(x, yy, 1)
        return float(a)
    except Exception:
        return np.nan


def _pretrend_check(df, t_idx, controls, days):
    """Pre-trend check: compare slope of target in window before trial vs controls.

    Returns (pass_flag, slope_trial, slope_ctrl_mean, diff_abs, reason).
    """
    days = int(max(3, days))
    y = pd.to_numeric(df[TARGET_COL], errors="coerce").to_numpy(dtype=float)

    def window(idx):
        a = max(0, int(idx) - days)
        b = int(idx)
        if b - a < 3:
            return None
        return y[a:b]

    yt = window(t_idx)
    if yt is None:
        return 1, np.nan, np.nan, np.nan, "too_early"

    st = _trend_slope(yt)
    if not np.isfinite(st):
        return 1, np.nan, np.nan, np.nan, "trial_slope_nan"

    sc = []
    for ci in controls:
        yc = window(ci)
        if yc is None:
            continue
        s = _trend_slope(yc)
        if np.isfinite(s):
            sc.append(float(s))
    if len(sc) < 3:
        return 1, float(st), np.nan, np.nan, "too_few_control_trends"

    sc_mean = float(np.mean(sc))
    diff = float(abs(float(st) - float(sc_mean)))

    # Scale by per-day target variability
    scale = float(np.nanstd(y))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    per_day = scale / float(days)
    thresh = float(PRETREND_MAX_DIFF) * per_day
    ok = int(diff <= thresh)
    return ok, float(st), sc_mean, diff, ""

# -----------------------------
# Matching
# -----------------------------
def match_controls(df, t_idx, covs):
    past = _past_indices(df, t_idx)
    if len(past) == 0:
        return [], {"reason": "no_past"}

    Xcols = []
    for c in covs:
        if c not in df.columns:
            continue
        col = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        Xcols.append(_robust_scale(col))

    if len(Xcols) == 0:
        return [], {"reason": "no_covariates"}

    X = np.vstack(Xcols).T
    x_t = X[t_idx]
    if not np.all(np.isfinite(x_t)):
        return [], {"reason": "trial_cov_missing"}

    dists = []
    for i in past:
        if not np.all(np.isfinite(X[i])):
            continue
        d = float(np.mean(np.abs(X[i] - x_t)))
        dists.append((int(i), d))

    dists.sort(key=lambda z: z[1])
    dists = [(i, d) for (i, d) in dists if d <= float(MATCH_DIST_MAX)]

    if len(dists) < int(MIN_MATCHED):
        return [], {"reason": "too_few_matches", "n": int(len(dists))}

    chosen = dists[: int(K_CONTROLS)]
    return [i for (i, _) in chosen], {"reason": "", "n_candidates": int(len(dists)), "avg_dist_top": float(np.mean([d for _, d in chosen]))}

def z_from_controls_col(df, t_idx, controls, col_name):
    y = pd.to_numeric(df[col_name], errors="coerce").to_numpy(dtype=float)
    if t_idx < 0 or t_idx >= len(y) or not np.isfinite(y[t_idx]):
        return np.nan, {"reason": "missing_outcome_at_trial", "outcome": str(col_name)}

    vals = y[controls]
    vals = vals[np.isfinite(vals)]
    if len(vals) < 2:
        return np.nan, {"reason": "too_few_control_targets"}

    mu = float(np.mean(vals))
    sd = float(np.std(vals))
    if not np.isfinite(sd) or sd < 1e-6:
        return np.nan, {"reason": "control_sd_zero", "mu": mu}

    z = float((y[t_idx] - mu) / sd)
    z = float(np.clip(z, -Z_CLIP, Z_CLIP))
    return z, {"mu": mu, "sd": sd, "n": int(len(vals))}


def z_from_controls(df, t_idx, controls):
    """Backwards-compatible helper for the main target outcome."""
    return z_from_controls_col(df, t_idx, controls, TARGET_COL)

# -----------------------------
# Main
# -----------------------------
def main(data_path=None):
    _ensure_out()

    if not os.path.exists(INSIGHTS_L2):
        raise FileNotFoundError("Missing %s (run Level 2.5 first)." % INSIGHTS_L2)
    trials_path = _pick_trials_path()
    if not os.path.exists(trials_path):
        # Graceful empty-run behavior: Level 3.2 depends on logged trials.
        # If no trials yet, write empty outputs and exit 0 (enterprise-friendly).
        _ensure_out()
        empty_trials = pd.DataFrame(columns=["insight_id","action_name","date","t_index","adherence_flag","dose","notes"])
        empty_trials.to_csv(OUT_TRIALS, index=False)
        empty_l3 = pd.DataFrame(columns=["insight_id","action_name","n_trials","success_rate","avg_z","median_z","verdict","notes"])
        empty_l3.to_csv(OUT_L3, index=False)
        empty_ledger = pd.DataFrame(columns=["date","event","details"])
        empty_ledger.to_csv(OUT_LEDGER, index=False)
        print("No experiment_results.csv found. Level 3.2 needs at least 1 logged trial.")
        print("Wrote empty outputs:")
        print(" -", OUT_TRIALS)
        print(" -", OUT_L3)
        print(" -", OUT_LEDGER)
        print("Next step: run `python pcb_experiments_level29.py log --insight_id I2-00001 --notes \"...\")` and then rerun Level 3.2.")
        return 0

    df_i = pd.read_csv(INSIGHTS_L2)
    df_r = pd.read_csv(trials_path)

    data_path = _load_data_path(data_path)
    df = pd.read_csv(data_path)
    df = _try_parse_date(df)
    df = _ensure_calendar_covs(df)

    if TARGET_COL not in df.columns:
        raise ValueError("Target column '%s' not found in data." % TARGET_COL)

    # Ensure previous-target covariate exists (regression-to-mean control).
    # Backward compatible: keep mood_prev and also create a neutral alias target_prev.
    if "mood_prev" not in df.columns:
        df["mood_prev"] = pd.to_numeric(df[TARGET_COL], errors="coerce").astype(float).shift(1)
    if "target_prev" not in df.columns and "mood_prev" in df.columns:
        df["target_prev"] = df["mood_prev"]

    # Covariates: start with past target + simple calendar/trend
    covs = []
    if "target_prev" in df.columns:
        covs.append("target_prev")
    elif "mood_prev" in df.columns:
        covs.append("mood_prev")

    for c in ["time_idx", "dow_sin", "dow_cos"]:
        if c in df.columns and c not in covs:
            covs.append(c)

    for c in _numeric_cols(df):
        if c not in (TARGET_COL, DATE_COL) and c not in covs:
            covs.append(c)
        if len(covs) >= 10:  # keep small & stable
            break

    # Derive an action indicator for propensity scoring (event-based treatment).
    # If the column doesn't exist in the dataset, we build it from experiment_results (t_index=1).
    if PROPENSITY_ACTION_COL not in df.columns:
        df[PROPENSITY_ACTION_COL] = 0
        for _t in pd.to_numeric(df_r.get("t_index"), errors="coerce").fillna(-999).astype(int).tolist():
            if 0 <= int(_t) < len(df):
                df.loc[int(_t), PROPENSITY_ACTION_COL] = 1

    # Compute propensity scores once (optional).
    propensity = np.full(len(df), np.nan, dtype=float)
    if ENABLE_PROPENSITY:
        propensity = _compute_propensity(df, PROPENSITY_ACTION_COL, covs)
        df["__propensity__"] = propensity

    # Trial enrichment
    trial_rows = []
    ledger_rows = []

    for _, tr in df_r.iterrows():
        iid = _as_str(tr.get("insight_id", "")).strip()
        t = _safe_float(tr.get("t_index", np.nan), np.nan)
        if not iid or not np.isfinite(t):
            continue
        t_idx = int(t)
        if t_idx < 0 or t_idx >= len(df):
            continue

        controls, mmeta = match_controls(df, t_idx, covs)
        z, zmeta = (np.nan, {})
        if controls:
            z, zmeta = z_from_controls(df, t_idx, controls)

        # --- Enterprise causal checks (propensity + pre-trend)
        propensity_t = np.nan
        propensity_c_mean = np.nan
        propensity_pass = 1
        if ENABLE_PROPENSITY and controls and ("__propensity__" in df.columns):
            pvec = pd.to_numeric(df["__propensity__"], errors="coerce").to_numpy(dtype=float)
            if 0 <= t_idx < len(pvec) and np.isfinite(pvec[t_idx]):
                propensity_t = float(pvec[t_idx])
                pcs = pvec[np.asarray(controls, dtype=int)]
                pcs = pcs[np.isfinite(pcs)]
                if len(pcs) > 0:
                    propensity_c_mean = float(np.mean(pcs))
                    propensity_pass = int(abs(float(propensity_t) - float(propensity_c_mean)) <= float(PROPENSITY_MAX_DIFF))

        pretrend_pass = 1
        pretrend_slope_trial = np.nan
        pretrend_slope_ctrl = np.nan
        pretrend_diff = np.nan
        pretrend_reason = ""
        if ENABLE_PRETREND_CHECK and controls:
            pretrend_pass, pretrend_slope_trial, pretrend_slope_ctrl, pretrend_diff, pretrend_reason = _pretrend_check(df, t_idx, controls, PRETREND_DAYS)

        eligible_flag = int((len(controls) > 0) and (propensity_pass == 1) and (pretrend_pass == 1))

        # Negative control: compute the same counterfactual z for a metric that should NOT move.
        z_negctrl = np.nan
        success_flag_negctrl = np.nan
        if NEGCTRL_ENABLE and controls and (NEGCTRL_OUTCOME_COL in df.columns):
            z_negctrl, _ = z_from_controls_col(df, t_idx, controls, NEGCTRL_OUTCOME_COL)
            if np.isfinite(z_negctrl):
                # We treat large absolute movements as "suspicious".
                success_flag_negctrl = int(abs(float(z_negctrl)) >= float(Z_SUCCESS))

        success_flag = np.nan
        if np.isfinite(z):
            success_flag = int(float(z) >= float(Z_SUCCESS))

        trial_rows.append({
            "insight_id": iid,
            "t_index": int(t_idx),
            "date": _as_str(tr.get("date", "")),
            "action_name": _as_str(tr.get("action_name", "")),
            "adherence_flag": _safe_float(tr.get("adherence_flag", np.nan), np.nan),
            "z_cf": float(z) if np.isfinite(z) else np.nan,
            "success_flag": success_flag,

            # Enterprise causal checks
            "eligible_flag": int(eligible_flag),
            "propensity_treated": float(propensity_t) if np.isfinite(propensity_t) else np.nan,
            "propensity_controls_mean": float(propensity_c_mean) if np.isfinite(propensity_c_mean) else np.nan,
            "propensity_pass": int(propensity_pass),
            "pretrend_pass": int(pretrend_pass),
            "pretrend_slope_trial": float(pretrend_slope_trial) if np.isfinite(pretrend_slope_trial) else np.nan,
            "pretrend_slope_controls_mean": float(pretrend_slope_ctrl) if np.isfinite(pretrend_slope_ctrl) else np.nan,
            "pretrend_diff_abs": float(pretrend_diff) if np.isfinite(pretrend_diff) else np.nan,
            "pretrend_reason": _as_str(pretrend_reason),

            # Enterprise robustness
            "z_negctrl": float(z_negctrl) if np.isfinite(z_negctrl) else np.nan,
            "success_flag_negctrl": success_flag_negctrl,
            "matched_n": int(len(controls)),
            "baseline_mu": float(zmeta.get("mu", np.nan)) if zmeta else np.nan,
            "baseline_sd": float(zmeta.get("sd", np.nan)) if zmeta else np.nan,
            "match_reason": _as_str(mmeta.get("reason", "")) if mmeta else "",
            "match_avg_dist_top": float(mmeta.get("avg_dist_top", np.nan)) if mmeta else np.nan,
            "covariates_used": "|".join(covs),
            "controls_idx": "|".join([str(i) for i in controls[: int(K_CONTROLS)]]),
        })

        # Ledger per-trial (audit)
        ledger_rows.append({
            "insight_id": iid,
            "t_index": int(t_idx),
            "matched_n": int(len(controls)),
            "controls_idx": "|".join([str(i) for i in controls[: int(K_CONTROLS)]]),
            "avg_dist_top": float(mmeta.get("avg_dist_top", np.nan)) if mmeta else np.nan,
            "z_cf": float(z) if np.isfinite(z) else np.nan,
            "success_flag": success_flag,

            "eligible_flag": int(eligible_flag),
            "propensity_treated": float(propensity_t) if np.isfinite(propensity_t) else np.nan,
            "propensity_controls_mean": float(propensity_c_mean) if np.isfinite(propensity_c_mean) else np.nan,
            "propensity_pass": int(propensity_pass),
            "pretrend_pass": int(pretrend_pass),
            "pretrend_slope_trial": float(pretrend_slope_trial) if np.isfinite(pretrend_slope_trial) else np.nan,
            "pretrend_slope_controls_mean": float(pretrend_slope_ctrl) if np.isfinite(pretrend_slope_ctrl) else np.nan,
            "pretrend_diff_abs": float(pretrend_diff) if np.isfinite(pretrend_diff) else np.nan,

            "z_negctrl": float(z_negctrl) if np.isfinite(z_negctrl) else np.nan,
            "success_flag_negctrl": success_flag_negctrl,
        })

    df_trials = pd.DataFrame(trial_rows)
    df_ledger = pd.DataFrame(ledger_rows)

    df_trials.to_csv(OUT_TRIALS, index=False)
    df_ledger.to_csv(OUT_LEDGER, index=False)

    # Aggregate per insight
    out_rows = []
    if len(df_trials) > 0:
        for iid, g in df_trials.groupby("insight_id"):
            g2 = g[(pd.to_numeric(g["z_cf"], errors="coerce").notna()) & (pd.to_numeric(g.get("eligible_flag", 1), errors="coerce").fillna(1).astype(int) == 1)].copy()
            n = int(len(g2))
            wins = int(np.sum((pd.to_numeric(g2["success_flag"], errors="coerce").fillna(0) > 0).astype(int))) if n > 0 else 0
            lb = float(_sr_lower_bound(wins, n)) if n > 0 else 0.0
            avgz = float(np.nanmean(pd.to_numeric(g2["z_cf"], errors="coerce").to_numpy(dtype=float))) if n > 0 else np.nan

            # Negative control aggregate (enterprise robustness)
            negctrl_lb = np.nan
            if NEGCTRL_ENABLE and ("success_flag_negctrl" in g.columns):
                gnc = g[pd.to_numeric(g["success_flag_negctrl"], errors="coerce").notna()].copy()
                nn = int(len(gnc))
                if nn > 0:
                    wins_nc = int(np.sum((pd.to_numeric(gnc["success_flag_negctrl"], errors="coerce").fillna(0) > 0).astype(int)))
                    negctrl_lb = float(_sr_lower_bound(wins_nc, nn))

            negctrl_pass = 1
            if NEGCTRL_ENABLE and np.isfinite(negctrl_lb):
                negctrl_pass = int(float(negctrl_lb) <= float(NEGCTRL_MAX_SUCCESS_LB))

            status = "candidate"
            if n >= int(MIN_TRIALS) and lb >= float(MIN_SUCCESS_LB) and (negctrl_pass == 1):
                status = "action_supported"

            out_rows.append({
                "insight_id": _as_str(iid),
                "n_trials": int(n),
                "n_wins": int(wins),
                "success_rate_lb": float(lb),
                "avg_z_cf": float(avgz) if np.isfinite(avgz) else np.nan,
                "negctrl_success_lb": float(negctrl_lb) if np.isfinite(negctrl_lb) else np.nan,
                "negctrl_pass": int(negctrl_pass),
                "status": status,
            })

    df_l3 = pd.DataFrame(out_rows)
    df_l3.to_csv(OUT_L3, index=False)

    print("\n=== PCB LEVEL 3.2 (counterfactual validation) ===")
    print("Data:", data_path)
    print("Inputs:", INSIGHTS_L2, "+", EXP_RESULTS)
    print("Saved:", OUT_TRIALS)
    print("Saved:", OUT_LEDGER)
    print("Saved:", OUT_L3)
    print("Trials enriched:", int(len(df_trials)))
    if len(df_trials) > 0 and "eligible_flag" in df_trials.columns:
        excl = int(np.sum(pd.to_numeric(df_trials["eligible_flag"], errors="coerce").fillna(1).astype(int) == 0))
        print("Trials excluded by propensity/pre-trend:", excl)
    print("Insights evaluated:", int(len(df_l3)))

    return df_l3, df_trials, df_ledger

# -----------------------------
# CLI
# -----------------------------
def build_argparser():
    p = argparse.ArgumentParser(
        prog="pcb_level3_engine_32.py",
        description="PCB Level 3.2 — Counterfactual causal validation (local-first)."
    )
    p.add_argument("--data", default=None, help="Optional data path (default: data.csv or out/data_clean.csv or out/demo_data.csv)")
    return p

def cli(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    args = build_argparser().parse_args(argv)
    main(data_path=args.data)
    return 0

if __name__ == "__main__":
    raise SystemExit(cli())
