#!/usr/bin/env python3
# FILE: pcb_insights_level25.py
# Python 3.7 compatible
#
# PCB – Level 2.5: Insight Discovery + Validation (local-first) + Guardrails v1.1
#
# Output:
#   out/edges.csv
#   out/insights_level2.csv
#   out/insights_level2.jsonl
#
# Dependencies: numpy, pandas only
#
import os
import json
import numpy as np
import pandas as pd

OUT_DIR = "out"

# -----------------------------
DEFAULT_DATA_CSV = "data.csv"
FALLBACK_DATA_CSV = os.path.join(OUT_DIR, "demo_data.csv")

# -----------------------------
# Product knobs (Level 2.5)
# -----------------------------
TARGET_COL = "target"
DATE_COL = "date"

DETREND_MODE = "none"   # "none" | "diff1"
MAX_LAG = 7
MIN_SUPPORT_N = 25
# NOTE: this set depends on TARGET_COL/DATE_COL and is recomputed after config override.
CANDIDATE_EXCLUDE = set([TARGET_COL, DATE_COL])

Q_LOW = 0.30
Q_HIGH = 0.70

WINDOW_LEN = 14
WINDOW_STRIDE = 7
MIN_WINDOWS = 3

BOOT_B = 200
BOOT_ALPHA = 0.10  # 90% CI

MAX_NAN_FRAC = 0.60
MIN_UNIQUE_NUM = 6
MIN_STD_EPS = 1e-6

MIN_STRENGTH = 0.30
MIN_P_SIGN = 0.55
MIN_EFFECT_ABS = 0.02

# --- Causality hardening (conservative defaults; override via pcb.json -> level25)
ADJUSTMENT_MODE = "full"   # "off" | "light" | "full"
PLACEBO_ENABLE = True
PLACEBO_FUTURE_ENABLE = True
PLACEBO_PERM_ENABLE = True
PLACEBO_PERM_B = 10
PLACEBO_BLOCK_LEN = 7
PLACEBO_MARGIN = 0.01

# --- Enterprise robustness upgrades
NEGCTRL_ENABLE = True
NEGCTRL_OUTCOME_COL = "negative_control_outcome"
NEGCTRL_MAX_STRENGTH = 0.35
NEGCTRL_MARGIN = 0.10

STABILITY_ENABLE = True
STABILITY_SLICES = ["weekday", "weekend", "first_half", "second_half"]
STABILITY_MIN_SCORE = 0.50

# -----------------------------
# Optional central config override via pcb_config.load_config (single source of truth)
# -----------------------------
try:
    from pcb_config import load_config  # local file
    _CFG = load_config()
except Exception:
    _CFG = {}

OUT_DIR = str(_CFG.get("out_dir", OUT_DIR))
DATE_COL = str(_CFG.get("date_col", DATE_COL))
TARGET_COL = str(_CFG.get("target_col") or _CFG.get("target") or TARGET_COL)

lvl25 = _CFG.get("level25", {}) if isinstance(_CFG, dict) else {}
DETREND_MODE = str(lvl25.get("detrend_mode", DETREND_MODE))
MAX_LAG = int(lvl25.get("max_lag", MAX_LAG))
MIN_SUPPORT_N = int(lvl25.get("min_support_n", MIN_SUPPORT_N))
MIN_STRENGTH = float(lvl25.get("min_strength", MIN_STRENGTH))
MIN_P_SIGN = float(lvl25.get("min_p_sign", MIN_P_SIGN))
MIN_EFFECT_ABS = float(lvl25.get("min_effect_abs", MIN_EFFECT_ABS))

ADJUSTMENT_MODE = str(lvl25.get("adjustment_mode", ADJUSTMENT_MODE))
PLACEBO_ENABLE = bool(lvl25.get("placebo_enable", PLACEBO_ENABLE))
PLACEBO_FUTURE_ENABLE = bool(lvl25.get("placebo_future_enable", PLACEBO_FUTURE_ENABLE))
PLACEBO_PERM_ENABLE = bool(lvl25.get("placebo_perm_enable", PLACEBO_PERM_ENABLE))
PLACEBO_PERM_B = int(lvl25.get("placebo_perm_B", PLACEBO_PERM_B))
PLACEBO_BLOCK_LEN = int(lvl25.get("placebo_block_len", PLACEBO_BLOCK_LEN))
PLACEBO_MARGIN = float(lvl25.get("placebo_margin", PLACEBO_MARGIN))

NEGCTRL_ENABLE = bool(lvl25.get("negative_control_enable", NEGCTRL_ENABLE))
NEGCTRL_OUTCOME_COL = str(lvl25.get("negative_control_outcome_col", NEGCTRL_OUTCOME_COL))
NEGCTRL_MAX_STRENGTH = float(lvl25.get("negative_control_max_strength", NEGCTRL_MAX_STRENGTH))
NEGCTRL_MARGIN = float(lvl25.get("negative_control_margin", NEGCTRL_MARGIN))

STABILITY_ENABLE = bool(lvl25.get("stability_enable", STABILITY_ENABLE))
STABILITY_SLICES = list(lvl25.get("stability_slices", STABILITY_SLICES))
STABILITY_MIN_SCORE = float(lvl25.get("stability_min_score", STABILITY_MIN_SCORE))

# recompute exclude set after overrides
CANDIDATE_EXCLUDE = set([TARGET_COL, DATE_COL])

# Derived paths (depend on OUT_DIR)
EDGES_PATH = os.path.join(OUT_DIR, "edges.csv")
OUT_CSV = os.path.join(OUT_DIR, "insights_level2.csv")
OUT_JSONL = os.path.join(OUT_DIR, "insights_level2.jsonl")
FALLBACK_DATA_CSV = os.path.join(OUT_DIR, "demo_data.csv")

W_EFF = 0.45
W_PSIGN = 0.35
W_N = 0.20

USE_CI_WIDTH_PENALTY = True
CI_WIDTH_SOFT_MAX = 1.0
CI_PENALTY_WEIGHT = 0.20

# -----------------------------
# Guardrails (enterprise-safe) — v1.1
# -----------------------------
ENABLE_GUARDRAILS = True

# Leakage v1.1 (computed per-lag)
LEAKAGE_FUTURE_CORR_HARD = 0.95
LEAKAGE_FUTURE_CORR_SOFT = 0.80
LEAKAGE_GAP_MIN = 0.20  # abs(corr_future) - abs(corr_now)

# Drift v1.1 (computed per-source)
DRIFT_TIME_CORR_SOURCE = 0.85
DRIFT_TIME_CORR_TARGET = 0.40

# If True, flagged insights are excluded from insights_level2.csv (but remain in edges.csv)
DROP_FLAGGED_INSIGHTS = True

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
        if x is None:
            return ""
        s = str(x)
        return "" if s.lower() == "nan" else s
    except Exception:
        return ""

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

def _target_col_name(target_col):
    if DETREND_MODE == "diff1":
        return target_col + "_diff1"
    return target_col

def _detrend_target_series(s):
    ss = pd.to_numeric(s, errors="coerce").astype(float)
    if DETREND_MODE == "diff1":
        return ss - ss.shift(1)
    return ss

def _save_csv(df, path):
    _ensure_out()
    df.to_csv(path, index=False)

def _save_jsonl(df, path):
    _ensure_out()
    with open(path, "w", encoding="utf-8") as f:
        for _, r in df.iterrows():
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

def _standardize(x):
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if (not np.isfinite(sd)) or sd < 1e-9:
        return x * np.nan
    return (x - mu) / sd

def _paired_xy(df, source, target, lag):
    x = pd.to_numeric(df[source], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[target], errors="coerce").to_numpy(dtype=float)

    n = len(df)
    lag = int(max(1, lag))
    if n <= lag:
        return np.array([], dtype=float), np.array([], dtype=float)

    x0 = x[: n - lag]
    y1 = y[lag: n]
    m = np.isfinite(x0) & np.isfinite(y1)
    return x0[m], y1[m]

def _paired_xy_future(df, source, target, lag):
    # Placebo: FUTURE X should not explain Y(t)
    x0 = pd.to_numeric(df[source], errors="coerce").to_numpy(dtype=float)
    y0 = pd.to_numeric(df[target], errors="coerce").to_numpy(dtype=float)
    lag = int(lag)
    if lag <= 0 or lag >= len(x0):
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    x_future = x0[lag:]
    y_now = y0[:-lag]
    m = np.isfinite(x_future) & np.isfinite(y_now)
    return x_future[m], y_now[m]

def _block_permute_series(x, block_len, rng):
    x = np.asarray(x, dtype=float)
    n = int(len(x))
    b = int(max(2, block_len))
    if n <= b:
        idx = np.arange(n)
        rng.shuffle(idx)
        return x[idx]
    blocks = []
    for i in range(0, n, b):
        blocks.append(x[i:min(n, i+b)])
    rng.shuffle(blocks)
    return np.concatenate(blocks, axis=0)

def _ensure_adjustment_covariates(df, target_used, mode):
    out = df.copy()
    covs = []
    mode = str(mode).lower()
    if mode in ["light", "full"]:
        out["target_prev"] = pd.to_numeric(out[target_used], errors="coerce").shift(1)
        covs.append("target_prev")
    if mode == "full":
        out["time_idx"] = np.arange(len(out), dtype=float)
        covs.append("time_idx")
        if DATE_COL in out.columns:
            dt = pd.to_datetime(out[DATE_COL], errors="coerce")
            if dt.notna().mean() > 0.2:
                dow = dt.dt.dayofweek.astype(float)
                out["dow_sin"] = np.sin(2.0*np.pi*dow/7.0)
                out["dow_cos"] = np.cos(2.0*np.pi*dow/7.0)
                covs += ["dow_sin", "dow_cos"]
    return out, covs

def _residualize_y(df, y_col, cov_cols):
    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
    if not cov_cols:
        return pd.Series(y, index=df.index)
    X = np.vstack([pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float) for c in cov_cols]).T
    m = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if int(np.sum(m)) < 25:
        return pd.Series(y, index=df.index)
    A = np.column_stack([np.ones(int(np.sum(m))), X[m]])
    beta, *_ = np.linalg.lstsq(A, y[m], rcond=None)
    yhat = np.column_stack([np.ones(len(X)), X]) @ beta
    return pd.Series(y - yhat, index=df.index)

def _delta_high_low(x, y, q_low=Q_LOW, q_high=Q_HIGH):
    if len(x) < 5:
        return np.nan, 0, 0
    lo = np.nanquantile(x, q_low)
    hi = np.nanquantile(x, q_high)
    low = y[x <= lo]
    high = y[x >= hi]
    low = low[np.isfinite(low)]
    high = high[np.isfinite(high)]
    if len(low) < 3 or len(high) < 3:
        return np.nan, int(len(low)), int(len(high))
    return float(np.mean(high) - np.mean(low)), int(len(low)), int(len(high))

def _bootstrap_ci_delta(x, y, b=BOOT_B, alpha=BOOT_ALPHA):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 10:
        return np.nan, np.nan
    rng = np.random.RandomState(12345)
    boots = np.zeros((int(b),), dtype=float)
    for i in range(int(b)):
        idx = rng.choice(np.arange(n), size=n, replace=True)
        d, _, _ = _delta_high_low(x[idx], y[idx], Q_LOW, Q_HIGH)
        boots[i] = d
    lo = 100.0 * (alpha / 2.0)
    hi = 100.0 * (1.0 - alpha / 2.0)
    return float(np.percentile(boots, lo)), float(np.percentile(boots, hi))

def _window_sign_prob(df, source, target, lag):
    x_full, y_full = _paired_xy(df, source, target, lag)
    if len(x_full) < MIN_SUPPORT_N:
        return np.nan, 0

    delta_full, _, _ = _delta_high_low(x_full, y_full, Q_LOW, Q_HIGH)
    if (not np.isfinite(delta_full)) or abs(delta_full) < 1e-12:
        return np.nan, 0

    sign_full = np.sign(delta_full)

    n = len(df)
    aligned_n = n - int(lag)
    if aligned_n < WINDOW_LEN:
        return np.nan, 0

    wins = []
    start = 0
    while start + WINDOW_LEN <= aligned_n:
        end = start + WINDOW_LEN
        xw = pd.to_numeric(df[source].iloc[start:end], errors="coerce").to_numpy(dtype=float)
        yw = pd.to_numeric(df[target].iloc[start + lag: end + lag], errors="coerce").to_numpy(dtype=float)
        m = np.isfinite(xw) & np.isfinite(yw)
        xw = xw[m]
        yw = yw[m]
        if len(xw) >= max(10, int(MIN_SUPPORT_N * 0.4)):
            dw, _, _ = _delta_high_low(xw, yw, Q_LOW, Q_HIGH)
            if np.isfinite(dw) and abs(dw) > 1e-12:
                wins.append(1 if np.sign(dw) == sign_full else 0)
        start += WINDOW_STRIDE

    if len(wins) < MIN_WINDOWS:
        return np.nan, int(len(wins))
    return float(np.mean(wins)), int(len(wins))

def _ci_width_penalty(ci_low, ci_high):
    if not USE_CI_WIDTH_PENALTY:
        return 1.0
    lo = _safe_float(ci_low, np.nan)
    hi = _safe_float(ci_high, np.nan)
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return 1.0
    w = float(hi - lo)
    if (not np.isfinite(w)) or w <= 0:
        return 1.0
    t = float(np.clip(w / float(CI_WIDTH_SOFT_MAX), 0.0, 3.0))
    mult = 1.0 - float(CI_PENALTY_WEIGHT) * float(np.clip((t - 1.0) / 2.0, 0.0, 1.0))
    return float(np.clip(mult, 0.0, 1.0))

def _strength_score(effect_size, p_sign, support_n, ci_low=np.nan, ci_high=np.nan):
    eff = abs(_safe_float(effect_size, np.nan))
    ps = _safe_float(p_sign, np.nan)
    n = _safe_float(support_n, 0.0)

    eff_score = float(np.clip(eff / 0.6, 0.0, 1.0)) if np.isfinite(eff) else 0.0
    ps_score = float(np.clip(ps, 0.0, 1.0)) if np.isfinite(ps) else 0.5
    n_score = float(np.clip(n / 60.0, 0.0, 1.0))

    base = float(W_EFF * eff_score + W_PSIGN * ps_score + W_N * n_score)
    mult = _ci_width_penalty(ci_low, ci_high)
    return float(np.clip(base * mult, 0.0, 1.0))


def _compute_strength_for_outcome(df, source, outcome_col, lag):
    """Compute core edge stats for a given outcome column (used for negative controls and slices)."""
    x, y = _paired_xy(df, source, outcome_col, lag)
    support_n = int(len(x))
    if support_n < int(MIN_SUPPORT_N):
        return {
            "support_n": support_n,
            "delta": np.nan,
            "effect_size": np.nan,
            "p_sign": np.nan,
            "n_windows": 0,
            "strength": np.nan,
        }

    delta, _, _ = _delta_high_low(x, y, Q_LOW, Q_HIGH)
    y_std = _standardize(y)
    delta_std, _, _ = _delta_high_low(x, y_std, Q_LOW, Q_HIGH)
    effect_size = float(delta_std) if np.isfinite(delta_std) else np.nan

    p_sign, n_windows = _window_sign_prob(df, source, outcome_col, lag)
    strength = float(_strength_score(effect_size, p_sign, support_n))

    return {
        "support_n": support_n,
        "delta": float(delta) if np.isfinite(delta) else np.nan,
        "effect_size": float(effect_size) if np.isfinite(effect_size) else np.nan,
        "p_sign": float(p_sign) if np.isfinite(p_sign) else np.nan,
        "n_windows": int(n_windows),
        "strength": float(strength) if np.isfinite(strength) else np.nan,
    }


def _slice_masks(df):
    """Return simple enterprise-friendly slice masks."""
    masks = {}
    n = int(len(df))
    if n <= 0:
        return masks

    # time split
    mid = max(1, n // 2)
    masks["first_half"] = np.arange(n) < mid
    masks["second_half"] = np.arange(n) >= mid

    # weekday/weekend if date is parseable
    if DATE_COL in df.columns:
        dt = pd.to_datetime(df[DATE_COL], errors="coerce")
        if dt.notna().mean() > 0.2:
            dow = dt.dt.dayofweek
            masks["weekday"] = (dow <= 4).to_numpy(dtype=bool)
            masks["weekend"] = (dow >= 5).to_numpy(dtype=bool)

    return masks


def _stability_score(df, source, outcome_col, lag):
    """Compute stability score across slices as the fraction of slices that preserve sign + strength."""
    if not STABILITY_ENABLE:
        return np.nan, {}, 1

    masks = _slice_masks(df)
    # filter to configured slices
    masks = {k: v for k, v in masks.items() if k in set([str(x) for x in STABILITY_SLICES])}
    if not masks:
        return np.nan, {}, 1

    full = _compute_strength_for_outcome(df, source, outcome_col, lag)
    full_delta = float(full.get("delta", np.nan))
    full_strength = float(full.get("strength", np.nan))
    if not np.isfinite(full_delta) or not np.isfinite(full_strength):
        return np.nan, {}, 1

    full_sign = np.sign(full_delta)
    passed = 0
    used = 0
    details = {}
    for name, m in masks.items():
        try:
            sdf = df.iloc[np.where(np.asarray(m, dtype=bool))[0]].copy()
        except Exception:
            continue
        if len(sdf) < int(MIN_SUPPORT_N) + int(lag) + 3:
            continue
        st = _compute_strength_for_outcome(sdf, source, outcome_col, lag)
        details[name] = {
            "strength": float(st.get("strength", np.nan)),
            "delta": float(st.get("delta", np.nan)),
            "support_n": int(st.get("support_n", 0)),
        }
        if not np.isfinite(details[name]["strength"]) or not np.isfinite(details[name]["delta"]):
            continue
        used += 1
        sign_ok = (np.sign(details[name]["delta"]) == full_sign)
        # allow some degradation vs full; still must be meaningfully strong
        strength_ok = (details[name]["strength"] >= max(0.0, 0.85 * full_strength))
        if sign_ok and strength_ok:
            passed += 1

    if used <= 0:
        return np.nan, details, 1

    score = float(passed) / float(used)
    return float(score), details, int(1 if score >= float(STABILITY_MIN_SCORE) else 0)

def _make_statement(source, target, lag, delta, ci_lo, ci_hi):
    src = source.replace("_", " ")
    tgt = target.replace("_", " ")
    d = _safe_float(delta, np.nan)
    lo = _safe_float(ci_lo, np.nan)
    hi = _safe_float(ci_hi, np.nan)

    if np.isfinite(d):
        direction = "higher" if d > 0 else "lower"
        if np.isfinite(lo) and np.isfinite(hi):
            return (
                "When %s is high, %s tends to be %s %d day(s) later (Δ=%.2f, CI %.2f..%.2f)."
                % (src, tgt, direction, int(lag), d, lo, hi)
            )
        return "When %s is high, %s tends to be %s %d day(s) later (Δ=%.2f)." % (src, tgt, direction, int(lag), d)

    return "Potential relationship: %s → %s (lag %d)." % (src, tgt, int(lag))

def _make_reco(source, target, lag, delta):
    src = source.replace("_", " ")
    tgt = target.replace("_", " ")
    d = _safe_float(delta, np.nan)
    if not np.isfinite(d):
        return "Track %s and %s for a few weeks and re-run PCB." % (src, tgt)
    if d > 0:
        return "Consider increasing %s (within safe limits) and observe %s over the next %d day(s)." % (src, tgt, int(lag))
    return "Consider reducing %s (within safe limits) and observe %s over the next %d day(s)." % (src, tgt, int(lag))

def _is_usable_source(series_num):
    x = pd.to_numeric(series_num, errors="coerce").astype(float)
    nan_frac = float(x.isna().mean())
    if nan_frac > float(MAX_NAN_FRAC):
        return False
    x2 = x.dropna()
    if len(x2) < max(MIN_SUPPORT_N, 10):
        return False
    if int(x2.nunique()) < int(MIN_UNIQUE_NUM):
        return False
    if float(np.nanstd(x2.to_numpy(dtype=float))) < float(MIN_STD_EPS):
        return False
    return True

def _corr_safe(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if int(m.sum()) < 15:
        return np.nan
    aa = a[m]
    bb = b[m]
    if float(np.nanstd(aa)) < 1e-9 or float(np.nanstd(bb)) < 1e-9:
        return np.nan
    try:
        return float(np.corrcoef(aa, bb)[0, 1])
    except Exception:
        return np.nan

# -----------------------------
# Guardrails v1.1
# -----------------------------
def _guardrails_drift(df, source, target_used):
    x = pd.to_numeric(df[source], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[target_used], errors="coerce").to_numpy(dtype=float)
    t = np.arange(len(df), dtype=float)

    corr_time_src = _corr_safe(x, t)
    corr_time_tgt = _corr_safe(y, t)

    drift_flag = 0
    if (np.isfinite(corr_time_src) and abs(float(corr_time_src)) >= float(DRIFT_TIME_CORR_SOURCE)
            and np.isfinite(corr_time_tgt) and abs(float(corr_time_tgt)) >= float(DRIFT_TIME_CORR_TARGET)):
        drift_flag = 1

    return {
        "drift_corr_time_source": float(corr_time_src) if np.isfinite(corr_time_src) else np.nan,
        "drift_corr_time_target": float(corr_time_tgt) if np.isfinite(corr_time_tgt) else np.nan,
        "drift_flag": int(drift_flag),
    }

def _guardrails_leakage_for_lag(df, source, target_used, lag):
    """
    Leakage v1.1 per-lag:
      corr_future = corr(source[t], target[t+lag])
      corr_now    = corr(source[t], target[t]) aligned to same length (n-lag)
    Flag if:
      abs(corr_future) >= HARD
      OR (abs(corr_future) >= SOFT AND abs(corr_future)-abs(corr_now) >= GAP)
    """
    x = pd.to_numeric(df[source], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[target_used], errors="coerce").to_numpy(dtype=float)

    n = len(df)
    lag = int(max(1, lag))
    if n <= lag + 2:
        return {"leakage_corr_future": np.nan, "leakage_corr_now": np.nan, "leakage_gap": np.nan, "leakage_flag": 0}

    xf = x[: n - lag]
    yf = y[lag: n]
    corr_future = _corr_safe(xf, yf)

    xn = x[: n - lag]
    yn = y[: n - lag]
    corr_now = _corr_safe(xn, yn)

    gap = np.nan
    if np.isfinite(corr_future) and np.isfinite(corr_now):
        gap = abs(float(corr_future)) - abs(float(corr_now))

    leakage_flag = 0
    if np.isfinite(corr_future) and abs(float(corr_future)) >= float(LEAKAGE_FUTURE_CORR_HARD):
        leakage_flag = 1
    elif (np.isfinite(corr_future) and abs(float(corr_future)) >= float(LEAKAGE_FUTURE_CORR_SOFT)
          and np.isfinite(gap) and float(gap) >= float(LEAKAGE_GAP_MIN)):
        leakage_flag = 1

    return {
        "leakage_corr_future": float(corr_future) if np.isfinite(corr_future) else np.nan,
        "leakage_corr_now": float(corr_now) if np.isfinite(corr_now) else np.nan,
        "leakage_gap": float(gap) if np.isfinite(gap) else np.nan,
        "leakage_flag": int(leakage_flag),
    }

def _guardrail_merge_reason(leakage_flag, drift_flag):
    if int(leakage_flag) == 1:
        return 1, "leakage_future_corr"
    if int(drift_flag) == 1:
        return 1, "drift_time_corr"
    return 0, ""

# -----------------------------
# Main
# -----------------------------
def main(data_csv_path=None, target_col=TARGET_COL, max_lag=MAX_LAG):
    _ensure_out()

    if data_csv_path is None:
        data_csv_path = _load_data_path()
    if not os.path.exists(data_csv_path):
        raise FileNotFoundError("Missing data.csv (or out/demo_data.csv).")

    df = pd.read_csv(data_csv_path)
    df = _try_parse_date(df)

    if target_col not in df.columns:
        raise ValueError("Target column '%s' not found in data CSV." % target_col)

    target_used = _target_col_name(target_col)
    target_for_effect = target_used
    if target_used != target_col:
        df[target_used] = _detrend_target_series(df[target_col])

    cols = [c for c in df.columns if c not in CANDIDATE_EXCLUDE]
    sources = []
    for c in cols:
        v = pd.to_numeric(df[c], errors="coerce")
        if _is_usable_source(v) and (v.notna().sum() >= max(MIN_SUPPORT_N, int(0.3 * len(df)))):
            sources.append(c)

    if len(sources) == 0:
        raise ValueError("No usable source columns found (need numeric columns besides mood/date).")

    rows = []

    for src in sources:
        drift_meta = {"drift_corr_time_source": np.nan, "drift_corr_time_target": np.nan, "drift_flag": 0}
        if ENABLE_GUARDRAILS:
            drift_meta = _guardrails_drift(df, src, target_used)

        for lag in range(1, int(max_lag) + 1):
            x, y = _paired_xy(df, src, target_for_effect, lag)
            support_n = int(len(x))
            if support_n < int(MIN_SUPPORT_N):
                continue

            delta, n_low, n_high = _delta_high_low(x, y, Q_LOW, Q_HIGH)
            if not np.isfinite(delta):
                continue

            y_std = _standardize(y)
            delta_std, _, _ = _delta_high_low(x, y_std, Q_LOW, Q_HIGH)
            effect_size = float(delta_std) if np.isfinite(delta_std) else np.nan

            p_sign, n_windows = _window_sign_prob(df, src, target_for_effect, lag)
            ci_lo, ci_hi = _bootstrap_ci_delta(x, y)
            strength = _strength_score(effect_size, p_sign, support_n, ci_lo, ci_hi)

            # ---- Placebo tests (optional)
            placebo_future_strength = np.nan
            placebo_perm_strength_p90 = np.nan
            placebo_future_pass = 1
            placebo_perm_pass = 1
            if PLACEBO_ENABLE:
                rng = np.random.RandomState(1337)
                if PLACEBO_FUTURE_ENABLE:
                    xf, yf = _paired_xy_future(df, src, target_for_effect, lag)
                    if len(xf) >= int(MIN_SUPPORT_N):
                        yf_std = _standardize(yf)
                        dstd_f, _, _ = _delta_high_low(xf, yf_std, Q_LOW, Q_HIGH)
                        es_f = float(dstd_f) if np.isfinite(dstd_f) else np.nan
                        placebo_future_strength = float(_strength_score(es_f, p_sign, int(len(xf))))
                        if np.isfinite(placebo_future_strength) and np.isfinite(strength):
                            placebo_future_pass = int(float(strength) >= float(placebo_future_strength) + float(PLACEBO_MARGIN))
                if PLACEBO_PERM_ENABLE and int(PLACEBO_PERM_B) > 0:
                    vals = []
                    for _ in range(int(PLACEBO_PERM_B)):
                        xp = _block_permute_series(x, int(PLACEBO_BLOCK_LEN), rng)
                        yps = _standardize(y)
                        dstd_p, _, _ = _delta_high_low(xp, yps, Q_LOW, Q_HIGH)
                        es_p = float(dstd_p) if np.isfinite(dstd_p) else np.nan
                        sp = float(_strength_score(es_p, p_sign, int(len(xp))))
                        if np.isfinite(sp):
                            vals.append(sp)
                    if len(vals) > 5:
                        placebo_perm_strength_p90 = float(np.quantile(np.asarray(vals, dtype=float), 0.90))
                        if np.isfinite(strength):
                            placebo_perm_pass = int(float(strength) >= float(placebo_perm_strength_p90) + float(PLACEBO_MARGIN))

            # ---- Slice stability (enterprise trust)
            stability_score = np.nan
            stability_pass = 1
            if STABILITY_ENABLE:
                stability_score, _slice_details, stability_pass = _stability_score(df, src, target_for_effect, lag)

            # ---- Negative control outcome (enterprise robustness)
            negctrl_strength = np.nan
            negctrl_pass = 1
            if NEGCTRL_ENABLE and (NEGCTRL_OUTCOME_COL in df.columns) and (NEGCTRL_OUTCOME_COL != target_for_effect):
                nc = _compute_strength_for_outcome(df, src, NEGCTRL_OUTCOME_COL, lag)
                negctrl_strength = float(nc.get("strength", np.nan))
                # Fail if negative-control looks strong, or almost as strong as the main outcome.
                if np.isfinite(negctrl_strength) and np.isfinite(strength):
                    if (float(negctrl_strength) >= float(NEGCTRL_MAX_STRENGTH)):
                        negctrl_pass = 0
                    elif (float(strength) - float(negctrl_strength)) < float(NEGCTRL_MARGIN):
                        negctrl_pass = 0

            leak_meta = {"leakage_corr_future": np.nan, "leakage_corr_now": np.nan, "leakage_gap": np.nan, "leakage_flag": 0}
            if ENABLE_GUARDRAILS:
                leak_meta = _guardrails_leakage_for_lag(df, src, target_used, lag)

            gr_flag, gr_reason = _guardrail_merge_reason(leak_meta["leakage_flag"], drift_meta["drift_flag"])

            rows.append({
                "edge_id": "E2-%s-%s-L%d" % (src, target_col, lag),
                "source": src,
                "target": target_col,
                "target_used": target_used,
                "lag": int(lag),

                "support_n": int(support_n),
                "n_low": int(n_low),
                "n_high": int(n_high),

                "delta": float(delta),
                "ci_low": float(ci_lo) if np.isfinite(ci_lo) else np.nan,
                "ci_high": float(ci_hi) if np.isfinite(ci_hi) else np.nan,
                "ci_width": float(ci_hi - ci_lo) if (np.isfinite(ci_lo) and np.isfinite(ci_hi)) else np.nan,

                "effect_size": float(effect_size) if np.isfinite(effect_size) else np.nan,
                "p_sign": float(p_sign) if np.isfinite(p_sign) else np.nan,
                "n_windows": int(n_windows),
                "strength": float(strength),

                # Placebo tests
                "placebo_future_strength": float(placebo_future_strength) if np.isfinite(placebo_future_strength) else np.nan,
                "placebo_perm_strength_p90": float(placebo_perm_strength_p90) if np.isfinite(placebo_perm_strength_p90) else np.nan,
                "placebo_future_pass": int(placebo_future_pass),
                "placebo_perm_pass": int(placebo_perm_pass),
                "placebo_pass": int(1 if (int(placebo_future_pass) == 1 and int(placebo_perm_pass) == 1) else 0),

                # Slice stability
                "stability_score": float(stability_score) if np.isfinite(stability_score) else np.nan,
                "stability_pass": int(stability_pass),

                # Negative control outcome
                "negctrl_strength": float(negctrl_strength) if np.isfinite(negctrl_strength) else np.nan,
                "negctrl_pass": int(negctrl_pass),

                "guardrail_flag": int(gr_flag),
                "guardrail_reason": _as_str(gr_reason),

                "leakage_corr_future": leak_meta.get("leakage_corr_future", np.nan),
                "leakage_corr_now": leak_meta.get("leakage_corr_now", np.nan),
                "leakage_gap": leak_meta.get("leakage_gap", np.nan),

                "drift_corr_time_source": drift_meta.get("drift_corr_time_source", np.nan),
                "drift_corr_time_target": drift_meta.get("drift_corr_time_target", np.nan),
            })

    df_edges = pd.DataFrame(rows)
    if len(df_edges) == 0:
        _save_csv(pd.DataFrame(columns=[]), EDGES_PATH)
        _save_csv(pd.DataFrame(columns=[]), OUT_CSV)
        _save_jsonl(pd.DataFrame(columns=[]), OUT_JSONL)
        print("No edges met MIN_SUPPORT_N. Check data coverage.")
        return

    df_edges = df_edges.sort_values(["strength", "support_n"], ascending=[False, False]).reset_index(drop=True)
    _save_csv(df_edges, EDGES_PATH)

    df_i = df_edges.copy().rename(columns={
        "support_n": "support_n_test",
        "effect_size": "effect_size_test",
        "p_sign": "p_sign_test",
        "delta": "delta_test",
        "ci_low": "ci_low_test",
        "ci_high": "ci_high_test",
    })

    df_i["insight_id"] = ["I2-%05d" % (i + 1) for i in range(len(df_i))]
    df_i["human_statement"] = df_i.apply(
        lambda r: _make_statement(r["source"], r["target"], int(r["lag"]), r["delta_test"], r["ci_low_test"], r["ci_high_test"]),
        axis=1,
    )
    df_i["recommendation"] = df_i.apply(
        lambda r: _make_reco(r["source"], r["target"], int(r["lag"]), r["delta_test"]),
        axis=1,
    )

    df_i["drop_reason"] = ""
    gate = (
        (df_i["support_n_test"].astype(float) >= float(MIN_SUPPORT_N))
        & (df_i["strength"].astype(float) >= float(MIN_STRENGTH))
        & (pd.to_numeric(df_i["p_sign_test"], errors="coerce").fillna(0.5) >= float(MIN_P_SIGN))
        & (pd.to_numeric(df_i["effect_size_test"], errors="coerce").abs().fillna(0.0) >= float(MIN_EFFECT_ABS))
    )

    # Must beat placebo (if enabled)
    if PLACEBO_ENABLE and ("placebo_pass" in df_i.columns):
        pp = pd.to_numeric(df_i["placebo_pass"], errors="coerce").fillna(1).astype(int)
        gate = gate & (pp == 1)

    # Slice stability (if enabled)
    if STABILITY_ENABLE and ("stability_pass" in df_i.columns):
        sp = pd.to_numeric(df_i["stability_pass"], errors="coerce").fillna(1).astype(int)
        gate = gate & (sp == 1)

    # Negative control outcome (if enabled)
    if NEGCTRL_ENABLE and ("negctrl_pass" in df_i.columns):
        npass = pd.to_numeric(df_i["negctrl_pass"], errors="coerce").fillna(1).astype(int)
        gate = gate & (npass == 1)
    df_i.loc[~gate, "drop_reason"] = "failed_gates(placebo/stability/negctrl/min_strength/min_psign/min_effect/min_support)"

    if ENABLE_GUARDRAILS and DROP_FLAGGED_INSIGHTS:
        gr_flag = pd.to_numeric(df_i.get("guardrail_flag", 0), errors="coerce").fillna(0).astype(int)
        bad = gr_flag == 1
        reason = df_i.loc[bad & gate, "guardrail_reason"].fillna("").astype(str)
        df_i.loc[bad & gate, "drop_reason"] = "guardrails(" + reason + ")"
        gate = gate & (~bad)

    df_i["kept"] = gate.astype(int)

    df_out = df_i[df_i["kept"] == 1].copy()
    df_out = df_out.sort_values(["strength", "support_n_test"], ascending=[False, False]).reset_index(drop=True)

    cols_out = [
        "insight_id",
        "source", "target", "lag",
        "strength", "support_n_test",
        "effect_size_test", "p_sign_test",
        "delta_test", "ci_low_test", "ci_high_test",

        # Enterprise trust columns
        "placebo_future_strength", "placebo_perm_strength_p90", "placebo_pass",
        "stability_score", "stability_pass",
        "negctrl_strength", "negctrl_pass",

        "human_statement", "recommendation",

        "guardrail_flag", "guardrail_reason",
        "leakage_corr_future", "leakage_corr_now", "leakage_gap",
        "drift_corr_time_source", "drift_corr_time_target",
    ]
    for c in cols_out:
        if c not in df_out.columns:
            df_out[c] = np.nan
    df_out = df_out[cols_out]

    _save_csv(df_out, OUT_CSV)
    _save_jsonl(df_out, OUT_JSONL)

    print("\n=== PCB LEVEL 2.5 (insights) ===")
    print("Data:", data_csv_path)
    print("Target used:", target_used)
    print("Saved:", EDGES_PATH)
    print("Saved:", OUT_CSV)
    print("Saved:", OUT_JSONL)
    print("Edges:", len(df_edges))
    print("Kept insights:", len(df_out))
    if ENABLE_GUARDRAILS:
        n_flagged = int(pd.to_numeric(df_i.get("guardrail_flag", 0), errors="coerce").fillna(0).astype(int).sum())
        print("Guardrails flagged (edges):", n_flagged)
        print("Guardrails drop policy:", "ON" if DROP_FLAGGED_INSIGHTS else "OFF (audit only)")

    if len(df_out) > 0:
        show = ["insight_id", "source", "lag", "strength", "support_n_test", "effect_size_test", "p_sign_test", "delta_test"]
        print("\nTop insights:")
        print(df_out[show].head(10).to_string(index=False))
    else:
        print("\n(No insights passed gates. Try lowering MIN_STRENGTH / MIN_P_SIGN or increase data.)")

if __name__ == "__main__":
    main()
