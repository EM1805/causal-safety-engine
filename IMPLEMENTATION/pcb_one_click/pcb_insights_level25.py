#!/usr/bin/env python3
# FILE: pcb_causal_discovery.py
# Python 3.8+ compatible
#
# PCB – Causal Discovery Engine v1.0
# Granger Causality (numpy/pandas only, no external stats libs)
#
# ┌─────────────────────────────────────────────────────────────────┐
# │  WHAT THIS DOES                                                 │
# │                                                                 │
# │  For every candidate column X, tests whether X Granger-causes  │
# │  the target Y:                                                  │
# │                                                                 │
# │    H0 (restricted):   Y(t) = f( Y(t-1..p) )                   │
# │    H1 (unrestricted): Y(t) = f( Y(t-1..p), X(t-1..k) )        │
# │                                                                 │
# │  Rejects H0 (X is causal) if:                                  │
# │    - F-test p-value < alpha  (classical Granger)               │
# │    - AND residual reduction ≥ min_rss_reduction                 │
# │    - AND placebo (future X) does NOT Granger-cause Y           │
# │    - AND permutation null distribution is beaten                │
# │    - AND rolling-window stability ≥ min_stability               │
# │    - AND negative-control outcome is NOT caused by X            │
# │    - AND guardrails (lag-0 leakage, time drift) pass            │
# │                                                                 │
# │  Output score: causal_score = f(F_stat, rss_reduction,         │
# │                                  stability, p_value)            │
# └─────────────────────────────────────────────────────────────────┘
#
# Output files:
#   out/causal_edges.csv       — all tested (source, target, lag) triples
#   out/causal_insights.csv    — edges that passed all gates
#   out/causal_insights.jsonl  — same, one JSON object per line
#
# Dependencies: numpy, pandas only
#
# Quick start:
#   python pcb_causal_discovery.py              # uses data.csv
#   python pcb_causal_discovery.py --selftest   # synthetic smoke-test
#

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("pcb.causal")


# ══════════════════════════════════════════════════════════════════
#  Configuration
# ══════════════════════════════════════════════════════════════════

@dataclass
class CausalConfig:
    # ── Paths ─────────────────────────────────────────────────────
    out_dir: str = "out"
    data_csv: str = "data.csv"

    # ── Columns ───────────────────────────────────────────────────
    target_col: str = "target"
    date_col: str = "date"

    # ── Granger VAR parameters ────────────────────────────────────
    max_lag: int = 7        # max lag k for X in unrestricted model
    ar_order: int = 2       # autoregressive order p for Y lags in both models
                            # rule of thumb: ~sqrt(T/5), min 1

    # ── Significance ──────────────────────────────────────────────
    granger_alpha: float = 0.05   # F-test p-value threshold
    min_rss_reduction: float = 0.05  # min relative RSS drop: (RSS_r - RSS_u)/RSS_r

    # ── Detrending ────────────────────────────────────────────────
    detrend_mode: str = "none"    # "none" | "diff1"
                                  # diff1 isolates change-causes-change signal

    # ── Data quality ──────────────────────────────────────────────
    min_obs: int = 40             # minimum observations after lag alignment
    max_nan_frac: float = 0.40
    min_unique: int = 5
    min_std_eps: float = 1e-6

    # ── Causal score weights ──────────────────────────────────────
    # causal_score = w_f*f_score + w_rss*rss_score + w_stab*stab_score
    w_f: float = 0.40
    w_rss: float = 0.35
    w_stab: float = 0.25

    # ── Placebo / permutation ─────────────────────────────────────
    placebo_future_enable: bool = True    # future-X placebo Granger test
    placebo_future_alpha: float = 0.20   # future placebo must NOT be significant
    placebo_perm_enable: bool = True
    placebo_perm_b: int = 30             # permutation iterations
    placebo_perm_seed: int = 1337
    placebo_block_len: int = 7           # block-permutation length

    # ── Rolling stability ─────────────────────────────────────────
    stability_enable: bool = True
    stability_window: int = 30           # rolling window length (obs)
    stability_stride: int = 7
    stability_min_windows: int = 3
    stability_min_score: float = 0.60    # fraction of windows where X is Granger-causal

    # ── Negative control outcome ──────────────────────────────────
    negctrl_enable: bool = True
    negctrl_outcome_col: str = "negative_control_outcome"
    negctrl_alpha: float = 0.10          # negctrl must NOT be significant at this level

    # ── Guardrails ────────────────────────────────────────────────
    enable_guardrails: bool = True
    lag0_corr_hard: float = 0.95         # same-day correlation → likely same variable
    drift_corr_source: float = 0.85      # |corr(X,t)| threshold
    drift_corr_target: float = 0.40      # |corr(Y,t)| threshold
    drop_flagged: bool = True

    # ── Output gates ──────────────────────────────────────────────
    min_causal_score: float = 0.35

    # ── Bootstrap CI on causal effect (mean diff) ─────────────────
    boot_b: int = 200
    boot_seed: int = 42
    boot_alpha: float = 0.10

    # ─────────────────────────────────────────────────────────────
    @classmethod
    def from_dict(cls, d: dict) -> "CausalConfig":
        cfg = cls()
        cfg.out_dir = str(d.get("out_dir", cfg.out_dir))
        cfg.date_col = str(d.get("date_col", cfg.date_col))
        cfg.target_col = str(d.get("target_col") or d.get("target") or cfg.target_col)
        lv = d.get("causal", d.get("level25", {})) if isinstance(d, dict) else {}
        _int_keys = ("max_lag", "ar_order", "min_obs", "min_unique",
                     "placebo_perm_b", "placebo_perm_seed", "placebo_block_len",
                     "stability_window", "stability_stride", "stability_min_windows",
                     "boot_b", "boot_seed")
        _float_keys = ("granger_alpha", "min_rss_reduction", "max_nan_frac", "min_std_eps",
                       "w_f", "w_rss", "w_stab", "placebo_future_alpha",
                       "stability_min_score", "negctrl_alpha", "lag0_corr_hard",
                       "drift_corr_source", "drift_corr_target", "min_causal_score",
                       "boot_alpha")
        _bool_keys = ("placebo_future_enable", "placebo_perm_enable", "stability_enable",
                      "negctrl_enable", "enable_guardrails", "drop_flagged")
        _str_keys = ("detrend_mode", "negctrl_outcome_col")
        for k in _int_keys:
            if k in lv: setattr(cfg, k, int(lv[k]))
        for k in _float_keys:
            if k in lv: setattr(cfg, k, float(lv[k]))
        for k in _bool_keys:
            if k in lv: setattr(cfg, k, bool(lv[k]))
        for k in _str_keys:
            if k in lv: setattr(cfg, k, str(lv[k]))
        return cfg

    @classmethod
    def load(cls) -> "CausalConfig":
        try:
            from pcb_config import load_config  # type: ignore
            return cls.from_dict(load_config())
        except Exception:
            return cls()


# ══════════════════════════════════════════════════════════════════
#  Low-level linear algebra helpers (pure numpy)
# ══════════════════════════════════════════════════════════════════

def _safe_float(x, default: float = np.nan) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else float(default)
    except Exception:
        return float(default)


def _as_str(x) -> str:
    try:
        s = "" if x is None else str(x)
        return "" if s.lower() == "nan" else s
    except Exception:
        return ""


def _corr_safe(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return np.nan
    aa, bb = a[m], b[m]
    if aa.std() < 1e-9 or bb.std() < 1e-9:
        return np.nan
    return float(np.corrcoef(aa, bb)[0, 1])


def _ols_rss(X: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    OLS regression y ~ X (X already includes intercept column).
    Returns (RSS, beta). Raises on ill-conditioned input.
    """
    cond = np.linalg.cond(X)
    if not np.isfinite(cond) or cond > 1e12:
        raise np.linalg.LinAlgError("Ill-conditioned design matrix (cond=%.2e)" % cond)
    beta, *_ = np.linalg.lstsq(X, y, rcond=1e-10)
    resid = y - X @ beta
    return float(resid @ resid), beta


def _f_test(rss_r: float, rss_u: float, df_r: int, df_u: int, n: int) -> Tuple[float, float]:
    """
    F-statistic for nested OLS models.

    F = ((RSS_r - RSS_u) / (df_r - df_u)) / (RSS_u / (n - df_u))

    Returns (F_stat, p_value) using F-distribution approximation via
    incomplete beta function (pure numpy — no scipy).
    df_r = number of params in restricted model
    df_u = number of params in unrestricted model
    """
    if rss_u <= 0 or df_u >= n or df_r >= df_u:
        return np.nan, np.nan
    num_df = df_u - df_r
    den_df = n - df_u
    if num_df <= 0 or den_df <= 0:
        return np.nan, np.nan
    f_stat = ((rss_r - rss_u) / num_df) / (rss_u / den_df)
    if not np.isfinite(f_stat) or f_stat < 0:
        return float(max(0.0, f_stat)) if np.isfinite(f_stat) else np.nan, np.nan
    p_value = _f_pvalue(float(f_stat), num_df, den_df)
    return float(f_stat), float(p_value)


def _f_pvalue(f: float, d1: int, d2: int) -> float:
    """
    P(F(d1,d2) > f) via regularised incomplete beta function.
    Implemented with the continued fraction expansion (Numerical Recipes).
    Accurate to ~1e-7 for typical F-stat ranges.
    """
    if f <= 0:
        return 1.0
    x = d2 / (d2 + d1 * f)
    # P(F > f) = I_x(d2/2, d1/2)  where I is regularised incomplete beta
    return _regularised_inc_beta(x, d2 / 2.0, d1 / 2.0)


def _log_beta(a: float, b: float) -> float:
    """log B(a,b) via Stirling / lgamma recurrence."""
    return _lgamma(a) + _lgamma(b) - _lgamma(a + b)


def _lgamma(x: float) -> float:
    """Lanczos approximation of log-gamma, accurate to ~1e-12."""
    g = 7
    c = [
        0.99999999999980993, 676.5203681218851, -1259.1392167224028,
        771.32342877765313, -176.61502916214059, 12.507343278686905,
        -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
    ]
    if x < 0.5:
        return np.log(np.pi / np.sin(np.pi * x)) - _lgamma(1.0 - x)
    x -= 1
    t = x + g + 0.5
    s = sum(c[i] / (x + i) for i in range(1, g + 2)) + c[0]
    return 0.5 * np.log(2 * np.pi) + (x + 0.5) * np.log(t) - t + np.log(s)


def _regularised_inc_beta(x: float, a: float, b: float) -> float:
    """
    I_x(a, b): regularised incomplete beta via continued fractions (Lentz).
    Returns value in [0, 1].
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    # Use the symmetry relation if x > (a+1)/(a+b+2)
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _regularised_inc_beta(1.0 - x, b, a)
    lbeta_ab = _log_beta(a, b)
    front = np.exp(np.log(x) * a + np.log(1.0 - x) * b - lbeta_ab) / a
    return front * _betacf(x, a, b)


def _betacf(x: float, a: float, b: float, max_iter: int = 200, eps: float = 3e-7) -> float:
    """Continued fraction for incomplete beta (Lentz's method)."""
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


# ══════════════════════════════════════════════════════════════════
#  VAR design-matrix builder
# ══════════════════════════════════════════════════════════════════

def _build_var_matrices(
    y: np.ndarray,
    x: Optional[np.ndarray],
    ar_order: int,
    x_lag: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], int]:
    """
    Build OLS design matrices for restricted (AR only) and unrestricted (AR + X lags) models.

    Restricted:   Y_t = a0 + sum_{i=1}^{ar_order} a_i * Y_{t-i}  + e_t
    Unrestricted: Y_t = a0 + sum_{i=1}^{ar_order} a_i * Y_{t-i}
                           + sum_{j=1}^{x_lag}    b_j * X_{t-j}  + e_t

    Returns (X_r, X_u, y_dep, mask, n_eff) or (None,...) on failure.
    n_eff = effective number of observations.
    """
    p = int(ar_order)
    k = int(x_lag)
    burn = max(p, k)
    n = len(y)
    if n - burn < 10:
        return None, None, None, None, 0

    T = n - burn   # effective obs

    # dependent variable: Y[burn:]
    y_dep = y[burn:].copy()

    # AR lags of Y
    ar_cols = [np.ones(T)]
    for i in range(1, p + 1):
        ar_cols.append(y[burn - i: n - i])

    X_r = np.column_stack(ar_cols)   # shape (T, 1+p)

    if x is None:
        return X_r, None, y_dep, None, T

    # X lags
    x_cols = []
    for j in range(1, k + 1):
        x_cols.append(x[burn - j: n - j])
    X_x = np.column_stack(x_cols)   # shape (T, k)

    X_u = np.column_stack([X_r, X_x])  # shape (T, 1+p+k)

    # Valid rows: no NaN in any column
    mask = (
        np.isfinite(y_dep)
        & np.all(np.isfinite(X_r), axis=1)
        & np.all(np.isfinite(X_u), axis=1)
    )
    n_eff = int(mask.sum())
    if n_eff < 10:
        return None, None, None, None, 0

    return X_r[mask], X_u[mask], y_dep[mask], mask, n_eff


# ══════════════════════════════════════════════════════════════════
#  Core Granger test (single lag)
# ══════════════════════════════════════════════════════════════════

def _granger_test(
    y: np.ndarray, x: np.ndarray, ar_order: int, x_lag: int
) -> dict:
    """
    Granger causality test: does X(t-1..x_lag) help predict Y beyond AR(ar_order)?

    Returns dict with:
        f_stat, p_value, rss_restricted, rss_unrestricted, rss_reduction,
        n_eff, df_restricted, df_unrestricted
    """
    empty = {
        "f_stat": np.nan, "p_value": np.nan,
        "rss_restricted": np.nan, "rss_unrestricted": np.nan,
        "rss_reduction": np.nan, "n_eff": 0,
        "df_restricted": 0, "df_unrestricted": 0,
    }
    X_r, X_u, y_dep, _, n_eff = _build_var_matrices(y, x, ar_order, x_lag)
    if X_r is None or n_eff == 0:
        return empty

    try:
        rss_r, _ = _ols_rss(X_r, y_dep)
        rss_u, _ = _ols_rss(X_u, y_dep)
    except np.linalg.LinAlgError as e:
        logger.debug("OLS failed: %s", e)
        return empty

    if rss_r <= 0:
        return empty

    rss_red = (rss_r - rss_u) / rss_r  # relative reduction

    df_r = X_r.shape[1]
    df_u = X_u.shape[1]
    f_stat, p_value = _f_test(rss_r, rss_u, df_r, df_u, n_eff)

    return {
        "f_stat": float(f_stat) if np.isfinite(f_stat) else np.nan,
        "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
        "rss_restricted": float(rss_r),
        "rss_unrestricted": float(rss_u),
        "rss_reduction": float(rss_red),
        "n_eff": n_eff,
        "df_restricted": df_r,
        "df_unrestricted": df_u,
    }


# ══════════════════════════════════════════════════════════════════
#  Bootstrap CI on mean causal effect (delta high-low)
# ══════════════════════════════════════════════════════════════════

def _delta_high_low(
    x: np.ndarray, y: np.ndarray, q_low: float = 0.30, q_high: float = 0.70
) -> Tuple[float, int, int]:
    if len(x) < 5:
        return np.nan, 0, 0
    lo = np.nanquantile(x, q_low)
    hi = np.nanquantile(x, q_high)
    ly = y[x <= lo]; hy = y[x >= hi]
    ly = ly[np.isfinite(ly)]; hy = hy[np.isfinite(hy)]
    if len(ly) < 3 or len(hy) < 3:
        return np.nan, int(len(ly)), int(len(hy))
    return float(np.mean(hy) - np.mean(ly)), int(len(ly)), int(len(hy))


def _bootstrap_ci(
    x: np.ndarray, y: np.ndarray, b: int, seed: int, alpha: float
) -> Tuple[float, float]:
    n = len(x)
    if n < 10:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    boots = np.empty(b, dtype=float)
    for i in range(b):
        idx = rng.integers(0, n, size=n)
        d, _, _ = _delta_high_low(x[idx], y[idx])
        boots[i] = d if np.isfinite(d) else np.nan
    v = boots[np.isfinite(boots)]
    if len(v) < 10:
        return np.nan, np.nan
    return float(np.percentile(v, 100 * alpha / 2)), float(np.percentile(v, 100 * (1 - alpha / 2)))


# ══════════════════════════════════════════════════════════════════
#  Causal Discovery Engine
# ══════════════════════════════════════════════════════════════════

class CausalDiscoveryEngine:
    """
    Granger-based causal discovery.
    Safe to import, instantiate multiple times, and test independently.
    """

    def __init__(self, cfg: Optional[CausalConfig] = None):
        self.cfg = cfg or CausalConfig.load()
        # Dynamic: computed from loaded config (not module-load-time globals)
        self._exclude: set = {self.cfg.target_col, self.cfg.date_col}

    # ── Data preparation ──────────────────────────────────────────

    def _load_df(self, path: Optional[str]) -> Tuple[pd.DataFrame, str]:
        cfg = self.cfg
        if path is None:
            fallback = os.path.join(cfg.out_dir, "demo_data.csv")
            path = cfg.data_csv if os.path.exists(cfg.data_csv) else fallback
        if not os.path.exists(path):
            raise FileNotFoundError("Data file not found: %s" % path)
        df = pd.read_csv(path)
        if cfg.date_col in df.columns:
            dt = pd.to_datetime(df[cfg.date_col], errors="coerce")
            if dt.notna().sum() >= max(5, int(0.2 * len(df))):
                df = df.copy()
                df[cfg.date_col] = dt
        return df, path

    def _target_col_name(self) -> str:
        return (self.cfg.target_col + "_diff1"
                if self.cfg.detrend_mode == "diff1"
                else self.cfg.target_col)

    def _apply_detrend(self, df: pd.DataFrame) -> pd.DataFrame:
        tc = self.cfg.target_col
        used = self._target_col_name()
        if used != tc:
            s = pd.to_numeric(df[tc], errors="coerce").astype(float)
            df = df.copy()
            df[used] = s - s.shift(1)
            logger.info("Detrending: %s → %s (diff1)", tc, used)
        return df

    def _usable_sources(self, df: pd.DataFrame) -> List[str]:
        cfg = self.cfg
        out = []
        for c in df.columns:
            if c in self._exclude:
                continue
            x = pd.to_numeric(df[c], errors="coerce").astype(float)
            if x.isna().mean() > cfg.max_nan_frac:
                continue
            x2 = x.dropna()
            if len(x2) < max(cfg.min_obs, 10):
                continue
            if int(x2.nunique()) < cfg.min_unique:
                continue
            if float(np.nanstd(x2.to_numpy())) < cfg.min_std_eps:
                continue
            out.append(c)
        return out

    def _to_arr(self, df: pd.DataFrame, col: str) -> np.ndarray:
        return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)

    # ── Causal score ─────────────────────────────────────────────

    def _causal_score(
        self, f_stat: float, rss_reduction: float, stability: float
    ) -> float:
        """
        Composite score in [0, 1]:
          f_score    = clip(F / F_ref, 0, 1)   F_ref ~ 10 (strong effect)
          rss_score  = clip(rss_reduction / 0.30, 0, 1)
          stab_score = stability fraction
        """
        cfg = self.cfg
        f_s = float(np.clip(_safe_float(f_stat, 0.0) / 10.0, 0.0, 1.0))
        r_s = float(np.clip(_safe_float(rss_reduction, 0.0) / 0.30, 0.0, 1.0))
        st_s = float(np.clip(_safe_float(stability, 0.5), 0.0, 1.0))
        return float(np.clip(cfg.w_f * f_s + cfg.w_rss * r_s + cfg.w_stab * st_s, 0.0, 1.0))

    # ── Rolling stability ─────────────────────────────────────────

    def _rolling_granger_stability(
        self, y_arr: np.ndarray, x_arr: np.ndarray, lag: int
    ) -> Tuple[float, int]:
        """
        Fraction of rolling windows where Granger test is significant.
        Returns (stability_score, n_windows_used).
        """
        cfg = self.cfg
        wlen = cfg.stability_window
        stride = cfg.stability_stride
        n = len(y_arr)
        if n < wlen + lag:
            return np.nan, 0

        results: List[int] = []
        for start in range(0, n - wlen + 1, stride):
            yw = y_arr[start: start + wlen]
            xw = x_arr[start: start + wlen]
            try:
                res = _granger_test(yw, xw, cfg.ar_order, lag)
                pv = res["p_value"]
                rd = res["rss_reduction"]
                if np.isfinite(pv) and np.isfinite(rd):
                    sig = int(pv < cfg.granger_alpha and rd >= cfg.min_rss_reduction)
                    results.append(sig)
            except Exception:
                pass

        if len(results) < cfg.stability_min_windows:
            return np.nan, len(results)
        return float(np.mean(results)), len(results)

    # ── Guardrails ────────────────────────────────────────────────

    def _guardrail_lag0(self, x_arr: np.ndarray, y_arr: np.ndarray) -> dict:
        """Flag if X and Y are near-perfectly correlated contemporaneously."""
        c = _corr_safe(x_arr, y_arr)
        flag = int(np.isfinite(c) and abs(float(c)) >= self.cfg.lag0_corr_hard)
        return {"lag0_corr": float(c) if np.isfinite(c) else np.nan, "lag0_flag": flag}

    def _guardrail_drift(
        self, x_arr: np.ndarray, y_arr: np.ndarray, n: int
    ) -> dict:
        t = np.arange(n, dtype=float)
        cst = _corr_safe(x_arr, t)
        ctt = _corr_safe(y_arr, t)
        cfg = self.cfg
        flag = int(
            np.isfinite(cst) and abs(float(cst)) >= cfg.drift_corr_source
            and np.isfinite(ctt) and abs(float(ctt)) >= cfg.drift_corr_target
        )
        return {
            "drift_corr_source": float(cst) if np.isfinite(cst) else np.nan,
            "drift_corr_target": float(ctt) if np.isfinite(ctt) else np.nan,
            "drift_flag": flag,
        }

    # ── Placebo tests ─────────────────────────────────────────────

    def _placebo_future(
        self, y_arr: np.ndarray, x_arr: np.ndarray, lag: int
    ) -> dict:
        """
        Future-X placebo: X(t+lag) should NOT Granger-cause Y(t).
        If it does, the 'causal' signal may be spurious (Y predicts X, not vice versa).
        """
        cfg = self.cfg
        n = len(y_arr)
        if lag >= n:
            return {"placebo_future_p": np.nan, "placebo_future_pass": 1}
        # align: X_future[t] = X[t+lag], Y[t] = Y[t]
        x_f = x_arr[lag:]
        y_f = y_arr[: n - lag]
        if len(y_f) < cfg.min_obs:
            return {"placebo_future_p": np.nan, "placebo_future_pass": 1}
        res = _granger_test(y_f, x_f, cfg.ar_order, lag)
        pv = res["p_value"]
        # Pass = future X does NOT significantly predict Y
        pass_ = int(not (np.isfinite(pv) and pv < cfg.placebo_future_alpha))
        return {"placebo_future_p": float(pv) if np.isfinite(pv) else np.nan,
                "placebo_future_pass": pass_}

    def _placebo_permutation(
        self, y_arr: np.ndarray, x_arr: np.ndarray, lag: int
    ) -> dict:
        """
        Block-permutation null: true F-stat must exceed 90th percentile of null F-stats.
        """
        cfg = self.cfg
        rng = np.random.default_rng(cfg.placebo_perm_seed)
        true_res = _granger_test(y_arr, x_arr, cfg.ar_order, lag)
        true_f = true_res["f_stat"]

        null_fs: List[float] = []
        for _ in range(cfg.placebo_perm_b):
            xp = _block_permute(x_arr, cfg.placebo_block_len, rng)
            rp = _granger_test(y_arr, xp, cfg.ar_order, lag)
            fp = rp["f_stat"]
            if np.isfinite(fp):
                null_fs.append(fp)

        if len(null_fs) < 5:
            return {"perm_f_p90": np.nan, "placebo_perm_pass": 1}

        p90 = float(np.quantile(null_fs, 0.90))
        pass_ = int(np.isfinite(true_f) and float(true_f) > p90)
        return {"perm_f_p90": p90, "placebo_perm_pass": pass_}

    # ── Negative control ──────────────────────────────────────────

    def _negctrl_test(
        self, df: pd.DataFrame, x_arr: np.ndarray, lag: int
    ) -> dict:
        """
        X should NOT Granger-cause the negative-control outcome.
        If it does, X is likely a proxy for a confound.
        """
        cfg = self.cfg
        if not cfg.negctrl_enable or cfg.negctrl_outcome_col not in df.columns:
            return {"negctrl_p": np.nan, "negctrl_pass": 1}
        nc_arr = self._to_arr(df, cfg.negctrl_outcome_col)
        res = _granger_test(nc_arr, x_arr, cfg.ar_order, lag)
        pv = res["p_value"]
        pass_ = int(not (np.isfinite(pv) and pv < cfg.negctrl_alpha))
        return {"negctrl_p": float(pv) if np.isfinite(pv) else np.nan,
                "negctrl_pass": pass_}

    # ── Human output ─────────────────────────────────────────────

    @staticmethod
    def _make_statement(
        source: str, target: str, lag: int,
        delta: float, ci_lo: float, ci_hi: float,
        p_value: float, f_stat: float,
    ) -> str:
        src = source.replace("_", " ")
        tgt = target.replace("_", " ")
        d = _safe_float(delta, np.nan)
        pv = _safe_float(p_value, np.nan)
        parts = []
        if np.isfinite(d):
            direction = "higher" if d > 0 else "lower"
            lo = _safe_float(ci_lo, np.nan)
            hi = _safe_float(ci_hi, np.nan)
            ci_str = " (CI %.2f..%.2f)" % (lo, hi) if (np.isfinite(lo) and np.isfinite(hi)) else ""
            parts.append(
                "Granger evidence: %s causes %s to be %s %d day(s) later "
                "(Δ=%.2f%s)." % (src, tgt, direction, lag, d, ci_str)
            )
        else:
            parts.append("Granger evidence: %s → %s at lag %d." % (src, tgt, lag))
        if np.isfinite(pv):
            parts.append("F-test p=%.4f." % pv)
        return " ".join(parts)

    @staticmethod
    def _make_reco(source: str, target: str, lag: int, delta: float) -> str:
        src = source.replace("_", " ")
        tgt = target.replace("_", " ")
        d = _safe_float(delta, np.nan)
        if not np.isfinite(d):
            return "Monitor %s and %s together and re-run causal discovery with more data." % (src, tgt)
        verb = "increasing" if d > 0 else "reducing"
        return (
            "Causal signal detected: consider %s %s (within safe limits) "
            "and track %s over the next %d day(s)." % (verb, src, tgt, lag)
        )

    # ── I/O ───────────────────────────────────────────────────────

    def _ensure_out(self):
        os.makedirs(self.cfg.out_dir, exist_ok=True)

    def _save_csv(self, df: pd.DataFrame, path: str):
        self._ensure_out()
        df.to_csv(path, index=False)

    def _save_jsonl(self, df: pd.DataFrame, path: str):
        self._ensure_out()
        with open(path, "w", encoding="utf-8") as fh:
            for _, r in df.iterrows():
                fh.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

    # ── Main pipeline ─────────────────────────────────────────────

    def run(self, data_csv_path: Optional[str] = None) -> pd.DataFrame:
        cfg = self.cfg
        out_dir = cfg.out_dir
        edges_path = os.path.join(out_dir, "edges.csv")
        insights_csv = os.path.join(out_dir, "insights_level2.csv")
        insights_jsonl = os.path.join(out_dir, "insights_level2.jsonl")

        df, data_path_used = self._load_df(data_csv_path)

        if cfg.target_col not in df.columns:
            raise ValueError("Target column '%s' not found in data." % cfg.target_col)

        df = self._apply_detrend(df)
        target_used = self._target_col_name()

        sources = self._usable_sources(df)
        if not sources:
            raise ValueError("No usable source columns found.")

        y_arr = self._to_arr(df, target_used)
        logger.info(
            "Causal discovery | sources=%d | lags=1..%d | AR order=%d | target=%s",
            len(sources), cfg.max_lag, cfg.ar_order, target_used,
        )

        rows: List[dict] = []

        for src in sources:
            x_arr = self._to_arr(df, src)

            # Per-source guardrails (lag-0 and drift)
            lag0_meta = self._guardrail_lag0(x_arr, y_arr) if cfg.enable_guardrails else {"lag0_corr": np.nan, "lag0_flag": 0}
            drift_meta = self._guardrail_drift(x_arr, y_arr, len(df)) if cfg.enable_guardrails else {"drift_corr_source": np.nan, "drift_corr_target": np.nan, "drift_flag": 0}

            gr_flag_src = int(lag0_meta["lag0_flag"] or drift_meta["drift_flag"])
            gr_reason_src = ""
            if lag0_meta["lag0_flag"]:
                gr_reason_src = "lag0_leakage"
            elif drift_meta["drift_flag"]:
                gr_reason_src = "drift"

            for lag in range(1, int(cfg.max_lag) + 1):

                # ── Core Granger test ──────────────────────────────
                res = _granger_test(y_arr, x_arr, cfg.ar_order, lag)
                f_stat = res["f_stat"]
                p_value = res["p_value"]
                rss_r = res["rss_restricted"]
                rss_u = res["rss_unrestricted"]
                rss_red = res["rss_reduction"]
                n_eff = res["n_eff"]

                if n_eff < cfg.min_obs:
                    continue
                if not np.isfinite(f_stat):
                    continue

                # ── Rolling stability ──────────────────────────────
                stability, n_stab_windows = np.nan, 0
                stab_pass = 1
                if cfg.stability_enable:
                    stability, n_stab_windows = self._rolling_granger_stability(y_arr, x_arr, lag)
                    if np.isfinite(stability):
                        stab_pass = int(stability >= cfg.stability_min_score)

                # ── Causal score ───────────────────────────────────
                causal_score = self._causal_score(f_stat, rss_red, stability)

                # ── Bootstrap CI on causal effect magnitude ────────
                # Align x[:-lag] → y[lag:] for delta computation
                n = len(y_arr)
                x_al = x_arr[: n - lag]
                y_al = y_arr[lag:]
                m = np.isfinite(x_al) & np.isfinite(y_al)
                x_c, y_c = x_al[m], y_al[m]
                delta, _, _ = _delta_high_low(x_c, y_c)
                ci_lo, ci_hi = _bootstrap_ci(x_c, y_c, cfg.boot_b, cfg.boot_seed, cfg.boot_alpha)

                # ── Placebo: future X ──────────────────────────────
                pf_meta = {"placebo_future_p": np.nan, "placebo_future_pass": 1}
                if cfg.placebo_future_enable:
                    pf_meta = self._placebo_future(y_arr, x_arr, lag)

                # ── Placebo: permutation ───────────────────────────
                pp_meta = {"perm_f_p90": np.nan, "placebo_perm_pass": 1}
                if cfg.placebo_perm_enable:
                    pp_meta = self._placebo_permutation(y_arr, x_arr, lag)

                placebo_pass = int(
                    pf_meta["placebo_future_pass"] == 1
                    and pp_meta["placebo_perm_pass"] == 1
                )

                # ── Negative control outcome ───────────────────────
                nc_meta = self._negctrl_test(df, x_arr, lag)

                rows.append({
                    "edge_id": "CG-%s-%s-L%d" % (src, cfg.target_col, lag),
                    "source": src,
                    "target": cfg.target_col,
                    "target_used": target_used,
                    "lag": int(lag),
                    "n_eff": int(n_eff),

                    # ── Granger statistics
                    "f_stat": float(f_stat),
                    "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
                    "rss_restricted": float(rss_r) if np.isfinite(rss_r) else np.nan,
                    "rss_unrestricted": float(rss_u) if np.isfinite(rss_u) else np.nan,
                    "rss_reduction": float(rss_red) if np.isfinite(rss_red) else np.nan,

                    # ── Effect magnitude
                    "delta": float(delta) if np.isfinite(delta) else np.nan,
                    "ci_low": float(ci_lo) if np.isfinite(ci_lo) else np.nan,
                    "ci_high": float(ci_hi) if np.isfinite(ci_hi) else np.nan,

                    # ── Composite score
                    "causal_score": float(causal_score),

                    # ── Stability
                    "stability_score": float(stability) if np.isfinite(stability) else np.nan,
                    "stability_n_windows": int(n_stab_windows),
                    "stability_pass": int(stab_pass),

                    # ── Placebo
                    "placebo_future_p": pf_meta["placebo_future_p"],
                    "placebo_future_pass": int(pf_meta["placebo_future_pass"]),
                    "perm_f_p90": pp_meta["perm_f_p90"],
                    "placebo_perm_pass": int(pp_meta["placebo_perm_pass"]),
                    "placebo_pass": int(placebo_pass),

                    # ── Negative control
                    "negctrl_p": nc_meta["negctrl_p"],
                    "negctrl_pass": int(nc_meta["negctrl_pass"]),

                    # ── Guardrails
                    "guardrail_flag": int(gr_flag_src),
                    "guardrail_reason": _as_str(gr_reason_src),
                    "lag0_corr": lag0_meta.get("lag0_corr", np.nan),
                    "drift_corr_source": drift_meta.get("drift_corr_source", np.nan),
                    "drift_corr_target": drift_meta.get("drift_corr_target", np.nan),
                })

        # ── Assemble edges dataframe ───────────────────────────────
        df_edges = pd.DataFrame(rows)
        if len(df_edges) == 0:
            for p in [edges_path, insights_csv]:
                self._save_csv(pd.DataFrame(), p)
            self._save_jsonl(pd.DataFrame(), insights_jsonl)
            logger.warning("No edges computed. Check min_obs=%d and data length.", cfg.min_obs)
            print("No causal edges computed. Check data length / min_obs setting.")
            return pd.DataFrame()

        df_edges = df_edges.sort_values(
            ["causal_score", "n_eff"], ascending=[False, False]
        ).reset_index(drop=True)
        self._save_csv(df_edges, edges_path)

        # ── Gate filtering with dynamic drop_reason ────────────────
        df_i = df_edges.copy()
        df_i["insight_id"] = ["CG-%05d" % (i + 1) for i in range(len(df_i))]

        gate = pd.Series([True] * len(df_i), index=df_i.index)
        drop_reasons: Dict[int, List[str]] = {i: [] for i in df_i.index}

        def _apply_gate(mask: pd.Series, label: str):
            nonlocal gate
            for idx in df_i[gate & ~mask].index:
                drop_reasons[idx].append(label)
            gate = gate & mask

        _apply_gate(df_i["n_eff"].astype(int) >= cfg.min_obs, "min_obs")
        _apply_gate(
            pd.to_numeric(df_i["p_value"], errors="coerce").fillna(1.0) < cfg.granger_alpha,
            "granger_p"
        )
        _apply_gate(
            pd.to_numeric(df_i["rss_reduction"], errors="coerce").fillna(0.0) >= cfg.min_rss_reduction,
            "rss_reduction"
        )
        _apply_gate(
            pd.to_numeric(df_i["causal_score"], errors="coerce").fillna(0.0) >= cfg.min_causal_score,
            "causal_score"
        )
        if cfg.placebo_future_enable:
            _apply_gate(
                pd.to_numeric(df_i["placebo_future_pass"], errors="coerce").fillna(1).astype(int) == 1,
                "placebo_future"
            )
        if cfg.placebo_perm_enable:
            _apply_gate(
                pd.to_numeric(df_i["placebo_perm_pass"], errors="coerce").fillna(1).astype(int) == 1,
                "placebo_perm"
            )
        if cfg.stability_enable:
            _apply_gate(
                pd.to_numeric(df_i["stability_pass"], errors="coerce").fillna(1).astype(int) == 1,
                "stability"
            )
        if cfg.negctrl_enable:
            _apply_gate(
                pd.to_numeric(df_i["negctrl_pass"], errors="coerce").fillna(1).astype(int) == 1,
                "negctrl"
            )

        df_i["drop_reason"] = [
            "+".join(drop_reasons[i]) if drop_reasons[i] else "" for i in df_i.index
        ]

        if cfg.enable_guardrails and cfg.drop_flagged:
            bad = pd.to_numeric(df_i.get("guardrail_flag", 0), errors="coerce").fillna(0).astype(int) == 1
            for idx in df_i[bad & gate].index:
                df_i.loc[idx, "drop_reason"] = "guardrail(%s)" % _as_str(df_i.loc[idx, "guardrail_reason"])
            gate = gate & ~bad

        df_i["kept"] = gate.astype(int)

        # ── Build human columns only for kept insights ─────────────
        df_out = df_i[df_i["kept"] == 1].copy().reset_index(drop=True)
        df_out["human_statement"] = df_out.apply(
            lambda r: self._make_statement(
                r["source"], r["target"], int(r["lag"]),
                r["delta"], r["ci_low"], r["ci_high"],
                r["p_value"], r["f_stat"],
            ), axis=1,
        )
        df_out["recommendation"] = df_out.apply(
            lambda r: self._make_reco(r["source"], r["target"], int(r["lag"]), r["delta"]),
            axis=1,
        )
        df_out = df_out.sort_values(["causal_score", "n_eff"], ascending=[False, False]).reset_index(drop=True)

        cols_out = [
            "insight_id", "source", "target", "lag", "n_eff",
            "f_stat", "p_value", "rss_reduction",
            "delta", "ci_low", "ci_high",
            "causal_score",
            "stability_score", "stability_n_windows",
            "placebo_future_p", "placebo_future_pass",
            "perm_f_p90", "placebo_perm_pass",
            "negctrl_p", "negctrl_pass",
            "guardrail_flag", "guardrail_reason",
            "lag0_corr", "drift_corr_source", "drift_corr_target",
            "human_statement", "recommendation",
        ]
        for c in cols_out:
            if c not in df_out.columns:
                df_out[c] = np.nan
        df_out = df_out[cols_out]

        self._save_csv(df_out, insights_csv)
        self._save_jsonl(df_out, insights_jsonl)

        # ── Console summary ────────────────────────────────────────
        n_flagged = int(
            pd.to_numeric(df_i.get("guardrail_flag", 0), errors="coerce")
            .fillna(0).astype(int).sum()
        )
        print("\n╔══ PCB CAUSAL DISCOVERY v1.0 ══════════════════════════╗")
        print("║  Data     :", data_path_used)
        print("║  Target   :", target_used, "(detrend=%s)" % cfg.detrend_mode)
        print("║  AR order :", cfg.ar_order, "  Max lag:", cfg.max_lag)
        print("║  Sources  :", len(sources))
        print("║  Edges    :", len(df_edges))
        print("║  Insights :", len(df_out))
        print("║  Guardrail flags:", n_flagged,
              " (drop=%s)" % ("ON" if cfg.drop_flagged else "OFF"))
        print("╚═══════════════════════════════════════════════════════╝")

        if len(df_out) > 0:
            show = ["insight_id", "source", "lag", "f_stat", "p_value",
                    "rss_reduction", "causal_score", "stability_score"]
            print("\nTop causal insights:")
            print(df_out[show].head(10).to_string(index=False))
        else:
            print("\n⚠  No causal insights passed all gates.")
            print("   Most common drop reasons:")
            print(df_i["drop_reason"].replace("", np.nan).dropna().value_counts().head(5).to_string())
            print("   Try: lower granger_alpha, min_rss_reduction, or collect more data.")

        logger.info(
            "Done. edges=%d kept=%d flagged=%d", len(df_edges), len(df_out), n_flagged
        )
        return df_out


# ══════════════════════════════════════════════════════════════════
#  Block permutation (used by placebo)
# ══════════════════════════════════════════════════════════════════

def _block_permute(x: np.ndarray, block_len: int, rng: np.random.Generator) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = len(x)
    b = int(max(2, block_len))
    if n <= b:
        idx = np.arange(n)
        rng.shuffle(idx)
        return x[idx]
    blocks = [x[i: min(n, i + b)] for i in range(0, n, b)]
    rng.shuffle(blocks)
    return np.concatenate(blocks)


# ══════════════════════════════════════════════════════════════════
#  Smoke-test
# ══════════════════════════════════════════════════════════════════

def _run_selftest():
    """
    Generates synthetic time-series with a known causal structure:
      - signal_a causes target at lag 3  (should be discovered)
      - noise_b is independent            (should be rejected)
    """
    import tempfile
    print("=" * 60)
    print("PCB Causal Discovery — self-test")
    print("=" * 60)
    rng = np.random.default_rng(99)
    n = 120

    # True causal signal: target(t) = 0.7 * signal_a(t-3) + noise
    signal_a = rng.normal(0, 1, n)
    target = np.zeros(n)
    for t in range(3, n):
        target[t] = 0.7 * signal_a[t - 3] + rng.normal(0, 0.4)

    # Independent noise column: should NOT be discovered
    noise_b = rng.normal(0, 1, n)

    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n),
        "target": target,
        "signal_a": signal_a,
        "noise_b": noise_b,
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = os.path.join(tmpdir, "data.csv")
        df.to_csv(data_path, index=False)

        cfg = CausalConfig()
        cfg.out_dir = os.path.join(tmpdir, "out")
        cfg.min_obs = 20
        cfg.stability_min_windows = 2
        cfg.placebo_perm_b = 10

        engine = CausalDiscoveryEngine(cfg)
        df_out = engine.run(data_path)

    found_a = "signal_a" in df_out["source"].values if len(df_out) > 0 else False
    found_noise = "noise_b" in df_out["source"].values if len(df_out) > 0 else False

    print("\n── Self-test results ──────────────────────────────────")
    print("Expected: signal_a discovered ✓ | noise_b rejected ✓")
    print("Got     : signal_a=%s | noise_b=%s" % (
        "✓ FOUND" if found_a else "✗ MISSED",
        "✓ REJECTED" if not found_noise else "✗ FALSE POSITIVE",
    ))
    if found_a and not found_noise:
        print("\nSelf-test PASSED ✓")
    elif found_a:
        print("\nSelf-test PARTIAL — signal found but false positive present.")
    else:
        print("\nSelf-test WARNING — try increasing n or lowering granger_alpha.")
    return df_out


# ══════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if "--selftest" in sys.argv:
        _run_selftest()
        return
    cfg = CausalConfig.load()
    engine = CausalDiscoveryEngine(cfg)
    engine.run()


if __name__ == "__main__":
    main()
