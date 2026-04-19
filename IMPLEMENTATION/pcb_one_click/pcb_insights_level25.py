#!/usr/bin/env python3
# FILE: pcb_insights_level25.py
# Python 3.8+ compatible
#
# PCB – Level 2.5: Insight Discovery + Validation (local-first) + Guardrails v2.0
#
# Output:
#   out/edges.csv
#   out/insights_level2.csv
#   out/insights_level2.jsonl
#
# Dependencies: numpy, pandas only
#
# Improvements over v1.1:
#   - Config encapsulated in a dataclass (no global mutation side-effects)
#   - DiscoveryEngine class: safe to import, test, and reuse
#   - CANDIDATE_EXCLUDE computed dynamically inside DiscoveryEngine (bug fix)
#   - diff1 detrending for causal-signal isolation
#   - Stability score via rolling correlation (not just sign/strength on slices)
#   - Guardrail: lag-0 leakage hard-flag (corr > 0.95 same-day)
#   - drop_reason built dynamically per gate (not a hardcoded string)
#   - Structured logging via logging module
#   - np.random.default_rng() replaces deprecated RandomState
#   - _residualize_y: condition-number guard before lstsq
#   - _window_sign_prob: pre-converts columns to numpy (O(1) per window)
#   - Runnable smoke-test via --selftest flag
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

logger = logging.getLogger("pcb.level25")


# ============================================================
# Configuration dataclass
# ============================================================

@dataclass
class PCBConfig:
    # Paths
    out_dir: str = "out"
    data_csv: str = "data.csv"

    # Columns
    target_col: str = "target"
    date_col: str = "date"

    # Detrending
    detrend_mode: str = "none"   # "none" | "diff1"

    # Core parameters
    max_lag: int = 7
    min_support_n: int = 25
    q_low: float = 0.30
    q_high: float = 0.70

    # Windowed sign probability
    window_len: int = 14
    window_stride: int = 7
    min_windows: int = 3

    # Bootstrap CI
    boot_b: int = 200
    boot_alpha: float = 0.10   # 90% CI
    boot_seed: int = 12345

    # Quality filters
    max_nan_frac: float = 0.60
    min_unique_num: int = 6
    min_std_eps: float = 1e-6

    # Gate thresholds
    min_strength: float = 0.45
    min_p_sign: float = 0.60
    min_effect_abs: float = 0.05

    # Strength score weights
    w_eff: float = 0.45
    w_psign: float = 0.35
    w_n: float = 0.20
    use_ci_width_penalty: bool = True
    ci_width_soft_max: float = 1.0
    ci_penalty_weight: float = 0.20

    # Causality hardening
    adjustment_mode: str = "full"   # "off" | "light" | "full"

    # Placebo tests
    placebo_enable: bool = True
    placebo_future_enable: bool = True
    placebo_perm_enable: bool = True
    placebo_perm_b: int = 20
    placebo_perm_seed: int = 1337
    placebo_block_len: int = 7
    placebo_margin: float = 0.02

    # Negative control outcome
    negctrl_enable: bool = True
    negctrl_outcome_col: str = "negative_control_outcome"
    negctrl_max_strength: float = 0.30
    negctrl_margin: float = 0.05

    # Slice stability
    stability_enable: bool = True
    stability_slices: List[str] = field(
        default_factory=lambda: ["weekday", "weekend", "first_half", "second_half"]
    )
    stability_min_score: float = 0.66

    # Guardrails
    enable_guardrails: bool = True
    leakage_lag0_hard: float = 0.95      # new: same-day leakage hard threshold
    leakage_future_corr_hard: float = 0.95
    leakage_future_corr_soft: float = 0.80
    leakage_gap_min: float = 0.20
    drift_time_corr_source: float = 0.85
    drift_time_corr_target: float = 0.40
    drop_flagged_insights: bool = True

    # ----------------------------------------------------------------
    @classmethod
    def from_dict(cls, d: dict) -> "PCBConfig":
        cfg = cls()
        cfg.out_dir = str(d.get("out_dir", cfg.out_dir))
        cfg.date_col = str(d.get("date_col", cfg.date_col))
        cfg.target_col = str(d.get("target_col") or d.get("target") or cfg.target_col)

        lv = d.get("level25", {}) if isinstance(d, dict) else {}
        for key in (
            "detrend_mode", "adjustment_mode",
        ):
            if key in lv:
                setattr(cfg, key, str(lv[key]))
        for key in ("max_lag", "min_support_n", "boot_b", "boot_seed",
                    "placebo_perm_b", "placebo_perm_seed", "placebo_block_len",
                    "min_unique_num"):
            if key in lv:
                setattr(cfg, key, int(lv[key]))
        for key in ("min_strength", "min_p_sign", "min_effect_abs",
                    "q_low", "q_high", "w_eff", "w_psign", "w_n",
                    "ci_width_soft_max", "ci_penalty_weight",
                    "placebo_margin", "negctrl_max_strength", "negctrl_margin",
                    "stability_min_score", "max_nan_frac", "min_std_eps",
                    "leakage_lag0_hard", "leakage_future_corr_hard",
                    "leakage_future_corr_soft", "leakage_gap_min",
                    "drift_time_corr_source", "drift_time_corr_target",
                    "boot_alpha"):
            if key in lv:
                setattr(cfg, key, float(lv[key]))
        for key in ("placebo_enable", "placebo_future_enable", "placebo_perm_enable",
                    "negctrl_enable", "stability_enable",
                    "enable_guardrails", "drop_flagged_insights",
                    "use_ci_width_penalty"):
            if key in lv:
                setattr(cfg, key, bool(lv[key]))
        if "stability_slices" in lv:
            cfg.stability_slices = list(lv["stability_slices"])
        if "negative_control_outcome_col" in lv:
            cfg.negctrl_outcome_col = str(lv["negative_control_outcome_col"])
        return cfg

    @classmethod
    def load(cls) -> "PCBConfig":
        try:
            from pcb_config import load_config  # type: ignore
            return cls.from_dict(load_config())
        except Exception:
            return cls()


# ============================================================
# Pure statistical helpers (no global state)
# ============================================================

def _safe_float(x, default: float = np.nan) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else float(default)
    except Exception:
        return float(default)


def _as_str(x) -> str:
    try:
        if x is None:
            return ""
        s = str(x)
        return "" if s.lower() == "nan" else s
    except Exception:
        return ""


def _standardize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if (not np.isfinite(sd)) or sd < 1e-9:
        return x * np.nan
    return (x - mu) / sd


def _corr_safe(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if int(m.sum()) < 15:
        return np.nan
    aa, bb = a[m], b[m]
    if np.nanstd(aa) < 1e-9 or np.nanstd(bb) < 1e-9:
        return np.nan
    try:
        return float(np.corrcoef(aa, bb)[0, 1])
    except Exception:
        return np.nan


def _paired_xy(
    x_arr: np.ndarray, y_arr: np.ndarray, lag: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Return aligned (x[:-lag], y[lag:]) with NaNs removed."""
    n = len(x_arr)
    lag = int(max(1, lag))
    if n <= lag:
        return np.array([], dtype=float), np.array([], dtype=float)
    x0, y1 = x_arr[: n - lag], y_arr[lag:]
    m = np.isfinite(x0) & np.isfinite(y1)
    return x0[m], y1[m]


def _paired_xy_future(
    x_arr: np.ndarray, y_arr: np.ndarray, lag: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Placebo: future x vs current y."""
    lag = int(lag)
    if lag <= 0 or lag >= len(x_arr):
        return np.array([], dtype=float), np.array([], dtype=float)
    x_f = x_arr[lag:]
    y_n = y_arr[:-lag]
    m = np.isfinite(x_f) & np.isfinite(y_n)
    return x_f[m], y_n[m]


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


def _delta_high_low(
    x: np.ndarray, y: np.ndarray, q_low: float, q_high: float
) -> Tuple[float, int, int]:
    if len(x) < 5:
        return np.nan, 0, 0
    lo_q = np.nanquantile(x, q_low)
    hi_q = np.nanquantile(x, q_high)
    low_y = y[x <= lo_q]
    high_y = y[x >= hi_q]
    low_y = low_y[np.isfinite(low_y)]
    high_y = high_y[np.isfinite(high_y)]
    if len(low_y) < 3 or len(high_y) < 3:
        return np.nan, int(len(low_y)), int(len(high_y))
    return float(np.mean(high_y) - np.mean(low_y)), int(len(low_y)), int(len(high_y))


def _bootstrap_ci_delta(
    x: np.ndarray, y: np.ndarray,
    b: int, alpha: float, seed: int,
    q_low: float, q_high: float,
) -> Tuple[float, float]:
    n = len(x)
    if n < 10:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    boots = np.empty(b, dtype=float)
    idx_pool = np.arange(n)
    for i in range(b):
        idx = rng.choice(idx_pool, size=n, replace=True)
        d, _, _ = _delta_high_low(x[idx], y[idx], q_low, q_high)
        boots[i] = d if np.isfinite(d) else np.nan
    lo_p = 100.0 * (alpha / 2.0)
    hi_p = 100.0 * (1.0 - alpha / 2.0)
    valid = boots[np.isfinite(boots)]
    if len(valid) < 10:
        return np.nan, np.nan
    return float(np.percentile(valid, lo_p)), float(np.percentile(valid, hi_p))


def _residualize_y(
    y: np.ndarray, cov_matrix: np.ndarray
) -> np.ndarray:
    """OLS residuals of y on cov_matrix columns. Returns y unchanged if ill-conditioned."""
    if cov_matrix.shape[1] == 0:
        return y
    m = np.isfinite(y) & np.all(np.isfinite(cov_matrix), axis=1)
    if int(np.sum(m)) < 25:
        return y
    A = np.column_stack([np.ones(int(np.sum(m))), cov_matrix[m]])
    cond = np.linalg.cond(A)
    if not np.isfinite(cond) or cond > 1e10:
        logger.debug("_residualize_y: ill-conditioned design matrix (cond=%.1e), skipping.", cond)
        return y
    beta, *_ = np.linalg.lstsq(A, y[m], rcond=1e-10)
    A_full = np.column_stack([np.ones(len(cov_matrix)), cov_matrix])
    yhat = A_full @ beta
    return y - yhat


# ============================================================
# DiscoveryEngine
# ============================================================

class DiscoveryEngine:
    """
    Self-contained PCB Level 2.5 engine.
    Safe to instantiate multiple times with different configs.
    """

    def __init__(self, cfg: Optional[PCBConfig] = None):
        self.cfg = cfg or PCBConfig.load()
        # Bug fix: exclude set computed from the ACTUAL loaded config, not module-load-time globals.
        self._exclude: set = {self.cfg.target_col, self.cfg.date_col}

    # ----------------------------------------------------------
    # Data preparation
    # ----------------------------------------------------------

    def _load_df(self, data_csv_path: Optional[str]) -> pd.DataFrame:
        if data_csv_path is None:
            fallback = os.path.join(self.cfg.out_dir, "demo_data.csv")
            data_csv_path = self.cfg.data_csv if os.path.exists(self.cfg.data_csv) else fallback
        if not os.path.exists(data_csv_path):
            raise FileNotFoundError("Data file not found: %s" % data_csv_path)
        df = pd.read_csv(data_csv_path)
        return self._parse_dates(df)

    def _parse_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        c = self.cfg.date_col
        if c not in df.columns:
            return df
        dt = pd.to_datetime(df[c], errors="coerce")
        if dt.notna().sum() >= max(5, int(0.2 * len(df))):
            out = df.copy()
            out[c] = dt
            return out
        return df

    def _target_col_name(self) -> str:
        if self.cfg.detrend_mode == "diff1":
            return self.cfg.target_col + "_diff1"
        return self.cfg.target_col

    def _apply_detrend(self, df: pd.DataFrame) -> pd.DataFrame:
        tc = self.cfg.target_col
        used = self._target_col_name()
        if used != tc:
            s = pd.to_numeric(df[tc], errors="coerce").astype(float)
            df = df.copy()
            df[used] = s - s.shift(1)
            logger.debug("Detrending applied (diff1): %s → %s", tc, used)
        return df

    def _build_covariates(self, df: pd.DataFrame, target_used: str) -> np.ndarray:
        """Build covariate matrix for residualization."""
        mode = self.cfg.adjustment_mode.lower()
        cols_data = []
        if mode in ["light", "full"]:
            prev = pd.to_numeric(df[target_used], errors="coerce").shift(1).to_numpy(dtype=float)
            cols_data.append(prev)
        if mode == "full":
            cols_data.append(np.arange(len(df), dtype=float))
            dc = self.cfg.date_col
            if dc in df.columns:
                dt = pd.to_datetime(df[dc], errors="coerce")
                if dt.notna().mean() > 0.2:
                    dow = dt.dt.dayofweek.astype(float).to_numpy()
                    cols_data.append(np.sin(2.0 * np.pi * dow / 7.0))
                    cols_data.append(np.cos(2.0 * np.pi * dow / 7.0))
        if not cols_data:
            return np.empty((len(df), 0), dtype=float)
        return np.column_stack(cols_data)

    def _usable_sources(self, df: pd.DataFrame) -> List[str]:
        cfg = self.cfg
        out = []
        for c in df.columns:
            if c in self._exclude:
                continue
            x = pd.to_numeric(df[c], errors="coerce").astype(float)
            nan_frac = float(x.isna().mean())
            if nan_frac > cfg.max_nan_frac:
                continue
            x2 = x.dropna()
            if len(x2) < max(cfg.min_support_n, 10):
                continue
            if int(x2.nunique()) < cfg.min_unique_num:
                continue
            if float(np.nanstd(x2.to_numpy(dtype=float))) < cfg.min_std_eps:
                continue
            if x.notna().sum() < max(cfg.min_support_n, int(0.3 * len(df))):
                continue
            out.append(c)
        return out

    # ----------------------------------------------------------
    # Strength scoring
    # ----------------------------------------------------------

    def _ci_width_penalty(self, ci_low: float, ci_high: float) -> float:
        if not self.cfg.use_ci_width_penalty:
            return 1.0
        lo = _safe_float(ci_low, np.nan)
        hi = _safe_float(ci_high, np.nan)
        if not (np.isfinite(lo) and np.isfinite(hi)):
            return 1.0
        w = float(hi - lo)
        if (not np.isfinite(w)) or w <= 0:
            return 1.0
        t = float(np.clip(w / self.cfg.ci_width_soft_max, 0.0, 3.0))
        mult = 1.0 - self.cfg.ci_penalty_weight * float(np.clip((t - 1.0) / 2.0, 0.0, 1.0))
        return float(np.clip(mult, 0.0, 1.0))

    def _strength_score(
        self,
        effect_size: float, p_sign: float, support_n: float,
        ci_low: float = np.nan, ci_high: float = np.nan,
    ) -> float:
        cfg = self.cfg
        eff = abs(_safe_float(effect_size, np.nan))
        ps = _safe_float(p_sign, np.nan)
        n = _safe_float(support_n, 0.0)
        eff_s = float(np.clip(eff / 0.6, 0.0, 1.0)) if np.isfinite(eff) else 0.0
        ps_s = float(np.clip(ps, 0.0, 1.0)) if np.isfinite(ps) else 0.5
        n_s = float(np.clip(n / 60.0, 0.0, 1.0))
        base = cfg.w_eff * eff_s + cfg.w_psign * ps_s + cfg.w_n * n_s
        return float(np.clip(base * self._ci_width_penalty(ci_low, ci_high), 0.0, 1.0))

    # ----------------------------------------------------------
    # Core per-outcome computation
    # ----------------------------------------------------------

    def _compute_edge_stats(
        self,
        x_arr: np.ndarray, y_arr: np.ndarray,
        df: pd.DataFrame, src: str, outcome_col: str, lag: int,
    ) -> dict:
        """Compute all numerical stats for one (source, outcome, lag) triple."""
        cfg = self.cfg
        x, y = _paired_xy(x_arr, y_arr, lag)
        support_n = int(len(x))
        if support_n < cfg.min_support_n:
            return {"support_n": support_n, "strength": np.nan}

        delta, n_low, n_high = _delta_high_low(x, y, cfg.q_low, cfg.q_high)
        if not np.isfinite(delta):
            return {"support_n": support_n, "strength": np.nan}

        y_std = _standardize(y)
        delta_std, _, _ = _delta_high_low(x, y_std, cfg.q_low, cfg.q_high)
        effect_size = float(delta_std) if np.isfinite(delta_std) else np.nan

        p_sign, n_windows = self._window_sign_prob(x_arr, y_arr, df, src, outcome_col, lag)
        ci_lo, ci_hi = _bootstrap_ci_delta(
            x, y, cfg.boot_b, cfg.boot_alpha, cfg.boot_seed, cfg.q_low, cfg.q_high
        )
        strength = self._strength_score(effect_size, p_sign, support_n, ci_lo, ci_hi)

        return {
            "support_n": support_n,
            "n_low": n_low,
            "n_high": n_high,
            "delta": float(delta),
            "ci_low": float(ci_lo) if np.isfinite(ci_lo) else np.nan,
            "ci_high": float(ci_hi) if np.isfinite(ci_hi) else np.nan,
            "effect_size": effect_size,
            "p_sign": float(p_sign) if np.isfinite(p_sign) else np.nan,
            "n_windows": int(n_windows),
            "strength": float(strength),
        }

    # ----------------------------------------------------------
    # Windowed sign probability (pre-vectorised)
    # ----------------------------------------------------------

    def _window_sign_prob(
        self,
        x_arr: np.ndarray, y_arr: np.ndarray,
        df: pd.DataFrame, src: str, outcome_col: str, lag: int,
    ) -> Tuple[float, int]:
        cfg = self.cfg
        x_full, y_full = _paired_xy(x_arr, y_arr, lag)
        if len(x_full) < cfg.min_support_n:
            return np.nan, 0

        delta_full, _, _ = _delta_high_low(x_full, y_full, cfg.q_low, cfg.q_high)
        if (not np.isfinite(delta_full)) or abs(delta_full) < 1e-12:
            return np.nan, 0

        sign_full = np.sign(delta_full)
        n = len(df)
        aligned_n = n - int(lag)
        if aligned_n < cfg.window_len:
            return np.nan, 0

        # Pre-fetch as numpy arrays (O(1) per window, not O(n))
        xa = pd.to_numeric(df[src], errors="coerce").to_numpy(dtype=float)
        ya = pd.to_numeric(df[outcome_col], errors="coerce").to_numpy(dtype=float)

        wins = []
        start = 0
        min_w = max(10, int(cfg.min_support_n * 0.4))
        while start + cfg.window_len <= aligned_n:
            end = start + cfg.window_len
            xw = xa[start:end]
            yw = ya[start + lag: end + lag]
            m = np.isfinite(xw) & np.isfinite(yw)
            if int(m.sum()) >= min_w:
                dw, _, _ = _delta_high_low(xw[m], yw[m], cfg.q_low, cfg.q_high)
                if np.isfinite(dw) and abs(dw) > 1e-12:
                    wins.append(1 if np.sign(dw) == sign_full else 0)
            start += cfg.window_stride

        if len(wins) < cfg.min_windows:
            return np.nan, int(len(wins))
        return float(np.mean(wins)), int(len(wins))

    # ----------------------------------------------------------
    # Stability (rolling correlation)
    # ----------------------------------------------------------

    def _rolling_corr_stability(
        self,
        x_arr: np.ndarray, y_arr: np.ndarray, lag: int,
    ) -> float:
        """
        Stability via rolling Pearson correlation.
        Returns fraction of windows where sign matches the full-sample sign.
        Provides a finer-grained signal than slice-only stability.
        """
        cfg = self.cfg
        x, y = _paired_xy(x_arr, y_arr, lag)
        if len(x) < cfg.min_support_n:
            return np.nan

        delta_full, _, _ = _delta_high_low(x, y, cfg.q_low, cfg.q_high)
        if not np.isfinite(delta_full) or abs(delta_full) < 1e-12:
            return np.nan

        sign_full = np.sign(delta_full)
        n = len(x)
        wlen = max(cfg.window_len, 10)
        corrs: List[float] = []
        for start in range(0, n - wlen + 1, cfg.window_stride):
            xw = x[start: start + wlen]
            yw = y[start: start + wlen]
            c = _corr_safe(xw, yw)
            if np.isfinite(c):
                corrs.append(c)

        if len(corrs) < cfg.min_windows:
            return np.nan

        sign_agree = [1 if (sign_full * c) > 0 else 0 for c in corrs]
        return float(np.mean(sign_agree))

    def _slice_masks(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        n = len(df)
        masks: Dict[str, np.ndarray] = {}
        if n <= 0:
            return masks
        mid = max(1, n // 2)
        masks["first_half"] = np.arange(n) < mid
        masks["second_half"] = np.arange(n) >= mid

        dc = self.cfg.date_col
        if dc in df.columns:
            dt = pd.to_datetime(df[dc], errors="coerce")
            if dt.notna().mean() > 0.2:
                dow = dt.dt.dayofweek
                masks["weekday"] = (dow <= 4).to_numpy(dtype=bool)
                masks["weekend"] = (dow >= 5).to_numpy(dtype=bool)
        return masks

    def _stability_score(
        self,
        df: pd.DataFrame,
        x_arr: np.ndarray, y_arr: np.ndarray,
        src: str, outcome_col: str, lag: int,
    ) -> Tuple[float, int]:
        """
        Combined stability: rolling correlation agreement + slice agreement.
        Returns (score 0..1, pass 0/1).
        """
        cfg = self.cfg
        if not cfg.stability_enable:
            return np.nan, 1

        roll_score = self._rolling_corr_stability(x_arr, y_arr, lag)

        masks = self._slice_masks(df)
        masks = {k: v for k, v in masks.items() if k in set(cfg.stability_slices)}

        if not masks:
            score = roll_score if np.isfinite(roll_score) else np.nan
            pass_ = int(score >= cfg.stability_min_score) if np.isfinite(score) else 1
            return score, pass_

        full_xa = pd.to_numeric(df[src], errors="coerce").to_numpy(dtype=float)
        full_ya = pd.to_numeric(df[outcome_col], errors="coerce").to_numpy(dtype=float)
        full_x_al, full_y_al = _paired_xy(full_xa, full_ya, lag)

        delta_full, _, _ = _delta_high_low(full_x_al, full_y_al, cfg.q_low, cfg.q_high)
        xs = _compute_strength_for_outcome(self, full_xa, full_ya, df, src, outcome_col, lag)
        full_strength = float(xs.get("strength", np.nan))

        if not np.isfinite(delta_full) or not np.isfinite(full_strength):
            return roll_score, int(np.isfinite(roll_score) and roll_score >= cfg.stability_min_score)

        full_sign = np.sign(delta_full)
        passed, used = 0, 0
        for name, m in masks.items():
            idx = np.where(np.asarray(m, dtype=bool))[0]
            sdf = df.iloc[idx].copy()
            if len(sdf) < cfg.min_support_n + int(lag) + 3:
                continue
            sx = pd.to_numeric(sdf[src], errors="coerce").to_numpy(dtype=float)
            sy = pd.to_numeric(sdf[outcome_col], errors="coerce").to_numpy(dtype=float)
            st = _compute_strength_for_outcome(self, sx, sy, sdf, src, outcome_col, lag)
            sd = float(st.get("delta", np.nan))
            ss = float(st.get("strength", np.nan))
            if not np.isfinite(sd) or not np.isfinite(ss):
                continue
            used += 1
            if np.sign(sd) == full_sign and ss >= max(0.0, 0.85 * full_strength):
                passed += 1

        slice_score = float(passed) / float(used) if used > 0 else np.nan

        # Combine rolling + slice (average if both available)
        scores = [s for s in [roll_score, slice_score] if np.isfinite(s)]
        if not scores:
            return np.nan, 1
        final = float(np.mean(scores))
        return final, int(final >= cfg.stability_min_score)

    # ----------------------------------------------------------
    # Guardrails
    # ----------------------------------------------------------

    def _guardrails_drift(
        self, x_arr: np.ndarray, y_arr: np.ndarray, n: int
    ) -> dict:
        t = np.arange(n, dtype=float)
        cst = _corr_safe(x_arr, t)
        ctt = _corr_safe(y_arr, t)
        cfg = self.cfg
        flag = int(
            np.isfinite(cst) and abs(float(cst)) >= cfg.drift_time_corr_source
            and np.isfinite(ctt) and abs(float(ctt)) >= cfg.drift_time_corr_target
        )
        return {
            "drift_corr_time_source": float(cst) if np.isfinite(cst) else np.nan,
            "drift_corr_time_target": float(ctt) if np.isfinite(ctt) else np.nan,
            "drift_flag": flag,
        }

    def _guardrails_leakage(
        self, x_arr: np.ndarray, y_arr: np.ndarray, lag: int
    ) -> dict:
        """
        Leakage v2.0:
          - Lag-0 hard flag: corr(x[t], y[t]) >= leakage_lag0_hard  (new)
          - Lag-k future flag: as before
        """
        cfg = self.cfg
        n = len(x_arr)
        lag = int(max(1, lag))

        # Lag-0 same-day leakage (new guardrail)
        corr_now_full = _corr_safe(x_arr, y_arr)
        if np.isfinite(corr_now_full) and abs(float(corr_now_full)) >= cfg.leakage_lag0_hard:
            logger.debug(
                "Lag-0 leakage detected: corr=%.3f (threshold %.2f)",
                corr_now_full, cfg.leakage_lag0_hard,
            )
            return {
                "leakage_corr_future": float(corr_now_full),
                "leakage_corr_now": float(corr_now_full),
                "leakage_gap": np.nan,
                "leakage_flag": 1,
            }

        if n <= lag + 2:
            return {"leakage_corr_future": np.nan, "leakage_corr_now": np.nan,
                    "leakage_gap": np.nan, "leakage_flag": 0}

        xf = x_arr[: n - lag]
        yf = y_arr[lag:]
        corr_future = _corr_safe(xf, yf)
        corr_now = _corr_safe(xf, y_arr[: n - lag])

        gap = np.nan
        if np.isfinite(corr_future) and np.isfinite(corr_now):
            gap = abs(float(corr_future)) - abs(float(corr_now))

        flag = 0
        if np.isfinite(corr_future) and abs(float(corr_future)) >= cfg.leakage_future_corr_hard:
            flag = 1
        elif (np.isfinite(corr_future) and abs(float(corr_future)) >= cfg.leakage_future_corr_soft
              and np.isfinite(gap) and float(gap) >= cfg.leakage_gap_min):
            flag = 1

        return {
            "leakage_corr_future": float(corr_future) if np.isfinite(corr_future) else np.nan,
            "leakage_corr_now": float(corr_now) if np.isfinite(corr_now) else np.nan,
            "leakage_gap": float(gap) if np.isfinite(gap) else np.nan,
            "leakage_flag": flag,
        }

    # ----------------------------------------------------------
    # Human output
    # ----------------------------------------------------------

    @staticmethod
    def _make_statement(
        source: str, target: str, lag: int,
        delta: float, ci_lo: float, ci_hi: float,
    ) -> str:
        src = source.replace("_", " ")
        tgt = target.replace("_", " ")
        d = _safe_float(delta, np.nan)
        lo = _safe_float(ci_lo, np.nan)
        hi = _safe_float(ci_hi, np.nan)
        if np.isfinite(d):
            direction = "higher" if d > 0 else "lower"
            if np.isfinite(lo) and np.isfinite(hi):
                return (
                    "When %s is high, %s tends to be %s %d day(s) later "
                    "(Δ=%.2f, CI %.2f..%.2f)." % (src, tgt, direction, lag, d, lo, hi)
                )
            return "When %s is high, %s tends to be %s %d day(s) later (Δ=%.2f)." % (
                src, tgt, direction, lag, d
            )
        return "Potential relationship: %s → %s (lag %d)." % (src, tgt, lag)

    @staticmethod
    def _make_reco(source: str, target: str, lag: int, delta: float) -> str:
        src = source.replace("_", " ")
        tgt = target.replace("_", " ")
        d = _safe_float(delta, np.nan)
        if not np.isfinite(d):
            return "Track %s and %s for a few weeks and re-run PCB." % (src, tgt)
        if d > 0:
            return (
                "Consider increasing %s (within safe limits) and observe %s over the next %d day(s)."
                % (src, tgt, lag)
            )
        return (
            "Consider reducing %s (within safe limits) and observe %s over the next %d day(s)."
            % (src, tgt, lag)
        )

    # ----------------------------------------------------------
    # I/O
    # ----------------------------------------------------------

    def _ensure_out(self):
        os.makedirs(self.cfg.out_dir, exist_ok=True)

    def _save_csv(self, df: pd.DataFrame, path: str):
        self._ensure_out()
        df.to_csv(path, index=False)

    def _save_jsonl(self, df: pd.DataFrame, path: str):
        self._ensure_out()
        with open(path, "w", encoding="utf-8") as f:
            for _, r in df.iterrows():
                f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

    # ----------------------------------------------------------
    # Main pipeline
    # ----------------------------------------------------------

    def run(self, data_csv_path: Optional[str] = None) -> pd.DataFrame:
        cfg = self.cfg
        out_dir = cfg.out_dir
        edges_path = os.path.join(out_dir, "edges.csv")
        out_csv = os.path.join(out_dir, "insights_level2.csv")
        out_jsonl = os.path.join(out_dir, "insights_level2.jsonl")

        df = self._load_df(data_csv_path)
        data_path_used = data_csv_path or cfg.data_csv

        if cfg.target_col not in df.columns:
            raise ValueError("Target column '%s' not found." % cfg.target_col)

        df = self._apply_detrend(df)
        target_used = self._target_col_name()

        sources = self._usable_sources(df)
        if not sources:
            raise ValueError("No usable source columns found.")
        logger.info("Sources: %d | Lags: 1..%d | Target: %s", len(sources), cfg.max_lag, target_used)

        # Pre-fetch target as numpy once
        y_full_arr = pd.to_numeric(df[target_used], errors="coerce").to_numpy(dtype=float)
        cov_matrix = self._build_covariates(df, target_used)
        if cfg.adjustment_mode.lower() != "off":
            y_full_arr = _residualize_y(y_full_arr, cov_matrix)

        rows = []
        perm_rng = np.random.default_rng(cfg.placebo_perm_seed)

        for src in sources:
            x_full_arr = pd.to_numeric(df[src], errors="coerce").to_numpy(dtype=float)

            drift_meta = {"drift_corr_time_source": np.nan, "drift_corr_time_target": np.nan, "drift_flag": 0}
            if cfg.enable_guardrails:
                drift_meta = self._guardrails_drift(x_full_arr, y_full_arr, len(df))

            for lag in range(1, int(cfg.max_lag) + 1):
                stats = self._compute_edge_stats(
                    x_full_arr, y_full_arr, df, src, target_used, lag
                )
                if not np.isfinite(stats.get("strength", np.nan)):
                    continue

                support_n = stats["support_n"]
                delta = stats["delta"]
                effect_size = stats["effect_size"]
                p_sign = stats["p_sign"]
                n_windows = stats["n_windows"]
                ci_lo = stats["ci_low"]
                ci_hi = stats["ci_high"]
                strength = stats["strength"]
                n_low = stats["n_low"]
                n_high = stats["n_high"]
                ci_width = (
                    float(ci_hi - ci_lo)
                    if (np.isfinite(ci_lo) and np.isfinite(ci_hi))
                    else np.nan
                )

                # ---- Placebo tests
                pf_strength = perm_p90 = np.nan
                placebo_future_pass = placebo_perm_pass = 1
                if cfg.placebo_enable:
                    if cfg.placebo_future_enable:
                        xf, yf = _paired_xy_future(x_full_arr, y_full_arr, lag)
                        if len(xf) >= cfg.min_support_n:
                            yf_std = _standardize(yf)
                            d_f, _, _ = _delta_high_low(xf, yf_std, cfg.q_low, cfg.q_high)
                            es_f = float(d_f) if np.isfinite(d_f) else np.nan
                            pf_strength = float(
                                self._strength_score(es_f, p_sign, int(len(xf)))
                            )
                            if np.isfinite(pf_strength) and np.isfinite(strength):
                                placebo_future_pass = int(
                                    strength >= pf_strength + cfg.placebo_margin
                                )

                    if cfg.placebo_perm_enable and cfg.placebo_perm_b > 0:
                        vals = []
                        for _ in range(cfg.placebo_perm_b):
                            xp = _block_permute(x_full_arr, cfg.placebo_block_len, perm_rng)
                            yp_std = _standardize(y_full_arr)
                            dp, _, _ = _delta_high_low(xp, yp_std, cfg.q_low, cfg.q_high)
                            sp_ = float(self._strength_score(float(dp) if np.isfinite(dp) else np.nan, p_sign, int(len(xp))))
                            if np.isfinite(sp_):
                                vals.append(sp_)
                        if len(vals) > 5:
                            perm_p90 = float(np.quantile(np.array(vals, dtype=float), 0.90))
                            if np.isfinite(strength):
                                placebo_perm_pass = int(strength >= perm_p90 + cfg.placebo_margin)

                placebo_pass = int(placebo_future_pass == 1 and placebo_perm_pass == 1)

                # ---- Slice stability (combined rolling + slice)
                stab_score, stab_pass = self._stability_score(
                    df, x_full_arr, y_full_arr, src, target_used, lag
                )

                # ---- Negative control outcome
                negctrl_strength = np.nan
                negctrl_pass = 1
                if cfg.negctrl_enable and cfg.negctrl_outcome_col in df.columns \
                        and cfg.negctrl_outcome_col != target_used:
                    nc_arr = pd.to_numeric(df[cfg.negctrl_outcome_col], errors="coerce").to_numpy(dtype=float)
                    nc = _compute_strength_for_outcome(self, x_full_arr, nc_arr, df, src, cfg.negctrl_outcome_col, lag)
                    negctrl_strength = float(nc.get("strength", np.nan))
                    if np.isfinite(negctrl_strength) and np.isfinite(strength):
                        if negctrl_strength >= cfg.negctrl_max_strength:
                            negctrl_pass = 0
                        elif (strength - negctrl_strength) < cfg.negctrl_margin:
                            negctrl_pass = 0

                # ---- Guardrails
                leak_meta = {"leakage_corr_future": np.nan, "leakage_corr_now": np.nan,
                             "leakage_gap": np.nan, "leakage_flag": 0}
                if cfg.enable_guardrails:
                    leak_meta = self._guardrails_leakage(x_full_arr, y_full_arr, lag)

                gr_flag = 0
                gr_reason = ""
                if leak_meta["leakage_flag"]:
                    gr_flag, gr_reason = 1, "leakage_future_corr"
                elif drift_meta["drift_flag"]:
                    gr_flag, gr_reason = 1, "drift_time_corr"

                rows.append({
                    "edge_id": "E2-%s-%s-L%d" % (src, cfg.target_col, lag),
                    "source": src,
                    "target": cfg.target_col,
                    "target_used": target_used,
                    "lag": int(lag),
                    "support_n": int(support_n),
                    "n_low": int(n_low),
                    "n_high": int(n_high),
                    "delta": float(delta),
                    "ci_low": float(ci_lo) if np.isfinite(ci_lo) else np.nan,
                    "ci_high": float(ci_hi) if np.isfinite(ci_hi) else np.nan,
                    "ci_width": float(ci_width) if np.isfinite(ci_width) else np.nan,
                    "effect_size": float(effect_size) if np.isfinite(effect_size) else np.nan,
                    "p_sign": float(p_sign) if np.isfinite(p_sign) else np.nan,
                    "n_windows": int(n_windows),
                    "strength": float(strength),
                    "placebo_future_strength": float(pf_strength) if np.isfinite(pf_strength) else np.nan,
                    "placebo_perm_strength_p90": float(perm_p90) if np.isfinite(perm_p90) else np.nan,
                    "placebo_future_pass": int(placebo_future_pass),
                    "placebo_perm_pass": int(placebo_perm_pass),
                    "placebo_pass": int(placebo_pass),
                    "stability_score": float(stab_score) if np.isfinite(stab_score) else np.nan,
                    "stability_pass": int(stab_pass),
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

        empty_cols: List[str] = []
        if len(df_edges) == 0:
            for p in [edges_path, out_csv]:
                self._save_csv(pd.DataFrame(columns=empty_cols), p)
            self._save_jsonl(pd.DataFrame(columns=empty_cols), out_jsonl)
            logger.warning("No edges met MIN_SUPPORT_N=%d. Check data coverage.", cfg.min_support_n)
            print("No edges met MIN_SUPPORT_N. Check data coverage.")
            return pd.DataFrame()

        df_edges = df_edges.sort_values(["strength", "support_n"], ascending=[False, False]).reset_index(drop=True)
        self._save_csv(df_edges, edges_path)

        # ---- Build insights with dynamic drop_reason
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
            lambda r: self._make_statement(
                r["source"], r["target"], int(r["lag"]),
                r["delta_test"], r["ci_low_test"], r["ci_high_test"]
            ), axis=1,
        )
        df_i["recommendation"] = df_i.apply(
            lambda r: self._make_reco(r["source"], r["target"], int(r["lag"]), r["delta_test"]),
            axis=1,
        )

        # Dynamic drop_reason: accumulate which gates failed
        gate = pd.Series([True] * len(df_i), index=df_i.index)
        drop_reasons: Dict[int, List[str]] = {i: [] for i in df_i.index}

        def _apply_gate(mask: pd.Series, label: str):
            nonlocal gate
            failed = gate & ~mask
            for idx in failed[failed].index:
                drop_reasons[idx].append(label)
            gate = gate & mask

        _apply_gate(
            df_i["support_n_test"].astype(float) >= float(cfg.min_support_n),
            "min_support"
        )
        _apply_gate(
            df_i["strength"].astype(float) >= float(cfg.min_strength),
            "min_strength"
        )
        _apply_gate(
            pd.to_numeric(df_i["p_sign_test"], errors="coerce").fillna(0.5) >= float(cfg.min_p_sign),
            "min_p_sign"
        )
        _apply_gate(
            pd.to_numeric(df_i["effect_size_test"], errors="coerce").abs().fillna(0.0) >= float(cfg.min_effect_abs),
            "min_effect"
        )
        if cfg.placebo_enable and "placebo_pass" in df_i.columns:
            _apply_gate(
                pd.to_numeric(df_i["placebo_pass"], errors="coerce").fillna(1).astype(int) == 1,
                "placebo"
            )
        if cfg.stability_enable and "stability_pass" in df_i.columns:
            _apply_gate(
                pd.to_numeric(df_i["stability_pass"], errors="coerce").fillna(1).astype(int) == 1,
                "stability"
            )
        if cfg.negctrl_enable and "negctrl_pass" in df_i.columns:
            _apply_gate(
                pd.to_numeric(df_i["negctrl_pass"], errors="coerce").fillna(1).astype(int) == 1,
                "negctrl"
            )

        df_i["drop_reason"] = [
            "+".join(drop_reasons[i]) if drop_reasons[i] else "" for i in df_i.index
        ]

        if cfg.enable_guardrails and cfg.drop_flagged_insights:
            gr_flag_col = pd.to_numeric(df_i.get("guardrail_flag", 0), errors="coerce").fillna(0).astype(int)
            bad = gr_flag_col == 1
            for idx in df_i[bad & gate].index:
                reason = _as_str(df_i.loc[idx, "guardrail_reason"])
                df_i.loc[idx, "drop_reason"] = "guardrail(%s)" % reason
            gate = gate & ~bad

        df_i["kept"] = gate.astype(int)

        df_out = df_i[df_i["kept"] == 1].copy()
        df_out = df_out.sort_values(["strength", "support_n_test"], ascending=[False, False]).reset_index(drop=True)

        cols_out = [
            "insight_id", "source", "target", "lag",
            "strength", "support_n_test",
            "effect_size_test", "p_sign_test",
            "delta_test", "ci_low_test", "ci_high_test",
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

        self._save_csv(df_out, out_csv)
        self._save_jsonl(df_out, out_jsonl)

        logger.info("Edges: %d | Insights kept: %d", len(df_edges), len(df_out))
        print("\n=== PCB LEVEL 2.5 (v2.0) ===")
        print("Data:", data_path_used)
        print("Target used:", target_used)
        print("Detrend mode:", cfg.detrend_mode)
        print("Saved:", edges_path)
        print("Saved:", out_csv)
        print("Saved:", out_jsonl)
        print("Edges:", len(df_edges))
        print("Kept insights:", len(df_out))
        if cfg.enable_guardrails:
            n_flagged = int(
                pd.to_numeric(df_i.get("guardrail_flag", 0), errors="coerce")
                .fillna(0).astype(int).sum()
            )
            print("Guardrails flagged (edges):", n_flagged)
            print("Drop policy:", "ON" if cfg.drop_flagged_insights else "OFF (audit only)")

        if len(df_out) > 0:
            show = ["insight_id", "source", "lag", "strength", "support_n_test",
                    "effect_size_test", "p_sign_test", "delta_test"]
            print("\nTop insights:")
            print(df_out[show].head(10).to_string(index=False))
        else:
            print("\n(No insights passed gates. Try lowering thresholds or add data.)")
            if df_i[~df_i["kept"].astype(bool)]["drop_reason"].value_counts().head(3).any():
                print("Top drop reasons:")
                print(df_i["drop_reason"].value_counts().head(5).to_string())

        return df_out


# ============================================================
# Module-level helper (used by stability internals)
# ============================================================

def _compute_strength_for_outcome(
    engine: DiscoveryEngine,
    x_arr: np.ndarray, y_arr: np.ndarray,
    df: pd.DataFrame, src: str, outcome_col: str, lag: int,
) -> dict:
    cfg = engine.cfg
    x, y = _paired_xy(x_arr, y_arr, lag)
    support_n = int(len(x))
    if support_n < cfg.min_support_n:
        return {"support_n": support_n, "delta": np.nan, "effect_size": np.nan,
                "p_sign": np.nan, "n_windows": 0, "strength": np.nan}
    delta, _, _ = _delta_high_low(x, y, cfg.q_low, cfg.q_high)
    y_std = _standardize(y)
    d_std, _, _ = _delta_high_low(x, y_std, cfg.q_low, cfg.q_high)
    effect_size = float(d_std) if np.isfinite(d_std) else np.nan
    p_sign, n_windows = engine._window_sign_prob(x_arr, y_arr, df, src, outcome_col, lag)
    strength = float(engine._strength_score(effect_size, p_sign, support_n))
    return {
        "support_n": support_n,
        "delta": float(delta) if np.isfinite(delta) else np.nan,
        "effect_size": float(effect_size) if np.isfinite(effect_size) else np.nan,
        "p_sign": float(p_sign) if np.isfinite(p_sign) else np.nan,
        "n_windows": int(n_windows),
        "strength": float(strength) if np.isfinite(strength) else np.nan,
    }


# ============================================================
# Smoke-test
# ============================================================

def _run_selftest():
    """Quick smoke-test with 90 rows of synthetic data. Run via: python pcb_insights_level25.py --selftest"""
    import tempfile
    print("Running self-test...")
    rng = np.random.default_rng(42)
    n = 90
    t = np.arange(n)
    x = rng.normal(0, 1, n)
    # y correlates with x lagged by 3 days
    y = np.roll(x, 3) + rng.normal(0, 0.5, n)
    y[:3] = rng.normal(0, 1, 3)

    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = os.path.join(tmpdir, "data.csv")
        pd.DataFrame({"date": pd.date_range("2024-01-01", periods=n), "target": y, "signal_a": x}).to_csv(data_path, index=False)

        cfg = PCBConfig()
        cfg.out_dir = os.path.join(tmpdir, "out")
        cfg.min_support_n = 10
        cfg.min_windows = 2

        engine = DiscoveryEngine(cfg)
        df_out = engine.run(data_path)

    if len(df_out) > 0:
        print("Self-test PASSED: %d insight(s) found." % len(df_out))
    else:
        print("Self-test WARNING: pipeline ran but no insights passed gates (may be OK for synthetic data).")
    return df_out


# ============================================================
# Entry point
# ============================================================

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if "--selftest" in sys.argv:
        _run_selftest()
        return
    cfg = PCBConfig.load()
    engine = DiscoveryEngine(cfg)
    engine.run()


if __name__ == "__main__":
    main()
