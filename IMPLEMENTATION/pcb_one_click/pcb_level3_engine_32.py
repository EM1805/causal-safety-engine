# FILE: pcb_level3_engine_32.py
# Python 3.7+ compatible
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
# Dependencies: numpy, pandas

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Tuple, List, Any
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ============================================================================
# LOGGING SETUP
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================
@dataclass
class Level32Config:
    """Configuration for Level 3.2 counterfactual validation"""
    
    # Directory and file paths
    out_dir: str = "out"
    insights_l2_file: str = "insights_level2.csv"
    exp_results_file: str = "experiment_results.csv"
    exp_intake_l31_file: str = "experiment_intake_level31.csv"
    out_l3_file: str = "insights_level3.csv"
    out_ledger_file: str = "insights_level3_ledger.csv"
    out_trials_file: str = "experiment_trials_enriched_level32.csv"
    
    # Data files
    default_data_csv: str = "data.csv"
    fallback_data_csv: str = "demo_data.csv"
    clean_data_csv: str = "data_clean.csv"
    
    # Column names
    target_col: str = "target"
    date_col: str = "date"
    
    # Lookback parameters
    lookback_days: int = 90
    lookback_rows: int = 90
    
    # Matching parameters
    k_controls: int = 10
    min_matched: int = 5
    match_dist_max: float = 1.5
    
    # Success thresholds
    z_success: float = 0.20
    z_clip: float = 6.0
    min_trials: int = 2
    min_success_lb: float = 0.55
    
    # Negative control configuration
    negctrl_enable: bool = True
    negctrl_outcome_col: str = "negative_control_outcome"
    negctrl_max_success_lb: float = 0.55
    
    # Propensity scoring
    enable_propensity: bool = True
    propensity_max_diff: float = 0.20
    propensity_action_col: str = "action_active"
    propensity_l2: float = 1.0
    propensity_steps: int = 300
    propensity_lr: float = 0.15
    propensity_min_samples: int = 25
    propensity_min_per_class: int = 5
    
    # Pre-trend check
    enable_pretrend_check: bool = True
    pretrend_days: int = 7
    pretrend_max_diff: float = 0.30
    pretrend_min_samples: int = 3
    
    # Covariate limits
    max_covariates: int = 10
    min_numeric_threshold: float = 0.3
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Level32Config':
        """Load configuration from dictionary"""
        # Handle nested level32 config
        if "level32" in config_dict:
            level32_dict = config_dict.get("level32", {})
            # Merge top-level and level32-specific configs
            merged = {**config_dict, **level32_dict}
            return cls(**{k: v for k, v in merged.items() if k in cls.__annotations__})
        
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__annotations__})
    
    @classmethod
    def load_from_file(cls, config_path: str = "pcb_config.py") -> 'Level32Config':
        """Load configuration from external file"""
        try:
            from pcb_config import load_config
            config_dict = load_config()
            return cls.from_dict(config_dict)
        except ImportError:
            logger.debug("No external config file found, using defaults")
            return cls()
        except Exception as e:
            logger.warning(f"Error loading config: {e}, using defaults")
            return cls()


# ============================================================================
# PATH MANAGER
# ============================================================================
class PathManager:
    """Manages all file paths for Level 3.2"""
    
    def __init__(self, config: Level32Config):
        self.config = config
        self.base = Path(config.out_dir)
        self.base.mkdir(exist_ok=True)
    
    @property
    def insights_l2(self) -> Path:
        return self.base / self.config.insights_l2_file
    
    @property
    def exp_results(self) -> Path:
        return self.base / self.config.exp_results_file
    
    @property
    def exp_intake_l31(self) -> Path:
        return self.base / self.config.exp_intake_l31_file
    
    @property
    def out_l3(self) -> Path:
        return self.base / self.config.out_l3_file
    
    @property
    def out_ledger(self) -> Path:
        return self.base / self.config.out_ledger_file
    
    @property
    def out_trials(self) -> Path:
        return self.base / self.config.out_trials_file
    
    def get_trials_path(self) -> Path:
        """Pick the correct trials file (prefer intake_level31 if exists)"""
        if self.exp_intake_l31.exists():
            logger.info(f"Using trials from: {self.exp_intake_l31}")
            return self.exp_intake_l31
        return self.exp_results
    
    def get_data_path(self, custom_path: Optional[str] = None) -> Path:
        """Resolve data file path with fallback chain"""
        if custom_path and Path(custom_path).exists():
            return Path(custom_path)
        
        # Prefer data_clean.csv if available
        clean_path = self.base / self.config.clean_data_csv
        if clean_path.exists():
            logger.info(f"Using cleaned data: {clean_path}")
            return clean_path
        
        default_path = Path(self.config.default_data_csv)
        if default_path.exists():
            return default_path
        
        fallback_path = self.base / self.config.fallback_data_csv
        if fallback_path.exists():
            return fallback_path
        
        raise FileNotFoundError(
            f"Data file not found. Tried: {custom_path}, "
            f"{clean_path}, {default_path}, {fallback_path}"
        )


# ============================================================================
# DATA UTILITIES
# ============================================================================
def _safe_float(x: Any, default: float = np.nan) -> float:
    """Safely convert to float"""
    try:
        v = float(x)
        return v if np.isfinite(v) else float(default)
    except (TypeError, ValueError):
        return float(default)


def _as_str(x: Any) -> str:
    """Safely convert to string"""
    try:
        return str(x) if x is not None else ""
    except Exception:
        return ""


def _try_parse_date(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """
    Attempt to parse date column as datetime
    
    Args:
        df: DataFrame to process
        date_col: Name of date column
        
    Returns:
        DataFrame with date column converted if successful
    """
    if date_col not in df.columns:
        logger.warning(f"Date column '{date_col}' not found")
        return df
    
    dt = pd.to_datetime(df[date_col], errors="coerce")
    valid_ratio = dt.notna().mean()
    
    if valid_ratio > 0.2:
        result = df.copy()
        result[date_col] = dt
        logger.info(f"Parsed {dt.notna().sum()}/{len(df)} dates ({valid_ratio:.1%} valid)")
        return result
    else:
        logger.warning(f"Only {valid_ratio:.1%} valid dates, keeping original format")
        return df


def _has_date(df: pd.DataFrame, date_col: str) -> bool:
    """Check if DataFrame has valid date column"""
    return (date_col in df.columns and 
            pd.api.types.is_datetime64_any_dtype(df[date_col]))


def _numeric_cols(df: pd.DataFrame, 
                  exclude_cols: List[str],
                  min_threshold: float = 0.3) -> List[str]:
    """
    Extract numeric columns from DataFrame
    
    Args:
        df: DataFrame to analyze
        exclude_cols: Columns to exclude
        min_threshold: Minimum ratio of valid numeric values
        
    Returns:
        List of numeric column names
    """
    numeric_cols = []
    
    for col in df.columns:
        if col in exclude_cols:
            continue
        
        series = pd.to_numeric(df[col], errors="coerce")
        valid_ratio = series.notna().sum() / len(df)
        
        if valid_ratio >= min_threshold:
            numeric_cols.append(col)
    
    logger.debug(f"Found {len(numeric_cols)} numeric columns")
    return numeric_cols


def _robust_scale(x: np.ndarray) -> np.ndarray:
    """
    Robust scaling using median and IQR
    
    Args:
        x: Array to scale
        
    Returns:
        Scaled array
    """
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    iqr = np.nanpercentile(x, 75) - np.nanpercentile(x, 25)
    
    if not np.isfinite(iqr) or iqr < 1e-6:
        iqr = np.nanstd(x)
    
    if not np.isfinite(iqr) or iqr < 1e-6:
        iqr = 1.0
    
    return (x - med) / iqr


def _sr_lower_bound(wins: int, n: int) -> float:
    """
    Conservative success rate lower bound
    
    Args:
        wins: Number of successes
        n: Total trials
        
    Returns:
        Lower bound estimate (one-success penalty)
    """
    if n <= 0:
        return 0.0
    return max(0.0, (wins - 1) / float(n))


# ============================================================================
# CALENDAR AND COVARIATE UTILITIES
# ============================================================================
def _ensure_calendar_covs(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """
    Add calendar and trend covariates for matching
    
    Args:
        df: DataFrame to augment
        date_col: Name of date column
        
    Returns:
        DataFrame with added covariates
    """
    result = df.copy()
    result["time_idx"] = np.arange(len(result), dtype=float)
    
    if _has_date(result, date_col):
        dow = result[date_col].dt.dayofweek.astype(float)
        result["dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
        result["dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0)
        logger.debug("Added calendar covariates: time_idx, dow_sin, dow_cos")
    else:
        logger.debug("Added trend covariate: time_idx")
    
    return result


def _past_indices(df: pd.DataFrame, 
                  t_idx: int,
                  date_col: str,
                  lookback_days: int,
                  lookback_rows: int) -> np.ndarray:
    """
    Get indices of past observations for baseline window
    
    Args:
        df: DataFrame with data
        t_idx: Current time index
        date_col: Name of date column
        lookback_days: Days to look back (if dates available)
        lookback_rows: Rows to look back (fallback)
        
    Returns:
        Array of past indices
    """
    if t_idx <= 0:
        return np.array([], dtype=int)
    
    # Try date-based lookback first
    if _has_date(df, date_col):
        d_t = df[date_col].iloc[t_idx]
        if pd.notna(d_t):
            start_date = d_t.normalize() - pd.Timedelta(days=lookback_days)
            mask = (df[date_col] < d_t) & (df[date_col] >= start_date)
            indices = df.index[mask].to_numpy(dtype=int)
            logger.debug(
                f"Date-based window: {len(indices)} points "
                f"(lookback {lookback_days} days)"
            )
            return indices
    
    # Fallback: row-based lookback
    start = max(0, t_idx - lookback_rows)
    indices = np.arange(start, t_idx, dtype=int)
    logger.debug(f"Row-based window: {len(indices)} points (rows {start} to {t_idx})")
    return indices


# ============================================================================
# LOGISTIC REGRESSION (for Propensity Scoring)
# ============================================================================
def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Sigmoid function with numerical stability
    
    Args:
        z: Input array
        
    Returns:
        Sigmoid transformed array
    """
    z = np.asarray(z, dtype=float)
    z = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def _fit_logistic_proba(X: np.ndarray, 
                        y: np.ndarray,
                        l2: float = 1.0,
                        steps: int = 250,
                        lr: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Lightweight logistic regression via gradient descent
    
    Args:
        X: Feature matrix (n, k)
        y: Binary target (n,) in {0, 1}
        l2: L2 regularization strength
        steps: Number of gradient descent steps
        lr: Learning rate
        
    Returns:
        Tuple of (weights, feature_means, feature_stds)
        
    Notes:
        Returns standardization parameters for later prediction
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, k = X.shape
    
    # Standardize features
    mu = np.nanmean(X, axis=0)
    sig = np.nanstd(X, axis=0)
    sig = np.where(np.isfinite(sig) & (sig > 1e-6), sig, 1.0)
    X_scaled = (X - mu) / sig
    
    # Add intercept
    A = np.column_stack([np.ones(n), X_scaled])
    
    # Initialize weights
    w = np.zeros(k + 1, dtype=float)
    l2 = float(max(0.0, l2))
    
    # Gradient descent
    for step in range(int(steps)):
        p = _sigmoid(A @ w)
        
        # Gradient
        grad = (A.T @ (p - y)) / float(n)
        
        # L2 regularization (not on intercept)
        grad[1:] += (l2 / float(n)) * w[1:]
        
        # Update
        w -= float(lr) * grad
    
    logger.debug(
        f"Fitted logistic regression: {k} features, "
        f"{steps} steps, final weights norm={np.linalg.norm(w):.3f}"
    )
    
    return w, mu, sig


def _predict_logistic_proba(X: np.ndarray,
                           w: np.ndarray,
                           mu: np.ndarray,
                           sig: np.ndarray) -> np.ndarray:
    """
    Predict probabilities using fitted logistic model
    
    Args:
        X: Feature matrix (n, k)
        w: Weights (k+1,) including intercept
        mu: Feature means for standardization
        sig: Feature stds for standardization
        
    Returns:
        Predicted probabilities (n,)
    """
    X = np.asarray(X, dtype=float)
    X_scaled = (X - mu) / sig
    A = np.column_stack([np.ones(len(X_scaled)), X_scaled])
    return _sigmoid(A @ w)


# ============================================================================
# PROPENSITY SCORE CALCULATOR
# ============================================================================
class PropensityScoreCalculator:
    """Computes propensity scores for treatment assignment"""
    
    def __init__(self, config: Level32Config):
        self.config = config
    
    def compute(self, 
                df: pd.DataFrame,
                action_col: str,
                covariates: List[str]) -> np.ndarray:
        """
        Compute propensity p(treated | covariates)
        
        Args:
            df: DataFrame with data
            action_col: Binary treatment indicator column
            covariates: List of covariate columns
            
        Returns:
            Array of propensity scores (NaN if not computable)
        """
        if action_col not in df.columns:
            logger.warning(f"Action column '{action_col}' not found")
            return np.full(len(df), np.nan, dtype=float)
        
        # Filter available covariates
        available_covs = [c for c in covariates if c in df.columns]
        if len(available_covs) == 0:
            logger.warning("No covariates available for propensity scoring")
            return np.full(len(df), np.nan, dtype=float)
        
        # Build feature matrix
        X_list = []
        for col in available_covs:
            X_list.append(pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float))
        X = np.vstack(X_list).T
        
        # Get treatment indicator
        y = pd.to_numeric(df[action_col], errors="coerce").fillna(0).to_numpy(dtype=float)
        
        # Filter to complete cases
        complete_mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        n_complete = int(np.sum(complete_mask))
        
        if n_complete < self.config.propensity_min_samples:
            logger.warning(
                f"Insufficient samples for propensity: {n_complete} < "
                f"{self.config.propensity_min_samples}"
            )
            return np.full(len(df), np.nan, dtype=float)
        
        # Check class balance
        y_complete = y[complete_mask]
        n_treated = int(np.sum(y_complete > 0.5))
        n_control = int(np.sum(y_complete <= 0.5))
        
        if (n_treated < self.config.propensity_min_per_class or 
            n_control < self.config.propensity_min_per_class):
            logger.warning(
                f"Insufficient class balance: treated={n_treated}, "
                f"control={n_control} (min={self.config.propensity_min_per_class})"
            )
            return np.full(len(df), np.nan, dtype=float)
        
        # Fit logistic regression
        X_complete = X[complete_mask]
        y_binary = (y_complete > 0.5).astype(float)
        
        w, mu, sig = _fit_logistic_proba(
            X_complete, y_binary,
            l2=self.config.propensity_l2,
            steps=self.config.propensity_steps,
            lr=self.config.propensity_lr
        )
        
        # Predict for all samples
        propensity = np.full(len(df), np.nan, dtype=float)
        propensity[complete_mask] = _predict_logistic_proba(X_complete, w, mu, sig)
        
        logger.info(
            f"Computed propensity scores: {n_complete} samples, "
            f"{len(available_covs)} covariates"
        )
        
        return propensity


# ============================================================================
# PRE-TREND CHECK
# ============================================================================
def _trend_slope(y: np.ndarray) -> float:
    """
    Compute linear trend slope
    
    Args:
        y: Time series values
        
    Returns:
        Slope coefficient or NaN if insufficient data
    """
    y = np.asarray(y, dtype=float)
    valid_mask = np.isfinite(y)
    
    if int(np.sum(valid_mask)) < 3:
        return np.nan
    
    x = np.arange(len(y), dtype=float)[valid_mask]
    y_valid = y[valid_mask]
    
    try:
        slope, _ = np.polyfit(x, y_valid, 1)
        return float(slope)
    except Exception as e:
        logger.debug(f"Trend slope calculation failed: {e}")
        return np.nan


class PreTrendChecker:
    """Checks for parallel pre-trends between treated and control units"""
    
    def __init__(self, config: Level32Config):
        self.config = config
    
    def check(self,
              df: pd.DataFrame,
              t_idx: int,
              controls: List[int],
              target_col: str) -> Tuple[int, float, float, float, str]:
        """
        Compare pre-trend slopes between trial and controls
        
        Args:
            df: DataFrame with data
            t_idx: Trial time index
            controls: List of control indices
            target_col: Target outcome column
            
        Returns:
            Tuple of (pass_flag, slope_trial, slope_controls_mean, diff_abs, reason)
        """
        days = max(self.config.pretrend_min_samples, self.config.pretrend_days)
        y = pd.to_numeric(df[target_col], errors="coerce").to_numpy(dtype=float)
        
        def get_window(idx: int) -> Optional[np.ndarray]:
            start = max(0, idx - days)
            end = idx
            if end - start < self.config.pretrend_min_samples:
                return None
            return y[start:end]
        
        # Trial window
        y_trial = get_window(t_idx)
        if y_trial is None:
            return 1, np.nan, np.nan, np.nan, "too_early"
        
        slope_trial = _trend_slope(y_trial)
        if not np.isfinite(slope_trial):
            return 1, np.nan, np.nan, np.nan, "trial_slope_nan"
        
        # Control windows
        slopes_control = []
        for c_idx in controls:
            y_control = get_window(c_idx)
            if y_control is None:
                continue
            
            slope = _trend_slope(y_control)
            if np.isfinite(slope):
                slopes_control.append(float(slope))
        
        if len(slopes_control) < self.config.pretrend_min_samples:
            return 1, float(slope_trial), np.nan, np.nan, "too_few_control_trends"
        
        slope_controls_mean = float(np.mean(slopes_control))
        diff_abs = float(abs(slope_trial - slope_controls_mean))
        
        # Scale by target variability
        scale = float(np.nanstd(y))
        if not np.isfinite(scale) or scale < 1e-6:
            scale = 1.0
        
        per_day_variability = scale / float(days)
        threshold = self.config.pretrend_max_diff * per_day_variability
        
        pass_flag = int(diff_abs <= threshold)
        
        logger.debug(
            f"Pre-trend check: trial_slope={slope_trial:.4f}, "
            f"control_mean={slope_controls_mean:.4f}, "
            f"diff={diff_abs:.4f}, threshold={threshold:.4f}, pass={pass_flag}"
        )
        
        return pass_flag, float(slope_trial), slope_controls_mean, diff_abs, ""


# ============================================================================
# COUNTERFACTUAL MATCHER
# ============================================================================
class CounterfactualMatcher:
    """Matches trial observations to historical controls"""
    
    def __init__(self, config: Level32Config):
        self.config = config
    
    def match(self,
              df: pd.DataFrame,
              t_idx: int,
              covariates: List[str],
              date_col: str) -> Tuple[List[int], Dict[str, Any]]:
        """
        Find matched control observations from past data
        
        Args:
            df: DataFrame with data
            t_idx: Trial time index
            covariates: List of covariate columns to match on
            date_col: Date column name
            
        Returns:
            Tuple of (control_indices, metadata)
        """
        # Get past observations
        past = _past_indices(
            df, t_idx, date_col,
            self.config.lookback_days,
            self.config.lookback_rows
        )
        
        if len(past) == 0:
            return [], {"reason": "no_past", "n_candidates": 0}
        
        # Build covariate matrix
        X_cols = []
        for cov in covariates:
            if cov not in df.columns:
                continue
            
            col_values = pd.to_numeric(df[cov], errors="coerce").to_numpy(dtype=float)
            X_cols.append(_robust_scale(col_values))
        
        if len(X_cols) == 0:
            return [], {"reason": "no_covariates", "n_candidates": len(past)}
        
        X = np.vstack(X_cols).T
        x_trial = X[t_idx]
        
        # Check trial observation completeness
        if not np.all(np.isfinite(x_trial)):
            return [], {
                "reason": "trial_cov_missing",
                "n_candidates": len(past),
                "n_covariates": len(X_cols)
            }
        
        # Calculate distances to all past observations
        distances = []
        for idx in past:
            x_control = X[idx]
            if not np.all(np.isfinite(x_control)):
                continue
            
            # Mean absolute distance
            dist = float(np.mean(np.abs(x_control - x_trial)))
            distances.append((int(idx), dist))
        
        # Sort by distance
        distances.sort(key=lambda x: x[1])
        
        # Filter by max distance threshold
        distances = [
            (idx, dist) for (idx, dist) in distances
            if dist <= self.config.match_dist_max
        ]
        
        if len(distances) < self.config.min_matched:
            return [], {
                "reason": "too_few_matches",
                "n_candidates": len(distances),
                "min_required": self.config.min_matched
            }
        
        # Select top K controls
        chosen = distances[:self.config.k_controls]
        control_indices = [idx for (idx, _) in chosen]
        avg_dist = float(np.mean([dist for (_, dist) in chosen]))
        
        logger.debug(
            f"Matched {len(control_indices)} controls "
            f"(avg_dist={avg_dist:.3f}, candidates={len(distances)})"
        )
        
        return control_indices, {
            "reason": "",
            "n_candidates": len(distances),
            "avg_dist_top": avg_dist,
            "n_covariates": len(X_cols)
        }
    
    def compute_z_score(self,
                       df: pd.DataFrame,
                       t_idx: int,
                       controls: List[int],
                       outcome_col: str) -> Tuple[float, Dict[str, Any]]:
        """
        Compute z-score of trial outcome vs control distribution
        
        Args:
            df: DataFrame with data
            t_idx: Trial time index
            controls: List of control indices
            outcome_col: Outcome column name
            
        Returns:
            Tuple of (z_score, metadata)
        """
        y = pd.to_numeric(df[outcome_col], errors="coerce").to_numpy(dtype=float)
        
        # Check trial outcome
        if t_idx < 0 or t_idx >= len(y) or not np.isfinite(y[t_idx]):
            return np.nan, {
                "reason": "missing_outcome_at_trial",
                "outcome": outcome_col
            }
        
        # Get control outcomes
        y_controls = y[controls]
        y_controls = y_controls[np.isfinite(y_controls)]
        
        if len(y_controls) < 2:
            return np.nan, {
                "reason": "too_few_control_outcomes",
                "n_controls": len(y_controls)
            }
        
        # Calculate statistics
        mu = float(np.mean(y_controls))
        sd = float(np.std(y_controls))
        
        if not np.isfinite(sd) or sd < 1e-6:
            return np.nan, {
                "reason": "control_sd_zero",
                "mu": mu,
                "n_controls": len(y_controls)
            }
        
        # Compute z-score with clipping
        z = float((y[t_idx] - mu) / sd)
        z = float(np.clip(z, -self.config.z_clip, self.config.z_clip))
        
        logger.debug(
            f"Z-score for {outcome_col}: {z:.2f} "
            f"(trial={y[t_idx]:.2f}, control: μ={mu:.2f}, σ={sd:.2f})"
        )
        
        return z, {
            "mu": mu,
            "sd": sd,
            "n": len(y_controls),
            "value_trial": float(y[t_idx])
        }


# ============================================================================
# TRIAL ENRICHER
# ============================================================================
class TrialEnricher:
    """Enriches trial data with counterfactual analysis"""
    
    def __init__(self, config: Level32Config, paths: PathManager):
        self.config = config
        self.paths = paths
        self.matcher = CounterfactualMatcher(config)
        self.propensity_calc = PropensityScoreCalculator(config)
        self.pretrend_checker = PreTrendChecker(config)
    
    def enrich_trials(self,
                     df_trials: pd.DataFrame,
                     df_data: pd.DataFrame,
                     covariates: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Enrich trial data with counterfactual matching and causal checks
        
        Args:
            df_trials: DataFrame with trial records
            df_data: DataFrame with full dataset
            covariates: List of covariate columns
            
        Returns:
            Tuple of (enriched_trials_df, ledger_df)
        """
        logger.info("=== Enriching Trials with Counterfactual Analysis ===")
        
        # Compute propensity scores if enabled
        propensity = np.full(len(df_data), np.nan, dtype=float)
        if self.config.enable_propensity:
            propensity = self.propensity_calc.compute(
                df_data,
                self.config.propensity_action_col,
                covariates
            )
            df_data["__propensity__"] = propensity
        
        trial_rows = []
        ledger_rows = []
        
        for idx, trial in df_trials.iterrows():
            enriched = self._enrich_single_trial(
                trial, df_data, covariates, propensity
            )
            
            if enriched:
                trial_rows.append(enriched["trial"])
                ledger_rows.append(enriched["ledger"])
        
        df_enriched = pd.DataFrame(trial_rows)
        df_ledger = pd.DataFrame(ledger_rows)
        
        logger.info(f"Enriched {len(df_enriched)} trials")
        
        if len(df_enriched) > 0 and "eligible_flag" in df_enriched.columns:
            n_excluded = int(
                (df_enriched["eligible_flag"] == 0).sum()
            )
            logger.info(f"Trials excluded by causal checks: {n_excluded}")
        
        return df_enriched, df_ledger
    
    def _enrich_single_trial(self,
                            trial: pd.Series,
                            df_data: pd.DataFrame,
                            covariates: List[str],
                            propensity: np.ndarray) -> Optional[Dict[str, Any]]:
        """Enrich a single trial observation"""
        
        insight_id = _as_str(trial.get("insight_id", "")).strip()
        t_idx_raw = _safe_float(trial.get("t_index", np.nan), np.nan)
        
        if not insight_id or not np.isfinite(t_idx_raw):
            logger.debug(f"Skipping invalid trial: id={insight_id}, t_index={t_idx_raw}")
            return None
        
        t_idx = int(t_idx_raw)
        if t_idx < 0 or t_idx >= len(df_data):
            logger.debug(f"Trial index out of bounds: {t_idx}")
            return None
        
        # Match controls
        controls, match_meta = self.matcher.match(
            df_data, t_idx, covariates, self.config.date_col
        )
        
        # Compute counterfactual z-score for main outcome
        z_cf = np.nan
        z_meta = {}
        if controls:
            z_cf, z_meta = self.matcher.compute_z_score(
                df_data, t_idx, controls, self.config.target_col
            )
        
        success_flag = np.nan
        if np.isfinite(z_cf):
            success_flag = int(z_cf >= self.config.z_success)
        
        # Propensity check
        propensity_t = np.nan
        propensity_c_mean = np.nan
        propensity_pass = 1
        
        if self.config.enable_propensity and controls:
            propensity_t, propensity_c_mean, propensity_pass = self._check_propensity(
                propensity, t_idx, controls
            )
        
        # Pre-trend check
        pretrend_pass = 1
        pretrend_slope_trial = np.nan
        pretrend_slope_ctrl = np.nan
        pretrend_diff = np.nan
        pretrend_reason = ""
        
        if self.config.enable_pretrend_check and controls:
            (pretrend_pass, pretrend_slope_trial, pretrend_slope_ctrl,
             pretrend_diff, pretrend_reason) = self.pretrend_checker.check(
                df_data, t_idx, controls, self.config.target_col
            )
        
        # Eligibility flag
        eligible_flag = int(
            len(controls) > 0 and 
            propensity_pass == 1 and 
            pretrend_pass == 1
        )
        
        # Negative control check
        z_negctrl = np.nan
        success_flag_negctrl = np.nan
        
        if (self.config.negctrl_enable and controls and 
            self.config.negctrl_outcome_col in df_data.columns):
            z_negctrl, _ = self.matcher.compute_z_score(
                df_data, t_idx, controls, self.config.negctrl_outcome_col
            )
            if np.isfinite(z_negctrl):
                success_flag_negctrl = int(abs(z_negctrl) >= self.config.z_success)
        
        # Build enriched trial record
        trial_record = {
            "insight_id": insight_id,
            "t_index": t_idx,
            "date": _as_str(trial.get("date", "")),
            "action_name": _as_str(trial.get("action_name", "")),
            "adherence_flag": _safe_float(trial.get("adherence_flag", np.nan), np.nan),
            "z_cf": float(z_cf) if np.isfinite(z_cf) else np.nan,
            "success_flag": success_flag,
            "eligible_flag": eligible_flag,
            
            # Propensity
            "propensity_treated": float(propensity_t) if np.isfinite(propensity_t) else np.nan,
            "propensity_controls_mean": float(propensity_c_mean) if np.isfinite(propensity_c_mean) else np.nan,
            "propensity_pass": propensity_pass,
            
            # Pre-trend
            "pretrend_pass": pretrend_pass,
            "pretrend_slope_trial": float(pretrend_slope_trial) if np.isfinite(pretrend_slope_trial) else np.nan,
            "pretrend_slope_controls_mean": float(pretrend_slope_ctrl) if np.isfinite(pretrend_slope_ctrl) else np.nan,
            "pretrend_diff_abs": float(pretrend_diff) if np.isfinite(pretrend_diff) else np.nan,
            "pretrend_reason": pretrend_reason,
            
            # Negative control
            "z_negctrl": float(z_negctrl) if np.isfinite(z_negctrl) else np.nan,
            "success_flag_negctrl": success_flag_negctrl,
            
            # Matching metadata
            "matched_n": len(controls),
            "baseline_mu": float(z_meta.get("mu", np.nan)) if z_meta else np.nan,
            "baseline_sd": float(z_meta.get("sd", np.nan)) if z_meta else np.nan,
            "match_reason": _as_str(match_meta.get("reason", "")),
            "match_avg_dist_top": float(match_meta.get("avg_dist_top", np.nan)) if match_meta else np.nan,
            "covariates_used": "|".join(covariates),
            "controls_idx": "|".join([str(i) for i in controls[:self.config.k_controls]]),
        }
        
        # Ledger record (audit trail)
        ledger_record = {
            "insight_id": insight_id,
            "t_index": t_idx,
            "matched_n": len(controls),
            "controls_idx": "|".join([str(i) for i in controls[:self.config.k_controls]]),
            "avg_dist_top": float(match_meta.get("avg_dist_top", np.nan)) if match_meta else np.nan,
            "z_cf": float(z_cf) if np.isfinite(z_cf) else np.nan,
            "success_flag": success_flag,
            "eligible_flag": eligible_flag,
            "propensity_treated": float(propensity_t) if np.isfinite(propensity_t) else np.nan,
            "propensity_controls_mean": float(propensity_c_mean) if np.isfinite(propensity_c_mean) else np.nan,
            "propensity_pass": propensity_pass,
            "pretrend_pass": pretrend_pass,
            "pretrend_slope_trial": float(pretrend_slope_trial) if np.isfinite(pretrend_slope_trial) else np.nan,
            "pretrend_slope_controls_mean": float(pretrend_slope_ctrl) if np.isfinite(pretrend_slope_ctrl) else np.nan,
            "pretrend_diff_abs": float(pretrend_diff) if np.isfinite(pretrend_diff) else np.nan,
            "z_negctrl": float(z_negctrl) if np.isfinite(z_negctrl) else np.nan,
            "success_flag_negctrl": success_flag_negctrl,
        }
        
        return {"trial": trial_record, "ledger": ledger_record}
    
    def _check_propensity(self,
                         propensity: np.ndarray,
                         t_idx: int,
                         controls: List[int]) -> Tuple[float, float, int]:
        """Check propensity balance between treated and controls"""
        
        if not (0 <= t_idx < len(propensity)):
            return np.nan, np.nan, 1
        
        p_t = propensity[t_idx]
        if not np.isfinite(p_t):
            return np.nan, np.nan, 1
        
        p_controls = propensity[np.asarray(controls, dtype=int)]
        p_controls = p_controls[np.isfinite(p_controls)]
        
        if len(p_controls) == 0:
            return float(p_t), np.nan, 1
        
        p_c_mean = float(np.mean(p_controls))
        diff = abs(p_t - p_c_mean)
        pass_flag = int(diff <= self.config.propensity_max_diff)
        
        logger.debug(
            f"Propensity check: treated={p_t:.3f}, "
            f"controls_mean={p_c_mean:.3f}, diff={diff:.3f}, pass={pass_flag}"
        )
        
        return float(p_t), p_c_mean, pass_flag


# ============================================================================
# INSIGHT AGGREGATOR
# ============================================================================
class InsightAggregator:
    """Aggregates trial results per insight"""
    
    def __init__(self, config: Level32Config):
        self.config = config
    
    def aggregate(self,
                  df_trials: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate trials by insight ID
        
        Args:
            df_trials: Enriched trials DataFrame
            
        Returns:
            DataFrame with per-insight statistics
        """
        logger.info("=== Aggregating Results by Insight ===")
        
        if len(df_trials) == 0:
            return self._create_empty_insights()
        
        insights = []
        
        for insight_id, group in df_trials.groupby("insight_id"):
            # Filter to eligible trials with valid z-scores
            eligible = group[
                (pd.to_numeric(group["z_cf"], errors="coerce").notna()) &
                (pd.to_numeric(group.get("eligible_flag", 1), errors="coerce")
                 .fillna(1).astype(int) == 1)
            ].copy()
            
            n_trials = len(eligible)
            if n_trials == 0:
                continue
            
            # Main outcome statistics
            n_wins = int(
                (pd.to_numeric(eligible["success_flag"], errors="coerce")
                 .fillna(0) > 0).sum()
            )
            
            success_lb = _sr_lower_bound(n_wins, n_trials)
            
            z_values = pd.to_numeric(eligible["z_cf"], errors="coerce").to_numpy(dtype=float)
            avg_z = float(np.nanmean(z_values))
            
            # Negative control statistics
            negctrl_lb = self._compute_negctrl_lb(group)
            negctrl_pass = self._check_negctrl_pass(negctrl_lb)
            
            # Determine status
            status = self._determine_status(n_trials, success_lb, negctrl_pass)
            
            insights.append({
                "insight_id": _as_str(insight_id),
                "n_trials": n_trials,
                "n_wins": n_wins,
                "success_rate_lb": float(success_lb),
                "avg_z_cf": float(avg_z) if np.isfinite(avg_z) else np.nan,
                "negctrl_success_lb": float(negctrl_lb) if np.isfinite(negctrl_lb) else np.nan,
                "negctrl_pass": negctrl_pass,
                "status": status,
            })
        
        df_insights = pd.DataFrame(insights)
        logger.info(f"Aggregated {len(df_insights)} insights")
        
        return df_insights
    
    def _create_empty_insights(self) -> pd.DataFrame:
        """Create empty insights DataFrame"""
        return pd.DataFrame(columns=[
            "insight_id", "n_trials", "n_wins", "success_rate_lb",
            "avg_z_cf", "negctrl_success_lb", "negctrl_pass", "status"
        ])
    
    def _compute_negctrl_lb(self, group: pd.DataFrame) -> float:
        """Compute negative control lower bound"""
        if not self.config.negctrl_enable:
            return np.nan
        
        if "success_flag_negctrl" not in group.columns:
            return np.nan
        
        negctrl_group = group[
            pd.to_numeric(group["success_flag_negctrl"], errors="coerce").notna()
        ].copy()
        
        n = len(negctrl_group)
        if n == 0:
            return np.nan
        
        wins = int(
            (pd.to_numeric(negctrl_group["success_flag_negctrl"], errors="coerce")
             .fillna(0) > 0).sum()
        )
        
        return _sr_lower_bound(wins, n)
    
    def _check_negctrl_pass(self, negctrl_lb: float) -> int:
        """Check if negative control passes threshold"""
        if not self.config.negctrl_enable:
            return 1
        
        if not np.isfinite(negctrl_lb):
            return 1
        
        return int(negctrl_lb <= self.config.negctrl_max_success_lb)
    
    def _determine_status(self,
                         n_trials: int,
                         success_lb: float,
                         negctrl_pass: int) -> str:
        """Determine insight validation status"""
        if (n_trials >= self.config.min_trials and
            success_lb >= self.config.min_success_lb and
            negctrl_pass == 1):
            return "action_supported"
        
        return "candidate"


# ============================================================================
# MAIN ENGINE
# ============================================================================
class Level32Engine:
    """Main engine for Level 3.2 counterfactual validation"""
    
    def __init__(self, config: Level32Config, paths: PathManager):
        self.config = config
        self.paths = paths
        self.enricher = TrialEnricher(config, paths)
        self.aggregator = InsightAggregator(config)
    
    def run(self, data_path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Run complete Level 3.2 validation pipeline
        
        Args:
            data_path: Optional custom data file path
            
        Returns:
            Tuple of (insights_l3_df, trials_enriched_df, ledger_df)
        """
        logger.info("="*70)
        logger.info("PCB LEVEL 3.2: COUNTERFACTUAL CAUSAL VALIDATION")
        logger.info("="*70)
        
        # Check required inputs
        if not self.paths.insights_l2.exists():
            raise FileNotFoundError(
                f"Missing {self.paths.insights_l2}. "
                "Run Level 2.5 first to generate insights."
            )
        
        trials_path = self.paths.get_trials_path()
        if not trials_path.exists():
            logger.warning("No experiment trials found - creating empty outputs")
            return self._create_empty_outputs()
        
        # Load data
        logger.info(f"Loading insights from: {self.paths.insights_l2}")
        df_insights = pd.read_csv(self.paths.insights_l2)
        
        logger.info(f"Loading trials from: {trials_path}")
        df_trials = pd.read_csv(trials_path)
        
        data_file = self.paths.get_data_path(data_path)
        logger.info(f"Loading data from: {data_file}")
        df_data = pd.read_csv(data_file)
        
        # Prepare data
        df_data = _try_parse_date(df_data, self.config.date_col)
        df_data = _ensure_calendar_covs(df_data, self.config.date_col)
        
        if self.config.target_col not in df_data.columns:
            raise ValueError(
                f"Target column '{self.config.target_col}' not found in data"
            )
        
        # Ensure previous-target covariate (regression-to-mean control)
        df_data = self._ensure_target_prev(df_data)
        
        # Select covariates
        covariates = self._select_covariates(df_data)
        logger.info(f"Selected {len(covariates)} covariates: {', '.join(covariates)}")
        
        # Derive action indicator
        df_data = self._derive_action_indicator(df_data, df_trials)
        
        # Enrich trials
        df_trials_enriched, df_ledger = self.enricher.enrich_trials(
            df_trials, df_data, covariates
        )
        
        # Aggregate insights
        df_insights_l3 = self.aggregator.aggregate(df_trials_enriched)
        
        # Save outputs
        self._save_outputs(df_insights_l3, df_trials_enriched, df_ledger)
        
        # Print summary
        self._print_summary(df_insights_l3, df_trials_enriched)
        
        return df_insights_l3, df_trials_enriched, df_ledger
    
    def _ensure_target_prev(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure previous-target covariate exists"""
        result = df.copy()
        
        if "target_prev" not in result.columns:
            if "mood_prev" in result.columns:
                result["target_prev"] = result["mood_prev"]
            else:
                result["mood_prev"] = pd.to_numeric(
                    result[self.config.target_col], errors="coerce"
                ).astype(float).shift(1)
                result["target_prev"] = result["mood_prev"]
        
        return result
    
    def _select_covariates(self, df: pd.DataFrame) -> List[str]:
        """Select covariates for matching"""
        covs = []
        
        # Previous target (most important)
        if "target_prev" in df.columns:
            covs.append("target_prev")
        elif "mood_prev" in df.columns:
            covs.append("mood_prev")
        
        # Calendar/trend covariates
        for col in ["time_idx", "dow_sin", "dow_cos"]:
            if col in df.columns and col not in covs:
                covs.append(col)
        
        # Additional numeric covariates
        exclude = [self.config.target_col, self.config.date_col] + covs
        numeric = _numeric_cols(
            df, exclude, self.config.min_numeric_threshold
        )
        
        for col in numeric:
            if col not in covs:
                covs.append(col)
            if len(covs) >= self.config.max_covariates:
                break
        
        return covs
    
    def _derive_action_indicator(self,
                                 df: pd.DataFrame,
                                 df_trials: pd.DataFrame) -> pd.DataFrame:
        """Derive binary action indicator from trials"""
        result = df.copy()
        
        if self.config.propensity_action_col not in result.columns:
            result[self.config.propensity_action_col] = 0
            
            trial_indices = pd.to_numeric(
                df_trials.get("t_index"), errors="coerce"
            ).fillna(-999).astype(int)
            
            for t_idx in trial_indices:
                if 0 <= t_idx < len(result):
                    result.loc[t_idx, self.config.propensity_action_col] = 1
        
        return result
    
    def _save_outputs(self,
                     df_insights: pd.DataFrame,
                     df_trials: pd.DataFrame,
                     df_ledger: pd.DataFrame) -> None:
        """Save all output files"""
        df_insights.to_csv(self.paths.out_l3, index=False)
        df_trials.to_csv(self.paths.out_trials, index=False)
        df_ledger.to_csv(self.paths.out_ledger, index=False)
        
        logger.info(f"Saved: {self.paths.out_l3}")
        logger.info(f"Saved: {self.paths.out_trials}")
        logger.info(f"Saved: {self.paths.out_ledger}")
    
    def _print_summary(self,
                      df_insights: pd.DataFrame,
                      df_trials: pd.DataFrame) -> None:
        """Print execution summary"""
        print("\n" + "="*70)
        print("LEVEL 3.2 SUMMARY")
        print("="*70)
        print(f"Trials enriched: {len(df_trials)}")
        print(f"Insights evaluated: {len(df_insights)}")
        
        if len(df_insights) > 0:
            n_supported = (df_insights["status"] == "action_supported").sum()
            print(f"Actions supported: {n_supported}")
            
            if n_supported > 0:
                print("\nSupported insights:")
                supported = df_insights[df_insights["status"] == "action_supported"]
                print(supported[["insight_id", "n_trials", "success_rate_lb", "avg_z_cf"]].to_string(index=False))
        
        print("="*70 + "\n")
    
    def _create_empty_outputs(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create empty output files when no trials exist"""
        empty_trials = pd.DataFrame(columns=[
            "insight_id", "action_name", "date", "t_index",
            "adherence_flag", "dose", "notes"
        ])
        empty_trials.to_csv(self.paths.out_trials, index=False)
        
        empty_insights = pd.DataFrame(columns=[
            "insight_id", "n_trials", "n_wins", "success_rate_lb",
            "avg_z_cf", "negctrl_success_lb", "negctrl_pass", "status"
        ])
        empty_insights.to_csv(self.paths.out_l3, index=False)
        
        empty_ledger = pd.DataFrame(columns=[
            "insight_id", "t_index", "matched_n"
        ])
        empty_ledger.to_csv(self.paths.out_ledger, index=False)
        
        logger.warning("No trials found - created empty outputs")
        print("\nNo experiment_results.csv found.")
        print("Level 3.2 requires at least 1 logged trial.")
        print("\nNext step: Log a trial using:")
        print("  python pcb_experiments_level29.py log --insight_id I2-00001")
        print("\nThen rerun Level 3.2.")
        
        return empty_insights, empty_trials, empty_ledger


# ============================================================================
# CLI
# ============================================================================
def build_argparser() -> argparse.ArgumentParser:
    """Build CLI argument parser"""
    parser = argparse.ArgumentParser(
        prog="pcb_level3_engine_32.py",
        description="PCB Level 3.2 — Counterfactual causal validation (local-first)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--data",
        default=None,
        help="Custom data file path (default: data.csv or out/data_clean.csv)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """
    Main entry point
    
    Args:
        argv: Command line arguments
        
    Returns:
        Exit code (0=success, non-zero=error)
    """
    if argv is None:
        argv = sys.argv[1:]
    
    parser = build_argparser()
    args = parser.parse_args(argv)
    
    # Setup logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    
    try:
        # Load configuration
        config = Level32Config.load_from_file()
        paths = PathManager(config)
        
        # Run engine
        engine = Level32Engine(config, paths)
        engine.run(data_path=args.data)
        
        logger.info("Level 3.2 completed successfully")
        return 0
        
    except KeyboardInterrupt:
        logger.warning("\nOperation cancelled by user")
        return 130
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
