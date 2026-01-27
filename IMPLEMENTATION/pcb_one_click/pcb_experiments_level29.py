# FILE: pcb_experiments_level29.py
# Python 3.7+ compatible
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
# Dependencies: numpy, pandas

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Tuple, List, Any
from functools import lru_cache

import numpy as np
import pandas as pd

# ============================================================================
# LOGGING SETUP
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================
class ExperimentConfig:
    """Configurazione centralizzata per esperimenti PCB"""
    
    # Directory paths
    OUT_DIR = "out"
    
    # File names
    INSIGHTS_L2_FILE = "insights_level2.csv"
    EXP_PLAN_FILE = "experiment_plan.csv"
    EXP_RESULTS_FILE = "experiment_results.csv"
    EXP_SUMMARY_FILE = "experiment_summary_level29.csv"
    EXP_SUMMARY_JSONL_FILE = "experiment_summary_level29.jsonl"
    
    # Data files
    DEFAULT_DATA_CSV = "data.csv"
    FALLBACK_DATA_CSV = "demo_data.csv"
    
    # Column names
    DATE_COL = "date"
    TARGET_COL = "target"
    
    # Plan defaults
    DEFAULT_WINDOW_DAYS = 1
    DEFAULT_COST = 1.0
    DEFAULT_DOSE = ""
    DEFAULT_NOTES = ""
    
    # Eval parameters
    LOOKBACK_DAYS = 60
    LOOKBACK_ROWS = 60
    HARD_WEEKDAY_MATCH = True
    MIN_BASELINE_N = 10
    Z_CLIP = 6.0
    Z_SUCCESS_THRESH = 0.2
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """Carica configurazione da dizionario"""
        for key, value in config_dict.items():
            key_upper = key.upper()
            if hasattr(cls, key_upper):
                setattr(cls, key_upper, value)
                logger.debug(f"Config override: {key_upper} = {value}")
        return cls
    
    @classmethod
    def load_from_file(cls, config_path: str = "pcb_config.py") -> 'ExperimentConfig':
        """Carica configurazione da file esterno"""
        try:
            from pcb_config import load_config
            config_dict = load_config()
            return cls.from_dict(config_dict)
        except ImportError:
            logger.debug("No external config file found, using defaults")
            return cls
        except Exception as e:
            logger.warning(f"Error loading config file: {e}, using defaults")
            return cls


# ============================================================================
# PATH MANAGER
# ============================================================================
class PathManager:
    """Gestisce tutti i path del progetto"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.base = Path(config.OUT_DIR)
        self.base.mkdir(exist_ok=True)
    
    @property
    def insights_l2(self) -> Path:
        return self.base / self.config.INSIGHTS_L2_FILE
    
    @property
    def exp_plan(self) -> Path:
        return self.base / self.config.EXP_PLAN_FILE
    
    @property
    def exp_results(self) -> Path:
        return self.base / self.config.EXP_RESULTS_FILE
    
    @property
    def exp_summary(self) -> Path:
        return self.base / self.config.EXP_SUMMARY_FILE
    
    @property
    def exp_summary_jsonl(self) -> Path:
        return self.base / self.config.EXP_SUMMARY_JSONL_FILE
    
    def get_data_path(self, custom_path: Optional[str] = None) -> Path:
        """Risolve il path del file dati"""
        if custom_path and Path(custom_path).exists():
            return Path(custom_path)
        
        default_path = Path(self.config.DEFAULT_DATA_CSV)
        if default_path.exists():
            return default_path
        
        fallback_path = self.base / self.config.FALLBACK_DATA_CSV
        if fallback_path.exists():
            return fallback_path
        
        raise FileNotFoundError(
            f"Data file not found. Tried: {custom_path}, "
            f"{default_path}, {fallback_path}"
        )
    
    def ensure_exists(self, path: Path) -> Path:
        """Verifica che un path esista"""
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")
        return path


# ============================================================================
# VALIDATION UTILITIES
# ============================================================================
def validate_dataframe(df: pd.DataFrame, 
                       required_cols: List[str],
                       name: str = "DataFrame") -> None:
    """
    Valida che un DataFrame abbia le colonne richieste
    
    Args:
        df: DataFrame da validare
        required_cols: Lista di colonne richieste
        name: Nome del DataFrame per messaggi di errore
        
    Raises:
        ValueError: Se mancano colonne o DataFrame è vuoto
    """
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")
    
    if len(df) == 0:
        logger.warning(f"{name} is empty")


def validate_date_column(df: pd.DataFrame, 
                        date_col: str) -> None:
    """
    Valida che la colonna date sia in formato datetime
    
    Args:
        df: DataFrame da validare
        date_col: Nome della colonna date
        
    Raises:
        ValueError: Se colonna mancante o tipo non corretto
    """
    if date_col not in df.columns:
        raise ValueError(f"Missing date column: {date_col}")
    
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        raise ValueError(
            f"Column {date_col} is not datetime type. "
            f"Current type: {df[date_col].dtype}"
        )


# ============================================================================
# DATA UTILITIES
# ============================================================================
def _safe_float(x: Any, default: float = np.nan) -> float:
    """
    Converte un valore in float in modo sicuro
    
    Args:
        x: Valore da convertire
        default: Valore di default se conversione fallisce
        
    Returns:
        float: Valore convertito o default
    """
    try:
        v = float(x)
        return v if np.isfinite(v) else float(default)
    except (TypeError, ValueError):
        return float(default)


def _as_str(x: Any) -> str:
    """
    Converte un valore in stringa in modo sicuro
    
    Args:
        x: Valore da convertire
        
    Returns:
        str: Valore convertito o stringa vuota
    """
    try:
        return str(x) if x is not None else ""
    except Exception:
        return ""


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    """
    Salva DataFrame come CSV
    
    Args:
        df: DataFrame da salvare
        path: Path del file di destinazione
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info(f"Saved CSV: {path} ({len(df)} rows)")
    except Exception as e:
        logger.error(f"Error saving CSV to {path}: {e}")
        raise


def _save_jsonl(df: pd.DataFrame, path: Path) -> None:
    """
    Salva DataFrame come JSONL
    
    Args:
        df: DataFrame da salvare
        path: Path del file di destinazione
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
        logger.info(f"Saved JSONL: {path} ({len(df)} rows)")
    except Exception as e:
        logger.error(f"Error saving JSONL to {path}: {e}")
        raise


def _try_parse_date(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """
    Tenta di parsare la colonna date come datetime
    
    Args:
        df: DataFrame da processare
        date_col: Nome della colonna date
        
    Returns:
        DataFrame con colonna date convertita se possibile
    """
    if date_col not in df.columns:
        logger.warning(f"Date column '{date_col}' not found")
        return df
    
    dt = pd.to_datetime(df[date_col], errors="coerce")
    valid_count = dt.notna().sum()
    threshold = max(5, int(0.2 * len(df)))
    
    if valid_count >= threshold:
        result = df.copy()
        result[date_col] = dt
        logger.info(f"Parsed {valid_count}/{len(df)} dates successfully")
        return result
    else:
        logger.warning(
            f"Only {valid_count}/{len(df)} valid dates found "
            f"(threshold: {threshold}), keeping original format"
        )
        return df


def _has_date(df: pd.DataFrame, date_col: str) -> bool:
    """
    Verifica se DataFrame ha colonna date valida
    
    Args:
        df: DataFrame da verificare
        date_col: Nome della colonna date
        
    Returns:
        bool: True se colonna date è presente e in formato datetime
    """
    return (date_col in df.columns and 
            pd.api.types.is_datetime64_any_dtype(df[date_col]))


# ============================================================================
# BASELINE CALCULATOR
# ============================================================================
class BaselineCalculator:
    """Gestisce calcolo baseline e z-scores"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
    
    def _weekday_series(self, df: pd.DataFrame) -> pd.Series:
        """
        Ottiene serie dei giorni della settimana
        
        Args:
            df: DataFrame con colonna date
            
        Returns:
            Series con weekday (0=lunedì, 6=domenica)
        """
        if _has_date(df, self.config.DATE_COL):
            return df[self.config.DATE_COL].dt.weekday
        # Fallback: pattern ciclico
        return pd.Series(np.arange(len(df)) % 7, index=df.index)
    
    def _past_window_indices(self, df: pd.DataFrame, 
                            t_idx: int) -> np.ndarray:
        """
        Ottiene indici della finestra temporale passata
        
        Args:
            df: DataFrame con dati
            t_idx: Indice temporale di riferimento
            
        Returns:
            Array di indici della finestra baseline
        """
        n = len(df)
        if t_idx <= 0 or t_idx >= n:
            return np.array([], dtype=int)
        
        # Se abbiamo date valide, usa lookback basato su date
        if _has_date(df, self.config.DATE_COL):
            d_t = df[self.config.DATE_COL].iloc[t_idx]
            if pd.notna(d_t):
                d0 = d_t.normalize() - pd.Timedelta(days=self.config.LOOKBACK_DAYS)
                mask = (df[self.config.DATE_COL] < d_t) & (df[self.config.DATE_COL] >= d0)
                indices = df.index[mask].to_numpy(dtype=int)
                logger.debug(
                    f"Date-based window: {len(indices)} points "
                    f"from {d0.date()} to {d_t.date()}"
                )
                return indices
        
        # Fallback: lookback basato su numero di righe
        start = max(0, t_idx - self.config.LOOKBACK_ROWS)
        indices = np.arange(start, t_idx, dtype=int)
        logger.debug(f"Row-based window: {len(indices)} points (rows {start} to {t_idx})")
        return indices
    
    def calculate_z_score(self, df: pd.DataFrame, 
                         t_idx: int, 
                         baseline_col: str,
                         baseline_exclude_mask: Optional[np.ndarray] = None
                         ) -> Tuple[float, Dict[str, Any]]:
        """
        Calcola z-score di un valore contro baseline storico
        
        Args:
            df: DataFrame con dati storici
            t_idx: Indice temporale del punto da valutare
            baseline_col: Nome colonna da usare per baseline
            baseline_exclude_mask: Maschera booleana per escludere punti
            
        Returns:
            Tuple[float, Dict]: (z-score, metadata del calcolo)
            
        Notes:
            - Usa solo dati passati (no data leakage)
            - Filtra per stesso giorno della settimana se HARD_WEEKDAY_MATCH
            - Richiede minimo MIN_BASELINE_N punti validi
        """
        # Validazione base
        if baseline_col not in df.columns:
            return np.nan, {"reason": "missing_col", "col": baseline_col}
        
        x = pd.to_numeric(df[baseline_col], errors="coerce").to_numpy(dtype=float)
        
        if t_idx < 0 or t_idx >= len(x) or not np.isfinite(x[t_idx]):
            return np.nan, {"reason": "missing_value_today", "t_idx": t_idx}
        
        # Ottieni finestra baseline
        idx = self._past_window_indices(df, t_idx)
        if idx.size == 0:
            return np.nan, {"reason": "no_past_window"}
        
        # Filtra per stesso giorno della settimana se richiesto
        wd = self._weekday_series(df).to_numpy(dtype=int)
        cand = idx
        
        if self.config.HARD_WEEKDAY_MATCH:
            cand = cand[wd[cand] == int(wd[t_idx])]
            method = "same_weekday"
        else:
            method = "window"
        
        # Applica maschera di esclusione se fornita
        if baseline_exclude_mask is not None:
            ex = np.asarray(baseline_exclude_mask, dtype=bool)
            cand = cand[~ex[cand]]
        
        # Estrai valori validi
        vals = x[cand]
        vals = vals[np.isfinite(vals)]
        
        if len(vals) < self.config.MIN_BASELINE_N:
            return np.nan, {
                "reason": "too_few_baseline",
                "baseline_n": int(len(vals)),
                "min_required": self.config.MIN_BASELINE_N
            }
        
        # Calcola statistiche baseline
        mu = float(np.mean(vals))
        sd = float(np.std(vals))
        
        if not np.isfinite(sd) or sd < 1e-9:
            return np.nan, {
                "reason": "sd_zero",
                "baseline_n": int(len(vals)),
                "mu": mu
            }
        
        # Calcola z-score
        z = float((x[t_idx] - mu) / sd)
        z = float(np.clip(z, -self.config.Z_CLIP, self.config.Z_CLIP))
        
        meta = {
            "baseline_n": int(len(vals)),
            "mu": mu,
            "sd": sd,
            "method": method,
            "value": float(x[t_idx])
        }
        
        logger.debug(
            f"Z-score: {z:.2f} (value={x[t_idx]:.2f}, "
            f"baseline: μ={mu:.2f}, σ={sd:.2f}, n={len(vals)})"
        )
        
        return z, meta


# ============================================================================
# DATE INDEX RESOLVER
# ============================================================================
@lru_cache(maxsize=128)
def _normalize_date_key(date_str: str) -> str:
    """
    Normalizza stringa data in formato ISO
    
    Args:
        date_str: Stringa data da normalizzare
        
    Returns:
        str: Data in formato YYYY-MM-DD o stringa vuota se invalida
    """
    try:
        d = pd.to_datetime(date_str, errors="coerce")
        if pd.isna(d):
            return ""
        return d.normalize().date().isoformat()
    except Exception:
        return ""


def _build_date_to_index_map(df: pd.DataFrame, date_col: str) -> Dict[str, int]:
    """
    Costruisce mappa data -> indice
    
    Args:
        df: DataFrame con colonna date
        date_col: Nome della colonna date
        
    Returns:
        Dict mapping date ISO string -> index
    """
    date_map = {"_len": int(len(df))}
    
    if _has_date(df, date_col):
        for i, d in enumerate(df[date_col]):
            if pd.notna(d):
                key = d.normalize().date().isoformat()
                if key not in date_map:
                    date_map[key] = int(i)
    
    return date_map


def _resolve_t_index(date_str: str, df: pd.DataFrame, date_col: str) -> int:
    """
    Risolve una data in un indice del DataFrame
    
    Args:
        date_str: Stringa data da risolvere
        df: DataFrame di riferimento
        date_col: Nome della colonna date
        
    Returns:
        int: Indice corrispondente o -1 se non trovato
    """
    if not (date_col in df.columns and _has_date(df, date_col)):
        return -1
    
    key = _normalize_date_key(date_str)
    if not key:
        return -1
    
    idx_map = _build_date_to_index_map(df, date_col)
    return int(idx_map.get(key, -1))


# ============================================================================
# EXPERIMENT PLAN GENERATOR
# ============================================================================
class ExperimentPlanGenerator:
    """Genera piano esperimenti da insights"""
    
    def __init__(self, config: ExperimentConfig, paths: PathManager):
        self.config = config
        self.paths = paths
    
    def generate(self) -> pd.DataFrame:
        """
        Genera piano esperimenti da insights_level2.csv
        
        Returns:
            DataFrame con piano esperimenti
            
        Raises:
            FileNotFoundError: Se file insights non esiste
        """
        logger.info("=== Generating Experiment Plan ===")
        
        # Carica insights
        insights_path = self.paths.ensure_exists(self.paths.insights_l2)
        df = pd.read_csv(insights_path)
        logger.info(f"Loaded {len(df)} insights from {insights_path}")
        
        if len(df) == 0:
            logger.warning("No insights found, creating empty plan")
            return self._create_empty_plan()
        
        # Valida colonne richieste
        required_cols = ["insight_id", "source", "target", "lag", 
                        "delta_test", "strength", "recommendation", 
                        "human_statement"]
        for col in required_cols:
            if col not in df.columns:
                df[col] = np.nan
        
        # Filtra per target column
        df = df[df["target"].astype(str) == self.config.TARGET_COL].copy()
        logger.info(f"Filtered to {len(df)} insights for target='{self.config.TARGET_COL}'")
        
        # Genera azioni
        actions = []
        for _, row in df.iterrows():
            action = self._create_action_from_insight(row)
            if action:
                actions.append(action)
        
        if not actions:
            logger.warning("No valid actions generated")
            return self._create_empty_plan()
        
        # Crea DataFrame e rimuovi duplicati
        plan_df = pd.DataFrame(actions)
        plan_df = plan_df.drop_duplicates(
            subset=["insight_id"], 
            keep="first"
        ).reset_index(drop=True)
        
        # Salva
        _save_csv(plan_df, self.paths.exp_plan)
        
        logger.info(f"Generated {len(plan_df)} unique actions")
        return plan_df
    
    def _create_empty_plan(self) -> pd.DataFrame:
        """Crea piano vuoto con colonne corrette"""
        empty = pd.DataFrame(columns=[
            "insight_id", "action_name", "action_type",
            "dose", "window_days", "cost",
            "expected_direction_on_mood",
            "expected_direction_on_target",
            "notes",
        ])
        _save_csv(empty, self.paths.exp_plan)
        return empty
    
    def _create_action_from_insight(self, row: pd.Series) -> Optional[Dict[str, Any]]:
        """
        Crea azione da una riga di insight
        
        Args:
            row: Riga del DataFrame insights
            
        Returns:
            Dict con dati azione o None se invalida
        """
        insight_id = _as_str(row.get("insight_id", "")).strip()
        source = _as_str(row.get("source", "")).strip()
        
        if not insight_id or not source:
            logger.debug(f"Skipping invalid insight: id={insight_id}, source={source}")
            return None
        
        lag = int(_safe_float(row.get("lag", 1), 1))
        delta = _safe_float(row.get("delta_test", np.nan), np.nan)
        
        # Determina direzione aspettata
        exp_dir = np.nan
        if np.isfinite(delta) and delta != 0:
            exp_dir = float(np.sign(delta))
        
        # Determina tipo azione
        if np.isfinite(delta) and delta > 0:
            action_type = "increase"
            action_name = f"increase_{source}"
        elif np.isfinite(delta) and delta < 0:
            action_type = "reduce"
            action_name = f"reduce_{source}"
        else:
            action_type = "adjust"
            action_name = f"adjust_{source}"
        
        return {
            "insight_id": insight_id,
            "action_name": action_name,
            "action_type": action_type,
            "dose": self.config.DEFAULT_DOSE,
            "window_days": self.config.DEFAULT_WINDOW_DAYS,
            "cost": self.config.DEFAULT_COST,
            "expected_direction_on_mood": exp_dir,
            "expected_direction_on_target": exp_dir,
            "notes": f"lag={lag} | source={source}",
        }


# ============================================================================
# EXPERIMENT LOGGER
# ============================================================================
class ExperimentLogger:
    """Gestisce logging dei trial sperimentali"""
    
    def __init__(self, config: ExperimentConfig, paths: PathManager):
        self.config = config
        self.paths = paths
    
    def log_trial(self, 
                  insight_id: str,
                  action_name: str = "",
                  date: str = "",
                  t_index: Optional[int] = None,
                  adherence_flag: int = 1,
                  dose: str = "",
                  notes: str = "",
                  data_path: Optional[str] = None) -> pd.DataFrame:
        """
        Registra un trial sperimentale
        
        Args:
            insight_id: ID dell'insight
            action_name: Nome dell'azione (opzionale, auto-risolto da plan)
            date: Data del trial in formato ISO (opzionale)
            t_index: Indice temporale esplicito (opzionale)
            adherence_flag: 0/1 se azione è stata eseguita
            dose: Descrizione dose/intensità
            notes: Note aggiuntive
            data_path: Path dati personalizzato
            
        Returns:
            DataFrame aggiornato con tutti i trial
        """
        logger.info("=== Logging Experiment Trial ===")
        
        # Validazione
        insight_id = _as_str(insight_id).strip()
        if not insight_id:
            raise ValueError("insight_id is required and cannot be empty")
        
        # Carica dati
        data_file = self.paths.get_data_path(data_path)
        df_data = pd.read_csv(data_file)
        df_data = _try_parse_date(df_data, self.config.DATE_COL)
        logger.info(f"Loaded data from {data_file} ({len(df_data)} rows)")
        
        # Risolvi t_index
        resolved_t_index = self._resolve_t_index(
            t_index, date, df_data
        )
        
        # Risolvi data string
        resolved_date = self._resolve_date_string(
            resolved_t_index, df_data, date
        )
        
        # Risolvi action_name da piano se non fornito
        if not action_name:
            action_name = self._resolve_action_name(insight_id)
        
        # Crea row
        trial_row = {
            "insight_id": insight_id,
            "action_name": _as_str(action_name).strip(),
            "date": resolved_date,
            "t_index": int(resolved_t_index),
            "adherence_flag": int(adherence_flag) if adherence_flag in (0, 1) else 1,
            "dose": _as_str(dose).strip(),
            "notes": _as_str(notes).strip(),
        }
        
        # Carica o crea DataFrame risultati
        if self.paths.exp_results.exists():
            df_results = pd.read_csv(self.paths.exp_results)
        else:
            df_results = pd.DataFrame(columns=list(trial_row.keys()))
        
        # Aggiungi nuovo trial
        df_results = pd.concat(
            [df_results, pd.DataFrame([trial_row])], 
            ignore_index=True
        )
        
        # Salva
        _save_csv(df_results, self.paths.exp_results)
        
        logger.info(f"Logged trial: {trial_row}")
        logger.info(f"Total trials: {len(df_results)}")
        
        return df_results
    
    def _resolve_t_index(self, 
                        t_index: Optional[int],
                        date_str: str,
                        df_data: pd.DataFrame) -> int:
        """Risolve t_index da parametri o date"""
        if t_index is not None and t_index >= 0:
            logger.info(f"Using explicit t_index: {t_index}")
            return int(t_index)
        
        if date_str:
            resolved = _resolve_t_index(date_str, df_data, self.config.DATE_COL)
            if resolved >= 0:
                logger.info(f"Resolved t_index from date '{date_str}': {resolved}")
                return resolved
            else:
                logger.warning(f"Could not resolve date '{date_str}', using last row")
        
        # Default: ultima riga
        last_idx = len(df_data) - 1
        logger.info(f"Using last row as t_index: {last_idx}")
        return last_idx
    
    def _resolve_date_string(self,
                            t_index: int,
                            df_data: pd.DataFrame,
                            provided_date: str) -> str:
        """Risolve stringa data da t_index o parametro"""
        if provided_date:
            return provided_date
        
        if (_has_date(df_data, self.config.DATE_COL) and 
            0 <= t_index < len(df_data)):
            d = df_data[self.config.DATE_COL].iloc[t_index]
            if pd.notna(d):
                return d.normalize().date().isoformat()
        
        return ""
    
    def _resolve_action_name(self, insight_id: str) -> str:
        """Risolve action_name da piano esperimenti"""
        if not self.paths.exp_plan.exists():
            return ""
        
        try:
            df_plan = pd.read_csv(self.paths.exp_plan)
            if "insight_id" in df_plan.columns and "action_name" in df_plan.columns:
                plan_map = dict(zip(
                    df_plan["insight_id"].astype(str),
                    df_plan["action_name"].astype(str)
                ))
                action = _as_str(plan_map.get(insight_id, "")).strip()
                if action:
                    logger.info(f"Resolved action_name from plan: '{action}'")
                return action
        except Exception as e:
            logger.warning(f"Could not load action_name from plan: {e}")
        
        return ""


# ============================================================================
# EXPERIMENT EVALUATOR
# ============================================================================
class ExperimentEvaluator:
    """Valuta risultati esperimenti con baseline comparison"""
    
    def __init__(self, config: ExperimentConfig, paths: PathManager):
        self.config = config
        self.paths = paths
        self.calculator = BaselineCalculator(config)
    
    def evaluate(self, data_path: Optional[str] = None) -> pd.DataFrame:
        """
        Valuta tutti i trial e genera summary
        
        Args:
            data_path: Path dati personalizzato
            
        Returns:
            DataFrame con summary per insight/action
        """
        logger.info("=== Evaluating Experiments ===")
        
        # Verifica esistenza risultati
        if not self.paths.exp_results.exists():
            logger.warning("No experiment results found")
            return self._create_empty_summary()
        
        # Carica risultati
        df_results = pd.read_csv(self.paths.exp_results)
        if len(df_results) == 0 or "insight_id" not in df_results.columns:
            logger.warning("Empty or invalid results file")
            return self._create_empty_summary()
        
        logger.info(f"Loaded {len(df_results)} trials")
        
        # Normalizza colonne
        df_results = self._normalize_results(df_results)
        
        # Carica dati
        data_file = self.paths.get_data_path(data_path)
        df_data = pd.read_csv(data_file)
        df_data = _try_parse_date(df_data, self.config.DATE_COL)
        
        if self.config.TARGET_COL not in df_data.columns:
            raise ValueError(
                f"Target column '{self.config.TARGET_COL}' "
                f"not found in {data_file}"
            )
        
        logger.info(f"Loaded data from {data_file} ({len(df_data)} rows)")
        
        # Calcola z-scores per ogni trial
        df_results_eval = self._evaluate_trials(df_results, df_data)
        
        # Genera summary per insight/action
        summary_df = self._generate_summary(df_results_eval)
        
        # Salva
        _save_csv(summary_df, self.paths.exp_summary)
        _save_jsonl(summary_df, self.paths.exp_summary_jsonl)
        
        # Report
        self._print_report(summary_df)
        
        return summary_df
    
    def _normalize_results(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalizza DataFrame risultati"""
        required_cols = [
            "insight_id", "action_name", "date", "t_index",
            "adherence_flag", "dose", "notes"
        ]
        
        for col in required_cols:
            if col not in df.columns:
                df[col] = np.nan
        
        df["insight_id"] = df["insight_id"].astype(str)
        df["action_name"] = df["action_name"].astype(str).replace({"nan": ""})
        df["t_index"] = pd.to_numeric(df["t_index"], errors="coerce")
        
        return df
    
    def _evaluate_trials(self, 
                        df_results: pd.DataFrame,
                        df_data: pd.DataFrame) -> pd.DataFrame:
        """Calcola z-scores per tutti i trial"""
        zs = []
        success_flags = []
        baseline_ns = []
        baseline_mus = []
        baseline_sds = []
        baseline_methods = []
        debug_reasons = []
        
        logger.info("Calculating z-scores for trials...")
        
        for idx, row in df_results.iterrows():
            t_idx_val = _safe_float(row.get("t_index", np.nan), np.nan)
            
            if not np.isfinite(t_idx_val):
                # Trial invalido
                zs.append(np.nan)
                success_flags.append(np.nan)
                baseline_ns.append(0)
                baseline_mus.append(np.nan)
                baseline_sds.append(np.nan)
                baseline_methods.append("")
                debug_reasons.append("missing_t_index")
                continue
            
            t_idx = int(t_idx_val)
            
            # Calcola z-score
            z, meta = self.calculator.calculate_z_score(
                df_data, 
                t_idx, 
                self.config.TARGET_COL,
                baseline_exclude_mask=None
            )
            
            if np.isfinite(z):
                # Success se z sopra threshold
                success = int(z >= self.config.Z_SUCCESS_THRESH)
                
                zs.append(float(z))
                success_flags.append(success)
                baseline_ns.append(int(meta.get("baseline_n", 0)))
                baseline_mus.append(float(meta.get("mu", np.nan)))
                baseline_sds.append(float(meta.get("sd", np.nan)))
                baseline_methods.append(_as_str(meta.get("method", "")))
                debug_reasons.append("")
            else:
                # Baseline fallito
                zs.append(np.nan)
                success_flags.append(np.nan)
                baseline_ns.append(int(meta.get("baseline_n", 0)))
                baseline_mus.append(float(meta.get("mu", np.nan)))
                baseline_sds.append(float(meta.get("sd", np.nan)))
                baseline_methods.append(_as_str(meta.get("method", "")))
                debug_reasons.append(_as_str(meta.get("reason", "baseline_failed")))
        
        # Aggiungi colonne al DataFrame
        df_eval = df_results.copy()
        df_eval["z_vs_baseline"] = zs
        df_eval["success_flag"] = success_flags
        df_eval["baseline_n"] = baseline_ns
        df_eval["baseline_mu"] = baseline_mus
        df_eval["baseline_sd"] = baseline_sds
        df_eval["baseline_method"] = baseline_methods
        df_eval["debug_reason"] = debug_reasons
        
        valid_z_count = pd.to_numeric(df_eval["z_vs_baseline"], errors="coerce").notna().sum()
        logger.info(f"Computed z-scores: {valid_z_count}/{len(df_eval)} valid")
        
        return df_eval
    
    def _generate_summary(self, df_results_eval: pd.DataFrame) -> pd.DataFrame:
        """Genera summary per insight/action"""
        rows = []
        group_cols = ["insight_id", "action_name"]
        
        for (insight_id, action_name), group in df_results_eval.groupby(group_cols):
            # Filtra trial con z-score valido
            valid_group = group[
                pd.to_numeric(group["z_vs_baseline"], errors="coerce").notna()
            ].copy()
            
            n_trials = len(valid_group)
            if n_trials <= 0:
                continue
            
            # Conta successi
            n_wins = int(np.sum(
                (pd.to_numeric(valid_group["success_flag"], errors="coerce")
                 .fillna(0) > 0).astype(int)
            ))
            
            success_rate = float(n_wins / n_trials) if n_trials > 0 else np.nan
            
            # Calcola statistiche z-score
            z_array = pd.to_numeric(
                valid_group["z_vs_baseline"], 
                errors="coerce"
            ).to_numpy(dtype=float)
            
            avg_z = float(np.nanmean(z_array)) if n_trials > 0 else np.nan
            median_z = float(np.nanmedian(z_array)) if n_trials > 0 else np.nan
            
            # Metodo baseline più recente
            method = _as_str(
                valid_group["baseline_method"].dropna().astype(str).iloc[-1]
            ) if n_trials > 0 else ""
            
            rows.append({
                "insight_id": _as_str(insight_id),
                "action_name": _as_str(action_name),
                "n_trials": int(n_trials),
                "n_wins": int(n_wins),
                "success_rate": float(success_rate) if np.isfinite(success_rate) else np.nan,
                "avg_z": float(avg_z) if np.isfinite(avg_z) else np.nan,
                "median_z": float(median_z) if np.isfinite(median_z) else np.nan,
                "baseline_method": method,
            })
        
        if not rows:
            return self._create_empty_summary()
        
        # Crea DataFrame e ordina per performance
        summary_df = pd.DataFrame(rows)
        summary_df = summary_df.sort_values(
            ["success_rate", "avg_z", "n_trials"],
            ascending=[False, False, False]
        ).reset_index(drop=True)
        
        return summary_df
    
    def _create_empty_summary(self) -> pd.DataFrame:
        """Crea summary vuoto"""
        empty = pd.DataFrame(columns=[
            "insight_id", "action_name",
            "n_trials", "n_wins", "success_rate",
            "avg_z", "median_z",
            "baseline_method",
        ])
        _save_csv(empty, self.paths.exp_summary)
        _save_jsonl(empty, self.paths.exp_summary_jsonl)
        return empty
    
    def _print_report(self, summary_df: pd.DataFrame) -> None:
        """Stampa report summary"""
        print(f"\n{'='*60}")
        print("EXPERIMENT EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total action groups: {len(summary_df)}")
        
        if len(summary_df) > 0:
            show_cols = [
                "insight_id", "action_name", "n_trials",
                "success_rate", "avg_z", "median_z"
            ]
            print("\nTop performing actions:")
            print(summary_df[show_cols].head(10).to_string(index=False))
        
        print(f"\nSaved: {self.paths.exp_summary}")
        print(f"Saved: {self.paths.exp_summary_jsonl}")
        print(f"{'='*60}\n")


# ============================================================================
# CLI COMMANDS
# ============================================================================
def cmd_plan(args, config: ExperimentConfig, paths: PathManager) -> pd.DataFrame:
    """Command: generate experiment plan"""
    try:
        generator = ExperimentPlanGenerator(config, paths)
        return generator.generate()
    except Exception as e:
        logger.error(f"Error in plan command: {e}", exc_info=True)
        raise


def cmd_log(args, config: ExperimentConfig, paths: PathManager) -> pd.DataFrame:
    """Command: log experiment trial"""
    try:
        logger_obj = ExperimentLogger(config, paths)
        return logger_obj.log_trial(
            insight_id=args.insight_id,
            action_name=args.action_name,
            date=args.date,
            t_index=args.t_index,
            adherence_flag=args.adherence_flag,
            dose=args.dose,
            notes=args.notes,
            data_path=args.data
        )
    except Exception as e:
        logger.error(f"Error in log command: {e}", exc_info=True)
        raise


def cmd_eval(args, config: ExperimentConfig, paths: PathManager) -> pd.DataFrame:
    """Command: evaluate experiments"""
    try:
        evaluator = ExperimentEvaluator(config, paths)
        return evaluator.evaluate(data_path=args.data)
    except Exception as e:
        logger.error(f"Error in eval command: {e}", exc_info=True)
        raise


# ============================================================================
# CLI PARSER
# ============================================================================
def build_argparser() -> argparse.ArgumentParser:
    """Costruisce parser CLI"""
    parser = argparse.ArgumentParser(
        prog="pcb_experiments_level29.py",
        description="PCB Level 2.9 — Experiment plan + logging + evaluation (local-first)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    subparsers = parser.add_subparsers(dest="cmd", help="Available commands")
    
    # Subcommand: plan
    parser_plan = subparsers.add_parser(
        "plan",
        help="Generate experiment plan from insights"
    )
    parser_plan.set_defaults(func=cmd_plan)
    
    # Subcommand: log
    parser_log = subparsers.add_parser(
        "log",
        help="Log an experiment trial"
    )
    parser_log.add_argument(
        "--insight_id",
        required=True,
        help="Insight ID (e.g., I2-00001)"
    )
    parser_log.add_argument(
        "--action_name",
        default="",
        help="Action name (optional, auto-resolved from plan)"
    )
    parser_log.add_argument(
        "--date",
        default="",
        help="Trial date in ISO format YYYY-MM-DD (optional)"
    )
    parser_log.add_argument(
        "--t_index",
        type=int,
        default=None,
        help="Explicit time index (optional, overrides date)"
    )
    parser_log.add_argument(
        "--adherence_flag",
        type=int,
        default=1,
        choices=[0, 1],
        help="Whether action was performed (0=no, 1=yes)"
    )
    parser_log.add_argument(
        "--dose",
        default="",
        help="Dose/intensity description (optional)"
    )
    parser_log.add_argument(
        "--notes",
        default="",
        help="Additional notes (optional)"
    )
    parser_log.add_argument(
        "--data",
        default=None,
        help="Custom data file path (optional)"
    )
    parser_log.set_defaults(func=cmd_log)
    
    # Subcommand: eval
    parser_eval = subparsers.add_parser(
        "eval",
        help="Evaluate experiment results"
    )
    parser_eval.add_argument(
        "--data",
        default=None,
        help="Custom data file path (optional)"
    )
    parser_eval.set_defaults(func=cmd_eval)
    
    return parser


# ============================================================================
# MAIN
# ============================================================================
def main(argv: Optional[List[str]] = None) -> int:
    """
    Main entry point
    
    Args:
        argv: Command line arguments (defaults to sys.argv[1:])
        
    Returns:
        int: Exit code (0=success, non-zero=error)
    """
    if argv is None:
        argv = sys.argv[1:]
    
    parser = build_argparser()
    args = parser.parse_args(argv)
    
    # Setup logging level
    if hasattr(args, 'verbose') and args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    
    # Check if command was provided
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    
    try:
        # Load configuration
        config = ExperimentConfig.load_from_file()
        paths = PathManager(config)
        
        # Execute command
        args.func(args, config, paths)
        
        logger.info("Command completed successfully")
        return 0
        
    except KeyboardInterrupt:
        logger.warning("\nOperation cancelled by user")
        return 130
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
