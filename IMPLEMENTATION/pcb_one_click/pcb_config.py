# FILE: pcb_config.py
# Python 3.7 compatible
#
# Minimal central config loader for PCB (local-first).
# If pcb.json exists in the project root, levels can override defaults
# like target_col/date_col/out_dir/max_lag/min_strength without editing code.

import json
import os

DEFAULT_CONFIG = {
    # Default uses neutral naming; override via pcb.json for domain-specific terms.
    "target_col": "target",
    "date_col": "date",
    "out_dir": "out",
    "data_primary": "data.csv",
    "data_fallback": os.path.join("out", "demo_data.csv"),
    "use_data_clean_if_present": True,
    "pipeline": {},
    "level25": {
        "max_lag": 7,
        "min_support_n": 25,
        "min_strength": 0.45,
        "min_p_sign": 0.60,
        "min_effect_abs": 0.05,
        "detrend_mode": "none",
        # --- Causality hardening (conservative defaults)
        "adjustment_mode": "full",  # "off" | "light" | "full"
        "placebo_enable": True,
        "placebo_future_enable": True,
        "placebo_perm_enable": True,
        "placebo_perm_B": 20,
        "placebo_block_len": 7,
        "placebo_margin": 0.02,

        # --- Enterprise robustness upgrades
        # Negative control outcome: a metric that should NOT respond to interventions.
        # If present and it shows a comparable "effect", the insight is likely confounded.
        "negative_control_enable": True,
        "negative_control_outcome_col": "negative_control_outcome",
        # Fail if negative-control strength is too high OR too close to the main strength.
        "negative_control_max_strength": 0.30,
        "negative_control_margin": 0.05,

        # Slice stability: require the same finding to hold across simple slices.
        # (Helps enterprise trust: weekday/weekend, early/late period.)
        "stability_enable": True,
        "stability_slices": ["weekday", "weekend", "first_half", "second_half"],
        "stability_min_score": 0.66,
    },
    "level28": {
        "z_trigger": 0.80,
        "max_alerts": 15,
        "min_strength": 0.45,
        "lookback_days": 90,
        "hard_weekday_match": True,
        "min_baseline_n": 10,
    },
    "level32": {
        "enable_metrics": True,
        "enable_matches_table": True,
        "enable_placebo": True,

        # --- Enterprise robustness upgrades
        "negative_control_enable": True,
        "negative_control_outcome_col": "negative_control_outcome",
        # Action is rejected if negative-control lower bound is high.
        "negative_control_max_success_lb": 0.55,

        # --- Enterprise causal upgrades (L3.2)
        "enable_propensity": True,
        "propensity_max_diff": 0.20,   # max allowed |p_treated - p_control_mean|
        "propensity_action_col": "action_active",

        "enable_pretrend_check": True,
        "pretrend_days": 7,
        "pretrend_max_diff": 0.30,     # max allowed relative slope difference

    }
}

def load_config(path="pcb.json"):
    """Load pcb.json if present; shallow-merge with defaults + merge known nested blocks."""
    cfg = dict(DEFAULT_CONFIG)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            user = json.load(f)
        # shallow merge
        for k, v in user.items():
            cfg[k] = v
        # nested merge
        for k in ["level25", "level28", "level32", "pipeline"]:
            if isinstance(cfg.get(k), dict) and isinstance(user.get(k), dict):
                merged = dict(DEFAULT_CONFIG.get(k, {}))
                merged.update(user.get(k, {}))
                cfg[k] = merged
    return cfg
