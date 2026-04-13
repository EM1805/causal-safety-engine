from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .transform import transform_events_to_timeseries

MIN_SAMPLES = 14
MAX_FEATURES = 8
BOOTSTRAP_ROUNDS = 200


def _safe_corr(series_a: pd.Series, series_b: pd.Series) -> float:
    if series_a.nunique(dropna=True) <= 1 or series_b.nunique(dropna=True) <= 1:
        return 0.0
    corr = series_a.corr(series_b)
    if pd.isna(corr):
        return 0.0
    return float(corr)


def _make_lag_features(frame: pd.DataFrame, features: list[str], lag: int = 1) -> pd.DataFrame:
    lagged = frame.copy()
    for feature in features:
        lagged[f"{feature}_lag{lag}"] = lagged[feature].shift(lag)
    return lagged


def _drop_noisy_or_collinear(frame: pd.DataFrame, features: list[str]) -> list[str]:
    kept: list[str] = []
    for feature in features:
        std = frame[feature].std(ddof=0)
        if pd.isna(std) or std == 0:
            continue
        kept.append(feature)

    final: list[str] = []
    for feature in kept:
        too_collinear = False
        for selected in final:
            if abs(_safe_corr(frame[feature], frame[selected])) >= 0.92:
                too_collinear = True
                break
        if not too_collinear:
            final.append(feature)

    return final[:MAX_FEATURES]


def _bootstrap_stats(frame: pd.DataFrame, feature: str, target: str) -> tuple[float, float, float]:
    sampled: list[float] = []
    clean = frame[[feature, target]].dropna()
    if len(clean) < 5:
        return 0.0, 0.0, 0.0

    for _ in range(BOOTSTRAP_ROUNDS):
        sample = clean.sample(n=len(clean), replace=True)
        sampled.append(_safe_corr(sample[feature], sample[target]))

    p10 = float(np.quantile(sampled, 0.10))
    p50 = float(np.quantile(sampled, 0.50))
    p90 = float(np.quantile(sampled, 0.90))
    return p10, p50, p90


def _stability_ratio(frame: pd.DataFrame, feature: str, target: str) -> float:
    if len(frame) < 8:
        return 0.0
    midpoint = len(frame) // 2
    first_half = frame.iloc[:midpoint]
    second_half = frame.iloc[midpoint:]

    c1 = _safe_corr(first_half[feature], first_half[target])
    c2 = _safe_corr(second_half[feature], second_half[target])

    if c1 == 0 and c2 == 0:
        return 0.0
    sign_match = np.sign(c1) == np.sign(c2)
    magnitude_alignment = max(0.0, 1.0 - abs(abs(c1) - abs(c2)))
    return float((1.0 if sign_match else 0.0) * magnitude_alignment)


def _driver_sentence(feature: str, effect: float) -> str:
    name = feature.replace("_", " ").title()
    if effect < 0:
        return f"Higher {name} likely contributed to the drop."
    return f"Higher {name} likely supported conversion."


def run_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    frame = transform_events_to_timeseries(rows).sort_values("date").reset_index(drop=True)

    if "conversion" not in frame.columns:
        raise ValueError("Expected conversion metric in transformed dataset")

    numeric_cols = frame.select_dtypes(include=["number"]).columns.tolist()
    base_features = [c for c in numeric_cols if c != "conversion"]

    if not base_features:
        return {
            "causes": [],
            "confidence": 0.0,
            "suggestion": "Add at least one driver metric (for example: price, traffic).",
        }

    with_lags = _make_lag_features(frame, base_features, lag=1).dropna().reset_index(drop=True)
    candidate_cols = [c for c in with_lags.select_dtypes(include=["number"]).columns if c != "conversion"]
    filtered = _drop_noisy_or_collinear(with_lags, candidate_cols)

    if not filtered:
        return {
            "causes": [],
            "confidence": 0.0,
            "suggestion": "No stable drivers after noise/collinearity filtering.",
        }

    drivers: list[dict[str, Any]] = []
    confidence_parts: list[float] = []

    for feature in filtered:
        effect = _safe_corr(with_lags[feature], with_lags["conversion"])
        p10, p50, p90 = _bootstrap_stats(with_lags, feature, "conversion")
        stable = _stability_ratio(with_lags, feature, "conversion")

        if abs(p50) < 0.05:
            continue

        falsification_pass = int(np.sign(p10) == np.sign(p90) and p10 != 0 and p90 != 0)
        confidence_local = min(1.0, abs(p50) * 0.6 + stable * 0.25 + falsification_pass * 0.15)
        confidence_parts.append(confidence_local)

        drivers.append(
            {
                "feature": feature,
                "effect": round(effect, 2),
                "stability": round(stable, 2),
                "bootstrap_p10": round(p10, 2),
                "bootstrap_p90": round(p90, 2),
                "explanation": _driver_sentence(feature, effect),
                "falsification_pass": bool(falsification_pass),
            }
        )

    drivers.sort(key=lambda item: abs(item["effect"]), reverse=True)
    top_drivers = drivers[:5]

    if not top_drivers:
        return {
            "causes": [],
            "confidence": 0.0,
            "suggestion": "No meaningful signal detected. Collect more data or expand features.",
        }

    confidence = round(float(np.mean(confidence_parts)), 2) if confidence_parts else 0.0

    if len(with_lags) < MIN_SAMPLES:
        confidence = round(min(confidence, 0.45), 2)

    top_negative = next((d for d in top_drivers if d["effect"] < 0), None)
    if top_negative is None:
        suggestion = "Run a controlled test on the strongest positive driver to confirm uplift."
    else:
        suggestion = (
            f"Prioritize an A/B test on {top_negative['feature']} and validate impact over the next 7 days."
        )

    return {
        "causes": top_drivers,
        "confidence": confidence,
        "suggestion": suggestion,
    }
