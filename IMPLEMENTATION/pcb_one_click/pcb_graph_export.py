#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FILE: pcb_graph_export.py
Python 3.7+

Builds a **Personal Causal Graph** artifact (machine-readable) from existing PCB outputs.

Primary inputs (best-effort):
- out/insights_level2.csv   (Level 2.5)
- out/edges.csv             (optional, if produced by other variants)

Output:
- out/personal_causal_graph.json

Notes:
- This is an *operational, directed predictive influence graph* (Granger-like),
  not clinical / ontological causality.
- Deterministic ordering: nodes and edges are sorted for stable diffs.
"""

import os
import json
from datetime import datetime

import pandas as pd
import numpy as np


DEFAULT_OUT_DIR = "out"
DEFAULT_L2 = os.path.join(DEFAULT_OUT_DIR, "insights_level2.csv")
DEFAULT_EDGES = os.path.join(DEFAULT_OUT_DIR, "edges.csv")
DEFAULT_GRAPH_JSON = os.path.join(DEFAULT_OUT_DIR, "personal_causal_graph.json")


def _exists(p):
    try:
        return bool(p) and os.path.exists(p)
    except Exception:
        return False


def _safe_float(x, default=np.nan):
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _pick_first_col(df, candidates):
    if df is None or len(df) == 0:
        return None
    cols = set(list(df.columns))
    for c in candidates:
        if c in cols:
            return c
    return None


def export_personal_causal_graph(out_dir=DEFAULT_OUT_DIR,
                                insights_l2_path=None,
                                edges_path=None,
                                out_json_path=None):
    """Export a PCG artifact as JSON. Returns output path (or None on failure)."""
    out_dir = out_dir or DEFAULT_OUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    insights_l2_path = insights_l2_path or os.path.join(out_dir, "insights_level2.csv")
    edges_path = edges_path or os.path.join(out_dir, "edges.csv")
    out_json_path = out_json_path or os.path.join(out_dir, "personal_causal_graph.json")

    src_used = None
    df = None

    # Prefer explicit edges.csv if present, else fall back to insights_level2.csv
    if _exists(edges_path):
        try:
            df = pd.read_csv(edges_path)
            src_used = os.path.basename(edges_path)
        except Exception:
            df = None

    if df is None and _exists(insights_l2_path):
        try:
            df = pd.read_csv(insights_l2_path)
            src_used = os.path.basename(insights_l2_path)
        except Exception:
            df = None

    graph = {
        "artifact_type": "personal_causal_graph",
        "schema_version": "v1",
        "generated_at_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "out_dir": out_dir,
        "source_artifact": src_used or "",
        "semantics": {
            "meaning": "directed_predictive_influence",
            "disclaimer": "Not RCT-grade causality. Operational, Granger-like directional influence."
        },
        "nodes": [],
        "edges": [],
        "notes": []
    }

    if df is None or len(df) == 0:
        graph["notes"].append("No inputs found or empty. Run Level 2.5 to generate insights_level2.csv.")
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(graph, f, ensure_ascii=False, indent=2)
        return out_json_path

    # Normalize expected columns
    source_col = _pick_first_col(df, ["source", "src", "from"])
    target_col = _pick_first_col(df, ["target", "trg", "to"])
    lag_col    = _pick_first_col(df, ["lag", "t_lag", "lag_days"])
    weight_col = _pick_first_col(df, ["weight", "coef", "beta", "delta_test", "delta", "effect", "avg_z_cf"])
    strength_col = _pick_first_col(df, ["strength", "score", "selection_confidence", "confidence", "pip"])
    tier_col = _pick_first_col(df, ["tier", "class", "label"])
    sign_cons_col = _pick_first_col(df, ["sign_consistency", "sign_stability"])

    if source_col is None or target_col is None:
        graph["notes"].append("Missing required columns (source/target) in input artifact.")
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(graph, f, ensure_ascii=False, indent=2)
        return out_json_path

    # Build nodes
    nodes = set()
    for _, r in df.iterrows():
        s = str(r.get(source_col, "")).strip()
        t = str(r.get(target_col, "")).strip()
        if s:
            nodes.add(s)
        if t:
            nodes.add(t)
    graph["nodes"] = [{"id": n} for n in sorted(nodes)]

    # Build edges
    edges = []
    for _, r in df.iterrows():
        s = str(r.get(source_col, "")).strip()
        t = str(r.get(target_col, "")).strip()
        if not s or not t:
            continue

        lag = r.get(lag_col, None) if lag_col else None
        try:
            lag = int(float(lag)) if lag is not None and str(lag) != "" else None
        except Exception:
            lag = None

        w = _safe_float(r.get(weight_col, np.nan), np.nan) if weight_col else np.nan
        strength = _safe_float(r.get(strength_col, np.nan), np.nan) if strength_col else np.nan
        tier = str(r.get(tier_col, "")).strip() if tier_col else ""
        sign_cons = _safe_float(r.get(sign_cons_col, np.nan), np.nan) if sign_cons_col else np.nan

        e = {
            "source": s,
            "target": t,
            "lag": lag,
            "weight": float(w) if np.isfinite(w) else None,
            "strength": float(strength) if np.isfinite(strength) else None,
            "sign_consistency": float(sign_cons) if np.isfinite(sign_cons) else None,
            "tier": tier or None,
        }
        edges.append(e)

    # Deterministic order
    def _edge_key(e):
        return (e.get("source",""), e.get("target",""), e.get("lag") if e.get("lag") is not None else 999999)
    edges = sorted(edges, key=_edge_key)

    graph["edges"] = edges

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)

    return out_json_path


def main():
    out = export_personal_causal_graph()
    print("[pcb] Saved:", out)


if __name__ == "__main__":
    main()
