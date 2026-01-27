# FILE: pcb_guardrails_audit.py
# Python 3.7 compatible
#
# Guardrails audit (standalone)
# Reads out/edges.csv and/or out/insights_level2.csv and writes:
#   out/guardrail_audit.csv
#   out/guardrail_audit_summary.csv
#
# Dependencies: numpy, pandas only

import os
import numpy as np
import pandas as pd

OUT_DIR = "out"

# -----------------------------
# Central config override (optional)
# -----------------------------
try:
    from pcb_config import load_config  # local file
    _CFG = load_config()
except Exception:
    _CFG = {{}}

# Allow overriding common columns/paths via pcb.json (keeps backward compatibility)
OUT_DIR = str(_CFG.get("out_dir", OUT_DIR))
EDGES_PATH = os.path.join(OUT_DIR, "edges.csv")
INSIGHTS_PATH = os.path.join(OUT_DIR, "insights_level2.csv")
AUDIT_CSV = os.path.join(OUT_DIR, "guardrail_audit.csv")
SUMMARY_CSV = os.path.join(OUT_DIR, "guardrail_audit_summary.csv")

DEDUP_CROSS_FILES = True


def _ensure_out():
    os.makedirs(OUT_DIR, exist_ok=True)


def _risk_type(reason):
    r = str(reason or "").lower()
    if "leakage" in r:
        return "leakage"
    if "drift" in r:
        return "drift"
    return "unknown"


def _read_csv(path):
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None


def _coerce_guardrail_cols(df):
    if df is None or len(df) == 0:
        return df
    if "guardrail_flag" not in df.columns:
        df["guardrail_flag"] = 0
    if "guardrail_reason" not in df.columns:
        df["guardrail_reason"] = ""
    df["guardrail_flag"] = pd.to_numeric(df["guardrail_flag"], errors="coerce").fillna(0).astype(int)
    df["guardrail_reason"] = df["guardrail_reason"].fillna("").astype(str)
    return df


def _pick(df, col, default=""):
    return df[col] if col in df.columns else pd.Series([default] * len(df))


def _dedup(df_a):
    if not DEDUP_CROSS_FILES or df_a is None or len(df_a) == 0:
        return df_a
    need = ["source", "target", "lag", "guardrail_reason", "source_file"]
    if any(c not in df_a.columns for c in need):
        return df_a
    pref = df_a["source_file"].map({"insights_level2.csv": 0, "edges.csv": 1}).fillna(2).astype(int)
    tmp = df_a.copy()
    tmp["_pref"] = pref
    tmp["lag"] = pd.to_numeric(tmp["lag"], errors="coerce")
    tmp = tmp.sort_values(["_pref", "risk_type", "guardrail_reason", "strength"], ascending=[True, True, True, False])
    tmp = tmp.drop_duplicates(subset=["source", "target", "lag", "guardrail_reason"], keep="first")
    return tmp.drop(columns=["_pref"]).reset_index(drop=True)


def main():
    _ensure_out()

    df_edges_all = _read_csv(EDGES_PATH)
    df_ins_all = _read_csv(INSIGHTS_PATH)

    df_edges = _coerce_guardrail_cols(df_edges_all.copy() if df_edges_all is not None else None)
    df_ins = _coerce_guardrail_cols(df_ins_all.copy() if df_ins_all is not None else None)

    parts = []

    if df_edges is not None and len(df_edges) > 0:
        e = df_edges[df_edges["guardrail_flag"] == 1].copy()
        if len(e) > 0:
            parts.append(pd.DataFrame({
                "source_file": "edges.csv",
                "edge_id": _pick(e, "edge_id", ""),
                "insight_id": "",
                "source": _pick(e, "source", ""),
                "target": _pick(e, "target", ""),
                "lag": pd.to_numeric(_pick(e, "lag", np.nan), errors="coerce"),
                "strength": pd.to_numeric(_pick(e, "strength", np.nan), errors="coerce"),
                "guardrail_reason": _pick(e, "guardrail_reason", ""),
                "leakage_corr_future": pd.to_numeric(_pick(e, "leakage_corr_future", np.nan), errors="coerce"),
                "leakage_corr_now": pd.to_numeric(_pick(e, "leakage_corr_now", np.nan), errors="coerce"),
                "leakage_gap": pd.to_numeric(_pick(e, "leakage_gap", np.nan), errors="coerce"),
                "drift_corr_time_source": pd.to_numeric(_pick(e, "drift_corr_time_source", np.nan), errors="coerce"),
                "drift_corr_time_target": pd.to_numeric(_pick(e, "drift_corr_time_target", np.nan), errors="coerce"),
            }))

    if df_ins is not None and len(df_ins) > 0:
        i = df_ins[df_ins["guardrail_flag"] == 1].copy()
        if len(i) > 0:
            parts.append(pd.DataFrame({
                "source_file": "insights_level2.csv",
                "edge_id": "",
                "insight_id": _pick(i, "insight_id", ""),
                "source": _pick(i, "source", ""),
                "target": _pick(i, "target", ""),
                "lag": pd.to_numeric(_pick(i, "lag", np.nan), errors="coerce"),
                "strength": pd.to_numeric(_pick(i, "strength", np.nan), errors="coerce"),
                "guardrail_reason": _pick(i, "guardrail_reason", ""),
                "leakage_corr_future": pd.to_numeric(_pick(i, "leakage_corr_future", np.nan), errors="coerce"),
                "leakage_corr_now": pd.to_numeric(_pick(i, "leakage_corr_now", np.nan), errors="coerce"),
                "leakage_gap": pd.to_numeric(_pick(i, "leakage_gap", np.nan), errors="coerce"),
                "drift_corr_time_source": pd.to_numeric(_pick(i, "drift_corr_time_source", np.nan), errors="coerce"),
                "drift_corr_time_target": pd.to_numeric(_pick(i, "drift_corr_time_target", np.nan), errors="coerce"),
            }))

    if len(parts) == 0:
        pd.DataFrame().to_csv(AUDIT_CSV, index=False)
        pd.DataFrame(columns=["group", "key", "count"]).to_csv(SUMMARY_CSV, index=False)
        print("No guardrail-flagged rows found.")
        return 0

    df_a = pd.concat(parts, ignore_index=True)
    df_a["risk_type"] = df_a["guardrail_reason"].apply(_risk_type)

    if DEDUP_CROSS_FILES:
        df_a = _dedup(df_a)

    df_a = df_a.sort_values(["risk_type", "guardrail_reason", "strength"], ascending=[True, True, False], na_position="last")
    df_a.to_csv(AUDIT_CSV, index=False)

    summary_rows = []
    summary_rows.append({"group": "executive", "key": "flagged_rows_total", "count": int(len(df_a))})
    summary_rows.append({"group": "executive", "key": "dedup_cross_files", "count": int(1 if DEDUP_CROSS_FILES else 0)})

    if df_edges_all is not None and len(df_edges_all) > 0:
        total_edges = int(len(df_edges_all))
        if "guardrail_flag" in df_edges_all.columns:
            flagged_edges = int(pd.to_numeric(df_edges_all["guardrail_flag"], errors="coerce").fillna(0).astype(int).sum())
        else:
            flagged_edges = 0
        summary_rows.append({"group": "executive", "key": "edges_total", "count": total_edges})
        summary_rows.append({"group": "executive", "key": "edges_flagged", "count": flagged_edges})
        share_permille = int(round(1000.0 * (float(flagged_edges) / float(total_edges)))) if total_edges > 0 else 0
        summary_rows.append({"group": "executive", "key": "edges_flagged_share_permille", "count": share_permille})

    for group, col in [("risk_type", "risk_type"), ("reason", "guardrail_reason"), ("source", "source")]:
        if col in df_a.columns:
            vc = df_a[col].fillna("").replace({"": "(empty)"}).value_counts()
            for k, c in vc.items():
                summary_rows.append({"group": group, "key": str(k), "count": int(c)})

    df_s = pd.DataFrame(summary_rows)
    df_s.to_csv(SUMMARY_CSV, index=False)

    print("Saved:", AUDIT_CSV)
    print("Saved:", SUMMARY_CSV)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
