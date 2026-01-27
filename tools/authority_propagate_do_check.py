#!/usr/bin/env python3
"""
Propagate do-calculus readiness results (out/DO_CHECK/do_identifiability.json)
into Authority artifacts (out/AUTHORITY/causal_authority.{csv,json}).

Conservative rules:
- Adds fields: do_identifiable, do_reason_code, do_assumptions, do_summary
- If not identifiable: set review_action=BLOCK_INTERVENTION and intervention_unsafe=True
- Merge do reason into reason_code (pipe-delimited) and append to evidence_summary
"""
import argparse, json, os
from typing import Any, Dict, Tuple, Optional, List

import pandas as pd

def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _dump_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def _norm_lag(lag) -> Optional[int]:
    if lag is None:
        return None
    try:
        return int(lag)
    except Exception:
        return None

def build_do_map(do_obj: Any) -> Dict[Tuple[str,str,Optional[int]], Dict[str,Any]]:
    if isinstance(do_obj, dict) and "results" in do_obj:
        do_obj = do_obj["results"]
    if isinstance(do_obj, dict):
        do_obj = [do_obj]
    m: Dict[Tuple[str,str,Optional[int]], Dict[str,Any]] = {}
    for r in do_obj:
        src = r.get("source") or r.get("x") or r.get("X")
        tgt = r.get("target") or r.get("y") or r.get("Y")
        lag = _norm_lag(r.get("lag"))
        if src is None or tgt is None:
            continue
        m[(str(src), str(tgt), lag)] = r
        # also allow lookup without lag
        m[(str(src), str(tgt), None)] = r
    return m

def merge_reason(prev: str, add: str) -> str:
    parts = set([p.strip() for p in str(prev).split("|") if p.strip() and p.lower() != "nan"])
    if add and str(add).strip():
        parts.add(str(add).strip())
    return "|".join(sorted(parts))

def run(out_dir: str) -> None:
    do_path = os.path.join(out_dir, "DO_CHECK", "do_identifiability.json")
    auth_csv = os.path.join(out_dir, "AUTHORITY", "causal_authority.csv")
    auth_json = os.path.join(out_dir, "AUTHORITY", "causal_authority.json")

    if not (os.path.exists(do_path) and os.path.exists(auth_csv) and os.path.exists(auth_json)):
        return

    do_map = build_do_map(_load_json(do_path))

    df = pd.read_csv(auth_csv)

    # Ensure columns exist
    for col in ["do_identifiable", "do_reason_code", "do_assumptions", "do_summary"]:
        if col not in df.columns:
            df[col] = ""

    for i, row in df.iterrows():
        src = str(row.get("source"))
        tgt = str(row.get("target"))
        lag_key = _norm_lag(row.get("lag"))
        rec = do_map.get((src, tgt, lag_key)) or do_map.get((src, tgt, None))
        if not rec:
            continue

        identifiable = bool(rec.get("identifiable")) if "identifiable" in rec else False
        reason = rec.get("reason_code") or rec.get("reason") or ("IDENTIFIABLE" if identifiable else "NOT_IDENTIFIABLE")
        assumptions = rec.get("assumptions_required") or rec.get("assumptions") or []
        if isinstance(assumptions, list):
            assumptions_str = "; ".join([str(a) for a in assumptions])
        else:
            assumptions_str = str(assumptions)

        df.at[i, "do_identifiable"] = identifiable
        df.at[i, "do_reason_code"] = reason
        df.at[i, "do_assumptions"] = assumptions_str
        df.at[i, "do_summary"] = f"do-check: {'identifiable' if identifiable else 'not-identifiable'}"

        if not identifiable:
            # Balanced–Safety: degrade to FLAG where evidence is directionally consistent and no hard risks.
            # Hard risks always stay BLOCK.
            leakage_gap = row.get("leakage_gap", None)
            leakage_future = row.get("leakage_corr_future", None)
            drift_s = row.get("drift_corr_time_source", None)
            drift_t = row.get("drift_corr_time_target", None)
            guardrail = row.get("guardrail_flag", 0)

            def _f(x):
                try:
                    return float(x)
                except Exception:
                    return float("nan")

            leakage_gap = _f(leakage_gap)
            leakage_future = _f(leakage_future)
            drift_s = _f(drift_s)
            drift_t = _f(drift_t)
            support = _f(row.get("support_n", float("nan")))
            strength = _f(row.get("strength", float("nan")))
            p_sign = _f(row.get("p_sign", float("nan")))
            sign_stability = _f(row.get("sign_stability", float("nan")))
            if sign_stability != sign_stability and p_sign == p_sign:
                sign_stability = abs(p_sign)

            leakage_gap_thr = 0.25
            leakage_future_thr = 0.20
            drift_thr = 0.35
            flag_sign = 0.65
            min_support_flag = 20
            min_stren = 0.02

            hard_risk = False
            if int(guardrail or 0) == 1:
                hard_risk = True
            if (leakage_gap == leakage_gap and leakage_gap > leakage_gap_thr) or (leakage_future == leakage_future and abs(leakage_future) > leakage_future_thr):
                hard_risk = True
            if (drift_s == drift_s and abs(drift_s) > drift_thr) or (drift_t == drift_t and abs(drift_t) > drift_thr):
                hard_risk = True

            can_flag = (not hard_risk) and (sign_stability == sign_stability) and (sign_stability >= flag_sign) and (support == support) and (support >= min_support_flag) and (strength == strength) and (strength >= min_stren)

            if "intervention_unsafe" in df.columns:
                df.at[i, "intervention_unsafe"] = True

            if "review_action" in df.columns:
                prev_action = str(df.at[i, "review_action"]) if not pd.isna(df.at[i, "review_action"]) else ""
                if can_flag:
                    df.at[i, "review_action"] = "FLAG"
                    if "reason_code" in df.columns:
                        df.at[i, "reason_code"] = merge_reason(df.at[i, "reason_code"], "DO_NOT_IDENTIFIABLE_FLAGGED")
                else:
                    df.at[i, "review_action"] = "BLOCK_INTERVENTION"

            if "reason_code" in df.columns:
                df.at[i, "reason_code"] = merge_reason(df.at[i, "reason_code"], str(reason))

            if "evidence_summary" in df.columns:
                prev_e = "" if pd.isna(df.at[i, "evidence_summary"]) else str(df.at[i, "evidence_summary"])
                add_e = "do=not-identifiable"
                if add_e not in prev_e:
                    df.at[i, "evidence_summary"] = (prev_e + ("; " if prev_e.strip() else "") + add_e).strip()


    df.to_csv(auth_csv, index=False)

    j = _load_json(auth_json)
    wrapper = None
    records = j
    if isinstance(j, dict) and "records" in j:
        wrapper = "records"
        records = j["records"]

    for rec_auth in records:
        src = str(rec_auth.get("source"))
        tgt = str(rec_auth.get("target"))
        lag_key = _norm_lag(rec_auth.get("lag"))
        rec = do_map.get((src, tgt, lag_key)) or do_map.get((src, tgt, None))
        if not rec:
            continue

        identifiable = bool(rec.get("identifiable")) if "identifiable" in rec else False
        reason = rec.get("reason_code") or rec.get("reason") or ("IDENTIFIABLE" if identifiable else "NOT_IDENTIFIABLE")
        assumptions = rec.get("assumptions_required") or rec.get("assumptions") or []
        assumptions_str = "; ".join([str(a) for a in assumptions]) if isinstance(assumptions, list) else str(assumptions)

        rec_auth["do_identifiable"] = identifiable
        rec_auth["do_reason_code"] = reason
        rec_auth["do_assumptions"] = assumptions_str
        rec_auth["do_summary"] = f"do-check: {'identifiable' if identifiable else 'not-identifiable'}"

        if not identifiable:
            rec_auth["review_action"] = "BLOCK_INTERVENTION"
            rec_auth["intervention_unsafe"] = True
            rec_auth["reason_code"] = merge_reason(rec_auth.get("reason_code",""), str(reason))
            prev_e = rec_auth.get("evidence_summary","") or ""
            add_e = "do=not-identifiable"
            if add_e not in prev_e:
                rec_auth["evidence_summary"] = (prev_e + ("; " if str(prev_e).strip() else "") + add_e).strip()

    out_obj = {wrapper: records} if wrapper else records
    _dump_json(auth_json, out_obj)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output directory containing AUTHORITY/ and DO_CHECK/")
    args = ap.parse_args()
    run(args.out)

if __name__ == "__main__":
    main()
