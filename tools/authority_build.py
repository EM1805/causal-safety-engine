#!/usr/bin/env python
"""
Build PCB Causal Authority artifacts from edges.csv outputs.

This script is intentionally conservative:
- PCB remains observational-only (identifiable=false).
- intervention_unsafe is true unless an edge passes strict stability gates.
"""

import argparse, os, json, datetime
import pandas as pd
import numpy as np

def classify_edge(r,
                  min_support_allow=30,
                  min_support_flag=20,
                  min_stren=0.02,
                  allow_sign=0.85,
                  flag_sign=0.65,
                  leakage_gap_thr=0.25,
                  leakage_future_thr=0.20,
                  drift_thr=0.35):
    """
    Balanced–Safety Authority policy (v1.0):

    Outputs one of: ALLOW / FLAG / BLOCK_INTERVENTION

    - ALLOW: strong, directionally stable, clean (no leakage/drift/guardrail)
    - FLAG: directionally consistent but weak evidence or not-identifiable (soft constraint)
    - BLOCK_INTERVENTION: unstable or risky (hard block)
    """
    reasons = []
    hard = []

    def f(x):
        try:
            return float(x)
        except Exception:
            return np.nan

    support = f(r.get("support_n"))
    strength = f(r.get("strength"))
    p_sign = f(r.get("p_sign"))
    leakage_gap = f(r.get("leakage_gap"))
    leakage_future = f(r.get("leakage_corr_future"))
    drift_s = f(r.get("drift_corr_time_source"))
    drift_t = f(r.get("drift_corr_time_target"))
    guardrail = int(r.get("guardrail_flag", 0) or 0)

    sign_stability = np.nan if np.isnan(p_sign) else abs(p_sign)

    # Hard safety / integrity blockers
    if guardrail == 1:
        hard.append("GUARDRAIL")
    if (not np.isnan(leakage_gap) and leakage_gap > leakage_gap_thr) or (not np.isnan(leakage_future) and abs(leakage_future) > leakage_future_thr):
        hard.append("LEAKAGE_RISK")
    if (not np.isnan(drift_s) and abs(drift_s) > drift_thr) or (not np.isnan(drift_t) and abs(drift_t) > drift_thr):
        hard.append("DRIFT_RISK")

    # Evidence checks
    if np.isnan(support) or support < min_support_flag:
        hard.append("LOW_SUPPORT")
    if np.isnan(strength) or strength < min_stren:
        reasons.append("LOW_SIGNAL")
    if np.isnan(sign_stability):
        hard.append("NO_SIGN")
    elif sign_stability < flag_sign:
        hard.append("SIGN_INSTABILITY")
    elif sign_stability < allow_sign:
        reasons.append("SIGN_WEAK")

    # Allow vs Flag support gating
    if (not np.isnan(support)) and support < min_support_allow:
        reasons.append("SUPPORT_WEAK")

    # Decision
    if len(hard) > 0:
        decision = "BLOCK_INTERVENTION"
        all_reasons = hard + reasons
        authority_state = "unstable"
        intervention_unsafe = True
    else:
        if len(reasons) == 0:
            decision = "ALLOW"
            all_reasons = ["OK"]
            authority_state = "stable"
            intervention_unsafe = False
        else:
            decision = "FLAG"
            all_reasons = reasons
            authority_state = "flagged"
            intervention_unsafe = True

    return decision, authority_state, intervention_unsafe, all_reasons, sign_stability

def build_for_out_dir(out_dir: str) -> dict:
    edges_csv = os.path.join(out_dir, "edges.csv")
    if not os.path.exists(edges_csv):
        return {"out_dir": out_dir, "skipped": True, "reason": "edges.csv not found"}

    df = pd.read_csv(edges_csv)
    recs = []
    for _, r in df.iterrows():
        decision, authority_state, intervention_unsafe, reasons, sign_stability = classify_edge(r)
        evidence = (
            f"support_n={r.get('support_n','NA')}; "
            f"strength={r.get('strength','NA')}; "
            f"p_sign={r.get('p_sign','NA')}; "
            f"leakage_gap={r.get('leakage_gap','NA')}; "
            f"drift_src={r.get('drift_corr_time_source','NA')}; "
            f"drift_tgt={r.get('drift_corr_time_target','NA')}; "
            f"guardrail={r.get('guardrail_flag','NA')}"
        )

        recs.append({
            "edge_id": r.get("edge_id", ""),
            "source": r.get("source", ""),
            "target": r.get("target", ""),
            "lag": int(r.get("lag", 0)) if pd.notna(r.get("lag", np.nan)) else None,

            # Raw evidence fields (keep numeric columns for downstream policy + audit)
            "support_n": r.get("support_n", np.nan),
            "strength": r.get("strength", np.nan),
            "p_sign": r.get("p_sign", np.nan),
            "sign_stability": sign_stability,
            "leakage_gap": r.get("leakage_gap", np.nan),
            "leakage_corr_future": r.get("leakage_corr_future", np.nan),
            "drift_corr_time_source": r.get("drift_corr_time_source", np.nan),
            "drift_corr_time_target": r.get("drift_corr_time_target", np.nan),
            "guardrail_flag": r.get("guardrail_flag", 0),

            # Authority decision
            "authority_state": authority_state,
            "identifiable": False,  # observational-only (do-check may override)
            "intervention_unsafe": bool(intervention_unsafe),
            "reason_code": "|".join(reasons) if isinstance(reasons, (list, tuple)) else str(reasons),
            "review_action": decision,
            "evidence_summary": evidence,
        })

    auth_df = pd.DataFrame(recs)
    auth_dir = os.path.join(out_dir, "AUTHORITY")
    os.makedirs(auth_dir, exist_ok=True)
    auth_df.to_csv(os.path.join(auth_dir, "causal_authority.csv"), index=False)
    auth_df.to_json(os.path.join(auth_dir, "causal_authority.json"), orient="records", indent=2)

    summary = {
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "out_dir": out_dir,
        "edges_total": int(len(auth_df)),
        "allow_edges": int((auth_df["review_action"] == "ALLOW").sum()),
        "flagged_edges": int((auth_df["review_action"] == "FLAG").sum()),
        "blocked_edges": int((auth_df["review_action"] == "BLOCK_INTERVENTION").sum()),
        "stable_edges": int((auth_df["authority_state"] == "stable").sum()),
        "flag_state_edges": int((auth_df["authority_state"] == "flagged").sum()),
        "unstable_edges": int((auth_df["authority_state"] == "unstable").sum()),
        "blocked_interventions": int((auth_df["review_action"] == "BLOCK_INTERVENTION").sum()),
        "policy": "PCB Authority Policy v1.0 (Balanced–Safety)",
        "notes": "Observational-only by default. Use DO_CHECK for do-identifiability; non-identifiable edges degrade to FLAG where possible."
    }
    with open(os.path.join(auth_dir, "authority_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Path to a PCB out/ directory (containing edges.csv).")
    ap.add_argument("--recursive", action="store_true", help="If set, scan for all out/ dirs containing edges.csv under --out.")
    args = ap.parse_args()

    results = []
    if args.recursive:
        for root, dirs, files in os.walk(args.out):
            if "edges.csv" in files:
                results.append(build_for_out_dir(root))
    else:
        results.append(build_for_out_dir(args.out))

    ok = [r for r in results if not r.get("skipped")]
    print(f"[authority_build] processed={len(ok)} skipped={len(results)-len(ok)}")
    for r in ok[:5]:
        print(f"- {r['out_dir']}: stable={r['stable_edges']} unstable={r['unstable_edges']}")

if __name__ == "__main__":
    main()
