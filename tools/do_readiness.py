#!/usr/bin/env python3
"""
PCB Do-Calculus Readiness (Identifiability) — Partner-safe (no extra deps)

Goal:
- Provide *formal* intervention-readiness checks WITHOUT estimating causal effects.
- Outputs are audit-friendly and conservative.

Inputs (defaults under --out):
- personal_causal_graph.json
- AUTHORITY/causal_authority.csv

Outputs (created under --out/DO_CHECK):
- do_identifiability.json      (per-edge checks)
- proof_trace.json             (audit-friendly trace)
- do_summary.json              (counts + metadata)

This tool intentionally implements a conservative, *structure-only* backdoor-style check
using only the observed DAG edges (stdlib only). Unknowns are surfaced as assumptions.
"""

import argparse, json, os, sys, datetime, csv
from typing import Dict, Any, List, Set, Tuple

def utc_now() -> str:
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

class DiGraph:
    def __init__(self):
        self.nodes: Set[str] = set()
        self.succ: Dict[str, Set[str]] = {}
        self.pred: Dict[str, Set[str]] = {}

    def add_node(self, n: str):
        if n not in self.nodes:
            self.nodes.add(n)
            self.succ[n] = set()
            self.pred[n] = set()

    def add_edge(self, u: str, v: str):
        self.add_node(u); self.add_node(v)
        self.succ[u].add(v)
        self.pred[v].add(u)

    def successors(self, n: str) -> Set[str]:
        return self.succ.get(n, set())

    def predecessors(self, n: str) -> Set[str]:
        return self.pred.get(n, set())

def build_graph(pcg: Dict[str, Any]) -> DiGraph:
    G = DiGraph()
    for n in pcg.get("nodes", []):
        nid = n.get("id")
        if nid:
            G.add_node(str(nid))
    for e in pcg.get("edges", []):
        s, t = e.get("source"), e.get("target")
        if s and t:
            G.add_edge(str(s), str(t))
    return G

def has_path_excluding(G: DiGraph, src: str, dst: str, blocked: Set[str]) -> bool:
    if src in blocked or dst in blocked:
        return False
    q = [src]
    seen = {src}
    while q:
        u = q.pop(0)
        if u == dst:
            return True
        for v in G.successors(u):
            if v in blocked or v in seen:
                continue
            seen.add(v)
            q.append(v)
    return False

def backdoor_confounders(G: DiGraph, X: str, Y: str) -> List[str]:
    """
    Conservative heuristic:
    - Any parent of X that can reach Y via directed paths (excluding X) is treated as a potential confounder.
    """
    conf = set()
    for z in G.predecessors(X):
        if z == Y:
            continue
        if has_path_excluding(G, z, Y, blocked={X}):
            conf.add(z)
    return sorted(conf)

def check_identifiability(G: DiGraph, X: str, Y: str) -> Dict[str, Any]:
    out = {
        "x": X,
        "y": Y,
        "identifiable": False,
        "method": "backdoor_conservative_structure_only",
        "adjustment_set_candidate": [],
        "assumptions_required": [],
        "reason": "",
        "proof_trace": []
    }

    if X == Y:
        out["reason"] = "Trivial query: X == Y."
        out["assumptions_required"] = ["Query is meaningful (X != Y)"]
        out["proof_trace"].append({"step":"sanity", "detail":"Rejected: X equals Y."})
        return out

    if X not in G.nodes or Y not in G.nodes:
        out["reason"] = "Nodes not present in graph."
        out["assumptions_required"] = ["Graph contains nodes X and Y"]
        out["proof_trace"].append({"step":"sanity", "detail":"Rejected: node missing."})
        return out

    conf = backdoor_confounders(G, X, Y)
    if not conf:
        out["identifiable"] = True
        out["adjustment_set_candidate"] = []
        out["assumptions_required"] = [
            "No unobserved confounding between X and Y",
            "Graph structure is correct for observed variables"
        ]
        out["reason"] = "No observed backdoor confounders detected (empty adjustment set)."
        out["proof_trace"].append({"step":"backdoor_scan", "detail":"No parent-of-X reaches Y; empty adjustment set."})
        return out

    out["adjustment_set_candidate"] = conf
    out["assumptions_required"] = [
        "All proposed adjustment variables are observed and correctly measured",
        "No unobserved confounding beyond proposed adjustment variables",
        "Graph structure is correct for observed variables"
    ]
    # Conservative but useful: identifiable under explicit assumptions
    out["identifiable"] = True
    out["reason"] = "Potential backdoor confounding detected; identifiable under adjustment assumptions."
    out["proof_trace"].append({"step":"backdoor_scan", "detail":f"Parents of X reaching Y detected: {conf}."})
    out["proof_trace"].append({"step":"adjustment", "detail":f"Propose adjustment set: {conf} (assumption-dependent)."})
    return out

def read_authority_csv(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def batch_from_authority(G: DiGraph, auth_rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    checks, traces = [], []
    for r in auth_rows:
        if (r.get("authority_state") or "").strip() != "stable":
            continue
        if (r.get("review_action") or "").strip() != "ALLOW":
            continue
        X, Y = (r.get("source") or "").strip(), (r.get("target") or "").strip()
        if not X or not Y:
            continue
        edge_id = r.get("edge_id") or f"{X}->{Y}"
        lag = r.get("lag")
        try:
            lag = int(lag) if lag not in (None, "", "nan") else None
        except Exception:
            lag = None

        res = check_identifiability(G, X, Y)
        res.update({"edge_id": edge_id, "lag": lag, "generated_at_utc": utc_now()})
        # derive partner-safe action from identifiability
        if not res["identifiable"]:
            res["do_review_action"] = "BLOCK_INTERVENTION"
        else:
            res["do_review_action"] = "FLAG" if len(res.get("adjustment_set_candidate", [])) > 0 else "ALLOW"

        checks.append({k:v for k,v in res.items() if k != "proof_trace"})
        traces.append({
            "edge_id": edge_id,
            "x": X, "y": Y, "lag": lag,
            "method": res.get("method"),
            "proof_trace": res.get("proof_trace", [])
        })
    return checks, traces

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output directory produced by PCB run (contains personal_causal_graph.json).")
    ap.add_argument("--graph", default=None, help="Override path to personal_causal_graph.json")
    ap.add_argument("--authority", default=None, help="Override path to AUTHORITY/causal_authority.csv")
    ap.add_argument("--x", default=None, help="Single query: X")
    ap.add_argument("--y", default=None, help="Single query: Y")
    args = ap.parse_args()

    out_dir = args.out
    graph_path = args.graph or os.path.join(out_dir, "personal_causal_graph.json")
    authority_path = args.authority or os.path.join(out_dir, "AUTHORITY", "causal_authority.csv")

    if not os.path.exists(graph_path):
        print(f"[do_readiness] Missing graph: {graph_path}", file=sys.stderr)
        sys.exit(2)

    pcg = load_json(graph_path)
    G = build_graph(pcg)

    do_dir = os.path.join(out_dir, "DO_CHECK")
    os.makedirs(do_dir, exist_ok=True)

    meta = {
        "artifact_type": "do_calculus_readiness",
        "schema_version": "1.0",
        "generated_at_utc": utc_now(),
        "graph_path": os.path.relpath(graph_path, out_dir),
        "mode": "batch" if (args.x is None and args.y is None) else "single"
    }

    if args.x and args.y:
        res = check_identifiability(G, args.x, args.y)
        with open(os.path.join(do_dir, "do_identifiability_single.json"), "w", encoding="utf-8") as f:
            json.dump({**meta, "result": res}, f, indent=2)
        with open(os.path.join(do_dir, "proof_trace_single.json"), "w", encoding="utf-8") as f:
            json.dump({**meta, "proof_trace": res.get("proof_trace", [])}, f, indent=2)
        with open(os.path.join(do_dir, "do_summary.json"), "w", encoding="utf-8") as f:
            json.dump({**meta, "counts": {"queries": 1}}, f, indent=2)
        return

    if not os.path.exists(authority_path):
        print(f"[do_readiness] Missing authority: {authority_path}", file=sys.stderr)
        sys.exit(3)

    auth_rows = read_authority_csv(authority_path)
    checks, traces = batch_from_authority(G, auth_rows)

    with open(os.path.join(do_dir, "do_identifiability.json"), "w", encoding="utf-8") as f:
        json.dump({**meta, "checks": checks}, f, indent=2)
    with open(os.path.join(do_dir, "proof_trace.json"), "w", encoding="utf-8") as f:
        json.dump({**meta, "traces": traces}, f, indent=2)

    summary = {
        **meta,
        "counts": {
            "stable_allow_edges_checked": len(checks),
            "allow": sum(1 for c in checks if c.get("do_review_action") == "ALLOW"),
            "flag": sum(1 for c in checks if c.get("do_review_action") == "FLAG"),
            "block": sum(1 for c in checks if c.get("do_review_action") == "BLOCK_INTERVENTION"),
        }
    }
    with open(os.path.join(do_dir, "do_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
