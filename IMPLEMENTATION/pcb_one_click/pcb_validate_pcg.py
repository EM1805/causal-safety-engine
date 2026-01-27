# -*- coding: utf-8 -*-
"""Validate Personal Causal Graph artifact.

This is an **OEM-grade guard**: the PCG artifact is the primary deliverable.
If it is missing or malformed, the run must fail.

Schema: v1 (lightweight, forward-compatible)
"""

from __future__ import annotations
from typing import Any, Dict, List, Tuple
import json
import os
import sys

REQUIRED_TOP_KEYS = [
    "artifact_type",
    "schema_version",
    "generated_at_utc",
    "nodes",
    "edges",
    "semantics",
]

def validate_pcg_dict(pcg: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errs: List[str] = []
    for k in REQUIRED_TOP_KEYS:
        if k not in pcg:
            errs.append(f"Missing required key: {k}")

    if pcg.get("artifact_type") not in (None, "personal_causal_graph"):
        errs.append("artifact_type must be 'personal_causal_graph'")

    if not isinstance(pcg.get("nodes", None), list):
        errs.append("nodes must be a list")
    if not isinstance(pcg.get("edges", None), list):
        errs.append("edges must be a list")
    if not isinstance(pcg.get("semantics", None), dict):
        errs.append("semantics must be an object/dict")

    # Soft checks (do not hard-fail on future extensions)
    sem = pcg.get("semantics") or {}
    if isinstance(sem, dict):
        if "meaning" not in sem:
            errs.append("semantics.meaning is required")
        if "disclaimer" not in sem:
            errs.append("semantics.disclaimer is required")

    return (len(errs) == 0), errs

def validate_pcg_file(path: str) -> Tuple[bool, List[str]]:
    if not os.path.exists(path):
        return False, [f"File not found: {path}"]
    try:
        with open(path, "r", encoding="utf-8") as f:
            pcg = json.load(f)
    except Exception as e:
        return False, [f"Invalid JSON: {e}"]
    if not isinstance(pcg, dict):
        return False, ["Top-level JSON must be an object/dict"]
    return validate_pcg_dict(pcg)

def main(argv=None) -> int:
    argv = argv or sys.argv[1:]
    if not argv:
        print("Usage: python pcb_validate_pcg.py out/personal_causal_graph.json", file=sys.stderr)
        return 2
    ok, errs = validate_pcg_file(argv[0])
    if ok:
        print("[pcg] OK:", argv[0])
        return 0
    print("[pcg] INVALID:", argv[0], file=sys.stderr)
    for e in errs:
        print(" -", e, file=sys.stderr)
    return 2

if __name__ == "__main__":
    raise SystemExit(main())
