"""
Gemini adapter for the Causal Safety Engine.

DESIGN CONTRACT
---------------
- Gemini can ONLY propose actions (JSON).
- The deterministic PCB CLI is the sole execution authority.
- Every proposal is validated, normalized, and persisted for auditability.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------

class GeminiProposalError(Exception):
    """Raised when the Gemini proposal is missing or invalid."""


class GeminiEngineError(Exception):
    """Raised when the Causal Safety Engine execution fails."""


# ---------------------------------------------------------------------
# Repository helpers
# ---------------------------------------------------------------------

def _repo_root() -> Path:
    """
    Resolve repository root.

    integrations/gemini/gemini_adapter.py
    -> parents[2] == repo root
    """
    return Path(__file__).resolve().parents[2]


def _out_dir() -> Path:
    return _repo_root() / "out"


def _ensure_out_dir() -> None:
    _out_dir().mkdir(parents=True, exist_ok=True)


def _resolve_input_path(path: str) -> str:
    """
    Resolve a path relative to repo root if not absolute.
    """
    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return str(candidate)

    repo_candidate = _repo_root() / candidate
    return str(repo_candidate)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


# ---------------------------------------------------------------------
# Proposal normalization
# ---------------------------------------------------------------------

def normalize_proposal(proposal: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a Gemini proposal into the canonical schema.

    Canonical schema:
    {
      "action": "adjust_features",
      "params": {
        "deltas": {
          "feature_name": number
        }
      },
      "rationale": "string"
    }

    Legacy compatibility supported:
    {
      "proposed_delta": { "feature": number }
    }
    """
    if not isinstance(proposal, dict):
        raise GeminiProposalError("proposal must be an object")

    # ---- Legacy payload support
    if "proposed_delta" in proposal:
        deltas = proposal.get("proposed_delta")
        if not isinstance(deltas, dict) or not deltas:
            raise GeminiProposalError("proposal.proposed_delta must be a non-empty object")

        canonical_deltas: Dict[str, float] = {}
        for key, value in deltas.items():
            if not isinstance(key, str) or not key.strip():
                raise GeminiProposalError("delta keys must be non-empty strings")
            if not _is_number(value):
                raise GeminiProposalError(f"delta '{key}' must be numeric")
            canonical_deltas[key.strip()] = float(value)

        rationale = proposal.get("rationale", "legacy payload normalized")
        if not isinstance(rationale, str):
            raise GeminiProposalError("proposal.rationale must be a string")

        return {
            "action": "adjust_features",
            "params": {"deltas": canonical_deltas},
            "rationale": rationale.strip(),
        }

    # ---- Canonical payload
    action = proposal.get("action")
    params = proposal.get("params")
    rationale = proposal.get("rationale", "")

    if not isinstance(action, str) or not action.strip():
        raise GeminiProposalError("proposal.action must be a non-empty string")
    if not isinstance(params, dict):
        raise GeminiProposalError("proposal.params must be an object")
    if not isinstance(rationale, str):
        raise GeminiProposalError("proposal.rationale must be a string")

    deltas = params.get("deltas")
    if not isinstance(deltas, dict) or not deltas:
        raise GeminiProposalError("proposal.params.deltas must be a non-empty object")

    canonical_deltas: Dict[str, float] = {}
    for key, value in deltas.items():
        if not isinstance(key, str) or not key.strip():
            raise GeminiProposalError("delta keys must be non-empty strings")
        if not _is_number(value):
            raise GeminiProposalError(f"delta '{key}' must be numeric")
        canonical_deltas[key.strip()] = float(value)

    return {
        "action": action.strip(),
        "params": {"deltas": canonical_deltas},
        "rationale": rationale.strip(),
    }


# ---------------------------------------------------------------------
# Engine invocation (deterministic boundary)
# ---------------------------------------------------------------------

def evaluate_with_causal_engine(
    proposal: Dict[str, Any],
    data_path: str,
    timeout_seconds: int = 120,
    config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Validate + normalize a Gemini proposal, invoke the PCB CLI,
    and return the deterministic engine result.
    """
    canonical_proposal = normalize_proposal(proposal)
    _ensure_out_dir()

    # ---- Persist proposal (audit-first)
    proposal_path = _out_dir() / "gemini_proposal.json"
    with proposal_path.open("w", encoding="utf-8") as f:
        json.dump(canonical_proposal, f, indent=2)

    # ---- Build PCB CLI command
    cli_path = _repo_root() / "IMPLEMENTATION" / "pcb_one_click" / "pcb_cli.py"

    cmd = [
        "python",
        str(cli_path),
        "run",
        "--data",
        _resolve_input_path(data_path),
    ]

    if config_path:
        cmd.extend(["--config", _resolve_input_path(config_path)])

    # ---- Execute deterministically
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise GeminiEngineError(
            f"engine timeout after {timeout_seconds}s"
        ) from exc

    # ---- Persist audit record
    audit_path = _out_dir() / "gemini_audit_record.json"
    audit_record = {
        "command": cmd,
        "timeout_seconds": timeout_seconds,
        "returncode": result.returncode,
        "stderr": result.stderr.strip(),
        "proposal_path": str(proposal_path),
    }

    with audit_path.open("w", encoding="utf-8") as f:
        json.dump(audit_record, f, indent=2)

    if result.returncode != 0:
        raise GeminiEngineError(result.stderr.strip() or "causal engine failed")

    # ---- Authoritative output
    return {
        "status": "ENGINE_EXECUTED",
        "proposal": canonical_proposal,
        "proposal_path": str(proposal_path),
        "audit_record_path": str(audit_path),
        "engine_stdout": result.stdout.strip(),
    }
