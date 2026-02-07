"""
Gemini Adapter for Causal Safety Engine

Gemini is used ONLY as an action proposal generator.
All actions are evaluated by the Causal Safety Engine
before any execution.
"""

import json
import subprocess
from typing import Dict, Any


class GeminiProposalError(Exception):
    pass


def evaluate_with_causal_engine(
    proposal: Dict[str, Any],
    data_path: str,
) -> Dict[str, Any]:
    """
    Sends the proposed action to the Causal Safety Engine
    and returns the deterministic verdict.
    """

    # Write proposal to temp file (audit-friendly)
    with open("out/gemini_proposal.json", "w") as f:
        json.dump(proposal, f, indent=2)

    # Call existing CLI (non-invasive)
    cmd = [
        "python",
        "IMPLEMENTATION/pcb_one_click/pcb_cli.py",
        "run",
        "--data",
        data_path,
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise GeminiProposalError(result.stderr)

    # Read deterministic stdout (correct contract)
    return {
        "status": "EXECUTED",
        "engine_stdout": result.stdout.strip(),
    }
