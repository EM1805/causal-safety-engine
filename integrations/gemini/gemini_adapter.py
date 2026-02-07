"""
Gemini Adapter for Causal Safety Engine

Gemini is used ONLY to propose counterfactual changes.
The Causal Safety Engine remains the sole decision authority.
"""

import json
import subprocess
import pandas as pd
from typing import Dict, Any


class GeminiProposalError(Exception):
    pass


def apply_proposal_to_data(
    proposal: Dict[str, Any],
    base_data_path: str,
    out_path: str = "out/counterfactual_data.csv",
) -> str:
    """
    Applies Gemini proposal as a counterfactual modification
    to the observational dataset.
    """

    df = pd.read_csv(base_data_path)

    for feature, delta in proposal.get("proposed_delta", {}).items():
        if feature in df.columns:
            df[feature] = df[feature] + delta

    df.to_csv(out_path, index=False)
    return out_path


def evaluate_with_causal_engine(
    proposal: Dict[str, Any],
    data_path: str,
) -> Dict[str, Any]:

    # 1. Convert proposal → counterfactual data
    cf_data_path = apply_proposal_to_data(
        proposal=proposal,
        base_data_path=data_path,
    )

    # 2. Call the REAL CLI (no fake flags)
    cmd = [
        "python",
        "IMPLEMENTATION/pcb_one_click/pcb_cli.py",
        "run",
        "--data",
        cf_data_path,
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise GeminiProposalError(result.stderr)

    # 3. Read deterministic Level 3 verdict
    with open("out/insights_level3.json") as f:
        decision = json.load(f)

    return decision
