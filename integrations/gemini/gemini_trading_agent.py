"""
Gemini Trading Proposal Agent (Safety-Governed)

Role separation (STRICT):
- Gemini: proposes ONE trade (JSON only)
- Deterministic Gate: validates, constrains, decides
- No execution is performed by this module

All proposals and decisions are persisted for auditability.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Any

import google.generativeai as genai

from gemini_llm_agent import extract_json, get_model
from live_trading_dataset import (
    LiveTradingDataset,
    append_full_decision_trace,
    gate_trade_proposal,
    validate_trade_proposal,
)


# ---------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------

def build_trading_prompt(context: str) -> str:
    """Build a strict JSON-only trading proposal prompt."""
    return f"""
You must output ONLY a valid JSON object.
No markdown. No explanations. No extra text.

Context:
{context}

Rules:
- Propose exactly ONE conservative trade
- Use realistic, small position sizing
- Always include explicit risk controls
- Assume capital preservation is the top priority

JSON schema:
{{
  "action": "propose_trade",
  "params": {{
    "symbol": "string",
    "side": "buy|sell",
    "quantity": number,
    "order_type": "market|limit",
    "max_slippage_bps": number,
    "stop_loss_pct": number,
    "take_profit_pct": number
  }},
  "rationale": "string"
}}
"""


# ---------------------------------------------------------------------
# Single-cycle execution (proposal → gate → audit)
# ---------------------------------------------------------------------

def run_once(
    context: str,
    dataset_path: str = "out/live_trading_dataset.jsonl",
) -> Dict[str, Any]:
    """
    Execute one Gemini → deterministic gate cycle.

    Returns:
        Dict containing proposal, decision, reasons, constraints,
        and audit dataset path.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")

    genai.configure(api_key=api_key)

    model = get_model()
    dataset = LiveTradingDataset(dataset_path)

    # --- Gemini proposal (untrusted) ---
    response = model.generate_content(build_trading_prompt(context))
    raw_payload = extract_json(response.text)

    # --- Deterministic validation ---
    proposal = validate_trade_proposal(raw_payload)

    # --- Persist market context snapshot ---
    dataset.append_event(
        "market_context",
        {"context": context},
    )

    # --- Deterministic gating (ALLOW / RESTRICT / BLOCK) ---
    gate_result = gate_trade_proposal(proposal, dataset)

    # --- Persist full decision trace ---
    append_full_decision_trace(
        dataset=dataset,
        proposal=proposal,
        gate=gate_result,
    )

    return {
        "proposal": proposal,
        "decision": gate_result.decision,
        "reasons": gate_result.reasons,
        "constraints": gate_result.constraints,
        "dataset_path": dataset_path,
    }


# ---------------------------------------------------------------------
# Demo entrypoint
# ---------------------------------------------------------------------

def main() -> None:
    context = """
Market snapshot:
- symbol: AAPL
- price: 220.15
- intraday volatility: medium
- spread_bps: 1.8
- trend: mild up
"""
    result = run_once(context)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
