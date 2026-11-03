"""
Live Trading Dataset & Deterministic Gating
==========================================

Purpose:
- Provide an append-only audit log for AI-proposed trades
- Compute conservative rolling risk features
- Enforce deterministic safety gating (ALLOW / RESTRICT / BLOCK)

Design invariants:
- Append-only storage (JSONL)
- Deterministic decisions
- Cold-start is RESTRICT, never ALLOW
- Proposal generation != execution authority
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Literal, Optional


# ---------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------

Decision = Literal["ALLOW", "RESTRICT", "BLOCK"]


@dataclass(frozen=True)
class GateResult:
    """Deterministic gating outcome."""
    decision: Decision
    reasons: List[str]
    constraints: List[str]


# ---------------------------------------------------------------------
# Dataset (append-only)
# ---------------------------------------------------------------------

class LiveTradingDataset:
    """
    Append-only JSONL dataset for live trading governance.

    Stored events:
    - market_context
    - gemini_proposal
    - gate_result
    - trade_outcome (optional, external)
    - execution (optional, external)
    """

    def __init__(self, dataset_path: str) -> None:
        self.path = Path(dataset_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def append_event(self, event_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Append a single immutable event."""
        record = {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "payload": payload,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def read_events(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Read all events or last N events."""
        lines = self.path.read_text(encoding="utf-8").splitlines()
        if limit is not None:
            lines = lines[-limit:]

        events: List[Dict[str, Any]] = []
        for line in lines:
            line = line.strip()
            if line:
                events.append(json.loads(line))
        return events

    def rolling_features(self, window: int = 50) -> Dict[str, Optional[float]]:
        """
        Compute conservative rolling risk metrics
        from recent trade_outcome events.
        """
        if window <= 0:
            raise ValueError("window must be > 0")

        outcomes: Deque[Dict[str, Any]] = deque(maxlen=window)
        for event in self.read_events():
            if event.get("event_type") == "trade_outcome":
                payload = event.get("payload")
                if isinstance(payload, dict):
                    outcomes.append(payload)

        if not outcomes:
            return {
                "n_outcomes": 0,
                "avg_pnl": None,
                "max_drawdown": None,
                "avg_slippage_bps": None,
            }

        pnls = [float(o.get("pnl", 0.0)) for o in outcomes]
        drawdowns = [float(o.get("drawdown", 0.0)) for o in outcomes]
        slippages = [float(o.get("slippage_bps", 0.0)) for o in outcomes]

        return {
            "n_outcomes": len(outcomes),
            "avg_pnl": sum(pnls) / len(pnls),
            "max_drawdown": max(drawdowns),
            "avg_slippage_bps": sum(slippages) / len(slippages),
        }


# ---------------------------------------------------------------------
# Proposal validation
# ---------------------------------------------------------------------

class TradingProposalError(Exception):
    """Raised when a trade proposal violates schema or constraints."""


def validate_trade_proposal(proposal: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize a trading proposal payload."""
    if not isinstance(proposal, dict):
        raise TradingProposalError("proposal must be an object")

    action = proposal.get("action")
    params = proposal.get("params")
    rationale = proposal.get("rationale")

    if action != "propose_trade":
        raise TradingProposalError("action must be 'propose_trade'")
    if not isinstance(params, dict):
        raise TradingProposalError("params must be an object")
    if not isinstance(rationale, str) or not rationale.strip():
        raise TradingProposalError("rationale must be a non-empty string")

    required_fields = [
        "symbol",
        "side",
        "quantity",
        "order_type",
        "max_slippage_bps",
        "stop_loss_pct",
        "take_profit_pct",
    ]
    for field in required_fields:
        if field not in params:
            raise TradingProposalError(f"missing params.{field}")

    symbol = params["symbol"]
    side = params["side"]
    order_type = params["order_type"]

    if not isinstance(symbol, str) or not symbol.strip():
        raise TradingProposalError("symbol must be non-empty string")
    if side not in {"buy", "sell"}:
        raise TradingProposalError("side must be 'buy' or 'sell'")
    if order_type not in {"market", "limit"}:
        raise TradingProposalError("order_type must be 'market' or 'limit'")

    for field in ["quantity", "max_slippage_bps", "stop_loss_pct", "take_profit_pct"]:
        value = params[field]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TradingProposalError(f"{field} must be numeric")
        if float(value) <= 0:
            raise TradingProposalError(f"{field} must be > 0")

    return {
        "action": "propose_trade",
        "params": {
            "symbol": symbol.strip().upper(),
            "side": side,
            "quantity": float(params["quantity"]),
            "order_type": order_type,
            "max_slippage_bps": float(params["max_slippage_bps"]),
            "stop_loss_pct": float(params["stop_loss_pct"]),
            "take_profit_pct": float(params["take_profit_pct"]),
        },
        "rationale": rationale.strip(),
    }


# ---------------------------------------------------------------------
# Deterministic gating logic
# ---------------------------------------------------------------------

def gate_trade_proposal(
    proposal: Dict[str, Any],
    dataset: LiveTradingDataset,
    min_outcomes_for_allow: int = 30,
    max_quantity: float = 100.0,
    max_drawdown_pct: float = 5.0,
    max_slippage_bps: float = 20.0,
) -> GateResult:
    """
    Deterministically gate a trade proposal.

    Policy:
    - Cold start => RESTRICT
    - Hard limit violations => BLOCK
    - Weak performance => RESTRICT
    """
    validated = validate_trade_proposal(proposal)
    params = validated["params"]
    features = dataset.rolling_features(window=min_outcomes_for_allow)

    # Hard limits
    if params["quantity"] > max_quantity:
        return GateResult(
            decision="BLOCK",
            reasons=[f"quantity {params['quantity']} exceeds max {max_quantity}"],
            constraints=["REDUCE_POSITION_SIZE"],
        )

    if params["max_slippage_bps"] > max_slippage_bps:
        return GateResult(
            decision="BLOCK",
            reasons=["slippage tolerance too high"],
            constraints=["TIGHTEN_SLIPPAGE_LIMIT"],
        )

    # Cold start
    if features["n_outcomes"] < min_outcomes_for_allow:
        return GateResult(
            decision="RESTRICT",
            reasons=["insufficient historical outcomes (cold start)"],
            constraints=["LOW_SIZE_ONLY", "HUMAN_REVIEW_REQUIRED"],
        )

    # Drawdown protection
    if features["max_drawdown"] is not None and features["max_drawdown"] > max_drawdown_pct:
        return GateResult(
            decision="BLOCK",
            reasons=[
                f"drawdown {features['max_drawdown']:.2f}% exceeds {max_drawdown_pct:.2f}%"
            ],
            constraints=["PAUSE_AUTOMATION"],
        )

    # Weak performance
    if features["avg_pnl"] is not None and features["avg_pnl"] <= 0:
        return GateResult(
            decision="RESTRICT",
            reasons=["non-positive rolling average pnl"],
            constraints=["REDUCE_SIZE", "LIMIT_FREQUENCY"],
        )

    return GateResult(
        decision="ALLOW",
        reasons=["all deterministic risk checks passed"],
        constraints=[],
    )


# ---------------------------------------------------------------------
# Audit trace persistence
# ---------------------------------------------------------------------

def append_full_decision_trace(
    dataset: LiveTradingDataset,
    proposal: Dict[str, Any],
    gate_result: GateResult,
    execution: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist proposal, gate decision and optional execution outcome."""
    dataset.append_event("gemini_proposal", proposal)
    dataset.append_event(
        "gate_result",
        {
            "decision": gate_result.decision,
            "reasons": gate_result.reasons,
            "constraints": gate_result.constraints,
        },
    )
    if execution is not None:
        dataset.append_event("execution", execution)
