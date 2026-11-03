"""
Live Trading Governance Demo
============================

Demonstrates deterministic trade gating over a live-updating
historical dataset.

Cases shown:
1) RESTRICT  — cold start (insufficient history)
2) BLOCK    — excessive position size
3) BLOCK    — excessive slippage tolerance

This demo does NOT execute trades.
It only shows safety decisions and audit traces.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from live_trading_dataset import (
    LiveTradingDataset,
    append_full_decision_trace,
    gate_trade_proposal,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def print_case(title: str, payload: Dict) -> None:
    """Pretty-print one demo case."""
    print(f"\n=== {title} ===")
    print(json.dumps(payload, indent=2))


# ---------------------------------------------------------------------
# Demo scenarios
# ---------------------------------------------------------------------

def demo_restrict_cold_start(dataset: LiveTradingDataset, dataset_path: Path) -> None:
    """Cold-start proposal -> RESTRICT."""
    proposal = {
        "action": "propose_trade",
        "params": {
            "symbol": "AAPL",
            "side": "buy",
            "quantity": 10,
            "order_type": "market",
            "max_slippage_bps": 5,
            "stop_loss_pct": 1.0,
            "take_profit_pct": 2.0,
        },
        "rationale": "Breakout attempt with bounded downside.",
    }

    decision = gate_trade_proposal(proposal, dataset)
    append_full_decision_trace(dataset, proposal, decision)

    print_case(
        "RESTRICT — cold start",
        {
            "decision": decision.decision,
            "reasons": decision.reasons,
            "constraints": decision.constraints,
            "dataset": str(dataset_path),
        },
    )


def demo_block_quantity(dataset: LiveTradingDataset) -> None:
    """Excessive position size -> BLOCK."""
    proposal = {
        "action": "propose_trade",
        "params": {
            "symbol": "AAPL",
            "side": "buy",
            "quantity": 500,
            "order_type": "market",
            "max_slippage_bps": 5,
            "stop_loss_pct": 1.0,
            "take_profit_pct": 2.0,
        },
        "rationale": "High-conviction oversized position.",
    }

    decision = gate_trade_proposal(proposal, dataset)
    append_full_decision_trace(dataset, proposal, decision)

    print_case(
        "BLOCK — position size",
        {
            "decision": decision.decision,
            "reasons": decision.reasons,
            "constraints": decision.constraints,
        },
    )


def demo_block_slippage(dataset: LiveTradingDataset) -> None:
    """Excessive slippage tolerance -> BLOCK."""
    proposal = {
        "action": "propose_trade",
        "params": {
            "symbol": "AAPL",
            "side": "sell",
            "quantity": 5,
            "order_type": "market",
            "max_slippage_bps": 50,
            "stop_loss_pct": 1.0,
            "take_profit_pct": 2.0,
        },
        "rationale": "Fast exit regardless of market impact.",
    }

    decision = gate_trade_proposal(proposal, dataset)
    append_full_decision_trace(dataset, proposal, decision)

    print_case(
        "BLOCK — slippage tolerance",
        {
            "decision": decision.decision,
            "reasons": decision.reasons,
            "constraints": decision.constraints,
        },
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    dataset_path = Path("out/live_trading_dataset.jsonl")
    dataset = LiveTradingDataset(str(dataset_path))

    # Simulate one market snapshot (audit-only, no execution)
    dataset.append_event(
        "market_data",
        {
            "symbol": "AAPL",
            "close": 220.15,
            "spread_bps": 1.8,
            "volatility_regime": "medium",
        },
    )

    demo_restrict_cold_start(dataset, dataset_path)
    demo_block_quantity(dataset)
    demo_block_slippage(dataset)


if __name__ == "__main__":
    main()
