"""Minimal demo for Gemini adapter + PCB engine evaluation."""

import json

from gemini_adapter import evaluate_with_causal_engine


def main() -> None:
    proposal = {
        "action": "adjust_features",
        "params": {
            "deltas": {
                "activity": -0.2,
                "stress": -0.3,
            }
        },
        "rationale": "High stress: reduce load conservatively.",
    }

    verdict = evaluate_with_causal_engine(
        proposal=proposal,
        data_path="IMPLEMENTATION/pcb_one_click/data.csv",
    )

    print("=== Proposal ===")
    print(json.dumps(verdict["proposal"], indent=2))
    print("\n=== Engine verdict ===")
    print(json.dumps({k: v for k, v in verdict.items() if k != "proposal"}, indent=2))


if __name__ == "__main__":
    main()
