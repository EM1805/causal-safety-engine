from gemini_adapter import evaluate_with_causal_engine

# Simulated Gemini output (JSON-only, no prose)
gemini_proposal = {
    "action": "adjust_threshold",
    "params": {"delta": 0.15},
    "rationale": "Reduce false negatives"
}

verdict = evaluate_with_causal_engine(
    proposal=gemini_proposal,
    data_path="data/observational.csv",
)

print("Causal verdict:", verdict)

if verdict["decision"] == "ALLOW":
    print("Action allowed → execution possible")
else:
    print("Action blocked or silenced → no execution")
