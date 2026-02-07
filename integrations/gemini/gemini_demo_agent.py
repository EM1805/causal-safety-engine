from gemini_adapter import evaluate_with_causal_engine

gemini_proposal = {
    "proposed_delta": {
        "activity": 1.2,
        "stress": -0.3
    }
}

verdict = evaluate_with_causal_engine(
    proposal=gemini_proposal,
    data_path="IMPLEMENTATION/pcb_one_click/data.csv",
)

print("Causal verdict:", verdict)
