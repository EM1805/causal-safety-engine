## Gemini Integration (Safety-Governed)

This integration connects **Gemini** to the **Causal Safety Engine** in a
**strictly governed, safety-first** manner.

Gemini is used **only** to generate a **proposal JSON**.
It never executes actions and never makes decisions.

The **deterministic Causal Safety Engine (PCB CLI)** is the sole authority that:
- evaluates proposals
- enforces causal guardrails
- decides whether an action is allowed, blocked, or silenced

Gemini has:
- no execution privileges
- no policy access
- no ability to override engine decisions

---

### Canonical proposal schema

All proposals are normalized to the following canonical schema before evaluation:

```json
{
  "action": "adjust_features",
  "params": {
    "deltas": {
      "activity": -0.2,
      "stress": -0.3,
      "sleep": 0.1
    }
  },
  "rationale": "short explanation"
}
