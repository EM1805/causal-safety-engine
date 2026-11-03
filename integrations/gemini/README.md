## Gemini Integration (Safety-Governed)

Gemini is used only to generate a proposal JSON.

The deterministic Causal Safety Engine (PCB CLI) is the execution authority:
- Gemini cannot execute commands.
- Gemini cannot override policy decisions.

### Canonical proposal schema
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
```

### Adapter behavior

`gemini_adapter.py`:
- validates and normalizes incoming payloads (`normalize_proposal`)
- supports legacy payloads (`proposed_delta`) for compatibility
- stores artifacts for auditing:
  - `out/gemini_proposal.json`
  - `out/gemini_audit_record.json`
- executes PCB CLI with timeout control (`timeout_seconds`, default `120`)

This integration demonstrates governed AI proposals, not autonomous decision-making.
