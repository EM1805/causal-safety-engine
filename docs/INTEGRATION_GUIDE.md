# Integration Guide

This guide explains a practical, low-friction way to integrate the Causal Safety Engine (CSE) in existing systems.

## Goal

Use CSE as a **decision gate** in front of any action executor.

- Your application can still produce candidate actions.
- CSE validates causal reliability and safety constraints.
- Only CSE-approved outputs can reach execution.

## Recommended architecture

Adopt a 4-stage pipeline:

1. **Propose**: your model/agent/app creates a proposal.
2. **Normalize**: convert proposal to a stable JSON schema.
3. **Evaluate**: run CSE (`run_pcb.py` / `pcb_cli.py`) on data + proposal.
4. **Enforce**: if blocked/silenced, do not execute; if approved, continue.

This keeps integration simple: your current business logic stays mostly unchanged,
and CSE is inserted as a deterministic gate.

## Minimal proposal contract

Standardize proposals to:

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

Why this helps integration:

- a single schema for all upstream systems
- easier validation and observability
- reduced adapter complexity

## Fast integration checklist

1. **Wrap CSE in a single function/service boundary**
   - e.g. `evaluate_proposal(data_path, proposal_json) -> decision`
2. **Make CSE output authoritative**
   - executor must require explicit `allow` state
3. **Persist artifacts per run**
   - store `run_id`, policy hash, and output folder
4. **Surface only 3 downstream states**
   - `ALLOW`, `BLOCK`, `SILENCE`
5. **Start in shadow mode**
   - evaluate and log CSE decisions without executing
6. **Promote to hard enforcement**
   - once false-positive and stability checks are acceptable

## Operational tips

- Keep data interfaces immutable per version.
- Version proposal schema and policy together.
- Add contract tests for adapters before end-to-end tests.
- Treat causal silence as a valid product outcome.

## Anti-patterns to avoid

- Allowing bypass flags in production.
- Letting LLM output directly reach executors.
- Mixing causal discovery and intervention authorization in one step.
- Ignoring artifact retention (auditability loss).

## Example integration flow (pseudo-code)

```python
proposal = upstream_agent.generate_proposal(context)
normalized = normalize_proposal(proposal)
decision = cse.evaluate(data_path="data.csv", proposal=normalized)

if decision.state == "ALLOW":
    execute(decision.authorized_action)
elif decision.state in {"BLOCK", "SILENCE"}:
    stop_and_log(decision)
```

## Migration path for existing systems

- **Phase 1**: add schema normalization and run CSE in parallel.
- **Phase 2**: require CSE decision before execution.
- **Phase 3**: enforce CI checks for invariants and determinism.
- **Phase 4**: publish run artifacts to your compliance/audit store.

This phased approach usually minimizes integration effort while preserving safety guarantees.
