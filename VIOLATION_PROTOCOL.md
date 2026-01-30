# VIOLATION_PROTOCOL.md
## PCB — Violation Protocol (Mandatory Handling)

A violation occurs when an automated system **bypasses, ignores, or overrides** PCB constraints.

Violations are not "edge cases". They are governance events.

---

## 1) What Counts as a Violation
A violation MUST be recorded when any of the following happen:

- An action is executed after PCB returned **BLOCK**
- A **RESTRICT** decision is executed without enforcing all constraints
- The integrator changes constitution documents without version pinning
- Downstream systems attempt to substitute population priors for individual evidence (Invariant 7)
- Any manual override occurs without a recorded justification (Invariant 8)

---

## 2) Immediate Response (Fail-Closed)
On violation detection, the integration MUST:

1. **Halt** the action (if still pending)
2. Enter a **SAFE MODE** for that individual/context:
   - default future decisions to RESTRICT/BLOCK
   - prevent escalation classes (plan change / automation / sensitive)
3. Emit a violation record to disk (audit-grade)

---

## 3) Required Violation Record (JSON)
A violation record MUST include:

```json
{
  "violation_id": "uuid",
  "timestamp": "ISO-8601",
  "integrator": {"system": "name", "version": "x.y.z"},
  "constitution": {"hash": "...", "version": "pinned"},
  "pcb_decision": "BLOCK | RESTRICT",
  "executed_action": {"class": "...", "description": "..."},
  "missing_constraints": ["..."],
  "reason": "human-entered justification or policy trigger",
  "artifacts": {
    "graph_ref": "...",
    "audit_ref": "...",
    "proof_ref": "..."
  },
  "accountability": {"actor": "human|system", "id": "..." }
}
```

---

## 4) Notification Policy (Configurable)
By default, PCB SHOULD support one or more notification channels (local logs are always required):

- local on-device alert (for user-owned deployments)
- operator/compliance notification (enterprise deployments)
- optional “integration health” telemetry **without** raw personal data

---

## 5) Recovery Procedure
A system may exit SAFE MODE only if:

- the integrator demonstrates enforcement of the last violated constraints
- the constitution version remains pinned
- a recovery record is written, referencing the original violation_id

---

## 6) Non-Repudiation
Violations MUST be:
- versioned
- attributable
- immutable (append-only logs)

If an integrator cannot produce violation records, PCB’s constraints cannot be trusted as enforced.

---

## 7) Principle
> **No silent overrides. No invisible harm.**
