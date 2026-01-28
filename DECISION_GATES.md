# DECISION_GATES.md
## PCB — Decision Gates (Operational Enforcement)

This document defines how PCB applies the constitution and invariants to **proposed actions**.

PCB does not generate user-facing text. It returns **structured constraints** that downstream systems must respect.

---

## 1) Gate Outcomes
Every proposed action MUST be classified as one of:

### ALLOW
Action is causally safe for the individual under current evidence.

### RESTRICT
Action may proceed only under explicit constraints (scope, intensity, frequency, duration, required disclaimers, required human review).

### BLOCK
Action is causally unsafe **or** not causally identifiable with sufficient confidence.

---

## 2) Default Rule (Conservative)
If evidence is insufficient to validate causal safety, PCB MUST return **RESTRICT** or **BLOCK** (see Invariant 6).

---

## 3) Action Classes
Integrations should map their actions into classes. Minimum recommended classes:

1. **Advice / Recommendation**
2. **Nudging / Repeated prompting**
3. **Plan change / escalation**
4. **Automation / execution** (an action that changes the world)
5. **Sensitive domain action** (health, finance, education, minors)

---

## 4) Gate Criteria (Non-Exhaustive)
PCB evaluates proposed actions against:

- **Protected state risk** (Invariant 1)
- **Causal sufficiency** of drivers (Invariant 2)
- **Fragility / non-identifiability** (Invariant 3)
- **Temporal respect** (Invariant 4)
- **Loop reinforcement risk** (Invariant 5)
- **Population override attempts** (Invariant 7)
- **Auditability completeness** (Invariant 9)
- **Optimization pressure** (Invariant 10)

---

## 5) Restrictions Catalog (What RESTRICT can mean)
When PCB returns RESTRICT, it may include one or more of:

- **SCOPE_LIMIT**: narrow the action scope to low-risk sub-actions
- **FREQUENCY_CAP**: cap repetitions per day/week
- **ESCALATION_LOCK**: forbid escalation beyond a bounded intensity
- **REQUIRE_ACK**: require explicit user acknowledgment / informed consent language
- **REQUIRE_HUMAN_REVIEW**: require clinician/coach/operator review
- **DATA_REQUEST**: request additional safe signals to reduce uncertainty (non-invasive)

---

## 6) Standard Response Payload (JSON)
Downstream systems should treat this payload as authoritative.

```json
{
  "decision": "ALLOW | RESTRICT | BLOCK",
  "action_class": "advice | nudge | plan_change | automation | sensitive",
  "constraints": [
    {"type": "FREQUENCY_CAP", "value": "max_1_per_day"},
    {"type": "SCOPE_LIMIT", "value": "sleep_hygiene_only"}
  ],
  "reasons": [
    {"invariant": "Invariant 3", "code": "FRAGILE_EDGE", "detail": "relationship sleep→anxiety unstable"},
    {"invariant": "Invariant 6", "code": "INSUFFICIENT_EVIDENCE", "detail": "cold-start window"}
  ],
  "artifacts": {
    "graph_ref": "out/personal_causal_graph.json",
    "audit_ref": "out/audit_log.jsonl",
    "proof_ref": "out/proof_trace.json"
  },
  "constitution_version": "pinned",
  "timestamp": "ISO-8601"
}
```

---

## 7) Hard Rules
- If **BLOCK**, downstream systems MUST NOT execute the action.
- If **RESTRICT**, downstream systems MUST apply all constraints or treat as BLOCK.
- If an integrator overrides a gate, it MUST trigger the **Violation Protocol**.

---

## 8) Minimum Gate Set for PoC
If you need a minimal integration, implement only these gates:

- BLOCK when: fragile/non-identifiable (Invariant 3) OR uncertainty default (Invariant 6) on sensitive actions
- RESTRICT when: mild risk to protected states (Invariant 1) without loop reinforcement
- ALLOW when: stable causal support and low-risk class
