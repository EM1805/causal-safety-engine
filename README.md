# PCB — Personal Causal Blueprint

PCB is a **local-first causal insight & validation engine** designed for
**auditability, conservative epistemics, and AI governance**.

## Constitutional Artifacts

PCB is governed by four normative documents:

- `CAUSAL_CONSTITUTION.md`
- `INVARIANTS.md`
- `DECISION_GATES.md`
- `VIOLATION_PROTOCOL.md`

Integrations should pin and reference these documents; they define non-negotiable causal constraints and enforcement semantics.


---

## 👉 How to run (important)

**You should run PCB from the IMPLEMENTATION folder.**

The `CANONICAL/` directory is a **reference artifact**:
- it defines the *correct, auditable behavior*
- it is **not** the entry point for demos
- you do **not** need to run anything inside it

### Quick start (recommended)

```bash
cd pcb_bundle_v1.0/IMPLEMENTATION/pcb_one_click
python demo.py
```

This will:
- run the pipeline end-to-end
- generate artifacts under `IMPLEMENTATION/pcb_one_click/out/`
- demonstrate discovery, monitoring, and (when data allows) validation

---

## Folder roles (do not mix them)

| Folder | Purpose |
|------|---------|
| **IMPLEMENTATION/** | Runnable build (demo / integration entry point) |
| **CANONICAL/** | Reference, audit, invariants, and expected behavior |
| **docs / root files** | Contractual documentation |

If you are evaluating or demoing PCB, **always use IMPLEMENTATION**.

---

## What PCB does (and does not)

PCB:
- infers **directed predictive influences** from longitudinal data
- supports monitoring, experiment planning, and validation
- prefers **silence over false positives**

PCB does **not**:
- claim clinical or ontological causality
- auto-apply interventions
- guarantee counterfactual outcomes

See `INVARIANTS.md` for non-negotiable guarantees.
