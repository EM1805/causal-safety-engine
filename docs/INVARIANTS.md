# PCB — Constitutional Causal Invariants

These invariants define **non-negotiable causal constraints** governing any automated system that acts on an individual **over time**.

They are **domain-agnostic**, **model-agnostic**, and **implementation-independent**.

---

## Invariant 1 — Non-Regression of Protected States
A system **MUST NOT** authorize actions that cause a sustained degradation of a protected individual state relative to the individual’s own historical baseline.

- Evaluation is **longitudinal**, not single-step.
- Short-term gains that lead to medium/long-term degradation **violate** this invariant.

## Invariant 2 — Causal Sufficiency Requirement
A system **MUST NOT** base repeated or escalating actions on variables that lack validated causal influence for that individual.

- Correlational signals may be observed.
- Correlational signals **must not** be used as decision drivers unless causally validated.

## Invariant 3 — Fragility-Aware Constraint
Actions relying on causally fragile, unstable, or non-identifiable relationships **MUST** be restricted, downgraded, or blocked.

Fragility includes:
- instability across time
- sensitivity to confounders
- insufficient longitudinal support

## Invariant 4 — Temporal Respect
A system **MUST NOT** compress, accelerate, or shortcut processes that require biological, cognitive, or social time to unfold.

- Escalation velocity must respect empirically observed temporal constraints.

## Invariant 5 — Feedback Loop Prevention
A system **MUST NOT** reinforce self-amplifying causal loops that degrade individual stability over time.

- Loop detection applies even if single actions appear harmless.

## Invariant 6 — Uncertainty-First Default
In the absence of sufficient causal validation, the system **MUST** default to restriction or non-action.

- Silence/refusal/neutrality are valid outcomes.
- Lack of evidence is treated as a safety condition.

## Invariant 7 — Individual Scope Supremacy
Population-level patterns **MUST NOT** override the individual causal model when making decisions affecting that individual.

- Aggregate statistics may inform priors.
- Individual causal evidence always has precedence.

## Invariant 8 — Non-Delegability of Causal Authority
No downstream system may override causal constraints without producing an explicit, auditable violation record.

- Overrides are treated as constitutional violations.
- All overrides must be logged, versioned, and attributable.

## Invariant 9 — Auditability
Every permitted, restricted, or blocked action **MUST** be causally traceable to specific invariant evaluations.

- No opaque decisions.
- No unverifiable approvals.

## Invariant 10 — Non-Optimization Principle
The system **MUST NOT** optimize for engagement, performance metrics, or behavioral change at the expense of causal safety.

- Safety invariants override all optimization objectives.

---

## Closing
If a decision cannot be explained causally (in terms of these invariants), it cannot be trusted as safe for that individual over time.


---

# Appendix A — Engineering / Implementation Invariants

PCB — System Invariants

This document defines **non-negotiable invariants**. Any version that violates them is **not PCB**.

## Epistemic invariants
- PCB estimates **directed predictive influence** (Granger-like), not ontological/clinical causality.
- PCB does **not** guarantee intervention validity or counterfactual correctness.
- Outputs are **hypothesis-generating** and must not be used as diagnosis or prescriptive advice.
- When evidence is insufficient, PCB prefers **silence over false positives**.

## Data invariants
- No future leakage: PCB must not use future information to explain/predict the past.
- Missingness is handled conservatively; invalid transitions are rejected rather than hallucinated.
- Any derived features must be time-causal (e.g., prev/lag features, not future-aware smoothing).

## Statistical invariants
- Directionality requires temporal precedence (explicit lags).
- Edges/insights must pass minimum support and robustness/stability thresholds.
- Unstable, sign-inconsistent, or poorly-supported effects are rejected or downgraded.

## Execution invariants
- Local-first execution: no mandatory network calls.
- Deterministic given fixed configuration and seed (where randomness is used).
- Artifacts are written explicitly to disk for audit and reproducibility.
