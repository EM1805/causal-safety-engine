# Causal Safety Engine  


## Overview

**Causal Safety Engine** (CSE) is a runtime safety layer for AI systems that prevents actions when causal reasoning is unreliable.
Instead of improving model predictions, CSE introduces a decision gate that evaluates whether a system’s output is supported by stable causal signals. If causal stability is insufficient, the system abstains from acting.


## Design Principle: Causal Silence

When causal identifiability is insufficient, the engine **intentionally produces no insights**.
Silence is treated as a correct and safe outcome, not a failure.

---

## Intervention Safety & Action Blocking

The Causal Safety Engine **never authorizes interventions by default**.

Causal discovery and causal action are treated as **strictly separate phases**.
Even when exploratory or tentative causal signals exist, the engine:

- does **not** recommend actions
- does **not** generate intervention plans
- does **not** expose “what-to-do” outputs

Interventions are **explicitly blocked** unless all of the following conditions are met:

- causal identifiability is satisfied
- robustness and stability tests pass
- no safety or silence gate is triggered
- the run is explicitly marked as *intervention-enabled*

When causal certainty is insufficient, the correct and safe behavior is **causal silence**:
no insights promoted, no actions suggested, no downstream activation.

This design prevents unsafe automation, decision leakage, and premature causal deployment
in high-stakes or regulated environments.

## What This Engine Is Not

- Not an AI assistant
- Not a decision-making system
- Not an intervention recommender
- Not an optimization or automation engine
- Not a replacement for human or regulatory judgment

---

## Key Capabilities

###  True Causal Discovery
- Identifies genuine causal relationships
- Rejects spurious correlations
- Handles confounders and common causal biases

###  Causal Safety & Guardrails
- Explicit rejection of:
  - Simpson’s paradox
  - collider bias
  - data leakage
  - spurious time trends
- Safety-first default behavior (no false positives by design)

###  Robustness & Stability
- Automated testing for:
  - stress scenarios
  - multi-run stability
  - reproducibility
- Consistent outputs under data perturbations

###  Audit & Certification Ready
- Every run is:
  - isolated
  - hashed
  - traceable
- Artifacts are preserved for verification and compliance



## Governance & Safety Invariants

The engine enforces non-negotiable safety invariants:

- Safety gates cannot be bypassed by configuration
- Causal silence overrides downstream demand for outputs
- No action authorization without explicit intervention enablement

These invariants take precedence over performance, convenience, or coverage.

---

## Repository Structure

```
IMPLEMENTATION/
  pcb_one_click/
    demo.py              # core causal engine
    data.csv             # example dataset
    stress_test/         # safety & stability tests


runs/
  <run_id>/
    data.csv
    out/
      edges.csv
      insights_*.csv
```
## Security

Causal Safety Engine is designed as a local-first analytical engine.
It does not expose network services and does not process untrusted remote input.


## Project Status

- Engine: **production-ready reference implementation**
- CI/CD: **fully automated**
- Safety & stability: **certified via tests**
