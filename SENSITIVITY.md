# Sensitivity Test (Level 2.5 Discovery)

This folder contains an **offline evaluation artifact** that assesses robustness of PCB's discovery stage
to reasonable hyperparameter changes.

**Important:** This is **not** a required runtime output of the pipeline. It is provided to support
audit, governance review, and partner evaluation.

## What was varied
One parameter group at a time in Level 2.5 discovery:
- `min_strength` (edge inclusion threshold)
- `min_support_n` (minimum transition support)
- `max_lag` (temporal lag horizon)

## What is reported
See `pcb_sensitivity_report.csv` for:
- counts of `edges.csv` and `insights_level2.csv`
- Jaccard overlap vs baseline (stability)
- edges/insights added/removed vs baseline

## How to interpret
- Higher overlap means the discovered structure is **stable** to knob changes.
- Stricter thresholds should reduce edges/insights (subset behavior).
- Looser thresholds should add marginal edges without replacing the core structure.

## Why it matters
This test supports the claim that PCB parameters primarily control **how much** the system speaks,
not **what** it concludes, improving trustworthiness for enterprise/OEM settings.
