# PCB — Causal Insight Engine
## Output Contract (v1)

This document defines the **output contract** produced by the PCB Causal Insight Engine.
It is designed to be **auditable, deterministic, and integration-friendly**.

### What “causal” means here (important)
PCB estimates **directed predictive influence** from time-series data (Granger-like logic + conservative falsification).
It does **not** claim RCT-grade causality. Outputs are **hypothesis-generating and decision-support**, not diagnosis.

### Determinism & reproducibility
Given:
- the same input dataset (`data.csv` / `out/data_clean.csv`)
- the same configuration (`pcb.json`, if used)
- the same code version
- the same random seeds (where applicable)

the engine should produce the same set of artifacts in `out/` (within expected numerical tolerances for bootstrap-based steps).

---

## Canonical Artifacts (Output Contract)

### Level 2.6 — Data preparation
**out/data_clean.csv** (REQUIRED when Level 2.6 is run)  
Purpose: cleaned & feature-enriched dataset used downstream.

### Level 2.5 — Insight discovery (validated)
**out/insights_level2.csv** (REQUIRED when Level 2.5 is run)  
Minimum columns:
- `insight_id` (e.g., `I2-00001`)
- `source`
- `target` (default: `mood`)
- `lag`
- `delta_test` (sign is meaningful)
- `strength`
- `support_n` (or equivalent)

Interpretation: sign(`delta_test`) defines expected direction; downstream validation may reject an L2 insight.

### Level 2.8 — Alerts (today)
**out/alerts_today_level28.csv** (OPTIONAL)  
Actionable advisory alerts; must not be framed as medical instruction.

### Level 2.9 — Experiments (plan/log/eval)
**out/experiment_plan.csv** (OPTIONAL)  
**out/experiment_results.csv** (OPTIONAL but IMPORTANT for L3 credibility)  
Append-only log: `insight_id`, `t_index` (or mappable date), `adherence_flag` (0/1), optional `action_name`, `dose`, `notes`.  
**out/experiment_summary_level29.csv** (OPTIONAL)

### Level 3.2 — Counterfactual validation (falsification)
**out/insights_level3.csv** (REQUIRED when Level 3.2 is run)  
Minimum columns:
- `insight_id`, `source`, `target`, `lag`
- `n_trials_scored`
- `success_rate_lb` (conservative lower bound)
- `avg_z_cf` (or equivalent)
- `status` in {`action_supported`, `candidate`, `tentative`, `insufficient`}

**out/insights_level3_ledger.csv** (REQUIRED when Level 3.2 is run)  
Per-trial audit trail: why each trial was scored or skipped (with reasons).

**out/experiment_trials_enriched.csv** (OPTIONAL)  
Enriched trial-level debug/audit output.

---

## Status gates (commercial-grade semantics)
- `action_supported`: supported under logged trials & matching constraints
- `candidate` / `tentative`: do not automate; collect more trials
- `insufficient`: not enough data / matching failures

---

## Integration notes (for data platforms)
- Local-first: no network calls.
- Input: CSV / SQL exports; Output: CSV/JSON-friendly artifacts.
- Designed to run downstream of data integration / BI pipelines.

---

## Versioning
This contract version is: **v1** (`pcb_output_contract_version = v1`).
Any breaking change to filenames or required columns must bump the contract version.
