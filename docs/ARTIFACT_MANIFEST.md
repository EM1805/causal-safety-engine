# Causal Safety Engine — Artifact Manifest

This document describes Causal Safety Engine artifacts, their meaning, and intended use.
Artifacts are written under:
- `IMPLEMENTATION/pcb_one_click/out/`

## Core discovery & profiling
| Artifact | Description | Intended use |
|---|---|---|
| `data_clean.csv` | Cleaned dataset used by the pipeline | Reproducibility |
| `data_report.json` | Data quality report and feature-derivation flags | Audit |
| `data_profile.csv` / `data_profile.jsonl` | Per-column profiling | Debugging / QA |
| `edges.csv` | Candidate directed lagged edges with robustness fields | Technical review |
| `insights_level2.csv` / `insights_level2.jsonl` | Human-readable insights derived from edges | Insight consumption |
| `README_OUT.txt` | Short description of the `out/` directory contents | Navigation |

## Monitoring (Level 2.8)
| Artifact | Description | Intended use |
|---|---|---|
| `metrics_level28.json` | Monitoring configuration + counts | Audit |
| `alerts_today_level28.csv` / `alerts_today_level28.jsonl` | Alerts computed for "today" | Operational monitoring |

## Experiment planning (Level 2.9)
| Artifact | Description | Intended use |
|---|---|---|
| `experiment_plan.csv` / `experiment_plan.jsonl` | Suggested actions/trials per insight | Trial planning |
| `experiment_summary_level29.csv` / `.jsonl` | Roll-up of experiment planning | Reporting |


## Trial intake (Level 3.1)
| Artifact | Description | Intended use |
|---|---|---|
| `experiment_intake_level31.csv` | Normalized and quality-gated trial log | Input to Level 3.2 validation |
| `metrics_level31.json` | Counts and gate outcomes for trial intake | Audit |
| `warnings_level31.csv` | Dropped trial rows and reasons | Debugging / QA |

## Trial validation (Level 3.2 / Level 3)
| Artifact | Description | Intended use |
|---|---|---|
| `experiment_results.csv` | Logged trials (actions and observed outcomes) | Input for validation |
| `experiment_trials_enriched_level32.csv` | Trials enriched with matching stats and counterfactual baselines | Audit-grade validation |
| `insights_level3.csv` | Validated insight statuses (e.g., supported / inconclusive) | Decision support |
| `insights_level3_ledger.csv` | Audit trail for validation (matches, distances, baselines) | Compliance / review |

## Reporting
| Artifact | Description | Intended use |
|---|---|---|
| `executive_summary.html` | Human-readable executive summary | Stakeholder review |
| `executive_summary.json` | Machine-readable summary | Integration |

## Evaluation (optional)
These artifacts are provided for review and are **not required** to run PCB.

| Artifact | Description | Intended use |
|---|---|---|
| `EVALUATION/pcb_sensitivity_report.csv` | Sensitivity analysis summary (counts + stability vs baseline) | Governance / robustness review |
| `EVALUATION/SENSITIVITY.md` | How to read and interpret the sensitivity report | Partner evaluation |



## Causal Authority Layer (Priority 1)

PCB emits *authority* metadata for each edge to support governance and AI safety:
- `AUTHORITY/causal_authority.csv` / `.json` with `authority_state`, `reason_code`, `review_action`, and `evidence_summary`.
- `AUTHORITY/authority_summary.json` (counts + generation timestamp).



## Do-Calculus Readiness Layer (Priority 2)

PCB provides a **partner-safe do-calculus readiness** check: it does **not** estimate causal effects or recommend actions.
Instead, it assesses whether hypothetical interventions would be **identifiable** from the current personal causal graph under explicit assumptions.

Artifacts (under `out/DO_CHECK/`):
- `do_identifiability.json` — per-edge identifiability checks (batch over stable/ALLOW edges).
- `proof_trace.json` — audit-friendly trace explaining the reasoning per check.
- `do_summary.json` — counts + metadata for evaluation.

Important:
- `identifiable: true` may still require explicit assumptions (see `assumptions_required`).
- These outputs are intended for governance / review and to **block unsafe use** of causal edges.


### DO_CHECK → AUTHORITY propagation
When `out/DO_CHECK/do_identifiability.json` is produced, PCB propagates do-check fields into `out/AUTHORITY/causal_authority.*` (`do_identifiable`, `do_reason_code`, `do_assumptions`). If not identifiable, `review_action` is escalated to `BLOCK_INTERVENTION`.
