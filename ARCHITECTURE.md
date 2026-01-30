# PCB — Architecture

This document describes PCB at a system level (not algorithm internals).

## High-level pipeline

```
Raw longitudinal data
        |
        v
[Level 2.6] Data profiling + conservative feature derivation
        |
        v
[Level 2.5] Directed lagged discovery (candidate edges)
        |
        v
[Level 2.0] Insight compilation (human-readable statements)
        |
        +------------------------------+
        |                              |
        v                              v
[Level 2.8] Monitoring / Alerts   [Level 2.9] Experiment planning
        |                              |
        +--------------+---------------+
                       v
              Logged trials (experiment_results.csv)
                       |
                       v
            [Level 3.1] Trial intake / normalization
                       |
                       v
            [Level 3.2] Counterfactual matching + validation
                       |
                       v
            [Level 3] Validated insight statuses + audit ledger
```

Outputs are written to:
- `CANONICAL/src/out/`
- `CANONICAL/pcb_one_click/out/`

## Key design properties
- Conservative epistemics: prefers silence over false positives.
- Auditability: artifacts + validation ledger for review.
- Local-first: no mandatory network calls.
- Model-agnostic: can audit actions from AI systems, humans, or hybrid systems.
