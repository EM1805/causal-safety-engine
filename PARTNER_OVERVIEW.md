# PCB Partner Package (multi-partner ready)

This package is prepared to be shareable with **multiple partner types** (non-exclusive conversations),
with clear separation between:
- code (local-first engine)
- output contract (audit / procurement)
- demo & reproducibility steps

## What problem does PCB solve?
Many platforms integrate data well, but customers still ask:
- “What actually drives the outcome we care about?”
- “Which actions are likely to improve it?”
- “How do we validate it without full RCTs?”

PCB provides a **conservative, auditable** “causal-ish activation layer”:
- discovers candidate directed influences (Granger-like)
- generates actionable alerts
- supports lightweight experiments (plan/log/eval)
- increases confidence via counterfactual-style falsification (Level 3.2)

## What you can do with it (partner POV)
- ship as an add-on “Causal Recommendations”
- bundle as an OEM component
- use as a behind-the-scenes engine powering your UX

## Minimal integration surface
- Input: a CSV extract (or any SQL → CSV export)
- Output: CSV + JSON artifacts in `out/` (see CAUSAL_CONTRACT.md)

## Where to start
1) `IMPLEMENTATION/pcb_one_click/README_QUICKSTART.md`
2) Run demo: `python demo.py` or `python pcb_cli.py run`
3) Open: `out/executive_summary.html` (after running summary generator)

## Notes
- Local-first; no external network calls.
- “Causality” is defined as directed predictive influence + falsification (not RCT causality).

**Causal Authority Layer (optional):** run outputs may include `AUTHORITY/causal_authority.csv/json`, providing formal `stable/unstable` states for each edge plus conservative `intervention_unsafe` flags for governance and audit.


## Do-Calculus Readiness (Priority 2)

PCB includes a partner-safe do-calculus readiness check that assesses whether hypothetical interventions are identifiable from the current personal causal graph, without estimating effects or recommending actions. Outputs are written under `out/DO_CHECK/`.
