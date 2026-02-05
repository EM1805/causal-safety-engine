# OEM / Partner hypotheses (non-binding)

This file is **not** a license and does **not** grant any rights.
It is a lightweight map of how PCB can be packaged with different partner types.

## What OEM means (in this context)
OEM = the partner embeds PCB as a component inside their product (white-label / under-the-hood),
and sells it bundled to their customers. Revenue can be structured as:
- per-seat / per-workspace fee
- per-connector / per-dataset fee
- per-tenant enterprise fee
- consumption-based (runs / rows / jobs)

## Where PCB fits best
PCB is strongest **downstream of integrated data**, when you already have a clean table:
- daily (or periodic) time-series per subject/entity
- numeric signals (usage, behavior, operations, wellbeing, finance, etc.)
PCB converts those into:
- validated, conservative “what likely drives what” insights
- today alerts
- experiment planning & trial logging
- counterfactual-style validation summaries

## Partner models (choose based on who is distributing)
### 1) Data integration & connectors vendors (e.g., “pipelines”)
- PCB runs after ELT/ETL and produces insights for BI/apps.
- Value: “activation layer” on top of integrated data.

### 2) BI / analytics vendors
- PCB becomes a module that produces causal recommendations, not just dashboards.

### 3) Digital health / wellbeing platforms (B2B)
- PCB is a local-first engine for personalized, conservative causal suggestions.
- Important: keep the “not medical / not diagnostic” positioning.

### 4) Device / wearable ecosystems (optional)
- On-device or companion-device compute can run simplified pipeline.
- Typically: data collected on device → processed on phone/desktop → results shown back.
- If a wearable is involved, the OEM would likely request:
  - smaller runtime footprint
  - fixed schemas
  - battery-aware scheduling
  - strict privacy & no-network default

## Non-binding commercial hypothesis (example)
- OEM fee: fixed annual minimum + per-customer royalty
- Or: per-customer annual fee with volume tiers
- Or: revenue share on an “insights add-on” SKU
