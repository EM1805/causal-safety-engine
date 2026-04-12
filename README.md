 # Causal Safety Engine (CSE)
 
A runtime safety layer for AI systems that blocks actions when causal evidence is not reliable enough.
 

> **Core principle:** when causal identifiability is insufficient, the correct behavior is **causal silence** (no actionable recommendation).
 
**Causal Safety Engine** (CSE) is a runtime safety layer for AI systems that prevents actions when causal reasoning is unreliable.

---

## Why this exists

CSE enforces a strict separation between two phases that are often mixed:
 
1. **Causal discovery** (analysis / insights)
2. **Action authorization** (operational decision)
 

This reduces the risk of:
 
- premature automation
- decision leakage
- causally unjustified interventions
- false positives in high-impact settings
 
---
 
###  Does
 
- Runs a local, deterministic causal pipeline.
- Applies guardrails (bias checks, robustness, stability).
- Produces auditable artifacts in `out/`.
- Exposes an integration-ready machine-readable state (`ALLOW | BLOCK | SILENCE`).
 
### Does not
 
- It is not an autonomous decision agent.
- It does not execute interventions by itself.
- It does not replace human judgment, governance, or compliance.
 
---

## Safety invariants
 
Invariants take precedence over performance and coverage:
 

- Safety gates must not be bypassable through operational config.
- Causal silence is a valid and preferred outcome over weak insights.
- No action authorization without causal and robustness conditions being met.
 
Further reading:
 
- `docs/INVARIANTS.md`
- `docs/DECISION_GATES.md`
- `docs/VIOLATION_PROTOCOL.md`
 
---
 

## Repository structure
 
```text
IMPLEMENTATION/pcb_one_click/
  pcb_cli.py                    # unified CLI
  run_pcb.py                    # legacy runner
  pcb_integration.py            # machine-readable decision contract
  demo.py                       # local demo
  data.csv                      # example dataset
 
integrations/gemini/
  ...                           # Gemini safety-governed integration adapters
 
docs/
  ARCHITECTURE.md
  INTEGRATION_GUIDE.md
  SECURITY.md
  INVARIANTS.md
```
 
---
 
## Requirements
 
- Python 3.7+
- Python dependencies from `IMPLEMENTATION/pcb_one_click/`
 

Quick install:
 
```bash
cd IMPLEMENTATION/pcb_one_click
pip install -r requirements.txt
```
 
---
 
## Quickstart (CLI)

From repository root:

```bash
python IMPLEMENTATION/pcb_one_click/pcb_cli.py init
python IMPLEMENTATION/pcb_one_click/pcb_cli.py run --data IMPLEMENTATION/pcb_one_click/data.csv --skip-32
```

Main commands:

- `init` → prepare `out/`
- `run` → full pipeline (with selective skip flags)
- `plan`, `log`, `eval` → experimental workflow (L2.9)
- `alerts` → daily alerts (L2.8)
- `validate` → level 3.2 validation only
- `decision` → machine-readable integration contract

Full help:

```bash
python IMPLEMENTATION/pcb_one_click/pcb_cli.py --help
```
 
 ---
 
## Code-first integration (recommended)

For production integrations, avoid parsing text logs and consume the JSON contract:
 
```bash
python IMPLEMENTATION/pcb_one_click/pcb_cli.py decision --print-json
 ```

Default output file:

- `out/integration_decision.json`

Example contract:

```json
{
  "schema_version": "v1",
  "generated_at_utc": "2026-01-01T00:00:00Z",
  "decision_state": "ALLOW",
  "decision_reason": "no_active_alerts",
  "metrics": {
    "kept_insights_count": 5,
    "alerts_count": 0
  },
  "artifacts": {
    "insights_level2_csv": "out/insights_level2.csv",
    "alerts_level28_csv": "out/alerts_today_level28.csv"
  }
}
 ```
 
### Decision mapping

- `SILENCE`: no kept insights are available
- `BLOCK`: kept insights exist and at least one alert is active
- `ALLOW`: kept insights exist and no active alerts are present

---

## Main outputs (`out/`)

- `data_clean.csv`
- `edges.csv`
- `insights_level2.csv`
- `alerts_today_level28.csv`
- `experiment_plan.csv`
- `experiment_results.csv`
- `experiment_summary_level29.csv`
- `insights_level3.csv`
- `insights_level3_ledger.csv`
- `experiment_trials_enriched_level32.csv`
- `integration_decision.json`

---

## LLM integrations

The `integrations/gemini/` folder shows a safety-governed pattern:

- LLM generates proposals only
- CSE remains the deterministic evaluation authority
- model-side override is not allowed

---

## Security and reproducibility

- Local-first: no network service exposed by default.
- Auditable per-run artifacts.
- See:
  - `docs/SECURITY.md`
  - `docs/REPRODUCIBILITY.md`
  - `docs/ARTIFACT_MANIFEST.md`

---

## Recommended rollout path for integrators

1. **Shadow mode**: run CSE and log decisions without enforcement.
2. **Soft gate**: block only critical cases.
3. **Hard gate**: full enforcement on `decision_state`.
4. **Audit loop**: artifact retention + periodic policy/config review.

Extended integration guide:

- `docs/INTEGRATION_GUIDE.md`

+---
 
## License
 
See `License.md`.
