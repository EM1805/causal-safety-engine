# Demo Guide (partner-friendly)

## 1) Run the pipeline (one command)
From `IMPLEMENTATION/pcb_one_click/`:

- Windows: `demo.bat`
- macOS/Linux: `bash demo.sh`

Or directly:
- `python pcb_cli.py run`

## 2) Generate the Executive Summary HTML
- `python pcb_exec_summary_ui.py`

This writes:
- `out/executive_summary.html`
- `out/executive_summary.json`

Open the HTML file in a browser.

## 3) What to look at
- `out/insights_level2.csv` (discovery candidates)
- `out/alerts_today_level28.csv` (actionable alerts)
- `out/experiment_plan.csv` (suggested trials)
- `out/experiment_results.csv` (trial log)
- `out/insights_level3.csv` + `out/insights_level3_ledger.csv` (validation + audit trail)

## 4) Reproducibility
Run twice with the same inputs; outputs should be stable (within bootstrap tolerance).


### Do-Calculus readiness outputs
After running the demo, review `out/DO_CHECK/do_summary.json` and `out/DO_CHECK/do_identifiability.json` for intervention identifiability checks over stable edges.
