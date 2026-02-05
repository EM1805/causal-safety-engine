# PCB — Reproducibility

PCB aims to be reproducible, but reproducibility depends on:
1) configuration/seeds, and
2) dependency versions.

## Determinism
- Keep seeds/config constant for stable outputs.
- Some steps may involve randomness (e.g., bootstrap); seeds control this.

## Dependencies (recommended for enterprise)
Use a dedicated environment and pin versions:
- `pip freeze > requirements.lock`
- Re-install from the lockfile for consistent reruns.

## Artifact-based reproducibility
Key artifacts supporting audit and replay:
- `run_metadata.json`
- `data_clean.csv`
- `data_report.json`
- `insights_level3_ledger.csv` (when validation is used)
