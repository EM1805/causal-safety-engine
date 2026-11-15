# Scoped intervention evidence

Level 3.2 can report `action_supported` for an explicit operator-owned
scope. An unscoped run preserves historical statistics as `candidate`.
The scope names one insight, action, dose, outcome, calendar-day horizon
and regime starting on a specified date.

```json
{
  "insight_id": "I2-00001",
  "action_name": "increase_treatment",
  "dose": "unit-1",
  "target_col": "target",
  "lag_days": 1,
  "regime_id": "current",
  "regime_start": "2025-01-01"
}
```

Record completed trials with the corresponding horizon and regime. Use
`--adherence-verified` only when the operator has checked the observation
in the original record. It records an attestation supplied by the caller.
The dataset must contain unique dates in chronological order and use the
same row indices that were used to log the trials.

```sh
python pcb_cli.py plan
python pcb_cli.py log --insight-id I2-00001 --action-name increase_treatment \
  --date 2025-04-16 --dose unit-1 --lag-days 1 --regime-id current \
  --adherence-flag 1 --adherence-verified --data data.csv
python pcb_cli.py intake
python pcb_cli.py validate --data data.csv --scope scope.json
```

The validator checks the scope against the Level 2 insight, outcome and
lag. It considers only matching, performed and verified trials in the
requested regime and excludes duplicate trial IDs or event indices. It
uses the exact outcome date for the trial and its matched historical
controls. Controls with a known intervention between their context date
and outcome date are excluded. The output records the scope and its
digest. The caller remains responsible for verifying logged adherence,
regime boundaries and input provenance and for supplying appropriate
pre-action covariates.

Intake preserves the fields used by validation and records source and
output hashes. Validation uses the current log when a previously generated
intake no longer matches it. Legacy log rows can be inspected, with stable
IDs added by intake when the original fields support it. Scoped promotion
requires explicit scope fields and an adherence attestation.

`action_supported` summarizes matched trial evidence under the declared
scope. Operational authorization remains a separate decision.

```sh
python IMPLEMENTATION/pcb_one_click/stress_test/test_scoped_intervention.py -v
```
