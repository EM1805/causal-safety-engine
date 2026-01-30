# PCB — Security and Privacy

PCB is designed as a **local-first** system.

## Network
- No mandatory network calls
- No telemetry collection required for operation
- All inputs and outputs remain on the executing machine

## Local orchestration
Run scripts and smoke tests may invoke **local subprocesses** to orchestrate pipeline steps.
These subprocesses do not imply network activity.

## Artifacts
- Artifacts are written to local disk only (under `CANONICAL/**/out/`)
- Users are responsible for storage, access control, and any organizational compliance requirements
