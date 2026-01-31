# Causal Safety Engine— Security and Privacy

Causal Safety Engine is designed as a **local-first** system.

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

# Security Policy

## Supported Versions
This project is provided as a local-first analytical tool.
No network-exposed services are included.

## Threat Model
- CLI-only execution
- Local filesystem access only
- No remote inputs, no authentication layer
- No multi-tenant or privileged execution

## Reporting a Vulnerability
If you believe you have found a security issue, please contact:
security@<your-domain> (or GitHub private disclosure)

## Known Limitations
Some CLI arguments are used as filesystem paths.
This is a conscious design choice for local-first workflows.

### Threat Model Summary

| Vector | Status |
|------|--------|
| Remote code execution | Not applicable |
| Network attack surface | None |
| Privilege escalation | Not applicable |
| Local file overwrite | Accepted risk (CLI user-controlled) |
| Data exfiltration | Not applicable |

### Accepted Security Risks

- Path traversal warnings reported by static analysis tools are accepted
  for CLI-controlled output paths.
- The tool assumes a trusted local execution environment.
