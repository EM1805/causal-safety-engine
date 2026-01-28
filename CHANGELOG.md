# Changelog

All notable changes to the **Causal Safety Engine SDK** are documented in this file.

This project follows the principles of **Keep a Changelog**  
and adheres to **Semantic Versioning (MAJOR.MINOR.PATCH)**.

The Causal Safety Engine is an **industrial-grade SDK**
for causal discovery, safety validation, and robustness certification
designed for regulated and high-risk AI systems.

---

## [1.1.1] – 2026-01-30

### Fixed
- Improved robustness when analyzing datasets affected by distribution drift
- Stabilized execution in the presence of partial or degraded causal signals
- Improved handling of datasets failing safety validation criteria
- Reduced false-positive execution failures during robustness checks

### Improved
- Clearer separation between certified and non-certifiable datasets
- More deterministic execution behavior across repeated runs
- Improved internal consistency of generated output artifacts

### Notes
- No changes to the core causal discovery logic
- Focused on stability and robustness improvements

---

## [1.1.0] – 2026-01-29

### Added
- Enhanced causal robustness validation mechanisms
- Extended safety checks for data leakage and spurious correlations
- Improved multi-level insight generation pipeline
- Additional structured output artifacts for audit and review
- Run-level isolation for deterministic analysis

### Changed
- Standardized internal execution flow across datasets
- Unified output artifact structure for certified analyses

### Notes
- This release strengthens the engine’s suitability for industrial and OEM integration

---

## [1.0.1] – 2026-01-28

### Fixed
- Minor execution stability issues in batch analysis
- Improved tolerance to missing or partially observed variables
- Improved CSV ingestion robustness

### Notes
- Maintenance release improving overall execution reliability

---

## [1.0.0] – 2026-01-28

### Added
- Initial release of the Causal Safety Engine SDK
- Core causal discovery and safety validation engine
- Automated detection of:
  - causal relationships
  - data leakage
  - distribution drift
  - spurious correlations
- Multi-level causal insight generation
- Safety and robustness indicators
- Structured, audit-ready output artifacts
- Deterministic and reproducible execution pipeline

### Design Principles
- Offline-first execution
- On-premise deployable
- Deterministic behavior
- Audit-ready outputs
- OEM and enterprise integration focused

### Notes
- This release establishes the foundation for certified causal safety analysis
- Designed for regulated, safety-critical, and high-stakes environments

---

## Planned (Roadmap – Non-binding)

### [1.2.0]
- Formal input schema validation
- Explicit causal target configuration
- Extended diagnostics for non-certifiable datasets

### [1.3.0]
- Versioned output schemas
- Formal safety score aggregation
- Extended robustness and stress testing capabilities

---

## Licensing Notes

- The Causal Safety Engine is distributed as a **licensed SDK**
- Intended for integration within OEM and enterprise systems
- Redistribution and embedding subject to licensing terms

---