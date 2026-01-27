# Causal Safety Engine  
**Industrial-grade causal discovery and safety certification engine**

---

## Overview

**Causal Safety Engine** is an industrial-grade engine for causal discovery and **certification of reliable insights**, designed for enterprise environments, regulated AI systems, and deep-tech startups that require:

- causality (not correlation)
- robustness
- multi-run stability
- auditability
- API-based integration

The system is designed as a **causal safety layer** on top of existing AI/ML pipelines.

---

## Key Capabilities

### ✔ True Causal Discovery
- Identifies genuine causal relationships
- Rejects spurious correlations
- Handles confounders and common causal biases

### ✔ Causal Safety & Guardrails
- Explicit rejection of:
  - Simpson’s paradox
  - collider bias
  - data leakage
  - spurious time trends
- Safety-first default behavior (no false positives by design)

### ✔ Robustness & Stability
- Automated testing for:
  - stress scenarios
  - multi-run stability
  - reproducibility
- Consistent outputs under data perturbations

### ✔ Audit & Certification Ready
- Every run is:
  - isolated
  - hashed
  - traceable
- Artifacts are preserved for verification and compliance

### ✔ API-First Architecture
- Engine exposed as a **service**
- Easy integration into enterprise pipelines
- Ready for industrial deployment

---

## Architecture

```
Client / System
      |
      |  POST /causal/run
      v
Causal Safety API (FastAPI)
      |
      |  isolated execution
      v
pcb_one_click engine
      |
      |  artifacts
      v
edges.csv / insights.csv
```

Each execution runs in an isolated directory identified by a unique `run_id`.

---

## Repository Structure

```
IMPLEMENTATION/
  pcb_one_click/
    demo.py              # core causal engine
    data.csv             # example dataset
    stress_test/         # safety & stability tests

api/
  causal_api_main.py     # production-grade API

runs/
  <run_id>/
    data.csv
    out/
      edges.csv
      insights_*.csv
```

---

## API Usage

### Health Check

```http
GET /health
```

Response:
```json
{
  "status": "ok",
  "engine": "available",
  "version": "1.1"
}
```

---

### Run Causal Analysis

```http
POST /causal/run
```

**Form-data**
- `file`: CSV dataset
- `target`: target variable name

Example:

```bash
curl -X POST http://localhost:8000/causal/run   -F "file=@data.csv"   -F "target=target"
```

---

## Safety & Certification Pipeline

The project includes a fully automated CI pipeline with:

- Functional engine tests
- Causal safety stress tests
- Multi-run stability tests
- API health and integration tests

---

## Deployment

The API is designed to run as:
- private internal service
- on-premise deployment
- containerized microservice
- controlled startup SaaS backend

---

## Project Status

- Engine: **production-ready**
- API: **production-grade**
- CI/CD: **fully automated**
- Safety & stability: **certified via tests**

---

## Partnerships & Licensing

This project is designed for:
- industrial partnerships
- OEM integration
- startup studio collaboration

For partnership, licensing, or deployment discussions, please contact the project owner.
