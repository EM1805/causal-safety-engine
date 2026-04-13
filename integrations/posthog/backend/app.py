from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .engine_adapter import run_analysis

app = FastAPI(title="Causal Safety Engine API", version="0.1.0")


class AnalyzeRequest(BaseModel):
    rows: list[dict[str, Any]] = Field(default_factory=list)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze")
def analyze(payload: AnalyzeRequest) -> dict[str, Any]:
    if not payload.rows:
        raise HTTPException(status_code=400, detail="rows cannot be empty")

    return run_analysis(payload.rows)
