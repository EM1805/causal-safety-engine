
"""
CAUSAL SAFETY ENGINE – API v1
Production-ready FastAPI wrapper
Safe for Render / Replit / Docker
"""

import os
import uuid
import shutil
import subprocess
import hashlib
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse

# =====================================================
# CONFIG
# =====================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

ENGINE_PATH = PROJECT_ROOT / "IMPLEMENTATION" / "pcb_one_click" / "demo.py"
RUNS_DIR = PROJECT_ROOT / "runs"

ENGINE_TIMEOUT = 300  # seconds

RUNS_DIR.mkdir(exist_ok=True)

if not ENGINE_PATH.exists():
    raise RuntimeError(f"ENGINE NOT FOUND: {ENGINE_PATH}")

# =====================================================
# APP
# =====================================================

app = FastAPI(
    title="Causal Safety Engine API",
    version="1.1",
    description="Industrial-grade causal discovery API"
)

# =====================================================
# HELPERS
# =====================================================

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_edges(path: Path):
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]

    if {"from", "to"}.issubset(df.columns):
        return (
            df[df["to"] == "target"]["from"]
            .astype(str)
            .str.lower()
            .tolist()
        )

    if {"source", "target"}.issubset(df.columns):
        return (
            df[df["target"] == "target"]["source"]
            .astype(str)
            .str.lower()
            .tolist()
        )

    return []


def parse_insights(path: Path):
    return pd.read_csv(path).to_dict(orient="records")

# =====================================================
# ENDPOINTS
# =====================================================

@app.post("/causal/run")
async def causal_run(
    file: UploadFile = File(...),
    target: str = Form(...)
):
    run_id = str(uuid.uuid4())
    run_dir = RUNS_DIR / run_id
    out_dir = run_dir / "out"

    run_dir.mkdir(parents=True)
    out_dir.mkdir()

    # Save dataset
    data_path = run_dir / "data.csv"
    with open(data_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Validate CSV
    try:
        df = pd.read_csv(data_path)
    except Exception:
        raise HTTPException(400, "Invalid CSV file")

    if target not in df.columns:
        raise HTTPException(400, f"Target '{target}' not in dataset")

    dataset_hash = sha256(data_path)

    # Run engine
    cmd = ["python", str(ENGINE_PATH), str(data_path), target]

    try:
        proc = subprocess.run(
            cmd,
            cwd=run_dir,
            capture_output=True,
            text=True,
            timeout=ENGINE_TIMEOUT
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(500, "Engine timeout")

    if proc.returncode != 0:
        raise HTTPException(
            500,
            f"Engine error:\n{proc.stderr[:600]}"
        )

    edges_path = out_dir / "edges.csv"

    if not edges_path.exists():
        raise HTTPException(500, "edges.csv not generated")

    insights_file = None
    for f in out_dir.iterdir():
        if f.name.startswith("insights") and f.suffix == ".csv":
            insights_file = f

    return JSONResponse({
        "status": "ok",
        "run_id": run_id,
        "target": target,
        "dataset_hash": dataset_hash,
        "causal_variables": parse_edges(edges_path),
        "insights": parse_insights(insights_file) if insights_file else [],
        "artifacts": {
            "edges": f"/causal/artifacts/{run_id}/edges",
            "insights": f"/causal/artifacts/{run_id}/insights"
        }
    })

# =====================================================
# ARTIFACTS
# =====================================================

@app.get("/causal/artifacts/{run_id}/edges")
def get_edges(run_id: str):
    path = RUNS_DIR / run_id / "out" / "edges.csv"
    if not path.exists():
        raise HTTPException(404, "edges not found")
    return FileResponse(path, media_type="text/csv")


@app.get("/causal/artifacts/{run_id}/insights")
def get_insights(run_id: str):
    out_dir = RUNS_DIR / run_id / "out"
    if not out_dir.exists():
        raise HTTPException(404, "run not found")

    for f in out_dir.iterdir():
        if f.name.startswith("insights") and f.suffix == ".csv":
            return FileResponse(f, media_type="text/csv")

    raise HTTPException(404, "insights not found")

# =====================================================
# HEALTH
# =====================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "engine": "available",
        "version": "1.1"
    }