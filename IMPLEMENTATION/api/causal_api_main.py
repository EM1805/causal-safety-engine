
"""
CAUSAL SAFETY ENGINE – OFFICIAL API v1 (PRODUCTION-READY)
Industrial-grade FastAPI wrapper for pcb_one_click demo engine
Portable: works on Render, GitLab CI, Docker, local
"""

import os
import uuid
import shutil
import subprocess
import hashlib

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse

import pandas as pd

# ---------------- CONFIG ----------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ENGINE_PATH = os.getenv(
    "ENGINE_PATH",
    os.path.abspath(
        os.path.join(
            BASE_DIR,
            "..",              # IMPLEMENTATION
            "pcb_one_click",
            "demo.py"
        )
    )
)

BASE_OUT = os.path.join(BASE_DIR, "runs")
ENGINE_TIMEOUT = 300  # seconds

os.makedirs(BASE_OUT, exist_ok=True)

if not os.path.exists(ENGINE_PATH):
    raise RuntimeError(f"ENGINE NOT FOUND: {ENGINE_PATH}")

# ---------------- APP ----------------

app = FastAPI(
    title="Causal Safety Engine API",
    version="1.1",
    description="Industrial-grade API for certified causal discovery"
)

# ---------------- UTILITIES ----------------

def file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_edges(path):
    edges = pd.read_csv(path)
    edges.columns = [c.lower() for c in edges.columns]

    causal = []

    if "to" in edges.columns and "from" in edges.columns:
        causal = edges[edges["to"] == "target"]["from"].astype(str).tolist()
    elif "target" in edges.columns and "source" in edges.columns:
        causal = edges[edges["target"] == "target"]["source"].astype(str).tolist()

    return causal


def parse_insights(path):
    df = pd.read_csv(path)
    return df.to_dict(orient="records")

# ---------------- ENDPOINTS ----------------

@app.post("/causal/run")
async def causal_run(
    file: UploadFile = File(...),
    target: str = Form(...)
):
    run_id = str(uuid.uuid4())
    run_dir = os.path.join(BASE_OUT, run_id)
    os.makedirs(run_dir, exist_ok=True)

    data_path = os.path.join(run_dir, "data.csv")
    with open(data_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    dataset_hash = file_hash(data_path)

    try:
        df = pd.read_csv(data_path)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid CSV file")

    if target not in df.columns:
        raise HTTPException(status_code=400, detail="Target column not found")

    cmd = ["python", ENGINE_PATH, data_path, target]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=ENGINE_TIMEOUT,
            cwd=run_dir
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="Engine timeout")

    if proc.returncode != 0:
        raise HTTPException(status_code=500, detail=proc.stderr[:500])

    out_dir = os.path.join(run_dir, "out")
    edges_path = os.path.join(out_dir, "edges.csv")

    if not os.path.exists(edges_path):
        raise HTTPException(status_code=500, detail="edges.csv missing")

    insights_path = None
    for f in os.listdir(out_dir):
        if f.startswith("insights") and f.endswith(".csv"):
            insights_path = os.path.join(out_dir, f)

    response = {
        "status": "ok",
        "run_id": run_id,
        "engine": "pcb_one_click",
        "target": target,
        "dataset_hash": dataset_hash,
        "causal_variables": parse_edges(edges_path),
        "insights": parse_insights(insights_path) if insights_path else [],
        "artifacts": {
            "edges": f"/causal/artifacts/{run_id}/edges",
            "insights": f"/causal/artifacts/{run_id}/insights"
        }
    }

    return JSONResponse(response)


@app.get("/causal/artifacts/{run_id}/edges")
def get_edges(run_id: str):
    path = os.path.join(BASE_OUT, run_id, "out", "edges.csv")
    if not os.path.exists(path):
        raise HTTPException(status_code=404)
    return FileResponse(path, media_type="text/csv")


@app.get("/causal/artifacts/{run_id}/insights")
def get_insights(run_id: str):
    out_dir = os.path.join(BASE_OUT, run_id, "out")
    if not os.path.exists(out_dir):
        raise HTTPException(status_code=404)

    for f in os.listdir(out_dir):
        if f.startswith("insights") and f.endswith(".csv"):
            return FileResponse(os.path.join(out_dir, f), media_type="text/csv")

    raise HTTPException(status_code=404)


@app.get("/health")
def health():
    return {"status": "ok", "engine": "available"}
