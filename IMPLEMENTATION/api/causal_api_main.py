
"""
CAUSAL SAFETY ENGINE – DEMO-ADAPTED API v1

This API wraps pcb_one_click/demo.py EXACTLY as implemented.
- demo.py ignores uploaded filenames and always reads pcb_one_click/data.csv
- demo.py always writes outputs to pcb_one_click/out/

This API:
- overwrites data.csv before each run
- runs demo.py with correct cwd
- reads artifacts from ./out
- returns structured JSON consistent with demo output
"""

import os
import uuid
import shutil
import subprocess
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PCB_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "pcb_one_click"))
DEMO_PATH = os.path.join(PCB_DIR, "demo.py")
DATA_PATH = os.path.join(PCB_DIR, "data.csv")
OUT_DIR = os.path.join(PCB_DIR, "out")

if not os.path.exists(DEMO_PATH):
    raise RuntimeError("demo.py not found")

app = FastAPI(
    title="Causal Safety Engine API (Demo-Adapted)",
    version="1.0",
    description="API wrapper aligned with pcb_one_click demo.py behavior"
)

@app.post("/causal/run")
async def causal_run(file: UploadFile = File(...)):
    run_id = str(uuid.uuid4())

    # overwrite data.csv (demo.py always reads this file)
    with open(DATA_PATH, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # clean previous outputs
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)

    cmd = ["python", "demo.py"]

    try:
        subprocess.check_call(cmd, cwd=PCB_DIR)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not os.path.exists(OUT_DIR):
        raise HTTPException(status_code=500, detail="Demo did not produce out/ directory")

    response = {
        "status": "ok",
        "run_id": run_id,
        "artifacts": {}
    }

    # collect artifacts
    for fname in os.listdir(OUT_DIR):
        if fname.endswith(".csv"):
            response["artifacts"][fname] = f"/causal/artifacts/{fname}"

    return JSONResponse(response)


@app.get("/causal/artifacts/{filename}")
def get_artifact(filename: str):
    path = os.path.join(OUT_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404)
    return FileResponse(path, media_type="text/csv")


@app.get("/health")
def health():
    return {"status": "ok", "engine": "pcb demo"}
