from fastapi import FastAPI
import subprocess
import json
import tempfile
import os

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyze")
def analyze(data: dict):

    # salva input temporaneo
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as f:
        json.dump(data, f)
        input_path = f.name

    # esegue il tuo engine
    result = subprocess.run(
        ["python", "IMPLEMENTATION/pcb_one_click/run_pcb.py", input_path],
        capture_output=True,
        text=True
    )

    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode
    }
