"""
CAUSAL ENGINE STABILITY CERTIFICATION TEST
Industrial-grade stability & reproducibility test
"""

import pandas as pd
import numpy as np
import subprocess
import sys
import os
import shutil

RUNS = 5
ENGINE_DIR = "IMPLEMENTATION/pcb_one_click"
OUT_DIR = os.path.join(ENGINE_DIR, "out")
ENGINE_CMD = [
    sys.executable,
    "demo.py",
    "data.csv",
    "target"
]

print("\n[STABILITY TEST] Starting multi-run stability certification...\n")

x_detected = 0
crashes = 0

for i in range(RUNS):
    print(f"\n--- RUN {i+1}/{RUNS} ---")

    # Clean previous output
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)

    # Run engine INSIDE pcb_one_click
    proc = subprocess.run(
        ENGINE_CMD,
        cwd=ENGINE_DIR,
        capture_output=True,
        text=True
    )

    # Save log
    os.makedirs("stability_out", exist_ok=True)
    with open(f"stability_out/run_{i+1}.log", "w") as f:
        f.write(proc.stdout + "\n" + proc.stderr)

    # Check output folder
    if not os.path.exists(OUT_DIR):
        print("[FAIL] Output folder not created")
        crashes += 1
        continue
    else:
        print("[OK] Output folder created")

    # Check edges.csv
    edges_path = os.path.join(OUT_DIR, "edges.csv")
    if not os.path.exists(edges_path):
        print("[FAIL] edges.csv missing")
        crashes += 1
        continue

    edges = pd.read_csv(edges_path)

    # Detect X -> target
    if "from" in edges.columns and "to" in edges.columns:
        causal = edges[edges["to"].str.lower() == "target"]["from"].str.lower().tolist()
        if "x" in causal:
            print("[OK] X detected as causal")
            x_detected += 1
        else:
            print("[WARN] X not detected in this run")
    else:
        print("[FAIL] edges.csv format unexpected")
        crashes += 1

# --------------------------------------------------
# FINAL REPORT
# --------------------------------------------------

print("\n================ STABILITY REPORT ================\n")
print(f"Runs executed        : {RUNS}")
print(f"Engine crashes       : {crashes}")
print(f"X detected in runs   : {x_detected}/{RUNS}")
print(f"Detection rate       : {x_detected / RUNS:.2f}")

FAILED = False

if crashes > 0:
    print("\n❌ Engine is NOT stable (crashes detected)")
    FAILED = True

if x_detected / RUNS < 0.8:
    print("\n❌ Causal detection NOT stable enough (<80%)")
    FAILED = True

if not FAILED:
    print("\n🏆 STABILITY CERTIFICATION PASSED (INDUSTRIAL GRADE)")
    sys.exit(0)
else:
    print("\n❌ STABILITY CERTIFICATION FAILED")
    sys.exit(1)
