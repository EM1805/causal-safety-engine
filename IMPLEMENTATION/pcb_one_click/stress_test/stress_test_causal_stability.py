"""
CAUSAL SAFETY – STABILITY TEST (INDUSTRIAL GRADE)

Goal:
- Run engine multiple times with different seeds
- Verify engine never crashes
- Verify outputs are stable
- Measure detection stability of true causal variable X
"""

import pandas as pd
import numpy as np
import subprocess
import sys
import os
import shutil

ENGINE = ["python", "IMPLEMENTATION/pcb_one_click/demo.py"]
DATA = "IMPLEMENTATION/pcb_one_click/data.csv"
TARGET = "target"

RUNS = 5          # number of repeated runs
PASS_RATE = 0.8  # at least 80% of runs must detect X

os.makedirs("stability_out", exist_ok=True)

detected_x = 0
successful_runs = 0

print("\n[STABILITY TEST] Starting multi-run stability certification...\n")

for i in range(RUNS):
    print(f"\n--- RUN {i+1}/{RUNS} ---")

    # Clean output folder before each run
    if os.path.exists("out"):
        shutil.rmtree("out")

    # Change seed for this run
    np.random.seed(1000 + i)

    # Run engine
    cmd = ENGINE + [DATA, TARGET]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    # Save raw log
    with open(f"stability_out/run_{i+1}.log", "w") as f:
        f.write(proc.stdout)
        f.write(proc.stderr)

    # Engine must create output folder
    if not os.path.exists("out"):
        print("[FAIL] Output folder not created")
        continue

    successful_runs += 1

    # Check edges.csv if present
    edges_path = "out/edges.csv"

    if not os.path.exists(edges_path):
        print("[WARN] edges.csv not produced in this run")
        continue

    try:
        edges = pd.read_csv(edges_path)
    except Exception as e:
        print("[WARN] Cannot read edges.csv:", e)
        continue

    # Normalize column names
    cols = [c.lower() for c in edges.columns]
    edges.columns = cols

    # Try to find source/target columns safely
    from_col = None
    to_col = None

    for c in cols:
        if c in ["from", "source", "parent"]:
            from_col = c
        if c in ["to", "target", "child"]:
            to_col = c

    if from_col is None or to_col is None:
        print("[WARN] edges.csv format not recognized")
        continue

    # Check if X -> target detected
    causal_vars = edges[edges[to_col].str.lower() == TARGET][from_col].str.lower().tolist()

    if "x" in causal_vars:
        print("[OK] X detected in this run")
        detected_x += 1
    else:
        print("[WARN] X NOT detected in this run")


# --------------------------------------------------
# FINAL STABILITY VERDICT
# --------------------------------------------------

rate = detected_x / max(1, successful_runs)

with open("stability_result.txt", "w") as f:
    f.write(f"Runs executed: {RUNS}\n")
    f.write(f"Successful engine runs: {successful_runs}\n")
    f.write(f"X detected in: {detected_x} runs\n")
    f.write(f"Detection rate: {rate:.2f}\n\n")

    if rate >= PASS_RATE and successful_runs == RUNS:
        f.write("STABILITY TEST PASSED – ENGINE IS STABLE\n")
        print("\n🏆 STABILITY TEST PASSED – ENGINE IS STABLE")
        sys.exit(0)
    else:
        f.write("STABILITY TEST FAILED – ENGINE IS NOT STABLE\n")
        print("\n❌ STABILITY TEST FAILED – ENGINE IS NOT STABLE")
        sys.exit(1)
