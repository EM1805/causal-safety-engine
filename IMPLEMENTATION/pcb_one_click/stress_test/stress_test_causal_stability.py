
import os
import sys
import subprocess
import pandas as pd
import numpy as np

np.random.seed(123)

RUNS = 5
ENGINE_CMD = [sys.executable, "IMPLEMENTATION/pcb_one_click/demo.py",
              "IMPLEMENTATION/pcb_one_click/data.csv", "target"]

print("\n[STABILITY TEST] Starting multi-run stability certification...\n")

crashes = 0
x_detected = 0

def detect_x(edges):
    cols = [c.lower() for c in edges.columns]
    edges.columns = cols

    # Common schemas supported
    if "to" in cols and "from" in cols:
        causal = edges[edges["to"].str.lower() == "target"]["from"].str.lower().tolist()
    elif "target" in cols and "source" in cols:
        causal = edges[edges["target"].str.lower() == "target"]["source"].str.lower().tolist()
    else:
        return False

    return "x" in causal

for i in range(RUNS):
    print(f"--- RUN {i+1}/{RUNS} ---")

    if os.path.exists("out"):
        for f in os.listdir("out"):
            os.remove(os.path.join("out", f))
    else:
        os.makedirs("out", exist_ok=True)

    proc = subprocess.run(ENGINE_CMD, capture_output=True, text=True)

    if not os.path.exists("out/edges.csv"):
        print("[FAIL] edges.csv not produced")
        crashes += 1
        continue

    try:
        edges = pd.read_csv("out/edges.csv")
    except Exception:
        print("[FAIL] edges.csv unreadable")
        crashes += 1
        continue

    if detect_x(edges):
        print("[OK] X detected")
        x_detected += 1
    else:
        print("[WARN] X not detected in this run")

print("\n========== STABILITY REPORT ==========")
print("Runs executed:", RUNS)
print("Engine crashes:", crashes)
print("X detected in runs:", x_detected)
rate = x_detected / RUNS
print("Detection rate:", round(rate, 2))

FAILED = False

if crashes > 0:
    print("❌ Engine is NOT stable (crashes detected)")
    FAILED = True

if rate < 0.6:
    print("❌ Causal detection NOT stable enough (<60%)")
    FAILED = True

if FAILED:
    print("\n❌ STABILITY CERTIFICATION FAILED")
    sys.exit(1)
else:
    print("\n🏆 STABILITY CERTIFICATION PASSED")
    sys.exit(0)
