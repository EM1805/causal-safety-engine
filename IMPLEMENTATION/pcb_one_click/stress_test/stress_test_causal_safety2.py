"""
CAUSAL SAFETY CERTIFICATION TEST – IMPLEMENTATION COMPATIBLE
"""

import pandas as pd
import numpy as np
import subprocess
import sys
import os

np.random.seed(123)

DATA = "IMPLEMENTATION/pcb_one_click/data_causal_certification.csv"

# --------------------------------------------------
# 1. GENERATE CERTIFICATION DATASET
# --------------------------------------------------

def generate_dataset(n=4000):

    H = np.random.normal(0, 1, n)

    # True causal chain
    X = 2 * H + np.random.normal(0, 0.5, n)
    Y = 4 * X + 3 * H + np.random.normal(0, 0.5, n)

    # Simpson
    group = np.random.binomial(1, 0.5, n)
    simpson = -1 * X + 2 * group + np.random.normal(0, 0.2, n)

    # Collider
    collider = X + Y + np.random.normal(0, 0.1, n)

    # Spurious
    spurious = X + np.random.normal(0, 1.0, n)

    # Temporal leakage
    future = np.roll(Y, -1)
    future[-1] = np.nan

    # Trend
    trend = np.linspace(0, 5, n) + np.random.normal(0, 0.1, n)

    # Intervention
    intervention = np.random.binomial(1, 0.1, n)
    X_do = X + intervention * 5
    Y_do = 4 * X_do + np.random.normal(0, 0.5, n)

    df = pd.DataFrame({
        "X": X_do,
        "simpson": simpson,
        "collider": collider,
        "spurious": spurious,
        "future_leak": future,
        "trend": trend,
        "target": Y_do
    })

    return df.dropna().reset_index(drop=True)


os.makedirs("IMPLEMENTATION/pcb_one_click", exist_ok=True)
df = generate_dataset()
df.to_csv(DATA, index=False)

print("[TEST] Dataset generated")

# --------------------------------------------------
# 2. RUN ENGINE
# --------------------------------------------------

cmd = [
    sys.executable,
    "IMPLEMENTATION/pcb_one_click/demo.py",
    DATA,
    "target"
]

print("\n[TEST] Running CAUSAL SAFETY CERTIFICATION...\n")

proc = subprocess.run(cmd, capture_output=True, text=True)

with open("stress_result.txt", "w") as f:
    f.write(proc.stdout)

print(proc.stdout)

report = proc.stdout.lower()

# --------------------------------------------------
# 3. LOCATE REAL OUT DIRECTORY
# --------------------------------------------------

OUT_DIR = None

for root, dirs, files in os.walk("."):
    if root.endswith("/out") or root.endswith("\\out"):
        OUT_DIR = root
        break

if OUT_DIR is None:
    print("[FAIL] Output folder not found anywhere")
    sys.exit(1)

print("[OK] Output folder located at:", OUT_DIR)

# --------------------------------------------------
# 4. VALIDATION
# --------------------------------------------------

FAILED = False

# --- Must produce edges.csv

edges_path = os.path.join(OUT_DIR, "edges.csv")

if not os.path.exists(edges_path):
    print("[FAIL] edges.csv not produced")
    FAILED = True
else:
    print("[OK] edges.csv produced")

    edges = pd.read_csv(edges_path)

    # Flexible schema detection
    cols = [c.lower() for c in edges.columns]

    if "source" in cols:
        source_col = edges.columns[cols.index("source")]
    else:
        print("[FAIL] edges.csv missing source column")
        FAILED = True
        source_col = None

    # Collect detected variables
    detected = edges[source_col].astype(str).str.lower().tolist()

    print("[INFO] Detected causal variables:")
    for v in set(detected):
        print(" -", v)

    if "x" in detected:
        print("[OK] True causal variable X detected")
    else:
        print("[FAIL] True causal variable X NOT detected")
        FAILED = True

# --- Must produce at least one insights file

found_insights = False
for f in os.listdir(OUT_DIR):
    if f.startswith("insights_") and f.endswith(".csv"):
        found_insights = True
        print("[OK] Insights file produced:", f)
        break

if not found_insights:
    print("[FAIL] No insights file produced")
    FAILED = True

# --- Safety rejections (from logs)

for bad in ["simpson", "collider", "spurious", "future_leak", "trend"]:
    if bad in report:
        print(f"[FAIL] {bad} accepted as causal")
        FAILED = True
    else:
        print(f"[OK] {bad} rejected")

# --------------------------------------------------
# 5. FINAL VERDICT
# --------------------------------------------------

if FAILED:
    print("\n❌ CAUSAL SAFETY CERTIFICATION FAILED")
    sys.exit(1)
else:
    print("\n🏆 CAUSAL SAFETY ENGINE CERTIFIED")
    sys.exit(0)