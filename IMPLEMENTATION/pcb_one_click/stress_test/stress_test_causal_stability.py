"""
CAUSAL STABILITY & ROBUSTNESS TEST – INDUSTRIAL GRADE

Verifies that:
✔ True causal relations are stable under noise
✔ No unstable false positives appear
✔ Insights remain consistent across perturbations

FAIL = unstable causality or noisy false positives
PASS = engine is production-grade robust
"""

import pandas as pd
import numpy as np
import subprocess
import sys
import os
from collections import Counter

np.random.seed(42)

BASE_DATA = "IMPLEMENTATION/pcb_one_click/data.csv"
TMP_DATA  = "IMPLEMENTATION/pcb_one_click/data_stability_tmp.csv"
OUT_DIR   = "out"

N_RUNS = 8
NOISE_STD = 0.15
MIN_STABILITY = 0.70

TARGET = "target"
TRUE_CAUSE = "x"   # cambia se il tuo nome è diverso

print("\n[TEST] Running CAUSAL STABILITY TEST...\n")

# --------------------------------------------------
# 1. LOAD BASE DATA
# --------------------------------------------------

if not os.path.exists(BASE_DATA):
    print("[FAIL] Base dataset not found:", BASE_DATA)
    sys.exit(1)

base_df = pd.read_csv(BASE_DATA)

# Select only numeric continuous features (safe to perturb)
numeric_cols = base_df.select_dtypes(include=["float64", "float32", "int64", "int32"]).columns.tolist()

# Remove target from perturbation
if TARGET in numeric_cols:
    numeric_cols.remove(TARGET)

print("[INFO] Numeric columns to perturb:", numeric_cols)

# --------------------------------------------------
# 2. MULTIPLE PERTURBED RUNS
# --------------------------------------------------

detected_vars = []

for run in range(N_RUNS):

    print(f"\n[RUN {run+1}/{N_RUNS}] Generating noisy dataset...")

    df = base_df.copy()

    # Add noise ONLY to numeric features
    for col in numeric_cols:
        df[col] = df[col] + np.random.normal(0, NOISE_STD, len(df))

    df.to_csv(TMP_DATA, index=False)

    # Run engine
    cmd = [
        sys.executable,
        "IMPLEMENTATION/pcb_one_click/demo.py",
        TMP_DATA,
        TARGET
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)

    # Collect edges
    edges_path = os.path.join(OUT_DIR, "edges.csv")

    if not os.path.exists(edges_path):
        print("[WARN] edges.csv missing in run", run+1)
        continue

    edges = pd.read_csv(edges_path)

    # Normalize column names (your engine sometimes changes them)
    cols = [c.lower() for c in edges.columns]
    edges.columns = cols

    if "from" not in cols or "to" not in cols:
        print("[WARN] edges.csv format unexpected in run", run+1)
        continue

    vars_to_target = (
        edges[edges["to"].str.lower() == TARGET]["from"]
        .str.lower()
        .tolist()
    )

    print("Detected causal vars:", vars_to_target)

    detected_vars.append(vars_to_target)

# --------------------------------------------------
# 3. STABILITY ANALYSIS
# --------------------------------------------------

print("\n[TEST] Stability analysis...\n")

flat = [v for run in detected_vars for v in run]
counts = Counter(flat)

FAILED = False

print("Variable frequency across runs:\n")

for var, c in counts.items():
    freq = c / N_RUNS
    print(f" - {var}: {c}/{N_RUNS} = {freq:.2f}")

    # Reject unstable variables
    if freq < MIN_STABILITY:
        print(f"[FAIL] Variable '{var}' is UNSTABLE (freq {freq:.2f})")
        FAILED = True

# --------------------------------------------------
# 4. TRUE CAUSE MUST BE STABLE
# --------------------------------------------------

true_freq = counts.get(TRUE_CAUSE.lower(), 0) / N_RUNS

if true_freq < 0.80:
    print(f"[FAIL] True causal variable '{TRUE_CAUSE}' UNSTABLE ({true_freq:.2f})")
    FAILED = True
else:
    print(f"[OK] True causal variable '{TRUE_CAUSE}' STABLE ({true_freq:.2f})")

# --------------------------------------------------
# 5. FINAL VERDICT
# --------------------------------------------------

if FAILED:
    print("\n❌ CAUSAL STABILITY TEST FAILED")
    sys.exit(1)
else:
    print("\n🏆 CAUSAL STABILITY TEST PASSED – ENGINE IS ROBUST & INDUSTRIAL GRADE")
    sys.exit(0)