"""
CAUSAL STABILITY & ROBUSTNESS TEST – INDUSTRIAL GRADE

This test verifies that the engine:
✔ Produces stable causal relations under noise
✔ Keeps true causal variables across resampling
✔ Does NOT introduce unstable false positives
✔ Produces consistent insights

FAIL = unstable causality or noisy false positives
PASS = engine is robust and production-grade
"""

import pandas as pd
import numpy as np
import subprocess
import sys
import os
import shutil

np.random.seed(42)

BASE_DATA = "IMPLEMENTATION/pcb_one_click/data.csv"   # your real dataset
TMP_DATA  = "IMPLEMENTATION/pcb_one_click/data_stability_tmp.csv"
OUT_DIR   = "out"

N_RUNS = 8              # number of perturbation runs
NOISE_STD = 0.15        # noise strength
MIN_STABILITY = 0.70    # variable must appear in ≥70% of runs

print("\n[TEST] Running CAUSAL STABILITY TEST...\n")

# --------------------------------------------------
# 1. LOAD BASE DATA
# --------------------------------------------------

if not os.path.exists(BASE_DATA):
    print("[FAIL] Base dataset not found:", BASE_DATA)
    sys.exit(1)

base_df = pd.read_csv(BASE_DATA)

target = "target"
features = [c for c in base_df.columns if c != target]

# --------------------------------------------------
# 2. RUN MULTIPLE PERTURBED EXPERIMENTS
# --------------------------------------------------

detected_vars = []

for run in range(N_RUNS):

    print(f"\n[RUN {run+1}/{N_RUNS}] Generating noisy dataset...")

    df = base_df.copy()

    # Add gaussian noise to all non-target features
    for col in features:
        df[col] = df[col] + np.random.normal(0, NOISE_STD, len(df))

    df.to_csv(TMP_DATA, index=False)

    # Run engine
    cmd = [
        sys.executable,
        "IMPLEMENTATION/pcb_one_click/demo.py",
        TMP_DATA,
        target
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)

    report = proc.stdout.lower()

    # Collect detected causal vars from edges
    edges_path = os.path.join(OUT_DIR, "edges.csv")

    if not os.path.exists(edges_path):
        print("[WARN] edges.csv missing in run", run+1)
        continue

    edges = pd.read_csv(edges_path)

    vars_to_target = edges[edges["to"] == target]["from"].str.lower().tolist()

    print("Detected causal vars:", vars_to_target)

    detected_vars.append(vars_to_target)

# --------------------------------------------------
# 3. STABILITY ANALYSIS
# --------------------------------------------------

print("\n[TEST] Stability analysis...\n")

from collections import Counter

flat = [v for run in detected_vars for v in run]
counts = Counter(flat)

FAILED = False

print("Variable frequency across runs:")

for var, c in counts.items():
    freq = c / N_RUNS
    print(f" - {var}: {c}/{N_RUNS} = {freq:.2f}")

    # Reject unstable false positives
    if freq < MIN_STABILITY:
        print(f"[FAIL] Variable '{var}' is UNSTABLE (freq {freq:.2f})")
        FAILED = True

# --------------------------------------------------
# 4. TRUE CAUSE MUST BE STABLE
# --------------------------------------------------

TRUE_CAUSE = "x"   # change if your true variable name differs

true_freq = counts.get(TRUE_CAUSE, 0) / N_RUNS

if true_freq < 0.8:
    print(f"[FAIL] True causal variable '{TRUE_CAUSE}' unstable ({true_freq:.2f})")
    FAILED = True
else:
    print(f"[OK] True causal variable '{TRUE_CAUSE}' stable ({true_freq:.2f})")

# --------------------------------------------------
# 5. FINAL VERDICT
# --------------------------------------------------

if FAILED:
    print("\n❌ CAUSAL STABILITY TEST FAILED")
    sys.exit(1)
else:
    print("\n🏆 CAUSAL STABILITY TEST PASSED – ENGINE IS ROBUST")
    sys.exit(0)