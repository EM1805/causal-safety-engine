"""
ULTIMATE CAUSAL SAFETY STRESS TEST – CERTIFICATION GRADE
"""

import pandas as pd
import numpy as np
import subprocess
import sys
import os

np.random.seed(123)

OUT = "out"
DATA = "IMPLEMENTATION/pcb_one_click/data_causal_ultimate.csv"

# --------------------------------------------------
# 1. GENERATE EXTREME CAUSAL DATASET
# --------------------------------------------------

def generate_dataset(n=4000):

    # Hidden confounder (NOT observable)
    H = np.random.normal(0, 1, n)

    # True causal chain
    X = 2 * H + np.random.normal(0, 0.5, n)
    Y = 4 * X + 3 * H + np.random.normal(0, 0.5, n)

    # Simpson’s paradox
    group = np.random.binomial(1, 0.5, n)
    S = -1 * X + 2 * group + np.random.normal(0, 0.2, n)

    # Collider
    C = X + Y + np.random.normal(0, 0.1, n)

    # Spurious (correlated with X, not causal)
    Z = X + np.random.normal(0, 1.0, n)

    # Temporal leakage
    future = np.roll(Y, -1)
    future[-1] = np.nan

    # Time trend
    trend = np.linspace(0, 5, n) + np.random.normal(0, 0.1, n)

    # Intervention do(X) — confounder removed after intervention
    intervention = np.random.binomial(1, 0.1, n)
    X_do = X + intervention * 5
    Y_do = 4 * X_do + np.random.normal(0, 0.5, n)

    df = pd.DataFrame({
        "X": X_do,
        "simpson": S,
        "collider": C,
        "spurious": Z,
        "future_leak": future,
        "trend": trend,
        "target": Y_do
    })

    df = df.dropna().reset_index(drop=True)
    return df


os.makedirs("IMPLEMENTATION/pcb_one_click", exist_ok=True)
df = generate_dataset()
df.to_csv(DATA, index=False)

print("[TEST] Dataset generated:", DATA)

# --------------------------------------------------
# 2. RUN ENGINE
# --------------------------------------------------

cmd = [
    sys.executable,
    "IMPLEMENTATION/pcb_one_click/demo.py",
    DATA,
    "target"
]

print("\n[TEST] Running ULTIMATE CAUSAL SAFETY STRESS TEST...\n")

proc = subprocess.run(cmd, capture_output=True, text=True)

with open("stress_result.txt", "w") as f:
    f.write(proc.stdout)

print(proc.stdout)

report = proc.stdout.lower()

# --------------------------------------------------
# 3. VALIDATION RULES
# --------------------------------------------------

FAILED = False

# --- Engine must produce insights file

if not os.path.exists("out"):
    print("[FAIL] Output folder not created")
    FAILED = True
else:
    found_insights = any(
        f.startswith("insights_") and f.endswith(".csv")
        for f in os.listdir("out")
    )

    if not found_insights:
        print("[FAIL] No insights file produced")
        FAILED = True
    else:
        print("[OK] Insights file produced")

# --- Must detect true causal X via edges

edges_path = "out/edges.csv"

if not os.path.exists(edges_path):
    print("[FAIL] edges.csv not produced – engine failed internally")
    FAILED = True
else:
    edges = pd.read_csv(edges_path)
    causal_vars = edges[edges["to"] == "target"]["from"].str.lower().tolist()

    if "x" in causal_vars:
        print("[OK] True causal variable X detected")
    else:
        print("[FAIL] True causal variable X NOT detected")
        FAILED = True

# --- Safety rejections

for bad in ["simpson", "collider", "spurious", "future_leak", "trend"]:
    if bad in report:
        print(f"[FAIL] {bad} accepted as causal")
        FAILED = True
    else:
        print(f"[OK] {bad} rejected")

# --------------------------------------------------
# 4. FINAL VERDICT
# --------------------------------------------------

if FAILED:
    print("\n❌ ULTIMATE CAUSAL SAFETY TEST FAILED")
    sys.exit(1)
else:
    print("\n🏆 ULTIMATE CAUSAL SAFETY ENGINE CERTIFIED")
    sys.exit(0)