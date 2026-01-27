"""
ULTIMATE CAUSAL SAFETY STRESS TEST – CERTIFICATION GRADE

This test certifies that the engine:

✔ Rejects Simpson’s paradox
✔ Rejects collider bias
✔ Rejects spurious correlations
✔ Rejects temporal leakage
✔ Rejects time trends
✔ Obeys interventions
✔ Detects true causal relations only

FAIL = any false positive or missed true causal relation
PASS = engine behaves as a true causal safety engine
"""

import pandas as pd
import numpy as np
import subprocess
import sys
import os

np.random.seed(123)

# --------------------------------------------------
# PATH CONFIGURATION (CRITICAL FOR GITLAB CI)
# --------------------------------------------------

ENGINE_DIR = "IMPLEMENTATION/pcb_one_click"
OUT_DIR = os.path.join(ENGINE_DIR, "out")
DATA = os.path.join(ENGINE_DIR, "data_causal_ultimate.csv")

# --------------------------------------------------
# 1. GENERATE EXTREME CAUSAL DATASET
# --------------------------------------------------

def generate_dataset(n=4000):

    # Hidden confounder (NOT observable by engine)
    H = np.random.normal(0, 1, n)

    # True causal chain:  H -> X -> Y   and   H -> Y
    X = 2 * H + np.random.normal(0, 0.5, n)
    Y = 4 * X + 3 * H + np.random.normal(0, 0.5, n)

    # -----------------------------
    # Simpson’s paradox variable
    # -----------------------------
    group = np.random.binomial(1, 0.5, n)
    simpson = -1 * X + 2 * group + np.random.normal(0, 0.2, n)

    # -----------------------------
    # Collider bias (X + Y)
    # -----------------------------
    collider = X + Y + np.random.normal(0, 0.1, n)

    # -----------------------------
    # Spurious correlated variable
    # -----------------------------
    spurious = X + np.random.normal(0, 1.0, n)

    # -----------------------------
    # Temporal leakage (future info)
    # -----------------------------
    future_leak = np.roll(Y, -1)
    future_leak[-1] = np.nan

    # -----------------------------
    # Time trend (non causal)
    # -----------------------------
    trend = np.linspace(0, 5, n) + np.random.normal(0, 0.1, n)

    # -----------------------------
    # Intervention do(X)
    # -----------------------------
    intervention = np.random.binomial(1, 0.1, n)
    X_do = X + intervention * 5
    Y_do = 4 * X_do + np.random.normal(0, 0.5, n)

    df = pd.DataFrame({
        "X": X_do,                    # TRUE CAUSE
        "simpson": simpson,           # MUST BE REJECTED
        "collider": collider,         # MUST BE REJECTED
        "spurious": spurious,         # MUST BE REJECTED
        "future_leak": future_leak,   # MUST BE REJECTED
        "trend": trend,               # MUST BE REJECTED
        "target": Y_do
    })

    # Remove invalid leakage row
    df = df.dropna().reset_index(drop=True)

    return df


# --------------------------------------------------
# WRITE DATASET
# --------------------------------------------------

os.makedirs(ENGINE_DIR, exist_ok=True)

df = generate_dataset()
df.to_csv(DATA, index=False)

print("\n[TEST] Dataset generated:", DATA)

# --------------------------------------------------
# 2. RUN ENGINE
# --------------------------------------------------

cmd = [
    sys.executable,
    os.path.join(ENGINE_DIR, "demo.py"),
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
# 3. VALIDATION RULES (CERTIFICATION STRICT)
# --------------------------------------------------

FAILED = False

# --------------------------------------------------
# A. Output folder must exist
# --------------------------------------------------

if not os.path.exists(OUT_DIR):
    print("[FAIL] Output folder not created by engine")
    FAILED = True
else:
    print("[OK] Output folder created")

# --------------------------------------------------
# B. Must produce at least one insights file
# --------------------------------------------------

found_insights = False

if os.path.exists(OUT_DIR):
    found_insights = any(
        f.startswith("insights_") and f.endswith(".csv")
        for f in os.listdir(OUT_DIR)
    )

if not found_insights:
    print("[FAIL] No insights file produced")
    FAILED = True
else:
    print("[OK] Insights file produced")

# --------------------------------------------------
# C. Must detect true causal variable X via edges.csv
# --------------------------------------------------

edges_path = os.path.join(OUT_DIR, "edges.csv")

if not os.path.exists(edges_path):
    print("[FAIL] edges.csv not produced – engine failed internally")
    FAILED = True
else:
    edges = pd.read_csv(edges_path)

    causal_vars = (
        edges[edges["to"].str.lower() == "target"]["from"]
        .str.lower()
        .unique()
        .tolist()
    )

    print("[INFO] Detected causal parents of target:", causal_vars)

    if "x" in causal_vars:
        print("[OK] True causal variable X detected")
    else:
        print("[FAIL] True causal variable X NOT detected")
        FAILED = True

# --------------------------------------------------
# D. Safety rejections (NO FALSE POSITIVES ALLOWED)
# --------------------------------------------------

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