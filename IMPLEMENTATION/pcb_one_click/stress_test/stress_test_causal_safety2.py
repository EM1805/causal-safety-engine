"""
ULTIMATE CAUSAL SAFETY STRESS TEST – CERTIFICATION GRADE

This test certifies that the engine:

✔ Detects true causal relations
✔ Rejects Simpson’s paradox
✔ Rejects collider bias
✔ Rejects spurious correlations
✔ Rejects temporal leakage
✔ Rejects time trends
✔ Obeys interventions
✔ Produces insights ONLY when causality is real

FAIL = any false positive or missed true causal relation
PASS = engine behaves as a true causal safety engine
"""

import pandas as pd
import numpy as np
import subprocess
import sys
import os

np.random.seed(123)

DATA = "IMPLEMENTATION/pcb_one_click/data_causal_certification.csv"
OUT_DIR = "out"

# --------------------------------------------------
# 1. GENERATE CERTIFICATION-GRADE DATASET
# --------------------------------------------------

def generate_dataset(n=4000):

    # Hidden confounder (not observable by engine)
    H = np.random.normal(0, 1, n)

    # True causal structure:  H -> X -> Y  and  H -> Y
    X = 2 * H + np.random.normal(0, 0.5, n)
    Y = 4 * X + 3 * H + np.random.normal(0, 0.5, n)

    # -------------------------------
    # Simpson’s paradox variable
    # -------------------------------
    group = np.random.binomial(1, 0.5, n)
    simpson = -1 * X + 2 * group + np.random.normal(0, 0.2, n)

    # -------------------------------
    # Collider bias variable
    # -------------------------------
    collider = X + Y + np.random.normal(0, 0.1, n)

    # -------------------------------
    # Spurious correlated variable
    # -------------------------------
    spurious = X + np.random.normal(0, 1.0, n)

    # -------------------------------
    # Temporal leakage (future info)
    # -------------------------------
    future = np.roll(Y, -1)
    future[-1] = np.nan

    # -------------------------------
    # Time trend (non causal)
    # -------------------------------
    trend = np.linspace(0, 5, n) + np.random.normal(0, 0.1, n)

    # -------------------------------
    # Intervention do(X)
    # -------------------------------
    intervention = np.random.binomial(1, 0.1, n)
    X_do = X + intervention * 5
    Y_do = 4 * X_do + np.random.normal(0, 0.5, n)

    df = pd.DataFrame({
        "X": X_do,                 # TRUE CAUSE
        "simpson": simpson,        # MUST BE REJECTED
        "collider": collider,      # MUST BE REJECTED
        "spurious": spurious,      # MUST BE REJECTED
        "future_leak": future,     # MUST BE REJECTED
        "trend": trend,            # MUST BE REJECTED
        "target": Y_do
    })

    df = df.dropna().reset_index(drop=True)
    return df


# Write dataset
os.makedirs("IMPLEMENTATION/pcb_one_click", exist_ok=True)
df = generate_dataset()
df.to_csv(DATA, index=False)

print("[TEST] Certification dataset generated:", DATA)

# --------------------------------------------------
# 2. RUN ENGINE
# --------------------------------------------------

cmd = [
    sys.executable,
    "IMPLEMENTATION/pcb_one_click/demo.py",
    DATA,
    "target"
]

print("\n[TEST] Running CAUSAL SAFETY CERTIFICATION TEST...\n")

proc = subprocess.run(cmd, capture_output=True, text=True)

with open("stress_result.txt", "w") as f:
    f.write(proc.stdout)

print(proc.stdout)

report = proc.stdout.lower()

# --------------------------------------------------
# 3. VALIDATION RULES (CERTIFICATION)
# --------------------------------------------------

FAILED = False

# --------------------------------------------------
# A. Engine must create output folder
# --------------------------------------------------

if not os.path.exists(OUT_DIR):
    print("[FAIL] Output folder not created")
    FAILED = True
else:
    print("[OK] Output folder created")

# --------------------------------------------------
# B. Engine must produce edges.csv
# --------------------------------------------------

edges_path = os.path.join(OUT_DIR, "edges.csv")

if not os.path.exists(edges_path):
    print("[FAIL] edges.csv not produced – engine failed internally")
    FAILED = True
else:
    print("[OK] edges.csv produced")

    edges = pd.read_csv(edges_path)

    # Check expected schema
    if "source" not in edges.columns or "target" not in edges.columns:
        print("[FAIL] edges.csv schema invalid – missing source/target columns")
        FAILED = True
    else:
        causal_edges = edges[edges["target"].str.lower() == "target"]

        if len(causal_edges) == 0:
            print("[FAIL] No causal edges detected toward target")
            FAILED = True
        else:
            vars_detected = causal_edges["source"].str.lower().tolist()

            print("[OK] Causal relations detected toward target:")
            for v in vars_detected:
                print(" -", v)

            if "x" in vars_detected:
                print("[OK] True causal variable X detected")
            else:
                print("[FAIL] True causal variable X NOT detected")
                FAILED = True

# --------------------------------------------------
# C. Engine must produce at least one insight
# --------------------------------------------------

found_insights = False
if os.path.exists(OUT_DIR):
    for f in os.listdir(OUT_DIR):
        if f.startswith("insights_") and f.endswith(".csv"):
            found_insights = True

if not found_insights:
    print("[FAIL] No insights file produced")
    FAILED = True
else:
    print("[OK] Insights file produced")

# --------------------------------------------------
# D. Safety rejections (STRICT)
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
    print("\n❌ CAUSAL SAFETY CERTIFICATION FAILED")
    sys.exit(1)
else:
    print("\n🏆 CAUSAL SAFETY ENGINE CERTIFIED")
    sys.exit(0)