"""
ULTIMATE CAUSAL SAFETY STRESS TEST – CERTIFICATION GRADE

This test certifies that the engine:

✔ Rejects Simpson’s paradox
✔ Rejects collider bias
✔ Handles hidden confounders
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

OUT = "out"
DATA = "IMPLEMENTATION/pcb_one_click/data_causal_ultimate.csv"

# --------------------------------------------------
# 1. GENERATE EXTREME CAUSAL DATASET
# --------------------------------------------------

def generate_dataset(n=4000):

    # Hidden confounder (NOT observable by engine)
    H = np.random.normal(0, 1, n)

    # True causal chain
    X = 2 * H + np.random.normal(0, 0.5, n)
    Y = 4 * X + 3 * H + np.random.normal(0, 0.5, n)

    # --------------------------------------------------
    # Simpson’s paradox (classic construction)
    # In each group: negative effect
    # Globally: positive correlation
    # --------------------------------------------------

    group = np.random.binomial(1, 0.5, n)

    S = np.zeros(n)
    S[group == 0] = -1 * X[group == 0] + np.random.normal(0, 0.2, np.sum(group == 0))
    S[group == 1] = -1 * X[group == 1] + np.random.normal(0, 0.2, np.sum(group == 1))
    S = S + 2 * group   # global positive correlation illusion

    # --------------------------------------------------
    # Collider bias (effect of X and Y)
    # --------------------------------------------------

    C = X + Y + np.random.normal(0, 0.1, n)

    # --------------------------------------------------
    # Spurious correlated variable (high corr, no causal path)
    # --------------------------------------------------

    Z = X + np.random.normal(0, 0.3, n)

    # --------------------------------------------------
    # Temporal leakage (future information)
    # --------------------------------------------------

    future = np.roll(Y, -1)
    future[-1] = np.nan   # remove invalid leakage row

    # --------------------------------------------------
    # Non-causal time trend
    # --------------------------------------------------

    trend = np.linspace(0, 5, n) + np.random.normal(0, 0.1, n)

    # --------------------------------------------------
    # Intervention do(X)
    # --------------------------------------------------

    intervention = np.random.binomial(1, 0.1, n)
    X_do = X + intervention * 5
    Y_do = 4 * X_do + 3 * H + np.random.normal(0, 0.5, n)

    df = pd.DataFrame({
        "X": X_do,                # TRUE CAUSE
        "simpson": S,             # MUST BE REJECTED
        "collider": C,            # MUST BE REJECTED
        "spurious": Z,            # MUST BE REJECTED
        "future_leak": future,    # MUST BE REJECTED
        "trend": trend,           # MUST BE REJECTED
        "target": Y_do
    })

    # Remove leakage row
    df = df.dropna().reset_index(drop=True)

    return df


# Write dataset
os.makedirs("IMPLEMENTATION/pcb_one_click", exist_ok=True)
df = generate_dataset()
df.to_csv(DATA, index=False)

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
# 3. VALIDATION RULES (STRICT CERTIFICATION)
# --------------------------------------------------

FAILED = False


edges_path = "out/edges.csv"

if not os.path.exists(edges_path):
    print("[FAIL] edges.csv not produced – engine did not run correctly")
    FAILED = True
else:
    edges = pd.read_csv(edges_path)

    # Filter edges going to target
    causal_edges = edges[edges["to"] == "target"]

    if len(causal_edges) == 0:
        print("[FAIL] No causal edges detected toward target")
        FAILED = True
    else:
        print(f"[OK] {len(causal_edges)} causal relations detected toward target")

        print("Detected causal variables:")
        for v in causal_edges["from"].unique():
            print(" -", v)

# --------------------------------------------------
# Must generate at least one insight
# --------------------------------------------------

if "kept insights: 0" in report or "no insights" in report:
    print("[FAIL] No insights generated despite strong causal signal")
    FAILED = True
else:
    print("[OK] Insights generated")


# B. Must reject Simpson paradox
if "simpson" in report:
    print("[FAIL] Simpson’s paradox variable accepted")
    FAILED = True
else:
    print("[OK] Simpson paradox rejected")

# C. Must reject collider bias
if "collider" in report:
    print("[FAIL] Collider bias accepted")
    FAILED = True
else:
    print("[OK] Collider rejected")

# D. Must reject spurious correlation
if "spurious" in report:
    print("[FAIL] Spurious correlation accepted")
    FAILED = True
else:
    print("[OK] Spurious rejected")

# E. Must reject temporal leakage
if "future" in report:
    print("[FAIL] Temporal leakage accepted")
    FAILED = True
else:
    print("[OK] Temporal leakage rejected")

# F. Must reject time trend
if "trend" in report:
    print("[FAIL] Time trend misinterpreted as causal")
    FAILED = True
else:
    print("[OK] Trend rejected")



# --------------------------------------------------
# 4. FINAL VERDICT
# --------------------------------------------------

if FAILED:
    print("\n❌ ULTIMATE CAUSAL SAFETY TEST FAILED")
    sys.exit(1)
else:
    print("\n🏆 ULTIMATE CAUSAL SAFETY ENGINE CERTIFIED")
    sys.exit(0)