
"""
CAUSAL ENGINE CERTIFICATION STRESS TEST – PRODUCTION GRADE
Compatibile con PCB One Click Engine
"""

import pandas as pd
import numpy as np
import subprocess
import sys
import os

ENGINE = "IMPLEMENTATION/pcb_one_click/demo.py"
DATA   = "IMPLEMENTATION/pcb_one_click/data_stress.csv"
OUT    = "out"

TRUE_CAUSE = "x"
FORBIDDEN  = ["spurious", "trend", "future", "leak"]

np.random.seed(42)

# --------------------------------------------------
# 1. DATASET CAUSALE FORTE
# --------------------------------------------------

def generate_dataset(n=3000):

    H = np.random.normal(0, 1, n)

    X = 2 * H + np.random.normal(0, 0.5, n)
    Y = 4 * X + 3 * H + np.random.normal(0, 0.5, n)

    spurious = np.random.normal(0, 1, n)
    trend = np.linspace(0, 5, n)

    df = pd.DataFrame({
        "x": X,
        "spurious": spurious,
        "trend": trend,
        "target": Y
    })

    return df


os.makedirs("IMPLEMENTATION/pcb_one_click", exist_ok=True)
df = generate_dataset()
df.to_csv(DATA, index=False)

print("[TEST] Dataset generated")

# --------------------------------------------------
# 2. RUN ENGINE
# --------------------------------------------------

cmd = [sys.executable, ENGINE, DATA, "target"]

print("\n[TEST] Running causal engine...\n")

proc = subprocess.run(cmd, capture_output=True, text=True)

print(proc.stdout)

FAILED = False

# --------------------------------------------------
# 3. CHECK OUTPUTS
# --------------------------------------------------

if not os.path.exists(OUT):
    print("[FAIL] Output folder not created")
    FAILED = True

# --- Insights

insights_path = os.path.join(OUT, "insights_level2.csv")

if not os.path.exists(insights_path):
    print("[FAIL] insights_level2.csv not produced")
    FAILED = True
else:
    print("[OK] Insights file produced")

# --- Edges

edges_path = os.path.join(OUT, "edges.csv")

if not os.path.exists(edges_path):
    print("[FAIL] edges.csv not produced")
    FAILED = True
else:
    edges = pd.read_csv(edges_path)
    edges.columns = [c.lower() for c in edges.columns]

    print("[OK] edges.csv produced")
    print("Columns:", list(edges.columns))

    from_col = None
    to_col = None

    for c in edges.columns:
        if "from" in c or "source" in c or "parent" in c:
            from_col = c
        if "to" in c or "target" in c or "child" in c:
            to_col = c

    if from_col is None or to_col is None:
        print("[FAIL] Cannot identify causal edge columns")
        FAILED = True
    else:
        causal = edges[edges[to_col].str.lower() == "target"][from_col].str.lower().tolist()

        print("Detected causal variables:", causal)

        if TRUE_CAUSE in causal:
            print("[OK] True causal variable detected")
        else:
            print("[FAIL] True causal variable NOT detected")
            FAILED = True

        for bad in FORBIDDEN:
            for v in causal:
                if bad in v:
                    print(f"[FAIL] Forbidden variable accepted: {v}")
                    FAILED = True

# --------------------------------------------------
# 4. FINAL VERDICT
# --------------------------------------------------

if FAILED:
    print("\n❌ CAUSAL ENGINE CERTIFICATION FAILED")
    sys.exit(1)
else:
    print("\n🏆 CAUSAL ENGINE CERTIFIED – PRODUCTION READY")
    sys.exit(0)
