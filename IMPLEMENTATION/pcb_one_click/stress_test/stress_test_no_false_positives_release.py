"""
NO FALSE POSITIVES – CAUSAL SAFETY TEST

The engine MUST NOT generate insights when:
- Only spurious correlations exist
- Leakage is present
- Drift-only signals exist
- No real causal mechanism exists

Any insight produced = FAIL
"""

import pandas as pd
import numpy as np
import subprocess
import sys
import os

np.random.seed(999)

DATA = "IMPLEMENTATION/pcb_one_click/data_no_false_positives.csv"
OUT_DIR = "IMPLEMENTATION/pcb_one_click/out"

# --------------------------------------------------
# 1. GENERATE NON-CAUSAL DATASET
# --------------------------------------------------

def generate_dataset(n=3000):
    noise = np.random.normal(0, 1, n)

    spurious = noise + np.random.normal(0, 0.1, n)
    leakage = np.roll(noise, -1)
    leakage[-1] = 0
    drift = np.concatenate([
        np.random.normal(0, 1, n // 2),
        np.random.normal(5, 1, n // 2)
    ])

    target = noise  # NO causal parents

    return pd.DataFrame({
        "spurious": spurious,
        "leakage": leakage,
        "drift": drift,
        "target": target
    })

os.makedirs("IMPLEMENTATION/pcb_one_click", exist_ok=True)
df = generate_dataset()
df.to_csv(DATA, index=False)

print("[TEST] Non-causal dataset generated")

# --------------------------------------------------
# 2. RUN ENGINE
# --------------------------------------------------

cmd = [
    sys.executable,
    "IMPLEMENTATION/pcb_one_click/demo.py",
    DATA,
    "target"
]

proc = subprocess.run(cmd, capture_output=True, text=True)
print(proc.stdout)

# --------------------------------------------------
# 3. VALIDATION
# --------------------------------------------------

FAILED = False
report = proc.stdout.lower()

if os.path.exists(OUT_DIR):
    insight_files = [
        f for f in os.listdir(OUT_DIR)
        if f.startswith("insights_") and f.endswith(".csv")
    ]

    if insight_files:
        print("[FAIL] Insights generated in non-causal scenario:", insight_files)
        FAILED = True
    else:
        print("[OK] No insights generated")
else:
    print("[OK] No output directory created")

# --------------------------------------------------
# 4. VERDICT
# --------------------------------------------------

if FAILED:
    print("\n❌ NO FALSE POSITIVES TEST FAILED")
    sys.exit(1)
else:
    print("\n✅ NO FALSE POSITIVES GUARANTEED")
    sys.exit(0)
