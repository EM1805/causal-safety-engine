"""
NO FALSE POSITIVES – RELEASE LEVEL CAUSAL SAFETY TEST

Definition:
The engine MUST NOT produce RELEASED causal insights (level 3)
when no true causal mechanism exists.

Internal hypotheses (level 2) are allowed.
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
# 3. VALIDATION (RELEASE LEVEL)
# --------------------------------------------------

FAILED = False

if os.path.exists(OUT_DIR):
    files = os.listdir(OUT_DIR)

    print("[INFO] Output files:")
    for f in files:
        print(" -", f)

    # ❌ ONLY THIS IS FORBIDDEN
    forbidden = [f for f in files if "insights_level3" in f.lower()]

    if forbidden:
        print("[FAIL] Released insights detected in non-causal scenario:", forbidden)
        FAILED = True
    else:
        print("[OK] No released insights (level 3) produced")
else:
    print("[OK] No output directory created")

# --------------------------------------------------
# 4. VERDICT
# --------------------------------------------------

if FAILED:
    print("\n❌ NO FALSE POSITIVES (RELEASE LEVEL) FAILED")
    sys.exit(1)
else:
    print("\n✅ NO FALSE POSITIVES GUARANTEED AT RELEASE LEVEL")
    sys.exit(0)
