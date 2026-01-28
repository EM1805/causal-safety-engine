"""
CAUSAL SILENCE TEST – NO FALSE POSITIVES (RELEASE GATE)

Purpose:
Verify that the engine remains causally silent when
NO valid intervention exists.

Allowed:
- Level 2 exploratory artifacts (non-causal, non-actionable)

Forbidden:
- Any promoted causal insight (Level >= 3)
- Any causal claim or recommendation

Any violation = FAIL
"""

import os
import sys
import subprocess
import pandas as pd

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

DATA = "IMPLEMENTATION/pcb_one_click/data_no_false_positives.csv"
OUT_DIR = "IMPLEMENTATION/pcb_one_click/out"

ENGINE_CMD = [
    sys.executable,
    "IMPLEMENTATION/pcb_one_click/demo.py",
    DATA,
    "target"
]

FORBIDDEN_LEVELS = ["level3", "level4"]

FORBIDDEN_WORDS = [
    "cause",
    "causal",
    "intervention confirmed",
    "leads to",
    "drives",
    "impact on"
]

print("\n[CAUSAL SILENCE TEST]")
print("[INFO] Dataset:", DATA)

# --------------------------------------------------
# 1. CLEAN PREVIOUS OUTPUTS
# --------------------------------------------------

if os.path.exists(OUT_DIR):
    for f in os.listdir(OUT_DIR):
        try:
            os.remove(os.path.join(OUT_DIR, f))
        except Exception:
            pass
else:
    os.makedirs(OUT_DIR, exist_ok=True)

# --------------------------------------------------
# 2. RUN ENGINE
# --------------------------------------------------

proc = subprocess.run(
    ENGINE_CMD,
    capture_output=True,
    text=True
)

print(proc.stdout)
report = proc.stdout.lower()

FAILED = False

# --------------------------------------------------
# 3. CHECK OUTPUT FILES
# --------------------------------------------------

files = os.listdir(OUT_DIR)
print("[INFO] Output files:", files)

insight_files = [f for f in files if f.startswith("insights_")]

# --------------------------------------------------
# 4. RULE A — NO PROMOTED CAUSAL INSIGHTS
# --------------------------------------------------

for f in insight_files:
    for lvl in FORBIDDEN_LEVELS:
        if lvl in f.lower():
            print(f"[FAIL] Forbidden promoted insight generated: {f}")
            FAILED = True

# --------------------------------------------------
# 5. RULE B — LEVEL 2 ALLOWED BUT NON-ACTIONABLE
# --------------------------------------------------

if "insights_level2.csv" in files:
    df = pd.read_csv(os.path.join(OUT_DIR, "insights_level2.csv"))
    if len(df) > 0:
        print("[INFO] Level 2 exploratory insights detected (allowed)")
    else:
        print("[OK] No level 2 insights")
else:
    print("[OK] No level 2 insights")

# --------------------------------------------------
# 6. RULE C — NO CAUSAL LANGUAGE
# --------------------------------------------------

for word in FORBIDDEN_WORDS:
    if word in report:
        print(f"[FAIL] Forbidden causal language found: '{word}'")
        FAILED = True

# --------------------------------------------------
# 7. FINAL VERDICT
# --------------------------------------------------

if FAILED:
    print("\n❌ CAUSAL SILENCE TEST FAILED")
    sys.exit(1)
else:
    print("\n✅ CAUSAL SILENCE VERIFIED")
    print("✅ No False Positives under Non-Causal Conditions")
    sys.exit(0)
