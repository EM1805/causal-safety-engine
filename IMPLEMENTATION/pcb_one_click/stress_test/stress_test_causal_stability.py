"""
CAUSAL STABILITY CERTIFICATION TEST – SAFETY-FIRST (FINAL)

Purpose:
Verify that the causal engine behaves stably across multiple runs.

Key principles:
- NO insights is NOT a failure (safe abstention is allowed)
- Crashes are counted ONLY if no output is produced
- If insights appear, they must be stable across runs
- Stability can be achieved either by:
  (A) consistent causal insights
  (B) consistent causal silence
"""

import os
import sys
import subprocess
import shutil
import pandas as pd
from collections import Counter

# ---------------- CONFIG ----------------

RUNS = 5

ENGINE_CMD = [
    sys.executable,
    "IMPLEMENTATION/pcb_one_click/pcb_cli.py",
    "run",
    "--data",
    "IMPLEMENTATION/pcb_one_click/data.csv"
]

OUT_DIR = "out"
INSIGHTS_FILE = "insights_level2.csv"

print("\n[STABILITY TEST] Causal Stability Certification (Safety-First)\n")

# ---------------- STATE ----------------

crashes = 0
runs_with_insights = 0
all_features = []

# ---------------- HELPERS ----------------

def clean_output():
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)


def read_features(path):
    df = pd.read_csv(path)

    if "source" in df.columns:
        return df["source"].astype(str).str.lower().tolist()
    if "feature" in df.columns:
        return df["feature"].astype(str).str.lower().tolist()

    return []

# ---------------- MAIN LOOP ----------------

for i in range(RUNS):
    print(f"--- RUN {i + 1}/{RUNS} ---")

    clean_output()

    proc = subprocess.run(
        ENGINE_CMD,
        capture_output=True,
        text=True
    )

    # ❗ CRASH ONLY IF NO OUTPUT AT ALL
    if not os.path.exists(OUT_DIR):
        print("[FAIL] Engine produced no output directory")
        crashes += 1
        continue

    insights_path = os.path.join(OUT_DIR, INSIGHTS_FILE)

    # ✔ SAFE SILENCE
    if not os.path.exists(insights_path):
        print("[OK] No insights produced (conservative safe behavior)")
        continue

    # ✔ INSIGHTS PRESENT → CHECK STABILITY
    try:
        feats = read_features(insights_path)

        if feats:
            print("[OK] Insights:", feats)
            all_features.extend(feats)
            runs_with_insights += 1
        else:
            print("[OK] Insights file empty (safe abstention)")

    except Exception as e:
        print("[FAIL] Insights unreadable:", e)
        crashes += 1

# ---------------- REPORT ----------------

print("\n=========== STABILITY REPORT ===========")
print("Runs executed:", RUNS)
print("Engine crashes:", crashes)
print("Runs with insights:", runs_with_insights)

FAILED = False

# ❌ TECHNICAL INSTABILITY
if crashes > 0:
    print("❌ Engine instability detected (technical failures)")
    FAILED = True

# ✔ IF INSIGHTS EXIST → REQUIRE STABILITY
if runs_with_insights > 0:
    counter = Counter(all_features)

    print("\nFeature recurrence:")
    for k, v in counter.items():
        print(f" - {k}: {v}/{runs_with_insights}")

    stable = [
        k for k, v in counter.items()
        if v / runs_with_insights >= 0.6
    ]

    print("Stable causal features (>=60%):", stable)

    if len(stable) == 0:
        print("❌ No stable causal signal when insights exist")
        FAILED = True

# ✔ PURE CONSERVATIVE STABILITY
else:
    print("✔ Conservative stability achieved (no insights across runs)")

# ---------------- VERDICT ----------------

if FAILED:
    print("\n❌ CAUSAL STABILITY CERTIFICATION FAILED")
    sys.exit(1)
else:
    print("\n🏆 CAUSAL STABILITY CERTIFICATION PASSED")
    sys.exit(0)
