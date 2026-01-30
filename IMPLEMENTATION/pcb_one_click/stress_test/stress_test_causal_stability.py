import os
import sys
import subprocess
import pandas as pd
from collections import Counter

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

print("\n[STABILITY TEST] Causal Stability Certification (Safety-Aware)\n")

crashes = 0
all_features = []
runs_with_insights = 0

def read_features(path):
    df = pd.read_csv(path)
    if "source" in df.columns:
        return df["source"].astype(str).str.lower().tolist()
    elif "feature" in df.columns:
        return df["feature"].astype(str).str.lower().tolist()
    else:
        return []

for i in range(RUNS):
    print(f"--- RUN {i+1}/{RUNS} ---")

    # clean outputs
    if os.path.exists(OUT_DIR):
        for f in os.listdir(OUT_DIR):
            try:
                os.remove(os.path.join(OUT_DIR, f))
            except:
                pass
    else:
        os.makedirs(OUT_DIR, exist_ok=True)

    proc = subprocess.run(ENGINE_CMD, capture_output=True, text=True)

    if proc.returncode != 0:
        print("[FAIL] Engine execution error")
        crashes += 1
        continue

    insights_path = os.path.join(OUT_DIR, INSIGHTS_FILE)

    if not os.path.exists(insights_path):
        print("[OK] No insights produced (conservative safe behavior)")
        continue

    try:
        feats = read_features(insights_path)
        if feats:
            print("[OK] Insights:", feats)
            all_features.extend(feats)
            runs_with_insights += 1
        else:
            print("[OK] Insights file empty (safe)")
    except Exception as e:
        print("[FAIL] Insights unreadable:", e)
        crashes += 1

# ---------------- REPORT ----------------

print("\n=========== STABILITY REPORT ===========")
print("Runs executed:", RUNS)
print("Engine crashes:", crashes)
print("Runs with insights:", runs_with_insights)

FAILED = False

if crashes > 0:
    print("❌ Engine instability detected (execution errors)")
    FAILED = True

if runs_with_insights > 0:
    counter = Counter(all_features)
    print("Feature recurrence:")
    for k, v in counter.items():
        print(f" - {k}: {v}/{runs_with_insights}")

    stable = [k for k, v in counter.items() if v / runs_with_insights >= 0.6]
    print("Stable causal features (>=60%):", stable)

    if len(stable) == 0:
        print("❌ No stable causal signal when insights exist")
        FAILED = True
else:
    print("✔ Conservative stability: no insights across runs")

if FAILED:
    print("\n❌ CAUSAL STABILITY CERTIFICATION FAILED")
    sys.exit(1)
else:
    print("\n🏆 CAUSAL STABILITY CERTIFICATION PASSED")
    sys.exit(0)
