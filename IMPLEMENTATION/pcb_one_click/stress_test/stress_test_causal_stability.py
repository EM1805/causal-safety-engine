
import os
import sys
import subprocess
import pandas as pd
import numpy as np

np.random.seed(123)

RUNS = 5

ENGINE_CMD = [
    sys.executable,
    "IMPLEMENTATION/pcb_one_click/demo.py",
    "IMPLEMENTATION/pcb_one_click/data.csv",
    "target"
]

# Feature realmente presenti nel dataset
EXPECTED_CAUSES = ["feature_activity", "feature_sleep", "feature_stress"]

print("\n[STABILITY TEST] Starting multi-run causal stability certification...\n")

crashes = 0
detections = {f: 0 for f in EXPECTED_CAUSES}

def extract_causes(edges):
    cols = [c.lower() for c in edges.columns]
    edges.columns = cols

    if "to" in cols and "from" in cols:
        causes = edges[edges["to"] == "target"]["from"].astype(str).str.lower().tolist()
    elif "target" in cols and "source" in cols:
        causes = edges[edges["target"] == "target"]["source"].astype(str).str.lower().tolist()
    else:
        return []

    return causes

for i in range(RUNS):
    print(f"\n--- RUN {i+1}/{RUNS} ---")

    if os.path.exists("out"):
        for f in os.listdir("out"):
            os.remove(os.path.join("out", f))
    else:
        os.makedirs("out", exist_ok=True)

    proc = subprocess.run(ENGINE_CMD, capture_output=True, text=True)

    if not os.path.exists("out/edges.csv"):
        print("[FAIL] edges.csv not produced")
        crashes += 1
        continue

    try:
        edges = pd.read_csv("out/edges.csv")
    except Exception:
        print("[FAIL] edges.csv unreadable")
        crashes += 1
        continue

    causes = extract_causes(edges)

    if len(causes) == 0:
        print("[WARN] No causal edges detected in this run")
        continue

    print("Detected causes:", causes)

    for f in EXPECTED_CAUSES:
        if f.lower() in causes:
            detections[f] += 1

print("\n========== STABILITY REPORT ==========")
print("Runs executed:", RUNS)
print("Engine crashes:", crashes)

print("\nDetection frequencies:")
for f, c in detections.items():
    print(f" - {f}: {c}/{RUNS}")

best_feature = max(detections, key=lambda k: detections[k])
best_rate = detections[best_feature] / RUNS

FAILED = False

if crashes > 0:
    print("\n❌ Engine is NOT stable (crashes detected)")
    FAILED = True

if best_rate < 0.6:
    print("\n❌ Causal detection NOT stable enough (<60% on any feature)")
    FAILED = True
else:
    print(f"\n[OK] Stable dominant causal feature: {best_feature} ({best_rate*100:.0f}%)")

if FAILED:
    print("\n❌ STABILITY CERTIFICATION FAILED")
    sys.exit(1)
else:
    print("\n🏆 STABILITY CERTIFICATION PASSED")
    sys.exit(0)
