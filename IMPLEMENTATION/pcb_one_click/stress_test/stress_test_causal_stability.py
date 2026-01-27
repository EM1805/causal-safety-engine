
import os
import sys
import subprocess
import pandas as pd
import numpy as np

np.random.seed(123)

RUNS = 5
TOP_K = 4   # engine usually generates ~4 insights per run

ENGINE_CMD = [
    sys.executable,
    "IMPLEMENTATION/pcb_one_click/demo.py",
    "IMPLEMENTATION/pcb_one_click/data_with_insights.csv",
    "target"
]

print("\n[STABILITY TEST] Multi-Insight Causal Stability Certification\n")

crashes = 0
feature_counts = {}

def read_insights():
    for fname in os.listdir("out"):
        if fname.startswith("insights") and fname.endswith(".csv"):
            try:
                return pd.read_csv(os.path.join("out", fname))
            except Exception:
                return None
    return None

def extract_features(insights, top_k):
    cols = [c.lower() for c in insights.columns]
    insights.columns = cols

    # Try common schemas used by the engine
    for col in ["source", "feature", "variable", "cause"]:
        if col in cols:
            return insights[col].astype(str).str.lower().head(top_k).tolist()

    # Fallback: first column
    return insights.iloc[:, 0].astype(str).str.lower().head(top_k).tolist()


for i in range(RUNS):
    print(f"\n--- RUN {i+1}/{RUNS} ---")

    # Clean output folder
    if os.path.exists("out"):
        for f in os.listdir("out"):
            os.remove(os.path.join("out", f))
    else:
        os.makedirs("out", exist_ok=True)

    proc = subprocess.run(ENGINE_CMD, capture_output=True, text=True)

    if proc.returncode != 0:
        print("[FAIL] Engine crashed")
        crashes += 1
        continue

    insights = read_insights()

    if insights is None or len(insights) == 0:
        print("[FAIL] No insights produced")
        crashes += 1
        continue

    feats = extract_features(insights, TOP_K)

    print("[OK] Insights this run:", feats)

    for f in feats:
        feature_counts[f] = feature_counts.get(f, 0) + 1


print("\n========== STABILITY REPORT ==========")
print("Runs executed:", RUNS)
print("Engine crashes:", crashes)

print("\nFeature recurrence:")
for f, n in sorted(feature_counts.items(), key=lambda x: -x[1]):
    print(f" - {f}: {n}/{RUNS}")

stable_features = [f for f, n in feature_counts.items() if n / RUNS >= 0.6]

print("\nStable causal features (>=60% runs):", stable_features)

FAILED = False

if crashes > 0:
    print("❌ Engine NOT stable (crashes detected)")
    FAILED = True

if len(stable_features) < 2:
    print("❌ Not enough stable causal features (need >=2)")
    FAILED = True

if FAILED:
    print("\n❌ MULTI-INSIGHT STABILITY CERTIFICATION FAILED")
    sys.exit(1)
else:
    print("\n🏆 MULTI-INSIGHT STABILITY CERTIFICATION PASSED")
    sys.exit(0)
