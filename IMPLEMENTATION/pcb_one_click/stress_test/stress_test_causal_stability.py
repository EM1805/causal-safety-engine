
"""
CAUSAL STABILITY STRESS TEST – INDUSTRIAL GRADE

This test verifies that:
- True causal variable X is stable across random seeds
- Small noise does NOT change causal ordering
- No spurious variable becomes causal
- Results are reproducible

FAIL = instability or false positives
"""

import pandas as pd
import numpy as np
import subprocess
import sys
import os

np.random.seed(123)

OUT = "out"
DATA = "IMPLEMENTATION/pcb_one_click/data_causal_stability.csv"

def generate_dataset(seed, n=3000, noise=0.5):

    rng = np.random.default_rng(seed)

    H = rng.normal(0, 1, n)
    X = 2 * H + rng.normal(0, noise, n)
    Y = 4 * X + 3 * H + rng.normal(0, noise, n)

    Z = X + rng.normal(0, 1.0, n)
    trend = np.linspace(0, 3, n) + rng.normal(0, 0.2, n)

    df = pd.DataFrame({
        "X": X,
        "spurious": Z,
        "trend": trend,
        "target": Y
    })

    return df


print("\n[STABILITY TEST] Running repeated causal discovery...\n")

detected_sets = []
SEEDS = [10, 20, 30, 40, 50]
FAILED = False

for seed in SEEDS:

    print(f"\n[RUN] Seed = {seed}")

    df = generate_dataset(seed)
    os.makedirs("IMPLEMENTATION/pcb_one_click", exist_ok=True)
    df.to_csv(DATA, index=False)

    cmd = [
        sys.executable,
        "IMPLEMENTATION/pcb_one_click/demo.py",
        DATA,
        "target"
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)

    if not os.path.exists("out"):
        print("[FAIL] Engine did not produce output folder")
        FAILED = True
        continue

    edges_path = "out/edges.csv"

    if not os.path.exists(edges_path):
        print("[FAIL] edges.csv not produced")
        FAILED = True
        continue

    edges = pd.read_csv(edges_path)

    causal_vars = (
        edges[edges["to"].str.lower() == "target"]["from"]
        .str.lower()
        .tolist()
    )

    print("Detected causal vars:", causal_vars)

    detected_sets.append(set(causal_vars))

    if "x" not in causal_vars:
        print("[FAIL] True causal X lost in this run")
        FAILED = True

    for bad in ["spurious", "trend"]:
        if bad in causal_vars:
            print(f"[FAIL] {bad} incorrectly accepted as causal")
            FAILED = True


print("\n[STABILITY CHECK] Comparing runs...\n")

reference = detected_sets[0]

for i, s in enumerate(detected_sets[1:], start=2):
    if s != reference:
        print(f"[FAIL] Run {i} differs from run 1")
        print("Run 1:", reference)
        print(f"Run {i}:", s)
        FAILED = True
    else:
        print(f"[OK] Run {i} consistent with run 1")


if FAILED:
    print("\n❌ CAUSAL STABILITY TEST FAILED")
    sys.exit(1)
else:
    print("\n🏆 CAUSAL STABILITY ENGINE VERIFIED (REPRODUCIBLE & ROBUST)")
    sys.exit(0)
