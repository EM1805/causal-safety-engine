Adversarial Causal Stress Test Suite

Level 5 — Hidden Confounder + Feedback + Regime Drift

Designed to break causal discovery & safety engines

import numpy as np import pandas as pd

=========================

CONFIGURATION

=========================

N = 600                # length of time series SEED = 42              # reproducibility DRIFT_POINT = N // 2   # regime change time

np.random.seed(SEED)

=========================

LATENT (HIDDEN) CONFOUNDER

=========================

Hidden driver that affects everything but is NOT in the dataset

U = np.random.normal(0, 1, N)

=========================

OBSERVED EXOGENOUS DRIVERS (confounded)

=========================

sleep = 0.9 * U + np.random.normal(0, 0.3, N) activity = 0.9 * U + np.random.normal(0, 0.3, N) stress = 0.9 * U + np.random.normal(0, 0.3, N)

=========================

CORE DYNAMICS

=========================

mood = np.zeros(N) target = np.zeros(N)

for t in range(1, N):

# -------- Regime 1 (normal causal structure)
if t < DRIFT_POINT:
    mood[t] = (
        0.6 * sleep[t-1] +
        0.4 * activity[t-1] -
        0.7 * stress[t-1] +
        0.5 * mood[t-1] +           # autoregression
        0.6 * U[t-1] +               # hidden confounder influence
        np.random.normal(0, 0.4)
    )

# -------- Regime 2 (causal inversion + feedback)
else:
    mood[t] = (
        -0.7 * sleep[t-1] +          # SIGN INVERSION (drift)
        0.8 * stress[t-1] +           # role reversal
        0.4 * mood[t-1] +             # persistence
        0.5 * target[t-1] +           # feedback loop from target
        0.6 * U[t-1] +                # hidden confounder still active
        np.random.normal(0, 0.4)
    )

# Target is noisy transformation of mood with memory
target[t] = (
    0.7 * mood[t] +
    0.3 * target[t-1] +
    np.random.normal(0, 0.2)
)

=========================

NORMALIZATION

=========================

mood = (mood - mood.mean()) / mood.std() target = (target - target.mean()) / target.std()

=========================

FINAL DATASET

=========================

stress_test_df = pd.DataFrame({ "sleep": sleep, "activity": activity, "stress": stress, "mood": mood, "target": target })

stress_test_df.to_csv("dataset_stress_level5.csv", index=False)

print("\n=== STRESS TEST LEVEL 5 GENERATED ===") print("Structure includes:") print(" - Hidden confounder U") print(" - Multi-driver") print(" - Autoregression") print(" - Feedback loop") print(" - Regime change (causal inversion)") print("\nSaved as: dataset_stress_level5.csv")

=========================

GROUND TRUTH SUMMARY (for evaluation)

=========================

print("\nGROUND TRUTH (for audit):") print("Regime 1 (t <", DRIFT_POINT, "):") print("  sleep(t-1)  → mood  ( + )") print("  activity(t-1) → mood  ( + )") print("  stress(t-1) → mood  ( - )") print("  mood(t-1) → mood") print("  U → all")

print("\nRegime 2 (t ≥", DRIFT_POINT, "):") print("  sleep(t-1)  → mood  ( - )  [inverted]") print("  stress(t-1) → mood  ( + )  [role swap]") print("  target(t-1) → mood  [feedback]") print("  mood(t-1) → mood") print("  U → all")

print("\nEXPECTED BEHAVIOR OF A SAFE ENGINE:") print(" - VERY FEW or ZERO insights") print(" - NO stable decision rules") print(" - Detection of drift / instability") print(" - NO false confident policies")