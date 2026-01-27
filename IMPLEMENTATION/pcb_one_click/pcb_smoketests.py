# pcb_smoketests.py
# Minimal, local-first smoke + sensitivity tests for PCB (Python 3.7+)
import os, sys, shutil
import pandas as pd

def assert_exists(path):
    if not os.path.exists(path):
        raise AssertionError("Missing expected file: %s" % path)

def run_cmd(cmd):
    import subprocess
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError("Command failed (%s)\n\nOutput:\n%s" % (" ".join(cmd), p.stdout))
    return p.stdout

def read_keys(csv_path):
    if not os.path.exists(csv_path):
        return set()
    df = pd.read_csv(csv_path)
    cols = set(df.columns)
    # stable key: source-target-lag for kept insights
    if "kept" in cols:
        df = df[df["kept"] == 1]
    if len(df) == 0:
        return set()
    for c in ["source","target","lag"]:
        if c not in cols:
            return set()
    return set((str(r["source"]), str(r["target"]), int(float(r["lag"]))) for _, r in df.iterrows())

def main():
    # 1) Full pipeline smoke
    out = run_cmd([sys.executable, "pcb_cli.py", "run", "--data", "data.csv"])
    # expected artifacts
    assert_exists("out/data_clean.csv")
    assert_exists("out/data_profile.csv")
    assert_exists("out/edges.csv")
    assert_exists("out/insights_level2.csv")
    assert_exists("out/alerts_today_level28.csv")
    # 3.2 outputs (if enabled in pipeline)
    # (these files may be empty, but should exist)
    assert_exists("out/insights_level3.csv")
    assert_exists("out/insights_level3_ledger.csv")
    assert_exists("out/experiment_trials_enriched_level32.csv")

    # 2) Sensitivity: rerun Level 2.5 with different max_lag and compare kept-edge overlap
    import pcb_insights_level25 as L25

    # baseline keys from current run
    base_keys = read_keys("out/insights_level2.csv")

    # run with smaller lag (will overwrite out/insights_level2.csv)
    L25.main(data_csv_path="data_clean.csv" if os.path.exists("data_clean.csv") else "data.csv",
            target_col="target",
            max_lag=3)

    alt_keys = read_keys("out/insights_level2.csv")

    # compute jaccard
    inter = len(base_keys & alt_keys)
    union = len(base_keys | alt_keys) if (base_keys or alt_keys) else 0
    jacc = (inter / union) if union else 1.0

    # restore by running full pipeline again (keeps the package in a consistent state)
    run_cmd([sys.executable, "pcb_cli.py", "run", "--data", "data.csv"])

    print("\n=== PCB TEST SUMMARY ===")
    print("Smoke test: PASS")
    print("Sensitivity (max_lag 7 vs 3) kept-edge Jaccard: %.3f" % jacc)
    # conservative threshold: if you have very few insights, overlap may be low; we just report it.

if __name__ == "__main__":
    main()
