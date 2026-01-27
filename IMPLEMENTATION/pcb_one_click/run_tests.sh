#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

count_flagged() {
python - <<'PY'
import pandas as pd
try:
    df = pd.read_csv('out/guardrail_audit.csv')
    print(len(df))
except Exception:
    print(0)
PY
}

count_by_reason() {
python - <<'PY'
import pandas as pd
from collections import Counter
try:
    df = pd.read_csv('out/guardrail_audit.csv')
    if 'guardrail_reason' not in df.columns:
        print('none')
    else:
        c = Counter([str(x) for x in df['guardrail_reason'].fillna('').tolist()])
        # print top 3
        for k,v in c.most_common(3):
            if k == '':
                k='(empty)'
            print(f"{k}:{v}")
except Exception:
    print('none')
PY
}

run_case () {
  local name="$1"; local src="$2"
  echo ""; echo "=== TEST: $name ==="
  cp "$src" data.csv
  rm -rf out
  mkdir -p out
  python pcb_insights_level25.py > out/_level25.log 2>&1
  python pcb_guardrails_audit.py > out/_audit.log 2>&1
  echo "flagged_rows=$(count_flagged)"
  echo "top_reasons:"; count_by_reason
}

run_case "baseline" "data_baseline.csv"
run_case "with_leak" "data_with_leak.csv"
run_case "with_drift" "data_with_drift.csv"

echo ""; echo "Logs saved in out/_level25.log and out/_audit.log for each run (overwritten per case)."
