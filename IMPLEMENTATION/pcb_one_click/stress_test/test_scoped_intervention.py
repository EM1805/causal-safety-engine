"""Regression coverage for the native log, intake and scoped validation path."""

import csv
from datetime import date, timedelta
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


ENGINE = Path(__file__).resolve().parents[1]
START = date(2025, 1, 1)
TIMES = list(range(105, 1261, 105))
SCOPE = {"insight_id": "I2-test", "action_name": "increase_dose", "dose": "unit-1", "target_col": "target",
         "lag_days": 1, "regime_id": "current", "regime_start": START.isoformat()}


class InterventionTests(unittest.TestCase):
    def run_trial(self, *, kind="benefit", scoped=True, intake=True):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "out").mkdir()
            scope = dict(SCOPE)
            if kind == "regime":
                scope["regime_start"] = (START + timedelta(days=TIMES[-3] - 90)).isoformat()
            (root / "scope.json").write_text(json.dumps(scope), encoding="utf-8")
            (root / "out/insights_level2.csv").write_text(
                "insight_id,source,target,lag,delta_test,strength\nI2-test,dose,target,1,1,0.9\n", encoding="utf-8",
            )
            with (root / "data.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["date", "target", "negative_control_outcome"])
                for i in range(1290):
                    endpoint = i - 1 in TIMES
                    gain = -12 if kind == "regime" and i - 1 >= TIMES[-3] else 12
                    target = 0.2 * math.sin(2 * math.pi * i / 7) + 0.02 * math.cos(2 * math.pi * i / 5)
                    writer.writerow([START + timedelta(days=i), target + (gain if endpoint else 0),
                                     0 if endpoint else 0.1 * math.cos(2 * math.pi * i / 9)])
            args = []
            for i in TIMES:
                recorded_action = "reduce_dose" if kind == "action" else scope["action_name"]
                recorded_dose = "unit-2" if kind == "dose" else scope["dose"]
                args.append(["log", "--insight-id", scope["insight_id"], "--t-index", str(i), "--dose", recorded_dose,
                             "--action-name", recorded_action, "--adherence-flag", "0" if kind == "unperformed" else "1",
                             "--lag-days", "2" if kind == "horizon" else "1", "--regime-id",
                             "previous" if kind == "regime" and i < TIMES[-3] else "current"]
                            + ([] if kind == "unverified" else ["--adherence-verified"]))
            if kind == "duplicate":
                args.append(args[-1])
            validation = ["validate", "--data", "data.csv"] + (["--scope", "scope.json"] if scoped else [])
            steps = [["plan"], *args] + ([["intake"]] if intake else [])
            if kind == "stale_intake":
                steps.append(args[-1])
            steps.append(validation)
            code = 'import json,sys; from pcb_cli import main\nfor step in json.loads(sys.argv[1]):\n if main(step): raise SystemExit(2)'
            result = subprocess.run([sys.executable, "-c", code, json.dumps(steps)], cwd=root,
                                    env={**os.environ, "PYTHONPATH": str(ENGINE)}, text=True, encoding="utf-8",
                                    capture_output=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

            def read(file):
                with (root / "out" / file).open(newline="", encoding="utf-8") as stream:
                    return list(csv.DictReader(stream))

            return read("insights_level3.csv"), read("experiment_trials_enriched_level32.csv")

    def test_matching_evidence_survives_intake_and_uses_next_day(self):
        evidence, trials = self.run_trial()
        self.assertEqual(evidence[0]["status"], "action_supported")
        self.assertEqual(evidence[0]["n_wins"], "12")
        self.assertEqual(len(trials), 12)
        self.assertTrue(all(int(row["outcome_index"]) == int(row["t_index"]) + 1 for row in trials))
        self.assertTrue(all(row["trial_id"] for row in trials))
        self.assertEqual(len(evidence[0]["scope_sha256"]), 64)

    def test_mismatches_and_missing_adherence_are_excluded(self):
        for kind in ("action", "dose", "horizon", "unperformed", "unverified"):
            with self.subTest(kind=kind):
                evidence, trials = self.run_trial(kind=kind)
                self.assertEqual(evidence, [])
                self.assertEqual(trials, [])

    def test_old_regime_trials_cannot_support_current_regime(self):
        evidence, trials = self.run_trial(kind="regime")
        self.assertEqual(evidence[0]["status"], "candidate")
        self.assertEqual(evidence[0]["n_wins"], "0")
        self.assertEqual([int(row["t_index"]) for row in trials], TIMES[-3:])

    def test_unscoped_history_remains_diagnostic(self):
        evidence, trials = self.run_trial(scoped=False, intake=False)
        self.assertEqual(evidence[0]["status"], "candidate")
        self.assertEqual(len(trials), 12)

    def test_duplicate_event_cannot_inflate_support(self):
        _, trials = self.run_trial(kind="duplicate")
        self.assertEqual([int(row["t_index"]) for row in trials], TIMES[:-1])

    def test_new_log_entry_invalidates_stale_intake(self):
        _, trials = self.run_trial(kind="stale_intake")
        self.assertEqual([int(row["t_index"]) for row in trials], TIMES[:-1])


if __name__ == "__main__":
    unittest.main()
