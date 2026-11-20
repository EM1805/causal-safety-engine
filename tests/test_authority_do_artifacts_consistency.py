"""Regression tests for synchronized do-check authority exports."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

SOURCE = Path(__file__).resolve().parents[1] / "tools" / "authority_propagate_do_check.py"
spec = importlib.util.spec_from_file_location("authority_do_check_under_test", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DoArtifactParityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)
        (self.out / "DO_CHECK").mkdir()
        (self.out / "AUTHORITY").mkdir()

        # Two not-identifiable edges: one can be FLAGged for review, but no
        # intervention; the other has a hard leakage risk and must BLOCK.
        rows = [
            self._row("A", "B", leakage_gap=0.0),
            self._row("C", "D", leakage_gap=0.8),
            self._row("E", "F", leakage_gap=0.0),
        ]
        pd.DataFrame(rows).to_csv(self.out / "AUTHORITY" / "causal_authority.csv", index=False)
        with (self.out / "AUTHORITY" / "causal_authority.json").open("w") as handle:
            json.dump({"records": rows}, handle)
        with (self.out / "DO_CHECK" / "do_identifiability.json").open("w") as handle:
            json.dump({"results": [
                {"source": "A", "target": "B", "lag": 1, "identifiable": False,
                 "reason_code": "DO_NOT_IDENTIFIABLE"},
                {"source": "C", "target": "D", "lag": 1, "identifiable": False,
                 "reason_code": "DO_NOT_IDENTIFIABLE"},
                {"source": "E", "target": "F", "lag": 1, "identifiable": True,
                 "reason_code": "IDENTIFIED"},
            ]}, handle)

    def _row(self, src, tgt, **override):
        data = {
            "source": src, "target": tgt, "lag": 1,
            "review_action": "ALLOW", "intervention_unsafe": False,
            "reason_code": "BASELINE", "evidence_summary": "baseline",
            "support_n": 40, "strength": 0.2, "p_sign": 0.9,
            "sign_stability": 0.9, "leakage_gap": 0.0,
            "leakage_corr_future": 0.0, "drift_corr_time_source": 0.0,
            "drift_corr_time_target": 0.0, "guardrail_flag": 0,
        }
        data.update(override)
        return data

    def _read_outputs(self):
        df = pd.read_csv(self.out / "AUTHORITY" / "causal_authority.csv").set_index("source")
        with (self.out / "AUTHORITY" / "causal_authority.json").open() as handle:
            records = {r["source"]: r for r in json.load(handle)["records"]}
        return df, records

    def test_review_actions_and_unsafe_flags_agree(self):
        module.run(str(self.out))
        df, records = self._read_outputs()
        for src, action in (("A", "FLAG"), ("C", "BLOCK_INTERVENTION"), ("E", "ALLOW")):
            with self.subTest(source=src):
                self.assertEqual(df.loc[src, "review_action"], action)
                self.assertEqual(records[src]["review_action"], action)
        for src in ("A", "C"):
            self.assertTrue(bool(df.loc[src, "intervention_unsafe"]))
            self.assertIs(records[src]["intervention_unsafe"], True)
        self.assertIn("DO_NOT_IDENTIFIABLE_FLAGGED", records["A"]["reason_code"])
        self.assertNotIn("DO_NOT_IDENTIFIABLE_FLAGGED", records["C"]["reason_code"])
        self.assertEqual(records["E"]["intervention_unsafe"], False)

    def test_repeated_propagation_does_not_duplicate_reasons(self):
        module.run(str(self.out))
        module.run(str(self.out))
        df, records = self._read_outputs()
        for src in ("A", "C"):
            reasons = records[src]["reason_code"].split("|")
            self.assertEqual(len(reasons), len(set(reasons)))
            self.assertEqual(records[src]["evidence_summary"].count("do=not-identifiable"), 1)
            self.assertEqual(df.loc[src, "review_action"], records[src]["review_action"])


    def test_invalid_guardrail_fails_closed_in_both_artifacts(self):
        csv_path = self.out / "AUTHORITY" / "causal_authority.csv"
        json_path = self.out / "AUTHORITY" / "causal_authority.json"
        frame = pd.read_csv(csv_path)
        frame.loc[frame["source"] == "A", "guardrail_flag"] = "unexpected"
        frame.to_csv(csv_path, index=False)
        records = json.loads(json_path.read_text())
        next(r for r in records["records"] if r["source"] == "A")["guardrail_flag"] = "unexpected"
        json_path.write_text(json.dumps(records))
        module.run(str(self.out))
        df, records = self._read_outputs()
        self.assertEqual(df.loc["A", "review_action"], "BLOCK_INTERVENTION")
        self.assertEqual(records["A"]["review_action"], "BLOCK_INTERVENTION")
        self.assertTrue(records["A"]["intervention_unsafe"])

    def test_json_hard_risk_must_not_be_overridden_by_csv_soft_flag(self):
        json_path = self.out / "AUTHORITY" / "causal_authority.json"
        records = json.loads(json_path.read_text())
        next(r for r in records["records"] if r["source"] == "A")["guardrail_flag"] = 1
        json_path.write_text(json.dumps(records))
        module.run(str(self.out))
        df, records = self._read_outputs()
        # The representations disagree about risk; do not silently promote
        # the higher-risk JSON artifact to a review-only FLAG.
        self.assertEqual(df.loc["A", "review_action"], "FLAG")
        self.assertEqual(records["A"]["review_action"], "BLOCK_INTERVENTION")
        self.assertTrue(records["A"]["intervention_unsafe"])



if __name__ == "__main__":
    unittest.main()
