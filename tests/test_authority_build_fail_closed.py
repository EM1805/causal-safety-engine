"""Protocol Genesis: falsification and non-regression checks for Authority."""
import importlib.util
import math
from pathlib import Path
import unittest

SOURCE = Path(__file__).resolve().parents[1] / "tools" / "authority_build.py"
spec = importlib.util.spec_from_file_location("pcb_authority_under_test", SOURCE)
authority = importlib.util.module_from_spec(spec)
spec.loader.exec_module(authority)


def valid_edge():
    return {
        "support_n": 40,
        "strength": 0.2,
        "p_sign": 0.9,
        "leakage_gap": 0.01,
        "leakage_corr_future": 0.0,
        "drift_corr_time_source": 0.0,
        "drift_corr_time_target": 0.0,
        "guardrail_flag": 0,
    }


class AuthorityFailClosedTest(unittest.TestCase):
    def classify(self, **overrides):
        edge = valid_edge()
        edge.update(overrides)
        return authority.classify_edge(edge)

    def test_valid_control_remains_allowed(self):
        decision, state, unsafe, reasons, sign = self.classify()
        self.assertEqual((decision, state, unsafe), ("ALLOW", "stable", False))
        self.assertEqual(reasons, ["OK"])
        self.assertEqual(sign, 0.9)

    def test_weak_but_valid_evidence_retains_flag(self):
        decision, state, unsafe, _, _ = self.classify(p_sign=0.7)
        self.assertEqual((decision, state, unsafe), ("FLAG", "flagged", True))

    def test_existing_instability_still_blocks(self):
        decision, _, unsafe, reasons, _ = self.classify(p_sign=0.4)
        self.assertEqual(decision, "BLOCK_INTERVENTION")
        self.assertTrue(unsafe)
        self.assertIn("SIGN_INSTABILITY", reasons)

    def test_nonfinite_values_never_allow(self):
        fields = (
            "support_n", "strength", "p_sign", "leakage_gap",
            "leakage_corr_future", "drift_corr_time_source",
            "drift_corr_time_target",
        )
        for field in fields:
            for value in (float("inf"), -float("inf")):
                with self.subTest(field=field, value=value):
                    decision, _, unsafe, reasons, _ = self.classify(**{field: value})
                    self.assertEqual(decision, "BLOCK_INTERVENTION")
                    self.assertTrue(unsafe)
                    self.assertIn("NONFINITE_EVIDENCE", reasons)

    def test_sign_probability_outside_interval_is_blocked(self):
        for value in (1.000001, -1.000001, 2, -3):
            with self.subTest(value=value):
                decision, _, unsafe, reasons, _ = self.classify(p_sign=value)
                self.assertEqual(decision, "BLOCK_INTERVENTION")
                self.assertTrue(unsafe)
                self.assertIn("INVALID_SIGN_STABILITY", reasons)

    def test_noncanonical_guardrails_block_without_exception(self):
        for value in (2, -1, "nan", float("nan"), float("inf"), "unexpected"):
            with self.subTest(value=value):
                decision, _, unsafe, reasons, _ = self.classify(guardrail_flag=value)
                self.assertEqual(decision, "BLOCK_INTERVENTION")
                self.assertTrue(unsafe)
                self.assertIn("INVALID_GUARDRAIL", reasons)

    def test_fractional_and_malformed_guardrails_fail_closed(self):
        for value in (
            0.5, 1.2, 1.9, -0.2, "0.5", "1.0", "0.0",
            True, False, None, "", [], float("nan"),
            authority.np.float64(0.5),
        ):
            with self.subTest(value=repr(value)):
                decision, _, unsafe, reasons, _ = self.classify(guardrail_flag=value)
                self.assertEqual(decision, "BLOCK_INTERVENTION")
                self.assertTrue(unsafe)
                self.assertIn("INVALID_GUARDRAIL", reasons)

    def test_canonical_guardrail_representations(self):
        for value in (0, 0.0, "0", " 0 ", authority.np.int64(0), authority.np.float64(0)):
            with self.subTest(value=repr(value)):
                self.assertEqual(self.classify(guardrail_flag=value)[0], "ALLOW")
        for value in (1, 1.0, "1", authority.np.int64(1), authority.np.float64(1)):
            with self.subTest(value=repr(value)):
                decision, _, unsafe, reasons, _ = self.classify(guardrail_flag=value)
                self.assertEqual(decision, "BLOCK_INTERVENTION")
                self.assertTrue(unsafe)
                self.assertIn("GUARDRAIL", reasons)

    def test_explicit_guardrail_is_still_blocking(self):
        decision, _, unsafe, reasons, _ = self.classify(guardrail_flag=1)
        self.assertEqual(decision, "BLOCK_INTERVENTION")
        self.assertTrue(unsafe)
        self.assertIn("GUARDRAIL", reasons)

    def test_missing_sign_still_blocks(self):
        decision, _, unsafe, reasons, _ = self.classify(p_sign=float("nan"))
        self.assertEqual(decision, "BLOCK_INTERVENTION")
        self.assertTrue(unsafe)
        self.assertIn("NO_SIGN", reasons)


if __name__ == "__main__":
    unittest.main()
