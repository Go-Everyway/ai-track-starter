"""lib/grader.py agreement maths and lib/harness.py's send gate. Mock mode, no network."""

import json
import os
import tempfile
import unittest
from pathlib import Path

import lib.llm as llm
from lib import grader, harness

ROOT = Path(__file__).resolve().parent.parent


class GraderTests(unittest.TestCase):
    def test_agreement_and_kappa(self):
        a = grader.agreement([True, False, True, False], [True, False, False, False])
        # 3 of 4 agree; chance agreement 0.5 x 0.25 + 0.5 x 0.75 = 0.5; kappa = (0.75 - 0.5) / 0.5
        self.assertAlmostEqual(a["accuracy"], 0.75)
        self.assertAlmostEqual(a["kappa"], 0.5)
        self.assertEqual((a["grader_too_easy"], a["grader_too_hard"]), (0, 1))

    def test_labelled_set_has_40_outputs(self):
        rows = [json.loads(l) for l in (ROOT / "evals/labels/outreach_labelled.jsonl").read_text().splitlines() if l.strip()]
        self.assertEqual(len(rows), 40)


class GateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        llm.LOG_PATH = Path(self.tmp.name) / "calls.jsonl"
        os.environ.pop("LLM_API_KEY", None)
        os.environ["LLM_PROVIDER"] = "mock"

    def tearDown(self):
        self.tmp.cleanup()

    def test_failed_step_is_blocked(self):
        allowed, reasons = harness.gate({"company_id": "C1", "description": "x"}, {"ok": False, "output": None, "error": "not JSON"})
        self.assertFalse(allowed)
        self.assertTrue(reasons)


if __name__ == "__main__":
    unittest.main()
