"""lib/ops.py: retry decisions, failover, alerts and the deploy gate. No network; failures are injected."""

import tempfile
import unittest
from pathlib import Path

import lib.llm as llm
from lib import ops


class HTTPError(Exception):
    def __init__(self, status_code, msg=""):
        super().__init__(msg)
        self.status_code = status_code


class OpsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_retry_decisions(self):
        self.assertTrue(ops.is_retryable(TimeoutError()))
        self.assertTrue(ops.is_retryable(HTTPError(503)))
        self.assertTrue(ops.is_retryable(HTTPError(429, "rate limited")))
        self.assertFalse(ops.is_retryable(HTTPError(429, "spend limit reached")))
        self.assertFalse(ops.is_retryable(HTTPError(400)))
        self.assertFalse(ops.is_retryable(ValueError("bad input")))

    def test_failover_to_fallback_when_primary_is_down(self):
        def call(messages, *, model=None, **params):
            if model == "primary-v1":
                raise HTTPError(503, "down")
            return llm.Reply("ok from fallback", model, 1, 1, 0.0, 0.0)

        client = ops.Resilient("primary-v1", "fallback-v1", pins={"primary-v1", "fallback-v1"},
                               call=call, attempts=2, sleep=lambda s: None, log_path=self.dir / "ops.jsonl")
        r = client.chat("hi")
        self.assertEqual((r.text, r.model), ("ok from fallback", "fallback-v1"))

    def test_unpinned_model_is_refused(self):
        client = ops.Resilient("primary-v1", pins={"primary-v1"}, call=lambda *a, **k: None,
                               log_path=self.dir / "ops.jsonl")
        with self.assertRaises(ops.Unpinned):
            client.chat("hi", model="some-latest-alias")

    def test_deploy_blocked_when_evals_fail(self):
        deployed = []
        with self.assertRaises(ops.DeployBlocked):
            ops.deploy_if_green(lambda: deployed.append(1), check=lambda: (False, "m07 0/20"))
        self.assertEqual(deployed, [])
        ops.deploy_if_green(lambda: deployed.append(1), check=lambda: (True, "all suites pass"))
        self.assertEqual(deployed, [1])


if __name__ == "__main__":
    unittest.main()
