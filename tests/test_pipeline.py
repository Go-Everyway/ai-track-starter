"""lib/pipeline.py: the approval gate and idempotent sends. These are code guarantees,
so they hold in mock mode."""

import os
import tempfile
import unittest
from pathlib import Path

import lib.llm as llm
from lib import pipeline


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        llm.LOG_PATH = Path(cls.tmp.name) / "calls.jsonl"
        os.environ.pop("LLM_API_KEY", None)
        os.environ["LLM_PROVIDER"] = "mock"
        cls.world = pipeline.World.load()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def ready_draft(self, draft_id="D-test"):
        self.world.drafts[draft_id] = {"draft_id": draft_id, "status": "ready_for_approval"}
        return draft_id

    def test_send_refuses_without_approval(self):
        d = self.ready_draft("D-noapprove")
        with self.assertRaises(pipeline.SendRefused):
            pipeline.send(self.world, d, None, idempotency_key="k-none")
        with self.assertRaises(pipeline.SendRefused):
            pipeline.send(self.world, d, "not-a-token", idempotency_key="k-bad")
        self.assertFalse([r for r in self.world.outbox if r["draft_id"] == d])

    def test_token_only_works_for_its_own_draft(self):
        a, b = self.ready_draft("D-a"), self.ready_draft("D-b")
        token = pipeline.approve(self.world, a, approver="tester")
        with self.assertRaises(pipeline.SendRefused):
            pipeline.send(self.world, b, token, idempotency_key="k-wrongdraft")

    def test_retry_after_lost_response_sends_once(self):
        d = self.ready_draft("D-retry")
        token = pipeline.approve(self.world, d, approver="tester")
        pipeline.send_with_retry(self.world, d, token, idempotency_key="k-retry", fail_first=True)
        self.assertEqual(len([r for r in self.world.outbox if r["draft_id"] == d]), 1)

    def test_unverified_draft_cannot_be_approved(self):
        self.world.drafts["D-review"] = {"draft_id": "D-review", "status": "needs_review"}
        with self.assertRaises(pipeline.SendRefused):
            pipeline.approve(self.world, "D-review", approver="tester")


if __name__ == "__main__":
    unittest.main()
