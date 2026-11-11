"""Tests for lib/llm.py. No network and no key: the real code path runs against a tiny
local server that speaks the OpenAI-compatible API."""

import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import lib.llm as llm


class FakeProvider(BaseHTTPRequestHandler):
    seen = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeProvider.seen.append((self.path, body))
        if self.path.endswith("/chat/completions"):
            out = {"id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
                   "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "hello"}}],
                   "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15, "cost": 0.4}}
        else:
            out = {"object": "list", "model": body["model"],
                   "data": [{"object": "embedding", "index": i, "embedding": [1.0, 0.0]} for i, _ in enumerate(body["input"])],
                   "usage": {"prompt_tokens": 5, "total_tokens": 5}}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


class LLMTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        llm.LOG_PATH = Path(self.tmp.name) / "calls.jsonl"
        self.env = dict(os.environ)
        for k in list(os.environ):
            if k.startswith(("LLM_", "EMBED_")):
                del os.environ[k]

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.env)
        self.tmp.cleanup()

    def log(self):
        return [json.loads(l) for l in llm.LOG_PATH.read_text().splitlines()]

    def test_mock_chat_and_embed_are_free_and_logged(self):
        os.environ["LLM_PROVIDER"] = "mock"
        r = llm.chat("Write an opener for a distributor.")
        self.assertTrue(r.text.startswith("[mock reply]"))
        self.assertEqual(r.cost_usd, 0.0)
        a, b, c = llm.embed(["forecast inventory demand", "inventory demand forecast", "dental clinic"])
        self.assertGreater(llm.cosine(a, b), llm.cosine(a, c))
        self.assertEqual([e["kind"] for e in self.log()], ["chat", "embed"])

    def test_no_key_defaults_to_mock(self):
        self.assertTrue(llm.chat("hi").text.startswith("[mock reply]"))

    def test_missing_key_is_a_clear_error(self):
        os.environ.update({"LLM_PROVIDER": "openai-compatible", "LLM_MODEL": "m"})
        with self.assertRaisesRegex(RuntimeError, "LLM_API_KEY"):
            llm.chat("hi")

    def test_real_path_reads_reported_cost_then_budget_stops_calls(self):
        srv = HTTPServer(("127.0.0.1", 0), FakeProvider)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            os.environ.update({"LLM_BASE_URL": f"http://127.0.0.1:{srv.server_port}/v1", "LLM_API_KEY": "test",
                               "LLM_MODEL": "chat-m", "EMBED_MODEL": "emb-m", "LLM_DAILY_BUDGET_USD": "0.5",
                               "LLM_PRICES": json.dumps({"emb-m": [2.0, 0.0]})})
            r = llm.chat([{"role": "user", "content": "hi"}], temperature=0)
            self.assertEqual((r.text, r.input_tokens, r.output_tokens, r.cost_usd), ("hello", 12, 3, 0.4))
            self.assertEqual(FakeProvider.seen[-1][1]["temperature"], 0)
            v = llm.embed(["a", "b"])
            self.assertEqual(len(v), 2)
            self.assertAlmostEqual(self.log()[-1]["cost_usd"], 5 * 2.0 / 1_000_000)  # from LLM_PRICES
            llm.chat("again")  # spend is now 0.4 + tiny, under 0.5
            with self.assertRaises(llm.BudgetExceeded):
                llm.chat("over budget")  # 0.8 logged, at or over 0.5
            self.assertEqual(len([e for e in self.log() if e["kind"] == "chat"]), 2)
        finally:
            srv.shutdown()


if __name__ == "__main__":
    unittest.main()
