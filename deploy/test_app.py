"""Local checks for deploy/app.py. No network, no key, no Flask needed.

    python deploy/test_app.py
"""

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

import app


class FakeProvider(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert body["max_tokens"] == app.MAX_TOKENS
        out = {"model": body["model"],
               "choices": [{"message": {"role": "assistant", "content": "It predicts what to reorder."}}],
               "usage": {"prompt_tokens": 14, "completion_tokens": 6, "cost": 0.00002}}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


class AppTests(unittest.TestCase):
    def test_mock_mode_without_a_key(self):
        r = app.answer({})
        self.assertTrue(r["ok"] and r["mock"])
        json.dumps(r)  # must be JSON-serializable

    def test_missing_model_is_an_error_not_a_crash(self):
        self.assertFalse(app.answer({"LLM_API_KEY": "k"})["ok"])

    def test_real_path_against_a_fake_provider(self):
        srv = HTTPServer(("127.0.0.1", 0), FakeProvider)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            r = app.answer({"LLM_API_KEY": "secret-test-key", "LLM_MODEL": "m",
                            "LLM_BASE_URL": f"http://127.0.0.1:{srv.server_port}/v1"})
            self.assertEqual((r["ok"], r["mock"], r["input_tokens"], r["cost_usd"]), (True, False, 14, 0.00002))
            self.assertNotIn("secret-test-key", json.dumps(r))
        finally:
            srv.shutdown()

    @unittest.skipIf(app.Flask is None, "Flask not installed")
    def test_flask_route_returns_json(self):
        resp = app.app.test_client().get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("reply", resp.get_json())


if __name__ == "__main__":
    unittest.main()
