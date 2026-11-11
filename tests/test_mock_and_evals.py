"""Mock structured output, mock tool calls, and the evals runner."""

import json
import os
import tempfile
import unittest
from pathlib import Path

import lib.llm as llm
from lib.evals import run_suite


class MockStructuredTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        llm.LOG_PATH = Path(self.tmp.name) / "calls.jsonl"
        os.environ.pop("LLM_API_KEY", None)
        os.environ["LLM_PROVIDER"] = "mock"

    def tearDown(self):
        self.tmp.cleanup()

    def test_json_schema_gets_a_valid_instance(self):
        schema = {"type": "object", "required": ["angle", "score", "tags", "tone"],
                  "properties": {"angle": {"type": "string"}, "score": {"type": "integer", "minimum": 1},
                                 "tags": {"type": "array", "items": {"type": "string"}},
                                 "tone": {"type": "string", "enum": ["plain", "warm"]}}}
        r = llm.chat("Write an angle for Norelley Distributors",
                     response_format={"type": "json_schema", "json_schema": {"name": "a", "schema": schema}})
        out = json.loads(r.text)
        self.assertEqual(set(out), {"angle", "score", "tags", "tone"})
        self.assertEqual((out["score"], out["tone"]), (1, "plain"))
        self.assertIsInstance(out["tags"], list)

    def test_tool_call_round_trip(self):
        tools = [{"type": "function", "function": {"name": "lookup_company", "parameters": {
            "type": "object", "required": ["company_id"], "properties": {"company_id": {"type": "string"}}}}}]
        msgs = [{"role": "user", "content": "Enrich C00004"}]
        r = llm.chat(msgs, tools=tools)
        self.assertEqual(r.tool_calls[0]["name"], "lookup_company")
        self.assertIn("company_id", r.tool_calls[0]["arguments"])
        msgs += [llm.assistant_message(r), llm.tool_result_message(r.tool_calls[0]["id"], {"employees": 99})]
        r2 = llm.chat(msgs, tools=tools)
        self.assertFalse(r2.tool_calls)
        self.assertTrue(r2.text)


class EvalsTests(unittest.TestCase):
    def test_run_suite(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "s.jsonl"
            cases = [
                {"id": "a", "input": {"ids": ["p1", "p2"]}, "expect": [
                    {"check": "field_nonempty", "field": "answer"},
                    {"check": "subset_of_input", "field": "citations", "input_field": "ids"}]},
                {"id": "b", "input": {"ids": ["p1"]}, "expect": [
                    {"check": "subset_of_input", "field": "citations", "input_field": "ids"},
                    {"check": "max_words", "field": "answer", "n": 2}]},
                {"id": "d", "input": {"ids": ["p1"]}, "expect": [{"check": "json_schema", "schema": {
                    "type": "object", "required": ["answer"], "properties": {"answer": {"type": "string"}}}}]},
                {"id": "c", "input": {}, "expect": [{"check": "field_nonempty", "field": "answer"}]},
            ]
            p.write_text("".join(json.dumps(c) + "\n" for c in cases))

            def step(inp):
                if not inp:
                    raise ValueError("no input")
                return json.dumps({"answer": "it ships nightly", "citations": ["p1", "p2"]})

            res = run_suite(p, step)
            self.assertEqual([r["passed"] for r in res.rows], [True, False, True, False])
            self.assertAlmostEqual(res.pass_rate, 2 / 4)
            self.assertEqual(len(res.failures()[0][1]), 2)  # case b: bad citation and too long
            self.assertIn("ValueError", res.failures()[1][1][0])


if __name__ == "__main__":
    unittest.main()
