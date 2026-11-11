# evals

Test sets for every model-powered step, and the harness that runs them.

Each suite is a JSONL file here, one case per line:

```json
{"id": "m07-001", "input": {"company_id": "C00004"}, "expect": [{"check": "field_nonempty", "field": "angle"}]}
```

`lib/evals.py` runs a suite against any function: `run_suite("evals/m07_outreach_angle.jsonl", step)`.
Built-in checks: `field_nonempty`, `max_words`, `contains`, `not_contains`, `equals`, `in_set`,
`json_schema`, `subset_of_input`. Register your own with `@check`.

- Module 7 adds the first suite: 20 cases for the outreach-angle prompt.
- Modules 8 and 9 add suites for retrieval and the pipeline.
- Module 10 turns them into one command with a pass rate, and blocks a send when a suite fails.
