"""Test sets for model-powered steps, and a small runner. Modules 7 to 9 add cases;
Module 10 turns them into a harness.

A suite is a JSONL file in evals/, one case per line:

    {"id": "m07-001", "input": {...}, "expect": [{"check": "field_nonempty", "field": "angle"}, ...]}

`run_suite(path, fn)` calls `fn(case["input"])` for every case and applies each check in
"expect" to what it returns (a dict, or a JSON string that parses to one). A case passes
when every check passes. Checks are plain functions; register your own with @check.

    from lib.evals import run_suite, check
    results = run_suite("evals/m07_outreach_angle.jsonl", my_step)
    print(results.pass_rate, results.failures())
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

CHECKS: dict[str, Callable] = {}


def check(fn: Callable) -> Callable:
    """Register a check. It gets (output, spec, case) and returns None if it passes,
    or a short string saying what failed."""
    CHECKS[fn.__name__] = fn
    return fn


def get(output, path: str):
    """Read a dotted path like "opener.text" or "citations.0" from a dict or list."""
    cur = output
    for part in path.split(".") if path else []:
        if isinstance(cur, list):
            cur = cur[int(part)] if part.isdigit() and int(part) < len(cur) else None
        elif isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
    return cur


# ---------------------------------------------------------------- built-in checks

@check
def field_nonempty(out, spec, case):
    v = get(out, spec["field"])
    return None if v not in (None, "", [], {}) else f"{spec['field']} is empty"


@check
def max_words(out, spec, case):
    n = len(str(get(out, spec["field"]) or "").split())
    return None if n <= spec["n"] else f"{spec['field']} has {n} words (max {spec['n']})"


@check
def contains(out, spec, case):
    v = str(get(out, spec["field"]) or "").lower()
    return None if spec["text"].lower() in v else f"{spec['field']} doesn't mention {spec['text']!r}"


@check
def not_contains(out, spec, case):
    v = str(get(out, spec["field"]) or "").lower()
    return f"{spec['field']} mentions {spec['text']!r}" if spec["text"].lower() in v else None


@check
def equals(out, spec, case):
    v = get(out, spec["field"])
    return None if v == spec["value"] else f"{spec['field']} is {v!r}, expected {spec['value']!r}"


@check
def in_set(out, spec, case):
    v = get(out, spec["field"])
    return None if v in spec["values"] else f"{spec['field']} is {v!r}, not one of {spec['values']}"


@check
def json_schema(out, spec, case):
    """The whole output (or one field) validates against a JSON schema given in the spec."""
    from jsonschema import Draft202012Validator

    errors = sorted(Draft202012Validator(spec["schema"]).iter_errors(get(out, spec.get("field", ""))), key=str)
    return "; ".join(e.message for e in errors[:3]) or None


@check
def subset_of_input(out, spec, case):
    """Every item in an output list must come from a list in the input, e.g. citations
    must be ids of passages that were actually retrieved."""
    got = get(out, spec["field"]) or []
    allowed = set(get(case["input"], spec["input_field"]) or [])
    extra = [x for x in got if x not in allowed]
    return f"{spec['field']} has items not in {spec['input_field']}: {extra}" if extra else None


# ---------------------------------------------------------------- running

@dataclass
class Results:
    suite: str
    rows: list = field(default_factory=list)  # [{"id", "passed", "failures", "output"}]

    @property
    def pass_rate(self) -> float:
        return sum(r["passed"] for r in self.rows) / len(self.rows) if self.rows else 0.0

    def failures(self) -> list:
        return [(r["id"], r["failures"]) for r in self.rows if not r["passed"]]

    def __repr__(self):
        return f"<{self.suite}: {sum(r['passed'] for r in self.rows)}/{len(self.rows)} passed ({self.pass_rate:.0%})>"


def load_suite(path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def run_suite(path, fn: Callable, cases: list[dict] | None = None) -> Results:
    path = Path(path)
    res = Results(path.stem)
    for case in cases if cases is not None else load_suite(path):
        failures = []
        try:
            out = fn(case["input"])
            if isinstance(out, str):
                out = json.loads(out)
        except Exception as e:  # a crash or unparseable output is a failure, not a stop
            out, failures = None, [f"{type(e).__name__}: {e}"]
        if out is not None:
            for spec in case.get("expect", []):
                fail = CHECKS[spec["check"]](out, spec, case)
                if fail:
                    failures.append(fail)
        res.rows.append({"id": case["id"], "passed": not failures, "failures": failures, "output": out})
    return res
