"""Gates that stop bad output from going out (Module 10).

Two places to block:

    before you deploy   `python evals/run.py` runs every test suite and exits non-zero when
                        any suite is under its threshold. Wire it into whatever runs before a
                        change ships (a pre-commit hook, CI, or a habit).
    before each send    `gate(company, result)` checks one output right before it would leave.
                        `queue_for_send()` refuses anything the gate blocks.

Nothing here sends email. `queue_for_send` appends to logs/outbox.jsonl, which is where a
real sending tool would pick messages up after a person approves them (Module 9).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from lib import llm
from lib.grader import code_grade, model_grade

ROOT = Path(__file__).resolve().parent.parent
OUTBOX = ROOT / "logs" / "outbox.jsonl"


def gate(company: dict, result: dict, *, use_model_grader: bool | None = None) -> tuple[bool, list[str]]:
    """Should this outreach-angle result be allowed to send? Returns (allowed, reasons).

    Runs the free code checks always, and the model grader only when a real model is set
    (in mock mode it would rubber-stamp everything). Any failure blocks."""
    if not result.get("ok"):
        return False, [f"step failed: {result.get('error')}"]
    reasons = []
    verdict = code_grade(company, result["output"])
    if verdict["verdict"] == "fail":
        reasons.append(f"code check: {verdict['failure_mode']} ({verdict['reason']})")
    if use_model_grader is None:
        use_model_grader = llm.settings()["provider"] != "mock"
    if use_model_grader and not reasons:
        verdict = model_grade(company, result["output"])
        if verdict["verdict"] == "fail":
            reasons.append(f"model grader: {verdict['failure_mode']} ({verdict['reason']})")
    return not reasons, reasons


def queue_for_send(company: dict, result: dict, **gate_kwargs) -> dict:
    """Queue a message only if it passes the gate. Returns {"queued": bool, "reasons": [...]}."""
    allowed, reasons = gate(company, result, **gate_kwargs)
    if allowed:
        OUTBOX.parent.mkdir(parents=True, exist_ok=True)
        with OUTBOX.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                "company_id": company.get("company_id"), "output": result["output"]}) + "\n")
    return {"queued": allowed, "reasons": reasons}
