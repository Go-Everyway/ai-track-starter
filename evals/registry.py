"""Which step each test suite runs against, and the pass rate it has to reach.

Each entry maps a suite file in evals/ (by stem) to:

    step        a function returning the step to test: step(case_input) -> output
    threshold   minimum pass rate; under it, the harness fails
    mock_skip   checks to drop in mock mode because they judge the model's own choices,
                which a mock can't make (on top of any check marked "needs_real_model")

To add a suite: write evals/<stem>.jsonl and add an entry here. Steps import lazily, so a
suite whose code doesn't exist yet is skipped with a message instead of breaking the run.
"""

from __future__ import annotations

import functools
from pathlib import Path

from lib import llm

ROOT = Path(__file__).resolve().parent.parent


def mock_mode() -> bool:
    return llm.settings()["provider"] == "mock"


# ---------------------------------------------------------------- Module 7: outreach angle

def outreach_step(version: str = "v3"):
    from lib.outreach import outreach_angle

    return lambda inp: outreach_angle(inp["company"], version=version)


# ---------------------------------------------------------------- Module 8: RAG

@functools.lru_cache(maxsize=1)
def _rag_index():
    from lib import rag

    # Module 8's "after" setup: 40-word chunks with the document title in front of each.
    return rag.build_index(rag.load_docs(ROOT / "data" / "docs"), embed=not mock_mode(),
                           size=40, overlap=10, prefix_title=True)


def rag_step():
    from lib import rag

    mode = "keyword" if mock_mode() else "hybrid"  # mock vectors only add noise
    return lambda inp: rag.answer(_rag_index(), inp["question"], mode=mode)


# ---------------------------------------------------------------- Module 9: pipeline

def pipeline_step():
    # Module 9 writes lib/pipeline.py with run_case(case_input) -> dict. Until it exists,
    # this raises ImportError and the harness skips the suite.
    from lib.pipeline import run_case

    return run_case


SUITES = {
    "m07_outreach_angle": {"step": outreach_step, "threshold": 1.0, "mock_skip": set()},
    "m08_rag": {"step": rag_step, "threshold": 0.9,
                # Mock always cites its top passage and never declines on its own, so these
                # two checks would grade the mock, not the system.
                "mock_skip": {"cites_answer_passage", "equals"}},
    "m09_pipeline": {"step": pipeline_step, "threshold": 1.0, "mock_skip": set()},
}
