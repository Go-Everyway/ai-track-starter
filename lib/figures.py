"""The numbers a module page quotes, written by the notebook that produces them.

    from lib.figures import record
    record("m04", precision_at_50=0.22, test_conversions=63)

Each call merges into figures/<mode>/mNN.json, where <mode> is "mock" or "real" depending on
whether a model key is set. The site reads these files instead of retyping numbers, so a page
can't quote a figure the notebook didn't produce. Record only values that are the same on
every run (no latencies in mock mode). FIGURES_DIR overrides the folder (tools/real_run.py
uses it to write a dated run).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from lib import llm

ROOT = Path(__file__).resolve().parent.parent


def figures_dir() -> Path:
    if os.environ.get("FIGURES_DIR"):
        return Path(os.environ["FIGURES_DIR"])
    return ROOT / "figures" / ("mock" if llm.settings()["provider"] == "mock" else "real")


def _plain(v):
    if hasattr(v, "item"):  # numpy scalars
        v = v.item()
    if isinstance(v, float):
        return round(v, 6)
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    return v


def record(module: str, **values) -> dict:
    """Merge named figures into the module's file and return the full set."""
    path = figures_dir() / f"{module}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    current = json.loads(path.read_text()) if path.exists() else {}
    current.update({k: _plain(v) for k, v in values.items()})
    path.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
    return current


def load(module: str, mode: str = "mock") -> dict:
    path = ROOT / "figures" / mode / f"{module}.json"
    return json.loads(path.read_text()) if path.exists() else {}
