"""Run every notebook in notebooks/ and solutions/ top to bottom, headless, and report pass/fail. Uses mock mode unless
LLM_PROVIDER is already set, so it costs nothing by default.

    python tools/run_notebooks.py
"""

import os
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("LLM_PROVIDER", "mock")

failed = 0
paths = sorted((ROOT / "notebooks").glob("*.ipynb")) + sorted((ROOT / "solutions").glob("*.ipynb"))
if len(sys.argv) > 1:  # optional filter, e.g. `run_notebooks.py 01`
    paths = [p for p in paths if any(p.name.startswith(a) for a in sys.argv[1:])]
for path in paths:
    nb = nbformat.read(path, as_version=4)
    t0 = time.perf_counter()
    try:
        NotebookClient(nb, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(path.parent)}}).execute()
        print(f"ok    {path.parent.name}/{path.name}  ({time.perf_counter() - t0:.1f}s)")
    except Exception as e:  # report and keep going
        failed += 1
        print(f"FAIL  {path.parent.name}/{path.name}: {str(e).splitlines()[0] if str(e) else type(e).__name__}")
sys.exit(1 if failed else 0)
