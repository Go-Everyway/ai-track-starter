"""Run solution notebooks against a real model, once, and save the results with model, date and cost.

This is how the course's published real-model results are made (fix list P1.1). It spends money,
so it asks before it starts and stops at a budget.

    1. Put your key in .env (LLM_API_KEY=...), never in this command or in a notebook.
    2. Pick models from your provider's list, then run:
       python tools/real_run.py --small <cheap-model-id> --large <bigger-model-id> --budget 10

Writes runs/<YYYY-MM-DD>-<small-model>/:
    notebooks/   the executed solution notebooks, outputs included
    figures/     the named figures each notebook recorded (lib/figures.py)
    logs/        one call log per module: tokens, cost, latency (no prompts or replies)
    summary.json models, date, base URL, per-module status, cost and tokens, and the total
Commit that folder; the site publishes from it. Before saving, every output file is checked for
your key and the run aborts if it appears anywhere.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_MODULES = ["06", "07", "08", "09", "10", "11"]


def main() -> int:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser(description="Run solution notebooks against a real model.")
    ap.add_argument("--small", default=os.environ.get("SMALL_MODEL"), help="cheap model id (also the default model)")
    ap.add_argument("--large", default=os.environ.get("LARGE_MODEL"), help="larger model id for modules that compare")
    ap.add_argument("--embed", default=os.environ.get("EMBED_MODEL"), help="embedding model id (Module 8)")
    ap.add_argument("--budget", type=float, default=10.0, help="stop when the run's logged spend reaches this many USD")
    ap.add_argument("--modules", nargs="*", default=DEFAULT_MODULES)
    ap.add_argument("--yes", action="store_true", help="don't ask for confirmation")
    a = ap.parse_args()

    key = os.environ.get("LLM_API_KEY", "")
    if not key:
        print("No LLM_API_KEY in the environment or .env. Add it to .env first.")
        return 2
    if not a.small:
        print("Pass --small <model id> (and --large for the model comparisons).")
        return 2

    out = ROOT / "runs" / f"{date.today().isoformat()}-{re.sub(r'[^A-Za-z0-9._-]+', '-', a.small)}"
    paths = [p for m in a.modules for p in sorted((ROOT / "solutions").glob(f"{m}-*.ipynb"))]
    print(f"Real run: {len(paths)} solution notebooks -> {out.relative_to(ROOT)}")
    print(f"  small/default model: {a.small}\n  large model: {a.large or '(none)'}\n  embedding model: {a.embed or '(none)'}")
    print(f"  base URL: {os.environ.get('LLM_BASE_URL', 'https://openrouter.ai/api/v1')}\n  budget: ${a.budget:.2f}")
    if not a.yes and input("This spends real money. Type yes to start: ").strip().lower() != "yes":
        print("Not started.")
        return 1

    import nbformat
    from nbclient import NotebookClient

    for sub in ("notebooks", "figures", "logs"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    env = {
        "LLM_PROVIDER": "openai-compatible", "LLM_MODEL": a.small, "SMALL_MODEL": a.small,
        "LARGE_MODEL": a.large or a.small, "EMBED_MODEL": a.embed or "",
        "FIGURES_DIR": str(out / "figures"), "LLM_DAILY_BUDGET_USD": str(a.budget),
    }
    os.environ.update({k: v for k, v in env.items() if v})

    summary = {"date": date.today().isoformat(), "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "models": {"small": a.small, "large": a.large, "embed": a.embed},
               "base_url": os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1"), "modules": {}}
    for path in paths:
        mod = path.name[:2]
        log = out / "logs" / f"m{mod}.jsonl"
        os.environ["LLM_LOG_PATH"] = str(log)
        nb = nbformat.read(path, as_version=4)
        status, error = "ok", None
        try:
            NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(path.parent)}}).execute()
        except Exception as e:  # keep going; record what broke
            msg = re.sub(r"\x1b\[[0-9;]*m", "", str(e)).strip()  # drop terminal colour codes
            status, error = "failed", (msg.splitlines() or [type(e).__name__])[-1][:300]
        nbformat.write(nb, out / "notebooks" / path.name)
        calls = [json.loads(l) for l in log.read_text().splitlines()] if log.exists() else []
        cost = sum(c["cost_usd"] or 0 for c in calls)
        summary["modules"][f"m{mod}"] = {
            "notebook": path.name, "status": status, "error": error, "calls": len(calls),
            "input_tokens": sum(c["input_tokens"] for c in calls), "output_tokens": sum(c["output_tokens"] for c in calls),
            "cost_usd": round(cost, 6), "calls_without_cost": sum(1 for c in calls if c["cost_usd"] is None),
        }
        print(f"{status:7} m{mod}  {len(calls)} calls  ${cost:.4f}" + (f"  ({error})" if error else ""))

    summary["total_cost_usd"] = round(sum(m["cost_usd"] for m in summary["modules"].values()), 6)
    summary["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    leaked = [p for p in out.rglob("*") if p.is_file() and key in p.read_text(errors="ignore")]
    if leaked:
        shutil.rmtree(out)
        print("ABORTED: your key appeared in the run's output, so the whole folder was deleted. Nothing was saved.")
        return 3
    print(f"\nTotal: ${summary['total_cost_usd']:.4f}. Saved to {out.relative_to(ROOT)}. Commit that folder.")
    return 0 if all(m["status"] == "ok" for m in summary["modules"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
