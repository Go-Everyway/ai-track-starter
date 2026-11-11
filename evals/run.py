"""Run every registered test suite and fail loudly if any is under its threshold.

    python evals/run.py                         # all suites
    python evals/run.py --only m07_outreach_angle
    python evals/run.py --outreach-version v1   # Module 7's suite against the broken v1 prompt

Exits 0 when every suite that ran met its threshold, 1 otherwise. In mock mode, checks that
judge the model's own choices are skipped (the count is printed), so a mock pass means the
plumbing works, not that the model is good.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import lib.outreach  # noqa: E402,F401  registers Module 7's checks
import lib.rag  # noqa: E402,F401  registers Module 8's checks
from evals.registry import SUITES, mock_mode  # noqa: E402
from lib.evals import load_suite, run_suite  # noqa: E402


def load_cases(stem: str, mock_skip: set) -> tuple[list[dict], int]:
    """Load a suite, dropping checks a mock can't meaningfully pass. Returns (cases, n_skipped)."""
    cases, skipped = load_suite(ROOT / "evals" / f"{stem}.jsonl"), 0
    if mock_mode():
        for c in cases:
            keep = [e for e in c["expect"] if not e.get("needs_real_model") and e["check"] not in mock_skip]
            skipped += len(c["expect"]) - len(keep)
            c["expect"] = keep
    return cases, skipped


def run_all(*, only=None, steps: dict | None = None, quiet: bool = False) -> dict:
    """Run the suites. `steps` overrides a suite's step, e.g. {"m07_outreach_angle": broken}.
    Returns {"ok": bool, "pass_rate": overall, "mock": bool, "suites": {stem: {...}}}."""
    report = {"ok": True, "suites": {}, "mock": mock_mode()}
    passed = total = 0
    for stem, cfg in SUITES.items():
        if only and stem not in only:
            continue
        path = ROOT / "evals" / f"{stem}.jsonl"
        if not path.exists():
            report["suites"][stem] = {"skipped": f"{path.name} not written yet"}
            continue
        try:
            step = (steps or {}).get(stem) or cfg["step"]()
        except ImportError as e:
            report["suites"][stem] = {"skipped": f"step not available yet ({e.name})"}
            continue
        cases, skipped = load_cases(stem, cfg["mock_skip"])
        res = run_suite(path, step, cases=cases)
        n_pass = sum(r["passed"] for r in res.rows)
        ok = res.pass_rate >= cfg["threshold"]
        report["ok"] &= ok
        passed, total = passed + n_pass, total + len(res.rows)
        report["suites"][stem] = {"passed": n_pass, "total": len(res.rows), "pass_rate": res.pass_rate,
                                  "threshold": cfg["threshold"], "ok": ok, "checks_skipped": skipped,
                                  "failures": res.failures()}
    report["pass_rate"] = passed / total if total else 0.0
    if not quiet:
        print_report(report)
    return report


def print_report(report: dict) -> None:
    print(f"mode: {'MOCK (plumbing only, not model quality)' if report['mock'] else 'real model'}")
    for stem, s in report["suites"].items():
        if "skipped" in s:
            print(f"  {stem:<20} skipped: {s['skipped']}")
            continue
        note = f", {s['checks_skipped']} checks skipped in mock" if s["checks_skipped"] else ""
        print(f"  {stem:<20} {s['passed']:>3}/{s['total']:<3} {s['pass_rate']:>5.0%}  "
              f"(needs {s['threshold']:.0%}{note})  {'PASS' if s['ok'] else 'FAIL'}")
        for cid, why in s["failures"][:3]:
            print(f"      {cid}: {'; '.join(why)[:110]}")
    print(f"overall {report['pass_rate']:.0%}  {'PASS' if report['ok'] else 'FAIL: do not ship'}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="*", help="suite stems to run")
    ap.add_argument("--outreach-version", help="run Module 7's suite against this prompt version")
    args = ap.parse_args(argv)
    steps = {}
    if args.outreach_version:
        from evals.registry import outreach_step
        steps["m07_outreach_angle"] = outreach_step(args.outreach_version)
    return 0 if run_all(only=args.only, steps=steps)["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
