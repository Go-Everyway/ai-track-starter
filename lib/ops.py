"""Production hardening for model calls (Module 13).

Everything here wraps `lib.llm` from the outside, so the rest of the kit doesn't change:

    from lib import ops
    client = ops.Resilient(primary="your-pinned-model", fallback="your-backup-model",
                           pins={"your-pinned-model", "your-backup-model"})
    with ops.hardened(client):          # every llm.chat in the kit now goes through it
        result = pipeline.run_lead(world, "L00004")

What it adds:
- timeouts, and retries with exponential backoff and jitter, for retryable errors only
- a token-bucket rate limiter
- model pinning: only ids on your pin list are allowed
- a fallback model when the primary fails, with the failure logged
- PII redaction for everything it writes to logs
- a per-day dashboard (cost, calls, error rate) and alert thresholds
- a pre-deploy check that runs the Module 10 suites and refuses to deploy on a failure

For tests, inject the call function (`call=`) and the clock (`sleep=`), so you can make the
primary fail on purpose without touching `lib/llm.py`.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import json
import random
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from lib import llm

ROOT = Path(__file__).resolve().parent.parent
OPS_LOG = ROOT / "logs" / "ops_events.jsonl"
ALERTS_LOG = ROOT / "logs" / "alerts.jsonl"


# ---------------------------------------------------------------- errors and retries

class Unpinned(ValueError):
    """The model id isn't on the pin list."""


class DeployBlocked(RuntimeError):
    """The pre-deploy check failed."""


# HTTP statuses worth retrying: timeouts, rate limits, server errors, overload. (A 409 conflict
# needs the conflict fixed first, so it isn't on the list.)
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504, 529}
RETRYABLE_TYPES = (TimeoutError, ConnectionError)


def is_retryable(exc: BaseException) -> bool:
    """Retry timeouts, dropped connections, 429s and 5xx. Don't retry bad requests, auth
    failures or a 429 with no retry-after that says a spend cap was reached: those fail
    again however long you wait."""
    if isinstance(exc, RETRYABLE_TYPES):
        return True
    status = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
    if status is None:
        return False
    if status == 429 and "spend" in str(exc).lower():
        return False
    return status in RETRYABLE_STATUS


def backoff_delays(attempts: int, *, base: float = 0.5, cap: float = 8.0, rng: random.Random | None = None) -> list[float]:
    """'Full jitter': each wait is random between 0 and base * 2**n, capped. Random spreads
    retries from many clients so they don't all hit the API at the same moment."""
    rng = rng or random.Random()
    return [rng.uniform(0, min(cap, base * 2 ** n)) for n in range(attempts - 1)]


def with_timeout(fn: Callable, seconds: float, *args, **kwargs):
    """Run fn in a worker thread; raise TimeoutError if it takes longer than `seconds`.
    (The thread can't be killed, so the work may finish later. Your SDK's own timeout
    setting is better when it has one; this guards everything else.)"""
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        return pool.submit(fn, *args, **kwargs).result(timeout=seconds)
    except concurrent.futures.TimeoutError:
        raise TimeoutError(f"no reply within {seconds}s") from None
    finally:
        pool.shutdown(wait=False)


class RateLimiter:
    """Token bucket: `rate` calls per second on average, bursts up to `burst`."""

    def __init__(self, rate: float, burst: int = 1, *, clock: Callable = time.monotonic, sleep: Callable = time.sleep):
        self.rate, self.burst, self.clock, self.sleep = rate, burst, clock, sleep
        self.tokens, self.last = float(burst), clock()
        self.lock = threading.Lock()

    def acquire(self) -> float:
        """Wait until a call is allowed. Returns how long it waited."""
        with self.lock:
            now = self.clock()
            self.tokens = min(self.burst, self.tokens + (now - self.last) * self.rate)
            self.last = now
            wait = 0.0 if self.tokens >= 1 else (1 - self.tokens) / self.rate
            if wait:
                self.sleep(wait)
                self.tokens, self.last = 1.0, self.clock()
            self.tokens -= 1
            return wait


# ---------------------------------------------------------------- redaction

EMAIL = re.compile(r"[\w.+-]+@[\w-]+(\.[\w-]+)+")
PHONE = re.compile(r"(?<!\w)(\+?\d[\d\s().-]{7,}\d)(?!\w)")


def redact(text: str, names: Iterable[str] = ()) -> str:
    """Replace emails, phone numbers and any of the given names before text is logged."""
    text = EMAIL.sub("[email]", text)
    text = PHONE.sub("[phone]", text)
    for n in sorted({n for n in names if n}, key=len, reverse=True):
        text = re.sub(re.escape(n), "[name]", text, flags=re.IGNORECASE)
    return text


def redact_record(rec, names: Iterable[str] = ()):
    """Redact every string inside a dict/list, recursively."""
    names = tuple(names)
    if isinstance(rec, str):
        return redact(rec, names)
    if isinstance(rec, dict):
        return {k: redact_record(v, names) for k, v in rec.items()}
    if isinstance(rec, list):
        return [redact_record(v, names) for v in rec]
    return rec


# ---------------------------------------------------------------- the resilient client

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Resilient:
    """A drop-in replacement for `llm.chat` with pinning, timeouts, retries and a fallback.

    `pins` is the allowlist of model ids you've tested. Anything else is refused, which is
    what stops a config typo or a floating alias from quietly changing the model in
    production. Check your provider's docs for which ids are fixed snapshots."""

    primary: str
    fallback: str | None = None
    pins: set = field(default_factory=set)
    call: Callable = None              # defaults to the real llm.chat, captured at creation
    timeout_s: float = 60.0
    attempts: int = 3
    limiter: RateLimiter | None = None
    sleep: Callable = time.sleep
    redact_names: tuple = ()
    log_path: Path = OPS_LOG
    rng: random.Random = field(default_factory=lambda: random.Random(0))

    def __post_init__(self):
        self.call = self.call or llm.chat
        for m in filter(None, [self.primary, self.fallback]):
            self.check_pinned(m)

    def check_pinned(self, model: str) -> None:
        if model not in self.pins:
            raise Unpinned(f"{model!r} is not on the pin list {sorted(self.pins)}")

    def log(self, **event) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(redact_record({"ts": _now(), **event}, self.redact_names)) + "\n")

    def _try(self, model: str, messages, params) -> llm.Reply:
        delays = backoff_delays(self.attempts, rng=self.rng)
        for attempt in range(self.attempts):
            if self.limiter:
                self.limiter.acquire()
            t0 = time.perf_counter()
            try:
                reply = with_timeout(self.call, self.timeout_s, messages, model=model, **params)
                self.log(kind="call", model=model, outcome="ok", attempt=attempt + 1,
                         cost_usd=reply.cost_usd, latency_s=round(time.perf_counter() - t0, 3))
                return reply
            except Exception as e:
                retry = is_retryable(e) and attempt < self.attempts - 1
                self.log(kind="call", model=model, outcome="error", attempt=attempt + 1,
                         error=f"{type(e).__name__}: {e}", will_retry=retry)
                if not retry:
                    raise
                self.sleep(delays[attempt])
        raise RuntimeError("unreachable")

    def chat(self, messages, *, model: str | None = None, **params) -> llm.Reply:
        """Same signature as llm.chat. A `model` argument from older code is checked against
        the pins but otherwise ignored: the pinned primary is what runs."""
        if model:
            self.check_pinned(model)
        try:
            return self._try(self.primary, messages, params)
        except Exception as primary_error:
            if not self.fallback:
                raise
            self.log(kind="fallback", model=self.fallback, outcome="switched",
                     reason=f"{type(primary_error).__name__}: {primary_error}")
            return self._try(self.fallback, messages, params)


@contextlib.contextmanager
def hardened(client: Resilient):
    """Route every `llm.chat` call in the kit (outreach, rag, pipeline) through `client`
    for the duration of the block, then put the original back."""
    original = llm.chat
    llm.chat = client.chat
    try:
        yield client
    finally:
        llm.chat = original


# ---------------------------------------------------------------- cost caps

class CostCap:
    """Per-user and per-day spend limits, checked before a call and recorded after."""

    def __init__(self, per_user_usd: float, per_day_usd: float):
        self.per_user, self.per_day = per_user_usd, per_day_usd
        self.spent: dict[tuple[str, str], float] = {}

    def allow(self, user: str, day: str | None = None) -> tuple[bool, str]:
        day = day or _now()[:10]
        user_total = self.spent.get((day, user), 0.0)
        day_total = sum(v for (d, _), v in self.spent.items() if d == day)
        if user_total >= self.per_user:
            return False, f"{user} has spent ${user_total:.2f} today (cap ${self.per_user:.2f})"
        if day_total >= self.per_day:
            return False, f"${day_total:.2f} spent today (cap ${self.per_day:.2f})"
        return True, "ok"

    def record(self, user: str, cost: float | None, day: str | None = None) -> None:
        key = (day or _now()[:10], user)
        self.spent[key] = self.spent.get(key, 0.0) + (cost or 0.0)


# ---------------------------------------------------------------- dashboard and alerts

def load_events(path: Path = OPS_LOG) -> list[dict]:
    if not Path(path).exists():
        return []
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def daily_dashboard(events: list[dict] | None = None, *, path: Path = OPS_LOG, chart: Path | None = None):
    """Cost, calls, errors and error rate per day from the ops log (or events you pass in).
    Saves a two-panel chart if `chart` is a path."""
    import pandas as pd

    rows = [e for e in (events if events is not None else load_events(path)) if e.get("kind") == "call"]
    if not rows:
        return pd.DataFrame(columns=["calls", "errors", "error_rate", "cost_usd"])
    df = pd.DataFrame(rows)
    df["day"] = df["ts"].str[:10]
    df["is_error"] = df["outcome"].eq("error")
    df["cost_usd"] = pd.to_numeric(df.get("cost_usd"), errors="coerce").fillna(0.0)
    out = df.groupby("day").agg(calls=("outcome", "size"), errors=("is_error", "sum"), cost_usd=("cost_usd", "sum"))
    out["error_rate"] = out.errors / out.calls
    out = out[["calls", "errors", "error_rate", "cost_usd"]]
    if chart:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (a, b) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
        a.bar(out.index, out.cost_usd, color="#8b5cf6")
        a.set_ylabel("cost (USD)")
        b.plot(out.index, out.error_rate, marker="o", color="#0e7490")
        b.set_ylabel("error rate")
        b.set_ylim(0, max(0.05, out.error_rate.max() * 1.2))
        fig.autofmt_xdate()
        fig.tight_layout()
        Path(chart).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(chart, dpi=110)
        plt.close(fig)
    return out


def check_alerts(dashboard, *, max_cost_per_day: float, max_error_rate: float,
                 min_calls: int = 20, path: Path = ALERTS_LOG) -> list[dict]:
    """Compare each day with the thresholds. Writes and returns one alert per breach.
    `min_calls` stops two errors out of three calls from paging anyone."""
    alerts = []
    for day, r in dashboard.iterrows():
        if r.cost_usd > max_cost_per_day:
            alerts.append({"day": day, "alert": "spend", "value": round(float(r.cost_usd), 4), "threshold": max_cost_per_day})
        if r.calls >= min_calls and r.error_rate > max_error_rate:
            alerts.append({"day": day, "alert": "error_rate", "value": round(float(r.error_rate), 4), "threshold": max_error_rate})
    if alerts:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("a", encoding="utf-8") as f:
            for a in alerts:
                f.write(json.dumps({"ts": _now(), **a}) + "\n")
    return alerts


# ---------------------------------------------------------------- pre-deploy

def predeploy_check(cmd: list[str] | None = None, *, runner: Callable = subprocess.run) -> tuple[bool, str]:
    """Run the Module 10 harness. Returns (passed, output). A non-zero exit means don't ship."""
    cmd = cmd or [sys.executable, str(ROOT / "evals" / "run.py")]
    proc = runner(cmd, capture_output=True, text=True, cwd=ROOT)
    return proc.returncode == 0, (proc.stdout or "") + (proc.stderr or "")


def deploy_if_green(deploy: Callable[[], object], *, check: Callable[[], tuple[bool, str]] = predeploy_check):
    """Call `deploy()` only if the pre-deploy check passes; otherwise raise DeployBlocked."""
    ok, output = check()
    if not ok:
        raise DeployBlocked("eval suite failed, not deploying:\n" + output[-1500:])
    return deploy()
