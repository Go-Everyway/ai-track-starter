# %% [markdown]
# ## 1. What you're hardening
#
# Module 9's pipeline: enrich a lead, score it, draft outreach, check the claims, wait for
# approval. Every model call in it goes through `llm.chat`. This module wraps that one
# function with timeouts, retries, a pinned model, a fallback, redacted logs, a dashboard,
# alerts and a pre-deploy check. All of it lives in `lib/ops.py`.
#
# Everything runs in mock mode here. Failures are injected on purpose, so you can watch
# each safety net catch something without waiting for a real outage.

# %%
import os
import json
import random
from lib import llm, ops, pipeline

OPS_LOG = ROOT / "logs" / "m13_ops.jsonl"         # this notebook's own log, fresh each run
ALERTS = ROOT / "logs" / "m13_alerts.jsonl"
for p in (OPS_LOG, ALERTS):
    p.unlink(missing_ok=True)

REAL_CHAT = llm.chat                               # the unwrapped call, kept for injection
world = pipeline.World.load()
print(f"{len(world.leads):,} leads loaded; mode: {llm.settings()['provider']}")

# %% [markdown]
# ## 2. Which errors deserve a retry
#
# Retry what might work a second later: timeouts, dropped connections, rate limits, server
# errors and overload. Don't retry what will fail the same way every time: a malformed
# request, a bad key, or a 429 that says you've hit a spend cap.

# %%
class APIError(Exception):
    """Stands in for an SDK error that carries an HTTP status."""
    def __init__(self, status, message="error"):
        super().__init__(message)
        self.status_code = status

examples = {
    "400 bad request": APIError(400),
    "401 bad key": APIError(401),
    "429 rate limited": APIError(429, "rate_limit_error"),
    "429 spend cap reached": APIError(429, "monthly spend cap reached"),
    "500 server error": APIError(500),
    "529 overloaded": APIError(529, "overloaded_error"),
    "timeout": TimeoutError("no reply"),
}
for name, exc in examples.items():
    print(f"{name:<24} retry: {ops.is_retryable(exc)}")

# %% [markdown]
# ## 3. Backoff with jitter
#
# Each wait is random, between zero and a ceiling that doubles every attempt. The doubling
# backs off a struggling service; the randomness stops every client retrying in lockstep.

# %%
delays = ops.backoff_delays(5, base=0.5, cap=8, rng=random.Random(1))
print("waits before retries 2-5:", [round(d, 2) for d in delays], "seconds")

# %% [markdown]
# ## 4. Pin the model
#
# A pin list is the set of model ids you've actually tested. The client refuses anything
# else, so a typo in config or a floating alias can't quietly swap the model under you.

# %%
PRIMARY = os.environ.get("LLM_MODEL") or "primary-model"
FALLBACK = os.environ.get("FALLBACK_MODEL") or "fallback-model"
PINS = {PRIMARY, FALLBACK}

try:
    ops.Resilient(primary="some-model-latest", pins=PINS)
except ops.Unpinned as e:
    print("refused:", e)

# %% [markdown]
# ## 5. YOUR TURN: kill the primary and make the fallback take over
#
# `failing_primary` raises an overload error whenever the primary model is asked. That's
# the outage, simulated. As written, the client has no fallback, so every model call in
# the pipeline fails after its retries. Give the client a fallback (it has to be on the pin
# list) and rerun the cell.

# %%
def failing_primary(messages, *, model, **params):
    if model == PRIMARY:
        raise APIError(529, "overloaded_error (simulated)")
    return REAL_CHAT(messages, model=model, **params)

client = ops.Resilient(
    primary=PRIMARY, fallback=None, pins=PINS,   # YOUR TURN: set the fallback
    call=failing_primary, attempts=3, timeout_s=30,
    sleep=lambda s: None,                 # don't actually wait in a notebook
    limiter=ops.RateLimiter(rate=50, burst=10),
    log_path=OPS_LOG,
)

with_profiles = [lid for lid in world.leads.index if world.leads.loc[lid, "company_id"] in world.companies]
results = []
with ops.hardened(client):
    for lid in with_profiles[:5]:
        try:
            results.append(pipeline.run_lead(world, lid))
        except Exception as e:  # with no fallback, the outage reaches the pipeline
            print(f"{lid}: failed: {type(e).__name__}: {e}")

events = ops.load_events(OPS_LOG)
fallbacks = [e for e in events if e["kind"] == "fallback"]
models_used = {e["model"] for e in events if e["kind"] == "call" and e["outcome"] == "ok"}
print(f"{len(results)} leads through the pipeline; statuses: {sorted({r['status'] for r in results})}")
print(f"fallback switches: {len(fallbacks)}; models that answered: {models_used}")

# %% [markdown]
# ## 6. A slow call is a failed call
#
# Without a timeout, one hung request holds the whole job. Here the primary sleeps past
# the limit, times out, and the fallback answers.

# %%
import time

def slow_primary(messages, *, model, **params):
    if model == PRIMARY:
        time.sleep(1.0)
    return REAL_CHAT(messages, model=model, **params)

slow_client = ops.Resilient(primary=PRIMARY, fallback=FALLBACK, pins=PINS, call=slow_primary,
                            attempts=2, timeout_s=0.2, sleep=lambda s: None, log_path=OPS_LOG)
reply = slow_client.chat("One sentence on why forecasts matter.")
print("answered by:", reply.model)

# %% [markdown]
# ## 7. Logs without personal data
#
# Contact names and emails end up in prompts, errors and traces. Redact before anything
# is written. Names come from a list you supply (here, the contacts in the messy export).

# %%
import pandas as pd

contacts = pd.read_csv(DATA / "leads_messy.csv", usecols=["contact_name", "contact_email"]).dropna()
names = tuple(contacts.contact_name.head(200))
sample = f"Draft for {contacts.contact_name.iloc[0]} <{contacts.contact_email.iloc[0]}>, mobile +1 555 010 2000"
print("before:", sample)
print("after: ", ops.redact(sample, names))

# %% [markdown]
# ## 8. Cost caps
#
# Two limits: one per user, so one runaway job can't spend everyone's budget, and one per
# day for the whole system. Check before the call, record after.

# %%
caps = ops.CostCap(per_user_usd=2.00, per_day_usd=10.00)
caps.record("rep-1", 1.95, day="2026-11-11")
caps.record("rep-1", 0.10, day="2026-11-11")
print(caps.allow("rep-1", day="2026-11-11"))
print(caps.allow("rep-2", day="2026-11-11"))

# %% [markdown]
# ## 9. The dashboard: cost and error rate per day
#
# Built from the ops log. Mock calls cost $0, so to see a real-looking week, the cell
# adds seven days of **synthetic** history in front of this run's events. The synthetic
# days are labelled in the code; your own log replaces them once you run with a key.

# %%
rng = random.Random(7)
synthetic = []
for day in range(4, 11):  # Nov 4-10, synthetic
    n, err_p, cost = 200, (0.18 if day == 9 else 0.02), (14.0 if day == 10 else 6.0)
    for i in range(n):
        bad = rng.random() < err_p
        synthetic.append({"ts": f"2026-11-{day:02d}T12:00:00", "kind": "call", "synthetic": True,
                          "outcome": "error" if bad else "ok", "cost_usd": 0 if bad else cost / n})

dashboard = ops.daily_dashboard(synthetic + events, chart=ROOT / "logs" / "m13_dashboard.png")
dashboard.round(3)

# %% [markdown]
# ## 10. YOUR TURN: alerts
#
# Thresholds turn the dashboard into something that wakes you up. Look at the dashboard
# above and pick a daily spend and an error rate that would mean something is wrong.
# `min_calls` keeps a quiet day with two errors out of three calls from paging anyone.

# %%
MAX_COST_PER_DAY = float("inf")   # YOUR TURN: dollars per day
MAX_ERROR_RATE = float("inf")     # YOUR TURN: share of calls, e.g. 0.05

alerts = ops.check_alerts(dashboard, max_cost_per_day=MAX_COST_PER_DAY, max_error_rate=MAX_ERROR_RATE,
                          min_calls=20, path=ALERTS)
for a in alerts:
    print(a)

# %% [markdown]
# ## 11. YOUR TURN: no deploy without a green eval suite
#
# The pre-deploy check runs Module 10's harness. If any suite is under its threshold, the
# deploy function never runs. The first deploy goes through the gate. The second, with the
# deliberately broken v1 prompt from Module 7, doesn't yet: route it through
# `ops.deploy_if_green(..., check=broken_check)` and watch it get blocked.

# %%
deployed, blocked = [], False
ops.deploy_if_green(lambda: deployed.append("v3"))
print("green suite, deployed:", deployed)

broken_check = lambda: ops.predeploy_check(
    [sys.executable, str(ROOT / "evals" / "run.py"), "--only", "m07_outreach_angle", "--outreach-version", "v1"])
try:
    deployed.append("v1")   # YOUR TURN: replace with ops.deploy_if_green(lambda: deployed.append("v1"), check=broken_check)
except ops.DeployBlocked as e:
    blocked = True
    print("blocked:", str(e).splitlines()[0])
print("deploys that happened:", deployed)

# %%
# YOUR TURN: what does mock mode prove here, and what still needs a real key and a real host
# to verify? A few sentences.
needs_a_real_deployment = """
"""

# %% [markdown]
# ## Done-when checks

# %%
from lib.checks import done_when

_ = done_when([
    ("The dashboard shows cost and error rate per day",
     {"cost_usd", "error_rate"} <= set(dashboard.columns) and len(dashboard) >= 2),
    ("An alert fired in a test", len(alerts) >= 1),
    ("Killing the primary model call triggered the fallback",
     len(fallbacks) >= 1 and models_used == {FALLBACK}),
    ("The eval suite runs before every deploy and blocks on failure",
     deployed == ["v3"] and blocked),
    ("You wrote down what still needs a real deployment to verify",
     len(needs_a_real_deployment.split()) >= 20),
])
