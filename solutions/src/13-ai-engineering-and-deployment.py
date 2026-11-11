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
# ## OPERATOR: change one value
#
# The error rate that should page someone: 0.10 means "alert when more than 10% of a day's
# calls fail". Step 10 checks every day on the dashboard against it. Try 0.05 (twitchy) or
# 0.5 (relaxed), run everything again, and see which days raise an alert.
# Sensible range: 0.02 to 0.9.

# %%
OPERATOR_VALUE = 0.10  # change this, then Runtime → Run all
OPERATOR_DEFAULT = 0.10
assert 0.0 < OPERATOR_VALUE < 1.0, "use an error rate between 0 and 1, e.g. 0.10 for 10%"

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
print("waits before attempts 2 to 5 (retries 1 to 4):", [round(d, 2) for d in delays], "seconds")
print("ceilings:", [min(8, 0.5 * 2 ** n) for n in range(4)])

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
# ## 5. Kill the primary and watch the fallback take over
#
# `failing_primary` raises an overload error whenever the primary model is asked. That's
# the outage, simulated. The client retries it with backoff, gives up, logs the switch and
# finishes the job on the fallback.

# %%
def failing_primary(messages, *, model, **params):
    if model == PRIMARY:
        raise APIError(529, "overloaded_error (simulated)")
    return REAL_CHAT(messages, model=model, **params)

client = ops.Resilient(
    primary=PRIMARY, fallback=FALLBACK, pins=PINS,
    call=failing_primary, attempts=3, timeout_s=30,
    sleep=lambda s: None,                 # don't actually wait in a notebook
    limiter=ops.RateLimiter(rate=50, burst=10),
    log_path=OPS_LOG,
)

with_profiles = [lid for lid in world.leads.index if world.leads.loc[lid, "company_id"] in world.companies]
with ops.hardened(client):
    results = [pipeline.run_lead(world, lid) for lid in with_profiles[:5]]

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
# ## 10. Alerts
#
# Thresholds turn the dashboard into something that wakes you up. `min_calls` keeps a
# quiet day with two errors out of three calls from paging anyone.

# %%
MAX_COST_PER_DAY = 10.00          # dollars
MAX_ERROR_RATE = OPERATOR_VALUE   # share of calls (the operator value; 0.10 = 10%)

alerts = ops.check_alerts(dashboard, max_cost_per_day=MAX_COST_PER_DAY, max_error_rate=MAX_ERROR_RATE,
                          min_calls=20, path=ALERTS)
print(f"{len(alerts)} alert(s) at an error-rate threshold of {MAX_ERROR_RATE:.0%} and a cost limit of ${MAX_COST_PER_DAY:.2f}/day:")
for a in alerts:
    print(a)

# %% [markdown]
# ## 11. No deploy without a green eval suite
#
# The pre-deploy check runs Module 10's harness. If any suite is under its threshold, the
# deploy function never runs. First with the working prompt, then with the deliberately
# broken v1 prompt from Module 7.

# %%
deployed, blocked = [], False
ops.deploy_if_green(lambda: deployed.append("v3"))
print("green suite, deployed:", deployed)

broken_check = lambda: ops.predeploy_check(
    [sys.executable, str(ROOT / "evals" / "run.py"), "--only", "m07_outreach_angle", "--outreach-version", "v1"])
try:
    ops.deploy_if_green(lambda: deployed.append("v1"), check=broken_check)
except ops.DeployBlocked as e:
    blocked = True
    print("blocked:", str(e).splitlines()[0])
print("deploys that happened:", deployed)

# %%
needs_a_real_deployment = """
Mock mode proves the logic, not the operations. With a real key and host I still need to check:
that my provider's SDK raises errors carrying status codes the way is_retryable expects; that
the timeout fits real latency for my prompts; that the fallback model passes the Module 10 suites
on its own before I trust it; that alerts actually reach a person (email or chat), not just a
file; and that redaction catches the personal data in my real traces.
"""

# %% [markdown]
# ## Done-when checks

# %%
from lib.checks import done_when

if OPERATOR_VALUE == OPERATOR_DEFAULT:
    from lib.figures import record
    calls_today = [e for e in events if e["kind"] == "call"]
    by_day = dashboard.reset_index().to_dict("records")
    record("m13",
           retryable={name: ops.is_retryable(exc) for name, exc in examples.items()},
           backoff_waits_seed1=[round(d, 2) for d in delays],
           leads_through_pipeline=len(results), fallback_switches=len(fallbacks), alert_min_calls=20, backoff_ceilings=[min(8, 0.5 * 2 ** n) for n in range(4)],
           models_that_answered=sorted(models_used),
           today_calls=len(calls_today), today_errors=sum(e["outcome"] == "error" for e in calls_today),
           timeout_s=0.2, slow_primary_s=1.0, slow_call_answered_by=reply.model,
           cost_cap_per_user=2.00, cost_cap_per_day=10.00,
           cap_rep1_spent=2.05, cap_rep1_allowed=caps.allow("rep-1", day="2026-11-11")[0],
           cap_rep2_allowed=caps.allow("rep-2", day="2026-11-11")[0],
           # The synthetic week only: today's row depends on the date you run this.
           synthetic_dashboard={r["day"]: {k: r[k] for k in ("calls", "errors", "error_rate", "cost_usd")}
                                for r in by_day if r["day"] in {e["ts"][:10] for e in synthetic}},
           alert_threshold_error_rate=MAX_ERROR_RATE, alert_threshold_cost=MAX_COST_PER_DAY,
           alerts=len(alerts), deployed=deployed, broken_deploy_blocked=blocked,
           # The two bad synthetic days, under keys a page can reference.
           synthetic_error_day=max(({k: r[k] for k in ("calls", "errors", "error_rate")} for r in by_day
                                    if r["day"] in {e["ts"][:10] for e in synthetic}), key=lambda r: r["error_rate"]),
           synthetic_spend_day_cost=max(r["cost_usd"] for r in by_day if r["day"] in {e["ts"][:10] for e in synthetic}),
           today_error_rate=sum(e["outcome"] == "error" for e in calls_today) / len(calls_today) if calls_today else None)

checks = [
    ("The dashboard shows cost and error rate per day",
     {"cost_usd", "error_rate"} <= set(dashboard.columns) and len(dashboard) >= 2),
    ("An alert fired in a test", len(alerts) >= 1),
    ("Killing the primary model call triggered the fallback",
     len(fallbacks) >= 1 and models_used == {FALLBACK}),
    ("The eval suite runs before every deploy and blocks on failure",
     deployed == ["v3"] and blocked),
    ("You wrote down what still needs a real deployment to verify",
     len(needs_a_real_deployment.split()) >= 20),
]
if OPERATOR_VALUE == OPERATOR_DEFAULT:
    assert done_when(checks)
else:
    print(f"OPERATOR_VALUE is {OPERATOR_VALUE}, not the default, so these are for reading, not grading:")
    _ = done_when(checks)
