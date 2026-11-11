# %% [markdown]
# ## 1. Where your settings come from
#
# `lib/llm.py` reads its settings from environment variables: a `.env` file on your laptop,
# Colab's Secrets panel in Colab. This cell shows what it found. It never prints the key itself.

# %%
import json
import os
from lib import llm

settings = llm.settings()
print("provider:     ", settings["provider"], "(no key set, so calls are free and canned)" if settings["provider"] == "mock" else "")
print("key:          ", "set" if settings["api_key_set"] else "not set")
print("model:        ", settings["model"] or "(not set)")
print("daily budget: ", f"${settings['budget']:.2f}", "(set by you)" if "LLM_DAILY_BUDGET_USD" in os.environ else "(the kit's default)")

# %% [markdown]
# ## 2. Make one call
#
# One short question, through the same client every later module uses.

# %%
reply = llm.chat("In one sentence, what is inventory forecasting software for?", max_tokens=60)
print(reply.text)
print(f"\n{reply.input_tokens} tokens in, {reply.output_tokens} out, cost {reply.cost_usd}, {reply.latency_s:.2f}s")

# %% [markdown]
# ## 3. Read the log
#
# Every call lands in `logs/llm_calls.jsonl`, one JSON line each. This is how you'll know
# what the course costs you, instead of guessing.

# %%
log_lines = llm.LOG_PATH.read_text(encoding="utf-8").splitlines()
last_call = json.loads(log_lines[-1])
print(json.dumps(last_call, indent=2))
print(f"\n{len(log_lines)} calls logged so far, ${llm.spend_today():.4f} spent today (UTC)")

# %% [markdown]
# ## 4. Set your spend caps
#
# Two caps, and you want both.
#
# - **At the provider.** On OpenRouter, give the key a credit limit when you create it. That
#   cap holds even when your code is wrong.
# - **In the kit.** `LLM_DAILY_BUDGET_USD` makes `lib/llm.py` refuse calls once today's
#   logged spend reaches it. Put it in `.env` (or Colab Secrets) so it applies every time.

# %%
# Normally this line lives in .env as LLM_DAILY_BUDGET_USD=1.00. It's set here so the
# solution runs anywhere.
os.environ.setdefault("LLM_DAILY_BUDGET_USD", "1.00")
budget_set = "LLM_DAILY_BUDGET_USD" in os.environ

# Set this to True once your key has a credit limit at the provider. Mock mode needs no key.
provider_cap_set = False
in_mock_mode = llm.settings()["provider"] == "mock"
print(f"kit budget: ${float(os.environ['LLM_DAILY_BUDGET_USD']):.2f} a day")
print("provider cap:", "set" if provider_cap_set else ("not needed in mock mode" if in_mock_mode else "NOT SET"))

# %% [markdown]
# ## 5. Is your key anywhere it shouldn't be?
#
# Two checks. Git should ignore `.env`, and your key should appear in no file git tracks.
# The second one only runs when a key is set.

# %%
import subprocess

def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)

if git("rev-parse", "--is-inside-work-tree").returncode != 0:
    env_ignored, key_in_repo = True, False
    print("not a git checkout, so nothing here can be committed by accident")
else:
    env_ignored = git("check-ignore", "-q", ".env").returncode == 0
    key = os.environ.get("LLM_API_KEY", "")
    # git grep exits 0 when it finds a match, 1 when it finds none
    key_in_repo = bool(key) and git("grep", "-q", "-F", "--", key).returncode == 0
    print(".env ignored by git:", env_ignored)
    print("key found in a tracked file:", key_in_repo if key else "no key set, nothing to find")
key_safe = env_ignored and not key_in_repo

# %% [markdown]
# ## 6. Run the function on your machine first
#
# `deploy/app.py` is the function you'll ship. It asks one fixed question and returns JSON.
# Run it here before you deploy it, so the first time you see it fail isn't on the internet.

# %%
import importlib.util

spec = importlib.util.spec_from_file_location("deploy_app", ROOT / "deploy" / "app.py")
deploy_app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deploy_app)

local_result = deploy_app.answer()
print(json.dumps(local_result, indent=2))

# %% [markdown]
# ## 7. Call your deployed URL
#
# Follow the deploy steps on the module page, then paste the production URL below
# (the short one ending in `.vercel.app`) and run the cell.

# %%
import urllib.request

DEPLOYED_URL = ""  # e.g. "https://your-project.vercel.app/"

deployed_ok = False
if DEPLOYED_URL:
    with urllib.request.urlopen(DEPLOYED_URL, timeout=30) as resp:
        deployed = json.load(resp)
    print(json.dumps(deployed, indent=2))
    deployed_ok = deployed.get("ok") is True and "reply" in deployed
else:
    print("No URL yet. The solution can't deploy for you, so this check is skipped here.")

# %% [markdown]
# ## Done-when checks

# %%
from lib.checks import done_when

assert done_when([
    ("One call made, with its tokens in the log", last_call["kind"] == "chat" and last_call["input_tokens"] > 0),
    ("Daily budget set (LLM_DAILY_BUDGET_USD)", budget_set),
    ("Provider credit limit on your key (or you're in mock mode)", provider_cap_set or in_mock_mode),
    ("Your key is in no tracked file, and git ignores .env", key_safe),
    ("The function returns JSON on your machine", local_result.get("ok") is True and "reply" in local_result),
    ("Your deployed URL answers with JSON" + ("" if DEPLOYED_URL else " (skipped: no URL in the solution)"),
     deployed_ok or not DEPLOYED_URL),
])
