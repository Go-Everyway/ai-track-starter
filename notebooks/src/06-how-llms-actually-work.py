# %% [markdown]
# ## 1. Which models, and are we in mock mode?
#
# This lab compares two models. Set `SMALL_MODEL` and `LARGE_MODEL` to model ids from your
# provider's model list (in `.env`, or Colab Secrets). If they're not set, both fall back to
# `LLM_MODEL`. With no key at all, everything runs in mock mode: replies are canned and
# identical every time, so the variation and quality numbers below mean nothing until you
# run them against real models.

# %%
import json
import os
import re
from collections import Counter

from lib import llm

settings = llm.settings()
MOCK = settings["provider"] == "mock"
SMALL_MODEL = os.environ.get("SMALL_MODEL") or settings["model"] or "mock-small"
LARGE_MODEL = os.environ.get("LARGE_MODEL") or settings["model"] or "mock-large"
print("provider:", settings["provider"], "(mock: numbers below show the mechanics only)" if MOCK else "")
print("small model:", SMALL_MODEL)
print("large model:", LARGE_MODEL)

# %% [markdown]
# ## 2. Tokens: what the model actually reads
#
# A model never sees letters or words. It sees tokens, chunks of text from a fixed vocabulary.
# `o200k_base` is the encoding OpenAI uses for its recent models; other providers use their own
# tokenizers and split differently, so treat these counts as a close guide, and your provider's
# usage numbers (in `logs/llm_calls.jsonl`) as the bill.

# %%
import tiktoken

enc = tiktoken.get_encoding("o200k_base")

def show_tokens(text):
    ids = enc.encode(text)
    pieces = [enc.decode([i]) for i in ids]
    return len(ids), pieces

samples = [
    "strawberry",
    "Tallyhawk",
    "Norelley Distributors is a mid-size wholesale distribution company focused on industrial fasteners.",
    "Lagerbestandsprognose",
    "12,500 SKUs across 5 warehouses",
]
token_counts = {}
for s in samples:
    n, pieces = show_tokens(s)
    token_counts[s] = n
    print(f"{n:>3} tokens  {pieces}")

# %% [markdown]
# "strawberry" arrives as three chunks, none of them a letter. That's why a model can stumble
# when you ask how many r's are in it: it has to reason about spelling it never directly saw.
#
# How far off is the "four characters per token" rule of thumb on our own data?

# %%
companies = pd.read_json(DATA / "companies.jsonl", lines=True)
real = sum(len(enc.encode(d)) for d in companies.description)
chars = int(companies.description.str.len().sum())
print(f"400 descriptions: {real:,} tokens (o200k_base), {chars:,} characters")
print(f"rule of thumb (chars / 4): {chars // 4:,} tokens")
print(f"actual characters per token: {chars / real:.2f}")

# %% [markdown]
# ## 3. A cost calculator
#
# Prices change, so you enter them. Find your models' prices per million input and output
# tokens on your provider's pricing page. Thinking or reasoning tokens are billed as output
# tokens on the major APIs, so a reasoning model's output count can be much bigger than the
# text you see.
#
# The prices below are **made-up example values** so the cell runs.
# YOUR TURN: replace them with your two models' real prices from your provider's pricing page.

# %%
PRICES = {  # USD per million tokens: (input, output). EXAMPLE VALUES ONLY.
    SMALL_MODEL: (0.50, 2.00),
    LARGE_MODEL: (5.00, 20.00),
}

def cost_usd(model, input_tokens, output_tokens):
    pin, pout = PRICES[model]
    return (input_tokens * pin + output_tokens * pout) / 1_000_000

# One outreach angle per lead: about 300 tokens in (a company description plus instructions),
# about 80 out. Both numbers are assumptions; measure yours from the log.
for m in [SMALL_MODEL, LARGE_MODEL]:
    print(f"{m}: ${cost_usd(m, 300, 80):.5f} per lead, ${1000 * cost_usd(m, 300, 80):.2f} per 1,000 leads")

# %% [markdown]
# ## 4. Same prompt, 20 times: how much does the answer move?
#
# Agreement rate: the share of runs that gave the most common answer. 100% means the model
# said the same thing every time.
#
# Many current models reject `temperature` entirely (several Anthropic and OpenAI models return
# an error when it's set, especially with reasoning on). So the second configuration *tries*
# temperature 0 and records it if the provider refuses.

# %%
PROMPT = (
    "Here is a company description:\n"
    + companies.description.iloc[0]
    + "\n\nIn three words or fewer, what is the single best reason for an inventory-forecasting "
    "vendor to contact them? Reply with the phrase only."
)

def normalize(text):
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()

# YOUR TURN: return the share of answers equal to the most common answer, after normalize().
# Counter(...).most_common(1) gives you [(answer, count)] for the most common one.
def agreement_rate(answers):
    return None

def run_many(model, n=20, **params):
    replies = []
    for _ in range(n):
        try:
            replies.append(llm.chat(PROMPT, model=model, max_tokens=20, **params))
        except llm.BudgetExceeded:
            raise
        except Exception as e:  # e.g. the provider rejects temperature on this model
            return {"error": f"{type(e).__name__}: {str(e)[:120]}"}
    return {
        "agreement": agreement_rate([r.text for r in replies]),
        "latency_s": sum(r.latency_s for r in replies) / n,
        "cost_per_call": cost_usd(model, sum(r.input_tokens for r in replies) / n, sum(r.output_tokens for r in replies) / n),
        "top_answer": Counter(normalize(r.text) for r in replies).most_common(1)[0][0],
    }

variation = {
    "default settings": run_many(SMALL_MODEL),
    "temperature 0": run_many(SMALL_MODEL, temperature=0),
}
for name, r in variation.items():
    print(name, "->", r)

# %% [markdown]
# ## 5. Small model versus large model, on ten checkable tasks
#
# Each task has one right answer we can check by code: a fact pulled from a company description.

# %%
tasks = []
for _, row in companies.head(10).iterrows():
    tasks.append({
        "prompt": f"{row.description}\n\nHow many employees does {row['name']} have? Reply with the number only.",
        "answer": str(row.employees),
    })

def run_tasks(model):
    rows = []
    for t in tasks:
        r = llm.chat(t["prompt"], model=model, max_tokens=10)
        rows.append({"correct": t["answer"] in r.text.replace(",", ""), "latency_s": r.latency_s,
                     "in": r.input_tokens, "out": r.output_tokens, "logged_cost": r.cost_usd})
    df = pd.DataFrame(rows)
    return {
        "quality": df.correct.mean(),
        "latency_s": df.latency_s.mean(),
        "cost_per_call": cost_usd(model, df["in"].mean(), df["out"].mean()) if model in PRICES else None,
    }

comparison = {"small": run_tasks(SMALL_MODEL), "large": run_tasks(LARGE_MODEL)}
for k, v in comparison.items():
    print(k, v)

# %% [markdown]
# ## 6. Your results table

# %%
table = pd.DataFrame([
    {"configuration": f"small, {name}", "model": SMALL_MODEL,
     "agreement": r.get("agreement"), "quality": None, "latency_s": r.get("latency_s"),
     "cost_per_call": r.get("cost_per_call"), "note": r.get("error", "")}
    for name, r in variation.items()
] + [
    {"configuration": f"{size}, default", "model": m, "agreement": None, "quality": comparison[size]["quality"],
     "latency_s": comparison[size]["latency_s"], "cost_per_call": comparison[size]["cost_per_call"], "note": ""}
    for size, m in [("small", SMALL_MODEL), ("large", LARGE_MODEL)]
])
if MOCK:
    table["note"] = table["note"].where(table["note"] != "", "mock: not real model behavior")
table

# %%
# YOUR TURN: pick the default model for the lead engine and say why, using the table:
# cost per 1,000 leads, quality on the ten tasks, latency, and how much the answers moved.
default_model = ""
why_this_model = """
"""

# %% [markdown]
# ## Done-when checks

# %%
from lib.checks import done_when

_ = done_when([
    ("Token counts for five inputs", len(token_counts) >= 5),
    ("A cost per 1,000 leads for both models from the calculator", all(m in PRICES for m in [SMALL_MODEL, LARGE_MODEL])),
    ("20 runs at two settings, with an agreement rate (or the provider's refusal) for each",
     len(variation) >= 2 and all(r.get("agreement") is not None or "error" in r for r in variation.values())),
    ("Small and large models compared on quality, latency and cost",
     {"quality", "latency_s", "cost_per_call"} <= set(table.columns) and len(table) >= 4),
    ("A default model for the project, with reasons", bool(default_model) and len(why_this_model.split()) >= 25),
])
