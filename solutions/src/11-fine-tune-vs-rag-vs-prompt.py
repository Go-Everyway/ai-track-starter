# %% [markdown]
# ## 1. The question
#
# Module 7's outreach-angle step works. Before anyone suggests fine-tuning it, compare the
# cheaper rungs of the ladder on the same test suite, with real token counts and your own
# prices. This notebook compares three approaches:
#
# - **A. Prompt only:** Module 7's system prompt with its worked example removed.
# - **B. Prompt + one example:** Module 7's v3, as shipped.
# - **C. Prompt + example + retrieved facts:** B, plus the three Tallyhawk passages Module 8's
#   keyword search finds for each company.
#
# In mock mode (no key), costs come from real token counts but quality is plumbing only.

# %%
import json
import time

import tiktoken

from lib import llm, outreach, rag
from evals.run import run_all
from evals.registry import mock_mode

enc = tiktoken.get_encoding("o200k_base")
tokens = lambda text: len(enc.encode(text))
print("mode:", "MOCK (costs from real token counts; quality is plumbing only)" if mock_mode() else "real model")

# %% [markdown]
# ## OPERATOR: change one value
#
# How many leads a month this step runs on. It changes the monthly cost of each approach and
# how long fine-tuning would take to pay for itself. Try 500 for a small team or 200000 for a
# big outbound operation, then run everything again and compare the table and the break-even.
# Sensible range: 100 to 1,000,000.

# %%
OPERATOR_VALUE = 2_000  # change this, then Runtime → Run all
OPERATOR_DEFAULT = 2_000
assert 1 <= OPERATOR_VALUE <= 10_000_000, "use a monthly lead count between 1 and 10 million"

# %% [markdown]
# ## 2. Build the three system prompts

# %%
SYSTEM_B = outreach.SYSTEM_V2
SYSTEM_A = SYSTEM_B.split("\n\nExample\n")[0]  # same rules, no worked example

kb = rag.build_index(rag.load_docs(DATA / "docs"), embed=False, size=40, overlap=10, prefix_title=True)

def system_c(company):
    hits = rag.search(kb, company["description"], k=3, mode="keyword")
    facts = "\n".join(f"- {h.chunk.text}" for h in hits)
    return SYSTEM_B + "\n\nFacts about Tallyhawk you may use in the opener (and nothing else about Tallyhawk):\n" + facts

APPROACHES = {
    "A: prompt only": lambda company: SYSTEM_A,
    "B: prompt + 1 example": lambda company: SYSTEM_B,
    "C: + retrieved facts": system_c,
}
for name, build in APPROACHES.items():
    print(f"{name:<24} system prompt for one company: {tokens(build({'description': 'wholesale plumbing parts distributor'})):>5} tokens")

# %% [markdown]
# ## 3. Run every approach on the same suite
#
# `run_all` takes a replacement step for the suite. Each approach is the same
# `outreach_angle` call with a different system prompt, so the only thing that changes is
# what goes in the context.

# %%
suite_cases = [json.loads(l) for l in (ROOT / "evals" / "m07_outreach_angle.jsonl").read_text().splitlines() if l.strip()]
# The JSON schema travels with every request (response_format), and it's billed as input too.
SCHEMA_TOKENS = tokens(json.dumps(outreach.SCHEMA))

def step_for(build):
    def step(inp):
        return outreach.outreach_angle(inp["company"], version="v3", system=build(inp["company"]))
    return step

def logged_calls():
    """Call-log lines so far (the log is per run: LLM_LOG_PATH)."""
    return llm.LOG_PATH.read_text().splitlines() if llm.LOG_PATH.exists() else []

runs = {}
for name, build in APPROACHES.items():
    t0 = time.perf_counter()
    before = len(logged_calls())
    # run_all never raises on a bad reply: a reply that isn't valid JSON, or a refusal, is a
    # failed case in the suite, so the comparison always completes.
    report = run_all(only=["m07_outreach_angle"], steps={"m07_outreach_angle": step_for(build)}, quiet=True)
    s = report["suites"]["m07_outreach_angle"]
    calls = [json.loads(l) for l in logged_calls()[before:]]
    real_cost = sum(c.get("cost_usd") or 0 for c in calls if c.get("kind") == "chat")
    in_tokens = [tokens(build(c["input"]["company"])) + SCHEMA_TOKENS
                 + tokens(outreach.fence(c["input"]["company"]["description"]) + "\nReply with the JSON object only.")
                 for c in suite_cases]
    runs[name] = {
        "pass_rate": s["pass_rate"],
        "input_tokens_per_lead": sum(in_tokens) / len(in_tokens),
        "prefix_tokens": tokens(build(suite_cases[0]["input"]["company"])) + SCHEMA_TOKENS if name != "C: + retrieved facts" else None,
        "seconds_per_lead": (time.perf_counter() - t0) / s["total"],
        "passed": s["passed"], "total": s["total"],
        "calls": len(calls),
        # What the provider actually charged for this approach's 20 cases ($0 in mock mode).
        "logged_cost_per_lead": real_cost / s["total"],
        "logged_output_tokens_per_lead": sum(c.get("output_tokens", 0) for c in calls) / s["total"],
    }
    print(f"{name:<24} pass {s['passed']}/{s['total']}  input tokens/lead {runs[name]['input_tokens_per_lead']:.0f}")
print(f"(each includes {SCHEMA_TOKENS} tokens of JSON schema)")

# %% [markdown]
# ## 4. Output tokens
#
# Output length is about the same for every approach: one JSON object. Estimate it from the
# worked example's reply in the prompt, which is the length you asked for. With a key, read
# the real number from `logs/llm_calls.jsonl` instead.

# %%
example_reply = SYSTEM_B.split("Reply:\n", 1)[1]
OUTPUT_TOKENS = tokens(example_reply)
print("estimated output tokens per lead:", OUTPUT_TOKENS)

# %% [markdown]
# ## 5. Your prices
#
# Prices change and differ by model, so they're inputs, not facts. Copy them from your
# provider's pricing page. The values below are example inputs for the exercise.

# %%
PRICES = {
    "input_per_million": 1.00,    # example input: USD per million input tokens
    "output_per_million": 4.00,   # example input: USD per million output tokens
}
LEADS_PER_MONTH = OPERATOR_VALUE  # example input: how many leads this step runs on (operator value)

def cost_per_lead(input_tokens, output_tokens=OUTPUT_TOKENS, prices=PRICES):
    return (input_tokens * prices["input_per_million"] + output_tokens * prices["output_per_million"]) / 1_000_000

table = pd.DataFrame({
    name: {
        "suite pass rate" + (" (MOCK)" if mock_mode() else ""): r["pass_rate"],
        "input tokens / lead": round(r["input_tokens_per_lead"]),
        "cost / lead (USD)": cost_per_lead(r["input_tokens_per_lead"]),
        "cost / month (USD)": cost_per_lead(r["input_tokens_per_lead"]) * LEADS_PER_MONTH,
        "seconds / lead" + (" (MOCK)" if mock_mode() else ""): r["seconds_per_lead"],
        **({} if mock_mode() else {"logged cost / lead (USD, real)": r["logged_cost_per_lead"]}),
    } for name, r in runs.items()
}).T
table

# %% [markdown]
# ## 6. Does caching change anything here?
#
# Caching discounts a repeated prefix, but only above a minimum length. Module 7 found
# minimums of 1,024 tokens on OpenAI's current models and 512 to 4,096 by model on
# Anthropic's (checked November 11, 2026). Compare the minimum with your measured prefix.

# %%
CACHE_MIN_TOKENS = 1024          # example input: your model's minimum cacheable prefix
prefix_b = runs["B: prompt + 1 example"]["prefix_tokens"]
print(f"B's fixed prefix (system prompt + schema) is {prefix_b} tokens; the minimum is {CACHE_MIN_TOKENS}.")
print("caching applies" if prefix_b >= CACHE_MIN_TOKENS else "too short to cache: caching saves nothing on this step")

# %% [markdown]
# ## 7. Fine-tuning break-even
#
# The honest case for fine-tuning B: a tuned model learns the format and style, so you can
# drop the worked example and send A's shorter prompt. The saving per lead is the example's
# tokens. Against that: a one-off training cost, and a higher price per token for the tuned
# model on most platforms. All three are inputs.

# %%
FT = {
    "training_cost": 50.00,         # example input: one-off USD for a training run
    "price_multiplier": 1.5,        # example input: tuned model's price vs the base model
}
plain_b = cost_per_lead(runs["B: prompt + 1 example"]["input_tokens_per_lead"])
tuned_prices = {k: v * FT["price_multiplier"] for k, v in PRICES.items()}
tuned_a = cost_per_lead(runs["A: prompt only"]["input_tokens_per_lead"], prices=tuned_prices)
saving = plain_b - tuned_a
print(f"B on the base model: ${plain_b:.6f} per lead")
print(f"A on a tuned model:  ${tuned_a:.6f} per lead")
if saving > 0:
    print(f"break-even after {FT['training_cost'] / saving:,.0f} leads")
else:
    print(f"never breaks even: the tuned model costs ${-saving:.6f} MORE per lead, before the training cost")

# Sensitivity: the best case, a tuned model at the base model's price.
best_saving = plain_b - cost_per_lead(runs["A: prompt only"]["input_tokens_per_lead"])
if best_saving > 0:
    print(f"best case (same price): saves ${best_saving:.6f} per lead, "
          f"break-even after {FT['training_cost'] / best_saving:,.0f} leads "
          f"= {FT['training_cost'] / best_saving / LEADS_PER_MONTH:,.1f} months at {LEADS_PER_MONTH:,} leads a month")
else:
    print("best case (same price): A's prompt is no shorter than B's, so there's nothing to save")

# %% [markdown]
# ## 8. The memo

# %%
memo = {
    "compared": list(runs),
    "suite": "m07_outreach_angle (20 cases)",
    "recommendation": (
        "Keep B, the v3 prompt with one worked example. It's the cheapest approach that keeps the "
        "example, which carries the format and tone, and the prefix is too short for caching to matter. "
        "C adds retrieved Tallyhawk facts for about 30% more input tokens, which only pays off "
        "if real-model tests show openers making unsupported product claims. Fine-tuning never breaks "
        "even at these inputs, because the tuned model's higher price outweighs the tokens it saves."
    ),
    "what_would_change_it": [
        "Real-model pass rates where A matches B: drop the example and save the tokens.",
        "Openers making product claims the suite flags as unsupported: switch to C.",
        "Volume in the hundreds of thousands of leads a month, a tuned-model price close to the base "
        "price, and a stable task: rerun the break-even.",
        "A provider that stops offering the base model: the decision restarts from the top.",
    ],
}
print(json.dumps(memo, indent=2))

# %% [markdown]
# ## Done-when checks

# %%
from lib.checks import done_when

if OPERATOR_VALUE == OPERATOR_DEFAULT:
    from lib.figures import record
    key = {"A: prompt only": "a", "B: prompt + 1 example": "b", "C: + retrieved facts": "c"}
    figs = {
        "schema_tokens": SCHEMA_TOKENS, "output_tokens_estimate": OUTPUT_TOKENS,
        "prefix_tokens_b": prefix_b, "cache_min_tokens_example": CACHE_MIN_TOKENS,
        "price_input_per_million_example": PRICES["input_per_million"],
        "price_output_per_million_example": PRICES["output_per_million"],
        "leads_per_month_example": LEADS_PER_MONTH,
        "ft_training_cost_example": FT["training_cost"], "ft_price_multiplier_example": FT["price_multiplier"],
        "cost_per_lead_b_base": plain_b, "cost_per_lead_a_tuned": tuned_a,
        "ft_saving_per_lead": saving, "ft_best_case_saving_per_lead": best_saving,
        "ft_best_case_breakeven_leads": FT["training_cost"] / best_saving if best_saving > 0 else None,
        "ft_best_case_breakeven_months": FT["training_cost"] / best_saving / LEADS_PER_MONTH if best_saving > 0 else None,
    }
    for name, r in runs.items():
        k = key[name]
        figs[f"input_tokens_per_lead_{k}"] = r["input_tokens_per_lead"]
        figs[f"cost_per_lead_{k}"] = cost_per_lead(r["input_tokens_per_lead"])
        figs[f"cost_per_month_{k}"] = cost_per_lead(r["input_tokens_per_lead"]) * LEADS_PER_MONTH
        # Quality per approach: the suite pass rate (plumbing only in mock mode).
        figs[f"suite_pass_rate_{k}"] = r["pass_rate"]
        figs[f"suite_passed_{k}"] = r["passed"]
        figs[f"suite_total_{k}"] = r["total"]
        if not mock_mode():  # real-only: what the provider charged, and how long it took
            figs[f"logged_cost_per_lead_{k}"] = r["logged_cost_per_lead"]
            figs[f"logged_output_tokens_per_lead_{k}"] = r["logged_output_tokens_per_lead"]
            figs[f"seconds_per_lead_{k}"] = r["seconds_per_lead"]
    figs["c_vs_b_input_tokens_increase"] = runs["C: + retrieved facts"]["input_tokens_per_lead"] / runs["B: prompt + 1 example"]["input_tokens_per_lead"] - 1
    figs["c_vs_b_cost_per_lead_increase"] = figs["cost_per_lead_c"] / figs["cost_per_lead_b"] - 1
    record("m11", **figs)

checks = [
    ("At least two approaches ran on the same eval suite", len(runs) >= 2 and all(r["pass_rate"] is not None for r in runs.values())),
    ("Cost per lead comes from measured tokens times prices you entered",
     all(v is not None for v in PRICES.values()) and table["cost / lead (USD)"].gt(0).all()),
    ("The memo makes a recommendation", len(memo["recommendation"].split()) >= 20),
    ("The memo says what would change the decision", len(memo["what_would_change_it"]) >= 2),
]
if OPERATOR_VALUE == OPERATOR_DEFAULT:
    assert done_when(checks)
else:
    print(f"OPERATOR_VALUE is {OPERATOR_VALUE:,}, not the default, so these are for reading, not grading:")
    _ = done_when(checks)
