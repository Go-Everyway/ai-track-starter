# %%
import json
from collections import Counter

from lib import llm
from lib.outreach import outreach_angle
from lib.grader import code_grade, model_grade, agreement, RUBRIC_V1, RUBRIC_V2
from lib.harness import gate, queue_for_send
from evals.run import run_all
from evals.registry import outreach_step

MOCK = llm.settings()["provider"] == "mock"
print("mode:", "MOCK: model outputs are filler, so judge the plumbing, not the quality" if MOCK else "real model")

# %% [markdown]
# ## 1. Read outputs before you measure anything
#
# Run the Module 7 step on 50 companies and read what comes back. Don't score anything yet.
# Write down every way an output goes wrong, in your own words, then group them. That list
# is your failure taxonomy, and every check and grader below comes from it.
#
# In mock mode the outputs are schema-valid filler, so there's nothing to read. The kit's
# labelled set (40 outputs written for the course, each judged by a person) stands in.

# %%
companies = pd.read_json(DATA / "companies.jsonl", lines=True).to_dict("records")
first_50 = [outreach_angle(c) for c in companies[:50]]
for c, r in list(zip(companies, first_50))[:3]:
    print(c["name"], "|", r["output"]["opener"] if r["ok"] else r["error"])

labelled = [json.loads(l) for l in (ROOT / "evals" / "labels" / "outreach_labelled.jsonl").read_text().splitlines()]
human_pass = [i["human"]["verdict"] == "pass" for i in labelled]
print(f"\nlabelled set: {len(labelled)} outputs, {sum(human_pass)} judged pass, {len(labelled) - sum(human_pass)} fail")
Counter(i["human"]["failure_mode"] for i in labelled if i["human"]["verdict"] == "fail").most_common()

# %%
taxonomy = {
    "invented_evidence": "Evidence that isn't in the description. The worst one: it puts a false claim in a rep's mouth.",
    "generic_opener": "An opener you could send to anyone. Some use stock phrases; some are just vague.",
    "wrong_pain_point": "The pain point doesn't match what the description supports.",
    "planted_link": "A link in a first email, often planted by text in the description.",
    "too_long": "An opener over 40 words.",
    "overconfident": "High confidence for a company that's a weak fit.",
}

# %% [markdown]
# ## 2. One command runs every suite
#
# `evals/run.py` runs every suite in `evals/registry.py` and fails when any is under its
# threshold. From a terminal: `python evals/run.py`. Same thing from here:

# %%
report = run_all()

# %% [markdown]
# ## 3. Break it on purpose
#
# A harness you've never seen fail isn't proven. Swap in Module 7's v1 prompt, which asks
# for JSON without enforcing it, and the suite has to go red.

# %%
broken_step = outreach_step("v1")
broken = run_all(only=["m07_outreach_angle"], steps={"m07_outreach_angle": broken_step})

# %% [markdown]
# ## 4. Is the grader right? Check it against people
#
# The code grader runs the rules from the taxonomy. Score all 40 labelled outputs with it,
# then compare with the human verdicts.

# %%
def grade_all(grader, **kw):
    return [grader(i["company"], i["output"], **kw)["verdict"] == "pass" for i in labelled]

code_v1 = agreement(human_pass, grade_all(code_grade, version="v1"))
print(f"code grader v1: {code_v1['accuracy']:.1%} agreement, kappa {code_v1['kappa']:.2f}")
print(f"  too easy (person failed it, grader passed): {code_v1['grader_too_easy']}")
print(f"  too hard (person passed it, grader failed): {code_v1['grader_too_hard']}")

# %%
for i, g in zip(labelled, grade_all(code_grade, version="v1")):
    if g != (i["human"]["verdict"] == "pass"):
        print(f"{i['id']}  person: {i['human']['verdict']:<4}  grader: {'pass' if g else 'fail'}  | {i['human']['reason']}")

# %% [markdown]
# Two misses are things rules can't see (a vague opener with no stock phrase, a pain point
# that's wrong but plausible). That's what a model grader is for. Three are the grader being
# stricter than a careful reviewer. One of those is fixable: evidence stitched from two exact
# phrases with "..." is faithful, and v2 of the rule accepts it.

# %%
code_v2 = agreement(human_pass, grade_all(code_grade, version="v2"))
print(f"code grader v2: {code_v2['accuracy']:.1%} agreement, kappa {code_v2['kappa']:.2f}")

# %% [markdown]
# Now the model grader, with the vague rubric and the specific one. **In mock mode this is
# meaningless**: the mock always answers "pass", so agreement equals the share of outputs
# people passed and kappa is 0. With a real key, this is the comparison that matters.

# %%
model_results = {}
for name, rubric in [("vague rubric", RUBRIC_V1), ("specific rubric", RUBRIC_V2)]:
    a = agreement(human_pass, grade_all(model_grade, rubric=rubric))
    model_results[name] = a
    print(f"model grader, {name}: {a['accuracy']:.1%} agreement, kappa {a['kappa']:.2f}{'  (MOCK)' if MOCK else ''}")

# %% [markdown]
# ## 5. Label some yourself
#
# Before you trust anyone's labels, including ours, judge at least 10 outputs yourself.
# Read the company description and the output, and decide pass or fail. Then compare.

# %%
for i in labelled[:3]:
    print(i["id"], "|", i["company"]["description"][:120], "...\n   ", json.dumps(i["output"])[:220], "\n")

# %%
my_labels = {i["id"]: i["human"]["verdict"] for i in labelled[:10]}  # the solution agrees with the course labels

mine = [my_labels[i["id"]] == "pass" for i in labelled if i["id"] in my_labels]
ours = [i["human"]["verdict"] == "pass" for i in labelled if i["id"] in my_labels]
me_vs_course = agreement(ours, mine) if my_labels else None
print(f"you vs the course labels on {len(my_labels)} outputs:",
      f"{me_vs_course['accuracy']:.0%}" if me_vs_course else "label some first")

# %% [markdown]
# ## 6. Gate every send
#
# The suite protects deploys. The gate protects each message: right before anything would
# go out, run the checks on that one output and refuse it if any fail.

# %%
decisions = [(i, queue_for_send(i["company"], {"ok": True, "output": i["output"]})) for i in labelled]
blocked = [i for i, d in decisions if not d["queued"]]
print(f"gate: {len(labelled) - len(blocked)} queued, {len(blocked)} blocked")
print("blocked, by what the person said was wrong:",
      dict(Counter(i["human"]["failure_mode"] for i in blocked)))

# %% [markdown]
# ## Done-when checks

# %%
from lib.checks import done_when

ran = [s for s in report["suites"].values() if "skipped" not in s]
link_items = [i for i in labelled if i["human"]["failure_mode"] == "planted_link"]
assert done_when([
    ("One command runs the suites and prints a pass rate", len(ran) >= 2 and "pass_rate" in report),
    ("The healthy suites pass", report["ok"]),
    ("The deliberately broken prompt fails the suite", not broken["ok"]),
    ("Grader-human agreement reported (accuracy and kappa)",
     all(k in code_v2 for k in ("accuracy", "kappa")) and len(model_results) == 2),
    ("You labelled at least 10 outputs yourself", len(my_labels) >= 10),
    ("Your failure taxonomy has at least four categories", len(taxonomy) >= 4),
    ("The gate blocks every output with a planted link", all(i in blocked for i in link_items)),
])
