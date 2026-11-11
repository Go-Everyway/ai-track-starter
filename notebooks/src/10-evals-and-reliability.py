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
# YOUR TURN: write your failure taxonomy. One entry per kind of failure you found, with a
# sentence on what it looks like. Use the counts above, and with a real key, your 50 outputs.
taxonomy = {
    # "invented_evidence": "Evidence that isn't in the description.",
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
# YOUR TURN: make a step that should fail. outreach_step("v1") is the loose prompt from
# Module 7; or write your own function that returns something broken. As written, this is
# the healthy step, so the suite passes and the done-when check below stays red.
broken_step = outreach_step("v3")
broken = run_all(only=["m07_outreach_angle"], steps={"m07_outreach_angle": broken_step})

# %% [markdown]
# ## 4. Is the grader right? Check it against people
#
# The code grader runs the rules from the taxonomy. Compare its verdicts with the person's.
#
# Split the labels first, so you develop on one half and measure on the other. Tuning a
# grader on the outputs you measure it on gives an in-sample number that flatters it. The
# rule here: within each kind of failure (and the passes), take the outputs in id order and
# alternate, development then held-out. Every label came from one person, so kappa below is
# agreement with that one labeler, not between people.

# %%
def grade_all(grader, items=None, **kw):
    return [grader(i["company"], i["output"], **kw)["verdict"] == "pass" for i in (items or labelled)]

by_mode = {}
for i in labelled:
    by_mode.setdefault(i["human"]["failure_mode"], []).append(i)
dev_ids, held_ids = set(), set()
for mode, items in by_mode.items():
    for k, i in enumerate(sorted(items, key=lambda x: x["id"])):
        (dev_ids if k % 2 == 0 else held_ids).add(i["id"])
dev = [i for i in labelled if i["id"] in dev_ids]
held = [i for i in labelled if i["id"] in held_ids]
human = lambda items: [i["human"]["verdict"] == "pass" for i in items]

dev_v1 = agreement(human(dev), grade_all(code_grade, dev, version="v1"))
print(f"code grader v1 on the development half: {dev_v1['accuracy']:.1%} agreement, kappa {dev_v1['kappa']:.2f}")

# %%
# Read the development disagreements only. Decide which are rule problems you can fix and
# which are judgment calls.
for i, g in zip(dev, grade_all(code_grade, dev, version="v1")):
    if g != (i["human"]["verdict"] == "pass"):
        print(f"{i['id']}  person: {i['human']['verdict']:<4}  grader: {'pass' if g else 'fail'}  | {i['human']['reason']}")

# %% [markdown]
# One is fixable: evidence stitched from two exact phrases with "..." is faithful, and v2 of
# the rule accepts it. Now measure both versions once on the held-out half.

# %%
held_v1 = agreement(human(held), grade_all(code_grade, held, version="v1"))
held_v2 = agreement(human(held), grade_all(code_grade, held, version="v2"))
print(f"held out: v1 {held_v1['accuracy']:.0%} (kappa {held_v1['kappa']:.2f}) -> v2 {held_v2['accuracy']:.0%} (kappa {held_v2['kappa']:.2f})")
code_v2 = held_v2

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
# YOUR TURN: judge at least 10 outputs yourself before reading the course labels.
# Print an item with print(json.dumps(labelled[0], indent=1)), cover the "human" part, decide.
my_labels = {
    # "L01": "pass",
}

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
_ = done_when([
    ("One command runs the suites and prints a pass rate", len(ran) >= 2 and "pass_rate" in report),
    ("The healthy suites pass", report["ok"]),
    ("The deliberately broken prompt fails the suite", not broken["ok"]),
    ("Grader-human agreement reported (accuracy and kappa)",
     all(k in code_v2 for k in ("accuracy", "kappa")) and len(model_results) == 2),
    ("You labelled at least 10 outputs yourself", len(my_labels) >= 10),
    ("Your failure taxonomy has at least four categories", len(taxonomy) >= 4),
    ("The gate blocks every output with a planted link", all(i in blocked for i in link_items)),
])
