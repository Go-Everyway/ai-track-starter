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
# ## OPERATOR: change one value
#
# Step 3 runs the outreach test suite against one version of Module 7's prompt. `"v1"` is the
# deliberately broken one: it asks for JSON but doesn't enforce it, so the suite should go
# red. Try `"v2"` or `"v3"` (the shipped version) and run everything again, then compare the
# pass rate step 3 prints. Allowed values: "v1", "v2", "v3".

# %%
OPERATOR_VALUE = "v1"  # change this, then Runtime → Run all
OPERATOR_DEFAULT = "v1"
assert OPERATOR_VALUE in ("v1", "v2", "v3"), "use v1, v2 or v3"

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
first_50 = [outreach_angle(c) for c in companies[:50]]  # never raises: a bad reply comes back with ok=False
for c, r in list(zip(companies, first_50))[:3]:
    print(c["name"], "|", (r.get("output") or {}).get("opener") if r["ok"] else r["error"])
print(f"{sum(not r['ok'] for r in first_50)} of 50 outputs failed outright (no valid JSON after a retry)")

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
# for JSON without enforcing it, and the suite has to go red. (Which version runs here is
# the operator value at the top.)

# %%
broken_step = outreach_step(OPERATOR_VALUE)
broken = run_all(only=["m07_outreach_angle"], steps={"m07_outreach_angle": broken_step})
print(f"prompt {OPERATOR_VALUE}: suite pass rate {broken['pass_rate']:.0%}, {'PASS' if broken['ok'] else 'FAIL'}")

# %% [markdown]
# ## 4. Is the grader right? Check it against people
#
# The code grader runs the rules from the taxonomy. Score all 40 labelled outputs with it,
# then compare with the human verdicts.

# %%
def grade_all(grader, **kw):
    return [grader(i["company"], i["output"], **kw)["verdict"] == "pass" for i in labelled]


def model_grade_all(**kw):
    """Model-grade every labelled output. A reply that isn't valid JSON, or a refusal, is
    counted as a grader failure and left out of the agreement, never a crash."""
    verdicts, failures = [], 0
    for i in labelled:
        try:
            v = model_grade(i["company"], i["output"], **kw).get("verdict")
        except Exception:
            v = None
        if v not in ("pass", "fail"):
            failures += 1
        verdicts.append(v)
    return verdicts, failures

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
model_results, model_failures = {}, {}
for name, rubric in [("vague rubric", RUBRIC_V1), ("specific rubric", RUBRIC_V2)]:
    verdicts, failures = model_grade_all(rubric=rubric)
    kept = [(h, v == "pass") for h, v in zip(human_pass, verdicts) if v in ("pass", "fail")]
    a = agreement([h for h, _ in kept], [g for _, g in kept]) if kept else {"n": 0, "accuracy": None, "kappa": None}
    model_results[name], model_failures[name] = a, failures
    shown = f"{a['accuracy']:.1%} agreement, kappa {a['kappa']:.2f}" if kept else "no usable verdicts"
    print(f"model grader, {name}: {shown} on {a['n']} outputs, {failures} unusable replies{'  (MOCK)' if MOCK else ''}")

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
def safe_queue(i):
    try:
        return queue_for_send(i["company"], {"ok": True, "output": i["output"]})
    except Exception as e:  # a grader that errors must block, never wave a message through
        return {"queued": False, "reasons": [f"gate error: {type(e).__name__}"]}

decisions = [(i, safe_queue(i)) for i in labelled]
blocked = [i for i, d in decisions if not d["queued"]]
queued = [i for i, d in decisions if d["queued"]]
blocked_human_pass = sum(i["human"]["verdict"] == "pass" for i in blocked)
queued_human_fail = sum(i["human"]["verdict"] == "fail" for i in queued)
print(f"gate: {len(queued)} queued, {len(blocked)} blocked")
print(f"  blocked although the person passed it: {blocked_human_pass}")
print(f"  queued although the person failed it (the gate's misses): {queued_human_fail}")
print("blocked, by what the person said was wrong:",
      dict(Counter(i["human"]["failure_mode"] for i in blocked)))

# %% [markdown]
# ## Done-when checks

# %%
from lib.checks import done_when

ran = [s for s in report["suites"].values() if "skipped" not in s]
link_items = [i for i in labelled if i["human"]["failure_mode"] == "planted_link"]

if OPERATOR_VALUE == OPERATOR_DEFAULT:
    from lib.figures import record
    record("m10",
           labelled_n=len(labelled), labelled_pass=sum(human_pass), labelled_fail=len(labelled) - sum(human_pass),
           labelled_failure_modes=dict(Counter(i["human"]["failure_mode"] for i in labelled if i["human"]["verdict"] == "fail")),
           taxonomy_categories=len(taxonomy),
           suite_pass_rates={k: v.get("pass_rate") for k, v in report["suites"].items() if "skipped" not in v},
           suite_overall_pass_rate=report["pass_rate"], suites_ok=report["ok"],
           broken_prompt_pass_rate=broken["pass_rate"], broken_prompt_ok=broken["ok"],
           code_grader_v1_accuracy=code_v1["accuracy"], code_grader_v1_kappa=code_v1["kappa"],
           code_grader_v1_too_easy=code_v1["grader_too_easy"], code_grader_v1_too_hard=code_v1["grader_too_hard"],
           code_grader_v2_accuracy=code_v2["accuracy"], code_grader_v2_kappa=code_v2["kappa"],
           model_grader_vague_accuracy=model_results["vague rubric"]["accuracy"],
           model_grader_vague_kappa=model_results["vague rubric"]["kappa"],
           model_grader_specific_accuracy=model_results["specific rubric"]["accuracy"],
           model_grader_specific_kappa=model_results["specific rubric"]["kappa"],
           model_grader_unusable_replies=model_failures,
           gate_queued=len(queued), gate_blocked=len(blocked),
           gate_blocked_human_pass=blocked_human_pass, gate_queued_human_fail=queued_human_fail,
           first_50_failed=sum(not r["ok"] for r in first_50))

checks = [
    ("One command runs the suites and prints a pass rate", len(ran) >= 2 and "pass_rate" in report),
    # Mock mode skips the real-model checks, so the healthy suites must pass there. With a real
    # model the pass rate is a finding to report, not a given.
    ("The healthy suites pass" if MOCK else "The suites ran and reported a pass rate",
     report["ok"] if MOCK else isinstance(report["pass_rate"], float)),
    ("The deliberately broken prompt fails the suite" if MOCK else "The prompt under test got a pass rate",
     (not broken["ok"]) if MOCK else isinstance(broken["pass_rate"], float)),
    ("Grader-human agreement reported (accuracy and kappa)",
     all(k in code_v2 for k in ("accuracy", "kappa")) and len(model_results) == 2),
    ("You labelled at least 10 outputs yourself", len(my_labels) >= 10),
    ("Your failure taxonomy has at least four categories", len(taxonomy) >= 4),
    ("The gate blocks every output with a planted link", all(i in blocked for i in link_items)),
]
if OPERATOR_VALUE == OPERATOR_DEFAULT:
    assert done_when(checks)
else:
    print(f"OPERATOR_VALUE is {OPERATOR_VALUE!r}, not the default, so these are for reading, not grading:")
    _ = done_when(checks)
