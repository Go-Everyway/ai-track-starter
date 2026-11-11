"""Build evals/labels/outreach_labelled.jsonl: 40 outreach-angle outputs with human labels.

Written for the course. The outputs are hand-designed from templates, not produced by a
model, so they're the same for everyone. Each one targets a real company profile in
data/companies.jsonl and is either good or carries one deliberate defect. Three are
borderline cases where a careful reviewer passes something a strict rule would fail;
those are what make grader validation interesting.

    python evals/labels/make_outreach_labelled.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from lib.grader import GOOD_FIT, code_grade  # noqa: E402

OUT = Path(__file__).with_name("outreach_labelled.jsonl")
GROWTH = {"announced a second distribution center", "raised a growth round",
          "launched a new product line", "expanded into a new region"}
POOR_FIT = {"Healthcare", "Software", "Education", "Professional services"}

companies = [json.loads(l) for l in (ROOT / "data" / "companies.jsonl").read_text().splitlines()]
used: set[str] = set()


def has(c, word):
    return word in c["description"]


def pick(pred):
    for c in companies:
        if c["company_id"] not in used and pred(c):
            used.add(c["company_id"])
            return c
    raise SystemExit("no company left for a spec")


def product(c):
    return c["description"].split("focused on ", 1)[1].split(".", 1)[0]


def signal(c):
    return c["signals"][0] if c["signals"] else None


def base_output(c, pain, confidence):
    """A good output for company c with the given pain point."""
    name, prod = c["name"], product(c)
    if pain == "stockouts":
        ev = ["reported stockouts during the last peak season"]
        angle = f"{name} ran short at peak, and Tallyhawk forecasts each item early enough to reorder in time."
        opener = f"Saw that {name} had stockouts last peak season. That's usually the forecast, not the warehouse. Worth comparing notes before the next one?"
    elif pain == "manual_planning":
        ev = ["a lot of spreadsheets"] if has(c, "a lot of spreadsheets") else ["mostly in spreadsheets"]
        angle = f"{name} plans stock in spreadsheets, and Tallyhawk replaces that with a nightly forecast per item."
        opener = f"Planning {prod} in spreadsheets gets harder with every new item. We help teams like {name} forecast each one, every night."
    elif pain == "growth_complexity":
        ev = [signal(c)]
        angle = f"{name} is growing, and more locations or lines make guessing what to stock where much harder."
        opener = f"Saw that {name} {signal(c)}. More moving parts usually means more guesswork on what to stock where. Happy to share how others handle it."
    else:
        ev = [product(c)]
        angle = f"Nothing in {name}'s profile points to an inventory problem yet."
        opener = f"{name} may never need forecasting software, but if planning {prod} gets harder, we'd be glad to help."
    return {"angle": angle, "pain_point": pain, "evidence": ev, "opener": opener, "confidence": confidence}


items = []


def add(c, output, verdict, mode, reason, expect_code_v1, expect_code_v2=None):
    """Record one labelled item, and assert the code grader behaves as designed."""
    for version, want in [("v1", expect_code_v1), ("v2", expect_code_v2 or expect_code_v1)]:
        got = code_grade(c, output, version=version)["verdict"]
        assert got == want, (c["company_id"], mode, version, got, code_grade(c, output, version=version))
    items.append({"id": f"L{len(items) + 1:02d}", "company_id": c["company_id"], "company": c, "output": output,
                  "human": {"verdict": verdict, "failure_mode": mode, "reason": reason}})


def good_fit(c):
    return c["industry"] in GOOD_FIT


# ---- 16 good outputs
for _ in range(4):
    c = pick(lambda c: good_fit(c) and has(c, "stockouts") and not has(c, "spreadsheets"))
    add(c, base_output(c, "stockouts", "high"), "pass", "none", "Specific, grounded in the stockout signal.", "pass")
for _ in range(5):
    c = pick(lambda c: good_fit(c) and has(c, "spreadsheets") and not has(c, "stockout"))
    add(c, base_output(c, "manual_planning", "high"), "pass", "none", "Spreadsheet planning is the clear pain; evidence is quoted.", "pass")
for _ in range(3):
    c = pick(lambda c: good_fit(c) and signal(c) in GROWTH and not has(c, "spreadsheets") and not has(c, "stockout"))
    add(c, base_output(c, "growth_complexity", "high"), "pass", "none", "Ties the growth signal to a stocking problem.", "pass")
for _ in range(2):
    c = pick(lambda c: c["industry"] == "Retail" and has(c, "spreadsheets") and not has(c, "stockout"))
    add(c, base_output(c, "manual_planning", "medium"), "pass", "none", "Reasonable fit, confidence kept at medium.", "pass")
for _ in range(2):
    c = pick(lambda c: c["industry"] in POOR_FIT and has(c, "spreadsheets") and not has(c, "stockout"))
    add(c, base_output(c, "manual_planning", "low"), "pass", "none", "Weak fit, and the output says so with low confidence.", "pass")

# ---- invented evidence
for fake in ["lost a major customer to stockouts", "plans to double its warehouse space",
             "runs its inventory on a legacy SAP system", "grew revenue 40 percent last year"]:
    c = pick(lambda c: good_fit(c) and has(c, "spreadsheets") and not has(c, "stockout"))
    o = base_output(c, "manual_planning", "high")
    o["evidence"] = o["evidence"] + [fake]
    add(c, o, "fail", "invented_evidence", f"\"{fake}\" isn't in the description.", "fail")

# ---- generic openers: three with stock phrases a rule catches, one that only a reader catches
for opener in ["Hope this finds you well. I'd love to tell you about Tallyhawk and how it can help your business.",
               "I wanted to reach out because Tallyhawk helps companies like yours with inventory.",
               "Quick question: who handles inventory planning on your team?"]:
    c = pick(lambda c: good_fit(c) and has(c, "spreadsheets") and not has(c, "stockout"))
    o = base_output(c, "manual_planning", "high")
    o["opener"] = opener
    add(c, o, "fail", "generic_opener", "Could be sent to any company unchanged.", "fail")
c = pick(lambda c: good_fit(c) and has(c, "spreadsheets") and not has(c, "stockout"))
o = base_output(c, "manual_planning", "high")
o["opener"] = "Companies like yours are always looking for ways to improve operations, and we'd love to show you what's possible."
add(c, o, "fail", "generic_opener", "No stock phrase, but nothing in it is about this company.", "pass")

# ---- too long
for _ in range(3):
    c = pick(lambda c: good_fit(c) and has(c, "spreadsheets") and not has(c, "stockout"))
    o = base_output(c, "manual_planning", "high")
    o["opener"] += (" Most of the teams we work with were in exactly the same place a year ago, with planners rebuilding"
                    " the same sheets every Monday, and they now spend that time on suppliers instead.")
    add(c, o, "fail", "too_long", "Opener runs well past 40 words.", "fail")

# ---- planted links
for link in ["Book a time at calendly.example/tallyhawk.", "Details at www.tallyhawk-demo.com.", "See https://tinyurl.example/forecast."]:
    c = pick(lambda c: good_fit(c) and has(c, "spreadsheets") and not has(c, "stockout"))
    o = base_output(c, "manual_planning", "high")
    o["opener"] = o["opener"] + " " + link
    add(c, o, "fail", "planted_link", "Contains a link; first emails go out without one.", "fail")

# ---- wrong pain point: three a rule catches, one only a reader catches
for _ in range(2):
    c = pick(lambda c: good_fit(c) and has(c, "spreadsheets") and not has(c, "stockout"))
    o = base_output(c, "manual_planning", "high")
    o["pain_point"] = "stockouts"
    add(c, o, "fail", "wrong_pain_point", "Claims stockouts the description never mentions.", "fail")
c = pick(lambda c: good_fit(c) and has(c, "stockouts") and not has(c, "spreadsheets"))
o = base_output(c, "stockouts", "high")
o["pain_point"] = "none_evident"
add(c, o, "fail", "wrong_pain_point", "The description reports stockouts; 'none evident' is wrong.", "fail")
c = pick(lambda c: good_fit(c) and has(c, "spreadsheets") and not has(c, "stockout"))
o = base_output(c, "manual_planning", "high")
o["pain_point"] = "excess_inventory"
add(c, o, "fail", "wrong_pain_point", "Nothing suggests excess stock; the evidence is about spreadsheets.", "pass")

# ---- overconfident
for _ in range(3):
    c = pick(lambda c: c["industry"] in POOR_FIT and has(c, "spreadsheets") and not has(c, "stockout"))
    add(c, base_output(c, "manual_planning", "high"), "fail", "overconfident",
        f"High confidence for a {c['industry'].lower()} company, which is a weak fit.", "fail")

# ---- borderline: a careful reviewer passes these; strict rules don't
c = pick(lambda c: good_fit(c) and has(c, "an older on-premise ERP and a lot of spreadsheets") and not has(c, "stockout"))
o = base_output(c, "manual_planning", "high")
o["evidence"] = ["an older on-premise ERP ... a lot of spreadsheets"]
add(c, o, "pass", "none", "Two exact phrases joined with an ellipsis. Faithful.", "fail", "pass")
c = pick(lambda c: good_fit(c) and has(c, "stockouts") and has(c, "spreadsheets"))
o = base_output(c, "manual_planning", "high")
add(c, o, "pass", "none", "Mentions stockouts too, but spreadsheets are the likely cause. Defensible.", "fail")
c = pick(lambda c: c["industry"] == "Construction" and has(c, "stockouts"))
add(c, base_output(c, "stockouts", "high"), "pass", "none",
    "Construction is usually a weak fit, but a reported stockout is a strong enough signal.", "fail")

assert len(items) == 40, len(items)
OUT.write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items), encoding="utf-8")
print(f"wrote {len(items)} items to {OUT.relative_to(ROOT)}: "
      f"{sum(i['human']['verdict'] == 'pass' for i in items)} pass, {sum(i['human']['verdict'] == 'fail' for i in items)} fail")
