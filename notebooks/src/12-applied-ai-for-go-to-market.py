# %% [markdown]
# ## 0. What this notebook does and doesn't do
#
# It assembles the system from Modules 4, 7, 8, 9 and 10: signals, a scored list, a holdout,
# drafts that are checked before anyone approves them, a send path that respects opt-outs, a
# sending-setup check, and a results page.
#
# It never sends real email, and the replies and meetings at the end are **SIMULATED** from
# assumptions you can see. That's enough to build and test the results code. It says nothing
# about what a real campaign would get.

# %%
import re
from datetime import date

from lib import campaign, outreach, pipeline

world = pipeline.World.load()
LOGS = ROOT / "logs"

# %% [markdown]
# ## 1. Write the before number down first
#
# Before the system touches a single account. `record_before` refuses to overwrite an existing
# record, because a before number you can edit later isn't one.
#
# The values below are example numbers for the old process. Use your own, from your CRM, for
# the same length of time you'll run the test.

# %%
# YOUR TURN: put your old process's numbers here (or the example ones: a 4% reply rate and
# 1.6 meetings per 100 accounts contacted), with the period they cover. Then rerun this cell.
MY_BEFORE = {}
MY_PERIOD = ""

BEFORE_PATH = LOGS / "before-my-campaign.json"
before = campaign.read_before(BEFORE_PATH)
if before is None and MY_BEFORE and MY_PERIOD:
    before = campaign.record_before(MY_BEFORE, period=MY_PERIOD, path=BEFORE_PATH)
before or print("No before number recorded yet. Fill in MY_BEFORE and MY_PERIOD.")

# %% [markdown]
# ## 2. Signals to offers, then a scored list
#
# A signal only counts if it maps to something Tallyhawk sells. The map lives in
# `campaign.SIGNAL_OFFERS`. Look at what's left unmapped: those signals are trivia, not reasons
# to write.

# %%
ranked = campaign.scored_list(world)
with_offer = ranked.offer.notna()
print(f"{len(ranked)} leads with a company profile; {with_offer.sum()} have a signal that maps to an offer")
unmapped = sorted({s for c in world.companies.values() for s in campaign.map_signals(c)["unmapped"]})
print("signals with no offer:", unmapped)
ranked.head(10)

# %% [markdown]
# ## 3. Split into treatment and holdout, before anything is sent
#
# Half the list gets the new system; half keeps the old process over the same weeks. The split
# is random, seeded, and stable per lead.

# %%
# YOUR TURN: replace this with campaign.assign_holdout(ranked.lead_id, share=0.5, seed=12).
# As written, everyone gets the new system, so there's nothing to compare against.
groups = {i: "treatment" for i in ranked.lead_id}
ranked["group"] = ranked.lead_id.map(groups)
print(ranked.group.value_counts().to_dict())
print("mean score by group (should be close, since the split is random):")
print(ranked.groupby("group").score.mean().round(4).to_dict())

# %% [markdown]
# ## 4. Draft, check, approve, send: three treatment leads by hand
#
# `run_lead` (Module 9) drafts and checks every claim. In mock mode the drafts are meaningless,
# so they land in `needs_review`, and a person rewrites them. That's the path a real draft with
# an unsupported claim takes too.

# %%
def recipient_for(company_id):
    slug = re.sub(r"[^a-z0-9]", "", world.companies[company_id]["name"].lower())[:24]
    return f"ops@{slug}.example"

demo = ranked[ranked.group == "treatment"].head(3)
stop = campaign.Suppression()
sent = []
for row in demo.itertuples():
    result = pipeline.run_lead(world, row.lead_id)
    profile = world.drafts[result["draft_id"]]["profile"]
    focus = re.search(r"focused on [^.]+", profile["description"]).group(0)
    revised = {
        "angle": f"{profile['name']} is {focus} and carries real stock, so demand per item matters to them.",
        "pain_point": "growth_complexity",
        "evidence": [focus],
        "opener": "Tallyhawk forecasts refresh every night using the previous day's orders, so planners start the day with current numbers.",
        "confidence": "medium",
    }
    check = pipeline.revise(world, result["draft_id"], revised)
    gate_ok, gate_reasons = __import__("lib.harness", fromlist=["gate"]).gate(profile, {"ok": True, "output": revised}, use_model_grader=False)
    print(row.lead_id, "| first draft:", result["status"], "| after review:", check["status"], "| gate:", "pass" if gate_ok else gate_reasons)
    if check["status"] == "ready_for_approval" and gate_ok:
        token = pipeline.approve(world, result["draft_id"], approver="you")
        sent.append(campaign.send_message(world, stop, result["draft_id"], token,
                                          recipient=recipient_for(row.company_id),
                                          idempotency_key=f"{row.lead_id}-touch-1"))
print(f"\n{len(sent)} sent")

# %% [markdown]
# ## 5. Opt-outs and suppression
#
# The second lead replies "stop". From that moment nothing else reaches them, including a
# follow-up that was already approved. A whole domain can be suppressed too.

# %%
second = demo.iloc[1]
# YOUR TURN: record the opt-out from the second lead, and suppress the third lead's whole domain:
#   stop.opt_out(recipient_for(second.company_id), received=date(2026, 11, 11))
#   stop.add("@" + recipient_for(demo.iloc[2].company_id).split("@")[1], "asked us not to contact anyone there")

suppression_blocked = []
for row, key in [(demo.iloc[1], "touch-2"), (demo.iloc[2], "touch-2")]:
    draft_id = f"D-{row.lead_id}"
    token = pipeline.approve(world, draft_id, approver="you")  # approved, and still not sent
    try:
        campaign.send_message(world, stop, draft_id, token, recipient=recipient_for(row.company_id),
                              idempotency_key=f"{row.lead_id}-{key}")
        suppression_blocked.append(False)
    except campaign.Suppressed as e:
        suppression_blocked.append(True)
        print("blocked:", e)
print(stop.entries)

# %% [markdown]
# ## 6. Check the sending setup
#
# Paste your sending domain's SPF, DKIM and DMARC records. These samples are for a made-up
# domain; the DKIM key is a placeholder that's valid base64, not a real key.

# %%
# YOUR TURN: two of these have a problem. Run the cell, read what the checker says, fix them.
SPF = "v=spf1 include:_spf.mailhost.example +all"
DKIM = "v=DKIM1; k=rsa; p=MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAsampleplaceholderkeyonlyAAAB"
DMARC = "v=DMARC1; p=monitor; rua=mailto:dmarc-reports@send.tallyhawk.example"

setup = campaign.check_sending_setup(SPF, DKIM, DMARC)
print(setup)
print("a record that lets anyone send as you:", campaign.check_spf("v=spf1 include:_spf.mailhost.example +all"))

YOUR_DOMAIN = ""  # e.g. "send.yourcompany.com"; looked up with `dig` if it's installed
if YOUR_DOMAIN:
    print("SPF:", campaign.lookup_txt(YOUR_DOMAIN))
    print("DMARC:", campaign.lookup_txt(f"_dmarc.{YOUR_DOMAIN}"))

# %% [markdown]
# ## 7. Results: treatment vs holdout
#
# **SIMULATED.** Real replies and meetings come from your CRM after the test period. These come
# from `simulate_outcomes`, with reply rates you can see in the call: 4% for the old process and
# 6% for the new one, before the score adjustment. They exist to show what an honest results
# page looks like, nothing more.

# %%
scores = dict(zip(ranked.lead_id, ranked.score))
outcomes = campaign.simulate_outcomes(groups, scores, seed=12, holdout_reply=0.04, treatment_reply=0.06)
if set(groups.values()) == {"treatment", "holdout"}:
    report = campaign.results(outcomes)
    print("SIMULATED outcomes. Before number on record:", before and before["values"])
    display(campaign.results_table(report))
else:
    report = {"metrics": {}, "n_treatment": len(outcomes), "n_holdout": 0}
    print("No holdout group, so there's nothing to compare against. Fix step 3.")

# %%
# YOUR TURN: what do these results show, and what don't they? Look at the counts behind each
# rate, the 90% ranges, and the word SIMULATED.
what_the_results_say = """
"""

# %% [markdown]
# ## Done-when checks

# %%
from lib.checks import done_when

before_rec = campaign.read_before(BEFORE_PATH)
_ = done_when([
    ("A before number was recorded first, with its period", before_rec is not None and before_rec["period"]),
    ("A holdout group exists and was assigned before sending",
     set(ranked.group) == {"treatment", "holdout"} and all(groups[s["draft_id"][2:]] == "treatment" for s in sent)),
    ("Results report replies and meetings for both groups, with sample sizes",
     {"replied", "meeting"} <= set(report["metrics"]) and report["n_treatment"] > 0 and report["n_holdout"] > 0),
    ("SPF, DKIM and DMARC records pass the check", setup["ok"]),
    ("Suppression and opt-out block sends, even approved ones", suppression_blocked == [True, True]),
    ("You wrote down what the results do and don't show", len(what_the_results_say.split()) >= 25),
])
