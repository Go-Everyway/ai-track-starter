# %% [markdown]
# ## 1. Look at the data
#
# 4,000 leads for Tallyhawk, the fictional forecasting company. `converted` is the thing
# we want to predict: did this lead become a customer?

# %%
leads = pd.read_csv(DATA / "leads_clean.csv")
print(f"{len(leads):,} leads, {leads.converted.sum()} converted ({leads.converted.mean():.1%})")
leads.head()

# %% [markdown]
# ## OPERATOR: change one value
#
# `OPERATOR_VALUE` is how many questions the "simple model" decision tree may ask per lead
# (its `max_depth`). The default is 3. Try anything from 1 to 10, then choose Runtime → Run all,
# and watch the train and test numbers in the results table. A small number means a few broad
# rules; a big number lets the tree memorize more.
#
# The done-when checks at the bottom were written for the default value, so a ✗ there after
# you change it is expected, not a mistake.

# %%
OPERATOR_VALUE = 3  # simple model's max_depth, 1 to 10. Change this, then Runtime → Run all
DEFAULT_VALUE = 3

# %% [markdown]
# ## 2. Hold some data back
#
# Split once, before anything else, and don't touch the test set until you're measuring.
# `stratify` keeps the same conversion rate in both halves. `random_state` makes the split
# the same every time you run it.

# %%
from sklearn.model_selection import train_test_split

train, test = train_test_split(leads, test_size=0.25, random_state=0, stratify=leads.converted)
print(f"train: {len(train):,} leads, {train.converted.sum()} converted")
print(f"test:  {len(test):,} leads, {test.converted.sum()} converted")

# %% [markdown]
# ## 3. One way to score anything: the catch rate
#
# Your team can call the top 10% of leads. Rank every lead by score, take the top 10%, and ask:
# of all the leads that converted, what share are in there? Pick at random and you catch
# about 10%. Rank perfectly and you catch all of them, because only about 6% convert.

# %%
import numpy as np

def catch_rate(scores, converted):
    """Share of all conversions that land in the top-scored 10% of leads."""
    scores, converted = np.asarray(scores, dtype=float), np.asarray(converted)
    k = max(1, len(scores) // 10)
    tiebreak = np.random.default_rng(0).random(len(scores))
    order = np.lexsort((tiebreak, -scores))  # highest score first, ties in random order
    return converted[order[:k]].sum() / converted.sum()

print(f"random ranking on the test set: {catch_rate(np.zeros(len(test)), test.converted):.1%}")

# %% [markdown]
# ## 4. Rules you write by hand
#
# Three rules a salesperson would give you. Each one a lead meets adds a point.

# %%
GOOD_FIT = {"Wholesale distribution", "Manufacturing", "Food & beverage", "Logistics"}

def rule_score(row):
    score = 0
    # 1. They asked for something: a demo, the pricing page, or a referral.
    score += row.lead_source in {"inbound_demo", "pricing_page", "referral"}
    # 2. They look like our customers: right industry, right size.
    score += row.industry in GOOD_FIT and 50 <= row.employees <= 1000
    # 3. They keep coming back to the pricing page.
    score += row.pricing_page_visits >= 2
    return score

rules_train = catch_rate(train.apply(rule_score, axis=1), train.converted)
rules_test = catch_rate(test.apply(rule_score, axis=1), test.converted)
print(f"rules: train {rules_train:.1%}  test {rules_test:.1%}")

# %% [markdown]
# ## 5. Let a model find the rules
#
# A decision tree learns rules like yours, but from the training data instead of from your
# head. `max_depth=OPERATOR_VALUE` (3 by default) caps how many questions it asks per lead.
# Print the tree and read the rules it found.

# %%
from sklearn.tree import DecisionTreeClassifier, export_text

FEATURES = ["industry", "employees", "country", "lead_source", "seniority", "job_function",
            "pages_viewed", "pricing_page_visits", "email_opens", "email_clicks",
            "webinar_attended", "erp_type", "hiring_ops_roles", "recent_funding"]

X_train = pd.get_dummies(train[FEATURES], dtype=float)
X_test = pd.get_dummies(test[FEATURES], dtype=float).reindex(columns=X_train.columns, fill_value=0)

simple = DecisionTreeClassifier(max_depth=OPERATOR_VALUE, random_state=0).fit(X_train, train.converted)
simple_train = catch_rate(simple.predict_proba(X_train)[:, 1], train.converted)
simple_test = catch_rate(simple.predict_proba(X_test)[:, 1], test.converted)
print(f"simple model: train {simple_train:.1%}  test {simple_test:.1%}\n")
print(f"the tree has {simple.get_n_leaves()} leaves")
if OPERATOR_VALUE <= 4:  # deeper trees are too long to read
    print(export_text(simple, feature_names=list(X_train.columns), show_weights=True))

# %% [markdown]
# ## 6. Overfit on purpose
#
# Same model with no limit on how many questions it can ask. It keeps splitting until
# every training lead is sorted perfectly.

# %%
overfit = DecisionTreeClassifier(max_depth=None, random_state=0).fit(X_train, train.converted)
overfit_train = catch_rate(overfit.predict_proba(X_train)[:, 1], train.converted)
overfit_test = catch_rate(overfit.predict_proba(X_test)[:, 1], test.converted)
print(f"overfit model: train {overfit_train:.1%}  test {overfit_test:.1%}")
print(f"it grew {overfit.get_n_leaves():,} leaves to sort {len(train):,} training leads")

# %% [markdown]
# ## 7. Your results

# %%
results = pd.DataFrame(
    {"train": [rules_train, simple_train, overfit_train], "test": [rules_test, simple_test, overfit_test]},
    index=["hand-written rules", f"simple model (depth {OPERATOR_VALUE})", "overfit model"],
)
results["gap"] = results.train - results.test
results.round(3)

# %% [markdown]
# ## 8. Does it hold on other splits?
#
# One split of 1,000 leads holds only 63 conversions, so each one moves the catch rate by about
# 1.6 points. Rerun the whole comparison on ten different random splits and look at the spread.

# %%
def compare_on_split(seed):
    tr, te = train_test_split(leads, test_size=0.25, random_state=seed, stratify=leads.converted)
    Xtr = pd.get_dummies(tr[FEATURES], dtype=float)
    Xte = pd.get_dummies(te[FEATURES], dtype=float).reindex(columns=Xtr.columns, fill_value=0)
    out = {"rules": catch_rate(te.apply(rule_score, axis=1), te.converted)}
    for name, depth in [("simple", OPERATOR_VALUE), ("overfit", None)]:
        m = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(Xtr, tr.converted)
        out[name] = catch_rate(m.predict_proba(Xte)[:, 1], te.converted)
    return out

splits = pd.DataFrame([compare_on_split(seed) for seed in range(10)])
print("test catch rate over 10 random splits (seeds 0-9):")
print(splits.agg(["mean", "std", "min", "max"]).round(3))

# %%
why_the_gap = """
The overfit tree memorized the training leads, noise and all, so it ranks them perfectly and
has almost nothing true to say about leads it hasn't seen. The more a method learns from the
training data, the bigger the gap: my rules learned nothing from it and barely moved, the
depth-3 tree learned a little and dropped some, and the unlimited tree learned everything and
collapsed.
"""

# %% [markdown]
# ## Done-when checks

# %%
from lib.checks import done_when

checks = [
    ("Train and test numbers for all three approaches",
     len(results) == 3 and results.notna().all().all()),
    ("The overfit model looks perfect on training data", results.loc["overfit model", "train"] >= 0.99),
    ("Your rules beat random on the test set (about 10%)", results.loc["hand-written rules", "test"] > 0.15),
    ("The overfit model has the biggest train-to-test gap", results.gap.idxmax() == "overfit model"),
    ("You explained the gap in a couple of sentences", len(why_the_gap.split()) >= 20),
]
if OPERATOR_VALUE == DEFAULT_VALUE:
    assert done_when(checks)
    from lib.figures import record
    _bars = dict(rules_train=rules_train, rules_test=rules_test, simple_train=simple_train,
                 simple_test=simple_test, overfit_train=overfit_train, overfit_test=overfit_test)
    record("m01",
           leads=len(leads), converted=int(leads.converted.sum()), conversion_rate=leads.converted.mean(),
           train_leads=len(train), train_converted=int(train.converted.sum()),
           test_leads=len(test), test_converted=int(test.converted.sum()),
           rules_train=rules_train, rules_test=rules_test,
           simple_train=simple_train, simple_test=simple_test,
           overfit_train=overfit_train, overfit_test=overfit_test, overfit_leaves=int(overfit.get_n_leaves()),
           random_test=catch_rate(np.zeros(len(test)), test.converted),
           train_leads_per_leaf=len(train) / overfit.get_n_leaves(),
           points_per_test_conversion=100 / int(test.converted.sum()),
           splits_rules_mean=splits.rules.mean(), splits_simple_mean=splits.simple.mean(),
           splits_overfit_mean=splits.overfit.mean(),
           splits_std_points_max=100 * float(splits.std().max()), splits_std_points_min=100 * float(splits.std().min()),
           nobody_converts_accuracy=1 - leads.converted.mean(),
           # bar heights and tops for the page's chart (200 px = 100%)
           bar={k: round(v * 200, 1) for k, v in _bars.items()},
           bar_y={k: round(210 - v * 200, 1) for k, v in _bars.items()})
else:
    _ = done_when(checks)
