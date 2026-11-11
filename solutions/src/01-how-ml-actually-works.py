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
# head. `max_depth=3` allows at most three questions per lead. Print the tree and read the
# rules it found.

# %%
from sklearn.tree import DecisionTreeClassifier, export_text

FEATURES = ["industry", "employees", "country", "lead_source", "seniority", "job_function",
            "pages_viewed", "pricing_page_visits", "email_opens", "email_clicks",
            "webinar_attended", "erp_type", "hiring_ops_roles", "recent_funding"]

X_train = pd.get_dummies(train[FEATURES], dtype=float)
X_test = pd.get_dummies(test[FEATURES], dtype=float).reindex(columns=X_train.columns, fill_value=0)

simple = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X_train, train.converted)
simple_train = catch_rate(simple.predict_proba(X_train)[:, 1], train.converted)
simple_test = catch_rate(simple.predict_proba(X_test)[:, 1], test.converted)
print(f"simple model: train {simple_train:.1%}  test {simple_test:.1%}\n")
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
    index=["hand-written rules", "simple model (depth 3)", "overfit model"],
)
results["gap"] = results.train - results.test
results.round(3)

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

assert done_when([
    ("Train and test numbers for all three approaches",
     set(results.index) == {"hand-written rules", "simple model (depth 3)", "overfit model"} and results.notna().all().all()),
    ("The overfit model looks perfect on training data", results.loc["overfit model", "train"] >= 0.99),
    ("Your rules beat random on the test set (about 10%)", results.loc["hand-written rules", "test"] > 0.15),
    ("The overfit model has the biggest train-to-test gap", results.gap.idxmax() == "overfit model"),
    ("You explained the gap in a couple of sentences", len(why_the_gap.split()) >= 20),
])
