# %% [markdown]
# ## 1. Same data, same split, same yardstick as Module 1
#
# Same 4,000 leads, same 75/25 split with the same `random_state`, and the same catch rate:
# the share of all conversions that land in the top 10% of your ranking. Random catches
# about 10%. Module 1's hand-written rules caught 38% of the test conversions.

# %%
import numpy as np
from sklearn.model_selection import train_test_split

leads = pd.read_csv(DATA / "leads_clean.csv")
train, test = train_test_split(leads, test_size=0.25, random_state=0, stratify=leads.converted)

def catch_rate(scores, converted):
    """Share of all conversions that land in the top-scored 10% of leads."""
    scores, converted = np.asarray(scores, dtype=float), np.asarray(converted)
    k = max(1, len(scores) // 10)
    tiebreak = np.random.default_rng(0).random(len(scores))
    order = np.lexsort((tiebreak, -scores))
    return converted[order[:k]].sum() / converted.sum()

GOOD_FIT = {"Wholesale distribution", "Manufacturing", "Food & beverage", "Logistics"}

def rule_score(row):
    return ((row.lead_source in {"inbound_demo", "pricing_page", "referral"})
            + (row.industry in GOOD_FIT and 50 <= row.employees <= 1000)
            + (row.pricing_page_visits >= 2))

rules_test = catch_rate(test.apply(rule_score, axis=1), test.converted)
print(f"baseline, Module 1 rules: {rules_test:.1%} of test conversions caught")

# %% [markdown]
# ## 2. Pick the columns
#
# Every column except the identifiers (`lead_id`, `company_id`, `company`), the date it was
# created, and the label. Numbers go in as numbers. Categories get one-hot encoded: one 0/1
# column per value.

# %%
NUMERIC = ["employees", "pages_viewed", "pricing_page_visits", "email_opens", "email_clicks",
           "webinar_attended", "hiring_ops_roles", "recent_funding", "meetings_held"]
CATEGORICAL = ["industry", "country", "lead_source", "seniority", "job_function", "erp_type"]

# %% [markdown]
# ## 3. Logistic regression
#
# It adds up a weight for every feature and turns the total into a probability. Scaling the
# numbers first puts every weight on the same footing: "what one typical step in this feature
# does to the odds".

# %%
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

def logistic(numeric):
    prep = ColumnTransformer([("num", StandardScaler(), numeric),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])
    return make_pipeline(prep, LogisticRegression(max_iter=2000))

cols = NUMERIC + CATEGORICAL
logreg = logistic(NUMERIC).fit(train[cols], train.converted)
logreg_train = catch_rate(logreg.predict_proba(train[cols])[:, 1], train.converted)
logreg_test = catch_rate(logreg.predict_proba(test[cols])[:, 1], test.converted)
print(f"logistic regression: train {logreg_train:.1%}  test {logreg_test:.1%}")

# %% [markdown]
# ## 4. A gradient-boosted tree
#
# Hundreds of small trees, each one trained on the mistakes the previous ones made. The
# settings keep each tree small (two questions) and each step cautious, because with fewer
# than 200 training conversions a boosted tree will happily memorize them.

# %%
from sklearn.ensemble import HistGradientBoostingClassifier

def boosted(numeric):
    prep = ColumnTransformer([("num", "passthrough", numeric),
                              ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL)])
    model = HistGradientBoostingClassifier(learning_rate=0.05, max_depth=2, min_samples_leaf=50,
                                           max_iter=200, random_state=0)
    return make_pipeline(prep, model)

boost = boosted(NUMERIC).fit(train[cols], train.converted)
boost_train = catch_rate(boost.predict_proba(train[cols])[:, 1], train.converted)
boost_test = catch_rate(boost.predict_proba(test[cols])[:, 1], test.converted)
print(f"boosted tree:        train {boost_train:.1%}  test {boost_test:.1%}")

# %% [markdown]
# Both models catch nearly every conversion in the test set. That's far better than anything
# in Module 1, and it should worry you. Find out what they're leaning on.

# %% [markdown]
# ## 5. Which features matter
#
# Permutation importance: shuffle one column in the test set, so it no longer lines up with
# the outcome, and measure how much the model gets worse. A big drop means the model leans on
# that column. It works the same way for any model.

# %%
from sklearn.inspection import permutation_importance

def importance(model, frame, columns):
    result = permutation_importance(model, frame[columns], frame.converted, scoring="average_precision",
                                    n_repeats=10, random_state=0)
    return pd.Series(result.importances_mean, index=columns).sort_values(ascending=False)

boost_importance = importance(boost, test, cols)
boost_importance.head(6).round(3)

# %% [markdown]
# One column carries the whole model. Shuffle anything else and the score barely moves.

# %%
suspicious = "meetings_held"
why_suspicious = """
meetings_held is worth more than every other column combined, and it's the only one that
records something sales does after a lead comes in, rather than something we know when it
arrives. A feature that predicts the outcome this well might be recording the outcome.
"""

# %% [markdown]
# ## 6. The same fight without that column
#
# Module 3 is about exactly what's wrong with `meetings_held`. For now, set it aside and
# compare the two models with the Module 1 rules on equal terms.

# %%
fair_numeric = [c for c in NUMERIC if c != "meetings_held"]
fair_cols = fair_numeric + CATEGORICAL
logreg_fair = logistic(fair_numeric).fit(train[fair_cols], train.converted)
boost_fair = boosted(fair_numeric).fit(train[fair_cols], train.converted)

results = pd.DataFrame({
    "train": [np.nan,
              catch_rate(logreg_fair.predict_proba(train[fair_cols])[:, 1], train.converted),
              catch_rate(boost_fair.predict_proba(train[fair_cols])[:, 1], train.converted),
              logreg_train, boost_train],
    "test": [rules_test,
             catch_rate(logreg_fair.predict_proba(test[fair_cols])[:, 1], test.converted),
             catch_rate(boost_fair.predict_proba(test[fair_cols])[:, 1], test.converted),
             logreg_test, boost_test],
}, index=["Module 1 rules", "logistic regression", "boosted tree",
          "logistic regression + meetings_held", "boosted tree + meetings_held"])
results.round(3)

# %% [markdown]
# Now the importance ranking means something, because no single column is doing all the work.

# %%
fair_importance = importance(boost_fair, test, fair_cols)
print(fair_importance.head(7).round(3).to_string())

# %%
top_five = ["lead_source", "pricing_page_visits", "industry", "job_function", "erp_type"]

# %% [markdown]
# ## 7. Read the logistic regression
#
# Each weight, turned into an odds multiplier. For a number, it's what one typical step up
# (one standard deviation) does to the odds of converting. For a category, it's that value
# against the average of the others.

# %%
names = logreg_fair[0].get_feature_names_out()
odds = pd.Series(np.exp(logreg_fair[-1].coef_[0]), index=names).sort_values()
print("biggest boosts to the odds:\n", odds.tail(6).round(2).to_string())
print("\nbiggest drags on the odds:\n", odds.head(4).round(2).to_string())
print("\nnumeric features, per one standard deviation:\n", odds[[n for n in names if n.startswith("num__")]].sort_values().round(2).to_string())

# %% [markdown]
# ## 8. What one prediction costs
#
# Time how long the boosted tree takes to score 1,000 leads, then turn that into money with a
# compute price. The price is your assumption: put in what an hour of one CPU core costs you.

# %%
import time

COMPUTE_PRICE_PER_HOUR = 0.10  # your assumption, in USD per CPU-core-hour

runs = 20
start = time.perf_counter()
for _ in range(runs):
    boost_fair.predict_proba(test[fair_cols])
seconds_per_prediction = (time.perf_counter() - start) / (runs * len(test))
cost_per_prediction = seconds_per_prediction * COMPUTE_PRICE_PER_HOUR / 3600
print(f"{seconds_per_prediction * 1e6:.1f} microseconds per lead")
print(f"${cost_per_prediction:.2e} per lead, ${cost_per_prediction * 1_000_000:.4f} per million leads")

# %% [markdown]
# Now the same job done by a language model: one call per lead. In mock mode (no key) the
# cost is $0 and the reply is canned. With a real key, the log shows what the call cost.

# %%
from lib import llm

lead = test.iloc[0]
prompt = (f"Score this lead from 0 to 100 for how likely they are to buy inventory-forecasting software. "
          f"Industry: {lead.industry}. Employees: {lead.employees}. Source: {lead.lead_source}. "
          f"Seniority: {lead.seniority}. Pricing page visits: {lead.pricing_page_visits}. Reply with a number.")
reply = llm.chat(prompt)
print(reply.text)
print(f"tokens in/out: {reply.input_tokens}/{reply.output_tokens}, cost: {reply.cost_usd}")
llm_cost_per_lead = reply.cost_usd
if llm_cost_per_lead:
    print(f"one model call costs {llm_cost_per_lead / cost_per_prediction:,.0f} times one tree prediction")
else:
    print("mock mode: no real cost logged. With a real key, rerun this cell and compare.")

# %% [markdown]
# ## Done-when checks

# %%
from lib.checks import done_when

assert done_when([
    ("Both models scored on the same held-out test set",
     results.loc[["logistic regression", "boosted tree"], "test"].notna().all()),
    ("You named five features, all in the fair model's top six",
     len(set(top_five)) == 5 and set(top_five) <= set(fair_importance.index[:6])),
    ("You flagged one feature as suspicious and said why", suspicious in cols and len(why_suspicious.split()) >= 15),
    ("Per-prediction cost computed from your own timing", cost_per_prediction > 0),
])
