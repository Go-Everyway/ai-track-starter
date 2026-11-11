# %% [markdown]
# ## 1. Same data, same split as Module 1
#
# 4,000 synthetic leads, 6.3% converted. The split is identical to Module 1's, so the test
# set is the same 1,000 leads with the same 63 conversions.

# %%
import numpy as np
from sklearn.model_selection import train_test_split

leads = pd.read_csv(DATA / "leads_clean.csv")
train, test = train_test_split(leads, test_size=0.25, random_state=0, stratify=leads.converted)
y_test = test.converted.to_numpy()
print(f"test set: {len(test):,} leads, {y_test.sum()} converted ({y_test.mean():.1%})")

# %% [markdown]
# ## OPERATOR: change one value
#
# `OPERATOR_VALUE` is how many leads your team calls each week: the "top k" the confusion
# matrix, the confidence range and the meetings estimate are built on. The default is 50.
# Try anything from 10 to 200, then choose Runtime → Run all, and watch precision, recall and
# the meetings sentence at the end. Calling more leads finds more buyers but wastes more calls.
#
# The done-when checks at the bottom were written for the default value, so a ✗ there after
# you change it is expected, not a mistake.

# %%
OPERATOR_VALUE = 50  # leads called per week, 10 to 200. Change this, then Runtime → Run all
DEFAULT_VALUE = 50
K = OPERATOR_VALUE

# %% [markdown]
# ## 2. The model that's right 94% of the time and useless
#
# Predict that nobody converts. Count how often that's right.

# %%
nobody = np.zeros(len(test), dtype=int)
accuracy_nobody = (nobody == y_test).mean()
print(f"'nobody converts' accuracy: {accuracy_nobody:.1%}, buyers found: {(nobody & y_test).sum()}")

# %% [markdown]
# ## 3. The model to evaluate
#
# The honest model from Module 3: logistic regression on everything we know about a lead
# before the outcome, without the leaked `meetings_held` column.
#
# We need to pick one setting (`C`, how hard the model is held back from fitting noise).
# Tuning it on the test set would leak the test set into our choices, so we carve a
# **validation set** out of the training data, tune on that, then retrain on all of the
# training data. The test set is used once, at the end.

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["industry", "employees", "country", "lead_source", "seniority", "job_function",
            "pages_viewed", "pricing_page_visits", "email_opens", "email_clicks",
            "webinar_attended", "erp_type", "hiring_ops_roles", "recent_funding"]

def features(df, columns=None):
    X = pd.get_dummies(df[FEATURES], dtype=float)
    return X if columns is None else X.reindex(columns=columns, fill_value=0)

def model_for(C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))

fit_part, valid = train_test_split(train, test_size=0.25, random_state=0, stratify=train.converted)
X_fit = features(fit_part)
validation = {}
for C in [0.01, 0.1, 1.0, 10.0]:
    m = model_for(C).fit(X_fit, fit_part.converted)
    validation[C] = average_precision_score(valid.converted, m.predict_proba(features(valid, X_fit.columns))[:, 1])
    print(f"C={C:<5} validation average precision {validation[C]:.3f}")

# %%
BEST_C = max(validation, key=validation.get)  # the best validation score above
print("using C =", BEST_C)
X_train = features(train)
model = model_for(BEST_C).fit(X_train, train.converted)
model_scores = model.predict_proba(features(test, X_train.columns))[:, 1]

# %% [markdown]
# ## 4. Baselines: random, and Module 1's rules

# %%
GOOD_FIT = {"Wholesale distribution", "Manufacturing", "Food & beverage", "Logistics"}

def rule_score(row):
    score = 0
    score += row.lead_source in {"inbound_demo", "pricing_page", "referral"}
    score += row.industry in GOOD_FIT and 50 <= row.employees <= 1000
    score += row.pricing_page_visits >= 2
    return score

rule_scores = test.apply(rule_score, axis=1).to_numpy()
random_scores = np.zeros(len(test))  # all tied, so the ranking is random

def ranked(scores):
    """Lead positions, best score first. Ties are broken at random, the same way every time."""
    tiebreak = np.random.default_rng(0).random(len(scores))
    return np.lexsort((tiebreak, -np.asarray(scores, dtype=float)))

# %% [markdown]
# ## 5. The confusion matrix for "call the top K"
#
# Your team works K leads a week (50 by default). Treat the 1,000 test leads as the backlog and
# call the model's top K. Every lead lands in one of four boxes.

# %%
def confusion_top_k(scores, y, k):
    flagged = np.zeros(len(y), dtype=bool)
    flagged[ranked(scores)[:k]] = True
    tp = int((flagged & (y == 1)).sum())
    fp = int((flagged & (y == 0)).sum())
    fn = int((~flagged & (y == 1)).sum())
    tn = int((~flagged & (y == 0)).sum())
    return tp, fp, fn, tn

tp, fp, fn, tn = confusion_top_k(model_scores, y_test, K)
precision = tp / (tp + fp) if tp + fp else 0.0
recall = tp / (tp + fn) if tp + fn else 0.0
f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
print(pd.DataFrame([[tp, fn], [fp, tn]], index=["actually converted", "didn't convert"],
                   columns=[f"called (top {K})", "not called"]))
print(f"\nprecision {precision:.1%}   recall {recall:.1%}   F1 {f1:.2f}   accuracy {(tp + tn) / len(y_test):.1%}")

# %% [markdown]
# ## 6. Thresholds: the trade-off
#
# Instead of "top 50", you can flag every lead whose predicted chance is above a cutoff.
# Lower the cutoff and you flag more leads: recall goes up, precision goes down.

# %%
rows = []
for t in [0.05, 0.10, 0.20, 0.30, 0.50]:
    flag = model_scores >= t
    tp_t = int((flag & (y_test == 1)).sum())
    rows.append({"cutoff": t, "flagged": int(flag.sum()),
                 "precision": tp_t / flag.sum() if flag.sum() else float("nan"),
                 "recall": tp_t / y_test.sum()})
thresholds = pd.DataFrame(rows)
thresholds.round(3)

# %% [markdown]
# ## 7. Precision@k against the baselines
#
# Precision@k: of the top k leads, what share converted. Lift: how many times better than
# the base rate. Module 1's catch rate is recall at k = 100.

# %%
def precision_at_k(scores, y, k):
    return y[ranked(scores)[:k]].mean()

KS = [25, 50, 100]
base_rate = y_test.mean()
p_at_k = pd.DataFrame({
    name: [precision_at_k(s, y_test, k) for k in KS]
    for name, s in [("random", random_scores), ("rules", rule_scores), ("model", model_scores)]
}, index=[f"precision@{k}" for k in KS])
lift = (p_at_k / base_rate).add_prefix("lift ")
pd.concat([p_at_k, lift], axis=1).round(2)

# %% [markdown]
# ## 8. How sure are we? Two ways to see the spread
#
# K leads is a small sample. First, resample the test set with replacement 1,000 times
# (a bootstrap) and see how precision@K moves. Second, redo the whole split ten times.

# %%
rng = np.random.default_rng(1)
boot = {"model": [], "rules": [], "model minus rules": []}
for _ in range(1000):
    idx = rng.integers(0, len(y_test), len(y_test))
    m = precision_at_k(model_scores[idx], y_test[idx], K)
    r = precision_at_k(rule_scores[idx], y_test[idx], K)
    boot["model"].append(m); boot["rules"].append(r); boot["model minus rules"].append(m - r)
spread = pd.DataFrame({k: np.percentile(v, [5, 50, 95]) for k, v in boot.items()}, index=["5th pct", "median", "95th pct"])
spread.round(3)

# %%
splits = []
for seed in range(10):
    tr, te = train_test_split(leads, test_size=0.25, random_state=seed, stratify=leads.converted)
    m = model_for(BEST_C).fit(features(tr), tr.converted)
    y = te.converted.to_numpy()
    splits.append({"seed": seed,
                   "model": precision_at_k(m.predict_proba(features(te, features(tr).columns))[:, 1], y, K),
                   "rules": precision_at_k(te.apply(rule_score, axis=1).to_numpy(), y, K)})
splits = pd.DataFrame(splits).set_index("seed")
print(splits.round(2).T)
print(f"\nmodel ahead of rules in {(splits.model > splits.rules).sum()} of 10 splits; "
      f"average precision@{K} model {splits.model.mean():.1%}, rules {splits.rules.mean():.1%}")

# %% [markdown]
# ## 9. Calibration: does "20% likely" mean 20%?

# %%
buckets = pd.qcut(model_scores, 5, labels=["lowest fifth", "2nd", "3rd", "4th", "highest fifth"])
pd.DataFrame({"avg predicted": model_scores, "actually converted": y_test}).groupby(buckets, observed=True).mean().round(3)

# %% [markdown]
# ## 10. The business number
#
# Turn precision@K into meetings a week. The two rates below are **assumptions**, not facts
# from this data. Write yours down and say where they came from.

# %%
LEADS_PER_WEEK = K
MEETING_RATE_BUYERS = 0.60     # assumption: share of eventual buyers who book a meeting when called
MEETING_RATE_OTHERS = 0.05     # assumption: share of everyone else who books one anyway

def meetings_per_week(p):
    return LEADS_PER_WEEK * (p * MEETING_RATE_BUYERS + (1 - p) * MEETING_RATE_OTHERS)

p_k = {name: precision_at_k(s, y_test, K) for name, s in
       [("random", random_scores), ("rules", rule_scores), ("model", model_scores)]}
meetings = {name: meetings_per_week(p) for name, p in p_k.items()}
print({k: round(v, 1) for k, v in meetings.items()})

business_sentence = (
    f"Calling the model's top {LEADS_PER_WEEK} leads a week should book about {meetings['model']:.0f} meetings, "
    f"against about {meetings['random']:.0f} from a random {LEADS_PER_WEEK}, assuming {MEETING_RATE_BUYERS:.0%} of "
    f"eventual buyers and {MEETING_RATE_OTHERS:.0%} of everyone else book a meeting, and knowing precision@{K} "
    f"swings between about {spread.loc['5th pct', 'model']:.0%} and {spread.loc['95th pct', 'model']:.0%}."
)
print(business_sentence)

# %% [markdown]
# ## Done-when checks

# %%
from lib.checks import done_when

checks = [
    (f"Confusion matrix for the top {K} adds up to the whole test set", tp + fp + fn + tn == len(y_test) and tp + fp == K),
    ("Precision@k for three values of k", len(p_at_k) == 3 and p_at_k.notna().all().all()),
    ("Compared against random and rule-based baselines", {"random", "rules", "model"} <= set(p_at_k.columns)),
    ("The model beats random at k = 50", p_at_k.loc["precision@50", "model"] > p_at_k.loc["precision@50", "random"]),
    ("One sentence with the business number and its assumptions",
     "assum" in business_sentence and any(ch.isdigit() for ch in business_sentence)),
]
if OPERATOR_VALUE == DEFAULT_VALUE:
    assert done_when(checks)
    from lib.figures import record
    calib = pd.DataFrame({"pred": model_scores, "actual": y_test}).groupby(buckets, observed=True).mean()
    record("m04",
           test_leads=len(test), test_converted=int(y_test.sum()), base_rate=base_rate,
           accuracy_nobody_converts=accuracy_nobody, validation_average_precision={str(c): v for c, v in validation.items()},
           best_c=BEST_C,
           top50_tp=tp, top50_fp=fp, top50_fn=fn, top50_tn=tn,
           top50_precision=precision, top50_recall=recall, top50_f1=f1, top50_accuracy=(tp + tn) / len(y_test),
           thresholds=thresholds.round(4).astype(object).where(thresholds.notna(), None).to_dict(orient="records"),
           precision_at_k=p_at_k.round(4).to_dict(), lift=lift.round(3).to_dict(),
           bootstrap_p50=spread.round(4).to_dict(),
           splits_model_ahead=int((splits.model > splits.rules).sum()),
           splits_model_mean=splits.model.mean(), splits_rules_mean=splits.rules.mean(),
           calibration_lowest_pred=calib.pred.iloc[0], calibration_lowest_actual=calib.actual.iloc[0],
           calibration_highest_pred=calib.pred.iloc[-1], calibration_highest_actual=calib.actual.iloc[-1],
           meetings_per_week=meetings)
else:
    _ = done_when(checks)
