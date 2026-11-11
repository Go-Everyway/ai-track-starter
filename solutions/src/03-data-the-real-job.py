# %% [markdown]
# ## 1. Look before you touch anything
#
# This is the same 4,000 leads as `leads_clean.csv`, the way a CRM actually exports them.
# Count the problems first. Every fix below is a decision, and `decisions` keeps a record of
# each one so someone else can check your work.

# %%
messy = pd.read_csv(DATA / "leads_messy.csv", dtype=str)
decisions = []

print(f"{len(messy):,} rows, {messy.shape[1]} columns")
print("\nblank cells per column:")
print(messy.isna().sum()[messy.isna().sum() > 0].to_string())
print(f"\nindustry has {messy.industry.nunique()} different spellings")
print(f"country has {messy.country.nunique()} different spellings")
print(f"lead_source has {messy.lead_source.nunique()} different spellings")

# The raw file's numbers, kept for the module page (recorded at the end).
raw_stats = {
    "raw_rows": len(messy),
    "raw_blanks": messy.isna().sum()[messy.isna().sum() > 0].to_dict(),
    "raw_rows_with_any_blank": int(messy[["industry", "employees", "country", "pages_viewed", "erp_type"]].isna().any(axis=1).sum()),
    "raw_industry_spellings": messy.industry.nunique(), "raw_country_spellings": messy.country.nunique(),
    "raw_source_spellings": messy.lead_source.nunique(),
    "raw_employee_ranges": int(messy.employees.str.contains("-", na=False).sum()),
    "raw_employee_commas": int(messy.employees.str.contains(",", na=False).sum()),
}
print(f"{raw_stats['raw_rows_with_any_blank']} rows have at least one blank")

# %% [markdown]
# ## OPERATOR: change one value
#
# `OPERATOR_VALUE` decides what happens to rows with blanks in step 7. `"fill"` (the default)
# keeps every row and fills the gaps; `"drop"` throws away any row with a blank. Switch it to
# `"drop"` (keep the quotes), then choose Runtime → Run all, and compare the row count after
# step 7 and the scores in step 9.
#
# The done-when checks at the bottom were written for the default value, so a ✗ there after
# you change it is expected, not a mistake.

# %%
OPERATOR_VALUE = "fill"  # "fill" or "drop". Change this, then Runtime → Run all
DEFAULT_VALUE = "fill"

# %% [markdown]
# ## 2. Personal data first
#
# The export carries contact names and emails. A model doesn't need either. The email is still
# useful for one job, spotting the same person entered twice, so swap it for a salted hash
# (a stand-in that matches itself but can't be read back), then drop both raw columns.
# Do this before the data goes anywhere else, including any AI tool.

# %%
import hashlib

SALT = "change-me-and-keep-it-out-of-git"
messy["contact_key"] = messy.contact_email.str.strip().str.lower().map(
    lambda e: hashlib.sha256((SALT + e).encode()).hexdigest()[:16]
)
messy = messy.drop(columns=["contact_name", "contact_email"])
decisions.append("pii: replaced contact_email with a salted hash (contact_key) for dedupe; dropped contact_name and contact_email")

# %% [markdown]
# ## 3. Dates in three formats
#
# `2025-03-14`, `03/14/2025` and `14 Mar 2025` all mean the same day. Parse each format
# explicitly instead of letting pandas guess, because a guess can swap day and month.

# %%
def parse_date(s):
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d %b %Y"):
        try:
            return pd.to_datetime(s, format=fmt)
        except (ValueError, TypeError):
            pass
    return pd.NaT

messy["created_date"] = messy.created_date.map(parse_date)
print("unparsed dates:", messy.created_date.isna().sum())
decisions.append("dates: parsed ISO, US (mm/dd/yyyy) and '14 Mar 2025' formats explicitly; no guessing")

# %% [markdown]
# ## 4. Duplicates
#
# The same contact shows up again under a new lead id, with the company typed a little
# differently. Match on the contact key, keep the first entry (the lowest lead id), drop the
# rest. Do it before you split: a duplicate that lands in both train and test lets the model
# see test answers during training.

# %%
from sklearn.model_selection import train_test_split

dupes = messy.contact_key.duplicated(keep=False)
print(f"{messy.contact_key.duplicated().sum()} extra rows belong to contacts already in the file")
print(messy[dupes].sort_values("contact_key")[["lead_id", "company", "lead_source", "converted"]].head(6).to_string())

# What would happen without dedupe: how many duplicate pairs straddle a random split?
tr, te = train_test_split(messy, test_size=0.25, random_state=0, stratify=messy.converted)
straddle = len(set(tr.contact_key) & set(te.contact_key))
print(f"\nwithout dedupe, {straddle} contacts would sit in both train and test")

messy = messy.sort_values("lead_id").drop_duplicates("contact_key", keep="first").reset_index(drop=True)
print(f"after dedupe: {len(messy):,} rows")
decisions.append(f"duplicates: matched on contact_key, kept the lowest lead_id, dropped {4120 - len(messy)} rows; done before the split")

# %% [markdown]
# ## 5. One spelling per category
#
# Normalize case and spacing, then map every variant to one name. Write the map out by hand.
# It's tedious, and it's the only way to be sure "Consulting" really means professional services.

# %%
INDUSTRY = {
    "wholesale distribution": "Wholesale distribution", "wholesale": "Wholesale distribution", "distribution": "Wholesale distribution",
    "manufacturing": "Manufacturing", "mfg": "Manufacturing", "manufacturer": "Manufacturing",
    "food & beverage": "Food & beverage", "food and beverage": "Food & beverage", "f&b": "Food & beverage",
    "logistics": "Logistics", "3pl": "Logistics", "logistics & warehousing": "Logistics",
    "retail": "Retail", "retail - consumer": "Retail", "ecommerce": "Retail",
    "healthcare": "Healthcare", "health care": "Healthcare", "medical": "Healthcare",
    "construction": "Construction", "construction & trades": "Construction",
    "professional services": "Professional services", "prof. services": "Professional services", "consulting": "Professional services",
    "software": "Software", "saas": "Software", "tech - software": "Software",
    "education": "Education", "edtech": "Education", "training": "Education",
}
COUNTRY = {
    "united states": "United States", "usa": "United States", "us": "United States", "u.s.": "United States",
    "united kingdom": "United Kingdom", "uk": "United Kingdom", "u.k.": "United Kingdom", "great britain": "United Kingdom", "england": "United Kingdom",
    "canada": "Canada", "ca": "Canada", "germany": "Germany", "de": "Germany", "deutschland": "Germany",
    "australia": "Australia", "au": "Australia",
}
SOURCE = {
    "inbound_demo": "inbound_demo", "inbound demo": "inbound_demo", "inbound-demo": "inbound_demo", "demo request": "inbound_demo",
    "pricing_page": "pricing_page", "pricing page": "pricing_page", "pricing": "pricing_page",
    "content_download": "content_download", "content download": "content_download", "ebook": "content_download", "content": "content_download",
    "outbound": "outbound", "cold email": "outbound", "sdr outbound": "outbound",
    "webinar": "webinar", "event": "event", "trade show": "event",
    "referral": "referral", "partner referral": "referral",
}

def normalize(col, mapping):
    key = messy[col].str.strip().str.lower()
    unmapped = sorted(set(key.dropna()) - set(mapping))
    assert not unmapped, f"{col}: no mapping for {unmapped}"
    return key.map(mapping)

for col, mapping in [("industry", INDUSTRY), ("country", COUNTRY), ("lead_source", SOURCE)]:
    messy[col] = normalize(col, mapping)
    print(f"{col}: {messy[col].nunique()} values")
decisions.append("categories: lower-cased, then mapped every variant of industry, country and lead_source to one name by hand; the code fails on any variant it hasn't seen")

# %% [markdown]
# ## 6. Numbers stored as text, and numbers that can't be true
#
# `employees` arrives as `97`, `1,200` or a range like `51-200`. Strip the commas, take the
# middle of a range, and treat 999,999 employees as a typo, not a giant company.
# `pages_viewed` has a couple of `-1`s, which can't happen.

# %%
def parse_employees(v):
    if pd.isna(v):
        return float("nan")
    v = str(v).replace(",", "").strip()
    if "-" in v:
        lo, hi = v.split("-")
        return (float(lo) + float(hi)) / 2
    return float(v)

messy["employees"] = messy.employees.map(parse_employees)
impossible = messy.employees > 100_000
print(f"{impossible.sum()} impossible employee counts set to blank")
messy.loc[impossible, "employees"] = float("nan")
decisions.append("employees: removed thousands separators, used the midpoint of ranges, set counts over 100,000 to blank as entry errors")

for c in ["pages_viewed", "pricing_page_visits", "email_opens", "email_clicks", "webinar_attended",
          "hiring_ops_roles", "recent_funding", "meetings_held", "converted"]:
    messy[c] = pd.to_numeric(messy[c])
bad_pages = messy.pages_viewed < 0
messy.loc[bad_pages, "pages_viewed"] = float("nan")
print(f"{bad_pages.sum()} negative page counts set to blank")
decisions.append("pages_viewed: negative values set to blank as entry errors")

# %% [markdown]
# ## 7. Blanks: pick a fix and know what it assumes
#
# - **Drop the row:** assumes blanks are random. They rarely are, and here you'd lose 997 of 4,120 rows.
# - **Fill numbers with the median:** assumes a blank lead is typical. Add a `_missing` flag so the
#   model can still learn if blank-ness itself means something.
# - **Fill categories with "unknown":** says "we don't know" out loud instead of guessing.

# %%
BLANKABLE = ["industry", "employees", "country", "pages_viewed", "erp_type"]
print("blanks before:", messy[BLANKABLE].isna().sum().to_dict())
if OPERATOR_VALUE == "drop":
    before = len(messy)
    messy = messy.dropna(subset=BLANKABLE).reset_index(drop=True)
    for c in ["employees", "pages_viewed"]:
        messy[f"{c}_missing"] = 0
    print(f"dropped {before - len(messy)} rows with a blank; {len(messy):,} left")
    decisions.append(f"missing values: dropped every row with a blank ({before - len(messy)} rows)")
else:
    for c in ["employees", "pages_viewed"]:
        messy[f"{c}_missing"] = messy[c].isna().astype(int)
        messy[c] = messy[c].fillna(messy[c].median())
    for c in ["industry", "country", "erp_type"]:
        messy[c] = messy[c].fillna("unknown")
    decisions.append("missing numbers: median fill plus a _missing flag column; missing categories: 'unknown' as its own value")
print("blanks after:", int(messy.isna().sum().sum()), f"| rows: {len(messy):,}")

# %% [markdown]
# ## 8. Hunt for leakage
#
# Score every numeric column on its own with Module 1's catch rate. A real signal catches
# maybe 20% to 30% of conversions by itself. Anything near 100% is too good to be true,
# and too good to be true almost always means the column was recorded after the outcome.

# %%
import numpy as np

def catch_rate(scores, converted):
    """Share of all conversions that land in the top-scored 10% of leads."""
    scores, converted = np.asarray(scores, dtype=float), np.asarray(converted)
    k = max(1, len(scores) // 10)
    tiebreak = np.random.default_rng(0).random(len(scores))
    order = np.lexsort((tiebreak, -scores))
    return converted[order[:k]].sum() / converted.sum()

numeric = ["employees", "pages_viewed", "pricing_page_visits", "email_opens", "email_clicks",
           "webinar_attended", "hiring_ops_roles", "recent_funding", "meetings_held"]
alone = pd.Series({c: catch_rate(messy[c], messy.converted) for c in numeric}).sort_values(ascending=False)
print(alone.map("{:.1%}".format).to_string())

# %%
print(pd.crosstab(messy.meetings_held, messy.converted, margins=True))

# %%
leaking_column = "meetings_held"
why_it_leaks = """
Meetings get logged as a deal moves forward, so a converted lead almost always has one or more
and an unconverted lead almost never does. At the moment you'd score a new lead, nobody has met
them yet, so this column records the outcome instead of predicting it.
"""

# %% [markdown]
# ## 9. Retrain, with and without the leak
#
# Same split as Module 1. Logistic regression this time, which needs its numbers on similar
# scales, so standardize them. `class_weight="balanced"` tells it to care about the rare
# converters as much as the common non-converters.

# %%
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORIES = ["industry", "country", "lead_source", "seniority", "job_function", "erp_type"]
ALL_NUMBERS = ["employees", "pages_viewed", "pricing_page_visits", "email_opens", "email_clicks",
               "webinar_attended", "hiring_ops_roles", "recent_funding", "meetings_held",
               "employees_missing", "pages_viewed_missing"]

def score(train, test, numbers):
    model = make_pipeline(
        ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORIES),
                           ("num", StandardScaler(), numbers)]),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    ).fit(train, train.converted)
    return catch_rate(model.predict_proba(test)[:, 1], test.converted)

train, test = train_test_split(messy, test_size=0.25, random_state=0, stratify=messy.converted)
with_leak = score(train, test, ALL_NUMBERS)
without_leak = score(train, test, [c for c in ALL_NUMBERS if c != leaking_column])
results = pd.Series({"every column": with_leak, f"without {leaking_column or '(nothing yet)'}": without_leak})
print(results.map("{:.1%}".format).to_string())
decisions.append(f"leakage: dropped {leaking_column} from the features")

# %%
why_lower_is_true = """
The higher score used a column that only exists after the deal is already moving, so it can't
be reproduced on a new lead. The lower score uses only what you know when the lead arrives,
which is the only number that tells you what the model will do next month.
"""

# %% [markdown]
# ## 10. Your record

# %%
for i, d in enumerate(decisions, 1):
    print(f"{i}. {d}")

# %% [markdown]
# ## Done-when checks

# %%
from lib.checks import done_when

checks = [
    ("No contact names or emails left in the data", not {"contact_name", "contact_email"} & set(messy.columns)),
    ("Every contact appears once", not messy.contact_key.duplicated().any()),
    ("One spelling per category (10 industries plus 'unknown')", messy.industry.nunique() == 11 and messy.lead_source.nunique() == 7),
    ("No blanks, no impossible values", messy.isna().sum().sum() == 0 and messy.employees.max() <= 100_000 and messy.pages_viewed.min() >= 0),
    ("Every cleaning choice is written down", sum(not d.startswith("TODO") for d in decisions) >= 8),
    ("You found the leaking column", leaking_column == "meetings_held"),
    ("You explained why it leaks", len(why_it_leaks.split()) >= 20),
    ("Before and after scores, and the leak inflated the score", with_leak > without_leak),
    ("You explained why the lower score is the true one", len(why_lower_is_true.split()) >= 20),
]
if OPERATOR_VALUE == DEFAULT_VALUE:
    assert done_when(checks)
    from lib.figures import record
    table = pd.crosstab(messy.meetings_held, messy.converted)
    two_plus = table[table.index >= 2]
    record("m03", **raw_stats,
           extra_duplicate_rows=4120 - len(messy), contacts_straddling_split=straddle, rows_after_dedupe=len(messy),
           impossible_employee_counts=int(impossible.sum()), negative_page_counts=int(bad_pages.sum()),
           catch_rate_alone=alone.round(4).to_dict(),
           zero_meetings_leads=int(table.loc[0].sum()), zero_meetings_converted=int(table.loc[0, 1]),
           two_plus_meetings_leads=int(two_plus.values.sum()), two_plus_meetings_converted=int(two_plus[1].sum()),
           converted=int(messy.converted.sum()), not_converted=int((messy.converted == 0).sum()),
           score_with_leak=with_leak, score_without_leak=without_leak)
else:
    _ = done_when(checks)
