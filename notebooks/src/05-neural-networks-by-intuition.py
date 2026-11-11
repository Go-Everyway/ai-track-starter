# %% [markdown]
# ## 1. Look at the companies
#
# 400 synthetic company profiles. 40 are marked `is_customer`: Tallyhawk's best accounts.
# The job: given any company, find the ones most like it, and use that to find prospects
# that look like the customers.

# %%
import numpy as np
from lib import llm

companies = pd.read_json(DATA / "companies.jsonl", lines=True)
print(f"{len(companies)} companies, {companies.is_customer.sum()} customers")
print("embedding mode:", "mock (word-hash vectors, mechanics only)" if llm._settings()["provider"] == "mock" else llm._settings()["embed_model"])
companies[["company_id", "name", "industry", "employees", "description"]].head()

# %% [markdown]
# ## 2. Cosine similarity by hand
#
# Three tiny made-up vectors. Cosine similarity is the dot product divided by the two lengths.
# 1 means pointing the same way, 0 means unrelated.

# %%
a = np.array([3.0, 1.0, 0.0])   # say: a distributor
b = np.array([2.0, 1.0, 0.0])   # another distributor
c = np.array([0.0, 1.0, 3.0])   # a dental clinic

def cos(u, v):
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))

print(f"a vs b: {cos(a, b):.3f}")
print(f"a vs c: {cos(a, c):.3f}")
print(f"b vs c: {cos(b, c):.3f}")

# %% [markdown]
# ## 3. Embed every description, once
#
# One call per batch of 50. Results are cached in `logs/`, so rerunning the notebook doesn't
# pay again. The cache key includes the model, so switching models re-embeds.

# %%
import hashlib, json

def embed_cached(texts, batch=50):
    s = llm._settings()
    model = "mock" if s["provider"] == "mock" else s["embed_model"]
    path = ROOT / "logs" / f"embeddings-{hashlib.md5(model.encode()).hexdigest()[:10]}.json"
    cache = json.loads(path.read_text()) if path.exists() else {}
    key = lambda t: hashlib.md5(t.encode()).hexdigest()
    todo = [t for t in dict.fromkeys(texts) if key(t) not in cache]
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        for t, v in zip(chunk, llm.embed(chunk)):
            cache[key(t)] = v
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(cache))
    return np.array([cache[key(t)] for t in texts])

E = embed_cached(companies.description.tolist())
E = E / np.linalg.norm(E, axis=1, keepdims=True)   # unit length, so a dot product is the cosine
print("embeddings:", E.shape)

# %% [markdown]
# ## 4. YOUR TURN: the lookup, accounts like this one

# %%
def lookalikes(company_id, vectors=E, k=10):
    """YOUR TURN: the k companies most similar to `company_id`, best first, excluding itself.

    1. Find the row number i for company_id:  companies.index[companies.company_id == company_id][0]
    2. Score every company against it. The vectors are unit length, so  vectors @ vectors[i]
       gives the cosine similarity of every row to row i in one go.
    3. Knock the company itself out (set its score to -np.inf), sort high to low, keep k.
    """
    out = companies.head(k)[["company_id", "name", "industry", "employees", "is_customer"]].copy()
    out["similarity"] = float("nan")   # replace with real scores
    return out.reset_index(drop=True)

example = companies[companies.is_customer].company_id.iloc[0]
print(companies.set_index("company_id").loc[example, "description"], "\n")
lookalikes(example)

# %% [markdown]
# ## 5. Prospects that look like your best customers
#
# Score each non-customer by its similarity to the closest customer, then rank.

# %%
cust = companies.is_customer.to_numpy()
best_match = (E[~cust] @ E[cust].T).max(axis=1)
prospects = companies[~cust][["company_id", "name", "industry", "employees"]].copy()
prospects["closest_customer_sim"] = best_match.round(3)
prospects = prospects.sort_values("closest_customer_sim", ascending=False).reset_index(drop=True)
prospects.head(10)

# %% [markdown]
# ## 6. YOUR TURN: judge the matches
#
# Read the neighbors for a few companies. A good match is a company a salesperson would
# call "like this one": similar business, similar size, similar situation. Record two bad
# matches and why the embedding put them there.

# %%
for cid in companies.company_id.sample(3, random_state=1):
    row = companies.set_index("company_id").loc[cid]
    print(f"\n=== {row['name']} ({row.industry}, {row.employees} employees)")
    print(lookalikes(cid, k=5)[["name", "industry", "employees", "similarity"]].to_string(index=False))

# %%
# Write down two bad matches you saw above, and why you think the vectors put them together.
# Key: "Company A -> Company B", value: your explanation in a sentence or two.
bad_matches = {
}

# %% [markdown]
# ## 7. Compare with keyword search
#
# TF-IDF scores words by how distinctive they are. It's the classic keyword baseline.
# One rough check for both: of each company's 10 neighbors, what share are in the same industry?

# %%
from sklearn.feature_extraction.text import TfidfVectorizer

T = TfidfVectorizer(stop_words="english").fit_transform(companies.description).toarray()
T = T / np.linalg.norm(T, axis=1, keepdims=True)

def same_industry_rate(vectors, k=10):
    sims = vectors @ vectors.T
    np.fill_diagonal(sims, -np.inf)
    top = np.argsort(-sims, axis=1)[:, :k]
    ind = companies.industry.to_numpy()
    return float((ind[top] == ind[:, None]).mean())

random_rate = float((companies.industry.value_counts(normalize=True) ** 2).sum())
comparison = pd.Series({
    "embeddings": same_industry_rate(E),
    "TF-IDF keywords": same_industry_rate(T),
    "random neighbors": random_rate,
}).round(3)
comparison

# %%
# YOUR TURN: when would plain keyword search have beaten embeddings here? Think about what
# a salesperson might search for that has to match exactly.
keyword_wins = """
"""

# %% [markdown]
# ## Done-when checks

# %%
from lib.checks import done_when

def lookup_works(cid):
    r = lookalikes(cid)
    return (len(r) == 10 and cid not in set(r.company_id)
            and r.similarity.notna().all() and r.similarity.is_monotonic_decreasing)

_ = done_when([
    ("The lookup returns ranked neighbors for any company",
     all(lookup_works(c) for c in companies.company_id.sample(5, random_state=0))),
    ("You compared embeddings with a keyword baseline", {"embeddings", "TF-IDF keywords"} <= set(comparison.index)),
    ("You explained at least two bad matches",
     len(bad_matches) >= 2 and all(len(v.split()) >= 8 for v in bad_matches.values())),
    ("You wrote down when keyword search would do better", len(keyword_wins.split()) >= 20),
])
