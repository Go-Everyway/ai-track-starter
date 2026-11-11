"""Generate every dataset in data/ from a fixed seed.

Everything here is synthetic. Company names, contact names and documents are assembled
from random syllables and templates; emails use the reserved .example domain. Nothing
describes a real person or business. Running this script twice gives byte-identical files.

Standard library only, on purpose: the output can't drift with numpy or pandas versions.

    python tools/generate_data.py            # writes data/
    python tools/generate_data.py --check    # regenerate in memory, fail if data/ differs
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import random
import sys
from datetime import date, timedelta
from pathlib import Path

SEED = 20261111
N_LEADS = 4000
TARGET_RATE = 0.06  # share of leads that convert, roughly
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

SELLER = "Tallyhawk"  # the fictional company whose lead engine the course builds

# ---------------------------------------------------------------- name parts

SYL_A = ["bren", "cal", "dor", "fen", "gal", "hal", "jor", "kel", "lum", "mar", "nor", "ost",
         "pel", "quen", "ral", "sor", "tam", "ul", "ver", "wen", "yar", "zel", "ash", "brin"]
SYL_B = ["a", "e", "i", "o", "u", "ar", "en", "is", "on", "ur", "el", "ix"]
SYL_C = ["vo", "dex", "ra", "lin", "mont", "ford", "wick", "ton", "ley", "dale", "mere",
         "stone", "brook", "field", "gate", "haven", "core", "line", "wise", "path"]
FIRST = ["Ari", "Bex", "Cato", "Dara", "Emry", "Fen", "Gale", "Hollis", "Ines", "Jory", "Kit",
         "Lior", "Mika", "Noor", "Oren", "Pell", "Quin", "Rhea", "Sol", "Tovi", "Ula", "Vesper",
         "Wren", "Yael", "Zuri"]

INDUSTRIES = {
    # name: (fit weight for Tallyhawk, suffixes, what they handle)
    "Wholesale distribution": (1.0, ["Supply", "Distributors", "Wholesale", "Trading Co."],
                               ["industrial fasteners", "plumbing parts", "packaging materials", "janitorial supplies", "electrical components", "auto parts"]),
    "Manufacturing": (0.8, ["Manufacturing", "Works", "Fabrication", "Industries"],
                      ["machined metal parts", "injection-molded housings", "commercial furniture", "HVAC components", "printed circuit boards", "precision valves"]),
    "Food & beverage": (0.6, ["Foods", "Provisions", "Beverage Co.", "Kitchens"],
                        ["specialty sauces", "frozen bakery goods", "craft beverages", "snack foods", "dairy alternatives", "prepared meals"]),
    "Logistics": (0.5, ["Logistics", "Freight", "Fulfillment", "Transport"],
                  ["third-party fulfillment", "cold-chain storage", "regional freight", "last-mile delivery", "cross-docking", "returns processing"]),
    "Retail": (0.3, ["Outfitters", "Goods", "Market", "Store"],
               ["outdoor gear", "home goods", "pet supplies", "sporting equipment", "kitchenware", "garden supplies"]),
    "Healthcare": (-0.4, ["Health", "Clinics", "Medical", "Care"],
                   ["outpatient clinics", "dental practices", "physical therapy", "home health services", "urgent care", "diagnostic imaging"]),
    "Construction": (-0.2, ["Builders", "Construction", "Contracting", "Group"],
                     ["commercial interiors", "roofing", "site preparation", "residential remodeling", "concrete work", "steel erection"]),
    "Professional services": (-0.7, ["Partners", "Advisors", "Consulting", "Associates"],
                              ["tax advisory", "management consulting", "legal services", "engineering design", "recruiting", "architecture"]),
    "Software": (-0.8, ["Labs", "Software", "Systems", "Digital"],
                 ["payroll software", "scheduling apps", "analytics dashboards", "e-signature tools", "learning platforms", "CRM add-ons"]),
    "Education": (-0.9, ["Academy", "Learning", "Institute", "Schools"],
                  ["vocational training", "online courses", "tutoring services", "corporate training", "test prep", "language schools"]),
}
INDUSTRY_SHARE = [0.16, 0.15, 0.09, 0.08, 0.10, 0.09, 0.08, 0.09, 0.09, 0.07]

COUNTRIES = [("United States", 0.60), ("Canada", 0.12), ("United Kingdom", 0.12), ("Germany", 0.08), ("Australia", 0.08)]
SOURCES = [("inbound_demo", 0.08, 1.4), ("pricing_page", 0.07, 0.9), ("referral", 0.06, 1.0),
           ("webinar", 0.12, 0.3), ("content_download", 0.25, 0.0), ("event", 0.10, 0.2),
           ("outbound", 0.32, -0.3)]
SENIORITY = [("c_level", 0.08, 0.2), ("vp", 0.12, 0.4), ("director", 0.22, 0.45),
             ("manager", 0.33, 0.1), ("individual", 0.25, -0.4)]
FUNCTIONS = [("operations", 0.24, 0.5), ("supply_chain", 0.16, 0.6), ("finance", 0.16, 0.2),
             ("it", 0.14, 0.0), ("sales", 0.15, -0.2), ("marketing", 0.15, -0.3)]
ERP = [("none", 0.30, -0.3), ("legacy", 0.40, 0.2), ("cloud", 0.30, 0.4)]

LEAD_COLUMNS = [
    "lead_id", "company_id", "company", "industry", "employees", "country", "lead_source",
    "seniority", "job_function", "created_date", "pages_viewed", "pricing_page_visits",
    "email_opens", "email_clicks", "webinar_attended", "erp_type", "hiring_ops_roles",
    "recent_funding", "meetings_held", "converted",
]


def pick(rng: random.Random, options):
    """Weighted choice over (value, weight, ...) tuples; returns the whole tuple."""
    r, acc = rng.random(), 0.0
    total = sum(o[1] for o in options)
    for o in options:
        acc += o[1] / total
        if r < acc:
            return o
    return options[-1]


def poisson(rng: random.Random, lam: float) -> int:
    l, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= l:
            return k
        k += 1


def company_name(rng: random.Random, suffixes) -> str:
    stem = rng.choice(SYL_A) + rng.choice(SYL_B) + rng.choice(SYL_C)
    return f"{stem.capitalize()} {rng.choice(suffixes)}"


def slug(name: str) -> str:
    return "".join(c for c in name.lower() if c.isalnum())[:24]


# ---------------------------------------------------------------- leads

def build_leads(rng: random.Random):
    names = list(INDUSTRIES)
    rows, latent = [], []
    used = set()
    start = date(2025, 1, 1)
    for i in range(N_LEADS):
        ind = rng.choices(names, weights=INDUSTRY_SHARE)[0]
        fit, suffixes, _ = INDUSTRIES[ind]
        name = company_name(rng, suffixes)
        while name in used:
            name = company_name(rng, suffixes)
        used.add(name)
        employees = int(min(5000, max(5, math.exp(rng.gauss(4.6, 1.2)))))
        country = pick(rng, COUNTRIES)[0]
        src = pick(rng, SOURCES)
        sen = pick(rng, SENIORITY)
        fn = pick(rng, FUNCTIONS)
        erp = pick(rng, ERP)
        created = start + timedelta(days=rng.randrange(365))
        engaged = src[0] in ("inbound_demo", "pricing_page", "referral")
        pages = poisson(rng, 6 if engaged else 3)
        pricing = poisson(rng, 1.2 if engaged else 0.3)
        opens = poisson(rng, 4)
        clicks = sum(1 for _ in range(opens) if rng.random() < 0.18)
        webinar = 1 if (src[0] == "webinar" or rng.random() < 0.05) else 0
        hiring = 1 if rng.random() < 0.18 else 0
        funding = 1 if rng.random() < 0.08 else 0

        size = 0.6 if 50 <= employees <= 1000 else (-0.8 if employees < 20 else 0.0)
        score = (fit + size + src[2] + sen[2] + fn[2] + erp[2]
                 + 0.35 * min(pricing, 4) + 0.15 * min(clicks, 4) + 0.04 * min(pages, 15)
                 + 0.5 * hiring + 0.3 * funding + 0.2 * webinar)
        latent.append(score)
        rows.append({
            "lead_id": f"L{i + 1:05d}", "company_id": f"C{i + 1:05d}", "company": name,
            "industry": ind, "employees": employees, "country": country, "lead_source": src[0],
            "seniority": sen[0], "job_function": fn[0], "created_date": created.isoformat(),
            "pages_viewed": pages, "pricing_page_visits": pricing, "email_opens": opens,
            "email_clicks": clicks, "webinar_attended": webinar, "erp_type": erp[0],
            "hiring_ops_roles": hiring, "recent_funding": funding,
        })

    # Pick the intercept so the expected conversion rate lands on TARGET_RATE.
    lo, hi = -15.0, 5.0
    for _ in range(60):
        mid = (lo + hi) / 2
        rate = sum(1 / (1 + math.exp(-(mid + s))) for s in latent) / len(latent)
        lo, hi = (mid, hi) if rate < TARGET_RATE else (lo, mid)
    b0 = (lo + hi) / 2

    for row, s in zip(rows, latent):
        converted = 1 if rng.random() < 1 / (1 + math.exp(-(b0 + s))) else 0
        # The planted leak: meetings are logged in the CRM, including the ones that happen
        # because the deal is already moving. It "predicts" conversion by recording it.
        if converted:
            meetings = 1 + sum(1 for _ in range(3) if rng.random() < 0.55)
        else:
            meetings = 1 if rng.random() < 0.07 else 0
        row["meetings_held"] = meetings
        row["converted"] = converted
    return rows


# ---------------------------------------------------------------- messy copy

INDUSTRY_VARIANTS = {
    "Wholesale distribution": ["Wholesale", "wholesale distribution", "Distribution", "WHOLESALE DISTRIBUTION"],
    "Manufacturing": ["Mfg", "manufacturing", "Manufacturer", "MANUFACTURING"],
    "Food & beverage": ["Food and Beverage", "F&B", "food & beverage"],
    "Logistics": ["logistics", "3PL", "Logistics & Warehousing"],
    "Retail": ["retail", "Retail - Consumer", "Ecommerce"],
    "Healthcare": ["Health care", "healthcare", "Medical"],
    "Construction": ["construction", "Construction & Trades"],
    "Professional services": ["Prof. Services", "Consulting", "professional services"],
    "Software": ["SaaS", "software", "Tech - Software"],
    "Education": ["education", "EdTech", "Training"],
}
COUNTRY_VARIANTS = {"United States": ["USA", "US", "U.S.", "united states"],
                    "United Kingdom": ["UK", "U.K.", "Great Britain", "England"],
                    "Canada": ["CA", "canada"], "Germany": ["DE", "Deutschland"], "Australia": ["AU", "australia"]}
SOURCE_VARIANTS = {"inbound_demo": ["Inbound Demo", "demo request", "inbound-demo"],
                   "pricing_page": ["Pricing Page", "pricing"],
                   "content_download": ["Content Download", "ebook", "content"],
                   "outbound": ["Outbound", "cold email", "SDR outbound"],
                   "webinar": ["Webinar"], "event": ["Event", "trade show"], "referral": ["Referral", "partner referral"]}


def employees_range(n: int) -> str:
    for lo, hi in [(1, 10), (11, 50), (51, 200), (201, 500), (501, 1000), (1001, 5000)]:
        if lo <= n <= hi:
            return f"{lo}-{hi}"
    return "5000+"


def messy_date(rng: random.Random, iso: str) -> str:
    d = date.fromisoformat(iso)
    r = rng.random()
    if r < 0.6:
        return iso
    if r < 0.85:
        return d.strftime("%m/%d/%Y")
    return d.strftime("%d %b %Y")


def build_messy(rng: random.Random, clean_rows):
    out = []
    for row in clean_rows:
        m = dict(row)
        first = rng.choice(FIRST)
        last = (rng.choice(SYL_A) + rng.choice(SYL_C)).capitalize()
        m["contact_name"] = f"{first} {last}"
        m["contact_email"] = f"{first.lower()}.{last.lower()}@{slug(row['company'])}.example"
        if rng.random() < 0.25:
            m["industry"] = rng.choice(INDUSTRY_VARIANTS[row["industry"]])
        if rng.random() < 0.20 and row["country"] in COUNTRY_VARIANTS:
            m["country"] = rng.choice(COUNTRY_VARIANTS[row["country"]])
        if rng.random() < 0.20:
            m["lead_source"] = rng.choice(SOURCE_VARIANTS[row["lead_source"]])
        r = rng.random()
        if r < 0.05:
            m["employees"] = employees_range(row["employees"])
        elif r < 0.10 and row["employees"] >= 1000:
            m["employees"] = f"{row['employees']:,}"
        m["created_date"] = messy_date(rng, row["created_date"])
        for col, p in [("employees", 0.07), ("industry", 0.05), ("pages_viewed", 0.04), ("country", 0.03), ("erp_type", 0.06)]:
            if rng.random() < p:
                m[col] = ""
        out.append(m)

    # A few data-entry errors.
    for idx in rng.sample(range(len(out)), 3):
        out[idx]["employees"] = 999999
    for idx in rng.sample(range(len(out)), 2):
        out[idx]["pages_viewed"] = -1

    # Duplicates: the same contact entered again later under a new lead id, with the
    # company typed slightly differently. Same person, same outcome.
    dupes = []
    for k, idx in enumerate(sorted(rng.sample(range(len(out)), int(0.03 * len(out))))):
        d = dict(out[idx])
        d["lead_id"] = f"L{N_LEADS + k + 1:05d}"
        d["company"] = rng.choice([d["company"].upper(), d["company"] + " ", d["company"].replace(" ", "  ", 1), d["company"] + " Inc."])
        if rng.random() < 0.5:
            d["lead_source"] = rng.choice(SOURCE_VARIANTS["outbound"] + ["webinar"])
        dupes.append(d)
    rows = out + dupes
    rng.shuffle(rows)
    return rows


# ---------------------------------------------------------------- companies.jsonl

SIGNALS = [
    "is hiring a demand planner", "posted two warehouse operations roles this month",
    "announced a second distribution center", "raised a growth round", "launched a new product line",
    "named a new VP of operations", "is replacing its ERP", "expanded into a new region",
    "reported stockouts during the last peak season", "is consolidating suppliers",
]
REGIONS = {
    "United States": ["the Midwest", "the Pacific Northwest", "the Southeast", "Texas", "New England"],
    "Canada": ["Ontario", "British Columbia", "Quebec", "the Prairies"],
    "United Kingdom": ["the UK Midlands", "the North West", "Scotland", "the South East"],
    "Germany": ["Bavaria", "the Rhine valley", "Saxony", "Lower Saxony"],
    "Australia": ["New South Wales", "Victoria", "Queensland"],
}


def build_companies(rng: random.Random, clean_rows):
    customers = [r for r in clean_rows if r["converted"] == 1]
    others = [r for r in clean_rows if r["converted"] == 0]
    picked = rng.sample(customers, 40) + rng.sample(others, 360)
    picked.sort(key=lambda r: r["company_id"])
    out = []
    for r in picked:
        _, _, products = INDUSTRIES[r["industry"]]
        product = rng.choice(products)
        region = rng.choice(REGIONS[r["country"]])
        size_word = "small" if r["employees"] < 50 else ("mid-size" if r["employees"] <= 1000 else "large")
        sents = [
            f"{r['company']} is a {size_word} {r['industry'].lower()} company focused on {product}.",
            f"It serves customers across {region} with about {r['employees']} employees.",
        ]
        if r["erp_type"] == "legacy":
            sents.append("Its operations still run on an older on-premise ERP and a lot of spreadsheets.")
        elif r["erp_type"] == "cloud":
            sents.append("It moved its finance and inventory onto a cloud ERP in the last few years.")
        else:
            sents.append("It tracks orders and stock mostly in spreadsheets.")
        sigs = rng.sample(SIGNALS, rng.choice([0, 1, 1, 2]))
        if r["hiring_ops_roles"] and "is hiring a demand planner" not in sigs:
            sigs.append("is hiring a demand planner")
        if sigs:
            sents.append(f"Recently it {sigs[0]}.")
        out.append({
            "company_id": r["company_id"], "name": r["company"], "industry": r["industry"],
            "employees": r["employees"], "country": r["country"],
            "is_customer": bool(r["converted"]), "signals": sigs, "description": " ".join(sents),
        })
    return out


# ---------------------------------------------------------------- docs + questions

def build_docs():
    """Tallyhawk's (fictional) knowledge base. Each doc lists its sentences; questions point
    at the sentence that answers them, so retrieval can be scored exactly."""
    docs = []

    def doc(doc_id, title, kind, paragraphs, qa):
        docs.append({"id": doc_id, "title": title, "kind": kind, "paragraphs": paragraphs, "qa": qa})

    doc("product-overview", "What Tallyhawk does", "product", [
        "Tallyhawk is inventory forecasting software for mid-size distributors and manufacturers.",
        "It predicts demand for each item at each warehouse, then recommends how much to reorder and when.",
        "Most customers have between 50 and 1,000 employees and carry between 2,000 and 50,000 stock-keeping units.",
        "Forecasts refresh every night using the previous day's orders.",
    ], [("Who is Tallyhawk built for?", 0), ("How often do forecasts refresh?", 3)])
    doc("how-forecasting-works", "How the forecast works", "product", [
        "Each item gets its own forecast built from at least 12 months of order history.",
        "Items with less than 12 months of history are grouped with similar items until they have enough data.",
        "The forecast accounts for seasonality, promotions you enter in advance, and supplier lead times.",
        "Every forecast shows a range, not a single number, so planners can see the uncertainty.",
    ], [("How much order history does Tallyhawk need per item?", 0), ("What happens to new items with little history?", 1), ("Does the forecast show a single number?", 3)])
    doc("reorder-recommendations", "Reorder recommendations", "product", [
        "Tallyhawk turns each forecast into a suggested purchase order.",
        "Suggestions respect minimum order quantities, pack sizes and supplier lead times.",
        "Planners approve, edit or reject every suggestion; nothing is sent to a supplier automatically.",
        "Approved orders can be exported as a CSV or pushed to a connected ERP.",
    ], [("Does Tallyhawk send purchase orders to suppliers automatically?", 2), ("Can reorder suggestions handle pack sizes?", 1)])
    doc("pricing", "Pricing", "commercial", [
        "Tallyhawk has three plans: Starter, Growth and Enterprise.",
        "Starter costs $900 per month and covers up to 5,000 stock-keeping units and one warehouse.",
        "Growth costs $2,400 per month and covers up to 25,000 stock-keeping units and five warehouses.",
        "Enterprise is priced by quote and covers unlimited items and warehouses.",
        "All plans are billed annually; monthly billing adds 15 percent.",
    ], [("How much does the Growth plan cost?", 2), ("How many warehouses does Starter include?", 1), ("Is there a monthly billing option?", 4)])
    doc("free-trial", "Trials and pilots", "commercial", [
        "Tallyhawk does not offer a self-serve free trial.",
        "Instead, qualified companies get a 30-day pilot on their own data for one warehouse.",
        "A pilot needs a CSV export of 12 months of orders and a list of current suppliers.",
    ], [("Is there a free trial?", 0), ("How long is a pilot?", 1), ("What data does a pilot need?", 2)])
    doc("implementation", "Implementation timeline", "onboarding", [
        "A typical implementation takes six weeks from contract to first live forecast.",
        "Week one is the data export and mapping; weeks two to four are model training and review.",
        "Weeks five and six are a parallel run, where planners compare Tallyhawk to their current process.",
        "Customers need one operations lead available for about four hours a week during implementation.",
    ], [("How long does implementation take?", 0), ("What is the parallel run?", 2), ("How much time does the customer need to commit during implementation?", 3)])
    doc("support", "Support and service levels", "onboarding", [
        "Support is available by email and chat from 8am to 6pm US Eastern, Monday to Friday.",
        "Growth and Enterprise customers get a named customer success manager.",
        "Critical issues receive a first response within two business hours.",
        "Enterprise customers can add 24/7 phone support.",
    ], [("What are Tallyhawk's support hours?", 0), ("How fast are critical issues answered?", 2), ("Who gets a named customer success manager?", 1)])
    doc("security", "Security", "trust", [
        "Customer data is encrypted in transit and at rest.",
        "Tallyhawk completed a SOC 2 Type II audit, and the report is available under NDA.",
        "Data is hosted in the United States by default; European customers can choose an EU region.",
        "Single sign-on through SAML is included on the Enterprise plan.",
    ], [("Can European customers keep their data in the EU?", 2), ("Is single sign-on available?", 3), ("Does Tallyhawk have a SOC 2 report?", 1)])
    doc("data-retention", "Data retention and deletion", "trust", [
        "Order history is kept for as long as the customer's contract is active.",
        "After cancellation, data is deleted within 30 days unless the customer asks for an export first.",
        "Tallyhawk does not use one customer's data to train forecasts for another customer.",
    ], [("What happens to my data if I cancel?", 1), ("Is my data used to train models for other customers?", 2)])
    doc("users-and-roles", "Users and roles", "product", [
        "Every plan includes unlimited read-only users.",
        "Planner seats, which can approve purchase orders, are included up to five on Starter and twenty on Growth.",
        "Admins control which warehouses each planner can see.",
    ], [("How many planner seats come with Growth?", 1), ("Are read-only users limited?", 0)])
    doc("api", "API access", "product", [
        "Tallyhawk has a REST API for reading forecasts and writing approved orders.",
        "API access is included on Growth and Enterprise plans.",
        "The API is rate limited to 600 requests per minute per account.",
    ], [("Is the API included on Starter?", 1), ("What is the API rate limit?", 2)])
    doc("cancellation", "Contracts and cancellation", "commercial", [
        "Contracts run for 12 months and renew automatically.",
        "Customers can cancel by giving 60 days' notice before the renewal date.",
        "Tallyhawk does not charge setup fees on Starter or Growth.",
    ], [("How much notice is needed to cancel?", 1), ("Are there setup fees?", 2)])

    integrations = [
        ("ledgerbase", "Ledgerbase ERP", "every hour", "Growth"),
        ("stockyard", "Stockyard WMS", "every 15 minutes", "Starter"),
        ("ordermill", "Ordermill commerce platform", "every hour", "Starter"),
        ("crateline", "Crateline shipping", "nightly", "Growth"),
        ("quarry", "Quarry accounting", "nightly", "Starter"),
        ("spreadsheet", "CSV and spreadsheet uploads", "whenever a file is uploaded", "Starter"),
    ]
    for key, name, freq, plan in integrations:
        doc(f"integration-{key}", f"Integration: {name}", "integration", [
            f"Tallyhawk connects to {name} (a fictional product) to read orders and item data.",
            f"Data syncs {freq}.",
            f"The {name} connection is available on the {plan} plan and above.",
        ], [(f"How often does Tallyhawk sync with {name}?", 1), (f"Which plan includes the {name} integration?", 2)])

    cases = [("a regional plumbing-parts distributor", "38 percent", "stockouts", 4),
             ("a food manufacturer with three plants", "21 percent", "excess inventory", 3),
             ("an industrial fastener wholesaler", "a third", "emergency freight spend", 5),
             ("a packaging supplier", "27 percent", "stockouts", 2),
             ("an outdoor-gear retailer's distribution arm", "18 percent", "excess inventory", 6),
             ("a precision-valve maker", "31 percent", "late orders", 3)]
    for k, (who, amount, metric, wh) in enumerate(cases, 1):
        doc(f"case-study-{k}", f"Case study {k}: {who}", "case-study", [
            f"This is a fictional case study written for the course: {who} with {wh} warehouses.",
            f"After six months on Tallyhawk, {metric} fell by {amount} compared with the same months a year earlier.",
            "The comparison is a simple before and after, with no control group, so other changes that year could explain part of it.",
        ], [(f"What changed for {who}?", 1), (f"How was the result for {who} measured?", 2)])
    return docs


def render_doc(d) -> str:
    return f"# {d['title']}\n\n" + "\n\n".join(d["paragraphs"]) + "\n"


# ---------------------------------------------------------------- write

def to_csv(rows, columns) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=columns, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({c: r.get(c, "") for c in columns})
    return buf.getvalue()


def build_all() -> dict[str, str]:
    rng = random.Random(SEED)
    leads = build_leads(rng)
    messy = build_messy(random.Random(SEED + 1), leads)
    companies = build_companies(random.Random(SEED + 2), leads)
    docs = build_docs()

    files = {
        "leads_clean.csv": to_csv(leads, LEAD_COLUMNS),
        "leads_messy.csv": to_csv(messy, LEAD_COLUMNS[:3] + ["contact_name", "contact_email"] + LEAD_COLUMNS[3:]),
        "companies.jsonl": "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in companies),
    }
    questions = []
    for d in docs:
        files[f"docs/{d['id']}.md"] = render_doc(d)
        for q, para in d["qa"]:
            questions.append({"id": f"q{len(questions) + 1:03d}", "question": q, "doc_id": d["id"], "answer_passage": d["paragraphs"][para]})
    files["questions.jsonl"] = "".join(json.dumps(q, ensure_ascii=False) + "\n" for q in questions)
    return files


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="fail if data/ differs from a fresh build")
    args = ap.parse_args()
    files = build_all()
    if args.check:
        bad = [p for p, text in files.items() if not (DATA / p).exists() or (DATA / p).read_text(encoding="utf-8") != text]
        if bad:
            print("data/ differs from the seed for:", *bad, sep="\n  ")
            return 1
        print(f"data/ matches seed {SEED} ({len(files)} files)")
        return 0
    for p, text in files.items():
        (DATA / p).parent.mkdir(parents=True, exist_ok=True)
        (DATA / p).write_text(text, encoding="utf-8")
    print(f"wrote {len(files)} files to {DATA} from seed {SEED}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
