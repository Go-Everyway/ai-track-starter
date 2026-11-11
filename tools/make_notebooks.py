"""Write the per-module notebook scaffolds in notebooks/.

Each notebook gets the same two setup cells (Colab bootstrap, then repo-root discovery)
and a starter cell that loads the data that module uses. The guided build for each
module is added when that module is written; until then the notebook says so.

    python tools/make_notebooks.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Colab clones from here.
REPO_URL = "https://github.com/Go-Everyway/ai-track-starter"

MODULES = [
    ("00", "Set up and ship something tiny", "from lib import llm\nreply = llm.chat('Say hello in five words.')\nprint(reply.text)\nprint(reply.input_tokens, reply.output_tokens, reply.cost_usd)"),
    ("01", "How ML actually works", "leads = pd.read_csv(DATA / 'leads_clean.csv')\nprint(leads.shape)\nprint(f\"conversion rate: {leads.converted.mean():.1%}\")\nleads.head()"),
    ("02", "Classic models, and when to use them", "leads = pd.read_csv(DATA / 'leads_clean.csv')\nleads.describe(include='all').T.head(20)"),
    ("03", "Data: the real job", "messy = pd.read_csv(DATA / 'leads_messy.csv')\nprint(messy.shape)\nmessy.isna().sum().sort_values(ascending=False).head(10)"),
    ("04", "Evaluation and metrics", "leads = pd.read_csv(DATA / 'leads_clean.csv')\nleads.converted.value_counts()"),
    ("05", "Neural networks, by intuition", "companies = pd.read_json(DATA / 'companies.jsonl', lines=True)\nprint(companies.shape, '| customers:', int(companies.is_customer.sum()))\ncompanies[['name', 'industry', 'description']].head()"),
    ("06", "How LLMs actually work", "from lib import llm\nprint(llm.chat('In one sentence: what is a token?').text)"),
    ("07", "Prompt and context engineering", "companies = pd.read_json(DATA / 'companies.jsonl', lines=True)\nprint(companies.description.iloc[0])"),
    ("08", "RAG: grounded, cited answers", "docs = {p.stem: p.read_text() for p in sorted((DATA / 'docs').glob('*.md'))}\nquestions = pd.read_json(DATA / 'questions.jsonl', lines=True)\nprint(len(docs), 'docs,', len(questions), 'questions')\nquestions.head()"),
    ("09", "Tools and agents", "leads = pd.read_csv(DATA / 'leads_clean.csv')\ncompanies = pd.read_json(DATA / 'companies.jsonl', lines=True)\nprint(len(leads), 'leads,', len(companies), 'company profiles')"),
    ("10", "Evals and reliability", "print(sorted(p.name for p in (ROOT / 'evals').iterdir()))"),
    ("11", "Fine-tune vs. RAG vs. prompt", "from lib import llm\nprint(f'spent today: ${llm.spend_today():.4f}')"),
    ("12", "Applied AI for go-to-market", "leads = pd.read_csv(DATA / 'leads_clean.csv')\nleads.groupby('lead_source').converted.agg(['count', 'mean']).sort_values('mean', ascending=False)"),
    ("13", "AI engineering and deployment", "log = ROOT / 'logs' / 'llm_calls.jsonl'\nprint(log.read_text()[-500:] if log.exists() else 'no calls logged yet')"),
]

COLAB = f"""# Runs only on Google Colab: fetch the starter kit and its packages, and copy any keys
# you saved in Colab's Secrets panel into the environment. Locally this does nothing.
import os, sys
if "google.colab" in sys.modules:
    !git clone --depth 1 {REPO_URL} /content/ai-track-starter
    %cd /content/ai-track-starter
    %pip install -q -r requirements.txt
    from google.colab import userdata
    for name in ["LLM_API_KEY", "LLM_PROVIDER", "LLM_BASE_URL", "LLM_MODEL", "EMBED_MODEL", "LLM_DAILY_BUDGET_USD"]:
        try:
            os.environ[name] = userdata.get(name)
        except Exception:
            pass  # not set in Secrets; lib/llm.py uses its default"""

SETUP = """# Find the repo root so paths work from notebooks/ or from Colab.
import sys
from pathlib import Path
import pandas as pd

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "data" / "leads_clean.csv").exists())
DATA = ROOT / "data"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
print("starter kit at", ROOT)"""


def cell(kind, src, cid):
    c = {"cell_type": kind, "id": cid, "metadata": {}, "source": src.splitlines(keepends=True)}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


def notebook(num, title, starter):
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                     "language_info": {"name": "python"}},
        "cells": [
            cell("markdown", f"# Module {int(num)}: {title}\n\nStarter notebook for the AI Track. "
                             "Run the cells top to bottom. All data in this kit is synthetic.\n\n"
                             "The guided build for this module is added when the module is published; "
                             "until then this notebook loads the data so you can explore it.", f"m{num}-intro"),
            cell("code", COLAB, f"m{num}-colab"),
            cell("code", SETUP, f"m{num}-setup"),
            cell("markdown", "## Start here", f"m{num}-start-md"),
            cell("code", starter, f"m{num}-start"),
        ],
    }


def main():
    out = ROOT / "notebooks"
    out.mkdir(exist_ok=True)
    for num, title, starter in MODULES:
        slug = title.lower().replace(":", "").replace(",", "").replace(".", "").replace(" ", "-")
        path = out / f"{num}-{slug}.ipynb"
        path.write_text(json.dumps(notebook(num, title, starter), indent=1) + "\n", encoding="utf-8")
    print(f"wrote {len(MODULES)} notebooks to {out}")


if __name__ == "__main__":
    main()
