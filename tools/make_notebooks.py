"""Build every notebook in notebooks/ and solutions/.

Modules are authored as plain Python cell scripts, which are easier to write and review
than notebook JSON:

    notebooks/src/NN-*.py   the learner's notebook, with the build left as marked gaps
    solutions/src/NN-*.py   the same notebook, completed, ending in its done-when checks

Cells are separated by `# %%` lines; `# %% [markdown]` starts a markdown cell whose lines
are written as `# text`. Every built notebook gets the same title, Colab and setup cells
first. A module with no source yet gets a short scaffold that just loads its data.

    python tools/make_notebooks.py          # all modules
    python tools/make_notebooks.py 02 03    # just these
"""

import json
import re
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


def parse_cells(text: str):
    """Split a `# %%` cell script into (kind, source) pairs."""
    cells, kind, lines = [], None, []

    def flush():
        src = "\n".join(lines).strip("\n")
        if kind and src:
            if kind == "markdown":
                src = "\n".join(re.sub(r"^# ?", "", l) for l in src.splitlines())
            cells.append((kind, src))

    for line in text.splitlines():
        m = re.match(r"^# %%( \[markdown\])?", line)
        if m:
            flush()
            kind, lines = ("markdown" if m.group(1) else "code"), []
        elif kind:
            lines.append(line)
    flush()
    return cells


def notebook(num, title, body, intro):
    cells = [
        cell("markdown", f"# Module {int(num)}: {title}\n\n{intro}", f"m{num}-intro"),
        cell("code", COLAB, f"m{num}-colab"),
        cell("code", SETUP, f"m{num}-setup"),
    ]
    cells += [cell(kind, src, f"m{num}-{i:02d}") for i, (kind, src) in enumerate(body)]
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                     "language_info": {"name": "python"}},
        "cells": cells,
    }


def slug_for(num, title):
    return f"{num}-" + re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


SCAFFOLD_INTRO = ("Starter notebook for the AI Track. Run the cells top to bottom. All data in this kit is synthetic.\n\n"
                  "The guided build for this module is added when the module is published; "
                  "until then this notebook loads the data so you can explore it.")
LEARNER_INTRO = ("Run the cells top to bottom. Cells marked **YOUR TURN** are the build: they run as they are, "
                 "but they won't pass the done-when checks until you fill them in. All data in this kit is synthetic.")
SOLUTION_INTRO = ("The worked solution. Try the build in `notebooks/` first; getting stuck for a while teaches more "
                  "than reading this. The last cells check every done-when criterion.")


def main(only=()):
    built = 0
    for num, title, starter in MODULES:
        if only and num not in only:
            continue
        slug = slug_for(num, title)
        for kind, intro in [("notebooks", LEARNER_INTRO), ("solutions", SOLUTION_INTRO)]:
            src = ROOT / kind / "src" / f"{slug}.py"
            out = ROOT / kind / f"{slug}.ipynb"
            if src.exists():
                body = parse_cells(src.read_text(encoding="utf-8"))
            elif kind == "notebooks":
                body, intro = [("markdown", "## Start here"), ("code", starter)], SCAFFOLD_INTRO
            else:
                continue
            out.parent.mkdir(exist_ok=True)
            out.write_text(json.dumps(notebook(num, title, body, intro), indent=1) + "\n", encoding="utf-8")
            built += 1
    print(f"built {built} notebooks")


if __name__ == "__main__":
    import sys
    main(tuple(sys.argv[1:]))
