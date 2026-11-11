# AI Track starter kit

The code and data for [the AI Track](https://goeveryway.com/resources/ai-expert-track), a free course on building AI that moves revenue. Every module's build happens here, and every build adds a piece to the same system: a lead engine for **Tallyhawk**, a fictional company that sells inventory-forecasting software to mid-size distributors and manufacturers.

All data in this repo is synthetic. Company names, contact names and documents are generated from a fixed seed. None of it describes a real person or business.

## Setup (about 15 minutes)

### Option A: no install (Google Colab)

Click a module below to open its notebook in Colab, then run the first two cells. The first cell clones this repo and installs the packages. It only runs on Colab.

| Module | Open in Colab |
|---|---|
| 0 | [Set up and ship something tiny](https://colab.research.google.com/github/Go-Everyway/ai-track-starter/blob/main/notebooks/00-set-up-and-ship-something-tiny.ipynb) |
| 1 | [How ml actually works](https://colab.research.google.com/github/Go-Everyway/ai-track-starter/blob/main/notebooks/01-how-ml-actually-works.ipynb) |
| 2 | [Classic models and when to use them](https://colab.research.google.com/github/Go-Everyway/ai-track-starter/blob/main/notebooks/02-classic-models-and-when-to-use-them.ipynb) |
| 3 | [Data the real job](https://colab.research.google.com/github/Go-Everyway/ai-track-starter/blob/main/notebooks/03-data-the-real-job.ipynb) |
| 4 | [Evaluation and metrics](https://colab.research.google.com/github/Go-Everyway/ai-track-starter/blob/main/notebooks/04-evaluation-and-metrics.ipynb) |
| 5 | [Neural networks by intuition](https://colab.research.google.com/github/Go-Everyway/ai-track-starter/blob/main/notebooks/05-neural-networks-by-intuition.ipynb) |
| 6 | [How llms actually work](https://colab.research.google.com/github/Go-Everyway/ai-track-starter/blob/main/notebooks/06-how-llms-actually-work.ipynb) |
| 7 | [Prompt and context engineering](https://colab.research.google.com/github/Go-Everyway/ai-track-starter/blob/main/notebooks/07-prompt-and-context-engineering.ipynb) |
| 8 | [Rag grounded cited answers](https://colab.research.google.com/github/Go-Everyway/ai-track-starter/blob/main/notebooks/08-rag-grounded-cited-answers.ipynb) |
| 9 | [Tools and agents](https://colab.research.google.com/github/Go-Everyway/ai-track-starter/blob/main/notebooks/09-tools-and-agents.ipynb) |
| 10 | [Evals and reliability](https://colab.research.google.com/github/Go-Everyway/ai-track-starter/blob/main/notebooks/10-evals-and-reliability.ipynb) |
| 11 | [Fine tune vs rag vs prompt](https://colab.research.google.com/github/Go-Everyway/ai-track-starter/blob/main/notebooks/11-fine-tune-vs-rag-vs-prompt.ipynb) |
| 12 | [Applied ai for go to market](https://colab.research.google.com/github/Go-Everyway/ai-track-starter/blob/main/notebooks/12-applied-ai-for-go-to-market.ipynb) |
| 13 | [Ai engineering and deployment](https://colab.research.google.com/github/Go-Everyway/ai-track-starter/blob/main/notebooks/13-ai-engineering-and-deployment.ipynb) |

Modules 1 to 4 need no model at all, and everything else runs in mock mode (below), so the whole course works in Colab for free. When you want real model calls, add `LLM_API_KEY`, `LLM_MODEL` and `EMBED_MODEL` in Colab's **Secrets** panel (the key icon) and turn on notebook access. The first cell copies them into the environment. Don't paste a key into a cell.

### Option B: on your machine

You need Python 3.11 to 3.13 and git.

```bash
git clone <this repo's URL> ai-track-starter
cd ai-track-starter
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env
jupyter notebook notebooks/
```

Then open `notebooks/01-how-ml-actually-works.ipynb` and run it top to bottom.

### Step 3: a model provider (from Module 0 on)

`lib/llm.py` talks to any OpenAI-compatible API. The worked default is OpenRouter, which gives you many providers' models behind one key. To use it:

1. Create an API key with your provider.
2. Put it in `.env` as `LLM_API_KEY`, set `LLM_PROVIDER=openai-compatible`, and set `LLM_MODEL` and `EMBED_MODEL` to model ids from your provider's model list. Model names change often, so copy current ones rather than ones from a tutorial.
3. Set a spend cap in two places:
   - **At the provider.** Set a credit or spend limit on the key in your provider's dashboard, if it supports one. This is the cap that holds even if your code has a bug.
   - **In this kit.** `LLM_DAILY_BUDGET_USD` (default $1.00). `lib/llm.py` checks today's logged spend before every call and refuses to go over.

Never put a key in a notebook cell or commit it. `.env` is gitignored for that reason.

**No key yet?** With no `LLM_API_KEY` set, the kit runs in mock mode automatically. Every notebook still runs, for free: chat calls return canned replies and embeddings are simple word-hash vectors. That's enough to see how the pieces fit, not to judge quality.

## What it costs

Every call is logged to `logs/llm_calls.jsonl` with tokens, cost and latency, so you can see your own numbers instead of trusting these.

| Modules | Model calls | Expected spend |
|---|---|---|
| 1 to 4 | None. Classic ML on your laptop. | $0 |
| 5 | Embeds 400 company descriptions, about 25,000 tokens in total (measured from `data/companies.jsonl`). | Cents with a typical embedding model. Check your model's price per million tokens. |
| 0, 6 to 13 | Planned per module; the exact number lands with each module. | Stated on each module page. |

The free route through the whole course is mock mode plus Modules 1 to 5 for real.

## What's in here

| Path | What it is |
|---|---|
| `data/leads_clean.csv` | 4,000 synthetic leads with a `converted` label (about 6% convert). |
| `data/leads_messy.csv` | The same leads the way a real CRM export arrives: duplicates, blanks, inconsistent categories, mixed date formats, contact details. |
| `data/companies.jsonl` | 400 synthetic company profiles for embeddings and outreach prompts. 40 are marked as existing customers. |
| `data/docs/` | Tallyhawk's (fictional) knowledge base: 24 short documents. |
| `data/questions.jsonl` | 54 questions about those documents, each with the passage that answers it, for scoring retrieval. |
| `lib/llm.py` | The provider-neutral client: `chat()`, `embed()`, `cosine()`, cost logging and the daily budget. |
| `notebooks/` | One notebook per module. |
| `evals/` | The test sets and harness that grow from Module 7. |
| `solutions/` | Worked solutions, added as modules are published. |
| `tools/` | `generate_data.py` rebuilds `data/` from the seed; `run_notebooks.py` runs every notebook headless. |

See [`data/README.md`](data/README.md) for every column.

## Reproducing the data

```bash
python tools/generate_data.py --check    # confirms data/ matches the seed
python tools/generate_data.py            # rebuilds it
```

The generator uses only Python's standard library, so the files are byte-identical on any machine.

## Checks

```bash
python -m unittest discover tests        # lib/llm.py, no network needed
python tools/run_notebooks.py            # every notebook, in mock mode
```

## Licence and errors

Code is MIT (see `LICENSE`). The course text on goeveryway.com is CC BY 4.0.

Found something wrong in the code, the data or a module page? Open an issue.
