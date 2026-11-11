# AI Track starter kit

The code and data for [the AI Track](https://goeveryway.com/resources/ai-expert-track), a free course on building AI that moves revenue. Every module's build happens here, and every build adds a piece to the same system: a lead engine for **Tallyhawk**, a fictional company that sells inventory-forecasting software to mid-size distributors and manufacturers.

All data in this repo is synthetic. Company names, contact names and documents are generated from a fixed seed. None of it describes a real person or business.

## Setup (about 15 minutes)

### Option A: no install (Google Colab)

Click a module below to open its notebook in Colab, then run the first two cells. The first cell clones this repo and installs the packages. It only runs on Colab.

| Module | Open in Colab |
|---|---|
| 0 | [Set up and ship something tiny](https://colab.research.google.com/github/Go-Everyway/ai-track-starter/blob/main/notebooks/00-set-up-and-ship-something-tiny.ipynb) |
| 1 | [How ML actually works](https://colab.research.google.com/github/Go-Everyway/ai-track-starter/blob/main/notebooks/01-how-ml-actually-works.ipynb) |
| 2 | [Classic models, and when to use them](https://colab.research.google.com/github/Go-Everyway/ai-track-starter/blob/main/notebooks/02-classic-models-and-when-to-use-them.ipynb) |
| 3 | [Data: the real job](https://colab.research.google.com/github/Go-Everyway/ai-track-starter/blob/main/notebooks/03-data-the-real-job.ipynb) |
| 4 | [Evaluation and metrics](https://colab.research.google.com/github/Go-Everyway/ai-track-starter/blob/main/notebooks/04-evaluation-and-metrics.ipynb) |
| 5 | [Neural networks, by intuition](https://colab.research.google.com/github/Go-Everyway/ai-track-starter/blob/main/notebooks/05-neural-networks-by-intuition.ipynb) |
| 6 | [How LLMs actually work](https://colab.research.google.com/github/Go-Everyway/ai-track-starter/blob/main/notebooks/06-how-llms-actually-work.ipynb) |
| 7 | [Prompt and context engineering](https://colab.research.google.com/github/Go-Everyway/ai-track-starter/blob/main/notebooks/07-prompt-and-context-engineering.ipynb) |
| 8 | [RAG: grounded, cited answers](https://colab.research.google.com/github/Go-Everyway/ai-track-starter/blob/main/notebooks/08-rag-grounded-cited-answers.ipynb) |
| 9 | [Tools and agents](https://colab.research.google.com/github/Go-Everyway/ai-track-starter/blob/main/notebooks/09-tools-and-agents.ipynb) |
| 10 | [Evals and reliability](https://colab.research.google.com/github/Go-Everyway/ai-track-starter/blob/main/notebooks/10-evals-and-reliability.ipynb) |
| 11 | [Fine-tune vs. RAG vs. prompt](https://colab.research.google.com/github/Go-Everyway/ai-track-starter/blob/main/notebooks/11-fine-tune-vs-rag-vs-prompt.ipynb) |
| 12 | [Applied AI for go-to-market](https://colab.research.google.com/github/Go-Everyway/ai-track-starter/blob/main/notebooks/12-applied-ai-for-go-to-market.ipynb) |
| 13 | [AI engineering and deployment](https://colab.research.google.com/github/Go-Everyway/ai-track-starter/blob/main/notebooks/13-ai-engineering-and-deployment.ipynb) |

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
| 5 | Embeds 400 company descriptions once: 19,623 tokens with the `o200k_base` tokenizer (Module 6 measures it). Your embedding model's own tokenizer will give a similar count. | That count times your embedding model's price per million tokens. Cached, so you pay once. |
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
| `lib/outreach.py` | Module 7's outreach-angle step: the prompt versions, the schema, validation with one retry, and the injection guard. |
| `lib/rag.py` | Module 8's retrieval: chunking, keyword/vector/hybrid search, citations limited to retrieved passages, `supports()` for checking claims, recall@k and MRR. |
| `lib/pipeline.py` | Module 9's pipeline: enrich (a tool call), score, draft, verify claims, then an approval gate and idempotent sends. |
| `lib/evals.py` | Test suites and the runner. Modules 7 to 9 add suites in `evals/`. |
| `lib/grader.py`, `lib/harness.py` | Module 10: a code grader and a model grader with agreement against human labels, and the gate that blocks a send when checks fail. |
| `lib/campaign.py` | Module 12: signal-to-offer map, seeded holdout assignment, the before number, suppression and opt-out, SPF/DKIM/DMARC record checks, and treatment-vs-holdout results with a range. |
| `lib/ops.py` | Module 13: retries with backoff and jitter, timeouts, rate limiting, pinned models with a fallback, PII redaction, cost caps, a daily dashboard, alerts, and a deploy that refuses to run when the eval suites fail. |
| `notebooks/` | One notebook per module. |
| `evals/` | The suites from Modules 7 to 9, `evals/run.py` (the harness), and `labels/`: 40 outreach outputs labelled by hand for grader validation, written for the course. |
| `solutions/` | Worked solutions, added as modules are published. |
| `deploy/` | Module 0's serverless function (Flask on Vercel's Python runtime) and its local test. |
| `tools/` | `generate_data.py` rebuilds `data/` from the seed; `make_notebooks.py` builds the notebooks from `notebooks/src` and `solutions/src`; `run_notebooks.py` runs them all headless. |

See [`data/README.md`](data/README.md) for every column.

## Real-model runs

The module pages publish one real run of the Track 2 solutions (Modules 6 to 11) beside the mock
numbers, labelled with model, date and cost. To make one yourself:

```bash
# key in .env as LLM_API_KEY, never on the command line
python tools/real_run.py --small <cheap-model-id> --large <bigger-model-id> --embed <embedding-model-id> --budget 10
```

It asks before it spends anything, stops at the budget, and saves `runs/<date>-<model>/`: the executed
notebooks, the figures each one recorded, per-module call logs (tokens, cost, latency; no prompts or
replies) and `summary.json` with the total. If your key shows up in any output, it deletes the folder.

## Reproducing the data

```bash
python tools/generate_data.py --check    # confirms data/ matches the seed
python tools/generate_data.py            # rebuilds it
```

The generator uses only Python's standard library, so the files are byte-identical on any machine.

## Checks

```bash
python -m unittest discover tests        # lib/llm.py, no network needed
python tools/run_notebooks.py            # every notebook and solution, in mock mode
python deploy/test_app.py                # Module 0's function, no network needed
python evals/run.py                      # every eval suite; exits 1 if any is under its threshold
```

## Licence and errors

Code is MIT (see `LICENSE`). The course text on goeveryway.com is CC BY 4.0.

Found something wrong in the code, the data or a module page? Open an issue.
