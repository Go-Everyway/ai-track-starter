"""A thin, provider-neutral LLM client for the AI Track.

Every call goes through `chat()` or `embed()`, and every call is logged to
logs/llm_calls.jsonl with its model, tokens, cost and latency. Before each call the
client checks today's logged spend against LLM_DAILY_BUDGET_USD and refuses to go over.

It speaks the OpenAI-compatible API, which most providers and routers accept. The
worked default is OpenRouter. Settings come from environment variables (or a .env file):

    LLM_PROVIDER          "openai-compatible" or "mock". Defaults to mock when no key is set.
    LLM_BASE_URL          default https://openrouter.ai/api/v1
    LLM_API_KEY           your key. Never put it in code or commit it.
    LLM_MODEL             chat model id, e.g. whatever your provider lists
    EMBED_MODEL           embedding model id
    LLM_DAILY_BUDGET_USD  default 1.00. The client stops calling once today's logged spend hits it.
    LLM_PRICES            optional JSON: {"model-id": [input_usd_per_million, output_usd_per_million]}
                          used only when the provider doesn't report cost itself.

LLM_PROVIDER=mock needs no key and costs nothing. Chat replies are canned and
embeddings are word-hash vectors, which is enough to run every notebook end to end
and see how the pieces fit, but not to judge quality.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

try:  # optional: read .env if python-dotenv is installed
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = Path(os.environ.get("LLM_LOG_PATH", ROOT / "logs" / "llm_calls.jsonl"))


class BudgetExceeded(RuntimeError):
    """Raised before a call that would run past LLM_DAILY_BUDGET_USD."""


@dataclass
class Reply:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float | None  # None when neither the provider nor LLM_PRICES gives a price
    latency_s: float
    raw: object = field(default=None, repr=False)


def _settings() -> dict:
    has_key = bool(os.environ.get("LLM_API_KEY"))
    return {
        "provider": os.environ.get("LLM_PROVIDER") or ("openai-compatible" if has_key else "mock"),
        "base_url": os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1"),
        "api_key": os.environ.get("LLM_API_KEY", ""),
        "model": os.environ.get("LLM_MODEL", ""),
        "embed_model": os.environ.get("EMBED_MODEL", ""),
        "budget": float(os.environ.get("LLM_DAILY_BUDGET_USD", "1.00")),
        "prices": json.loads(os.environ.get("LLM_PRICES", "{}") or "{}"),
    }


# ---------------------------------------------------------------- logging + budget

def _log(entry: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def spend_today() -> float:
    """Total logged cost for today (UTC). Calls with unknown cost count as zero."""
    if not LOG_PATH.exists():
        return 0.0
    today = datetime.now(timezone.utc).date().isoformat()
    total = 0.0
    for line in LOG_PATH.read_text(encoding="utf-8").splitlines():
        e = json.loads(line)
        if e.get("ts", "").startswith(today) and e.get("cost_usd"):
            total += e["cost_usd"]
    return total


def _check_budget(cfg: dict) -> None:
    spent = spend_today()
    if spent >= cfg["budget"]:
        raise BudgetExceeded(
            f"Today's logged spend is ${spent:.4f}, at or over LLM_DAILY_BUDGET_USD=${cfg['budget']:.2f}. "
            "Raise the budget on purpose if you mean to keep going."
        )


def _price(cfg: dict, model: str, tin: int, tout: int, reported: float | None) -> float | None:
    if reported is not None:
        return float(reported)
    if model in cfg["prices"]:
        pin, pout = cfg["prices"][model]
        return (tin * pin + tout * pout) / 1_000_000
    return None


# ---------------------------------------------------------------- mock provider

def _tokens_estimate(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))  # rough rule of thumb, mock mode only


def _mock_vector(text: str, dims: int = 256) -> list[float]:
    v = [0.0] * dims
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        h = int(hashlib.md5(w.encode()).hexdigest(), 16)
        v[h % dims] += 1.0 if (h >> 8) % 2 else -1.0
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


# ---------------------------------------------------------------- public API

def _client(cfg: dict):
    from openai import OpenAI  # imported lazily so mock mode works without the package

    if not cfg["api_key"]:
        raise RuntimeError("Set LLM_API_KEY (or LLM_PROVIDER=mock to run without one). See README, step 3.")
    return OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"])


def chat(messages, *, model: str | None = None, **params) -> Reply:
    """Send a chat request. `messages` is a list of {"role", "content"} dicts or a plain string.

    Extra keyword arguments (temperature, max_tokens, response_format, ...) pass straight
    through to the provider, so check its docs for what the model supports.
    """
    cfg = _settings()
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    model = model or cfg["model"]
    _check_budget(cfg)
    t0 = time.perf_counter()

    if cfg["provider"] == "mock":
        last = messages[-1]["content"] if messages else ""
        text = f"[mock reply] {last[:80]}"
        tin = sum(_tokens_estimate(m["content"]) for m in messages)
        reply = Reply(text, model or "mock", tin, _tokens_estimate(text), 0.0, time.perf_counter() - t0)
    else:
        if not model:
            raise RuntimeError("Set LLM_MODEL to a chat model id your provider lists.")
        extra = params.pop("extra_body", {}) or {}
        if "openrouter.ai" in cfg["base_url"]:
            extra.setdefault("usage", {"include": True})  # ask OpenRouter to report cost
        resp = _client(cfg).chat.completions.create(model=model, messages=messages, extra_body=extra or None, **params)
        u = resp.usage
        tin, tout = (u.prompt_tokens, u.completion_tokens) if u else (0, 0)
        reported = (getattr(u, "model_extra", None) or {}).get("cost") if u else None
        reply = Reply(
            resp.choices[0].message.content or "", resp.model or model, tin, tout,
            _price(cfg, model, tin, tout, reported), time.perf_counter() - t0, resp,
        )

    _log({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "kind": "chat",
          "provider": cfg["provider"], "model": reply.model, "input_tokens": reply.input_tokens,
          "output_tokens": reply.output_tokens, "cost_usd": reply.cost_usd,
          "latency_s": round(reply.latency_s, 3)})
    return reply


def embed(texts, *, model: str | None = None) -> list[list[float]]:
    """Embed one string or a list of strings. Returns one vector per input."""
    cfg = _settings()
    if isinstance(texts, str):
        texts = [texts]
    model = model or cfg["embed_model"]
    _check_budget(cfg)
    t0 = time.perf_counter()

    if cfg["provider"] == "mock":
        vectors, tin, cost, used = [_mock_vector(t) for t in texts], sum(_tokens_estimate(t) for t in texts), 0.0, model or "mock"
    else:
        if not model:
            raise RuntimeError("Set EMBED_MODEL to an embedding model id your provider lists.")
        resp = _client(cfg).embeddings.create(model=model, input=list(texts))
        vectors = [d.embedding for d in resp.data]
        tin = resp.usage.prompt_tokens if resp.usage else 0
        reported = (getattr(resp.usage, "model_extra", None) or {}).get("cost") if resp.usage else None
        cost, used = _price(cfg, model, tin, 0, reported), resp.model or model

    _log({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "kind": "embed",
          "provider": cfg["provider"], "model": used, "input_tokens": tin, "output_tokens": 0,
          "cost_usd": cost, "latency_s": round(time.perf_counter() - t0, 3), "n_inputs": len(texts)})
    return vectors


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0
