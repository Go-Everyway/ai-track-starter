"""A tiny serverless function: one fixed question to a model, answered as JSON.

Module 0 deploys this folder to Vercel, which loads the Flask `app` from app.py.
It reads the same settings as lib/llm.py, from environment variables set on the host:

    LLM_API_KEY    your provider key. Set it on the host, never in this file.
    LLM_MODEL      a chat model id from your provider's list
    LLM_BASE_URL   default https://openrouter.ai/api/v1

With no key it answers in mock mode, so you can test it anywhere for free.

The prompt is fixed and replies are capped at 60 tokens on purpose. Anyone who finds a
public URL can call it, and every call spends your money. Never let the caller choose
the prompt or the length on an endpoint like this.
"""

import json
import os
import time
import urllib.error
import urllib.request

try:
    from flask import Flask, jsonify
except ImportError:  # the local check in the notebook runs without Flask installed
    Flask = None

QUESTION = "In one sentence, what is inventory forecasting software for?"
MAX_TOKENS = 60


def answer(env=None) -> dict:
    """Ask the fixed question and return the reply with its token counts and cost."""
    env = os.environ if env is None else env
    key = env.get("LLM_API_KEY", "")
    model = env.get("LLM_MODEL", "")
    base_url = env.get("LLM_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
    t0 = time.perf_counter()

    if not key:
        reply = f"[mock reply] {QUESTION}"
        return {"ok": True, "mock": True, "question": QUESTION, "reply": reply, "model": "mock",
                "input_tokens": len(QUESTION) // 4, "output_tokens": len(reply) // 4,
                "cost_usd": 0.0, "latency_s": round(time.perf_counter() - t0, 3)}
    if not model:
        return {"ok": False, "error": "Set LLM_MODEL on the host."}

    body = {"model": model, "max_tokens": MAX_TOKENS,
            "messages": [{"role": "user", "content": QUESTION}]}
    if "openrouter.ai" in base_url:
        body["usage"] = {"include": True}  # ask OpenRouter to report the cost
    req = urllib.request.Request(
        f"{base_url}/chat/completions", data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as e:  # report the status, never the key
        return {"ok": False, "error": f"provider returned HTTP {e.code}"}
    except (urllib.error.URLError, TimeoutError) as e:
        return {"ok": False, "error": f"could not reach the provider: {e}"}

    usage = data.get("usage") or {}
    return {"ok": True, "mock": False, "question": QUESTION,
            "reply": data["choices"][0]["message"]["content"], "model": data.get("model", model),
            "input_tokens": usage.get("prompt_tokens"), "output_tokens": usage.get("completion_tokens"),
            "cost_usd": usage.get("cost"), "latency_s": round(time.perf_counter() - t0, 3)}


if Flask is not None:
    app = Flask(__name__)

    @app.get("/")
    def index():
        result = answer()
        return jsonify(result), (200 if result["ok"] else 502)
