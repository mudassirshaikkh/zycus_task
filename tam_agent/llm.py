"""Thin LLM client. Pick a provider with environment variables (see .env.example).

Supported out of the box:
  LLM_PROVIDER=ollama     free, local (https://ollama.com) - no API key needed
  LLM_PROVIDER=gemini     Google AI Studio (free tier). Needs GEMINI_API_KEY, no extra package
  LLM_PROVIDER=anthropic  needs ANTHROPIC_API_KEY and `pip install anthropic`
  LLM_PROVIDER=openai     needs OPENAI_API_KEY and `pip install openai`

You may replace this file, add native tool calling, or use a framework
(LangGraph, LlamaIndex, etc.). Keep the provider configurable via env vars.
"""
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

# minimal .env loader (no extra dependency)
_env = Path(__file__).resolve().parent.parent / ".env"
if _env.exists():
    for _line in _env.read_text().splitlines():
        _line = _line.split("#", 1)[0].strip()
        if "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())


class LLMClient:
    def __init__(self, provider: str | None = None, model: str | None = None):
        self.provider = (provider or os.getenv("LLM_PROVIDER", "ollama")).lower()
        self.model = model or os.getenv("LLM_MODEL", {
            "ollama": "llama3.1:8b", "anthropic": "claude-sonnet-4-5", "openai": "gpt-4o-mini",
            "gemini": "gemini-2.5-flash"}[self.provider])
        self.calls = 0  # simple counter you can report for cost/latency

    def chat(self, messages: list[dict], system: str | None = None, max_tokens: int = 1024,
             temperature: float = 0.0) -> str:
        """messages: [{"role": "user"|"assistant", "content": str}] -> assistant text."""
        self.calls += 1
        if self.provider == "anthropic":
            import anthropic
            client = anthropic.Anthropic()
            resp = client.messages.create(model=self.model, max_tokens=max_tokens, temperature=temperature,
                                          system=system or "", messages=messages)
            return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        if self.provider == "openai":
            from openai import OpenAI
            client = OpenAI()
            msgs = ([{"role": "system", "content": system}] if system else []) + messages
            resp = client.chat.completions.create(model=self.model, messages=msgs, max_tokens=max_tokens,
                                                  temperature=temperature)
            return resp.choices[0].message.content or ""
        if self.provider == "ollama":
            host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
            msgs = ([{"role": "system", "content": system}] if system else []) + messages
            body = json.dumps({"model": self.model, "messages": msgs, "stream": False,
                               "options": {"temperature": temperature, "num_predict": max_tokens,
                                           "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "8192"))}}).encode()
            req = urllib.request.Request(f"{host}/api/chat", data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as r:
                return json.loads(r.read())["message"]["content"]
        if self.provider == "gemini":  # Google AI Studio free tier via its OpenAI-compatible endpoint (no SDK needed)
            key = os.getenv("GEMINI_API_KEY")
            if not key:
                raise RuntimeError("GEMINI_API_KEY is not set (put it in .env; never commit that file)")
            host = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai")
            msgs = ([{"role": "system", "content": system}] if system else []) + messages
            body = json.dumps({"model": self.model, "messages": msgs, "max_tokens": max_tokens,
                               "temperature": temperature}).encode()
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
            for attempt in range(6):  # free tier is rate limited: back off and retry on 429 / 5xx
                req = urllib.request.Request(f"{host}/chat/completions", data=body, headers=headers)
                try:
                    with urllib.request.urlopen(req, timeout=120) as r:
                        return json.loads(r.read())["choices"][0]["message"]["content"] or ""
                except urllib.error.HTTPError as e:
                    if e.code in (429, 500, 502, 503, 504) and attempt < 5:
                        time.sleep(min(60, 5 * 2 ** attempt))
                        continue
                    raise RuntimeError(f"Gemini API error {e.code}: {e.read().decode(errors='replace')[:300]}") from None
                except urllib.error.URLError:
                    if attempt < 5:
                        time.sleep(5 * (attempt + 1))
                        continue
                    raise
        raise ValueError(f"Unknown LLM_PROVIDER: {self.provider}")
