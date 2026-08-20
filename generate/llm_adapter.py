"""
LLM adapter — the one place a specific model provider is named.

generate/narrative_generator.py (and anything else that needs an LLM call)
depends only on the LLMAdapter interface, never on a specific provider's
SDK. Swapping providers later (e.g. to Anthropic or OpenAI, per plan.md's
original assumption) means adding one more subclass here and one more
branch in get_default_adapter() — no changes anywhere else.

Three implementations exist, all hit in practice during this build:
Gemini (free tier: 20 requests/day/model — exhausted first), Groq (free
tier: 200k tokens/day/model — exhausted second, across both viable
models), and Ollama (local inference — no quota, limited instead by this
machine's GPU memory). None of these is Anthropic/OpenAI as plan.md
originally assumed, but the point of this file is that it no longer
matters which one it is — see get_default_adapter() for how the active
one is selected.
"""

import json
import os
import time
from abc import ABC, abstractmethod

from dotenv import load_dotenv

load_dotenv()

_MAX_RETRIES = 5


class LLMAdapter(ABC):
    @abstractmethod
    def generate_json(self, prompt: str, schema: dict) -> dict:
        """Call the model and return a dict parsed from its structured JSON
        response, validated against `schema` (a JSON Schema object)."""


class GeminiAdapter(LLMAdapter):
    # Default model is gemini-2.5-flash-lite, not gemini-2.5-flash: on this
    # key's free tier, gemini-2.5-flash is capped at 20 requests/DAY (hit
    # that during initial smoke-testing, well short of the ~200-session
    # generation run this is for). flash-lite has a separate, much higher
    # free-tier quota bucket. Space calls out preemptively rather than
    # relying on retry-after-429 for every single call.
    _MIN_INTERVAL_SECONDS = 4.0

    def __init__(self, api_key: str | None = None, model: str = "gemini-2.5-flash-lite"):
        from google import genai

        self._client = genai.Client(api_key=api_key or os.environ["GOOGLE_API_KEY"])
        self._model = model
        self._last_call_at = 0.0

    def generate_json(self, prompt: str, schema: dict) -> dict:
        from google.genai import errors

        elapsed = time.monotonic() - self._last_call_at
        if elapsed < self._MIN_INTERVAL_SECONDS:
            time.sleep(self._MIN_INTERVAL_SECONDS - elapsed)

        for attempt in range(_MAX_RETRIES):
            try:
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=prompt,
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": schema,
                    },
                )
                self._last_call_at = time.monotonic()
                return json.loads(response.text)
            except errors.ClientError as e:
                self._last_call_at = time.monotonic()
                if e.code != 429 or attempt == _MAX_RETRIES - 1:
                    raise
                delay = _retry_delay_seconds(e) or 15
                time.sleep(delay)
            except errors.ServerError as e:
                # Transient overload (503) — not a quota problem, back off and retry.
                self._last_call_at = time.monotonic()
                if attempt == _MAX_RETRIES - 1:
                    raise
                time.sleep(15 * (attempt + 1))
        raise RuntimeError("unreachable")  # loop always returns or raises


def _retry_delay_seconds(error) -> float | None:
    """Pulls the server-suggested retry delay out of a 429 error, if present."""
    try:
        details = error.details.get("error", {}).get("details", [])
        for d in details:
            if d.get("@type", "").endswith("RetryInfo"):
                return float(d["retryDelay"].rstrip("s"))
    except (AttributeError, KeyError, ValueError):
        pass
    return None


_GROQ_MAX_RETRY_SLEEP_SECONDS = 20.0  # beyond this, fail fast instead of blocking silently


class GroqQuotaExhausted(RuntimeError):
    """Raised instead of retrying when Groq reports a long/daily-scale
    wait. A short per-minute rate limit is worth sleeping through; a
    multi-minute-or-longer one almost always means the day's token/request
    budget for this model is gone, and blindly retrying just produces a
    long silent-looking hang with no better odds of success."""


class GroqAdapter(LLMAdapter):
    # openai/gpt-oss-120b's free-tier daily token budget (200k TPD) was
    # exhausted during testing this build; openai/gpt-oss-20b is a
    # separate model with its own separate quota bucket.
    _MIN_INTERVAL_SECONDS = 1.0

    def __init__(self, api_key: str | None = None, model: str = "openai/gpt-oss-20b"):
        from groq import Groq

        self._client = Groq(api_key=api_key or os.environ["GROQ_API_KEY"])
        self._model = model
        self._last_call_at = 0.0

    def generate_json(self, prompt: str, schema: dict) -> dict:
        import groq

        elapsed = time.monotonic() - self._last_call_at
        if elapsed < self._MIN_INTERVAL_SECONDS:
            time.sleep(self._MIN_INTERVAL_SECONDS - elapsed)

        strict_schema = _require_no_additional_properties(schema)
        for attempt in range(_MAX_RETRIES):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {"name": "session_fields", "schema": strict_schema, "strict": True},
                    },
                )
                self._last_call_at = time.monotonic()
                return json.loads(response.choices[0].message.content)
            except groq.RateLimitError as e:
                self._last_call_at = time.monotonic()
                delay = _groq_retry_delay_seconds(e) or 10
                if delay > _GROQ_MAX_RETRY_SLEEP_SECONDS:
                    raise GroqQuotaExhausted(
                        f"model={self._model} suggested a {delay:.0f}s wait — treating as "
                        f"quota exhaustion, not a transient limit. Original: {e}"
                    ) from e
                if attempt == _MAX_RETRIES - 1:
                    raise
                time.sleep(delay)
            except groq.BadRequestError as e:
                # The smaller/fast models occasionally emit output that fails
                # strict JSON-schema validation server-side — a one-off
                # generation glitch, not a real bad request. Worth a couple
                # of quick retries (same prompt, fresh sample) before giving up.
                self._last_call_at = time.monotonic()
                code = (getattr(e, "body", None) or {}).get("error", {}).get("code")
                if code != "json_validate_failed" or attempt == _MAX_RETRIES - 1:
                    raise
                time.sleep(2)
            except groq.APIStatusError as e:
                self._last_call_at = time.monotonic()
                if e.status_code < 500 or attempt == _MAX_RETRIES - 1:
                    raise
                time.sleep(10 * (attempt + 1))
        raise RuntimeError("unreachable")  # loop always returns or raises


def _require_no_additional_properties(schema: dict) -> dict:
    """Groq/OpenAI strict JSON-schema mode requires additionalProperties:
    false on every object node. Injected here, adapter-side, so the plain
    JSON Schema objects narrative_generator.py defines stay
    provider-agnostic rather than carrying a Groq-specific quirk."""
    schema = dict(schema)
    if schema.get("type") == "object":
        schema["additionalProperties"] = False
        if "properties" in schema:
            schema["properties"] = {
                k: _require_no_additional_properties(v) for k, v in schema["properties"].items()
            }
    elif schema.get("type") == "array" and "items" in schema:
        schema["items"] = _require_no_additional_properties(schema["items"])
    return schema


def _groq_retry_delay_seconds(error) -> float | None:
    """Groq returns a Retry-After header on 429s."""
    try:
        return float(error.response.headers.get("retry-after"))
    except (AttributeError, TypeError, ValueError):
        return None


class OllamaAdapter(LLMAdapter):
    """Local inference via Ollama (http://localhost:11434) — no API key,
    no daily quota, since it runs on this machine's own GPU/CPU. Added
    after both Gemini and Groq hit free-tier daily token caps mid-run;
    trades quota limits for local compute limits (model size vs available
    VRAM) instead.

    qwen2.5:3b-instruct (chosen for a 6GB-VRAM card) needs an explicit
    system message or it drifts into replying as a customer-service
    chatbot (e.g. emitting a greeting instead of session data) rather than
    filling out the requested fields — the bigger hosted models didn't
    need this. Kept adapter-side rather than changed in
    narrative_generator.py's prompts, since it's a quirk of this specific
    small local model, not the task description."""

    _SYSTEM_PROMPT = (
        "You output ONLY a JSON record describing a fictional data sample "
        "for a fraud-detection dataset. You are not a chatbot and must not "
        "greet or converse. Fill every field with concrete, specific "
        "fictional content matching what's asked."
    )

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "qwen2.5:3b-instruct"):
        self._base_url = base_url
        self._model = model

    def generate_json(self, prompt: str, schema: dict) -> dict:
        import requests

        response = requests.post(
            f"{self._base_url}/api/chat",
            json={
                "model": self._model,
                "messages": [
                    {"role": "system", "content": self._SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "format": schema,
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
        return json.loads(response.json()["message"]["content"])


def get_default_adapter() -> LLMAdapter:
    """Picks an adapter based on which provider's API key is set in the
    environment. Checked in this order; extend here when adding a provider."""
    if os.environ.get("USE_OLLAMA"):
        return OllamaAdapter()
    if os.environ.get("GROQ_API_KEY"):
        return GroqAdapter()
    if os.environ.get("GOOGLE_API_KEY"):
        return GeminiAdapter()
    raise RuntimeError(
        "No supported LLM API key found. Set GROQ_API_KEY, GOOGLE_API_KEY, "
        "or USE_OLLAMA=1 in the repo-root .env file."
    )
