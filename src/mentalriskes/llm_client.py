"""LLM client for MentalRiskES Task 1.

Supports five providers:
  - ollama:      Ollama native or /v1 OpenAI-compatible endpoint (local GPU)
  - together:    TogetherAI API (https://api.together.xyz/v1) — streaming, pay-per-token
  - deepinfra:   DeepInfra API (https://api.deepinfra.com/v1/openai) — streaming, pay-per-token
  - openai:      OpenAI-compatible (non-streaming by default; set base_url for any endpoint)
  - huggingface: HF Inference API via huggingface_hub (serverless, no GPU needed)

DeepInfra is the recommended remote option for the test replay: Llama-3.3-70B-Instruct
at competitive per-token pricing with stable streaming and no Hugging Face TPM caps.

All providers expose the same interface: complete(), chat_completion(), stats.
Use create_llm_client() factory to get the right one from config.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field

import requests

from .config import LLMConfig

logger = logging.getLogger(__name__)


@dataclass
class LLMClient:
    """OpenAI-compatible chat completions client with streaming support."""

    provider: str = "ollama"
    base_url: str = ""
    api_key: str = ""
    model: str = "llama3.3:70b"
    temperature: float = 0.1
    max_tokens: int = 4096
    max_retries: int = 3
    rate_limit_delay: float = 1.0
    timeout: int = 180

    # Fallback client (used when primary fails after all retries)
    _fallback: LLMClient | None = field(default=None, init=False, repr=False)

    # Tracking
    _call_count: int = field(default=0, init=False, repr=False)
    _total_tokens: int = field(default=0, init=False, repr=False)
    _total_latency_ms: float = field(default=0.0, init=False, repr=False)
    _fallback_count: int = field(default=0, init=False, repr=False)

    @classmethod
    def from_config(cls, cfg: LLMConfig, model_override: str | None = None) -> LLMClient:
        return cls(
            provider=cfg.provider,
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            model=model_override or cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            timeout=cfg.timeout,
        )

    def with_fallback(self, fallback: LLMClient) -> LLMClient:
        """Attach a fallback client used when the primary fails after all retries."""
        self._fallback = fallback
        return self

    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Send chat completion and return content string."""
        response = self.chat_completion(messages, temperature, max_tokens)
        return self._get_content(response)

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict:
        """Send chat completion, falling back to secondary client on failure."""
        try:
            return self._do_chat_completion(messages, temperature, max_tokens)
        except Exception as primary_err:
            if self._fallback is None:
                raise
            logger.warning(
                "Primary LLM failed (%s/%s): %s — falling back to %s/%s",
                self.provider, self.model, primary_err,
                self._fallback.provider, self._fallback.model,
            )
            self._fallback_count += 1
            return self._fallback._do_chat_completion(messages, temperature, max_tokens)

    def _do_chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict:
        """Send a chat completion request and return response."""
        use_native_ollama = self.provider == "ollama" and "/v1" not in self.base_url
        # Stream for: Ollama (native + /v1), Together, DeepInfra (avoids long non-streaming timeouts).
        # Non-streaming for: generic openai-compatible endpoints (simpler, safer).
        use_streaming = use_native_ollama or self.provider in ("ollama", "together", "deepinfra")

        if use_native_ollama:
            url = f"{self.base_url.rstrip('/')}/api/chat"
        else:
            url = f"{self.base_url.rstrip('/')}/chat/completions"

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        temp = temperature if temperature is not None else self.temperature
        mtok = max_tokens if max_tokens is not None else self.max_tokens

        if use_native_ollama:
            payload: dict = {
                "model": self.model,
                "messages": messages,
                "stream": True,
                "options": {"temperature": temp, "num_predict": mtok},
            }
        elif use_streaming:
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": True,
                "stream_options": {"include_usage": True},
                "temperature": temp,
                "max_tokens": mtok,
            }
        else:
            # Newer OpenAI models (gpt-5-nano, o-series) require max_completion_tokens
            # and don't support temperature
            is_new_openai = any(self.model.startswith(p) for p in ("gpt-5-nano", "gpt-5-mini", "o1", "o3", "o4"))
            payload: dict = {
                "model": self.model,
                "messages": messages,
                "stream": False,
            }
            if is_new_openai:
                payload["max_completion_tokens"] = mtok
                # These models only support temperature=1 (default); omit it
            else:
                payload["max_tokens"] = mtok
                payload["temperature"] = temp

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                t0 = time.monotonic()
                read_timeout = max(self.timeout, 300)
                if use_streaming:
                    response = requests.post(
                        url, headers=headers, json=payload,
                        timeout=(30, read_timeout), stream=True,
                    )
                    response.raise_for_status()
                    result = self._consume_stream(response, use_native_ollama)
                else:
                    response = requests.post(
                        url, headers=headers, json=payload,
                        timeout=(30, read_timeout),
                    )
                    response.raise_for_status()
                    result = response.json()
                elapsed = time.monotonic() - t0

                self._call_count += 1
                usage = result.get("usage", {})
                self._total_tokens += usage.get("total_tokens", 0)
                self._total_latency_ms += elapsed * 1000

                content = self._get_content(result)
                logger.info(
                    "LLM call %d [%s/%s] %.1fs — tokens=%d, content_len=%d",
                    self._call_count, self.provider, self.model, elapsed,
                    usage.get("total_tokens", 0), len(content),
                )

                if self.rate_limit_delay > 0:
                    time.sleep(self.rate_limit_delay)
                return result

            except requests.exceptions.HTTPError as e:
                last_error = e
                status = e.response.status_code if e.response is not None else None
                if status == 429 or (status is not None and status >= 500):
                    wait = 2 ** attempt
                    logger.warning("LLM call failed (attempt %d/%d, status %s), retrying in %ds",
                                   attempt, self.max_retries, status, wait)
                    time.sleep(wait)
                else:
                    raise
            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError,
            ) as e:
                last_error = e
                wait = 2 ** attempt
                logger.warning(
                    "LLM transport error (attempt %d/%d): %s, retrying in %ds",
                    attempt, self.max_retries, type(e).__name__, wait,
                )
                time.sleep(wait)
            except Exception as e:
                # Catch stream disconnections and other transient errors
                err_name = type(e).__name__
                err_str = str(e)
                if (
                    "Disconnected" in err_name
                    or "RemoteDisconnected" in err_str
                    or "ConnectionReset" in err_name
                    or "IncompleteRead" in err_name
                    or "ChunkedEncodingError" in err_name
                    or "ProtocolError" in err_name
                ):
                    last_error = e
                    wait = 2 ** attempt
                    logger.warning("LLM stream error (attempt %d/%d): %s, retrying in %ds",
                                   attempt, self.max_retries, err_name, wait)
                    time.sleep(wait)
                else:
                    raise

        raise last_error  # type: ignore[misc]

    @staticmethod
    def _get_content(response: dict) -> str:
        try:
            return response["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            return ""

    @staticmethod
    def _consume_stream(response: requests.Response, native_ollama: bool) -> dict:
        """Read SSE / NDJSON stream and assemble into a single response dict."""
        content_parts: list[str] = []
        usage: dict = {}

        try:
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                if native_ollama:
                    chunk = json.loads(raw_line)
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        content_parts.append(token)
                    if chunk.get("done"):
                        pt = chunk.get("prompt_eval_count", 0)
                        ct = chunk.get("eval_count", 0)
                        usage = {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}
                else:
                    line = raw_line
                    if line.startswith("data: "):
                        line = line[6:]
                    elif line.startswith("data:"):
                        line = line[5:]
                    else:
                        continue
                    if line.strip() == "[DONE]":
                        break
                    chunk = json.loads(line)
                    choices = chunk.get("choices", [])
                    if choices:
                        token = choices[0].get("delta", {}).get("content", "")
                        if token:
                            content_parts.append(token)
                    if "usage" in chunk:
                        usage = chunk["usage"]
        finally:
            response.close()

        return {
            "choices": [{"message": {"role": "assistant", "content": "".join(content_parts)}}],
            "usage": usage,
        }

    @property
    def stats(self) -> dict:
        s = {
            "provider": self.provider,
            "model": self.model,
            "call_count": self._call_count,
            "total_tokens": self._total_tokens,
            "total_latency_ms": round(self._total_latency_ms, 1),
        }
        if self._fallback_count > 0:
            s["fallback_count"] = self._fallback_count
            s["fallback_provider"] = self._fallback.provider if self._fallback else None
            s["fallback_model"] = self._fallback.model if self._fallback else None
        return s


# ---------------------------------------------------------------------------
# HuggingFace Inference API client (serverless, no GPU)
# ---------------------------------------------------------------------------


@dataclass
class HFInferenceClient:
    """HuggingFace Inference API client with the same interface as LLMClient.

    Uses huggingface_hub.InferenceClient under the hood. No GPU or local
    model needed — inference runs on HF's serverless infrastructure.

    Requires HF_TOKEN env var or api_key parameter.
    """

    provider: str = "huggingface"
    base_url: str = ""  # unused, kept for interface compatibility
    api_key: str = ""   # HF token
    model: str = "meta-llama/Llama-3.3-70B-Instruct"
    temperature: float = 0.1
    max_tokens: int = 4096
    max_retries: int = 3
    rate_limit_delay: float = 1.0
    timeout: int = 180

    # Fallback client (used when primary fails after all retries)
    _fallback: object | None = field(default=None, init=False, repr=False)

    # Tracking
    _call_count: int = field(default=0, init=False, repr=False)
    _total_tokens: int = field(default=0, init=False, repr=False)
    _total_latency_ms: float = field(default=0.0, init=False, repr=False)
    _fallback_count: int = field(default=0, init=False, repr=False)

    # Lazy-loaded client
    _client: object = field(default=None, init=False, repr=False)

    def with_fallback(self, fallback) -> HFInferenceClient:
        """Attach a fallback client used when the primary fails after all retries."""
        self._fallback = fallback
        return self

    @classmethod
    def from_config(cls, cfg: LLMConfig, model_override: str | None = None) -> HFInferenceClient:
        import os
        api_key = cfg.api_key or os.environ.get("HF_TOKEN", "")
        return cls(
            provider="huggingface",
            api_key=api_key,
            model=model_override or cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            max_retries=5,          # more retries for rate limits
            rate_limit_delay=2.0,   # 2s between calls to avoid TPM spikes
            timeout=cfg.timeout,
        )

    def _get_client(self):
        if self._client is None:
            from huggingface_hub import InferenceClient
            self._client = InferenceClient(
                model=self.model,
                token=self.api_key,
                timeout=self.timeout,
            )
        return self._client

    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Send chat completion and return content string."""
        response = self.chat_completion(messages, temperature, max_tokens)
        return self._get_content(response)

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict:
        """Send chat completion via HF Inference API."""
        temp = temperature if temperature is not None else self.temperature
        mtok = max_tokens if max_tokens is not None else self.max_tokens
        client = self._get_client()

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                t0 = time.monotonic()
                response = client.chat_completion(
                    messages=messages,
                    temperature=max(temp, 0.01),  # HF doesn't accept 0
                    max_tokens=mtok,
                )
                elapsed = time.monotonic() - t0

                content = response.choices[0].message.content or ""
                usage = getattr(response, "usage", None)
                prompt_tok = getattr(usage, "prompt_tokens", 0) if usage else 0
                completion_tok = getattr(usage, "completion_tokens", 0) if usage else 0
                total_tok = prompt_tok + completion_tok

                self._call_count += 1
                self._total_tokens += total_tok
                self._total_latency_ms += elapsed * 1000

                logger.info(
                    "HF call %d [%s] %.1fs — tokens=%d, content_len=%d",
                    self._call_count, self.model, elapsed, total_tok, len(content),
                )

                if self.rate_limit_delay > 0:
                    time.sleep(self.rate_limit_delay)

                return {
                    "choices": [{"message": {"role": "assistant", "content": content}}],
                    "usage": {
                        "prompt_tokens": prompt_tok,
                        "completion_tokens": completion_tok,
                        "total_tokens": total_tok,
                    },
                }

            except Exception as e:
                last_error = e
                err_str = str(e)
                # Rate limit: wait longer (parse retry-after or use 10s base)
                if "429" in err_str or "rate limit" in err_str.lower():
                    wait = max(10, 2 ** (attempt + 1))
                    logger.warning(
                        "HF rate limited (attempt %d/%d), waiting %ds",
                        attempt, self.max_retries, wait,
                    )
                else:
                    wait = 2 ** attempt
                    logger.warning(
                        "HF call failed (attempt %d/%d): %s — retrying in %ds",
                        attempt, self.max_retries, e, wait,
                    )
                if attempt < self.max_retries:
                    time.sleep(wait)

        # All retries exhausted — try fallback
        if self._fallback is not None:
            logger.warning(
                "HF primary failed (%s): %s — falling back to %s/%s",
                self.model, last_error,
                getattr(self._fallback, 'provider', '?'),
                getattr(self._fallback, 'model', '?'),
            )
            self._fallback_count += 1
            return self._fallback.chat_completion(messages, temperature, max_tokens)

        raise last_error  # type: ignore[misc]

    @staticmethod
    def _get_content(response: dict) -> str:
        try:
            return response["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            return ""

    @property
    def stats(self) -> dict:
        s = {
            "provider": self.provider,
            "model": self.model,
            "call_count": self._call_count,
            "total_tokens": self._total_tokens,
            "total_latency_ms": round(self._total_latency_ms, 1),
        }
        if self._fallback_count > 0:
            s["fallback_count"] = self._fallback_count
            s["fallback_provider"] = getattr(self._fallback, 'provider', None)
            s["fallback_model"] = getattr(self._fallback, 'model', None)
        return s


# ---------------------------------------------------------------------------
# Factory: create the right client from config
# ---------------------------------------------------------------------------


_TOGETHER_DEFAULT_BASE_URL = "https://api.together.xyz/v1"
_DEEPINFRA_DEFAULT_BASE_URL = "https://api.deepinfra.com/v1/openai"


def create_llm_client(
    cfg: LLMConfig,
    model_override: str | None = None,
) -> LLMClient | HFInferenceClient:
    """Create an LLM client based on the provider in config.

    Supported providers:
      - "ollama":      local Ollama (streaming, native or /v1)
      - "together":    TogetherAI (streaming SSE, OpenAI-compatible)
                       base_url defaults to https://api.together.xyz/v1
                       api_key from TOGETHER_API_KEY env var
      - "deepinfra":   DeepInfra (streaming SSE, OpenAI-compatible)
                       base_url defaults to https://api.deepinfra.com/v1/openai
                       api_key from DEEPINFRA_API_KEY env var
      - "openai":      OpenAI or any OpenAI-compatible endpoint (non-streaming)
      - "huggingface": HF Inference API (serverless)

    Args:
        cfg: LLM configuration.
        model_override: Optional model name override (e.g., from run config).

    Returns:
        LLMClient (ollama/together/deepinfra/openai) or HFInferenceClient (huggingface).
    """
    if cfg.provider == "huggingface":
        logger.info("Using HuggingFace Inference API: %s", model_override or cfg.model)
        return HFInferenceClient.from_config(cfg, model_override)

    if cfg.provider == "together":
        import os
        base_url = cfg.base_url or os.environ.get("TOGETHER_BASE_URL", _TOGETHER_DEFAULT_BASE_URL)
        api_key = cfg.api_key or os.environ.get("TOGETHER_API_KEY", "")
        if not api_key:
            logger.warning("Together provider selected but TOGETHER_API_KEY is not set")
        client = LLMClient(
            provider="together",
            base_url=base_url,
            api_key=api_key,
            model=model_override or cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            timeout=cfg.timeout,
            rate_limit_delay=0.5,   # Together handles bursts well; small delay avoids RPM limits
        )
        logger.info("Using TogetherAI: %s @ %s", client.model, base_url)
        return client

    if cfg.provider == "deepinfra":
        import os
        base_url = cfg.base_url or os.environ.get("DEEPINFRA_BASE_URL", _DEEPINFRA_DEFAULT_BASE_URL)
        api_key = cfg.api_key or os.environ.get("DEEPINFRA_API_KEY", "")
        if not api_key:
            logger.warning("DeepInfra provider selected but DEEPINFRA_API_KEY is not set")
        client = LLMClient(
            provider="deepinfra",
            base_url=base_url,
            api_key=api_key,
            model=model_override or cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            timeout=cfg.timeout,
            rate_limit_delay=0.3,   # DeepInfra has generous throughput; small delay
        )
        logger.info("Using DeepInfra: %s @ %s", client.model, base_url)
        return client

    logger.info("Using %s client: %s", cfg.provider, model_override or cfg.model)
    return LLMClient.from_config(cfg, model_override)


def parse_json_response(text: str) -> dict | None:
    """Extract and parse JSON from LLM output.

    Handles:
    - JSON inside ```json ... ``` code fences
    - Bare JSON objects in text
    - Truncated JSON (attempts bracket-closing repair)
    """
    # Strategy 1: find JSON in markdown code fence
    fence_match = re.search(r'```(?:json)?\s*(\{[\s\S]*)', text)
    if fence_match:
        candidate = fence_match.group(1)
        # Try to find the closing fence
        end = candidate.find("```")
        if end > 0:
            candidate = candidate[:end].strip()
        result = _try_parse_json(candidate)
        if result is not None:
            return result

    # Strategy 2: find outermost JSON object using bracket matching
    start = text.find("{")
    if start >= 0:
        candidate = text[start:]
        # Find the matching closing brace by counting brackets
        depth = 0
        end = -1
        for i, ch in enumerate(candidate):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end > 0:
            result = _try_parse_json(candidate[:end + 1])
            if result is not None:
                return result

        # Strategy 3: truncated JSON — try to repair by closing open brackets
        result = _try_parse_json(candidate)
        if result is not None:
            return result

    return None


def _try_parse_json(text: str) -> dict | None:
    """Try to parse JSON, with truncation repair as fallback."""
    text = text.strip()
    # Direct parse
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # Truncation repair: close any unclosed brackets/braces
    # Strip trailing comma and whitespace
    repaired = re.sub(r',\s*$', '', text)
    # Count open vs close
    open_braces = repaired.count("{") - repaired.count("}")
    open_brackets = repaired.count("[") - repaired.count("]")

    if open_braces > 0 or open_brackets > 0:
        # Close any open strings (heuristic: odd number of unescaped quotes)
        # Simple approach: add closing brackets
        repaired += "]" * max(0, open_brackets)
        repaired += "}" * max(0, open_braces)
        try:
            result = json.loads(repaired)
            if isinstance(result, dict):
                logger.debug("Repaired truncated JSON (%d braces, %d brackets closed)",
                             open_braces, open_brackets)
                return result
        except json.JSONDecodeError:
            pass

    return None
