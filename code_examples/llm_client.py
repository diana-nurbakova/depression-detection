"""Provider-agnostic LLM client using the OpenAI-compatible chat completions API.

Supports Ollama, DeepSeek, Mistral, and OpenAI — all use the same
``/v1/chat/completions`` endpoint format. Uses ``requests`` directly
(consistent with the existing codebase).
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import requests

logger = logging.getLogger(__name__)


def _is_openai_reasoning_model(model: str) -> bool:
    """Check if a model is an OpenAI reasoning model (o-series, gpt-5-nano).

    Reasoning models don't support ``temperature`` and require
    ``developer`` role instead of ``system``.
    """
    m = model.lower()
    # o1, o3, o4-mini, etc.
    if m.startswith(("o1", "o3", "o4")):
        return True
    # gpt-5-nano is reasoning-based (all completion tokens are reasoning tokens)
    if "gpt-5-nano" in m:
        return True
    return False


@dataclass
class LLMClient:
    """OpenAI-compatible chat completions client."""

    provider: str
    base_url: str
    api_key: str
    model: str
    temperature: float = 0.3
    max_tokens: int = 1024
    max_retries: int = 3
    rate_limit_delay: float = 1.0
    timeout: int = 120

    # Tracking
    _call_count: int = field(default=0, init=False, repr=False)
    _total_tokens: int = field(default=0, init=False, repr=False)
    _total_prompt_tokens: int = field(default=0, init=False, repr=False)
    _total_completion_tokens: int = field(default=0, init=False, repr=False)
    _total_latency_ms: float = field(default=0.0, init=False, repr=False)

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict:
        """Send a chat completion request.

        Args:
            messages: List of message dicts with "role" and "content" keys.
            temperature: Override default temperature for this call.
            max_tokens: Override default max_tokens for this call.

        Returns:
            Parsed JSON response from the API.  For Ollama native API
            responses, the result is wrapped to match the OpenAI format
            so callers don't need to distinguish providers.

        Raises:
            requests.HTTPError: After exhausting retries.
        """
        use_native_ollama = self.provider == "ollama" and "/v1" not in self.base_url
        if use_native_ollama:
            url = f"{self.base_url.rstrip('/')}/api/chat"
        else:
            url = f"{self.base_url.rstrip('/')}/chat/completions"

        headers = {
            "Content-Type": "application/json",
        }

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        temp = temperature if temperature is not None else self.temperature
        mtok = max_tokens if max_tokens is not None else self.max_tokens

        is_reasoning = (
            self.provider == "openai"
            and _is_openai_reasoning_model(self.model)
        )

        # Reasoning models require "developer" role instead of "system"
        if is_reasoning:
            messages = [
                {**m, "role": "developer"} if m["role"] == "system" else m
                for m in messages
            ]

        if use_native_ollama:
            # Ollama native /api/chat format
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": True,
                "options": {
                    "temperature": temp,
                    "num_predict": mtok,
                },
            }
        else:
            # OpenAI-compatible format
            payload: dict = {
                "model": self.model,
                "messages": messages,
                "stream": True,
                # Request token usage in the final streamed chunk
                "stream_options": {"include_usage": True},
            }
            # Reasoning models don't support temperature (only default=1)
            if not is_reasoning:
                payload["temperature"] = temp
            # Newer OpenAI models (GPT-4o+, GPT-5) require
            # max_completion_tokens; older models and other providers
            # use max_tokens.
            if self.provider == "openai":
                payload["max_completion_tokens"] = mtok
            else:
                payload["max_tokens"] = mtok

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                t0 = time.monotonic()
                # Use (connect_timeout, read_timeout) tuple.
                # Read timeout must be generous for reasoning models
                # that may not send chunks during their thinking phase.
                read_timeout = max(self.timeout, 300)
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=(30, read_timeout),
                    stream=True,
                )
                elapsed_connect = time.monotonic() - t0

                response.raise_for_status()
                result = self._consume_stream(
                    response, use_native_ollama,
                )
                elapsed = time.monotonic() - t0

                # Track usage
                self._call_count += 1
                elapsed_ms = elapsed * 1000
                usage = result.get("usage", {})
                prompt_tok = usage.get("prompt_tokens", 0)
                completion_tok = usage.get("completion_tokens", 0)
                total_tok = usage.get("total_tokens", 0)

                self._total_tokens += total_tok
                self._total_prompt_tokens += prompt_tok
                self._total_completion_tokens += completion_tok
                self._total_latency_ms += elapsed_ms

                content = ""
                try:
                    content = result["choices"][0]["message"]["content"]
                except (KeyError, IndexError, TypeError):
                    pass
                content_len = len(content)

                # Reasoning token breakdown (OpenAI reasoning models)
                completion_details = usage.get("completion_tokens_details", {})
                reasoning_tok = completion_details.get("reasoning_tokens", 0)

                logger.info(
                    "LLM call %d [%s/%s] %.1fs — prompt=%d, "
                    "completion=%d (reasoning=%d), content_len=%d",
                    self._call_count,
                    self.provider,
                    self.model,
                    elapsed,
                    prompt_tok,
                    completion_tok,
                    reasoning_tok,
                    content_len,
                )

                # Emit structured record for JSONL handler
                logger.debug(
                    "llm_call",
                    extra={"api_call": {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "event_type": "llm_call",
                        "provider": self.provider,
                        "model": self.model,
                        "call_number": self._call_count,
                        "latency_ms": round(elapsed_ms, 1),
                        "prompt_tokens": prompt_tok,
                        "completion_tokens": completion_tok,
                        "reasoning_tokens": reasoning_tok,
                        "total_tokens": total_tok,
                        "content_length": content_len,
                        "success": True,
                        "url": url,
                    }},
                )

                # Rate limiting
                if self.rate_limit_delay > 0:
                    time.sleep(self.rate_limit_delay)

                return result

            except requests.exceptions.HTTPError as e:
                last_error = e
                status = e.response.status_code if e.response is not None else None
                # Log the error response body for diagnosis
                resp_body = ""
                if e.response is not None:
                    try:
                        resp_body = e.response.text[:500]
                    except Exception:
                        pass
                if status == 429 or (status is not None and status >= 500):
                    wait = 2 ** attempt
                    logger.warning(
                        "LLM call failed (attempt %d/%d, status %s), "
                        "retrying in %ds: %s — %s",
                        attempt, self.max_retries, status, wait, e,
                        resp_body,
                    )
                    time.sleep(wait)
                else:
                    logger.error(
                        "LLM call failed (status %s): %s — %s",
                        status, e, resp_body,
                    )
                    raise
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                last_error = e
                wait = 2 ** attempt
                logger.warning(
                    "LLM connection error (attempt %d/%d), retrying in %ds: %s",
                    attempt, self.max_retries, wait, e,
                )
                time.sleep(wait)

        raise last_error  # type: ignore[misc]

    def get_content(self, response: dict) -> str:
        """Extract the assistant message content from a chat completion response."""
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            logger.error("Unexpected response structure: %s", json.dumps(response)[:500])
            return ""

    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Convenience: send a chat completion and return the content string."""
        response = self.chat_completion(messages, temperature, max_tokens)
        return self.get_content(response)

    @staticmethod
    def _consume_stream(
        response: requests.Response,
        native_ollama: bool,
    ) -> dict:
        """Read an SSE / NDJSON stream and assemble into a single response dict.

        For OpenAI-compatible SSE streams, lines look like:
            data: {"choices":[{"delta":{"content":"tok"}}]}
            data: [DONE]

        For Ollama native NDJSON streams, lines look like:
            {"message":{"content":"tok"},"done":false}
            {"message":{"content":""},"done":true,...}

        Returns a dict in the standard OpenAI chat completion format.
        """
        content_parts: list[str] = []
        role = "assistant"
        finish_reason = "stop"
        usage: dict = {}
        chunk_count = 0
        t_first_chunk = None
        t_start = time.monotonic()

        try:
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue

                chunk_count += 1
                if t_first_chunk is None:
                    t_first_chunk = time.monotonic()
                    wait_s = t_first_chunk - t_start
                    logger.debug(
                        "Stream: first chunk after %.1fs", wait_s,
                    )

                # Log progress every 50 chunks
                if chunk_count % 50 == 0:
                    logger.debug(
                        "Stream: %d chunks, %d content parts so far "
                        "(%.1fs elapsed)",
                        chunk_count, len(content_parts),
                        time.monotonic() - t_start,
                    )

                if native_ollama:
                    # Ollama native NDJSON: each line is a JSON object
                    chunk = json.loads(raw_line)
                    msg = chunk.get("message", {})
                    token = msg.get("content", "")
                    if token:
                        content_parts.append(token)
                    if chunk.get("done"):
                        prompt_tok = chunk.get("prompt_eval_count", 0)
                        comp_tok = chunk.get("eval_count", 0)
                        usage = {
                            "prompt_tokens": prompt_tok,
                            "completion_tokens": comp_tok,
                            "total_tokens": prompt_tok + comp_tok,
                        }
                else:
                    # OpenAI SSE: lines prefixed with "data: "
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
                        delta = choices[0].get("delta", {})
                        token = delta.get("content", "")
                        if token:
                            content_parts.append(token)
                        if choices[0].get("finish_reason"):
                            finish_reason = choices[0]["finish_reason"]
                    if "usage" in chunk:
                        usage = chunk["usage"]
        finally:
            response.close()

        logger.debug(
            "Stream complete: %d chunks, %d content parts, %.1fs total",
            chunk_count, len(content_parts),
            time.monotonic() - t_start,
        )

        return {
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": role,
                        "content": "".join(content_parts),
                    },
                    "finish_reason": finish_reason,
                },
            ],
            "usage": usage,
        }

    @staticmethod
    def _normalize_ollama_response(ollama_resp: dict) -> dict:
        """Wrap an Ollama native /api/chat response into OpenAI format."""
        message = ollama_resp.get("message", {})
        # Ollama reports tokens in prompt_eval_count + eval_count
        prompt_tokens = ollama_resp.get("prompt_eval_count", 0)
        completion_tokens = ollama_resp.get("eval_count", 0)
        return {
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": message.get("role", "assistant"),
                        "content": message.get("content", ""),
                    },
                    "finish_reason": "stop",
                },
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    @property
    def stats(self) -> dict:
        avg_latency = (
            round(self._total_latency_ms / self._call_count, 1)
            if self._call_count > 0
            else 0.0
        )
        tokens_per_sec = (
            round(self._total_tokens / (self._total_latency_ms / 1000), 1)
            if self._total_latency_ms > 0
            else 0.0
        )
        return {
            "provider": self.provider,
            "model": self.model,
            "calls": self._call_count,
            "total_tokens": self._total_tokens,
            "prompt_tokens": self._total_prompt_tokens,
            "completion_tokens": self._total_completion_tokens,
            "total_latency_s": round(self._total_latency_ms / 1000, 2),
            "avg_latency_ms": avg_latency,
            "tokens_per_second": tokens_per_sec,
        }


def make_client(
    provider_config: dict,
    provider_name: str,
    model: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 1024,
    max_retries: int = 3,
    rate_limit_delay: float = 1.0,
) -> LLMClient:
    """Create an LLMClient from a provider config dict.

    The config dict should have keys: base_url or base_url_env,
    api_key_env, and models.
    """
    # Resolve base URL
    if "base_url_env" in provider_config:
        base_url = os.getenv(provider_config["base_url_env"], "")
    else:
        base_url = provider_config.get("base_url", "")

    if not base_url:
        raise ValueError(
            f"No base_url configured for provider {provider_name!r}"
        )

    # Ollama behind a reverse proxy typically only exposes OpenAI-compatible
    # endpoints (/v1/chat/completions).  Auto-append /v1 so users only need
    # to set the bare Ollama URL in their .env.
    if provider_name == "ollama" and "/v1" not in base_url:
        base_url = base_url.rstrip("/") + "/v1"

    # Resolve API key
    api_key_env = provider_config.get("api_key_env", "")
    api_key = os.getenv(api_key_env, "") if api_key_env else ""

    # Resolve model
    if model is None:
        available = provider_config.get("models", [])
        if not available:
            raise ValueError(f"No models configured for provider {provider_name!r}")
        model = available[0]

    return LLMClient(
        provider=provider_name,
        base_url=base_url,
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=max_retries,
        rate_limit_delay=rate_limit_delay,
    )


def make_client_from_config(
    raw_config: dict,
    provider: str | None = None,
    model: str | None = None,
) -> LLMClient:
    """Create an LLMClient from the full YAML config dict.

    Uses the ``detection.llm_providers`` and ``detection.llm_agents`` sections.
    """
    det = raw_config["detection"]
    agents_cfg = det["llm_agents"]

    provider = provider or agents_cfg["default_provider"]
    model = model or agents_cfg["default_model"]

    providers = det["llm_providers"]
    if provider not in providers:
        raise ValueError(
            f"Unknown provider {provider!r}. Available: {list(providers.keys())}"
        )

    return make_client(
        provider_config=providers[provider],
        provider_name=provider,
        model=model,
        temperature=agents_cfg.get("temperature", 0.3),
        max_tokens=agents_cfg.get("max_tokens", 1024),
        max_retries=agents_cfg.get("max_retries", 3),
        rate_limit_delay=agents_cfg.get("rate_limit_delay", 1.0),
    )
