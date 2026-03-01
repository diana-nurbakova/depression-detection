"""Provider-agnostic LLM client using the OpenAI-compatible chat completions API.

Adapted from code_examples/llm_client.py with proven anti-disconnection
patterns for remote Ollama servers:
- Streaming (keeps connection alive for slow 70B models)
- Generous read timeout (300s between chunks)
- Auto /v1 append for Ollama behind reverse proxy
- Exponential backoff retries
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
    if m.startswith(("o1", "o3", "o4")):
        return True
    if "gpt-5-nano" in m:
        return True
    return False


@dataclass
class LLMClient:
    """OpenAI-compatible chat completions client with streaming."""

    provider: str
    base_url: str
    api_key: str
    model: str
    temperature: float = 0.3
    max_tokens: int = 512
    max_retries: int = 5
    rate_limit_delay: float = 0.5
    timeout: int = 300

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
        """Send a chat completion request with streaming.

        Returns:
            Parsed response dict in OpenAI format with usage info.

        Raises:
            requests.HTTPError: After exhausting retries.
        """
        use_native_ollama = (
            self.provider == "ollama" and "/v1" not in self.base_url
        )
        if use_native_ollama:
            url = f"{self.base_url.rstrip('/')}/api/chat"
        else:
            url = f"{self.base_url.rstrip('/')}/chat/completions"

        headers: dict[str, str] = {"Content-Type": "application/json"}
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
            payload: dict = {
                "model": self.model,
                "messages": messages,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            if not is_reasoning:
                payload["temperature"] = temp
            if self.provider == "openai":
                payload["max_completion_tokens"] = mtok
            else:
                payload["max_tokens"] = mtok

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                t0 = time.monotonic()
                read_timeout = max(self.timeout, 300)
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=(30, read_timeout),
                    stream=True,
                )
                response.raise_for_status()

                result = self._consume_stream(response, use_native_ollama)
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

                # Reasoning token breakdown
                completion_details = usage.get(
                    "completion_tokens_details", {},
                )
                reasoning_tok = completion_details.get("reasoning_tokens", 0)

                logger.info(
                    "LLM call %d [%s/%s] %.1fs — prompt=%d, "
                    "completion=%d (reasoning=%d), content_len=%d",
                    self._call_count, self.provider, self.model,
                    elapsed, prompt_tok, completion_tok,
                    reasoning_tok, len(content),
                )

                # Store stream metadata in result for the call logger
                result["_meta"] = {
                    "latency_ms": round(elapsed_ms, 1),
                    "time_to_first_chunk_ms": result.get(
                        "_time_to_first_chunk_ms",
                    ),
                    "stream_chunks": result.get("_stream_chunks", 0),
                }

                if self.rate_limit_delay > 0:
                    time.sleep(self.rate_limit_delay)

                return result

            except requests.exceptions.HTTPError as e:
                last_error = e
                status = (
                    e.response.status_code if e.response is not None else None
                )
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
            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
            ) as e:
                last_error = e
                wait = 2 ** attempt
                logger.warning(
                    "LLM connection error (attempt %d/%d), "
                    "retrying in %ds: %s",
                    attempt, self.max_retries, wait, e,
                )
                time.sleep(wait)

        raise last_error  # type: ignore[misc]

    def get_content(self, response: dict) -> str:
        """Extract the assistant message content from a response."""
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            logger.error(
                "Unexpected response structure: %s",
                json.dumps(response)[:500],
            )
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
        """Read an SSE / NDJSON stream and assemble into a single response."""
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

                if native_ollama:
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

        ttfc_ms = None
        if t_first_chunk is not None:
            ttfc_ms = round((t_first_chunk - t_start) * 1000, 1)

        result = {
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
            "_time_to_first_chunk_ms": ttfc_ms,
            "_stream_chunks": chunk_count,
        }
        return result

    @property
    def stats(self) -> dict:
        avg_latency = (
            round(self._total_latency_ms / self._call_count, 1)
            if self._call_count > 0 else 0.0
        )
        tokens_per_sec = (
            round(
                self._total_tokens / (self._total_latency_ms / 1000), 1,
            )
            if self._total_latency_ms > 0 else 0.0
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


def make_llm_client(provider_config) -> LLMClient:
    """Create an LLMClient from an LLMProviderConfig.

    Args:
        provider_config: An LLMProviderConfig from the pipeline config.
    """
    base_url = provider_config.base_url

    # Auto-append /v1 for Ollama behind a reverse proxy
    if provider_config.name == "ollama" and "/v1" not in base_url:
        base_url = base_url.rstrip("/") + "/v1"

    return LLMClient(
        provider=provider_config.name,
        base_url=base_url,
        api_key=provider_config.api_key,
        model=provider_config.model,
        temperature=provider_config.temperature,
        max_tokens=provider_config.max_tokens,
    )
