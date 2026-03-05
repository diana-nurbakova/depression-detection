"""LLM client for Task 1 pipeline.

Wraps the battle-tested LLMClient from code_examples/llm_client.py,
adapting it for our config dataclasses and providing convenience helpers
(JSON parsing, per-agent factory).
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import requests

from .config import ModelConfig, OllamaConfig, OpenAIConfig, TogetherConfig, PipelineConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_openai_reasoning_model(model: str) -> bool:
    """Check if a model is an OpenAI reasoning model (o-series, gpt-5-nano)."""
    m = model.lower()
    if m.startswith(("o1", "o3", "o4")):
        return True
    if "gpt-5-nano" in m:
        return True
    return False


# ---------------------------------------------------------------------------
# Core client (streaming, retries, Ollama native + OpenAI SSE)
# ---------------------------------------------------------------------------


@dataclass
class LLMClient:
    """OpenAI-compatible chat completions client with streaming support."""

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
        """Send a streaming chat completion request and return assembled response."""
        use_native_ollama = self.provider == "ollama" and "/v1" not in self.base_url
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
            self.provider == "openai" and _is_openai_reasoning_model(self.model)
        )

        # Reasoning models: developer role instead of system
        if is_reasoning:
            messages = [
                {**m, "role": "developer"} if m["role"] == "system" else m
                for m in messages
            ]

        if use_native_ollama:
            payload: dict = {
                "model": self.model,
                "messages": messages,
                "stream": True,
                "options": {
                    "temperature": temp,
                    "num_predict": mtok,
                },
            }
        else:
            payload = {
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

                completion_details = usage.get("completion_tokens_details", {})
                reasoning_tok = completion_details.get("reasoning_tokens", 0)

                logger.info(
                    "LLM call %d [%s/%s] %.1fs — prompt=%d, "
                    "completion=%d (reasoning=%d), content_len=%d",
                    self._call_count, self.provider, self.model, elapsed,
                    prompt_tok, completion_tok, reasoning_tok, len(content),
                )

                if self.rate_limit_delay > 0:
                    time.sleep(self.rate_limit_delay)

                return result

            except requests.exceptions.HTTPError as e:
                last_error = e
                status = e.response.status_code if e.response is not None else None
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
                        attempt, self.max_retries, status, wait, e, resp_body,
                    )
                    time.sleep(wait)
                else:
                    logger.error("LLM call failed (status %s): %s — %s", status, e, resp_body)
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
        """Convenience: send chat completion and return content string."""
        response = self.chat_completion(messages, temperature, max_tokens)
        return self.get_content(response)

    @staticmethod
    def _consume_stream(
        response: requests.Response,
        native_ollama: bool,
    ) -> dict:
        """Read SSE / NDJSON stream and assemble into a single response dict."""
        content_parts: list[str] = []
        role = "assistant"
        finish_reason = "stop"
        usage: dict = {}
        chunk_count = 0
        t_start = time.monotonic()

        try:
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                chunk_count += 1

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

        logger.debug(
            "Stream complete: %d chunks, %d content parts, %.1fs",
            chunk_count, len(content_parts), time.monotonic() - t_start,
        )

        return {
            "choices": [{
                "index": 0,
                "message": {"role": role, "content": "".join(content_parts)},
                "finish_reason": finish_reason,
            }],
            "usage": usage,
        }

    @property
    def stats(self) -> dict:
        avg_latency = (
            round(self._total_latency_ms / self._call_count, 1)
            if self._call_count > 0 else 0.0
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
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_clients(config: PipelineConfig) -> dict[str, LLMClient]:
    """Create all LLM clients needed by the pipeline.

    Returns a dict keyed by role: interviewer, assessor, orchestrator, justificator.
    """
    # Ollama base URL — auto-append /v1 for proxy setups
    ollama_url = config.ollama.base_url.rstrip("/")
    if "/v1" not in ollama_url:
        ollama_url = ollama_url + "/v1"

    def _make(mc: ModelConfig) -> LLMClient:
        if mc.provider == "ollama":
            return LLMClient(
                provider="ollama",
                base_url=ollama_url,
                api_key=config.ollama.api_key,
                model=mc.model,
                temperature=mc.temperature,
                max_tokens=mc.max_tokens,
                max_retries=config.ollama.retry_attempts,
                timeout=config.ollama.timeout_seconds,
                rate_limit_delay=0.0,
            )
        elif mc.provider == "openai":
            return LLMClient(
                provider="openai",
                base_url="https://api.openai.com/v1",
                api_key=config.openai.api_key,
                model=mc.model,
                temperature=mc.temperature,
                max_tokens=mc.max_tokens,
                max_retries=config.openai.retry_attempts,
                timeout=config.openai.timeout_seconds,
                rate_limit_delay=0.5,
            )
        elif mc.provider == "together":
            return LLMClient(
                provider="together",
                base_url=config.together.base_url,
                api_key=config.together.api_key,
                model=mc.model,
                temperature=mc.temperature,
                max_tokens=mc.max_tokens,
                max_retries=config.together.retry_attempts,
                timeout=config.together.timeout_seconds,
                rate_limit_delay=1.0,
            )
        else:
            raise ValueError(f"Unknown provider: {mc.provider}")

    return {
        "interviewer": _make(config.interviewer),
        "assessor": _make(config.assessor),
        "orchestrator": _make(config.orchestrator_llm),
        "justificator": _make(config.justificator),
    }


# ---------------------------------------------------------------------------
# JSON response parsing
# ---------------------------------------------------------------------------


def parse_json_response(text: str) -> Optional[dict]:
    """Extract JSON from LLM response, handling markdown fences."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Markdown fences
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # First { to last }
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    return None
