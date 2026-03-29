"""LLM client for MentalRiskES Task 1.

Reuses the streaming pattern from erisk_task1.llm_client, simplified for
this pipeline's needs: JSON-structured assessment outputs.
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

    # Tracking
    _call_count: int = field(default=0, init=False, repr=False)
    _total_tokens: int = field(default=0, init=False, repr=False)
    _total_latency_ms: float = field(default=0.0, init=False, repr=False)

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

        if use_native_ollama:
            payload: dict = {
                "model": self.model,
                "messages": messages,
                "stream": True,
                "options": {"temperature": temp, "num_predict": mtok},
            }
        else:
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": True,
                "stream_options": {"include_usage": True},
                "temperature": temp,
                "max_tokens": mtok,
            }

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                t0 = time.monotonic()
                read_timeout = max(self.timeout, 300)
                response = requests.post(
                    url, headers=headers, json=payload,
                    timeout=(30, read_timeout), stream=True,
                )
                response.raise_for_status()
                result = self._consume_stream(response, use_native_ollama)
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
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                last_error = e
                wait = 2 ** attempt
                logger.warning("LLM connection error (attempt %d/%d), retrying in %ds",
                               attempt, self.max_retries, wait)
                time.sleep(wait)

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
        return {
            "call_count": self._call_count,
            "total_tokens": self._total_tokens,
            "total_latency_ms": round(self._total_latency_ms, 1),
        }


def parse_json_response(text: str) -> dict | None:
    """Extract and parse the first JSON object found in LLM output."""
    # Try to find JSON block in markdown code fence first
    fence_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', text)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # Fall back to finding any JSON object
    json_match = re.search(r'\{[\s\S]*\}', text)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    return None
