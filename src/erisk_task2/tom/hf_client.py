"""HuggingFace Inference client — drop-in replacement for OllamaClient.

Uses the HF Inference API (serverless), so no local LLM needed.
Requires HF_TOKEN env var or explicit token parameter.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)


class HFInferenceClient:
    """HuggingFace Inference API client with the same interface as OllamaClient."""

    def __init__(
        self,
        model: str = "meta-llama/Llama-3.3-70B-Instruct",
        token: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        timeout: int = 120,
        max_retries: int = 3,
    ):
        self.model = model
        self.token = token or os.getenv("HF_TOKEN", "")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = None

    def _get_client(self):
        if self._client is None:
            from huggingface_hub import InferenceClient
            self._client = InferenceClient(
                model=self.model,
                token=self.token,
                timeout=self.timeout,
            )
        return self._client

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
    ) -> tuple[str, float]:
        """Call HF Inference API with chat completion.

        Returns (response_text, elapsed_seconds).
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        temp = temperature or self.temperature

        for attempt in range(self.max_retries):
            try:
                start = time.monotonic()
                response = self._get_client().chat_completion(
                    messages=messages,
                    temperature=temp,
                    max_tokens=self.max_tokens,
                )
                elapsed = time.monotonic() - start

                text = response.choices[0].message.content or ""
                return text, elapsed

            except Exception as e:
                logger.warning(
                    "HF Inference request failed (attempt %d/%d): %s",
                    attempt + 1, self.max_retries, e,
                )
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)

        logger.error("All %d HF Inference attempts failed", self.max_retries)
        return "", 0.0

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
    ) -> tuple[Optional[dict], float]:
        """Call HF Inference API and parse response as JSON.

        Returns (parsed_dict_or_None, elapsed_seconds).
        """
        text, elapsed = self.generate(system_prompt, user_prompt, temperature)
        if not text:
            return None, elapsed

        parsed = _extract_json(text)
        if parsed is None:
            logger.warning("Failed to parse JSON from HF response: %s", text[:200])
        return parsed, elapsed

    def is_available(self) -> bool:
        """Check if HF Inference API is reachable."""
        try:
            client = self._get_client()
            # Simple health check — list models endpoint
            client.model_info(self.model)
            return True
        except Exception:
            return False


def _extract_json(text: str) -> Optional[dict]:
    """Extract JSON from LLM response, handling markdown code blocks."""
    text = text.strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code block
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            cleaned = part.strip()
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                continue

    # Try finding JSON object boundaries
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    return None
