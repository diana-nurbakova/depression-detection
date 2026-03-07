"""Ollama LLM client with caching strategy (Spec Sections 12-13).

Handles all 4 prompt types with retry logic and JSON parsing.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)


class OllamaClient:
    """HTTP client for Ollama API with system prompt caching."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.3:70b",
        num_ctx: int = 8192,
        keep_alive: str = "24h",
        temperature: float = 0.1,
        timeout: int = 120,
        max_retries: int = 3,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.num_ctx = num_ctx
        self.keep_alive = keep_alive
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self._session = requests.Session()

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
    ) -> tuple[str, float]:
        """Call Ollama generate API.

        Returns (response_text, elapsed_seconds).
        """
        payload = {
            "model": self.model,
            "system": system_prompt,
            "prompt": user_prompt,
            "stream": False,
            "options": {
                "num_ctx": self.num_ctx,
                "temperature": temperature or self.temperature,
            },
            "keep_alive": self.keep_alive,
        }

        url = f"{self.base_url}/api/generate"

        for attempt in range(self.max_retries):
            try:
                start = time.monotonic()
                resp = self._session.post(url, json=payload, timeout=self.timeout)
                elapsed = time.monotonic() - start

                if resp.status_code == 200:
                    data = resp.json()
                    return data.get("response", ""), elapsed
                else:
                    logger.warning(
                        "Ollama returned %d on attempt %d: %s",
                        resp.status_code, attempt + 1, resp.text[:200],
                    )
            except requests.RequestException as e:
                logger.warning("Ollama request failed (attempt %d): %s", attempt + 1, e)

            if attempt < self.max_retries - 1:
                time.sleep(2 ** attempt)

        logger.error("All %d Ollama attempts failed", self.max_retries)
        return "", 0.0

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
    ) -> tuple[Optional[dict], float]:
        """Call Ollama and parse response as JSON.

        Returns (parsed_dict_or_None, elapsed_seconds).
        """
        text, elapsed = self.generate(system_prompt, user_prompt, temperature)
        if not text:
            return None, elapsed

        parsed = _extract_json(text)
        if parsed is None:
            logger.warning("Failed to parse JSON from Ollama response: %s", text[:200])
        return parsed, elapsed

    def is_available(self) -> bool:
        """Check if Ollama server is reachable."""
        try:
            resp = self._session.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except requests.RequestException:
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
