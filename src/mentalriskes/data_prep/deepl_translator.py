"""DeepL translation utility for MentalRiskES data preparation.

Translates text batches EN->ES using the DeepL API with rate limiting,
caching, and batch support. Uses the REST API directly (no SDK dependency).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

DEEPL_API_URL = "https://api-free.deepl.com/v2/translate"
DEEPL_API_URL_PRO = "https://api.deepl.com/v2/translate"

# DeepL free tier: 500,000 chars/month, no rate limit documented but be polite
MAX_BATCH_SIZE = 50  # texts per request
DELAY_BETWEEN_REQUESTS = 0.5  # seconds


class DeepLTranslator:
    """Batch translator using DeepL API with disk cache."""

    def __init__(
        self,
        auth_key: str | None = None,
        cache_dir: str | Path = "output/mentalriskes/translation_cache",
        source_lang: str = "EN",
        target_lang: str = "ES",
    ):
        self.auth_key = auth_key or os.environ.get("DEEPL_AUTH_KEY", "")
        if not self.auth_key:
            raise ValueError("DEEPL_AUTH_KEY not set in environment or passed as argument")

        # Free vs Pro API endpoint detection (free keys end with ":fx")
        if self.auth_key.endswith(":fx"):
            self.api_url = DEEPL_API_URL
        else:
            self.api_url = DEEPL_API_URL_PRO

        self.source_lang = source_lang
        self.target_lang = target_lang
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, str] = {}
        self._load_cache()

        # Stats
        self.chars_translated = 0
        self.cache_hits = 0
        self.api_calls = 0

    def _cache_key(self, text: str) -> str:
        """Generate a deterministic cache key for a text."""
        h = hashlib.sha256(
            f"{self.source_lang}:{self.target_lang}:{text}".encode()
        ).hexdigest()[:16]
        return h

    def _cache_path(self) -> Path:
        return self.cache_dir / f"deepl_{self.source_lang}_{self.target_lang}.json"

    def _load_cache(self) -> None:
        path = self._cache_path()
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                self._cache = json.load(f)
            logger.info("Loaded %d cached translations from %s", len(self._cache), path)

    def _save_cache(self) -> None:
        path = self._cache_path()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._cache, f, ensure_ascii=False, indent=2)

    def translate(self, text: str) -> str:
        """Translate a single text string."""
        results = self.translate_batch([text])
        return results[0]

    def translate_batch(self, texts: list[str]) -> list[str]:
        """Translate a batch of texts, using cache where possible."""
        results: list[str | None] = [None] * len(texts)
        to_translate: list[tuple[int, str]] = []

        # Check cache first
        for i, text in enumerate(texts):
            key = self._cache_key(text)
            if key in self._cache:
                results[i] = self._cache[key]
                self.cache_hits += 1
            else:
                to_translate.append((i, text))

        if not to_translate:
            logger.debug("All %d texts found in cache", len(texts))
            return results  # type: ignore

        logger.info(
            "Translating %d texts (%d cached, %d new)",
            len(texts), self.cache_hits, len(to_translate),
        )

        # Batch translate uncached texts
        for batch_start in range(0, len(to_translate), MAX_BATCH_SIZE):
            batch = to_translate[batch_start : batch_start + MAX_BATCH_SIZE]
            batch_texts = [t for _, t in batch]

            translated = self._api_translate(batch_texts)

            for (orig_idx, orig_text), trans_text in zip(batch, translated):
                results[orig_idx] = trans_text
                key = self._cache_key(orig_text)
                self._cache[key] = trans_text

            if batch_start + MAX_BATCH_SIZE < len(to_translate):
                time.sleep(DELAY_BETWEEN_REQUESTS)

        self._save_cache()
        return results  # type: ignore

    def _api_translate(self, texts: list[str]) -> list[str]:
        """Call DeepL API for a batch of texts."""
        total_chars = sum(len(t) for t in texts)
        logger.debug("DeepL API call: %d texts, %d chars", len(texts), total_chars)

        response = requests.post(
            self.api_url,
            headers={"Authorization": f"DeepL-Auth-Key {self.auth_key}"},
            data={
                "text": texts,
                "source_lang": self.source_lang,
                "target_lang": self.target_lang,
                "formality": "default",
            },
            timeout=60,
        )
        response.raise_for_status()

        data = response.json()
        translations = [t["text"] for t in data["translations"]]

        self.chars_translated += total_chars
        self.api_calls += 1

        logger.debug("API response: %d translations", len(translations))
        return translations

    def get_usage(self) -> dict:
        """Get DeepL API usage statistics."""
        response = requests.get(
            self.api_url.replace("/translate", "/usage"),
            headers={"Authorization": f"DeepL-Auth-Key {self.auth_key}"},
            timeout=10,
        )
        response.raise_for_status()
        return response.json()

    def stats(self) -> dict:
        return {
            "chars_translated": self.chars_translated,
            "cache_hits": self.cache_hits,
            "api_calls": self.api_calls,
            "cache_size": len(self._cache),
        }
