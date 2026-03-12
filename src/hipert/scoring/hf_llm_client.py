"""HuggingFace Transformers LLM client for local inference on Colab.

Drop-in replacement for the HTTP-based LLMClient. Loads a model
via transformers and runs inference locally on GPU.

Designed for Llama 3.1 8B Instruct in 4-bit quantization (~5GB VRAM).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import torch

logger = logging.getLogger(__name__)

# Lazy-loaded to avoid import overhead when not used
_pipeline = None
_tokenizer = None


def _load_model(model_id: str, max_memory: str = "14GiB"):
    """Load model with 4-bit quantization for Colab T4/L4 GPUs."""
    global _pipeline, _tokenizer

    if _pipeline is not None:
        return _pipeline, _tokenizer

    from transformers import AutoTokenizer, pipeline, BitsAndBytesConfig

    logger.info("Loading HF model: %s (4-bit quantized)", model_id)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    _tokenizer = AutoTokenizer.from_pretrained(model_id)
    _pipeline = pipeline(
        "text-generation",
        model=model_id,
        tokenizer=_tokenizer,
        model_kwargs={"quantization_config": bnb_config},
        device_map="auto",
        torch_dtype=torch.float16,
    )

    logger.info("Model loaded: %s", model_id)
    return _pipeline, _tokenizer


@dataclass
class HFLLMClient:
    """Local HuggingFace LLM client with same interface as LLMClient."""

    provider: str = "huggingface"
    base_url: str = ""
    api_key: str = ""
    model: str = "meta-llama/Llama-3.1-8B-Instruct"
    temperature: float = 0.1
    max_tokens: int = 512
    max_retries: int = 1
    rate_limit_delay: float = 0.0
    timeout: int = 300

    # Tracking
    _call_count: int = field(default=0, init=False, repr=False)
    _total_tokens: int = field(default=0, init=False, repr=False)
    _total_prompt_tokens: int = field(default=0, init=False, repr=False)
    _total_completion_tokens: int = field(default=0, init=False, repr=False)
    _total_latency_ms: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self):
        """Load model on first initialization."""
        _load_model(self.model)

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict:
        """Run local inference matching OpenAI response format."""
        pipe, tokenizer = _load_model(self.model)

        temp = temperature if temperature is not None else self.temperature
        mtok = max_tokens if max_tokens is not None else self.max_tokens

        t0 = time.monotonic()

        # Use the chat template
        outputs = pipe(
            messages,
            max_new_tokens=mtok,
            temperature=max(temp, 0.01),  # transformers doesn't accept 0
            do_sample=temp > 0,
            return_full_text=False,
        )

        elapsed = time.monotonic() - t0
        content = outputs[0]["generated_text"]

        # Estimate token counts
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        prompt_tokens = len(tokenizer.encode(prompt_text))
        completion_tokens = len(tokenizer.encode(content))

        self._call_count += 1
        self._total_tokens += prompt_tokens + completion_tokens
        self._total_prompt_tokens += prompt_tokens
        self._total_completion_tokens += completion_tokens
        self._total_latency_ms += elapsed * 1000

        logger.info(
            "HF LLM call %d [%s] %.1fs — prompt=%d, completion=%d, content_len=%d",
            self._call_count, self.model, elapsed,
            prompt_tokens, completion_tokens, len(content),
        )

        return {
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "_meta": {
                "latency_ms": round(elapsed * 1000, 1),
                "time_to_first_chunk_ms": None,
                "stream_chunks": 0,
            },
        }

    def get_content(self, response: dict) -> str:
        """Extract the assistant message content from a response."""
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
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
