"""Persona model loader and inference using PEFT/LoRA on Llama-3-8B.

The persona model is loaded via transformers + peft (NOT via Ollama).
Each persona is a LoRA adapter on top of meta-llama/Meta-Llama-3-8B-Instruct.
"""

from __future__ import annotations

import logging
from typing import Optional

from .config import PersonaModelConfig
from .prompts import PERSONA_SYSTEM_PROMPT

_BitsAndBytesConfig = None

logger = logging.getLogger(__name__)

# Lazy imports — these are heavy and may not be available in all envs
_torch = None
_AutoTokenizer = None
_AutoModelForCausalLM = None
_PeftModel = None


def _ensure_imports():
    global _torch, _AutoTokenizer, _AutoModelForCausalLM, _PeftModel, _BitsAndBytesConfig
    if _torch is None:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel

        _torch = torch
        _AutoTokenizer = AutoTokenizer
        _AutoModelForCausalLM = AutoModelForCausalLM
        _PeftModel = PeftModel
        _BitsAndBytesConfig = BitsAndBytesConfig


class PersonaModel:
    """Manages a single persona: base model + LoRA adapter."""

    def __init__(self, config: PersonaModelConfig, quantize_4bit: bool = False):
        self.config = config
        self.quantize_4bit = quantize_4bit
        self.tokenizer = None
        self.model = None
        self._loaded_adapter: Optional[str] = None

    def load_base(self):
        """Load the base model and tokenizer (one-time, shared across personas)."""
        _ensure_imports()
        logger.info("Loading base model: %s", self.config.base_model)

        dtype_map = {
            "float16": _torch.float16,
            "bfloat16": _torch.bfloat16,
            "float32": _torch.float32,
        }
        dtype = dtype_map.get(self.config.torch_dtype, _torch.float16)

        self.tokenizer = _AutoTokenizer.from_pretrained(self.config.base_model)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"

        load_kwargs = dict(
            torch_dtype=dtype,
            device_map=self.config.device_map,
        )
        if self.quantize_4bit:
            load_kwargs["quantization_config"] = _BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
            )
            logger.info("Using 4-bit quantization (NF4)")

        self.model = _AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            **load_kwargs,
        )
        logger.info("Base model loaded")

    def load_adapter(self, persona_id: int):
        """Load a specific persona's LoRA adapter."""
        _ensure_imports()
        adapter_name = self.config.adapter_pattern.format(id=persona_id)

        if self._loaded_adapter == adapter_name:
            logger.debug("Adapter %s already loaded", adapter_name)
            return

        logger.info("Loading adapter: %s", adapter_name)
        if self._loaded_adapter is not None:
            # Unload previous adapter
            self.model = self.model.base_model.model
        self.model = _PeftModel.from_pretrained(self.model, adapter_name)
        self._loaded_adapter = adapter_name
        logger.info("Adapter loaded: %s", adapter_name)

    def generate(self, conversation: list[dict[str, str]]) -> str:
        """Generate a persona response given conversation history.

        Args:
            conversation: List of {"role": "user"|"assistant", "content": "..."}

        Returns:
            Generated response text.
        """
        _ensure_imports()

        # Build chat messages with mandatory system prompt
        messages = [{"role": "system", "content": PERSONA_SYSTEM_PROMPT}]
        messages.extend(conversation)

        # Apply chat template
        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)

        with _torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only new tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return response.strip()

    def unload(self):
        """Free GPU memory."""
        _ensure_imports()
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self._loaded_adapter = None
        if _torch is not None:
            _torch.cuda.empty_cache()
        logger.info("Persona model unloaded")
