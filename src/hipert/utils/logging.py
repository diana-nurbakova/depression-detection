"""Structured logging for the HiPerT-ADHD pipeline.

Three log streams:
1. Console: human-readable progress (INFO+)
2. Pipeline event log: JSONL of pipeline events
3. LLM call log: JSONL with full request/response for every LLM call
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


class JsonlHandler(logging.Handler):
    """Logging handler that writes JSON lines to a file."""

    def __init__(self, filepath: Path) -> None:
        super().__init__()
        self.filepath = filepath
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(filepath, "a", encoding="utf-8")

    def emit(self, record: logging.LogRecord) -> None:
        try:
            # Check for structured data in the extra dict
            data = getattr(record, "json_data", None)
            if data is not None:
                line = json.dumps(data, ensure_ascii=False, default=str)
                self._file.write(line + "\n")
                self._file.flush()
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        self._file.close()
        super().close()


class PipelineEventLogger:
    """Logs pipeline-level events to a JSONL file."""

    def __init__(self, log_dir: Path, run_id: str | None = None) -> None:
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.filepath = log_dir / f"pipeline_{self.run_id}.jsonl"
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.filepath, "a", encoding="utf-8")

    def log(self, event_type: str, **kwargs: Any) -> None:
        """Write a structured event record."""
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "event_type": event_type,
            **kwargs,
        }
        self._file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


class LLMCallLogger:
    """Logs every LLM call with full request/response details.

    Each call gets a unique call_id. System prompts are hashed to avoid
    repeating the ~2KB text in every record; the full text is logged once
    at pipeline start.
    """

    def __init__(self, log_dir: Path, run_id: str | None = None) -> None:
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.filepath = log_dir / f"llm_calls_{self.run_id}.jsonl"
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.filepath, "a", encoding="utf-8")
        self._system_prompt_hashes: dict[str, bool] = {}
        self._call_count = 0

    def log_system_prompt(self, prompt_text: str) -> str:
        """Log the full system prompt once and return its hash."""
        h = hashlib.sha256(prompt_text.encode()).hexdigest()[:16]
        if h not in self._system_prompt_hashes:
            self._system_prompt_hashes[h] = True
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "run_id": self.run_id,
                "event_type": "system_prompt",
                "prompt_hash": h,
                "prompt_text": prompt_text,
            }
            self._file.write(
                json.dumps(record, ensure_ascii=False, default=str) + "\n",
            )
            self._file.flush()
        return h

    def log_call(
        self,
        *,
        provider: str,
        model: str,
        symptom_id: int,
        sentence_id: str,
        sentence_text: str,
        is_escalation: bool,
        system_prompt_hash: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        raw_response: str,
        parsed: Optional[dict] = None,
        parse_success: bool = True,
        parse_warnings: Optional[list[str]] = None,
        escalation_triggered: bool = False,
        escalation_triggers: Optional[list[str]] = None,
        gpt_call_id: Optional[str] = None,
        llama_call_id: Optional[str] = None,
        final_label: Optional[int] = None,
        confidence_weight: Optional[float] = None,
        resolution_source: Optional[str] = None,
        latency_ms: float = 0.0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        time_to_first_chunk_ms: Optional[float] = None,
        stream_chunks: Optional[int] = None,
        success: bool = True,
        error: Optional[str] = None,
        llama_output_for_escalation: Optional[dict] = None,
    ) -> str:
        """Log a single LLM call and return its call_id."""
        self._call_count += 1
        call_id = str(uuid.uuid4())[:12]

        record: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "event_type": "llm_call",
            "call_id": call_id,
            "call_number": self._call_count,
            "provider": provider,
            "model": model,
            "symptom_id": symptom_id,
            "sentence_id": sentence_id,
            "sentence_text": sentence_text,
            "is_escalation": is_escalation,
            "request": {
                "system_prompt_hash": system_prompt_hash,
                "user_prompt": user_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            "response": {
                "raw_text": raw_response,
                "parsed": parsed,
                "parse_success": parse_success,
                "parse_warnings": parse_warnings or [],
            },
            "escalation": {
                "triggered": escalation_triggered,
                "triggers": escalation_triggers or [],
                "gpt_call_id": gpt_call_id,
                "llama_call_id": llama_call_id,
            },
            "metrics": {
                "latency_ms": round(latency_ms, 1),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "time_to_first_chunk_ms": (
                    round(time_to_first_chunk_ms, 1)
                    if time_to_first_chunk_ms is not None else None
                ),
                "stream_chunks": stream_chunks,
            },
            "success": success,
            "error": error,
        }

        # Include resolution info if available (filled after escalation decision)
        if final_label is not None:
            record["resolution"] = {
                "final_label": final_label,
                "confidence_weight": confidence_weight,
                "source": resolution_source,
            }

        # For escalation calls, include the Llama output that prompted escalation
        if is_escalation and llama_output_for_escalation is not None:
            record["llama_context"] = llama_output_for_escalation

        self._file.write(
            json.dumps(record, ensure_ascii=False, default=str) + "\n",
        )
        self._file.flush()
        return call_id

    @property
    def call_count(self) -> int:
        return self._call_count

    def close(self) -> None:
        self._file.close()


def setup_logging(
    log_dir: Path,
    level: str = "INFO",
    run_id: str | None = None,
) -> tuple[PipelineEventLogger, LLMCallLogger]:
    """Configure the full logging stack.

    Sets up:
    1. Console handler (human-readable, INFO+)
    2. PipelineEventLogger (JSONL, all pipeline events)
    3. LLMCallLogger (JSONL, all LLM calls with full detail)

    Args:
        log_dir: Directory for log files.
        level: Minimum log level for console output.
        run_id: Optional run identifier. If None, uses timestamp.

    Returns:
        Tuple of (PipelineEventLogger, LLMCallLogger).
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Configure root logger for console output
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers to avoid duplicates on re-init
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    console_fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(console_fmt)
    root_logger.addHandler(console_handler)

    # Create specialized loggers
    pipeline_logger = PipelineEventLogger(log_dir, run_id)
    llm_logger = LLMCallLogger(log_dir, run_id)

    logging.getLogger(__name__).info(
        "Logging initialized — run_id=%s, log_dir=%s", run_id, log_dir,
    )

    return pipeline_logger, llm_logger
