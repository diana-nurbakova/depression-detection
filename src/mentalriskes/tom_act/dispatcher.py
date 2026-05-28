"""Persistence and logging engine (spec §6).

The JSONL logs are the authoritative system of record: every LLM call produces
exactly one line, *regardless of parse success*, with the full prompts and raw
response so a third party can reconstruct what the model saw. Writes are
flushed + fsynced. A resume scan lets any interrupted pass continue without
re-running completed ``(session, round[, candidate])`` calls.

Single-process / multi-threaded safe via per-path locks. Multi-process
parallelism would additionally need OS-level file locks (noted, not used here).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .recovery import RecoveryResult, recover

logger = logging.getLogger(__name__)

_PROMPT_VERSION = "v1"
_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _lock_for(path: Path) -> threading.Lock:
    key = str(path)
    with _locks_guard:
        if key not in _locks:
            _locks[key] = threading.Lock()
        return _locks[key]


def _sha256(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return "git:" + out.stdout.strip()
    except Exception:
        pass
    return "git:unknown"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def input_signature(session_id: str, round_n: int, signal_type: str,
                    candidate: int | None = None) -> str:
    """Stable resume key, e.g. ``S01:r03:tom_stance:opt2:v1``."""
    parts = [session_id, f"r{round_n:02d}", signal_type]
    if candidate is not None:
        parts.append(f"opt{candidate}")
    parts.append(_PROMPT_VERSION)
    return ":".join(parts)


@dataclass
class _Completed:
    parsed: dict | None
    attempts: int
    success: bool


class Dispatcher:
    """Routes LLM calls through recovery + atomic JSONL persistence with resume."""

    def __init__(self, run_root: str | Path, max_attempts: int = 3) -> None:
        self.root = Path(run_root)
        self.logs = self.root / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.max_attempts = max_attempts
        self.code_version = _git_commit()
        self.meta_path = self.logs / "meta.jsonl"
        # Per-signal resume index: {signal_type: {signature: _Completed}}
        self._index: dict[str, dict[str, _Completed]] = {}

    # -- persistence ------------------------------------------------------

    def _log_path(self, signal_type: str) -> Path:
        return self.logs / f"{signal_type}.jsonl"

    def _append(self, path: Path, obj: dict) -> None:
        line = json.dumps(obj, ensure_ascii=False)
        with _lock_for(path):
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())

    def meta(self, event: str, **fields) -> None:
        self._append(self.meta_path, {"timestamp_utc": _now(), "event": event, **fields})

    # -- resume index -----------------------------------------------------

    def _load_index(self, signal_type: str) -> dict[str, _Completed]:
        if signal_type in self._index:
            return self._index[signal_type]
        idx: dict[str, _Completed] = {}
        path = self._log_path(signal_type)
        if path.exists():
            with open(path, encoding="utf-8") as f:
                for raw_line in f:
                    raw_line = raw_line.strip()
                    if not raw_line:
                        continue
                    try:
                        rec = json.loads(raw_line)
                    except json.JSONDecodeError:
                        continue
                    sig = rec.get("input_signature")
                    if not sig:
                        continue
                    prev = idx.get(sig)
                    attempts = max(prev.attempts if prev else 0, rec.get("attempt", 1))
                    success = (prev.success if prev else False) or rec.get("parse_success", False)
                    parsed = rec.get("response_parsed") if rec.get("parse_success") else (
                        prev.parsed if prev else None)
                    idx[sig] = _Completed(parsed=parsed, attempts=attempts, success=success)
        self._index[signal_type] = idx
        return idx

    # -- main entry -------------------------------------------------------

    def process(
        self,
        *,
        signal_type: str,
        session_id: str,
        round_n: int,
        system_prompt: str,
        user_prompt: str,
        client,
        model_id: str,
        provider: str,
        schema: str | None = None,
        candidate: int | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict | None:
        """Issue (or skip) one LLM call; persist exactly one JSONL line.

        Returns the parsed response dict, or None on failure / exhausted retries.
        """
        idx = self._load_index(signal_type)
        sig = input_signature(session_id, round_n, signal_type, candidate)
        prior = idx.get(sig)

        # Resume: already succeeded → skip.
        if prior and prior.success:
            return prior.parsed
        # Exhausted retries → skip + flag.
        if prior and prior.attempts >= self.max_attempts:
            self.meta("retries_exhausted", input_signature=sig, attempts=prior.attempts)
            return None

        attempt = (prior.attempts + 1) if prior else 1
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        t0 = time.monotonic()
        raw = ""
        tokens_in = tokens_out = 0
        call_error: str | None = None
        try:
            resp = client.chat_completion(messages, temperature, max_tokens)
            raw = client._get_content(resp)
            usage = resp.get("usage", {}) or {}
            tokens_in = usage.get("prompt_tokens", 0)
            tokens_out = usage.get("completion_tokens", 0)
        except Exception as e:  # transport already retried inside client
            call_error = f"{type(e).__name__}: {e}"
            self.meta("call_exception", input_signature=sig, attempt=attempt, error=call_error)

        latency_ms = (time.monotonic() - t0) * 1000.0

        rec: RecoveryResult = (
            recover(raw, schema) if call_error is None
            else RecoveryResult(None, False, None, error=call_error)
        )
        if rec.notes:
            self.meta("recovery_notes", input_signature=sig, stage=rec.stage, notes=rec.notes)

        entry = {
            "timestamp_utc": _now(),
            "session_id": session_id,
            "round": round_n,
            "signal_type": signal_type,
            "model_id": model_id,
            "provider": provider,
            "prompt_system_hash": _sha256(system_prompt),
            "prompt_system": system_prompt,
            "prompt_user_hash": _sha256(user_prompt),
            "prompt_user": user_prompt,
            "response_raw": raw,
            "response_parsed": rec.parsed,
            "parse_success": rec.success,
            "parse_error": rec.error,
            "recovery_stage": rec.stage,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "latency_ms": round(latency_ms, 1),
            "attempt": attempt,
            "input_signature": sig,
            "code_version": self.code_version,
        }
        if candidate is not None:
            entry["candidate"] = candidate

        self._append(self._log_path(signal_type), entry)
        idx[sig] = _Completed(parsed=rec.parsed, attempts=attempt, success=rec.success)

        if not rec.success:
            logger.warning("Parse failed [%s] attempt %d: %s", sig, attempt, rec.error)
        return rec.parsed if rec.success else None
