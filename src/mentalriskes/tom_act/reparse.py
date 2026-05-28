"""Offline re-parsing of failed JSONL lines with updated recovery logic (spec §6.9).

The JSONL logs retain every raw response, so a parser improvement can salvage
previously-failed calls **without any new LLM calls**. For each signal type, any
``input_signature`` whose latest line failed (and has no successful line) is
re-run through :func:`recovery.recover` on its saved ``response_raw``; on
success a corrected line is appended (``reparsed: true``). Both the resume index
and the aggregator use latest-success semantics, so reparsed records are then
skipped by future runs and picked up by aggregation.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from .constants import GEMMA_VIEW_SIGNALS
from .dispatcher import Dispatcher
from .llama_regen import SIG_ASSESS, SIG_ASSESS_COMBINED, SIG_STATE_UPDATE
from .recovery import recover

logger = logging.getLogger(__name__)

# Recovery schema per signal type.
SIGNAL_SCHEMA: dict[str, str | None] = {
    SIG_STATE_UPDATE: None,
    SIG_ASSESS_COMBINED: "view",
    "llama_assess_phq9": "assessor:PHQ-9",
    "llama_assess_gad7": "assessor:GAD-7",
    "llama_assess_compact10": "assessor:CompACT-10",
    **{v: "view" for v in GEMMA_VIEW_SIGNALS},
    "tom_tier_patient": "tom_tier_patient",
    "tom_stance": "tom_stance",
    "presencia": "presencia",
}


def reparse(run_root: str | Path, signals: list[str] | None = None) -> dict[str, int]:
    """Re-parse failed lines from saved raws; append corrected lines. No LLM calls.

    Returns ``{signal_type: n_recovered}``.
    """
    run_root = Path(run_root)
    disp = Dispatcher(run_root)
    targets = signals or list(SIGNAL_SCHEMA)
    summary: dict[str, int] = {}

    for signal_type in targets:
        path = run_root / "logs" / f"{signal_type}.jsonl"
        if not path.exists():
            continue
        schema = SIGNAL_SCHEMA.get(signal_type)

        # Latest line + whether any success exists, per signature.
        latest: dict[str, dict] = {}
        has_success: set[str] = set()
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sig = rec.get("input_signature")
                if not sig:
                    continue
                latest[sig] = rec
                if rec.get("parse_success"):
                    has_success.add(sig)

        recovered = 0
        for sig, rec in latest.items():
            if sig in has_success:
                continue
            raw = rec.get("response_raw") or ""
            res = recover(raw, schema)
            if not res.success:
                continue
            corrected = dict(rec)
            corrected.update({
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "response_parsed": res.parsed,
                "parse_success": True,
                "parse_error": None,
                "recovery_stage": res.stage,
                "reparsed": True,
            })
            disp._append(path, corrected)
            recovered += 1

        if recovered:
            disp.meta("reparse_recovered", signal_type=signal_type, count=recovered)
            logger.info("reparse [%s]: recovered %d previously-failed call(s)",
                        signal_type, recovered)
        summary[signal_type] = recovered

    return summary
