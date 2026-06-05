"""Aggregation step (spec §6.6): JSONL logs → tidy parquet tables.

The JSONL logs are the system of record; these parquet tables are derivative and
rebuildable at any time. For each ``(session, round[, candidate])`` we take the
latest line with ``parse_success: true``. Reports per-signal recovery-stage
proportions (for the paper's methods section) and flags missing pairs.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

import pandas as pd

from .constants import ACT_PROCESS_KEYS, GEMMA_VIEW_SIGNALS, canonical_phase
from .llama_regen import (
    SIG_ASSESS,
    SIG_ASSESS_COMBINED,
    SIG_STATE_UPDATE,
    _extract_scores,
)

logger = logging.getLogger(__name__)

VIEW_INST_MAP = {"phq9": "PHQ-9", "gad7": "GAD-7", "compact10": "CompACT-10"}
ASSESS_INST = {v: k for k, v in SIG_ASSESS.items()}  # signal_type -> instrument


def _log_path(run_root: Path, signal_type: str) -> Path:
    return Path(run_root) / "logs" / f"{signal_type}.jsonl"


def _read_latest_success(run_root: Path, signal_type: str) -> dict[tuple, dict]:
    """{(session, round, candidate): record} keeping the latest successful line."""
    path = _log_path(run_root, signal_type)
    out: dict[tuple, dict] = {}
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not rec.get("parse_success"):
                continue
            key = (rec["session_id"], rec["round"], rec.get("candidate"))
            out[key] = rec  # later lines overwrite earlier (latest wins)
    return out


def recovery_stage_report(run_root: Path, signal_type: str) -> dict:
    """Per-signal recovery-stage proportions over all logged lines."""
    path = _log_path(run_root, signal_type)
    counts: Counter = Counter()
    total = 0
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                total += 1
                counts[rec.get("recovery_stage") if rec.get("parse_success") else "failed"] += 1
    return {"signal_type": signal_type, "total_lines": total, "stages": dict(counts)}


# ---------------------------------------------------------------------------
# Long-table builders
# ---------------------------------------------------------------------------

def _view_block_rows(sid, rnd, parsed: dict, source: str) -> list[dict]:
    """Rows from a 3-instrument view-format payload (combined assessor / views)."""
    out = []
    for jkey, instrument in VIEW_INST_MAP.items():
        block = parsed.get(jkey, {})
        items = block.get("items", []) if isinstance(block, dict) else []
        for it in items:
            if not isinstance(it, dict):
                continue
            try:
                out.append({"session_id": sid, "round": rnd, "instrument": instrument,
                            "item": int(it.get("item")), "score": int(it.get("score")),
                            "source": source})
            except (ValueError, TypeError):
                continue
    return out


def build_llama_assessors_long(run_root: Path) -> pd.DataFrame:
    """Assemble Llama assessor item-vectors from per-instrument and/or combined logs.

    Per-instrument rows take precedence over combined when both exist for the
    same (session, round, instrument, item).
    """
    rows = []
    # Per-instrument (task1 CoT) signals.
    for sig, instrument in ASSESS_INST.items():
        for (sid, rnd, _), rec in _read_latest_success(run_root, sig).items():
            scores = _extract_scores(rec.get("response_parsed"), instrument)
            if scores is None:
                continue
            for i, s in enumerate(scores, 1):
                rows.append({"session_id": sid, "round": rnd, "instrument": instrument,
                             "item": i, "score": s, "source": "per_instrument"})
    # Combined single-call signal (view format).
    for (sid, rnd, _), rec in _read_latest_success(run_root, SIG_ASSESS_COMBINED).items():
        rows.extend(_view_block_rows(sid, rnd, rec.get("response_parsed") or {}, "combined"))

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Sort descending so "per_instrument" precedes "combined"; keep first => the
    # higher-fidelity per-instrument vector wins when both exist.
    df = (df.sort_values("source", ascending=False, kind="stable")
            .drop_duplicates(["session_id", "round", "instrument", "item"], keep="first")
            .drop(columns="source")
            .reset_index(drop=True))
    return df


def build_llama_state_long(run_root: Path) -> pd.DataFrame:
    rows = []
    for (sid, rnd, _), rec in _read_latest_success(run_root, SIG_STATE_UPDATE).items():
        parsed = rec.get("response_parsed") or {}
        procesos = parsed.get("procesos_act", {}) or {}
        row = {"session_id": sid, "round": rnd,
               "fase_terapeutica": canonical_phase(parsed.get("fase_terapeutica"))}
        for k in ACT_PROCESS_KEYS:
            try:
                row[k] = float(procesos.get(k)) if procesos.get(k) is not None else None
            except (ValueError, TypeError):
                row[k] = None
        rows.append(row)
    return pd.DataFrame(rows)


def build_gemma_views_long(run_root: Path) -> pd.DataFrame:
    rows = []
    for view in GEMMA_VIEW_SIGNALS:
        for (sid, rnd, _), rec in _read_latest_success(run_root, view).items():
            parsed = rec.get("response_parsed") or {}
            for jkey, instrument in VIEW_INST_MAP.items():
                block = parsed.get(jkey, {})
                items = block.get("items", []) if isinstance(block, dict) else []
                for it in items:
                    if not isinstance(it, dict):
                        continue
                    item_no = it.get("item")
                    score = it.get("score")
                    try:
                        rows.append({"session_id": sid, "round": rnd, "view": view,
                                     "instrument": instrument, "item": int(item_no),
                                     "score": int(score)})
                    except (ValueError, TypeError):
                        continue
    return pd.DataFrame(rows)


def build_tom_tier(run_root: Path) -> pd.DataFrame:
    rows = []
    for (sid, rnd, _), rec in _read_latest_success(run_root, "tom_tier_patient").items():
        parsed = rec.get("response_parsed") or {}
        soft = parsed.get("soft_scores", {}) or {}
        rows.append({"session_id": sid, "round": rnd, "argmax": parsed.get("argmax"),
                     "somatico": soft.get("somatico"), "cognitivo": soft.get("cognitivo"),
                     "afectivo": soft.get("afectivo")})
    return pd.DataFrame(rows)


def build_candidate_signal(run_root: Path, signal_type: str, field: str) -> pd.DataFrame:
    rows = []
    for (sid, rnd, cand), rec in _read_latest_success(run_root, signal_type).items():
        parsed = rec.get("response_parsed") or {}
        rows.append({"session_id": sid, "round": rnd, "candidate": cand,
                     field: parsed.get(field)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

_BUILDERS = {
    "llama_assessors": build_llama_assessors_long,
    "llama_state": build_llama_state_long,
    "gemma_views": build_gemma_views_long,
    "tom_tier": build_tom_tier,
    "tom_stance": lambda rr: build_candidate_signal(rr, "tom_stance", "stance"),
    "presencia": lambda rr: build_candidate_signal(rr, "presencia", "presencia"),
}

_ALL_LOG_SIGNALS = (
    [SIG_STATE_UPDATE, SIG_ASSESS_COMBINED] + list(SIG_ASSESS.values()) + GEMMA_VIEW_SIGNALS
    + ["tom_tier_patient", "tom_stance", "presencia"]
)


def aggregate_all(run_root: str | Path) -> dict[str, pd.DataFrame]:
    """Build and persist all aggregated parquet tables; print reports."""
    run_root = Path(run_root)
    out_dir = run_root / "outputs" / "aggregated"
    out_dir.mkdir(parents=True, exist_ok=True)

    tables: dict[str, pd.DataFrame] = {}
    for name, builder in _BUILDERS.items():
        df = builder(run_root)
        tables[name] = df
        path = out_dir / f"{name}.parquet"
        df.to_parquet(path, index=False)
        logger.info("Wrote %s (%d rows) -> %s", name, len(df), path)

    # Recovery-stage report.
    reports = [recovery_stage_report(run_root, s) for s in _ALL_LOG_SIGNALS]
    with open(out_dir / "recovery_report.json", "w", encoding="utf-8") as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)
    for r in reports:
        if r["total_lines"]:
            logger.info("Recovery [%s]: %s", r["signal_type"], r["stages"])

    return tables
