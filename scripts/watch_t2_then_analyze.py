"""Wait for T2 (gen-gemma, gold-only candidates) to finish, then fire reparse +
RQ2/RQ3 analysis (spec §15.3 T2 checkpoint).

Mirrors the T1 watcher but scoped to the four T2 signal types and the
T2-relevant analyses. ``reparse`` is filtered to T2 signals so it can't touch
log files the T1 process or T1 watcher may still be writing.

Completion is detected via a ``tier_pass_complete`` event with
``tier=T2, pass_=gen-gemma`` in either ``meta.T2.jsonl`` or ``meta.jsonl``,
with a quiescence fallback.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Force UTF-8 on stdout/stderr — Windows defaults to cp1252 when redirected,
# which can't encode non-ASCII characters and crashes the watcher mid-print.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

ROOT = Path("runs/tom_act_explanatory")
LOGS = ROOT / "logs"

# T2 signal types (spec §15.1): Self-B + Observer-PT + stance/presencia × gold candidate.
T2_SIGS = ["self_b", "observer_pt", "tom_stance", "presencia"]
EXPECTED = 568                  # patient-rounds across the 10 sessions

POLL_SECONDS = 90
QUIESCENCE_POLLS = 8
QUIESCENCE_MIN_FRACTION = 0.98

# Initial pass uses attempt 1; max_attempts=3 ⇒ at most 2 useful re-runs.
MAX_REGEN_RETRIES = 2


def _stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{_stamp()}] {msg}", flush=True)


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def count_success_unique(path: Path) -> int:
    """Unique (session, round, candidate) tuples with a successful line."""
    if not path.exists():
        return 0
    seen: set[tuple] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("parse_success"):
                seen.add((rec.get("session_id"), rec.get("round"),
                          rec.get("candidate")))
    return len(seen)


def count_failed_signatures() -> dict[str, int]:
    """Per signal, count input_signatures with NO successful line."""
    out: dict[str, int] = {}
    for s in T2_SIGS:
        p = LOGS / f"{s}.jsonl"
        if not p.exists():
            out[s] = 0
            continue
        has_success: set[str] = set()
        all_sigs: set[str] = set()
        with open(p, encoding="utf-8") as f:
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
                all_sigs.add(sig)
                if rec.get("parse_success"):
                    has_success.add(sig)
        out[s] = len(all_sigs - has_success)
    return out


def tier_complete_event_seen() -> bool:
    for fname in ("meta.T2.jsonl", "meta.jsonl"):
        p = LOGS / fname
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (rec.get("event") == "tier_pass_complete"
                        and rec.get("tier") == "T2"
                        and rec.get("pass_") == "gen-gemma"):
                    return True
    return False


def run_cmd(cmd: list[str]) -> int:
    log(f"running: {' '.join(cmd)}")
    env = dict(os.environ)
    env.pop("VIRTUAL_ENV", None)
    r = subprocess.run(cmd, env=env)
    log(f"exit code: {r.returncode}")
    return r.returncode


def wait_for_t2() -> None:
    log(f"polling every {POLL_SECONDS}s; expected {EXPECTED} rounds per signal")
    prev_total = -1
    quiet_polls = 0

    while True:
        if tier_complete_event_seen():
            log("tier_pass_complete event for T2/gen-gemma seen → T2 finished.")
            return

        line_counts = {s: count_lines(LOGS / f"{s}.jsonl") for s in T2_SIGS}
        succ_counts = {s: count_success_unique(LOGS / f"{s}.jsonl") for s in T2_SIGS}
        total_lines = sum(line_counts.values())

        report = " ".join(f"{s}={succ_counts[s]}" for s in T2_SIGS)
        log(f"total={total_lines}  T2[{report}]")

        if total_lines == prev_total:
            quiet_polls += 1
            if (quiet_polls >= QUIESCENCE_POLLS
                    and all(succ_counts[s] >= QUIESCENCE_MIN_FRACTION * EXPECTED
                            for s in T2_SIGS)):
                log(f"quiescence: {quiet_polls} polls without growth and T2 signals "
                    f"≥ {QUIESCENCE_MIN_FRACTION:.0%} of target → assuming T2 done.")
                return
        else:
            quiet_polls = 0
            prev_total = total_lines

        time.sleep(POLL_SECONDS)


def main() -> int:
    log("=== T2 watcher started ===")
    wait_for_t2()

    signals_arg = ",".join(T2_SIGS)

    for cycle in range(MAX_REGEN_RETRIES + 1):
        log(f"=== cycle {cycle}: firing reparse (T2 signals only) ===")
        rc = run_cmd(["uv", "run", "mentalriskes-tom-act",
                      "--config", "config/tom_act.yaml",
                      "reparse", "--signals", signals_arg])
        if rc != 0:
            log("reparse failed; aborting.")
            return rc

        still_failed = count_failed_signatures()
        total_failed = sum(still_failed.values())
        log(f"after reparse: still-failed by signal = {still_failed}  total = {total_failed}")

        if total_failed == 0:
            log("nothing left to retry.")
            break
        if cycle == MAX_REGEN_RETRIES:
            log(f"reached MAX_REGEN_RETRIES={MAX_REGEN_RETRIES}; the {total_failed} "
                f"remaining signature(s) will be left as retries_exhausted gaps.")
            break

        log(f"=== re-running T2 generation to retry {total_failed} stragglers "
            f"(cycle {cycle + 1}/{MAX_REGEN_RETRIES}) ===")
        rc = run_cmd(["uv", "run", "mentalriskes-tom-act",
                      "--config", "config/tom_act.yaml", "--tier", "T2",
                      "gen-gemma"])
        if rc != 0:
            log("T2 retry pass failed; aborting.")
            return rc

    # Final aggregate (idempotent across watchers) + RQ2/RQ3 checkpoint per spec §15.3.
    log("=== firing wasserstein aggregate analyze --rq 2 ===")
    rc = run_cmd(["uv", "run", "mentalriskes-tom-act",
                  "--config", "config/tom_act.yaml",
                  "wasserstein", "aggregate", "analyze", "--rq", "2"])
    if rc != 0:
        log("RQ2 analysis failed; aborting.")
        return rc

    log("=== firing analyze --rq 3 ===")
    rc = run_cmd(["uv", "run", "mentalriskes-tom-act",
                  "--config", "config/tom_act.yaml", "analyze", "--rq", "3"])
    if rc != 0:
        log("RQ3 analysis failed; aborting.")
        return rc

    log("=== T2 watcher: done. RQ2/RQ3 results under runs/tom_act_explanatory/outputs/analysis/ ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
