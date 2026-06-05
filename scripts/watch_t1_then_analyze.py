"""Wait for T1 (regen-llama → gen-gemma) to finish, then fire reparse + RQ1 analysis.

Completion is detected via either:
  (a) a ``tier_pass_complete`` event with ``tier=T1, pass_=gen-gemma`` in any of
      ``meta.jsonl`` / ``meta.T1.jsonl`` (the dispatcher writes one when the
      gen-gemma pass exits normally), or
  (b) a quiescence fallback: no new JSONL lines across the 7 T1 signal logs for
      a sustained window, with the gen-gemma signal counts having reached the
      expected target.

When done, runs ``reparse`` then ``wasserstein aggregate analyze --rq 1``.
Intended to be launched in the background so we get notified on exit.
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

LLAMA_SIGS = ["llama_state_update", "llama_assess_phq9", "llama_assess_gad7",
              "llama_assess_compact10"]
GEMMA_SIGS = ["self_a", "observer_p", "tom_tier_patient"]
ALL_SIGS = LLAMA_SIGS + GEMMA_SIGS
EXPECTED = 568                  # patient-rounds across the 10 sessions

POLL_SECONDS = 90               # how often to re-scan
QUIESCENCE_POLLS = 8            # ~12 min of silence => fallback completion
QUIESCENCE_MIN_FRACTION = 0.98  # all gen-gemma signals must be ≥ this fraction of EXPECTED

# After T1 finishes we loop reparse → re-run T1 generation → reparse until either
# no signatures are still failed OR max_attempts (3) have been used. The initial
# pass uses attempt 1, so at most 2 re-runs are useful before retries_exhausted.
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
    """Per signal, count input_signatures that have NO successful line.

    These are the genuine stragglers — what reparse couldn't salvage and what a
    re-run with attempts < max_attempts would re-call.
    """
    out: dict[str, int] = {}
    for s in ALL_SIGS:
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
    for fname in ("meta.T1.jsonl", "meta.jsonl"):
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
                        and rec.get("tier") == "T1"
                        and rec.get("pass_") == "gen-gemma"):
                    return True
    return False


def run_cmd(cmd: list[str]) -> int:
    log(f"running: {' '.join(cmd)}")
    env = dict(os.environ)
    env.pop("VIRTUAL_ENV", None)         # let uv pick the project venv
    r = subprocess.run(cmd, env=env)
    log(f"exit code: {r.returncode}")
    return r.returncode


def wait_for_t1() -> None:
    log(f"polling every {POLL_SECONDS}s; expected {EXPECTED} rounds per signal")
    prev_total = -1
    quiet_polls = 0

    while True:
        if tier_complete_event_seen():
            log("tier_pass_complete event for T1/gen-gemma seen → T1 finished.")
            return

        line_counts = {s: count_lines(LOGS / f"{s}.jsonl") for s in ALL_SIGS}
        succ_counts = {s: count_success_unique(LOGS / f"{s}.jsonl") for s in ALL_SIGS}
        total_lines = sum(line_counts.values())

        llama = " ".join(f"{s.split('_',1)[1]}={succ_counts[s]}" for s in LLAMA_SIGS)
        gemma = " ".join(f"{s}={succ_counts[s]}" for s in GEMMA_SIGS)
        log(f"total={total_lines}  llama[{llama}]  gemma[{gemma}]")

        # Quiescence fallback.
        if total_lines == prev_total:
            quiet_polls += 1
            if (quiet_polls >= QUIESCENCE_POLLS
                    and all(succ_counts[s] >= QUIESCENCE_MIN_FRACTION * EXPECTED
                            for s in GEMMA_SIGS)):
                log(f"quiescence: {quiet_polls} polls with no growth and gen-gemma "
                    f"≥ {QUIESCENCE_MIN_FRACTION:.0%} of target → assuming T1 done.")
                return
        else:
            quiet_polls = 0
            prev_total = total_lines

        time.sleep(POLL_SECONDS)


def main() -> int:
    log("=== T1 watcher started ===")
    wait_for_t1()

    # Loop reparse → re-run T1 → reparse until either nothing's still failed or
    # we've exhausted the 3 max_attempts budget (initial + MAX_REGEN_RETRIES).
    for cycle in range(MAX_REGEN_RETRIES + 1):
        log(f"=== cycle {cycle}: firing reparse ===")
        rc = run_cmd(["uv", "run", "mentalriskes-tom-act",
                      "--config", "config/tom_act.yaml", "reparse"])
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

        log(f"=== re-running T1 generation to retry {total_failed} stragglers "
            f"(cycle {cycle + 1}/{MAX_REGEN_RETRIES}) ===")
        rc = run_cmd(["uv", "run", "mentalriskes-tom-act",
                      "--config", "config/tom_act.yaml", "--tier", "T1",
                      "regen-llama", "gen-gemma"])
        if rc != 0:
            log("T1 retry pass failed; aborting.")
            return rc

    # Final aggregate + RQ1 checkpoint.
    log("=== firing wasserstein aggregate analyze --rq 1 ===")
    rc = run_cmd(["uv", "run", "mentalriskes-tom-act",
                  "--config", "config/tom_act.yaml",
                  "wasserstein", "aggregate", "analyze", "--rq", "1"])
    if rc != 0:
        log("analysis failed; aborting.")
        return rc

    log("=== T1 watcher: done. RQ1 results under runs/tom_act_explanatory/outputs/analysis/ ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
