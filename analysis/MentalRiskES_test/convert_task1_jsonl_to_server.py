"""Convert Task 1 trial JSONL prediction logs to server-submission format.

The Task 1 trial pipeline writes per-prediction JSONL records to
`output/.../logs/predictions_<run>.jsonl` (one line per (session, round, run)).
The analysis scripts in `analysis/MentalRiskES_test/` expect server-style
per-round JSON files at `<output_dir>/predictions/round{N}_run{R}.json`,
matching the format the live submission pipeline produced.

This script bridges the two: it reads the JSONL logs and emits one
`round{N}_run{R}.json` file per (round, run) pair, where R is the integer
index parsed from the run name (run0_A5 -> 0, run1_A3 -> 1, run2_A1 -> 2).

Usage:
    uv run python analysis/MentalRiskES_test/convert_task1_jsonl_to_server.py \
        --logs-dir output/mentalriskes_test_replay/logs \
        --out-dir output/mentalriskes_test_replay/predictions
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

logger = logging.getLogger("convert_task1")


_RUN_NAME_RE = re.compile(r"^run(\d+)")


def parse_run_index(run_name: str) -> int:
    """run0_A5 -> 0, run1_A3 -> 1, run2_A1 -> 2."""
    m = _RUN_NAME_RE.match(run_name)
    if not m:
        raise ValueError(f"Cannot parse run index from '{run_name}'")
    return int(m.group(1))


def convert_one(jsonl_path: Path, out_dir: Path) -> int:
    """Convert one predictions_<run>.jsonl into per-round server-format JSONs.

    Returns the number of round files written.
    """
    run_name = jsonl_path.stem.replace("predictions_", "")
    run_idx = parse_run_index(run_name)

    # Group predictions by round
    per_round: dict[int, list[dict]] = {}
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            rnd = int(entry["round"])
            per_round.setdefault(rnd, []).append({
                "id": entry["session_id"],
                "round": rnd,
                "prediction": {
                    "GAD-7": entry["gad7"],
                    "PHQ-9": entry["phq9"],
                    "CompACT-10": entry["compact10"],
                },
            })

    # If a session was assessed multiple times in the same round (re-runs),
    # keep the LAST prediction.
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for rnd, preds in sorted(per_round.items()):
        # Deduplicate by session id, keeping the final occurrence
        latest_by_sid: dict[str, dict] = {}
        for p in preds:
            latest_by_sid[p["id"]] = p
        ordered = sorted(latest_by_sid.values(), key=lambda p: p["id"])
        out_path = out_dir / f"round{rnd}_run{run_idx}.json"
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(
                [{"predictions": ordered, "emissions": {}}],
                fh, ensure_ascii=False, indent=2,
            )
        written += 1
    logger.info(
        "Converted %s -> %d round files (run%d) in %s",
        jsonl_path.name, written, run_idx, out_dir,
    )
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-dir", required=True, help="Directory containing predictions_*.jsonl")
    parser.add_argument("--out-dir", required=True, help="Where to write round{N}_run{R}.json files")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logs_dir = Path(args.logs_dir)
    out_dir = Path(args.out_dir)
    files = sorted(logs_dir.glob("predictions_*.jsonl"))
    if not files:
        logger.warning("No predictions_*.jsonl found in %s", logs_dir)
        return
    total = 0
    for fp in files:
        total += convert_one(fp, out_dir)
    logger.info("Done: wrote %d round files across %d runs", total, len(files))


if __name__ == "__main__":
    main()
