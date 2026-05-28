"""Diff each persona's results_{1,2,3}.json between the new official_submission folder
and the per-batch task1_results_<date> folders, to identify which run/persona files
were overwritten after submission."""

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
NEW_ROOT = REPO_ROOT / "runs" / "task1" / "official_submission" / "task1-llms-results"
BATCH_ROOT = REPO_ROOT / "runs" / "task1"

# Map patient_id (= LLM id) -> list of per-batch persona dirs that contain it.
BATCHES_FOR_PATIENT = {
    2: ["task1_results_20260308/persona01"],
    3: ["task1_results_20260308/persona02"],
    4: ["task1_results_20260307_2/persona04", "task1_results_20260308/persona03"],
    5: ["task1_results_20260307_2/persona05", "task1_results_20260308/persona04"],
    6: ["task1_results_20260308/persona05"],
    7: ["task1_results_20260309/persona06"],
    8: ["task1_results_20260309/persona07"],
    9: ["task1_results_20260321/persona08"],
    10: ["task1_results_20260321/persona09"],
    11: ["task1_results_20260323/persona10"],
    12: ["task1_results_20260323/persona11"],
    13: ["task1_results_20260331/persona12", "task1_results_20260331_tom/persona12"],
    14: ["task1_results_20260331/persona13", "task1_results_20260331_tom/persona13"],
    15: ["task1_results_20260408/persona14"],
    16: ["task1_results_20260408/persona15"],
    17: ["task1_results_20260413/persona16"],
    18: ["task1_results_20260413/persona17"],
    19: ["task1_results_20260420/persona18"],
    20: ["task1_results_20260420/persona19"],
}


def _load(p: Path):
    if not p.exists():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, list) and data:
        return data[0]
    return data


def _summarise(d):
    if not d:
        return "MISSING"
    return f"bdi={d.get('bdi-score')!r:>4} syms={d.get('key-symptoms', [])}"


def main() -> None:
    rows = []
    for pid, batch_dirs in BATCHES_FOR_PATIENT.items():
        new_dir = NEW_ROOT / f"persona{pid - 1}"
        for run_idx in (1, 2, 3):
            new_file = new_dir / f"results_{run_idx}.json"
            new_data = _load(new_file)
            for bd in batch_dirs:
                old_file = BATCH_ROOT / bd / f"results_{run_idx}.json"
                old_data = _load(old_file)
                if old_data is None:
                    continue
                match = (
                    new_data is not None
                    and new_data.get("bdi-score") == old_data.get("bdi-score")
                    and new_data.get("key-symptoms") == old_data.get("key-symptoms")
                )
                rows.append((pid, run_idx, bd, match,
                             _summarise(new_data), _summarise(old_data)))

    by_run = {1: [], 2: [], 3: []}
    for r in rows:
        by_run[r[1]].append(r)

    for run_idx in (1, 2, 3):
        rs = by_run[run_idx]
        matches = sum(1 for r in rs if r[3])
        print(f"=== Run {run_idx} ===  identical bytes in {matches}/{len(rs)} (pid, batch) pairs")
        for pid, _, bd, ok, new_s, old_s in rs:
            tag = "OK" if ok else "DIFF"
            print(f"  pid={pid:>2} {tag:>4} batch={bd}")
            if not ok:
                print(f"    new:   {new_s}")
                print(f"    batch: {old_s}")


if __name__ == "__main__":
    main()
