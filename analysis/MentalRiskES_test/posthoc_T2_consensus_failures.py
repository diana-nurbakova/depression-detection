"""Task 2 consensus-failure analysis.

Identifies (round, session) pairs in the test set where EVERY system we
tested missed the gold response. Useful for the paper's "task floor"
discussion — these rounds expose either inherent ambiguity in the task,
gold-label quirks, or genuine clinical-judgment limits beyond what current
LLMs achieve.

Systems considered (only those with predictions on disk):
  - Submitted Run 2 (HYB B+, R1-30)
  - Submitted Run 2 replay (HYB B+, full 82 rounds)
  - Gemma 4 31B bare (S, S2, S3, S4, R2)
  - Gemma 3 27B bare (S)
  - Llama-3.3-70B bare (S)

Outputs:
  W_t2_consensus_failure_stats.csv     per-class consensus-failure counts
  W_t2_consensus_failures.md           Markdown with K stratified examples,
                                        DeepL English glosses (if --gloss),
                                        full transcript + 3 candidates +
                                        every system's pick.

Usage:
  python analysis/MentalRiskES_test/posthoc_T2_consensus_failures.py \
      --k-per-class 6 --gloss
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter
from pathlib import Path

import pandas as pd

from utils import load_config, load_task2_gold, load_task2_test, repo_path
from qualitative_case_studies import _make_deepl, _md_escape  # noqa: F401

logger = logging.getLogger("consensus_t2")


def _opt_to_int(opt: str) -> int:
    return int(opt.replace("option_", ""))


def _load_server_format(predictions_dir: Path, run_idx: int | None = None) -> dict[tuple[int, str], int]:
    """Load round{N}.json or round{N}_run{R}.json into {(round, sid): pred}."""
    pattern = f"round*_run{run_idx}.json" if run_idx is not None else "round*.json"
    out: dict[tuple[int, str], int] = {}
    for fp in sorted(predictions_dir.glob(pattern)):
        rnd_part = fp.stem.split("_")[0].replace("round", "")
        try:
            rnd = int(rnd_part)
        except ValueError:
            continue
        with open(fp, encoding="utf-8") as fh:
            payload = json.load(fh)
        for entry in payload[0]["predictions"]:
            out[(rnd, entry["id"])] = int(entry["prediction"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k-per-class", type=int, default=6,
                        help="Examples per gold class (1/2/3) in the case-study Markdown")
    parser.add_argument("--gloss", action="store_true", help="DeepL English gloss inline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    cfg = load_config()
    out_dir = repo_path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    gold = load_task2_gold(cfg)
    test = load_task2_test(cfg)

    # Discover all the prediction sources we have on disk
    sources: dict[str, dict[tuple[int, str], int]] = {}
    sources["Submitted Run 2 (R1-30)"] = _load_server_format(
        repo_path("output/mentalriskes_task2/server_submissions"), run_idx=2)
    sources["Submitted Run 2 replay (full)"] = _load_server_format(
        repo_path("output/mentalriskes_task2_test_replay/server_submissions"), run_idx=2)
    bare_root = repo_path("output/mentalriskes_task2_bare_llm")
    if bare_root.exists():
        for d in sorted(bare_root.iterdir()):
            if not d.is_dir() or "__" not in d.name:
                continue
            # Only test-cohort runs (no `__test`/`__trial`/`__simulated` suffix beyond the legacy two-piece)
            parts = d.name.split("__")
            if len(parts) != 2:  # has cohort suffix, skip
                continue
            preds = _load_server_format(d, run_idx=None)
            if preds:
                model_short, mode = parts
                sources[f"{model_short} ({mode})"] = preds

    if not sources:
        logger.warning("No prediction sources found")
        return

    # Build per-(round, session) record: gold + every system's pick + correct?
    rows = []
    for rnd, sess_gold in gold.items():
        for sid, opt_str in sess_gold.items():
            g = _opt_to_int(opt_str)
            picks = {}
            in_all = True
            for sys_name, preds in sources.items():
                if (rnd, sid) in preds:
                    picks[sys_name] = preds[(rnd, sid)]
                else:
                    in_all = False
            if not in_all:
                continue
            n_correct = sum(1 for v in picks.values() if v == g)
            rows.append({
                "round": rnd, "session": sid, "gold": g,
                "n_systems": len(picks), "n_correct": n_correct,
                "all_wrong": int(n_correct == 0),
                "all_correct": int(n_correct == len(picks)),
                **{f"sys::{k}": v for k, v in picks.items()},
            })

    if not rows:
        logger.warning("No (round, session) pairs covered by ALL systems")
        return

    df = pd.DataFrame(rows)
    n_total = len(df)
    sys_cols = [c for c in df.columns if c.startswith("sys::")]
    sys_names = [c.replace("sys::", "") for c in sys_cols]

    # Stats: per-class consensus-failure counts
    stats = []
    for g in (1, 2, 3):
        sub = df[df["gold"] == g]
        if len(sub) == 0:
            continue
        stats.append({
            "gold_class": g, "n": len(sub),
            "n_all_wrong": int(sub["all_wrong"].sum()),
            "pct_all_wrong": float(sub["all_wrong"].mean()),
            "n_all_correct": int(sub["all_correct"].sum()),
            "pct_all_correct": float(sub["all_correct"].mean()),
            "mean_correct_systems": float(sub["n_correct"].mean()),
        })
    stats.append({
        "gold_class": "ALL", "n": n_total,
        "n_all_wrong": int(df["all_wrong"].sum()),
        "pct_all_wrong": float(df["all_wrong"].mean()),
        "n_all_correct": int(df["all_correct"].sum()),
        "pct_all_correct": float(df["all_correct"].mean()),
        "mean_correct_systems": float(df["n_correct"].mean()),
    })
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(out_dir / "W_t2_consensus_failure_stats.csv", index=False)
    print("Per-class consensus stats:")
    print(stats_df.to_string(index=False))

    # Sample stratified by gold class
    rng = random.Random(args.seed)
    failures = df[df["all_wrong"] == 1].copy()
    sampled: dict[int, list[dict]] = {}
    for g in (1, 2, 3):
        candidates = failures[failures["gold"] == g].to_dict("records")
        rng.shuffle(candidates)
        sampled[g] = candidates[: args.k_per_class]

    translator = _make_deepl(args.gloss)

    md = ["# Task 2 — Consensus failures (every tested system wrong)\n"]
    md.append(f"Total comparable (round, session) pairs (those covered by every "
              f"system in the registry): **{n_total}**.\n")
    md.append("Systems compared: " + "; ".join(f"`{n}`" for n in sys_names) + ".\n")
    md.append("**Per-class consensus statistics**\n")
    md.append("| Gold class | n | All-wrong | All-correct | Mean correct systems |")
    md.append("| --- | --- | --- | --- | --- |")
    for r in stats:
        md.append(f"| {r['gold_class']} | {r['n']} | "
                  f"{r['n_all_wrong']} ({r['pct_all_wrong']:.1%}) | "
                  f"{r['n_all_correct']} ({r['pct_all_correct']:.1%}) | "
                  f"{r['mean_correct_systems']:.2f} / {len(sys_names)} |")
    md.append("")

    # Bucket Markdown
    for g in (1, 2, 3):
        md.append(f"## Gold = {g} ({len(failures[failures['gold'] == g])} consensus-failures, "
                  f"sampling {len(sampled[g])})\n")
        if not sampled[g]:
            md.append("_(none)_\n")
            continue
        for i, rec in enumerate(sampled[g], 1):
            rnd = int(rec["round"]); sid = rec["session"]
            md.append(f"### #{i} — {sid} round {rnd} (gold = {g})")
            md.append("")

            # Recent transcript
            md.append("**Recent transcript:**")
            md.append("")
            for r_back in range(max(1, rnd - 3), rnd + 1):
                if r_back not in test or sid not in test[r_back]:
                    continue
                turn = test[r_back][sid]
                pi_raw = turn.get("patient_input", "")
                line = f"- **R{r_back} patient (es)**: _{_md_escape(pi_raw)[:300]}_"
                if translator and pi_raw.strip():
                    en = translator.translate(pi_raw)
                    if en:
                        line += f"\n  - **(en)**: _{_md_escape(en)[:300]}_"
                md.append(line)
            md.append("")

            # Candidates
            md.append("**Candidate options:**")
            md.append("")
            round_data = test[rnd][sid]
            for k in (1, 2, 3):
                label = round_data.get(f"option_{k}", "")
                tag = "**GOLD**" if k == g else ""
                tag_str = f" — {tag}" if tag else ""
                md.append(f"- **Option {k}**{tag_str} (es): _{_md_escape(label)[:500]}_")
                if translator and label.strip():
                    en = translator.translate(label)
                    if en:
                        md.append(f"  - **(en)**: _{_md_escape(en)[:500]}_")
            md.append("")

            # Picks per system
            md.append("**System picks** (none correct):")
            md.append("")
            md.append("| System | Pick |")
            md.append("| --- | --- |")
            for sys_name in sys_names:
                pick = rec.get(f"sys::{sys_name}")
                md.append(f"| {sys_name} | {pick} |")
            md.append("")
            md.append("**Notes:** _(fill in)_")
            md.append("")
            md.append("---")
            md.append("")

    out_path = out_dir / "W_t2_consensus_failures.md"
    out_path.write_text("\n".join(md), encoding="utf-8")
    if translator:
        translator.flush()
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
