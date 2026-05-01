"""Side-by-side disagreement Markdown: Submitted Run 2 vs S2 bare-LLM (Gemma 4 31B + guardrails).

For every (round, session) in the test set with both predictions on file
(R1-30 inner join), classifies the case into one of five buckets:

    both_correct           — both pick the gold response
    s2_wins                — S2 picks gold, submitted misses
    submitted_wins         — submitted picks gold, S2 misses
    both_wrong_same        — both miss the gold but agree with each other
    both_wrong_disagree    — both miss the gold and pick differently

Then samples K examples per "interesting" bucket (s2_wins, submitted_wins,
both_wrong_disagree, both_wrong_same) with full transcript context, the
three candidate responses (with English glosses if --gloss is set),
the two systems' picks, the gold pick, and S2's brief_reason.

Outputs:
    outputs/qualitative_T2_submitted_vs_s2.md     full Markdown report
    outputs/W_t2_submitted_vs_s2_summary.csv      per-bucket counts +
                                                   per-gold-class breakdown

Usage:
    python analysis/MentalRiskES_test/qualitative_T2_submitted_vs_s2.py \
        --k-per-bucket 8 --gloss
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from utils import load_config, load_task2_gold, load_task2_test, repo_path
from qualitative_case_studies import _make_deepl, _md_escape, DeepLTranslator  # noqa: F401


SUBMITTED_DIR_REL = "output/mentalriskes_task2/server_submissions"
S2_DIR_REL = "output/mentalriskes_task2_bare_llm/google_gemma-4-31b-it__S2"
S2_MODEL_LABEL = "Gemma 4 31B + guardrails (S2)"
SUBMITTED_LABEL = "Submitted Run 2 (HYB B+)"

logger = logging.getLogger("qualitative_compare")


def opt_to_int(opt: str) -> int:
    return int(opt.replace("option_", ""))


def _load_submitted(predictions_dir: Path, run_idx: int = 2) -> dict[tuple[int, str], int]:
    """{(round, sid): pred_int}."""
    out: dict[tuple[int, str], int] = {}
    for fp in sorted(predictions_dir.glob(f"round*_run{run_idx}.json")):
        rnd = int(fp.stem.split("_")[0].replace("round", ""))
        with open(fp, encoding="utf-8") as fh:
            payload = json.load(fh)
        for entry in payload[0]["predictions"]:
            out[(rnd, entry["id"])] = int(entry["prediction"])
    return out


def _load_s2(predictions_dir: Path) -> dict[tuple[int, str], int]:
    """{(round, sid): pred_int} from S2 server-format JSONs."""
    out: dict[tuple[int, str], int] = {}
    for fp in sorted(predictions_dir.glob("round*.json")):
        rnd = int(fp.stem.replace("round", ""))
        with open(fp, encoding="utf-8") as fh:
            payload = json.load(fh)
        for entry in payload[0]["predictions"]:
            out[(rnd, entry["id"])] = int(entry["prediction"])
    return out


def _load_s2_raw(jsonl_path: Path) -> dict[tuple[int, str], dict]:
    """Load S2 raw.jsonl keyed by (round, session) for brief_reason / model output."""
    out: dict[tuple[int, str], dict] = {}
    if not jsonl_path.exists():
        return out
    with open(jsonl_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                out[(int(rec["round"]), rec["session"])] = rec
            except Exception:
                continue
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k-per-bucket", type=int, default=8,
                        help="Examples per interesting bucket")
    parser.add_argument("--gloss", action="store_true",
                        help="DeepL English gloss inline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    cfg = load_config()
    out_dir = repo_path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out or (out_dir / "qualitative_T2_submitted_vs_s2.md")

    gold = load_task2_gold(cfg)
    test = load_task2_test(cfg)
    submitted = _load_submitted(repo_path(SUBMITTED_DIR_REL), run_idx=2)
    s2 = _load_s2(repo_path(S2_DIR_REL))
    s2_raw = _load_s2_raw(repo_path(S2_DIR_REL) / "raw.jsonl")
    translator = _make_deepl(args.gloss)

    # Build joined records over the (round, session) inner join
    records = []
    for rnd, sess_gold in gold.items():
        for sid, opt_str in sess_gold.items():
            key = (rnd, sid)
            if key not in submitted or key not in s2:
                continue
            g = opt_to_int(opt_str)
            sub = submitted[key]
            s2p = s2[key]
            sub_correct = sub == g
            s2_correct = s2p == g
            if sub_correct and s2_correct:
                bucket = "both_correct"
            elif s2_correct:
                bucket = "s2_wins"
            elif sub_correct:
                bucket = "submitted_wins"
            elif sub == s2p:
                bucket = "both_wrong_same"
            else:
                bucket = "both_wrong_disagree"
            records.append({
                "round": rnd, "session": sid, "gold": g,
                "submitted": sub, "s2": s2p, "bucket": bucket,
            })

    df = pd.DataFrame(records)
    if df.empty:
        out_path.write_text("# Submitted vs S2 disagreement\n\n_No overlapping predictions found._\n", encoding="utf-8")
        return

    # Summary table
    bucket_counts = df["bucket"].value_counts().to_dict()
    n_total = len(df)
    sub_acc = (df["submitted"] == df["gold"]).mean()
    s2_acc = (df["s2"] == df["gold"]).mean()
    agree_rate = (df["submitted"] == df["s2"]).mean()
    by_class_rows = []
    for g in (1, 2, 3):
        sub = df[df["gold"] == g]
        if len(sub) == 0:
            continue
        by_class_rows.append({
            "gold_class": g,
            "n": len(sub),
            "submitted_acc": (sub["submitted"] == g).mean(),
            "s2_acc": (sub["s2"] == g).mean(),
            "agreement_rate": (sub["submitted"] == sub["s2"]).mean(),
        })

    summary_csv = out_dir / "W_t2_submitted_vs_s2_summary.csv"
    pd.DataFrame([
        {"metric": k, "value": v} for k, v in [
            ("n_total", n_total),
            ("submitted_acc_on_join", round(sub_acc, 4)),
            ("s2_acc_on_join", round(s2_acc, 4)),
            ("agreement_rate", round(agree_rate, 4)),
            ("both_correct", bucket_counts.get("both_correct", 0)),
            ("s2_wins", bucket_counts.get("s2_wins", 0)),
            ("submitted_wins", bucket_counts.get("submitted_wins", 0)),
            ("both_wrong_same", bucket_counts.get("both_wrong_same", 0)),
            ("both_wrong_disagree", bucket_counts.get("both_wrong_disagree", 0)),
        ]
    ]).to_csv(summary_csv, index=False)
    pd.DataFrame(by_class_rows).to_csv(out_dir / "W_t2_submitted_vs_s2_per_class.csv", index=False)

    # Sampling per interesting bucket — stratify by gold class to keep coverage
    rng = random.Random(args.seed)
    interesting = ("s2_wins", "submitted_wins", "both_wrong_disagree", "both_wrong_same")
    sampled: dict[str, list[dict]] = {}
    for b in interesting:
        sub = df[df["bucket"] == b].copy()
        if sub.empty:
            sampled[b] = []
            continue
        # stratify by gold class
        per_class = []
        for g, group in sub.groupby("gold"):
            recs = group.to_dict("records")
            rng.shuffle(recs)
            per_class.append(recs)
        # interleave classes round-robin to get diverse first samples
        merged = []
        idx = 0
        while len(merged) < args.k_per_bucket and any(per_class):
            for class_recs in per_class:
                if class_recs:
                    merged.append(class_recs.pop(0))
                if len(merged) >= args.k_per_bucket:
                    break
        sampled[b] = merged

    # Build the Markdown
    md: list[str] = []
    md.append("# Task 2 — Submitted Run 2 vs S2 (Gemma 4 31B + guardrails)")
    md.append("")
    md.append(f"**Comparison cohort:** {n_total} (round, session) pairs (rounds 1–30 × 10 sessions, "
              "the inner join of submitted predictions and S2 predictions on the test set).")
    md.append("")
    md.append("**Headline accuracy on this slice**")
    md.append("")
    md.append("| System | Accuracy on join | Comment |")
    md.append("| --- | --- | --- |")
    md.append(f"| {SUBMITTED_LABEL} | **{sub_acc:.3f}** | matches official leaderboard 0.247 |")
    md.append(f"| {S2_MODEL_LABEL} | **{s2_acc:.3f}** | bare LLM + anti-bias guardrails |")
    md.append(f"| Agreement rate | **{agree_rate:.3f}** | both systems pick the same option |")
    md.append("")
    md.append("**Bucket counts**")
    md.append("")
    md.append("| Bucket | Count | Share |")
    md.append("| --- | --- | --- |")
    for b in ("both_correct", "s2_wins", "submitted_wins", "both_wrong_same", "both_wrong_disagree"):
        c = bucket_counts.get(b, 0)
        md.append(f"| `{b}` | {c} | {c/n_total:.1%} |")
    md.append("")
    md.append("**Per-gold-class accuracy**")
    md.append("")
    md.append("| Gold | n | Submitted acc | S2 acc | Agreement |")
    md.append("| --- | --- | --- | --- | --- |")
    for r in by_class_rows:
        md.append(f"| {r['gold_class']} | {r['n']} | {r['submitted_acc']:.3f} | "
                  f"{r['s2_acc']:.3f} | {r['agreement_rate']:.3f} |")
    md.append("")
    md.append("---")
    md.append("")

    # Detail sections
    section_titles = {
        "s2_wins": "Section A — S2 wins (S2 picks gold, Submitted misses)",
        "submitted_wins": "Section B — Submitted wins (Submitted picks gold, S2 misses)",
        "both_wrong_disagree": "Section C — Both wrong, disagree",
        "both_wrong_same": "Section D — Both wrong, same answer",
    }
    for bucket, title in section_titles.items():
        items = sampled[bucket]
        md.append(f"## {title}")
        md.append("")
        md.append(f"_{bucket_counts.get(bucket, 0)} cases total; sampling {len(items)} stratified by gold class._")
        md.append("")
        if not items:
            md.append("_(none)_\n")
            continue
        for i, rec in enumerate(items, 1):
            rnd = rec["round"]; sid = rec["session"]
            g = rec["gold"]; sub = rec["submitted"]; s2p = rec["s2"]
            md.append(f"### #{i} — {sid} round {rnd}  (gold={g}, submitted={sub}, S2={s2p})")
            md.append("")

            # Recent transcript (last 4 patient turns up to and including current)
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

            md.append("**Candidate options:**")
            md.append("")
            round_data = test[rnd][sid]
            for k in (1, 2, 3):
                label = round_data.get(f"option_{k}", "")
                tags = []
                if k == g: tags.append("**GOLD**")
                if k == sub: tags.append("**SUBMITTED**")
                if k == s2p: tags.append("**S2**")
                tag_str = f" — {' / '.join(tags)}" if tags else ""
                md.append(f"- **Option {k}**{tag_str} (es): _{_md_escape(label)[:500]}_")
                if translator and label.strip():
                    en = translator.translate(label)
                    if en:
                        md.append(f"  - **(en)**: _{_md_escape(en)[:500]}_")
            md.append("")

            # S2 reasoning if available
            s2_rec = s2_raw.get((rnd, sid))
            if s2_rec is not None:
                raw_payload = s2_rec.get("raw")
                if isinstance(raw_payload, dict):
                    reason = raw_payload.get("brief_reason", "")
                    if reason:
                        md.append(f"**S2 reasoning:** _{_md_escape(reason)}_")
                        md.append("")

            md.append("**Notes:** _(fill in)_")
            md.append("")
            md.append("---")
            md.append("")

    out_path.write_text("\n".join(md), encoding="utf-8")
    if translator:
        translator.flush()

    print(f"Wrote {out_path}")
    print(f"\nBucket counts (n_total={n_total}):")
    for b in ("both_correct", "s2_wins", "submitted_wins", "both_wrong_same", "both_wrong_disagree"):
        c = bucket_counts.get(b, 0)
        print(f"  {b:25s} {c:>4} ({c/n_total:.1%})")
    print(f"\nSubmitted acc on join: {sub_acc:.3f}")
    print(f"S2 acc on join: {s2_acc:.3f}")
    print(f"Agreement rate: {agree_rate:.3f}")


if __name__ == "__main__":
    main()
