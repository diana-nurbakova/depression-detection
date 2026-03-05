"""
Exploratory analysis of BDI-Sen-2.0 and eRisk-2025 datasets.

Reproduces all statistics in analysis/dataset_exploration.md.

Usage:
    uv run python analysis/explore_datasets.py
"""

from __future__ import annotations

import csv
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
BDISEN_DIR = DATA_ROOT / "BDI-Sen" / "full_dataset"
ERISK_DIR = DATA_ROOT / "eRisk-2025" / "eRisk25-datasets"
T1_DIR = ERISK_DIR / "t1-depression-symptom-ranking"
T2_DIR = ERISK_DIR / "t2-early-contextualized-depression"


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def fmt_pct(n: int, total: int) -> str:
    return f"{n:,} ({n / total * 100:.1f}%)" if total else "0"


def print_header(title: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}")


def print_table(headers: list[str], rows: list[list], col_align: str | None = None) -> None:
    """Simple aligned table printer."""
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    hdr = " | ".join(str(h).ljust(w) for h, w in zip(headers, widths))
    sep = "-+-".join("-" * w for w in widths)
    print(hdr)
    print(sep)
    for row in rows:
        print(" | ".join(str(c).ljust(w) for c, w in zip(row, widths)))


# ── BDI-Sen ──────────────────────────────────────────────────────────────────

def analyse_bdisen() -> None:
    print_header("BDI-Sen-2.0")

    # --- Load files ---
    mv_path = BDISEN_DIR / "bdi_majority_vote.jsonl"
    uni_path = BDISEN_DIR / "bdi_unified.jsonl"
    if not mv_path.exists():
        print(f"  [SKIP] {mv_path} not found"); return

    mv = load_jsonl(mv_path)
    uni = load_jsonl(uni_path)

    # --- File-level counts ---
    print(f"\n  bdi_majority_vote.jsonl : {len(mv):,} records (flat: one per sentence×symptom)")
    print(f"  bdi_unified.jsonl      : {len(uni):,} records (one per unique sentence)")

    # --- Split counts ---
    splits_dir = BDISEN_DIR / "splits"
    if splits_dir.exists():
        print("\n  Splits:")
        for sp in sorted(splits_dir.glob("*.jsonl")):
            recs = load_jsonl(sp)
            print(f"    {sp.name:<30s} {len(recs):>5,} records")

    # --- Label distribution ---
    total = len(mv)
    pos = sum(1 for r in mv if r["label"] == 1)
    neg = total - pos
    print(f"\n  Label distribution (flat):")
    print(f"    label=1 (positive) : {fmt_pct(pos, total)}")
    print(f"    label=0 (negative) : {fmt_pct(neg, total)}")

    # --- Severity distribution ---
    sev_counts: Counter = Counter()
    for r in mv:
        sev_counts[r.get("severity")] += 1

    print(f"\n  Severity distribution:")
    rows = []
    for s in [0, 1, 2, 3, None]:
        label = str(s) if s is not None else "null"
        rows.append([label, f"{sev_counts[s]:,}", f"{sev_counts[s] / total * 100:.1f}%"])
    print_table(["Severity", "Count", "%"], rows)

    # --- Severity × Label cross-tab ---
    print(f"\n  Severity × Label cross-tab:")
    crosstab: dict[str, Counter] = {}
    for r in mv:
        sev_key = str(r.get("severity")) if r.get("severity") is not None else "null"
        crosstab.setdefault(sev_key, Counter())[r["label"]] += 1
    rows = []
    for s in ["0", "1", "2", "3", "null"]:
        c = crosstab.get(s, Counter())
        rows.append([f"severity={s}", str(c.get(0, 0)), str(c.get(1, 0))])
    print_table(["", "label=0", "label=1"], rows)

    # --- Per-symptom breakdown ---
    symp_total: Counter = Counter()
    symp_pos: Counter = Counter()
    for r in mv:
        symp_total[r["symptom"]] += 1
        if r["label"] == 1:
            symp_pos[r["symptom"]] += 1

    print(f"\n  Per-symptom breakdown (sorted by total annotations desc):")
    rows = []
    for sym, tot in symp_total.most_common():
        p = symp_pos[sym]
        rows.append([sym, str(tot), str(p), f"{p / tot * 100:.1f}%"])
    print_table(["Symptom", "Annotations", "Positive", "Pos%"], rows)

    # --- Annotations per sentence ---
    ann_counts = Counter(len(r["annotations"]) for r in uni)
    print(f"\n  Annotations per sentence:")
    mean_ann = sum(k * v for k, v in ann_counts.items()) / len(uni)
    max_ann = max(ann_counts)
    multi = sum(v for k, v in ann_counts.items() if k > 1)
    rows = []
    for k in sorted(ann_counts):
        rows.append([str(k), str(ann_counts[k]), f"{ann_counts[k] / len(uni) * 100:.1f}%"])
    print_table(["# Annotations", "Sentences", "%"], rows)
    print(f"    Mean: {mean_ann:.2f}  |  Max: {max_ann}  |  Multi-symptom: {multi} ({multi / len(uni) * 100:.1f}%)")

    # --- Sentence length ---
    lengths = [len(r["sentence"].split()) for r in uni]
    lengths.sort()
    n = len(lengths)
    mean_len = sum(lengths) / n
    median_len = lengths[n // 2]
    print(f"\n  Sentence length (words):")
    print(f"    Mean: {mean_len:.1f}  |  Median: {median_len}  |  Min: {lengths[0]}  |  Max: {lengths[-1]}")

    # --- Unique sentence stats ---
    sents_with_pos = sum(1 for r in uni if any(a["label"] == 1 for a in r["annotations"]))
    sents_only_neg = len(uni) - sents_with_pos
    print(f"\n  Sentence-level: {sents_with_pos} with >=1 positive label, {sents_only_neg} all-negative")


# ── eRisk-2025 Task 1 ───────────────────────────────────────────────────────

def count_trec_docs(path: Path) -> int:
    """Count <DOC> elements in a .trec file (regex, fast)."""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text.count("<DOCNO>")


def analyse_erisk_t1() -> None:
    print_header("eRisk-2025 — Task 1: Depression Symptom Ranking")

    # --- Corpus ---
    trec_dir = T1_DIR / "erisk25-t1-dataset" / "erisk25-t1-dataset"
    if not trec_dir.exists():
        print(f"  [SKIP] {trec_dir} not found"); return

    trec_files = sorted(trec_dir.glob("s_*.trec"))
    print(f"\n  .trec files (users): {len(trec_files):,}")

    # Count sentences per file (sample first 20 for speed, then full)
    print("  Counting sentences (this may take a while on 6k+ files)...")
    per_user: list[int] = []
    for tf in trec_files:
        per_user.append(count_trec_docs(tf))

    total_sents = sum(per_user)
    per_user_sorted = sorted(per_user)
    n = len(per_user_sorted)
    print(f"  Total sentences: {total_sents:,}")
    print(f"  Sentences/user — Mean: {total_sents / n:,.1f}  |  Median: {per_user_sorted[n // 2]:,}")
    print(f"    Min: {per_user_sorted[0]:,}  |  Max: {per_user_sorted[-1]:,}")
    print(f"    P10: {per_user_sorted[int(n * 0.1)]:,}  |  P90: {per_user_sorted[int(n * 0.9)]:,}")

    # --- qrels ---
    for qrels_name in ["qrels_consensus_merged.csv", "qrels_majority_merged.csv"]:
        qrels_path = T1_DIR / qrels_name
        if not qrels_path.exists():
            print(f"  [SKIP] {qrels_path} not found"); continue

        with open(qrels_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows_list = list(reader)

        total_q = len(rows_list)
        relevant = sum(1 for r in rows_list if r["relevant"] == "True")
        queries = Counter(r["query"] for r in rows_list)
        users = set(r["doc_id"].split("_")[0] for r in rows_list)

        print(f"\n  {qrels_name}:")
        print(f"    Total annotations: {total_q:,}")
        print(f"    Relevant (True): {fmt_pct(relevant, total_q)}")
        print(f"    Unique users referenced: {len(users):,}")
        print(f"    Queries (symptoms): {len(queries)} — range {min(queries.values())}-{max(queries.values())} annotations each")

        # Per-query relevance
        query_rel: dict[str, list[str]] = {}
        for r in rows_list:
            query_rel.setdefault(r["query"], []).append(r["relevant"])

        print(f"    Per-query relevance rate:")
        table_rows = []
        for q in sorted(query_rel, key=lambda x: int(x)):
            vals = query_rel[q]
            rel = sum(1 for v in vals if v == "True")
            table_rows.append([f"Q{q}", str(len(vals)), str(rel), f"{rel / len(vals) * 100:.1f}%"])
        print_table(["Query", "Total", "Relevant", "Rate"], table_rows)

    # --- Agreement between consensus & majority ---
    cons_path = T1_DIR / "qrels_consensus_merged.csv"
    maj_path = T1_DIR / "qrels_majority_merged.csv"
    if cons_path.exists() and maj_path.exists():
        cons_data = {}
        with open(cons_path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                cons_data[(r["query"], r["doc_id"])] = r["relevant"]
        maj_data = {}
        with open(maj_path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                maj_data[(r["query"], r["doc_id"])] = r["relevant"]

        common = set(cons_data) & set(maj_data)
        agree = sum(1 for k in common if cons_data[k] == maj_data[k])
        borderline = sum(1 for k in common if maj_data[k] == "True" and cons_data[k] == "False")
        print(f"\n  Consensus vs Majority agreement:")
        print(f"    Common pairs: {len(common):,}")
        print(f"    Agree: {agree:,} ({agree / len(common) * 100:.1f}%)")
        print(f"    Borderline (majority=True, consensus=False): {borderline:,}")


# ── eRisk-2025 Task 2 ───────────────────────────────────────────────────────

def analyse_erisk_t2() -> None:
    print_header("eRisk-2025 — Task 2: Early Contextualized Depression Detection")

    data_dir = T2_DIR / "final-eriskt2-dataset-with-ground-truth" / "final-eriskt2-dataset-with-ground-truth"
    combined_dir = data_dir / "all_combined"
    gt_path = data_dir / "shuffled_ground_truth_labels.txt"

    if not combined_dir.exists():
        print(f"  [SKIP] {combined_dir} not found"); return

    # --- Ground truth ---
    labels: dict[str, int] = {}
    if gt_path.exists():
        with open(gt_path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    labels[parts[0]] = int(parts[1])

    total_users = len(labels)
    depressed = sum(1 for v in labels.values() if v == 1)
    control = total_users - depressed
    print(f"\n  Subjects: {total_users}")
    print(f"    Depressed (label=1): {fmt_pct(depressed, total_users)}")
    print(f"    Control   (label=0): {fmt_pct(control, total_users)}")

    # --- JSON files ---
    json_files = sorted(combined_dir.glob("subject_*.json"))
    print(f"  JSON files: {len(json_files):,}")

    total_submissions = 0
    total_comments = 0
    target_submissions = 0
    target_comments = 0
    per_user_stats: list[dict] = []

    for jf in json_files:
        with open(jf, encoding="utf-8") as f:
            data = json.load(f)

        user_subs = 0
        user_comments = 0
        user_target_subs = 0
        user_target_comments = 0

        for entry in data:
            sub = entry.get("submission", {})
            user_subs += 1
            if sub.get("target"):
                user_target_subs += 1

            for comm in entry.get("comments", []):
                user_comments += 1
                if comm.get("target"):
                    user_target_comments += 1

        total_submissions += user_subs
        total_comments += user_comments
        target_submissions += user_target_subs
        target_comments += user_target_comments

        uid = jf.stem  # e.g. "subject_01ZzrIT"
        label = labels.get(uid, -1)
        per_user_stats.append({
            "uid": uid,
            "label": label,
            "entries": len(data),
            "subs": user_subs,
            "comments": user_comments,
            "target_writings": user_target_subs + user_target_comments,
        })

    print(f"\n  Total submissions: {total_submissions:,}")
    print(f"  Total comments: {total_comments:,}")
    print(f"  Total writings: {total_submissions + total_comments:,}")
    print(f"  Target-subject submissions: {target_submissions:,}")
    print(f"  Target-subject comments: {target_comments:,}")

    # --- Per-label averages ---
    for lbl, lbl_name in [(1, "Depressed"), (0, "Control")]:
        subset = [u for u in per_user_stats if u["label"] == lbl]
        if not subset:
            continue
        avg_entries = sum(u["entries"] for u in subset) / len(subset)
        avg_target = sum(u["target_writings"] for u in subset) / len(subset)
        print(f"\n  {lbl_name} users (n={len(subset)}):")
        print(f"    Mean entries: {avg_entries:,.1f}  |  Mean target writings: {avg_target:,.1f}")

    # --- ADHD mention check ---
    adhd_mentions = 0
    files_with_adhd = 0
    for jf in json_files:
        text = jf.read_text(encoding="utf-8", errors="replace")
        count = len(re.findall(r"(?i)\badhd\b", text))
        if count > 0:
            files_with_adhd += 1
            adhd_mentions += count

    print(f"\n  ADHD mentions in T2 user posts:")
    print(f"    Files with 'ADHD': {files_with_adhd} / {len(json_files)}")
    print(f"    Total mentions: {adhd_mentions:,}")
    print(f"    (No ADHD-specific annotations or labels exist)")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Dataset Exploratory Analysis")
    print(f"Data root: {DATA_ROOT}")

    analyse_bdisen()
    analyse_erisk_t1()
    analyse_erisk_t2()

    print(f"\n{'=' * 72}")
    print("  Done.")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
