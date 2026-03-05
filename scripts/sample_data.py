"""
Sample data explorer for RedSM5 and eRisk2023_T1 datasets.

Prints a small representative sample from each file so you can
quickly understand the structure and content of both corpora.
"""

import csv
import os
import re
import random
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

SEPARATOR = "=" * 80


def print_section(title: str):
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


# ─────────────────────────────────────────────
#  1. RedSM5
# ─────────────────────────────────────────────

def sample_redsm5():
    print_section("RedSM5 — redsm5_posts.csv  (Reddit posts, sentence-level depression annotations)")

    posts_path = DATA_DIR / "RedSM5" / "redsm5_posts.csv"
    annot_path = DATA_DIR / "RedSM5" / "redsm5_annotations.csv"

    # --- Posts ---
    with open(posts_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        posts = list(reader)

    print(f"\nTotal posts: {len(posts)}")
    print(f"Columns: {list(posts[0].keys())}")

    print("\n--- 3 sample posts (truncated to 300 chars) ---")
    random.seed(42)
    for post in random.sample(posts, 3):
        text = post["text"][:300] + ("..." if len(post["text"]) > 300 else "")
        print(f"\n  post_id : {post['post_id']}")
        print(f"  text    : {text}")

    # --- Annotations ---
    with open(annot_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        annotations = list(reader)

    print(f"\n\nTotal annotations: {len(annotations)}")
    print(f"Columns: {list(annotations[0].keys())}")

    # Symptom distribution
    symptom_counts = {}
    for row in annotations:
        sym = row["DSM5_symptom"]
        symptom_counts[sym] = symptom_counts.get(sym, 0) + 1
    print("\nSymptom distribution:")
    for sym, cnt in sorted(symptom_counts.items(), key=lambda x: -x[1]):
        print(f"  {sym:25s} {cnt:5d}")

    # Status distribution
    status_counts = {}
    for row in annotations:
        s = row["status"]
        status_counts[s] = status_counts.get(s, 0) + 1
    print(f"\nStatus distribution: {dict(status_counts)}")

    print("\n--- 5 sample annotations ---")
    for row in random.sample(annotations, 5):
        print(f"\n  post_id      : {row['post_id']}")
        print(f"  sentence_id  : {row['sentence_id']}")
        print(f"  sentence_text: {row['sentence_text'][:200]}")
        print(f"  DSM5_symptom : {row['DSM5_symptom']}")
        print(f"  status       : {row['status']}")
        print(f"  explanation  : {row['explanation'][:200]}")


# ─────────────────────────────────────────────
#  2. eRisk2023_T1
# ─────────────────────────────────────────────

def sample_erisk():
    print_section("eRisk2023_T1 — Relevance judgments (g_qrels_majority_2.csv)")

    qrels_path = DATA_DIR / "eRisk2023_T1" / "g_qrels_majority_2.csv"
    consenso_path = DATA_DIR / "eRisk2023_T1" / "g_rels_consenso.csv"
    trec_dir = DATA_DIR / "eRisk2023_T1" / "new_data"

    # --- qrels majority ---
    with open(qrels_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        qrels = list(reader)

    print(f"\nTotal rows: {len(qrels)}")
    print(f"Columns: {list(qrels[0].keys())}")

    # Query / relevance distribution
    queries = set(row["query"] for row in qrels)
    rel_counts = {}
    for row in qrels:
        r = row["rel"]
        rel_counts[r] = rel_counts.get(r, 0) + 1
    print(f"Unique queries: {len(queries)}  (values: {sorted(queries)})")
    print(f"Relevance distribution: {dict(sorted(rel_counts.items()))}")

    print("\n--- 5 sample rows ---")
    random.seed(42)
    for row in random.sample(qrels, 5):
        print(f"  query={row['query']}  docid={row['docid']}  rel={row['rel']}")

    # --- consenso ---
    print_section("eRisk2023_T1 — Relevance judgments (g_rels_consenso.csv)")

    with open(consenso_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        consenso = list(reader)

    print(f"\nTotal rows: {len(consenso)}")
    print(f"Columns: {list(consenso[0].keys())}")

    rel_counts_c = {}
    for row in consenso:
        r = row["rel"]
        rel_counts_c[r] = rel_counts_c.get(r, 0) + 1
    print(f"Relevance distribution: {dict(sorted(rel_counts_c.items()))}")

    print("\n--- 5 sample rows ---")
    for row in random.sample(consenso, 5):
        print(f"  query={row['query']}  docid={row['docid']}  rel={row['rel']}")

    # --- .trec files ---
    print_section("eRisk2023_T1 — TREC user files (new_data/*.trec)")

    trec_files = sorted(trec_dir.glob("*.trec"))
    print(f"\nTotal .trec files (users): {len(trec_files)}")

    # Sample 2 files, show first few docs from each
    sample_files = random.sample(trec_files, min(2, len(trec_files)))
    for trec_file in sample_files:
        print(f"\n--- File: {trec_file.name} ---")
        content = trec_file.read_text(encoding="utf-8", errors="replace")

        # Count documents
        doc_count = content.count("<DOC>")
        print(f"  Total <DOC> entries: {doc_count}")

        # Extract first 3 docs
        docs = re.findall(
            r"<DOCNO>(.*?)</DOCNO>\s*<TEXT>(.*?)</TEXT>",
            content,
            re.DOTALL,
        )
        print(f"  First 3 documents:")
        for docno, text in docs[:3]:
            text_clean = text.strip()[:200]
            print(f"    DOCNO: {docno.strip()}")
            print(f"    TEXT : {text_clean}")
            print()

    # --- Cross-reference: link docids in qrels to trec content ---
    print_section("Cross-reference: qrels docid -> TREC content (2 examples)")

    # Pick 2 relevant docs (rel=1) from qrels
    relevant = [r for r in qrels if r["rel"] == "1"]
    samples = random.sample(relevant, min(2, len(relevant)))

    for row in samples:
        docid = row["docid"]
        # docid format: s_<user>_<post>_<sentence>
        parts = docid.split("_")
        user_id = parts[1] if len(parts) >= 2 else "?"
        user_file = trec_dir / f"s_{user_id}.trec"

        print(f"\n  docid: {docid}  (query={row['query']}, rel={row['rel']})")
        if user_file.exists():
            content = user_file.read_text(encoding="utf-8", errors="replace")
            match = re.search(
                rf"<DOCNO>\s*{re.escape(docid)}\s*</DOCNO>\s*<TEXT>(.*?)</TEXT>",
                content,
                re.DOTALL,
            )
            if match:
                print(f"  TEXT  : {match.group(1).strip()[:300]}")
            else:
                print(f"  (docid not found in {user_file.name})")
        else:
            print(f"  (file {user_file.name} not found)")


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("DATA SAMPLE EXPLORER")
    print(f"Data directory: {DATA_DIR}")
    sample_redsm5()
    sample_erisk()
    print(f"\n{SEPARATOR}")
    print("  Done!")
    print(SEPARATOR)
