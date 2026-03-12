"""Export qualitative analysis JSONs: one per symptom with ranked sentences and text.

Usage:
    python scripts/export_qualitative.py --run INSALyon_LLM_cascade --top 50
    python scripts/export_qualitative.py --run all --top 30

Reads TREC ranking files from output/rankings/final/, looks up sentence text
from the corpus, and writes one JSON per symptom to output/qualitative/<run>/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def load_symptoms(symptoms_path: Path) -> dict[int, dict]:
    """Load symptom metadata from symptoms.yaml."""
    with open(symptoms_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    symptoms = {}
    for s in data["symptoms"]:
        sid = s["item_number"]
        symptoms[sid] = {
            "symptom_id": sid,
            "symptom_name": f"ASRS Item {sid}",
            "symptom_text": s["text"],
            "factor": s["factor"],
            "subcluster": s["subcluster"],
        }
    return symptoms


def parse_trec_ranking(trec_path: Path, top_n: int) -> dict[int, list[dict]]:
    """Parse a TREC ranking file into {symptom_id: [{docno, rank, score}]}."""
    per_symptom: dict[int, list[dict]] = {}

    with open(trec_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 6:
                parts = line.strip().split()
            if len(parts) < 6:
                continue

            symptom_id = int(parts[0])
            docno = parts[2]
            rank = int(parts[3])
            score = float(parts[4])

            if symptom_id not in per_symptom:
                per_symptom[symptom_id] = []

            if len(per_symptom[symptom_id]) < top_n:
                per_symptom[symptom_id].append({
                    "docno": docno,
                    "rank": rank,
                    "score": score,
                })

    return per_symptom


def load_sentence_texts(corpus_dir: Path, docnos: set[str]) -> dict[str, str]:
    """Load sentence texts from TREC corpus for given docnos.

    Scans all corpus files since filenames don't correspond to docno user IDs.
    """
    from hipert.data.trec_parser import parse_trec_file, iter_trec_files
    from tqdm import tqdm

    remaining = set(docnos)
    texts: dict[str, str] = {}
    trec_files = list(iter_trec_files(corpus_dir))

    print(f"  Scanning {len(trec_files)} corpus files for {len(docnos)} sentences...")

    for trec_path in tqdm(trec_files, desc="  Loading texts", unit="file"):
        sentences = parse_trec_file(trec_path)
        for sent in sentences:
            if sent.docno in remaining:
                texts[sent.docno] = sent.text
                remaining.discard(sent.docno)

        if not remaining:
            break

    print(f"  Found text for {len(texts)}/{len(docnos)} sentences")
    return texts


def main():
    parser = argparse.ArgumentParser(description="Export qualitative analysis JSONs")
    parser.add_argument(
        "--run", default="all",
        help="Run name (e.g. INSALyon_LLM_cascade) or 'all'",
    )
    parser.add_argument("--top", type=int, default=50, help="Top N sentences per symptom")
    parser.add_argument(
        "--rankings-dir", type=Path,
        default=PROJECT_ROOT / "output" / "rankings" / "final",
    )
    parser.add_argument(
        "--corpus-dir", type=Path, default=None,
        help="Corpus directory (default: from pipeline.yaml)",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "output" / "qualitative",
    )
    args = parser.parse_args()

    # Load corpus dir from config if not specified
    if args.corpus_dir is None:
        config_path = PROJECT_ROOT / "config" / "pipeline.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        args.corpus_dir = PROJECT_ROOT / cfg["data"]["corpus_dir"]

    # Determine which runs to process
    if args.run == "all":
        trec_files = list(args.rankings_dir.glob("*.trec"))
    else:
        trec_files = [args.rankings_dir / f"{args.run}.trec"]
        if not trec_files[0].exists():
            print(f"ERROR: {trec_files[0]} not found")
            sys.exit(1)

    # Load symptom metadata
    symptoms = load_symptoms(PROJECT_ROOT / "config" / "symptoms.yaml")

    for trec_path in sorted(trec_files):
        run_name = trec_path.stem
        print(f"\nProcessing {run_name} (top {args.top})...")

        # Parse rankings
        per_symptom = parse_trec_ranking(trec_path, args.top)

        # Collect all needed docnos
        all_docnos = set()
        for entries in per_symptom.values():
            for e in entries:
                all_docnos.add(e["docno"])

        # Load sentence texts from corpus
        texts = load_sentence_texts(args.corpus_dir, all_docnos)

        # Write one JSON per symptom
        out_dir = args.output_dir / run_name
        out_dir.mkdir(parents=True, exist_ok=True)

        for symptom_id in range(1, 19):
            meta = symptoms.get(symptom_id, {
                "symptom_id": symptom_id,
                "symptom_name": f"ASRS Item {symptom_id}",
                "symptom_text": "",
                "factor": "",
                "subcluster": "",
            })

            entries = per_symptom.get(symptom_id, [])
            sentences = []
            for e in entries:
                sentences.append({
                    "rank": e["rank"],
                    "sentence_id": e["docno"],
                    "score": e["score"],
                    "text": texts.get(e["docno"], "[text not found]"),
                })

            output = {
                **meta,
                "run": run_name,
                "total_ranked": len(sentences),
                "sentences": sentences,
            }

            out_path = out_dir / f"symptom_{symptom_id}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"  Written {len(per_symptom)} symptom files to {out_dir}")


if __name__ == "__main__":
    main()
