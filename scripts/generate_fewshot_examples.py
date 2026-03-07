"""Generate few-shot examples for all 18 ASRS symptoms.

Reads candidate TSVs (already extracted from RedSM5, BDI-Sen, eRisk 2025/2023,
eRisk 2026), selects the best candidate per score level (0-3), and uses
GPT-4o-mini to produce structured annotations following the annotation protocol.

Usage:
    uv run python scripts/generate_fewshot_examples.py
    uv run python scripts/generate_fewshot_examples.py --symptoms 1,2,3
    uv run python scripts/generate_fewshot_examples.py --dry-run
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
import time
from pathlib import Path

import requests
import yaml
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")

CANDIDATES_DIR = PROJECT_ROOT / "candidates"
ANNOTATIONS_DIR = PROJECT_ROOT / "annotations"
SYMPTOMS_YAML = PROJECT_ROOT / "config" / "symptoms.yaml"
DEPRESYM_DIR = PROJECT_ROOT / "depresym_analysis"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL = "gpt-4o-mini"

# Source priority per score level (from annotation_protocol_spec_v3.md Section 2.5)
SOURCE_PRIORITY = {
    0: ["score0_random", "erisk2025_nonrel", "retrieval"],
    1: ["bdisen", "redsm5", "erisk2025_boundary", "erisk2023_boundary", "retrieval"],
    2: ["retrieval", "erisk2025_agreement", "erisk2023_agreement"],
    3: ["retrieval"],
}


def load_symptoms() -> dict[int, dict]:
    """Load symptom definitions from YAML."""
    with open(SYMPTOMS_YAML, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return {s["item_number"]: s for s in raw["symptoms"]}


def load_candidates(symptom_id: int) -> list[dict]:
    """Load candidates TSV for a symptom."""
    path = CANDIDATES_DIR / f"symptom_{symptom_id:02d}_candidates.tsv"
    if not path.exists():
        logger.warning("No candidates for symptom %d at %s", symptom_id, path)
        return []

    candidates = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            candidates.append(row)
    return candidates


def load_score0_pool() -> list[dict]:
    """Load the shared score-0 pool."""
    path = CANDIDATES_DIR / "score0_pool.tsv"
    if not path.exists():
        return []
    pool = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            pool.append(row)
    return pool


def load_external_pools(symptom_id: int) -> dict[str, list[dict]]:
    """Load external candidate pools for a symptom.

    Returns dict of source_category -> list of candidates, normalized to
    have at least: text, source, pre, post, docno keys.
    """
    pools: dict[str, list[dict]] = {}

    # Fallback DSM-5 depression symptoms for ASRS items without direct mapping
    _REDSM5_FALLBACK = {
        3: "COGNITIVE_ISSUES",   # remembering → cognitive
        14: "PSYCHOMOTOR",       # difficulty stopping → psychomotor
        15: "PSYCHOMOTOR",       # talking too much → psychomotor
        16: "COGNITIVE_ISSUES",  # finishing sentences → cognitive
        17: "PSYCHOMOTOR",       # waiting turn → psychomotor
        18: "PSYCHOMOTOR",       # interrupting → psychomotor
    }

    # RedSM5 confounder pool
    redsm5_path = CANDIDATES_DIR / "redsm5_confounder_pool.tsv"
    if redsm5_path.exists():
        with open(redsm5_path, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                row_sid = int(row.get("symptom_id", -1))
                # Direct match OR fallback by DSM-5 symptom type
                fallback_dsm5 = _REDSM5_FALLBACK.get(symptom_id)
                if row_sid == symptom_id or (
                    fallback_dsm5
                    and row.get("dsm5_symptom") == fallback_dsm5
                    and row_sid != symptom_id
                ):
                    pools.setdefault("redsm5", []).append({
                        "docno": row.get("sentence_id", ""),
                        "text": row.get("sentence_text", ""),
                        "pre": "",
                        "post": "",
                        "source": "redsm5",
                        "source_detail": {
                            "dsm5_symptom": row.get("dsm5_symptom", ""),
                            "explanation": row.get("explanation", ""),
                        },
                    })

    # BDI-Sen confounder pool (prefer severity=1 for score-1)
    bdisen_path = CANDIDATES_DIR / "bdisen_confounder_pool.tsv"
    if bdisen_path.exists():
        with open(bdisen_path, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                if int(row.get("symptom_id", -1)) == symptom_id:
                    pools.setdefault("bdisen", []).append({
                        "docno": f"bdisen_{row.get('bdisen_symptom', '')}",
                        "text": row.get("sentence_text", ""),
                        "pre": "",
                        "post": "",
                        "source": "bdisen",
                        "source_detail": {
                            "bdisen_symptom": row.get("bdisen_symptom", ""),
                            "severity": int(row.get("severity", 0)),
                        },
                    })
        # Sort BDI-Sen by severity (prefer severity=1)
        if "bdisen" in pools:
            pools["bdisen"].sort(key=lambda x: x["source_detail"].get("severity", 9))

    # eRisk 2025 T1 boundary pool
    erisk2025_path = CANDIDATES_DIR / "erisk2025_t1_boundary_pool.tsv"
    if erisk2025_path.exists():
        with open(erisk2025_path, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                if int(row.get("symptom_id", -1)) == symptom_id:
                    pools.setdefault("erisk2025_boundary", []).append({
                        "docno": row.get("docid", ""),
                        "text": row.get("text", ""),
                        "pre": row.get("pre", ""),
                        "post": row.get("post", ""),
                        "source": f"erisk2025_{row.get('type', 'boundary')}",
                        "source_detail": {
                            "bdi_query": row.get("bdi_query", ""),
                            "bdi_name": row.get("bdi_name", ""),
                            "type": row.get("type", ""),
                        },
                    })

    # eRisk 2023 boundary pool
    erisk2023_path = CANDIDATES_DIR / "erisk2023_boundary_pool.tsv"
    if erisk2023_path.exists():
        with open(erisk2023_path, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                if int(row.get("symptom_id", -1)) == symptom_id:
                    pools.setdefault("erisk2023_boundary", []).append({
                        "docno": row.get("docid", ""),
                        "text": row.get("text", ""),
                        "pre": row.get("pre", ""),
                        "post": row.get("post", ""),
                        "source": f"erisk2023_{row.get('type', 'boundary')}",
                        "source_detail": {
                            "bdi_query": row.get("bdi_query", ""),
                            "bdi_name": row.get("bdi_name", ""),
                            "type": row.get("type", ""),
                        },
                    })

    return pools


def select_candidates_per_score(
    candidates: list[dict],
    score0_pool: list[dict],
    external_pools: dict[str, list[dict]],
    global_used_texts: set[str] | None = None,
    num_per_score: int = 2,
) -> dict[int, list[dict]]:
    """Select candidates for each score level using source priority.

    Returns dict mapping score -> list of candidates (up to num_per_score).
    Ensures no duplicate sentences across score levels AND across symptoms
    (via global_used_texts).
    """
    selected: dict[int, list[dict]] = {s: [] for s in range(4)}
    used_texts: set[str] = set(global_used_texts or set())

    def _pick_unused(pool: list[dict]) -> dict | None:
        for c in pool:
            if c.get("text", "") not in used_texts:
                return c
        return None

    def _mark_used(candidate: dict) -> None:
        used_texts.add(candidate.get("text", ""))

    # Retrieval candidates from per-symptom TSV
    retrieval = [c for c in candidates if c.get("source", "") == "retrieval"]

    # --- Score 0: from score0 pool (random, unrelated sentences) ---
    for _ in range(num_per_score):
        if score0_pool:
            pick = _pick_unused(score0_pool)
            if pick:
                pick = {**pick, "source": "score0_random"}
                selected[0].append(pick)
                _mark_used(pick)
                continue
        # fallback: low-ranked retrieval
        if retrieval:
            pick = _pick_unused(list(reversed(retrieval)))
            if pick:
                selected[0].append(pick)
                _mark_used(pick)

    # --- Score 1: prefer bdisen > redsm5 > erisk boundary > retrieval ---
    for _ in range(num_per_score):
        picked = False
        for source_cat in ["bdisen", "redsm5", "erisk2025_boundary", "erisk2023_boundary"]:
            if source_cat in external_pools:
                pick = _pick_unused(external_pools[source_cat])
                if pick:
                    selected[1].append(pick)
                    _mark_used(pick)
                    picked = True
                    break
        if not picked and retrieval:
            mid = retrieval[len(retrieval) // 2:]
            pick = _pick_unused(mid)
            if pick:
                selected[1].append(pick)
                _mark_used(pick)

    # --- Score 3: top-ranked retrieval (most relevant) — pick before score 2 ---
    for _ in range(num_per_score):
        if retrieval:
            pick = _pick_unused(retrieval)
            if pick:
                selected[3].append(pick)
                _mark_used(pick)

    # --- Score 2: mid-ranked retrieval (moderate relevance) ---
    for _ in range(num_per_score):
        if retrieval:
            mid_range = retrieval[3:20] if len(retrieval) > 3 else retrieval
            pick = _pick_unused(mid_range)
            if pick:
                selected[2].append(pick)
                _mark_used(pick)

    # Remove empty score levels
    return {s: picks for s, picks in selected.items() if picks}


ANNOTATION_PROMPT = """\
You are a clinical psychologist specializing in adult ADHD assessment. \
You need to annotate a sentence for its relevance to a specific ADHD symptom.

ASRS SYMPTOM #{item_number}: "{symptom_text}"
FACTOR: {factor}

CLINICAL CONTEXT:
{clinical_definition}

The sentence should be annotated at SCORE LEVEL {target_score}.

SCORING CRITERIA:
- SCORE 0: Sentence does not address this symptom at all.
- SCORE 1: Sentence touches on the symptom area but is vague, indirect, or ambiguous. \
A connection exists but is not explicit. Common for depression confounders that RESEMBLE \
this ADHD symptom.
- SCORE 2: Sentence clearly addresses this symptom AND conveys the writer's own experience, \
but lacks specific situational detail.
- SCORE 3: Sentence explicitly describes the writer's own experience of this exact symptom \
with concrete detail, specific situations, or clear behavioral examples.

SENTENCE TO ANNOTATE:
BEFORE: {pre}
>>> TARGET: {text} <<<
AFTER: {post}

SOURCE: {source}
{source_detail}

Produce the annotation in EXACTLY this JSON format:
{{
  "symptom_match": "YES" or "PARTIAL" or "NO",
  "self_reference": "DIRECT" or "INDIRECT" or "NONE",
  "detail_level": "HIGH" or "MEDIUM" or "LOW" or "NONE",
  "confounders": "NONE" or a brief description,
  "score": {target_score},
  "confidence": 1-5,
  "reasoning": "1-2 sentences explaining why this score"
}}

IMPORTANT RULES:
- For score 0: SYMPTOM_MATCH must be NO
- For score 1: SYMPTOM_MATCH should be PARTIAL; if from RedSM5/BDI-Sen, \
note depression as confounder
- For score 2+: SELF_REFERENCE must be DIRECT
- For score 3: DETAIL_LEVEL must be HIGH
- Be conservative: if this sentence doesn't fit score {target_score} well, \
note that in your reasoning but still assign the requested score

Respond with ONLY the JSON object, no other text."""


def annotate_with_llm(
    symptom: dict,
    candidate: dict,
    target_score: int,
) -> dict:
    """Call GPT-4o-mini to annotate a candidate at a target score level."""
    source_detail = ""
    source = candidate.get("source", "unknown")
    if "redsm5" in source.lower():
        source_detail = (
            "This sentence comes from RedSM5 (depression dataset) — "
            "it describes a depression symptom that resembles this ADHD symptom. "
            "The confounder is depression."
        )
    elif "bdisen" in source.lower():
        source_detail = (
            "This sentence comes from BDI-Sen (graded depression severity) — "
            "it has mild depression relevance, making it an ambiguous confounder for ADHD."
        )
    elif "boundary" in source.lower():
        source_detail = (
            "This sentence is a borderline case from eRisk — "
            "human annotators disagreed on its relevance to the mapped BDI-II symptom."
        )

    prompt = ANNOTATION_PROMPT.format(
        item_number=symptom["item_number"],
        symptom_text=symptom["text"],
        factor=symptom["factor"],
        clinical_definition=symptom.get("layers", {}).get("L1_clinical", "N/A"),
        pre=candidate.get("pre", "") or "",
        text=candidate.get("text", ""),
        post=candidate.get("post", "") or "",
        source=source,
        source_detail=source_detail,
        target_score=target_score,
    )

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}",
    }
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 300,
    }

    for attempt in range(3):
        try:
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()

            # Parse JSON from response
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            annotation = json.loads(content)
            return annotation

        except (json.JSONDecodeError, KeyError, requests.HTTPError) as e:
            logger.warning("Attempt %d failed: %s", attempt + 1, e)
            time.sleep(2 ** attempt)

    # Fallback: return empty annotation
    logger.error("Failed to annotate after 3 attempts")
    return {
        "symptom_match": "PARTIAL" if target_score > 0 else "NO",
        "self_reference": "DIRECT" if target_score >= 2 else "NONE",
        "detail_level": ["NONE", "LOW", "MEDIUM", "HIGH"][target_score],
        "confounders": "NONE",
        "score": target_score,
        "confidence": 3,
        "reasoning": "Auto-generated fallback annotation.",
    }


def generate_examples_for_symptom(
    symptom: dict,
    score0_pool: list[dict],
    dry_run: bool = False,
    global_used_texts: set[str] | None = None,
) -> dict:
    """Generate the full example set for one symptom."""
    symptom_id = symptom["item_number"]
    candidates = load_candidates(symptom_id)

    if not candidates:
        logger.warning("No candidates for symptom %d, skipping", symptom_id)
        return {}

    external_pools = load_external_pools(symptom_id)
    if external_pools:
        logger.info(
            "  External pools: %s",
            {k: len(v) for k, v in external_pools.items()},
        )
    selected = select_candidates_per_score(
        candidates, score0_pool, external_pools, global_used_texts,
    )

    # Register all selected texts in global tracker
    if global_used_texts is not None:
        for picks in selected.values():
            for c in picks:
                global_used_texts.add(c.get("text", ""))

    examples = []
    for score_level in range(4):
        if score_level not in selected:
            logger.warning(
                "  No candidate for score %d, symptom %d", score_level, symptom_id,
            )
            continue

        for idx, candidate in enumerate(selected[score_level]):
            logger.info(
                "  Score %d [%d]: docno=%s, source=%s, text=%.60s...",
                score_level, idx, candidate.get("docno", "?"),
                candidate.get("source", "?"), candidate.get("text", "")[:60],
            )

            if dry_run:
                annotation = {
                    "symptom_match": "PARTIAL" if score_level > 0 else "NO",
                    "self_reference": "DIRECT" if score_level >= 2 else "NONE",
                    "detail_level": ["NONE", "LOW", "MEDIUM", "HIGH"][score_level],
                    "confounders": "NONE",
                    "score": score_level,
                    "confidence": 3,
                    "reasoning": f"DRY RUN — would annotate at score {score_level}",
                }
            else:
                annotation = annotate_with_llm(symptom, candidate, score_level)

            source = candidate.get("source", "unknown")
            source_detail = None
            if "redsm5" in source.lower() or "bdisen" in source.lower() or "boundary" in source.lower():
                source_detail = {"raw_source": source}

            examples.append({
                "score": score_level,
                "docno": candidate.get("docno", ""),
                "source": source,
                "source_detail": source_detail,
                "synthetic": False,
                "pre": candidate.get("pre", "") or "",
                "text": candidate.get("text", ""),
                "post": candidate.get("post", "") or "",
                "annotation": annotation,
            })

    result = {
        "symptom_id": symptom_id,
        "symptom_text": symptom["text"],
        "symptom_factor": symptom["factor"],
        "annotator": "GPT-4o-mini (auto-generated, requires human review)",
        "annotation_date": "2026-03-07",
        "examples": examples,
    }

    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--symptoms", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    symptoms = load_symptoms()
    score0_pool = load_score0_pool()

    if args.symptoms:
        symptom_ids = [int(s.strip()) for s in args.symptoms.split(",")]
    else:
        symptom_ids = sorted(symptoms.keys())

    logger.info(
        "Generating examples for %d symptoms (dry_run=%s)",
        len(symptom_ids), args.dry_run,
    )

    ANNOTATIONS_DIR.mkdir(exist_ok=True)
    all_results = {}
    score0_idx = 0  # round-robin through score-0 pool
    global_used_texts: set[str] = set()  # prevent duplicate sentences across symptoms

    for sid in symptom_ids:
        if sid not in symptoms:
            logger.warning("Symptom %d not in config, skipping", sid)
            continue

        # Rotate score-0 pool so each symptom gets a different sentence
        rotated_pool = score0_pool[score0_idx:] + score0_pool[:score0_idx]
        score0_idx = (score0_idx + 1) % max(len(score0_pool), 1)

        logger.info("=== Symptom %d: %s ===", sid, symptoms[sid]["text"][:60])
        result = generate_examples_for_symptom(
            symptoms[sid], rotated_pool, dry_run=args.dry_run,
            global_used_texts=global_used_texts,
        )

        if result:
            out_path = ANNOTATIONS_DIR / f"symptom_{sid:02d}_examples.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.info("  Saved to %s", out_path)
            all_results[sid] = result

    # Summary
    logger.info("")
    logger.info("=== Generation Summary ===")
    for sid, result in sorted(all_results.items()):
        scores_found = [e["score"] for e in result["examples"]]
        sources = [e["source"] for e in result["examples"]]
        logger.info(
            "  Symptom %2d: scores=%s, sources=%s",
            sid, scores_found, sources,
        )

    missing = [sid for sid in symptom_ids if sid not in all_results]
    if missing:
        logger.warning("Missing examples for symptoms: %s", missing)

    logger.info("Done! Review the annotations in %s", ANNOTATIONS_DIR)


if __name__ == "__main__":
    main()
