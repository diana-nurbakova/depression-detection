"""Qualitative analysis of Task 3 (ADHD symptom ranking) results.

Analyses:
  1. Cross-run agreement (Jaccard overlap @ K, Kendall's tau)
  2. LLM-as-judge relevance audit (precision@K, error categorization)
  3. Per-factor/subcluster breakdown
  4. Diversity analysis (unique relevant sentences per run)
  5. Failure mode comparison (divergence detection + side-by-side)

Usage:
    python scripts/analyze_qualitative.py --qualitative-dir output/qualitative
    python scripts/analyze_qualitative.py --analysis agreement      # run only one analysis
    python scripts/analyze_qualitative.py --analysis relevance --llm-judge  # with LLM judge
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ─── Symptom metadata ───────────────────────────────────────────────────────

FACTORS = {
    "Inattention": [1, 2, 3, 4, 7, 8, 9, 10, 11],
    "Motor_HI": [5, 6, 12, 13, 14],
    "Verbal_HI": [15, 16, 17, 18],
}

SUBCLUSTERS = {
    "Organization_Planning": [1, 2],
    "Memory_Avoidance": [3, 4],
    "Sustained_Attention": [7, 8, 9, 10, 11],
    "Fidgeting_Restlessness": [5, 13],
    "Internal_Drive": [6, 12, 14],
    "Output_Control": [15, 16],
    "Turn_Taking": [17, 18],
}

RUNS = [
    "INSALyon_HiPerT_full",
    "INSALyon_LLM_cascade",
    "INSALyon_Ensemble",
    "INSALyon_DepTransfer",
    "INSALyon_BiEnc_baseline",
]

SHORT = {
    "INSALyon_HiPerT_full": "HiPerT",
    "INSALyon_LLM_cascade": "LLM",
    "INSALyon_Ensemble": "Ens",
    "INSALyon_DepTransfer": "DepTr",
    "INSALyon_BiEnc_baseline": "BiEnc",
}


# ─── Data loading ────────────────────────────────────────────────────────────

def load_all(qualitative_dir: Path) -> dict[str, dict[int, dict]]:
    """Load all runs → {run_name: {symptom_id: {meta + sentences}}}."""
    data = {}
    for run in RUNS:
        run_dir = qualitative_dir / run
        if not run_dir.exists():
            print(f"  WARNING: {run_dir} not found, skipping")
            continue
        data[run] = {}
        for sid in range(1, 19):
            path = run_dir / f"symptom_{sid}.json"
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    data[run][sid] = json.load(f)
    return data


def get_sentence_ids(run_data: dict, symptom_id: int, top_k: int) -> list[str]:
    """Get top-K sentence IDs for a run/symptom."""
    sentences = run_data.get(symptom_id, {}).get("sentences", [])
    return [s["sentence_id"] for s in sentences[:top_k]]


def get_sentence_map(run_data: dict, symptom_id: int) -> dict[str, dict]:
    """Get {sentence_id: {rank, score, text}} for a run/symptom."""
    sentences = run_data.get(symptom_id, {}).get("sentences", [])
    return {s["sentence_id"]: s for s in sentences}


# ─── 1. Cross-run agreement ─────────────────────────────────────────────────

def jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    return len(set_a & set_b) / len(union) if union else 0.0


def analyze_agreement(data: dict, output_dir: Path):
    """Compute Jaccard@K and Kendall's tau between all run pairs."""
    print("\n" + "=" * 70)
    print("ANALYSIS 1: Cross-Run Agreement")
    print("=" * 70)

    ks = [5, 10, 20]
    run_names = [r for r in RUNS if r in data]
    pairs = list(combinations(run_names, 2))

    # ── Jaccard@K ──
    results = {}
    for k in ks:
        results[k] = {}
        for r1, r2 in pairs:
            jaccards = []
            for sid in range(1, 19):
                s1 = set(get_sentence_ids(data[r1], sid, k))
                s2 = set(get_sentence_ids(data[r2], sid, k))
                jaccards.append(jaccard(s1, s2))
            pair_key = f"{SHORT[r1]}-{SHORT[r2]}"
            results[k][pair_key] = {
                "mean": np.mean(jaccards),
                "std": np.std(jaccards),
                "per_symptom": jaccards,
            }

    print("\n── Jaccard Overlap @ K (mean ± std across 18 symptoms) ──\n")
    header = f"{'Pair':<14}" + "".join(f"{'@'+str(k):>14}" for k in ks)
    print(header)
    print("-" * len(header))
    for r1, r2 in pairs:
        pair_key = f"{SHORT[r1]}-{SHORT[r2]}"
        row = f"{pair_key:<14}"
        for k in ks:
            m = results[k][pair_key]["mean"]
            s = results[k][pair_key]["std"]
            row += f"{m:.3f} ± {s:.3f}  "
        print(row)

    # ── Kendall's tau on shared sentences ──
    print("\n── Kendall's Tau (rank correlation on shared sentences) ──\n")
    tau_results = {}
    header = f"{'Pair':<14}{'Mean Tau':>10}{'Mean p':>10}{'Shared %':>10}"
    print(header)
    print("-" * len(header))

    for r1, r2 in pairs:
        taus, pvals, shared_pcts = [], [], []
        for sid in range(1, 19):
            map1 = get_sentence_map(data[r1], sid)
            map2 = get_sentence_map(data[r2], sid)
            shared = set(map1.keys()) & set(map2.keys())
            total = set(map1.keys()) | set(map2.keys())
            shared_pcts.append(len(shared) / len(total) if total else 0)

            if len(shared) < 3:
                continue
            shared_sorted = sorted(shared)
            ranks1 = [map1[s]["rank"] for s in shared_sorted]
            ranks2 = [map2[s]["rank"] for s in shared_sorted]
            tau, p = kendalltau(ranks1, ranks2)
            if not np.isnan(tau):
                taus.append(tau)
                pvals.append(p)

        pair_key = f"{SHORT[r1]}-{SHORT[r2]}"
        tau_results[pair_key] = {
            "mean_tau": np.mean(taus) if taus else float("nan"),
            "mean_p": np.mean(pvals) if pvals else float("nan"),
            "shared_pct": np.mean(shared_pcts),
        }
        print(
            f"{pair_key:<14}"
            f"{tau_results[pair_key]['mean_tau']:>10.3f}"
            f"{tau_results[pair_key]['mean_p']:>10.4f}"
            f"{tau_results[pair_key]['shared_pct']:>9.1%}"
        )

    # ── Per-symptom heatmap data ──
    heatmap = {}
    for sid in range(1, 19):
        heatmap[sid] = {}
        for r1, r2 in pairs:
            pair_key = f"{SHORT[r1]}-{SHORT[r2]}"
            s1 = set(get_sentence_ids(data[r1], sid, 10))
            s2 = set(get_sentence_ids(data[r2], sid, 10))
            heatmap[sid][pair_key] = jaccard(s1, s2)

    # Save detailed results
    out = output_dir / "agreement.json"
    save = {
        "jaccard": {str(k): {p: {"mean": r["mean"], "std": r["std"], "per_symptom": r["per_symptom"]}
                              for p, r in v.items()} for k, v in results.items()},
        "kendall_tau": tau_results,
        "jaccard_at_10_heatmap": heatmap,
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(save, f, indent=2, default=str)
    print(f"\n  Saved to {out}")


# ─── 2. LLM-as-judge relevance audit ────────────────────────────────────────

# Relevance labels:
#   3 = clearly relevant (directly describes the symptom)
#   2 = partially relevant (related ADHD experience, indirect match)
#   1 = wrong symptom (ADHD-related but maps to a different ASRS item)
#   0 = off-topic (not ADHD-related or too vague)

RELEVANCE_PROMPT = """You are an expert in ADHD clinical assessment using the ASRS-v1.1 screening tool.

Given an ADHD symptom and a candidate sentence from a social media post, rate the sentence's relevance to the symptom.

## Symptom
Item {item_number}: "{symptom_text}"
Factor: {factor} | Subcluster: {subcluster}

## Sentence
"{sentence_text}"

## Rating Scale
3 = CLEARLY RELEVANT: The sentence directly describes this specific symptom or a concrete personal experience of it.
2 = PARTIALLY RELEVANT: The sentence is related to this symptom but only indirectly (e.g., mentions a related concept without describing the specific experience).
1 = WRONG SYMPTOM: The sentence describes an ADHD experience but maps better to a different ASRS item.
0 = OFF-TOPIC: The sentence is not about ADHD symptoms or is too vague to be useful.

## Response
Reply with ONLY a JSON object: {{"score": <0-3>, "category": "<off-topic|wrong-symptom|partial|relevant>", "brief_reason": "<1 sentence>"}}"""


_llm_backend = None  # set in main()


def _init_llm_backend(backend: str):
    """Initialize the LLM backend (ollama or huggingface)."""
    global _llm_backend
    _llm_backend = backend

    if backend == "huggingface":
        from huggingface_hub import InferenceClient
        import os
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
        token = os.environ.get("HF_TOKEN", "")
        _init_llm_backend._hf_client = InferenceClient(
            model="meta-llama/Llama-3.1-8B-Instruct",
            token=token,
            timeout=120,
        )
    elif backend == "ollama":
        import os
        import requests
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
        _init_llm_backend._ollama_url = os.environ.get(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )
        _init_llm_backend._ollama_key = os.environ.get("OLLAMA_API_KEY", "")
        # Verify Ollama is reachable
        try:
            headers = {}
            if _init_llm_backend._ollama_key:
                headers["Authorization"] = f"Bearer {_init_llm_backend._ollama_key}"
            requests.get(
                f"{_init_llm_backend._ollama_url}/api/tags",
                headers=headers, timeout=10,
            )
        except Exception as e:
            raise ConnectionError(f"Cannot reach Ollama at {_init_llm_backend._ollama_url}: {e}")


def judge_with_llm(symptom_meta: dict, sentence_text: str) -> dict:
    """Judge sentence relevance using configured LLM backend.

    Returns {score, category, brief_reason}.
    """
    prompt = RELEVANCE_PROMPT.format(
        item_number=symptom_meta["symptom_id"],
        symptom_text=symptom_meta["symptom_text"],
        factor=symptom_meta["factor"],
        subcluster=symptom_meta["subcluster"],
        sentence_text=sentence_text,
    )

    messages = [
        {"role": "system", "content": "You are an expert ADHD clinical assessor. Respond with JSON only."},
        {"role": "user", "content": prompt},
    ]

    try:
        if _llm_backend == "ollama":
            text = _judge_ollama(messages)
        else:
            text = _judge_huggingface(messages)

        # Extract JSON from response
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            parsed = json.loads(text[start:end])
            if "score" in parsed and "category" in parsed:
                parsed["score"] = int(parsed["score"])
                return parsed
    except Exception as e:
        return {"score": -1, "category": "error", "brief_reason": str(e)[:100]}

    return {"score": -1, "category": "error", "brief_reason": "failed to parse"}


def _judge_ollama(messages: list[dict]) -> str:
    """Call Ollama (local or remote pagoda) for chat completion."""
    import requests

    base_url = _init_llm_backend._ollama_url
    api_key = _init_llm_backend._ollama_key

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    resp = requests.post(
        f"{base_url}/api/chat",
        headers=headers,
        json={
            "model": "llama3.3:70b",
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 150},
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


def _judge_huggingface(messages: list[dict]) -> str:
    """Call HuggingFace Inference API for chat completion."""
    import time as _time

    client = _init_llm_backend._hf_client
    response = client.chat_completion(
        messages=messages,
        max_tokens=150,
        temperature=0.1,
    )
    # Rate-limit for HF free tier
    _time.sleep(1.0)
    return response.choices[0].message.content.strip()


def analyze_relevance(data: dict, output_dir: Path, use_llm: bool = False, top_k: int = 10):
    """LLM-as-judge relevance audit or heuristic-based proxy."""
    print("\n" + "=" * 70)
    print("ANALYSIS 2: Relevance Audit")
    print("=" * 70)

    run_names = [r for r in RUNS if r in data]
    results = {}

    # Check for cached results to allow resuming
    cache_path = output_dir / "relevance_cache.json"
    cache = {}
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
        print(f"  Loaded {sum(len(v) for v in cache.values())} cached judgments")

    total_calls = len(run_names) * 18 * top_k
    done_calls = 0

    for run in run_names:
        run_results = {}
        run_cache = cache.get(run, {})

        for sid in range(1, 19):
            sym = data[run].get(sid, {})
            sentences = sym.get("sentences", [])[:top_k]
            meta = {
                "symptom_id": sym.get("symptom_id", sid),
                "symptom_text": sym.get("symptom_text", ""),
                "factor": sym.get("factor", ""),
                "subcluster": sym.get("subcluster", ""),
            }

            judgments = []
            for sent in sentences:
                done_calls += 1
                cache_key = f"{sid}_{sent['sentence_id']}"

                if use_llm and cache_key in run_cache:
                    judgment = run_cache[cache_key]
                elif use_llm:
                    judgment = judge_with_llm(meta, sent["text"])
                    # Cache the result
                    if run not in cache:
                        cache[run] = {}
                    cache[run][cache_key] = judgment
                    # Save cache periodically (every 50 calls)
                    if done_calls % 50 == 0:
                        with open(cache_path, "w", encoding="utf-8") as f:
                            json.dump(cache, f, indent=2, default=str)
                else:
                    judgment = heuristic_relevance(meta, sent["text"])

                judgments.append({
                    "rank": sent["rank"],
                    "sentence_id": sent["sentence_id"],
                    "text": sent["text"][:100],
                    **judgment,
                })

                if use_llm and done_calls % 25 == 0:
                    print(f"  Progress: {done_calls}/{total_calls} ({done_calls/total_calls:.0%})")

            scores = [j["score"] for j in judgments if j["score"] >= 0]
            categories = [j["category"] for j in judgments if j["category"] != "error"]

            run_results[sid] = {
                "mean_relevance": np.mean(scores) if scores else 0,
                "precision_at_k": sum(1 for s in scores if s >= 2) / len(scores) if scores else 0,
                "category_dist": {
                    cat: categories.count(cat) / len(categories) if categories else 0
                    for cat in ["relevant", "partial", "wrong-symptom", "off-topic"]
                },
                "judgments": judgments,
            }

        results[run] = run_results

    # Save final cache
    if use_llm and cache:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, default=str)
        print(f"  Cache saved: {cache_path}")

    # Summary table
    method = "LLM" if use_llm else "Heuristic"
    print(f"\n── Precision@{top_k} by Run ({method} judge) ──\n")
    header = f"{'Run':<12}{'Mean P@K':>10}{'Relevance':>12}{'Off-topic%':>12}{'WrongSym%':>12}"
    print(header)
    print("-" * len(header))

    for run in run_names:
        p_at_k = np.mean([results[run][sid]["precision_at_k"] for sid in range(1, 19)])
        mean_rel = np.mean([results[run][sid]["mean_relevance"] for sid in range(1, 19)])
        off_topic = np.mean([results[run][sid]["category_dist"].get("off-topic", 0) for sid in range(1, 19)])
        wrong_sym = np.mean([results[run][sid]["category_dist"].get("wrong-symptom", 0) for sid in range(1, 19)])
        print(
            f"{SHORT[run]:<12}"
            f"{p_at_k:>10.3f}"
            f"{mean_rel:>12.2f}"
            f"{off_topic:>11.1%}"
            f"{wrong_sym:>11.1%}"
        )

    out = output_dir / "relevance.json"
    # Serialize for JSON
    serializable = {}
    for run, rr in results.items():
        serializable[run] = {}
        for sid, sr in rr.items():
            serializable[run][str(sid)] = {
                "mean_relevance": float(sr["mean_relevance"]),
                "precision_at_k": float(sr["precision_at_k"]),
                "category_dist": {k: float(v) for k, v in sr["category_dist"].items()},
                "judgments": sr["judgments"],
            }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\n  Saved to {out}")

    return results


def heuristic_relevance(meta: dict, text: str) -> dict:
    """Keyword-based heuristic proxy for relevance scoring."""
    import yaml

    text_lower = text.lower()

    # Load keywords from symptoms.yaml (cached)
    if not hasattr(heuristic_relevance, "_keywords_cache"):
        symptoms_path = PROJECT_ROOT / "config" / "symptoms.yaml"
        with open(symptoms_path, "r", encoding="utf-8") as f:
            syms = yaml.safe_load(f)
        heuristic_relevance._keywords_cache = {}
        heuristic_relevance._all_keywords = {}
        for s in syms["symptoms"]:
            sid = s["item_number"]
            heuristic_relevance._keywords_cache[sid] = [kw.lower() for kw in s.get("keywords", [])]
        # Build set of all keywords per factor
        for s in syms["symptoms"]:
            factor = s["factor"]
            if factor not in heuristic_relevance._all_keywords:
                heuristic_relevance._all_keywords[factor] = set()
            for kw in s.get("keywords", []):
                heuristic_relevance._all_keywords[factor].add(kw.lower())

    sid = meta["symptom_id"]
    target_kw = heuristic_relevance._keywords_cache.get(sid, [])

    # Count target keyword hits
    target_hits = sum(1 for kw in target_kw if kw in text_lower)

    # Check for other symptom keywords (wrong symptom detection)
    other_hits = 0
    for other_sid, other_kw in heuristic_relevance._keywords_cache.items():
        if other_sid == sid:
            continue
        other_hits += sum(1 for kw in other_kw if kw in text_lower and kw not in target_kw)

    # Scoring
    if target_hits >= 3:
        return {"score": 3, "category": "relevant", "brief_reason": f"{target_hits} keyword matches"}
    elif target_hits >= 1:
        if other_hits > target_hits:
            return {"score": 1, "category": "wrong-symptom", "brief_reason": f"More matches for other symptoms ({other_hits} vs {target_hits})"}
        return {"score": 2, "category": "partial", "brief_reason": f"{target_hits} keyword match(es)"}
    elif other_hits >= 1:
        return {"score": 1, "category": "wrong-symptom", "brief_reason": f"Matches other ADHD symptoms only"}
    else:
        # Check for general ADHD terms
        general = ["adhd", "add", "attention", "hyperactive", "impulsive", "executive function", "dopamine"]
        gen_hits = sum(1 for g in general if g in text_lower)
        if gen_hits >= 1:
            return {"score": 1, "category": "partial", "brief_reason": "General ADHD mention only"}
        return {"score": 0, "category": "off-topic", "brief_reason": "No keyword matches"}


# ─── 3. Per-factor/subcluster breakdown ─────────────────────────────────────

def analyze_per_factor(data: dict, relevance_results: dict | None, output_dir: Path):
    """Break down agreement and relevance by factor and subcluster."""
    print("\n" + "=" * 70)
    print("ANALYSIS 3: Per-Factor / Subcluster Breakdown")
    print("=" * 70)

    run_names = [r for r in RUNS if r in data]

    # ── Agreement by factor ──
    print("\n── Jaccard@10 by Factor ──\n")
    pairs = list(combinations(run_names, 2))

    header = f"{'Factor':<25}" + "".join(f"{SHORT[r1]}-{SHORT[r2]:>6}  " for r1, r2 in pairs)
    print(header)
    print("-" * len(header))

    factor_agreement = {}
    for factor, items in FACTORS.items():
        factor_agreement[factor] = {}
        row = f"{factor:<25}"
        for r1, r2 in pairs:
            jacs = []
            for sid in items:
                s1 = set(get_sentence_ids(data[r1], sid, 10))
                s2 = set(get_sentence_ids(data[r2], sid, 10))
                jacs.append(jaccard(s1, s2))
            pair_key = f"{SHORT[r1]}-{SHORT[r2]}"
            mean_j = np.mean(jacs)
            factor_agreement[factor][pair_key] = mean_j
            row += f"{mean_j:>8.3f}  "
        print(row)

    # ── Relevance by factor (if available) ──
    if relevance_results:
        print("\n── Precision@10 by Factor ──\n")
        header = f"{'Factor':<25}" + "".join(f"{SHORT[r]:>8}" for r in run_names)
        print(header)
        print("-" * len(header))

        for factor, items in FACTORS.items():
            row = f"{factor:<25}"
            for run in run_names:
                precs = [relevance_results[run][sid]["precision_at_k"] for sid in items]
                row += f"{np.mean(precs):>8.3f}"
            print(row)

        print("\n── Precision@10 by Subcluster ──\n")
        header = f"{'Subcluster':<25}" + "".join(f"{SHORT[r]:>8}" for r in run_names)
        print(header)
        print("-" * len(header))

        for subcluster, items in SUBCLUSTERS.items():
            row = f"{subcluster:<25}"
            for run in run_names:
                precs = [relevance_results[run][sid]["precision_at_k"] for sid in items]
                row += f"{np.mean(precs):>8.3f}"
            print(row)

    out = output_dir / "per_factor.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"factor_agreement": factor_agreement}, f, indent=2, default=str)
    print(f"\n  Saved to {out}")


# ─── 4. Diversity analysis ──────────────────────────────────────────────────

def analyze_diversity(data: dict, output_dir: Path, top_k: int = 10):
    """How many unique sentences does each run contribute to the pool?"""
    print("\n" + "=" * 70)
    print("ANALYSIS 4: Diversity Analysis")
    print("=" * 70)

    run_names = [r for r in RUNS if r in data]

    print(f"\n── Unique Contribution @ top-{top_k} (sentences only in this run) ──\n")
    header = f"{'Symptom':<12}" + "".join(f"{SHORT[r]:>8}" for r in run_names) + f"{'Union':>8}"
    print(header)
    print("-" * len(header))

    diversity = {}
    for sid in range(1, 19):
        all_sets = {run: set(get_sentence_ids(data[run], sid, top_k)) for run in run_names}
        union = set()
        for s in all_sets.values():
            union |= s

        unique_counts = {}
        for run in run_names:
            others = set()
            for other_run in run_names:
                if other_run != run:
                    others |= all_sets[other_run]
            unique_counts[run] = len(all_sets[run] - others)

        diversity[sid] = {SHORT[r]: unique_counts[r] for r in run_names}
        diversity[sid]["union"] = len(union)

        row = f"Item {sid:<6}"
        for run in run_names:
            row += f"{unique_counts[run]:>8}"
        row += f"{len(union):>8}"
        print(row)

    # Totals
    print("-" * len(header))
    row = f"{'Total':<12}"
    for run in run_names:
        total_unique = sum(diversity[sid][SHORT[run]] for sid in range(1, 19))
        row += f"{total_unique:>8}"
    total_union = sum(diversity[sid]["union"] for sid in range(1, 19))
    row += f"{total_union:>8}"
    print(row)

    # Ensemble value: does ensemble surface sentences not in its components?
    if "INSALyon_Ensemble" in data and "INSALyon_LLM_cascade" in data and "INSALyon_BiEnc_baseline" in data:
        print("\n── Ensemble Value (sentences in Ensemble but not in LLM or BiEnc) ──\n")
        ens_only = 0
        for sid in range(1, 19):
            ens = set(get_sentence_ids(data["INSALyon_Ensemble"], sid, top_k))
            llm = set(get_sentence_ids(data["INSALyon_LLM_cascade"], sid, top_k))
            bienc = set(get_sentence_ids(data["INSALyon_BiEnc_baseline"], sid, top_k))
            ens_exclusive = ens - llm - bienc
            ens_only += len(ens_exclusive)
            if ens_exclusive:
                print(f"  Item {sid}: {len(ens_exclusive)} exclusive sentences")
        print(f"  Total exclusive ensemble sentences: {ens_only} / {18 * top_k}")

    out = output_dir / "diversity.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(diversity, f, indent=2)
    print(f"\n  Saved to {out}")


# ─── 5. Failure mode comparison ─────────────────────────────────────────────

def analyze_failure_modes(data: dict, output_dir: Path, top_k: int = 10):
    """Find symptoms where runs diverge most; show side-by-side top sentences."""
    print("\n" + "=" * 70)
    print("ANALYSIS 5: Failure Mode Comparison")
    print("=" * 70)

    run_names = [r for r in RUNS if r in data]
    pairs = list(combinations(run_names, 2))

    # Find most divergent symptom for each pair
    print(f"\n── Most Divergent Symptoms (lowest Jaccard@{top_k} per pair) ──\n")

    divergences = []
    for r1, r2 in pairs:
        worst_sid, worst_jac = None, 1.0
        for sid in range(1, 19):
            s1 = set(get_sentence_ids(data[r1], sid, top_k))
            s2 = set(get_sentence_ids(data[r2], sid, top_k))
            j = jaccard(s1, s2)
            if j < worst_jac:
                worst_jac = j
                worst_sid = sid
        divergences.append((r1, r2, worst_sid, worst_jac))
        sym_text = data[r1].get(worst_sid, {}).get("symptom_text", "")[:60]
        print(f"  {SHORT[r1]}-{SHORT[r2]}: Item {worst_sid} (J={worst_jac:.3f}) — {sym_text}...")

    # Side-by-side for top 3 most divergent overall
    divergences.sort(key=lambda x: x[3])
    print(f"\n── Side-by-Side: Top-5 for Most Divergent Cases ──")

    side_by_side = []
    seen_symptoms = set()
    for r1, r2, sid, jac in divergences[:5]:
        if sid in seen_symptoms:
            continue
        seen_symptoms.add(sid)

        sym_text = data[r1].get(sid, {}).get("symptom_text", "")
        print(f"\n  Item {sid}: {sym_text}")
        print(f"  Jaccard@{top_k} between {SHORT[r1]} and {SHORT[r2]}: {jac:.3f}\n")

        comparison = {"symptom_id": sid, "symptom_text": sym_text, "runs": {}}

        for run in [r1, r2]:
            sents = data[run].get(sid, {}).get("sentences", [])[:5]
            comparison["runs"][SHORT[run]] = [
                {"rank": s["rank"], "text": s["text"][:80]} for s in sents
            ]
            print(f"  {SHORT[run]} top-5:")
            for s in sents:
                print(f"    #{s['rank']}: {s['text'][:80]}...")
            print()

        side_by_side.append(comparison)

    # ── Cross-run rank displacement for shared sentences ──
    print(f"\n── Rank Displacement (mean |rank_A - rank_B| for shared sentences) ──\n")
    header = f"{'Pair':<14}{'Mean Δrank':>12}{'Max Δrank':>12}{'Worst item':>12}"
    print(header)
    print("-" * len(header))

    displacement_data = {}
    for r1, r2 in pairs:
        pair_key = f"{SHORT[r1]}-{SHORT[r2]}"
        all_displacements = []
        per_symptom_mean = {}

        for sid in range(1, 19):
            map1 = get_sentence_map(data[r1], sid)
            map2 = get_sentence_map(data[r2], sid)
            shared = set(map1.keys()) & set(map2.keys())
            displacements = [abs(map1[s]["rank"] - map2[s]["rank"]) for s in shared]
            all_displacements.extend(displacements)
            per_symptom_mean[sid] = np.mean(displacements) if displacements else 0

        worst_sid = max(per_symptom_mean, key=per_symptom_mean.get) if per_symptom_mean else 0
        displacement_data[pair_key] = {
            "mean": np.mean(all_displacements) if all_displacements else 0,
            "max": max(all_displacements) if all_displacements else 0,
            "worst_symptom": worst_sid,
            "per_symptom": per_symptom_mean,
        }
        print(
            f"{pair_key:<14}"
            f"{displacement_data[pair_key]['mean']:>12.1f}"
            f"{displacement_data[pair_key]['max']:>12}"
            f"{'Item ' + str(worst_sid):>12}"
        )

    out = output_dir / "failure_modes.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "divergences": [
                {"run1": SHORT[r1], "run2": SHORT[r2], "symptom": sid, "jaccard": jac}
                for r1, r2, sid, jac in divergences
            ],
            "side_by_side": side_by_side,
            "rank_displacement": displacement_data,
        }, f, indent=2, default=str)
    print(f"\n  Saved to {out}")


# ─── Main ────────────────────────────────────────────────────────────────────

ANALYSES = {
    "agreement": "Cross-run agreement (Jaccard, Kendall)",
    "relevance": "Relevance audit (heuristic or LLM)",
    "factor": "Per-factor/subcluster breakdown",
    "diversity": "Diversity analysis",
    "failure": "Failure mode comparison",
}


def main():
    parser = argparse.ArgumentParser(
        description="Qualitative analysis of Task 3 ADHD symptom ranking results"
    )
    parser.add_argument(
        "--qualitative-dir", type=Path,
        default=PROJECT_ROOT / "output" / "qualitative",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "output" / "analysis",
    )
    parser.add_argument(
        "--analysis", type=str, default="all",
        choices=["all"] + list(ANALYSES.keys()),
        help="Which analysis to run",
    )
    parser.add_argument("--llm-judge", action="store_true", help="Use LLM for relevance judging")
    parser.add_argument(
        "--llm-backend", type=str, default="ollama",
        choices=["ollama", "huggingface"],
        help="LLM backend for judging (default: ollama)",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Top K for analysis")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.llm_judge:
        _init_llm_backend(args.llm_backend)
        print(f"  LLM backend: {args.llm_backend}")

    print(f"Loading data from {args.qualitative_dir}...")
    data = load_all(args.qualitative_dir)
    print(f"  Loaded {len(data)} runs: {', '.join(SHORT[r] for r in data)}")

    analyses_to_run = list(ANALYSES.keys()) if args.analysis == "all" else [args.analysis]

    relevance_results = None

    for analysis in analyses_to_run:
        if analysis == "agreement":
            analyze_agreement(data, args.output_dir)
        elif analysis == "relevance":
            relevance_results = analyze_relevance(
                data, args.output_dir, use_llm=args.llm_judge, top_k=args.top_k
            )
        elif analysis == "factor":
            # Run relevance first if needed
            if relevance_results is None and "relevance" not in analyses_to_run:
                relevance_results = analyze_relevance(data, args.output_dir, top_k=args.top_k)
            analyze_per_factor(data, relevance_results, args.output_dir)
        elif analysis == "diversity":
            analyze_diversity(data, args.output_dir, top_k=args.top_k)
        elif analysis == "failure":
            analyze_failure_modes(data, args.output_dir, top_k=args.top_k)

    print("\n" + "=" * 70)
    print(f"All analyses complete. Results saved to {args.output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
