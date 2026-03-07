"""DepreSym LLM Bias Profiling: Steps 1-4, 6 from spec.

Uses the DepreSym dataset (4 qrels: consensus, majority, ChatGPT, GPT-4)
to empirically characterize LLM scoring biases on depression symptoms
that overlap with ADHD. Outputs calibration data for the HiPerT pipeline.

Usage:
    uv run python scripts/depresym_bias_profiling.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# === Configuration ===

PROJECT_ROOT = Path(__file__).parent.parent
DEPRESYM_DIR = PROJECT_ROOT / "data" / "DepreSym_dataset"
OUTPUT_DIR = PROJECT_ROOT / "depresym_analysis"

# BDI-II symptoms that overlap with ASRS
OVERLAPPING_SYMPTOMS = {
    11: {"name": "Agitation", "maps_to_asrs": [5, 6, 12, 13]},
    13: {"name": "Indecisiveness", "maps_to_asrs": [1, 2]},
    15: {"name": "Loss of Energy", "maps_to_asrs": [4, 7]},
    16: {"name": "Sleep Changes", "maps_to_asrs": [7]},
    19: {"name": "Concentration Difficulty", "maps_to_asrs": [8, 9, 10, 11]},
}


# === Step 1: Load and Align Qrels ===

def load_qrels(filepath: Path, source_name: str) -> pd.DataFrame:
    """Load TREC-format qrels: SYMPTOM_ID 0 sentence-id RELEVANCE."""
    df = pd.read_csv(
        filepath, sep=r"\s+", header=None,
        names=["symptom_id", "_", "sentence_id", "relevance"],
        engine="python",
    )
    df = df.drop(columns=["_"])
    df = df.rename(columns={"relevance": source_name})
    return df


def load_and_align() -> pd.DataFrame:
    """Load all 4 qrels and merge into a unified matrix."""
    logger.info("Step 1: Loading and aligning qrels...")

    consensus = load_qrels(DEPRESYM_DIR / "qrels-consensus.txt", "human_consensus")
    majority = load_qrels(DEPRESYM_DIR / "qrels-majority-2.txt", "human_majority")
    chatgpt = load_qrels(DEPRESYM_DIR / "qrels-chatgpt.txt", "chatgpt")
    gpt4 = load_qrels(DEPRESYM_DIR / "qrels-gpt4.txt", "gpt4")

    unified = consensus.merge(majority, on=["symptom_id", "sentence_id"], how="outer")
    unified = unified.merge(chatgpt, on=["symptom_id", "sentence_id"], how="outer")
    unified = unified.merge(gpt4, on=["symptom_id", "sentence_id"], how="outer")
    unified = unified.fillna(-1).astype({"human_consensus": int, "human_majority": int,
                                          "chatgpt": int, "gpt4": int})

    logger.info(
        "Loaded %d (symptom, sentence) pairs, %d unique sentences, %d symptoms",
        len(unified), unified["sentence_id"].nunique(),
        unified["symptom_id"].nunique(),
    )
    return unified


# === Step 2: Agreement Metrics ===

def compute_agreement(df: pd.DataFrame, source_a: str, source_b: str) -> dict | None:
    """Compute agreement metrics between two sources."""
    mask = (df[source_a] >= 0) & (df[source_b] >= 0)
    subset = df[mask]
    if len(subset) == 0:
        return None

    a = subset[source_a].values
    b = subset[source_b].values

    kappa = cohen_kappa_score(a, b)
    cm = confusion_matrix(a, b, labels=[0, 1])

    return {
        "kappa": round(float(kappa), 4),
        "agreement_rate": round(float((a == b).mean()), 4),
        "n": int(len(subset)),
        "confusion_matrix": cm.tolist(),
    }


def compute_global_agreement(unified: pd.DataFrame) -> dict:
    """Step 2: Compute global agreement between all source pairs."""
    logger.info("Step 2: Computing global agreement...")

    pairs = [
        ("human_consensus", "human_majority"),
        ("human_consensus", "gpt4"),
        ("human_consensus", "chatgpt"),
        ("human_majority", "gpt4"),
        ("human_majority", "chatgpt"),
        ("gpt4", "chatgpt"),
    ]

    results = {}
    for sa, sb in pairs:
        key = f"{sa}_vs_{sb}"
        agreement = compute_agreement(unified, sa, sb)
        results[key] = agreement
        if agreement:
            logger.info(
                "  %s: agreement=%.3f, κ=%.3f (n=%d)",
                key, agreement["agreement_rate"], agreement["kappa"], agreement["n"],
            )

    return results


def compute_per_symptom_agreement(unified: pd.DataFrame) -> dict:
    """Compute per-symptom agreement for all source pairs."""
    logger.info("Step 2b: Per-symptom agreement...")

    sources = ["human_consensus", "human_majority", "chatgpt", "gpt4"]
    results = {}

    for symptom_id in sorted(unified["symptom_id"].unique()):
        symptom_df = unified[unified["symptom_id"] == symptom_id]
        results[int(symptom_id)] = {}
        for i, sa in enumerate(sources):
            for sb in sources[i + 1:]:
                key = f"{sa}_vs_{sb}"
                results[int(symptom_id)][key] = compute_agreement(symptom_df, sa, sb)

    return results


# === Step 3: LLM Overestimation Profile ===

def overestimation_profile(
    df: pd.DataFrame, llm_source: str, human_source: str = "human_consensus",
) -> dict:
    """Compute overestimation metrics for an LLM vs human ground truth."""
    mask = (df[llm_source] >= 0) & (df[human_source] >= 0)
    subset = df[mask]
    if len(subset) == 0:
        return {}

    human_pos = int((subset[human_source] == 1).sum())
    human_neg = int((subset[human_source] == 0).sum())
    llm_pos = int((subset[llm_source] == 1).sum())

    false_pos = int(((subset[llm_source] == 1) & (subset[human_source] == 0)).sum())
    false_neg = int(((subset[llm_source] == 0) & (subset[human_source] == 1)).sum())

    return {
        "n": int(len(subset)),
        "human_positive_rate": round(human_pos / len(subset), 4),
        "llm_positive_rate": round(llm_pos / len(subset), 4),
        "overestimation_ratio": round(llm_pos / max(human_pos, 1), 4),
        "false_positive_rate": round(false_pos / max(human_neg, 1), 4),
        "false_negative_rate": round(false_neg / max(human_pos, 1), 4),
        "net_bias": round((llm_pos - human_pos) / len(subset), 4),
        "false_positives": false_pos,
        "false_negatives": false_neg,
    }


def compute_overestimation(unified: pd.DataFrame) -> dict:
    """Step 3: Per-symptom overestimation profile for GPT-4 and ChatGPT."""
    logger.info("Step 3: Computing LLM overestimation profiles...")

    results = {}
    for symptom_id in sorted(unified["symptom_id"].unique()):
        symptom_df = unified[unified["symptom_id"] == symptom_id]
        results[int(symptom_id)] = {
            "gpt4": overestimation_profile(symptom_df, "gpt4"),
            "chatgpt": overestimation_profile(symptom_df, "chatgpt"),
        }

    # Print focused report on overlapping symptoms
    logger.info("")
    logger.info("=== LLM OVERESTIMATION ON ADHD-OVERLAPPING SYMPTOMS ===")
    for sid, info in OVERLAPPING_SYMPTOMS.items():
        if sid not in results:
            continue
        profile = results[sid]
        logger.info("Symptom %d (%s) → ASRS %s:", sid, info["name"], info["maps_to_asrs"])
        for llm in ["gpt4", "chatgpt"]:
            p = profile[llm]
            if p:
                logger.info(
                    "  %s: overest_ratio=%.2f, FPR=%.3f, FNR=%.3f, net_bias=%+.3f (FP=%d, FN=%d)",
                    llm, p["overestimation_ratio"], p["false_positive_rate"],
                    p["false_negative_rate"], p["net_bias"],
                    p["false_positives"], p["false_negatives"],
                )

    return results


# === Step 4: False Positive Extraction ===

def extract_false_positives(unified: pd.DataFrame) -> dict:
    """Step 4: Extract GPT-4 false positive sentence IDs for overlapping symptoms."""
    logger.info("Step 4: Extracting false positives for overlapping symptoms...")

    fp_data = {}
    for sid, info in OVERLAPPING_SYMPTOMS.items():
        # GPT-4 false positives
        mask_gpt4 = (
            (unified["symptom_id"] == sid)
            & (unified["gpt4"] == 1)
            & (unified["human_consensus"] == 0)
        )
        gpt4_fps = unified[mask_gpt4]["sentence_id"].tolist()

        # ChatGPT false positives
        mask_chatgpt = (
            (unified["symptom_id"] == sid)
            & (unified["chatgpt"] == 1)
            & (unified["human_consensus"] == 0)
        )
        chatgpt_fps = unified[mask_chatgpt]["sentence_id"].tolist()

        # Both LLMs agree on false positive (shared blind spot)
        mask_both = mask_gpt4 & mask_chatgpt
        shared_fps = unified[mask_both]["sentence_id"].tolist()

        fp_data[int(sid)] = {
            "symptom_name": info["name"],
            "maps_to_asrs": info["maps_to_asrs"],
            "gpt4_false_positives": gpt4_fps,
            "gpt4_fp_count": len(gpt4_fps),
            "chatgpt_false_positives": chatgpt_fps,
            "chatgpt_fp_count": len(chatgpt_fps),
            "shared_false_positives": shared_fps,
            "shared_fp_count": len(shared_fps),
        }

        logger.info(
            "  Symptom %d (%s): GPT-4 FP=%d, ChatGPT FP=%d, shared=%d",
            sid, info["name"], len(gpt4_fps), len(chatgpt_fps), len(shared_fps),
        )

    return fp_data


# === Step 6: ChatGPT vs GPT-4 Correction Rate ===

def compute_correction_rates(unified: pd.DataFrame) -> dict:
    """Step 6: Does GPT-4 correct ChatGPT's errors?"""
    logger.info("Step 6: Computing ChatGPT → GPT-4 correction rates...")

    results = {}
    for sid, info in OVERLAPPING_SYMPTOMS.items():
        symptom_df = unified[unified["symptom_id"] == sid]
        mask = (symptom_df["chatgpt"] >= 0) & (symptom_df["gpt4"] >= 0) & (symptom_df["human_consensus"] >= 0)
        subset = symptom_df[mask]

        chatgpt_wrong = subset["chatgpt"] != subset["human_consensus"]
        chatgpt_errors = int(chatgpt_wrong.sum())

        gpt4_corrects = int(
            (chatgpt_wrong & (subset["gpt4"] == subset["human_consensus"])).sum()
        )

        correction_rate = gpt4_corrects / max(chatgpt_errors, 1)

        # Also: GPT-4 errors that ChatGPT gets right (reverse direction)
        gpt4_wrong = subset["gpt4"] != subset["human_consensus"]
        gpt4_errors = int(gpt4_wrong.sum())
        chatgpt_corrects_gpt4 = int(
            (gpt4_wrong & (subset["chatgpt"] == subset["human_consensus"])).sum()
        )

        verdict = (
            "Escalation well-motivated"
            if correction_rate >= 0.6
            else "Escalation has moderate value"
            if correction_rate >= 0.4
            else "Escalation has limited value — invest in prompt improvement"
        )

        results[int(sid)] = {
            "symptom_name": info["name"],
            "chatgpt_errors": chatgpt_errors,
            "gpt4_corrects": gpt4_corrects,
            "correction_rate": round(correction_rate, 4),
            "gpt4_errors": gpt4_errors,
            "chatgpt_corrects_gpt4": chatgpt_corrects_gpt4,
            "reverse_correction_rate": round(chatgpt_corrects_gpt4 / max(gpt4_errors, 1), 4),
            "verdict": verdict,
        }

        logger.info(
            "  Symptom %d (%s): ChatGPT errors=%d, GPT-4 corrects=%d (%.1f%%) — %s",
            sid, info["name"], chatgpt_errors, gpt4_corrects,
            correction_rate * 100, verdict,
        )

    return results


# === Integration: Derive Symptom Weight Adjustments ===

def derive_weight_adjustments(overestimation: dict) -> dict:
    """Derive symptom_weight adjustments from overestimation data.

    Formula: symptom_weight = base_weight × (1 - 0.5 × gpt4_fpr)
    """
    logger.info("Deriving symptom weight adjustments...")

    adjustments = {}
    for sid, info in OVERLAPPING_SYMPTOMS.items():
        if sid not in overestimation:
            continue
        gpt4_data = overestimation[sid].get("gpt4", {})
        chatgpt_data = overestimation[sid].get("chatgpt", {})

        gpt4_fpr = gpt4_data.get("false_positive_rate", 0)
        chatgpt_fpr = chatgpt_data.get("false_positive_rate", 0)

        # Use GPT-4 FPR as proxy for GPT-4o-mini (conservative)
        correction = 0.5 * gpt4_fpr
        adjusted_weight = round(1.0 - correction, 4)

        for asrs_item in info["maps_to_asrs"]:
            adjustments[asrs_item] = {
                "source_bdi_symptom": sid,
                "source_name": info["name"],
                "gpt4_fpr": gpt4_fpr,
                "chatgpt_fpr": chatgpt_fpr,
                "overestimation_correction": round(correction, 4),
                "recommended_symptom_weight": adjusted_weight,
                "escalation_priority": (
                    "HIGH" if gpt4_fpr > 0.10
                    else "MEDIUM" if gpt4_fpr > 0.05
                    else "LOW"
                ),
            }

        logger.info(
            "  BDI %d (%s) → ASRS %s: FPR=%.3f, weight=%.3f, priority=%s",
            sid, info["name"], info["maps_to_asrs"],
            gpt4_fpr, adjusted_weight,
            adjustments[info["maps_to_asrs"][0]]["escalation_priority"],
        )

    return adjustments


# === Main ===

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load and align
    unified = load_and_align()
    unified.to_csv(OUTPUT_DIR / "unified_qrels_matrix.csv", index=False)
    logger.info("Saved unified matrix to %s", OUTPUT_DIR / "unified_qrels_matrix.csv")

    # Step 2: Global agreement
    global_agreement = compute_global_agreement(unified)
    per_symptom_agreement = compute_per_symptom_agreement(unified)

    # Step 3: Overestimation profiles
    overestimation = compute_overestimation(unified)

    # Step 4: False positive extraction
    false_positives = extract_false_positives(unified)

    # Step 6: Correction rates
    correction_rates = compute_correction_rates(unified)

    # Derive integration outputs
    weight_adjustments = derive_weight_adjustments(overestimation)

    # === Save all outputs ===

    # Output 1: Per-symptom overestimation
    with open(OUTPUT_DIR / "per_symptom_overestimation.json", "w") as f:
        json.dump(overestimation, f, indent=2)

    # Output 2: False positive sentences (per overlapping symptom)
    fp_dir = OUTPUT_DIR / "false_positive_sentences"
    fp_dir.mkdir(exist_ok=True)
    for sid, data in false_positives.items():
        name = OVERLAPPING_SYMPTOMS[sid]["name"].lower().replace(" ", "_")
        with open(fp_dir / f"symptom_{sid}_{name}_fps.json", "w") as f:
            json.dump(data, f, indent=2)

    # Output 3: Weight adjustments (integration recommendation)
    with open(OUTPUT_DIR / "symptom_weight_adjustments.json", "w") as f:
        json.dump(weight_adjustments, f, indent=2)

    # Output 4: Agreement data
    agreement_output = {
        "global": global_agreement,
        "per_symptom_overlapping": {
            str(sid): per_symptom_agreement.get(sid, {})
            for sid in OVERLAPPING_SYMPTOMS
        },
    }
    with open(OUTPUT_DIR / "agreement_analysis.json", "w") as f:
        json.dump(agreement_output, f, indent=2)

    # Output 5: Correction rates
    with open(OUTPUT_DIR / "gpt4_correction_rate.json", "w") as f:
        json.dump(correction_rates, f, indent=2)

    # Summary report
    summary = {
        "total_pairs": len(unified),
        "unique_sentences": int(unified["sentence_id"].nunique()),
        "symptoms": int(unified["symptom_id"].nunique()),
        "overlapping_symptoms_analyzed": list(OVERLAPPING_SYMPTOMS.keys()),
        "global_agreement": {
            k: {"kappa": v["kappa"], "agreement_rate": v["agreement_rate"]}
            for k, v in global_agreement.items()
            if v is not None
        },
        "overlapping_overestimation_summary": {
            str(sid): {
                "name": info["name"],
                "gpt4_overest_ratio": overestimation[sid]["gpt4"].get("overestimation_ratio"),
                "gpt4_fpr": overestimation[sid]["gpt4"].get("false_positive_rate"),
                "chatgpt_overest_ratio": overestimation[sid]["chatgpt"].get("overestimation_ratio"),
                "chatgpt_fpr": overestimation[sid]["chatgpt"].get("false_positive_rate"),
                "gpt4_fp_count": false_positives[sid]["gpt4_fp_count"],
                "shared_fp_count": false_positives[sid]["shared_fp_count"],
                "correction_rate": correction_rates[sid]["correction_rate"],
                "verdict": correction_rates[sid]["verdict"],
            }
            for sid, info in OVERLAPPING_SYMPTOMS.items()
            if sid in overestimation
        },
        "asrs_weight_adjustments": weight_adjustments,
    }
    with open(OUTPUT_DIR / "profiling_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("")
    logger.info("=== All outputs saved to %s ===", OUTPUT_DIR)
    logger.info("Files: unified_qrels_matrix.csv, per_symptom_overestimation.json,")
    logger.info("       agreement_analysis.json, gpt4_correction_rate.json,")
    logger.info("       symptom_weight_adjustments.json, profiling_summary.json,")
    logger.info("       false_positive_sentences/")


if __name__ == "__main__":
    main()
