"""BERTopic extraction pipeline for RedSM5 depression topic discovery.

Implements the pipeline from specs/bertopic_redsm5_spec.md:
1. Preprocess RedSM5 posts and annotations
2. Run post-level BERTopic (primary)
3. Map topics -> DSM-5 categories via annotation-based plurality vote
4. Extract per-DSM5 vocabulary
5. Compute overlap with Kang et al. ADHD topics
6. Run sentence-level validation
7. Generate all deliverables

Usage:
    uv run python scripts/extract_redsm5_topics.py
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from umap import UMAP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "redsm5"
ANALYSIS_DIR = ROOT / "analysis"
MODEL_DIR = ANALYSIS_DIR / "bertopic_model"

ANALYSIS_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── Kang et al. ADHD topic descriptions ──────────────────────────────────────

ADHD_TOPICS = {
    "focus_self_regulation": "focus concentration attention self-regulation distracted hyperfocus",
    "emotional_dysregulation": "emotions frustration anger overwhelmed mood swings",
    "self_criticism": "shame guilt failure not good enough self-hate lazy",
    "substance_use": "alcohol weed cannabis self-medicate drugs drinking",
    "exhaustion": "tired burnout exhausted no energy fatigue drained",
    "medication": "adderall vyvanse ritalin stimulant dosage side effects",
    "diagnosis": "diagnosed assessment evaluation finally found out adult diagnosis",
    "relationships": "partner spouse friend family understand support",
    "work_academic": "job school deadline boss performance review fired",
    "daily_management": "routine habits organization planning chores cleaning",
}

# DSM-5 categories that overlap with ADHD (for confounder analysis)
OVERLAP_DSM5 = [
    "COGNITIVE_ISSUES", "PSYCHOMOTOR", "FATIGUE", "SLEEP_ISSUES", "ANHEDONIA",
]

ALL_DSM5 = [
    "ANHEDONIA", "APPETITE_CHANGE", "COGNITIVE_ISSUES", "DEPRESSED_MOOD",
    "FATIGUE", "PSYCHOMOTOR", "SLEEP_ISSUES", "SPECIAL_CASE",
    "SUICIDAL_THOUGHTS", "WORTHLESSNESS",
]


# ── Step 1: Preprocessing ───────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Remove URLs, user/subreddit mentions, normalize whitespace."""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"u/\w+", "", text)
    text = re.sub(r"r/\w+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def load_and_preprocess() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load RedSM5 data and apply preprocessing."""
    posts = pd.read_csv(DATA_DIR / "redsm5_posts.csv")
    annotations = pd.read_csv(DATA_DIR / "redsm5_annotations.csv")

    log.info("Loaded %d posts, %d annotations", len(posts), len(annotations))

    # Clean post texts
    posts["clean_text"] = posts["text"].apply(clean_text)

    # Filter short posts (< 10 words)
    word_counts = posts["clean_text"].str.split().str.len()
    posts = posts[word_counts >= 10].reset_index(drop=True)
    log.info("After filtering short posts: %d posts", len(posts))

    # Clean annotation sentences
    annotations["clean_sentence"] = annotations["sentence_text"].apply(clean_text)
    sent_word_counts = annotations["clean_sentence"].str.split().str.len()
    annotations = annotations[sent_word_counts >= 3].reset_index(drop=True)
    log.info("After filtering short sentences: %d annotations", len(annotations))

    return posts, annotations


# ── Step 2: Embedding ────────────────────────────────────────────────────────

def encode_documents(
    model: SentenceTransformer, texts: list[str], desc: str = ""
) -> np.ndarray:
    """Encode texts with progress bar."""
    log.info("Encoding %d documents (%s)...", len(texts), desc)
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    log.info("Encoding complete. Shape: %s", embeddings.shape)
    return embeddings


# ── Step 3: Post-level BERTopic ──────────────────────────────────────────────

def build_post_topic_model(
    documents: list[str],
    embeddings: np.ndarray,
    embedding_model: SentenceTransformer,
) -> tuple[BERTopic, list[int], np.ndarray]:
    """Run BERTopic on post-level documents."""
    log.info("Building post-level BERTopic model...")

    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=15,
        min_samples=5,
        metric="euclidean",
        prediction_data=True,
    )
    representation_model = KeyBERTInspired(top_n_words=15)

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        representation_model=representation_model,
        verbose=True,
    )

    topics, probs = topic_model.fit_transform(documents, embeddings)

    n_topics = len(set(topics)) - (1 if -1 in topics else 0)
    n_outliers = sum(1 for t in topics if t == -1)
    log.info(
        "Post-level: %d topics, %d outliers (%.1f%%)",
        n_topics, n_outliers, 100 * n_outliers / len(topics),
    )
    return topic_model, topics, probs


# ── Step 4: Topic -> DSM-5 Mapping (annotation-based) ───────────────────────

def map_topics_to_dsm5(
    topic_model: BERTopic,
    topics: list[int],
    posts: pd.DataFrame,
    annotations: pd.DataFrame,
) -> dict:
    """Map each topic to DSM-5 categories via annotation-based plurality vote."""
    log.info("Mapping topics to DSM-5 categories...")

    # Build post_id -> list of DSM-5 labels (only status=1 annotations)
    positive_annotations = annotations[annotations["status"] == 1]
    post_to_dsm5 = (
        positive_annotations.groupby("post_id")["DSM5_symptom"]
        .apply(list)
        .to_dict()
    )

    topic_dsm5_dist: dict[int, Counter] = {}
    topic_post_ids: dict[int, list[str]] = {}

    for topic_id in set(topics):
        if topic_id == -1:
            continue
        mask = [i for i, t in enumerate(topics) if t == topic_id]
        topic_posts = posts.iloc[mask]
        topic_post_ids[topic_id] = topic_posts["post_id"].tolist()

        dsm5_counts: Counter = Counter()
        for pid in topic_posts["post_id"]:
            if pid in post_to_dsm5:
                dsm5_counts.update(post_to_dsm5[pid])
        topic_dsm5_dist[topic_id] = dsm5_counts

    # Assign primary and secondary DSM-5 categories
    topic_mappings = {}
    for topic_id, counts in topic_dsm5_dist.items():
        if not counts:
            topic_mappings[topic_id] = {
                "primary": "UNKNOWN",
                "secondary": None,
                "distribution": {},
                "total_annotations": 0,
            }
            continue

        sorted_cats = counts.most_common()
        primary = sorted_cats[0][0]
        primary_count = sorted_cats[0][1]

        secondary = None
        if len(sorted_cats) > 1 and sorted_cats[1][1] > 0.2 * primary_count:
            secondary = sorted_cats[1][0]

        topic_mappings[topic_id] = {
            "primary": primary,
            "secondary": secondary,
            "distribution": dict(counts),
            "total_annotations": sum(counts.values()),
            "purity": primary_count / sum(counts.values()),
        }

        keywords = [w for w, _ in topic_model.get_topic(topic_id)][:10]
        log.info(
            "  Topic %2d -> %-18s (%.0f%% purity) | keywords: %s",
            topic_id,
            primary,
            100 * topic_mappings[topic_id]["purity"],
            ", ".join(keywords[:5]),
        )

    return {
        "mappings": topic_mappings,
        "post_ids": topic_post_ids,
        "dsm5_distributions": {
            k: dict(v) for k, v in topic_dsm5_dist.items()
        },
    }


# ── Step 5: Vocabulary Extraction ────────────────────────────────────────────

def extract_vocabulary(
    topic_model: BERTopic,
    topic_mappings: dict,
    posts: pd.DataFrame,
    topics: list[int],
) -> dict:
    """Extract merged vocabulary per DSM-5 category."""
    log.info("Extracting per-DSM5 vocabulary...")

    mappings = topic_mappings["mappings"]
    dsm5_vocabulary = {}

    for dsm5_cat in ALL_DSM5:
        # Topics where this is primary OR secondary category
        cat_topics = []
        for tid, m in mappings.items():
            if m["primary"] == dsm5_cat or m.get("secondary") == dsm5_cat:
                cat_topics.append(tid)

        if not cat_topics:
            continue

        # Merge topic keywords
        merged_words = []
        for tid in cat_topics:
            topic_words = topic_model.get_topic(tid)
            if topic_words:
                merged_words.extend([(w, score) for w, score in topic_words])

        # Deduplicate, keep highest c-TF-IDF score
        word_scores: dict[str, float] = {}
        for w, score in merged_words:
            word_scores[w] = max(word_scores.get(w, 0), score)

        sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)

        # Extract characteristic expressions from top docs in these topics
        expressions = _extract_expressions(posts, topics, cat_topics)

        dsm5_vocabulary[dsm5_cat] = {
            "high_frequency": [w for w, _ in sorted_words[:10]],
            "medium_frequency": [w for w, _ in sorted_words[10:20]],
            "distinctive": [w for w, _ in sorted_words[20:30]],
            "all_keywords": {w: round(s, 4) for w, s in sorted_words},
            "topic_count": len(cat_topics),
            "topic_ids": cat_topics,
            "characteristic_expressions": expressions,
        }

        log.info(
            "  %-18s: %d topics, %d keywords, top: %s",
            dsm5_cat,
            len(cat_topics),
            len(sorted_words),
            ", ".join([w for w, _ in sorted_words[:5]]),
        )

    return dsm5_vocabulary


def _extract_expressions(
    posts: pd.DataFrame,
    topics: list[int],
    cat_topics: list[int],
    max_expressions: int = 10,
) -> list[str]:
    """Extract short, characteristic expressions from posts in given topics."""
    cat_indices = [i for i, t in enumerate(topics) if t in cat_topics]
    if not cat_indices:
        return []

    cat_posts = posts.iloc[cat_indices]
    expressions = []
    for text in cat_posts["clean_text"].head(50):
        # Extract sentences that are 5-25 words
        sentences = re.split(r"[.!?]+", text)
        for sent in sentences:
            sent = sent.strip()
            wc = len(sent.split())
            if 5 <= wc <= 25:
                expressions.append(sent)
                if len(expressions) >= max_expressions:
                    break
        if len(expressions) >= max_expressions:
            break

    return expressions[:max_expressions]


# ── Step 6: Overlap with Kang et al. ADHD Topics ────────────────────────────

def compute_adhd_overlap(
    topic_model: BERTopic,
    topics: list[int],
    embedding_model: SentenceTransformer,
    topic_mappings: dict,
) -> dict:
    """Compute semantic similarity between depression topics and ADHD topics."""
    log.info("Computing overlap with Kang et al. ADHD topics...")

    # Build depression topic text representations (keywords)
    dep_topic_ids = [t for t in set(topics) if t != -1]
    dep_topic_texts = {}
    for tid in dep_topic_ids:
        words = topic_model.get_topic(tid)
        if words:
            dep_topic_texts[tid] = " ".join([w for w, _ in words])

    if not dep_topic_texts:
        log.warning("No depression topics to compare!")
        return {}

    # Encode both sets
    dep_ids = list(dep_topic_texts.keys())
    dep_texts = list(dep_topic_texts.values())
    adhd_names = list(ADHD_TOPICS.keys())
    adhd_texts = list(ADHD_TOPICS.values())

    dep_embs = embedding_model.encode(dep_texts)
    adhd_embs = embedding_model.encode(adhd_texts)

    overlap_matrix = cosine_similarity(dep_embs, adhd_embs)

    # Build overlap results
    mappings = topic_mappings["mappings"]
    overlap_results = {
        "matrix": {
            "depression_topic_ids": dep_ids,
            "adhd_topic_names": adhd_names,
            "similarities": overlap_matrix.tolist(),
        },
        "per_depression_topic": {},
        "per_adhd_topic": {},
        "overlap_zones": {"high": [], "moderate": [], "low": []},
    }

    # Per depression topic: find best ADHD match
    for i, tid in enumerate(dep_ids):
        best_idx = int(np.argmax(overlap_matrix[i]))
        best_sim = float(overlap_matrix[i, best_idx])
        dsm5_cat = mappings.get(tid, {}).get("primary", "UNKNOWN")

        overlap_results["per_depression_topic"][tid] = {
            "dsm5_category": dsm5_cat,
            "best_adhd_match": adhd_names[best_idx],
            "best_similarity": round(best_sim, 3),
            "all_similarities": {
                name: round(float(overlap_matrix[i, j]), 3)
                for j, name in enumerate(adhd_names)
            },
            "overlap_level": (
                "HIGH" if best_sim > 0.6
                else "MODERATE" if best_sim > 0.3
                else "LOW"
            ),
        }

    # Per ADHD topic: find best depression match
    for j, aname in enumerate(adhd_names):
        best_idx = int(np.argmax(overlap_matrix[:, j]))
        best_sim = float(overlap_matrix[best_idx, j])
        best_dep_id = dep_ids[best_idx]
        dsm5_cat = mappings.get(best_dep_id, {}).get("primary", "UNKNOWN")

        overlap_results["per_adhd_topic"][aname] = {
            "best_depression_topic": best_dep_id,
            "best_depression_dsm5": dsm5_cat,
            "best_similarity": round(best_sim, 3),
            "overlap_level": (
                "HIGH" if best_sim > 0.6
                else "MODERATE" if best_sim > 0.3
                else "LOW"
            ),
        }

    # Classify overlap zones
    for i, tid in enumerate(dep_ids):
        dsm5_cat = mappings.get(tid, {}).get("primary", "UNKNOWN")
        for j, aname in enumerate(adhd_names):
            sim = float(overlap_matrix[i, j])
            entry = {
                "depression_topic": tid,
                "dsm5_category": dsm5_cat,
                "adhd_topic": aname,
                "similarity": round(sim, 3),
            }
            if sim > 0.6:
                overlap_results["overlap_zones"]["high"].append(entry)
            elif sim > 0.3:
                overlap_results["overlap_zones"]["moderate"].append(entry)
            else:
                overlap_results["overlap_zones"]["low"].append(entry)

    # Sort zones by similarity descending
    for zone in overlap_results["overlap_zones"].values():
        zone.sort(key=lambda x: x["similarity"], reverse=True)

    log.info(
        "  HIGH overlap: %d pairs, MODERATE: %d, LOW: %d",
        len(overlap_results["overlap_zones"]["high"]),
        len(overlap_results["overlap_zones"]["moderate"]),
        len(overlap_results["overlap_zones"]["low"]),
    )

    return overlap_results


# ── Step 7: Depression Signal Phrases ────────────────────────────────────────

def extract_signal_phrases(
    topic_model: BERTopic,
    topic_mappings: dict,
    overlap_results: dict,
) -> dict:
    """Extract depression-only signal phrases from LOW overlap topics."""
    log.info("Extracting depression signal phrases...")

    mappings = topic_mappings["mappings"]
    per_topic = overlap_results.get("per_depression_topic", {})

    # Topics with LOW overlap = depression-only
    depression_only_topics = []
    moderate_overlap_topics = []
    for tid_str, info in per_topic.items():
        tid = int(tid_str) if isinstance(tid_str, str) else tid_str
        if info["overlap_level"] == "LOW":
            depression_only_topics.append(tid)
        elif info["overlap_level"] == "MODERATE":
            moderate_overlap_topics.append(tid)

    # Extract vocabulary from depression-only topics
    strong_signals = []
    moderate_signals = []
    for tid in depression_only_topics:
        words = topic_model.get_topic(tid)
        if words:
            for w, score in words[:10]:
                strong_signals.append((w, score))

    for tid in moderate_overlap_topics:
        words = topic_model.get_topic(tid)
        if words:
            for w, score in words[:5]:
                moderate_signals.append((w, score))

    # Deduplicate
    strong_words = list(dict.fromkeys([w for w, _ in strong_signals]))
    moderate_words = list(dict.fromkeys([w for w, _ in moderate_signals]))

    result = {
        "depression_only_signals": {
            "strong": strong_words[:20],
            "moderate": moderate_words[:15],
            "source_topics": {
                "low_overlap": depression_only_topics,
                "moderate_overlap": moderate_overlap_topics,
            },
        },
        "adhd_only_signals": {
            "note": "From Kang et al. / Layer 3 definitions - not extracted here",
            "examples": [
                "hyperfocus", "fidget", "time blindness", "doom scrolling",
                "wall of awful", "executive dysfunction", "stimming",
                "body doubling", "interest-based nervous system",
            ],
        },
    }

    log.info(
        "  Depression-only: %d strong, %d moderate signal terms",
        len(result["depression_only_signals"]["strong"]),
        len(result["depression_only_signals"]["moderate"]),
    )
    return result


# ── Step 8: Per-Symptom Confounder Phrases ───────────────────────────────────

# ASRS items that overlap with depression DSM-5 categories
ASRS_DEPRESSION_OVERLAP = {
    "asrs_item_7": {
        "symptom": "difficulty with attention to detail",
        "dsm5_categories": ["COGNITIVE_ISSUES", "FATIGUE", "SLEEP_ISSUES"],
    },
    "asrs_item_8": {
        "symptom": "difficulty concentrating on what people say",
        "dsm5_categories": ["COGNITIVE_ISSUES"],
    },
    "asrs_item_9": {
        "symptom": "difficulty remembering appointments or obligations",
        "dsm5_categories": ["COGNITIVE_ISSUES"],
    },
    "asrs_item_10": {
        "symptom": "difficulty concentrating on boring tasks",
        "dsm5_categories": ["COGNITIVE_ISSUES", "ANHEDONIA"],
    },
    "asrs_item_11": {
        "symptom": "difficulty keeping attention on repetitive tasks",
        "dsm5_categories": ["COGNITIVE_ISSUES", "ANHEDONIA"],
    },
    "asrs_item_4": {
        "symptom": "avoiding or delaying getting started on tasks",
        "dsm5_categories": ["FATIGUE", "ANHEDONIA"],
    },
    "asrs_item_5": {
        "symptom": "fidgeting or squirming",
        "dsm5_categories": ["PSYCHOMOTOR"],
    },
    "asrs_item_6": {
        "symptom": "feeling overly active or compelled to do things",
        "dsm5_categories": ["PSYCHOMOTOR"],
    },
    "asrs_item_12": {
        "symptom": "leaving seat in situations where remaining seated is expected",
        "dsm5_categories": ["PSYCHOMOTOR"],
    },
    "asrs_item_13": {
        "symptom": "feeling restless or fidgety",
        "dsm5_categories": ["PSYCHOMOTOR"],
    },
}


def extract_per_symptom_confounders(
    topic_model: BERTopic,
    topic_mappings: dict,
    dsm5_vocabulary: dict,
) -> dict:
    """Build per-ASRS-item confounder phrase lists from depression topic vocabulary."""
    log.info("Extracting per-symptom confounder phrases...")

    mappings = topic_mappings["mappings"]
    result = {}

    for item_key, item_info in ASRS_DEPRESSION_OVERLAP.items():
        relevant_cats = item_info["dsm5_categories"]

        # Collect all phrases from relevant DSM-5 categories
        phrases = []
        source_topics = []
        source_count = 0

        for cat in relevant_cats:
            if cat in dsm5_vocabulary:
                vocab = dsm5_vocabulary[cat]
                phrases.extend(vocab.get("high_frequency", []))
                phrases.extend(vocab.get("medium_frequency", []))
                source_topics.extend(
                    [f"Topic {t}" for t in vocab.get("topic_ids", [])]
                )
                source_count += vocab.get("topic_count", 0)

        # Deduplicate
        phrases = list(dict.fromkeys(phrases))

        result[item_key] = {
            "symptom": item_info["symptom"],
            "depression_confounders": {
                "phrases": phrases[:15],
                "dsm5_categories": relevant_cats,
                "source_topics": list(set(source_topics)),
                "source_topic_count": source_count,
            },
        }

    return result


# ── Step 9: Sentence-Level Validation ────────────────────────────────────────

def run_sentence_validation(
    annotations: pd.DataFrame,
    embedding_model: SentenceTransformer,
) -> dict:
    """Run BERTopic on annotated sentences to validate topic-DSM5 alignment."""
    log.info("Running sentence-level validation...")

    sentences = annotations["clean_sentence"].tolist()
    sent_embeddings = encode_documents(
        embedding_model, sentences, desc="sentence-level"
    )

    umap_model = UMAP(
        n_neighbors=10,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=10,
        min_samples=3,
        metric="euclidean",
        prediction_data=True,
    )

    sent_topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        representation_model=KeyBERTInspired(top_n_words=15),
        verbose=True,
    )

    sent_topics, sent_probs = sent_topic_model.fit_transform(
        sentences, sent_embeddings
    )

    # Compute purity per topic
    validation_results = {
        "n_sentences": len(sentences),
        "n_topics": len(set(sent_topics)) - (1 if -1 in sent_topics else 0),
        "n_outliers": sum(1 for t in sent_topics if t == -1),
        "topics": {},
        "dsm5_category_summary": {},
    }

    for topic_id in set(sent_topics):
        if topic_id == -1:
            continue
        mask = [i for i, t in enumerate(sent_topics) if t == topic_id]
        topic_labels = annotations.iloc[mask]["DSM5_symptom"]
        label_counts = topic_labels.value_counts()
        dominant = label_counts.index[0]
        purity = float(label_counts.iloc[0] / len(topic_labels))

        keywords = sent_topic_model.get_topic(topic_id)
        kw_list = [w for w, _ in keywords[:10]] if keywords else []

        validation_results["topics"][topic_id] = {
            "size": len(mask),
            "dominant_dsm5": dominant,
            "purity": round(purity, 3),
            "label_distribution": label_counts.to_dict(),
            "keywords": kw_list,
        }

        log.info(
            "  Sent-Topic %2d: purity=%.2f, dominant=%-18s, size=%d, kw=%s",
            topic_id, purity, dominant, len(mask),
            ", ".join(kw_list[:5]),
        )

    # Summary by DSM-5 category: how well does each cluster?
    dsm5_purities: dict[str, list[float]] = defaultdict(list)
    for tid, info in validation_results["topics"].items():
        dsm5_purities[info["dominant_dsm5"]].append(info["purity"])

    for cat, purities in dsm5_purities.items():
        validation_results["dsm5_category_summary"][cat] = {
            "n_topics": len(purities),
            "mean_purity": round(float(np.mean(purities)), 3),
            "max_purity": round(float(max(purities)), 3),
            "min_purity": round(float(min(purities)), 3),
        }

    return validation_results


# ── Step 10: Overlap Heatmap ─────────────────────────────────────────────────

def save_overlap_heatmap(overlap_results: dict, topic_mappings: dict) -> None:
    """Save overlap heatmap as a plotly HTML (no matplotlib dependency)."""
    try:
        import plotly.graph_objects as go

        matrix_data = overlap_results.get("matrix", {})
        dep_ids = matrix_data.get("depression_topic_ids", [])
        adhd_names = matrix_data.get("adhd_topic_names", [])
        sims = np.array(matrix_data.get("similarities", []))

        if sims.size == 0:
            log.warning("No overlap data to plot.")
            return

        mappings = topic_mappings["mappings"]
        dep_labels = []
        for tid in dep_ids:
            dsm5 = mappings.get(tid, {}).get("primary", "?")
            dep_labels.append(f"T{tid} ({dsm5})")

        adhd_labels = [n.replace("_", " ").title() for n in adhd_names]

        fig = go.Figure(data=go.Heatmap(
            z=sims,
            x=adhd_labels,
            y=dep_labels,
            colorscale="RdYlBu_r",
            zmin=0, zmax=1,
            text=np.round(sims, 2),
            texttemplate="%{text}",
            textfont={"size": 9},
        ))
        fig.update_layout(
            title="Depression Topic vs. ADHD Topic Overlap (Cosine Similarity)",
            xaxis_title="Kang et al. ADHD Topics",
            yaxis_title="RedSM5 Depression Topics (BERTopic)",
            width=1000,
            height=max(400, 40 * len(dep_ids)),
        )
        out_path = ANALYSIS_DIR / "overlap_heatmap.html"
        fig.write_html(str(out_path))
        log.info("Saved overlap heatmap to %s", out_path)

    except Exception as e:
        log.warning("Could not generate heatmap: %s", e)


# ── Main Pipeline ────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=" * 60)
    log.info("BERTopic Extraction Pipeline: RedSM5 Depression Topics")
    log.info("=" * 60)

    # Step 1: Preprocess
    posts, annotations = load_and_preprocess()

    # Step 2: Encode
    embedding_model = SentenceTransformer("all-mpnet-base-v2")
    post_texts = posts["clean_text"].tolist()
    post_embeddings = encode_documents(embedding_model, post_texts, desc="post-level")

    # Step 3: Post-level BERTopic
    topic_model, topics, probs = build_post_topic_model(
        post_texts, post_embeddings, embedding_model
    )

    # Print topic overview
    topic_info = topic_model.get_topic_info()
    log.info("\nTopic Overview:")
    for _, row in topic_info.iterrows():
        if row["Topic"] == -1:
            log.info("  Topic -1 (outliers): %d documents", row["Count"])
        else:
            words = topic_model.get_topic(row["Topic"])
            kw = ", ".join([w for w, _ in words[:8]]) if words else ""
            log.info("  Topic %2d: %4d docs | %s", row["Topic"], row["Count"], kw)

    # Step 4: Map to DSM-5
    topic_mapping_results = map_topics_to_dsm5(
        topic_model, topics, posts, annotations
    )

    # Step 5: Extract vocabulary
    dsm5_vocabulary = extract_vocabulary(topic_model, topic_mapping_results, posts, topics)

    # Step 6: Overlap with ADHD
    overlap_results = compute_adhd_overlap(
        topic_model, topics, embedding_model, topic_mapping_results
    )

    # Step 7: Depression signal phrases
    signal_phrases = extract_signal_phrases(
        topic_model, topic_mapping_results, overlap_results
    )

    # Step 8: Per-symptom confounders
    confounder_phrases = extract_per_symptom_confounders(
        topic_model, topic_mapping_results, dsm5_vocabulary
    )

    # Step 9: Sentence-level validation
    sentence_validation = run_sentence_validation(annotations, embedding_model)

    # Step 10: Save heatmap
    save_overlap_heatmap(overlap_results, topic_mapping_results)

    # ── Save all deliverables ────────────────────────────────────────────

    log.info("\nSaving deliverables to %s ...", ANALYSIS_DIR)

    def _save_json(data: dict, name: str) -> None:
        path = ANALYSIS_DIR / name
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        log.info("  Saved %s", path.name)

    _save_json(dsm5_vocabulary, "depression_topic_vocabulary.json")
    _save_json(overlap_results, "adhd_depression_overlap_map.json")
    _save_json(signal_phrases, "depression_signal_phrases.json")
    _save_json(confounder_phrases, "per_symptom_confounder_phrases.json")
    _save_json(sentence_validation, "sentence_level_topic_validation.json")

    # Save topic mapping details
    _save_json(topic_mapping_results, "topic_dsm5_mappings.json")

    # Save BERTopic model
    model_path = MODEL_DIR / "post_level_model"
    topic_model.save(str(model_path), serialization="safetensors", save_ctfidf=True)
    log.info("  Saved BERTopic model to %s", model_path)

    # ── Summary ──────────────────────────────────────────────────────────

    log.info("\n" + "=" * 60)
    log.info("PIPELINE COMPLETE")
    log.info("=" * 60)
    log.info("Posts processed: %d", len(posts))
    log.info("Post-level topics: %d", len(set(topics)) - (1 if -1 in topics else 0))
    log.info("Sentence-level topics: %d", sentence_validation["n_topics"])
    log.info("DSM-5 categories with vocabulary: %d", len(dsm5_vocabulary))
    log.info(
        "ADHD overlap zones: %d high, %d moderate, %d low",
        len(overlap_results.get("overlap_zones", {}).get("high", [])),
        len(overlap_results.get("overlap_zones", {}).get("moderate", [])),
        len(overlap_results.get("overlap_zones", {}).get("low", [])),
    )
    log.info("Deliverables saved to: %s", ANALYSIS_DIR)


if __name__ == "__main__":
    main()
