#!/usr/bin/env python3
"""Exploratory Data Analysis (EDA) for eRisk 2026 Task 3 (ADHD symptom sentence ranking).

Streams the official test corpus one .trec file at a time (4521 files, ~4.17M
sentences) accumulating aggregate statistics only -- it never holds all sentences
in memory. The qrels (majority + unanimity) are read fully; the small set of judged
doc_ids (~4.3k unique) is captured during the single corpus pass so that the text of
judged / relevant sentences can be compared against the corpus at large.

Outputs:
  - analysis/eda_task3/eda_task3.json   (all numeric stats)
  - analysis/eda_task3/*.png            (figures, dpi=150)

Re-runnable and self-contained. Repo root is resolved relative to this file.
"""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

# Make console output robust to non-ASCII on Windows (cp1252) terminals.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parents[1]

CORPUS_DIR = (
    REPO_ROOT
    / "data"
    / "eRisk-2026"
    / "task3-adhd-symptom-ranking-20260204T094934Z-3-001"
    / "task3-adhd-symptom-ranking"
    / "output_trec_files"
    / "output_trec_files"
)

QRELS_DIR = (
    REPO_ROOT
    / "data"
    / "eRisk-2026"
    / "eRisk26-datasets-20260519T175618Z-3-001"
    / "eRisk26-datasets"
    / "task3-adhd-symptom-ranking"
    / "golden-data"
)
QRELS_MAJORITY = QRELS_DIR / "qrels_majority-final.csv"
QRELS_UNANIMITY = QRELS_DIR / "qrels_unanimity-final.csv"

OUT_DIR = REPO_ROOT / "analysis" / "eda_task3"
OUT_JSON = OUT_DIR / "eda_task3.json"

# Set to an int N to sample every Nth corpus file (records sampling in JSON).
# Default None => full corpus pass.
CORPUS_FILE_SAMPLE = None

# ASRS-v1.1 symptom names for the 18 query ids.
ASRS_SYMPTOMS = {
    1: "Trouble wrapping up final details",
    2: "Difficulty getting things in order",
    3: "Problems remembering appointments/obligations",
    4: "Avoiding/delaying tasks requiring thought",
    5: "Fidgeting hands/feet",
    6: "Feeling overly active/compelled",
    7: "Careless mistakes on boring/difficult work",
    8: "Difficulty keeping attention on boring/repetitive work",
    9: "Difficulty concentrating on what people say",
    10: "Misplacing/finding things",
    11: "Distracted by activity/noise",
    12: "Leaving seat when seated expected",
    13: "Feeling restless/fidgety",
    14: "Difficulty unwinding/relaxing",
    15: "Talking too much",
    16: "Finishing others' sentences",
    17: "Difficulty waiting your turn",
    18: "Interrupting others",
}
# Compact labels for figure axes.
ASRS_SHORT = {
    1: "wrap-up details",
    2: "getting in order",
    3: "remember appts",
    4: "avoid thought tasks",
    5: "fidget hands/feet",
    6: "overly active",
    7: "careless mistakes",
    8: "keep attention",
    9: "concentrate on speech",
    10: "misplacing things",
    11: "distracted noise",
    12: "leaving seat",
    13: "restless/fidgety",
    14: "unwinding",
    15: "talking too much",
    16: "finish sentences",
    17: "waiting turn",
    18: "interrupting",
}

# Symptoms our runs missed entirely at retrieval (per docs/task3_results_analysis.md).
MISSED_SYMPTOMS = {7, 9, 10}

# --------------------------------------------------------------------------- #
# Regexes for streaming TREC parse
# --------------------------------------------------------------------------- #
RE_DOCNO = re.compile(r"<DOCNO>\s*(.*?)\s*</DOCNO>")
RE_TEXT = re.compile(r"<TEXT>(.*?)</TEXT>", re.DOTALL)


def word_count(text: str) -> int:
    """Whitespace word count, robust to empty/None."""
    if not text:
        return 0
    return len(text.split())


# --------------------------------------------------------------------------- #
# Running statistics accumulators (constant memory)
# --------------------------------------------------------------------------- #
class RunningStats:
    """Welford-style accumulator for count/mean/std plus min/max and a histogram.

    Median is approximated from the histogram (the corpus is far too large to keep
    every value); the bin width is small so the approximation is tight.
    """

    def __init__(self, hist_max: int, hist_bin: int = 1):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.vmin = math.inf
        self.vmax = -math.inf
        self.hist_max = hist_max
        self.hist_bin = hist_bin
        self.nbins = hist_max // hist_bin + 1  # last bin = overflow
        self.hist = np.zeros(self.nbins + 1, dtype=np.int64)

    def add(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (x - self.mean)
        if x < self.vmin:
            self.vmin = x
        if x > self.vmax:
            self.vmax = x
        b = int(x) // self.hist_bin
        if b > self.nbins:
            b = self.nbins
        self.hist[b] += 1

    @property
    def std(self) -> float:
        return math.sqrt(self.m2 / self.n) if self.n > 1 else 0.0

    def approx_median(self) -> float:
        if self.n == 0:
            return 0.0
        target = self.n / 2.0
        cum = 0
        for b, c in enumerate(self.hist):
            cum += c
            if cum >= target:
                return float(b * self.hist_bin)
        return float(self.vmax)

    def summary(self) -> dict:
        return {
            "count": int(self.n),
            "min": (None if self.n == 0 else float(self.vmin)),
            "max": (None if self.n == 0 else float(self.vmax)),
            "mean": float(self.mean),
            "median_approx": self.approx_median(),
            "std": self.std,
        }

    def hist_for_plot(self):
        """Return (bin_left_edges, counts) trimmed to the populated range."""
        edges = np.arange(self.nbins + 1) * self.hist_bin
        counts = self.hist[:-1]  # drop overflow bin for plotting
        # trim trailing zeros
        nz = np.nonzero(counts)[0]
        if len(nz) == 0:
            return edges[:1], counts[:1]
        last = nz[-1] + 1
        return edges[:last], counts[:last]


# --------------------------------------------------------------------------- #
# Step 1: read qrels (full)
# --------------------------------------------------------------------------- #
def read_qrels(path: Path):
    """Return (rows, judged_docids).

    rows: list of (query:int, doc_id:str, relevant:bool)
    judged_docids: set of doc_id strings.
    """
    rows = []
    judged = set()
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            q = int(r["query"])
            doc_id = r["doc_id"].strip()
            rel = r["relevant"].strip().lower() == "true"
            rows.append((q, doc_id, rel))
            judged.add(doc_id)
    return rows, judged


def qrels_per_symptom(rows):
    """Per-symptom judged-pool size, relevant count, relevance rate."""
    pool = defaultdict(int)
    rel = defaultdict(int)
    for q, _doc, r in rows:
        pool[q] += 1
        if r:
            rel[q] += 1
    out = {}
    for sid in range(1, 19):
        judged = pool.get(sid, 0)
        relevant = rel.get(sid, 0)
        out[sid] = {
            "symptom": ASRS_SYMPTOMS[sid],
            "judged": judged,
            "relevant": relevant,
            "relevance_rate": (relevant / judged if judged else 0.0),
        }
    return out


# --------------------------------------------------------------------------- #
# Step 2: stream the corpus
# --------------------------------------------------------------------------- #
def stream_corpus(judged_docids):
    """Single streaming pass over all .trec files.

    Returns a dict of accumulated stats and a {doc_id: text} map for judged docs.
    """
    files = sorted(
        CORPUS_DIR.glob("s_*.trec"),
        key=lambda p: int(re.search(r"s_(\d+)\.trec", p.name).group(1)),
    )
    if CORPUS_FILE_SAMPLE:
        files = files[::CORPUS_FILE_SAMPLE]

    n_files = len(files)
    print(f"[corpus] {n_files} files to process "
          f"(sampling every {CORPUS_FILE_SAMPLE} file)" if CORPUS_FILE_SAMPLE
          else f"[corpus] {n_files} files to process (full pass)")

    sent_len_words = RunningStats(hist_max=200, hist_bin=1)
    sent_len_chars = RunningStats(hist_max=1000, hist_bin=5)
    sents_per_subject = RunningStats(hist_max=5000, hist_bin=25)
    sents_per_post = RunningStats(hist_max=100, hist_bin=1)
    posts_per_subject = RunningStats(hist_max=2000, hist_bin=10)

    total_sentences = 0
    total_subjects = 0
    empty_text = 0
    malformed_blocks = 0

    judged_text = {}  # doc_id -> TEXT (only for judged docs)

    t0 = time.time()
    for i, fp in enumerate(files, 1):
        try:
            raw = fp.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            print(f"[warn] could not read {fp.name}: {e}")
            continue

        total_subjects += 1
        subj_sent_count = 0
        # postIndex -> sentence count for this subject
        post_counts = defaultdict(int)

        # Split on <DOC> ... </DOC> blocks. Robust to missing tags.
        for block in raw.split("<DOC>"):
            if "</DOC>" not in block:
                continue
            m_doc = RE_DOCNO.search(block)
            m_txt = RE_TEXT.search(block)
            if m_doc is None:
                malformed_blocks += 1
                continue
            doc_id = m_doc.group(1).strip()
            text = m_txt.group(1) if m_txt else ""
            # Unescape minimal entities that may appear in TEXT.
            if "&" in text:
                text = (text.replace("&amp;", "&")
                            .replace("&lt;", "<")
                            .replace("&gt;", ">")
                            .replace("&#39;", "'")
                            .replace("&quot;", '"'))
            text = text.strip()

            # parse postIndex from doc_id ({subject}_{post}_{sentence})
            parts = doc_id.rsplit("_", 2)
            post_idx = None
            if len(parts) == 3:
                try:
                    post_idx = int(parts[1])
                except ValueError:
                    post_idx = None

            total_sentences += 1
            subj_sent_count += 1
            if post_idx is not None:
                post_counts[post_idx] += 1

            wc = word_count(text)
            cc = len(text)
            if wc == 0:
                empty_text += 1
            sent_len_words.add(wc)
            sent_len_chars.add(cc)

            if doc_id in judged_docids:
                # keep the first occurrence (doc_id should be unique anyway)
                judged_text.setdefault(doc_id, text)

        sents_per_subject.add(subj_sent_count)
        posts_per_subject.add(len(post_counts))
        for c in post_counts.values():
            sents_per_post.add(c)

        if i % 500 == 0 or i == n_files:
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed else 0
            print(f"[corpus] {i}/{n_files} files | "
                  f"{total_sentences:,} sentences | "
                  f"{elapsed:.0f}s | {rate:.1f} files/s")

    return {
        "n_files": n_files,
        "total_subjects": total_subjects,
        "total_sentences": total_sentences,
        "empty_text": empty_text,
        "malformed_blocks": malformed_blocks,
        "sent_len_words": sent_len_words,
        "sent_len_chars": sent_len_chars,
        "sents_per_subject": sents_per_subject,
        "sents_per_post": sents_per_post,
        "posts_per_subject": posts_per_subject,
        "judged_text": judged_text,
        "elapsed_s": time.time() - t0,
    }


# --------------------------------------------------------------------------- #
# Step 3: relevant vs judged vs corpus text length
# --------------------------------------------------------------------------- #
def text_length_comparison(maj_rows, judged_text):
    """Word-length stats for relevant (majority) vs all judged sentences.

    Returns (relevant_lengths, judged_lengths, stats_dict).
    """
    relevant_doc_ids = set()
    for q, doc_id, rel in maj_rows:
        if rel:
            relevant_doc_ids.add(doc_id)

    judged_lengths = []
    relevant_lengths = []
    missing = 0
    for doc_id, text in judged_text.items():
        wc = word_count(text)
        judged_lengths.append(wc)
        if doc_id in relevant_doc_ids:
            relevant_lengths.append(wc)

    # doc_ids judged but not found in corpus pass
    for doc_id in relevant_doc_ids:
        if doc_id not in judged_text:
            missing += 1

    def stat(vals):
        if not vals:
            return {"count": 0, "mean": 0, "median": 0, "std": 0, "min": 0, "max": 0}
        return {
            "count": len(vals),
            "mean": float(statistics.fmean(vals)),
            "median": float(statistics.median(vals)),
            "std": float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0,
            "min": int(min(vals)),
            "max": int(max(vals)),
        }

    stats = {
        "relevant_majority": stat(relevant_lengths),
        "all_judged": stat(judged_lengths),
        "n_relevant_docids_unique": len(relevant_doc_ids),
        "n_judged_text_captured": len(judged_text),
        "relevant_docids_missing_from_corpus": missing,
    }
    return relevant_lengths, judged_lengths, stats


# --------------------------------------------------------------------------- #
# Step 4: majority vs unanimity agreement
# --------------------------------------------------------------------------- #
def agreement_analysis(maj_per_sym, una_per_sym):
    """Per-symptom shrinkage: unanimity relevant / majority relevant."""
    out = {}
    tot_maj = 0
    tot_una = 0
    for sid in range(1, 19):
        m = maj_per_sym[sid]["relevant"]
        u = una_per_sym[sid]["relevant"]
        tot_maj += m
        tot_una += u
        out[sid] = {
            "symptom": ASRS_SYMPTOMS[sid],
            "majority_relevant": m,
            "unanimity_relevant": u,
            "survive_ratio": (u / m if m else 0.0),
        }
    overall = {
        "total_majority_relevant": tot_maj,
        "total_unanimity_relevant": tot_una,
        "overall_shrinkage_ratio": (tot_una / tot_maj if tot_maj else 0.0),
    }
    return out, overall


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_sentences_per_subject(rs: RunningStats):
    edges, counts = rs.hist_for_plot()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(edges, counts, width=rs.hist_bin, align="edge", color="#4C72B0",
           edgecolor="none")
    ax.set_yscale("log")
    ax.axvline(rs.mean, color="crimson", linestyle="--", linewidth=1.5,
               label=f"mean = {rs.mean:.0f}")
    ax.set_xlabel("sentences per subject")
    ax.set_ylabel("number of subjects (log scale)")
    ax.set_title("Distribution of sentences per subject (Task 3 test corpus)")
    ax.legend()
    fig.tight_layout()
    p = OUT_DIR / "sentences_per_subject.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_sentence_length_dist(rs: RunningStats):
    edges, counts = rs.hist_for_plot()
    fig, ax = plt.subplots(figsize=(9, 5))
    # focus on 0..60 words where the mass lives
    xmax = 60
    mask = edges < xmax
    ax.bar(edges[mask], counts[mask], width=rs.hist_bin, align="edge",
           color="#55A868", edgecolor="none")
    ax.axvline(rs.mean, color="crimson", linestyle="--", linewidth=1.5,
               label=f"mean = {rs.mean:.1f} words")
    ax.axvline(rs.approx_median(), color="navy", linestyle=":", linewidth=1.5,
               label=f"median ≈ {rs.approx_median():.0f} words")
    ax.set_xlabel("words per sentence")
    ax.set_ylabel("number of sentences")
    ax.set_title("Sentence length distribution (words) — corpus overall")
    ax.legend()
    fig.tight_layout()
    p = OUT_DIR / "sentence_length_dist.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_qrels_relevant_per_symptom(maj_per_sym, una_per_sym):
    sids = list(range(1, 19))
    maj = [maj_per_sym[s]["relevant"] for s in sids]
    una = [una_per_sym[s]["relevant"] for s in sids]
    x = np.arange(len(sids))
    w = 0.4
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - w / 2, maj, w, label="majority (≥2/3)", color="#4C72B0")
    ax.bar(x + w / 2, una, w, label="unanimity (3/3)", color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n{ASRS_SHORT[s]}" for s in sids],
                       rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("# relevant docs")
    ax.set_title("Relevant docs per symptom: majority vs unanimity qrels")
    ax.legend()
    fig.tight_layout()
    p = OUT_DIR / "qrels_relevant_per_symptom.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_relevance_rate_per_symptom(maj_per_sym):
    sids = list(range(1, 19))
    pairs = sorted(sids, key=lambda s: maj_per_sym[s]["relevance_rate"],
                   reverse=True)
    rates = [maj_per_sym[s]["relevance_rate"] for s in pairs]
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#DD8452" if s in MISSED_SYMPTOMS else "#4C72B0" for s in pairs]
    ax.bar(range(len(pairs)), rates, color=colors)
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels([f"{s}\n{ASRS_SHORT[s]}" for s in pairs],
                       rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("relevance rate (relevant / judged)")
    ax.set_title("Relevance rate per symptom (majority qrels), sorted "
                 "— orange = our retrieval misses (7,9,10)")
    fig.tight_layout()
    p = OUT_DIR / "relevance_rate_per_symptom.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_majority_vs_unanimity_agreement(agree):
    sids = list(range(1, 19))
    una = [agree[s]["unanimity_relevant"] for s in sids]
    drop = [agree[s]["majority_relevant"] - agree[s]["unanimity_relevant"]
            for s in sids]
    x = np.arange(len(sids))
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x, una, label="survive unanimity (3/3)", color="#55A868")
    ax.bar(x, drop, bottom=una, label="lost majority→unanimity",
           color="#C44E52", alpha=0.8)
    # annotate survival ratio
    for i, s in enumerate(sids):
        tot = agree[s]["majority_relevant"]
        if tot:
            ax.text(i, tot + 0.5, f"{agree[s]['survive_ratio']*100:.0f}%",
                    ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n{ASRS_SHORT[s]}" for s in sids],
                       rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("# majority-relevant docs")
    ax.set_title("Majority→unanimity strictness per symptom "
                 "(% = fraction surviving unanimity)")
    ax.legend()
    fig.tight_layout()
    p = OUT_DIR / "majority_vs_unanimity_agreement.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_relevant_vs_corpus_length(relevant_lengths, corpus_rs: RunningStats):
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.arange(0, 61, 2)
    # corpus from histogram -> reconstruct density up to 60 words
    edges, counts = corpus_rs.hist_for_plot()
    ax.hist(np.clip(relevant_lengths, 0, 60), bins=bins, density=True,
            alpha=0.6, color="#C44E52", label="relevant (majority)")
    # corpus density: build a normalized step from RunningStats histogram
    c_edges = edges[edges < 60]
    c_counts = counts[: len(c_edges)].astype(float)
    if c_counts.sum() > 0:
        c_density = c_counts / c_counts.sum()
        ax.bar(c_edges, c_density, width=corpus_rs.hist_bin, align="edge",
               alpha=0.4, color="#4C72B0", label="corpus overall")
    ax.axvline(np.mean(relevant_lengths), color="#C44E52", linestyle="--",
               linewidth=1.5,
               label=f"relevant mean = {np.mean(relevant_lengths):.1f}")
    ax.axvline(corpus_rs.mean, color="#4C72B0", linestyle="--", linewidth=1.5,
               label=f"corpus mean = {corpus_rs.mean:.1f}")
    ax.set_xlabel("words per sentence")
    ax.set_ylabel("density")
    ax.set_title("Sentence length: relevant (majority) vs corpus overall")
    ax.legend()
    fig.tight_layout()
    p = OUT_DIR / "relevant_vs_corpus_length.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[setup] repo root: {REPO_ROOT}")
    print(f"[setup] corpus dir: {CORPUS_DIR}")
    print(f"[setup] output dir: {OUT_DIR}")

    # --- qrels ---
    print("[qrels] reading majority + unanimity ...")
    maj_rows, maj_judged = read_qrels(QRELS_MAJORITY)
    una_rows, una_judged = read_qrels(QRELS_UNANIMITY)
    all_judged = maj_judged | una_judged
    print(f"[qrels] majority rows={len(maj_rows)} unanimity rows={len(una_rows)}")
    print(f"[qrels] unique judged doc_ids (union)={len(all_judged)}")

    maj_per_sym = qrels_per_symptom(maj_rows)
    una_per_sym = qrels_per_symptom(una_rows)
    agree, agree_overall = agreement_analysis(maj_per_sym, una_per_sym)

    # --- corpus streaming pass ---
    print("[corpus] starting streaming pass ...")
    corpus = stream_corpus(all_judged)
    print(f"[corpus] DONE: {corpus['total_sentences']:,} sentences across "
          f"{corpus['total_subjects']} subjects in {corpus['elapsed_s']:.0f}s")
    print(f"[corpus] judged-text captured: {len(corpus['judged_text'])}/"
          f"{len(all_judged)}")

    # --- text length comparison ---
    relevant_lengths, judged_lengths, text_stats = text_length_comparison(
        maj_rows, corpus["judged_text"]
    )

    # --- per-symptom highlights ---
    by_rel = sorted(range(1, 19), key=lambda s: maj_per_sym[s]["relevant"])
    by_rate = sorted(range(1, 19),
                     key=lambda s: maj_per_sym[s]["relevance_rate"])

    highlights = {
        "most_relevant_symptoms": [
            {"sid": s, "symptom": ASRS_SYMPTOMS[s],
             "relevant": maj_per_sym[s]["relevant"]} for s in by_rel[::-1][:3]
        ],
        "fewest_relevant_symptoms": [
            {"sid": s, "symptom": ASRS_SYMPTOMS[s],
             "relevant": maj_per_sym[s]["relevant"]} for s in by_rel[:3]
        ],
        "highest_relevance_rate": [
            {"sid": s, "symptom": ASRS_SYMPTOMS[s],
             "relevance_rate": maj_per_sym[s]["relevance_rate"]}
            for s in by_rate[::-1][:3]
        ],
        "lowest_relevance_rate": [
            {"sid": s, "symptom": ASRS_SYMPTOMS[s],
             "relevance_rate": maj_per_sym[s]["relevance_rate"]}
            for s in by_rate[:3]
        ],
    }

    # --- figures ---
    print("[figures] writing ...")
    figs = {}
    figs["sentences_per_subject"] = str(
        fig_sentences_per_subject(corpus["sents_per_subject"]))
    figs["sentence_length_dist"] = str(
        fig_sentence_length_dist(corpus["sent_len_words"]))
    figs["qrels_relevant_per_symptom"] = str(
        fig_qrels_relevant_per_symptom(maj_per_sym, una_per_sym))
    figs["relevance_rate_per_symptom"] = str(
        fig_relevance_rate_per_symptom(maj_per_sym))
    figs["majority_vs_unanimity_agreement"] = str(
        fig_majority_vs_unanimity_agreement(agree))
    figs["relevant_vs_corpus_length"] = str(
        fig_relevant_vs_corpus_length(relevant_lengths,
                                      corpus["sent_len_words"]))

    # verify figures written
    for name, p in figs.items():
        pth = Path(p)
        ok = pth.exists() and pth.stat().st_size > 0
        print(f"[figures] {name}: {'OK' if ok else 'MISSING'} ({p})")
        if not ok:
            raise RuntimeError(f"figure not written: {p}")

    # --- assemble JSON ---
    result = {
        "meta": {
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "corpus_dir": str(CORPUS_DIR),
            "qrels_majority": str(QRELS_MAJORITY),
            "qrels_unanimity": str(QRELS_UNANIMITY),
            "corpus_file_sample_every_n": CORPUS_FILE_SAMPLE,
            "sampled": CORPUS_FILE_SAMPLE is not None,
            "corpus_pass_seconds": round(corpus["elapsed_s"], 1),
        },
        "corpus": {
            "n_files": corpus["n_files"],
            "total_subjects": corpus["total_subjects"],
            "total_sentences": corpus["total_sentences"],
            "total_sentences_millions": round(
                corpus["total_sentences"] / 1e6, 3),
            "empty_text_sentences": corpus["empty_text"],
            "malformed_blocks": corpus["malformed_blocks"],
            "sentence_length_words": corpus["sent_len_words"].summary(),
            "sentence_length_chars": corpus["sent_len_chars"].summary(),
            "sentences_per_subject": corpus["sents_per_subject"].summary(),
            "sentences_per_post": corpus["sents_per_post"].summary(),
            "posts_per_subject": corpus["posts_per_subject"].summary(),
        },
        "qrels": {
            "majority": {
                "n_rows": len(maj_rows),
                "n_unique_doc_ids": len(maj_judged),
                "total_judged": sum(maj_per_sym[s]["judged"] for s in range(1, 19)),
                "total_relevant": sum(maj_per_sym[s]["relevant"]
                                      for s in range(1, 19)),
                "per_symptom": {str(s): maj_per_sym[s] for s in range(1, 19)},
            },
            "unanimity": {
                "n_rows": len(una_rows),
                "n_unique_doc_ids": len(una_judged),
                "total_judged": sum(una_per_sym[s]["judged"] for s in range(1, 19)),
                "total_relevant": sum(una_per_sym[s]["relevant"]
                                      for s in range(1, 19)),
                "per_symptom": {str(s): una_per_sym[s] for s in range(1, 19)},
            },
            "highlights": highlights,
        },
        "agreement_majority_vs_unanimity": {
            "per_symptom": {str(s): agree[s] for s in range(1, 19)},
            "overall": agree_overall,
        },
        "text_length_comparison": text_stats,
        "missed_symptoms_retrieval": {
            "sids": sorted(MISSED_SYMPTOMS),
            "note": ("Symptoms 7/9/10 had zero relevant docs retrieved in our "
                     "runs' top-1000 (docs/task3_results_analysis.md §5)."),
            "majority_relevant_in_missed": {
                str(s): maj_per_sym[s]["relevant"] for s in sorted(MISSED_SYMPTOMS)
            },
            "share_of_total_majority_relevant": round(
                sum(maj_per_sym[s]["relevant"] for s in MISSED_SYMPTOMS)
                / max(1, sum(maj_per_sym[s]["relevant"] for s in range(1, 19))),
                4),
        },
        "figures": figs,
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[json] wrote {OUT_JSON}")

    # console summary
    print("\n===== SUMMARY =====")
    print(f"subjects (files)      : {corpus['total_subjects']}")
    print(f"total sentences       : {corpus['total_sentences']:,} "
          f"({corpus['total_sentences']/1e6:.3f}M)")
    print(f"sent length (words)   : mean {corpus['sent_len_words'].mean:.1f}, "
          f"median~{corpus['sent_len_words'].approx_median():.0f}, "
          f"std {corpus['sent_len_words'].std:.1f}")
    print(f"majority relevant     : {result['qrels']['majority']['total_relevant']}")
    print(f"unanimity relevant    : {result['qrels']['unanimity']['total_relevant']}")
    print(f"overall shrinkage     : {agree_overall['overall_shrinkage_ratio']:.3f}")
    print(f"relevant mean words   : {text_stats['relevant_majority']['mean']:.1f}")
    print(f"judged mean words     : {text_stats['all_judged']['mean']:.1f}")
    print(f"corpus mean words     : {corpus['sent_len_words'].mean:.1f}")
    return result


if __name__ == "__main__":
    main()
