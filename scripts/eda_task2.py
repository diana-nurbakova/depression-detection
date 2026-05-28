"""Exploratory Data Analysis for eRisk 2026 Task 2 (contextualized early depression detection) official test set.

Streams the 500 released round files one at a time, aggregating per-subject statistics,
compares POSITIVE vs CONTROL groups, and emits:
  - analysis/eda_task2/eda_task2.json   (all computed statistics)
  - analysis/eda_task2/*.png            (figures, dpi=150)
  - docs/task2_test_eda.md is written separately by hand; this script only produces data + figures.

Re-runnable and self-contained. Does NOT hold all rounds in memory.
"""

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = (
    ROOT
    / "data/eRisk-2026/eRisk26-datasets-20260519T175618Z-3-001/eRisk26-datasets/"
    "task2-contextualized-depression/golden-data/risk_golden_truth_t2_2026.txt"
)
ROUNDS_DIR = ROOT / "runs/task2/train/server_responses"
OUT_DIR = ROOT / "analysis/eda_task2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WORD_RE = re.compile(r"[a-zA-Z']+")
FIRST_PERSON = {"i", "me", "my", "mine", "myself", "im", "ive", "id", "ill"}
# small negative-emotion lexicon (cheap proxy)
NEG_EMO = {
    "sad", "depressed", "depression", "anxious", "anxiety", "lonely", "alone", "hopeless",
    "worthless", "tired", "exhausted", "cry", "crying", "hurt", "pain", "fear", "afraid",
    "scared", "angry", "hate", "hates", "hated", "stress", "stressed", "worried", "worry",
    "guilt", "guilty", "ashamed", "empty", "numb", "miserable", "suicidal", "die", "death",
    "kill", "fail", "failure", "broken", "lost", "useless", "panic", "overwhelmed",
}

STOPWORDS = set(
    """a about above after again against all am an and any are aren't as at be because been before
being below between both but by can't cannot could couldn't did didn't do does doesn't doing don't
down during each few for from further had hadn't has hasn't have haven't having he he'd he'll he's
her here here's hers herself him himself his how how's i i'd i'll i'm i've if in into is isn't it
it's its itself let's me more most mustn't my myself no nor not of off on once only or other ought
our ours ourselves out over own same shan't she she'd she'll she's should shouldn't so some such than
that that's the their theirs them themselves then there there's these they they'd they'll they're
they've this those through to too under until up very was wasn't we we'd we'll we're we've were
weren't what what's when when's where where's which while who who's whom why why's with won't would
wouldn't you you'd you'll you're you've your yours yourself yourselves
just like get got would also one really even much make made way still going go im ive dont cant
thats youre theyre well think know want need see say said us yeah yes ok okay oh thing things lot
people time day good bad new now back come came take took give gave find found feel felt let put
something someone anything everything nothing every always never sometimes maybe probably actually
""".split()
)


def load_golden():
    labels = {}
    with open(GOLDEN, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            labels[parts[0]] = int(parts[1])
    return labels


def tokenize(text):
    return [w.lower() for w in WORD_RE.findall(text or "")]


def content_words(text):
    return [w for w in tokenize(text) if w not in STOPWORDS and len(w) > 2]


def main():
    labels = load_golden()
    n_pos = sum(1 for v in labels.values() if v == 1)
    n_ctrl = sum(1 for v in labels.values() if v == 0)
    n_total = len(labels)
    print(f"Golden: total={n_total} pos={n_pos} ctrl={n_ctrl}")

    round_files = sorted(ROUNDS_DIR.glob("round_*.json"))
    print(f"Found {len(round_files)} round files")

    # Per-subject aggregates
    subj = defaultdict(
        lambda: {
            "n_threads": 0,            # threads where this subject is the target
            "n_target_submissions": 0,  # submissions authored by target
            "n_target_comments": 0,     # comments authored by target
            "n_context_submissions": 0, # submissions in target's threads authored by others
            "n_context_comments": 0,    # comments authored by others
            "n_comments_total": 0,      # all comments across target's threads
            "n_context_authors": set(), # distinct other contributors
            "target_chars": 0,
            "target_words": 0,
            "n_target_titles": 0,
            "first_date": None,
            "last_date": None,
        }
    )

    # Streaming-dynamics aggregates
    active_subjects_per_round = []      # number of target subjects present in round r
    target_writings_per_round = []      # total target-authored writings (subs+comments) in round r

    # Text-length samples (per target-authored writing) split by label
    len_words = {0: [], 1: []}
    len_chars = {0: [], 1: []}

    # Vocabulary counters (target-authored text) by label
    vocab = {0: Counter(), 1: Counter()}
    total_tokens = {0: 0, 1: 0}          # all content tokens (denominator for log-odds)
    first_person_hits = {0: 0, 1: 0}
    neg_emo_hits = {0: 0, 1: 0}
    all_tokens_for_lex = {0: 0, 1: 0}    # all tokens (incl stopwords) for pronoun/neg-emo rate

    # comments-per-thread distribution and thread depth (distinct contributors per thread)
    comments_per_thread = []
    thread_depth = []  # distinct context authors per thread

    DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})T(\d{2}):(\d{2}):(\d{2})")

    def parse_dt(s):
        # returns float seconds-ish ordinal for span computation; cheap, no tz handling needed
        m = DATE_RE.match(s or "")
        if not m:
            return None
        y, mo, d = int(s[0:4]), int(s[5:7]), int(s[8:10])
        hh, mm, ss = int(m.group(2)), int(m.group(3)), int(m.group(4))
        # days since epoch-ish (proleptic, good enough for spans)
        import datetime

        try:
            return datetime.datetime(y, mo, d, hh, mm, ss).timestamp()
        except Exception:
            return None

    global_min_date = None
    global_max_date = None

    for ridx, fp in enumerate(round_files):
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)

        round_active = set()
        round_target_writings = 0

        for thread in data:
            tgt = thread.get("targetSubject")
            if tgt is None:
                continue
            round_active.add(tgt)
            s = subj[tgt]
            s["n_threads"] += 1

            sub_author = thread.get("author")
            body = thread.get("body") or ""
            title = thread.get("title") or ""
            sdate = thread.get("date")

            # submission-level
            if sub_author == tgt:
                s["n_target_submissions"] += 1
                round_target_writings += 1
                txt = (title + " " + body).strip()
                w = tokenize(txt)
                if w:
                    lbl = labels.get(tgt)
                    if lbl is not None:
                        len_words[lbl].append(len(w))
                        len_chars[lbl].append(len(txt))
                        cw = [x for x in w if x not in STOPWORDS and len(x) > 2]
                        vocab[lbl].update(cw)
                        total_tokens[lbl] += len(cw)
                        all_tokens_for_lex[lbl] += len(w)
                        first_person_hits[lbl] += sum(1 for x in w if x in FIRST_PERSON)
                        neg_emo_hits[lbl] += sum(1 for x in w if x in NEG_EMO)
                s["target_chars"] += len(txt)
                s["target_words"] += len(w)
                if title:
                    s["n_target_titles"] += 1
            else:
                s["n_context_submissions"] += 1
                if sub_author:
                    s["n_context_authors"].add(sub_author)

            # dates (submission)
            dt = parse_dt(sdate)
            if dt is not None:
                if s["first_date"] is None or dt < s["first_date"]:
                    s["first_date"] = dt
                if s["last_date"] is None or dt > s["last_date"]:
                    s["last_date"] = dt
                if global_min_date is None or dt < global_min_date:
                    global_min_date = dt
                if global_max_date is None or dt > global_max_date:
                    global_max_date = dt

            # comments
            comments = thread.get("comments") or []
            comments_per_thread.append(len(comments))
            depth_authors = set()
            for c in comments:
                cauthor = c.get("author")
                cbody = c.get("body") or ""
                s["n_comments_total"] += 1
                cdt = parse_dt(c.get("date"))
                if cdt is not None:
                    if s["first_date"] is None or cdt < s["first_date"]:
                        s["first_date"] = cdt
                    if s["last_date"] is None or cdt > s["last_date"]:
                        s["last_date"] = cdt
                    if global_min_date is None or cdt < global_min_date:
                        global_min_date = cdt
                    if global_max_date is None or cdt > global_max_date:
                        global_max_date = cdt
                if cauthor == tgt:
                    s["n_target_comments"] += 1
                    round_target_writings += 1
                    w = tokenize(cbody)
                    if w:
                        lbl = labels.get(tgt)
                        if lbl is not None:
                            len_words[lbl].append(len(w))
                            len_chars[lbl].append(len(cbody))
                            cw = [x for x in w if x not in STOPWORDS and len(x) > 2]
                            vocab[lbl].update(cw)
                            total_tokens[lbl] += len(cw)
                            all_tokens_for_lex[lbl] += len(w)
                            first_person_hits[lbl] += sum(1 for x in w if x in FIRST_PERSON)
                            neg_emo_hits[lbl] += sum(1 for x in w if x in NEG_EMO)
                    s["target_chars"] += len(cbody)
                    s["target_words"] += len(w)
                else:
                    s["n_context_comments"] += 1
                    if cauthor and cauthor != tgt:
                        depth_authors.add(cauthor)
                        s["n_context_authors"].add(cauthor)
            thread_depth.append(len(depth_authors))

        active_subjects_per_round.append(len(round_active))
        target_writings_per_round.append(round_target_writings)

        if ridx % 100 == 0:
            print(f"  processed round {ridx}/{len(round_files)}  active={len(round_active)}")

    print(f"Done streaming. Subjects seen in rounds: {len(subj)}")

    # ---- Overlap check ----
    gold_ids = set(labels.keys())
    seen_ids = set(subj.keys())
    overlap = gold_ids & seen_ids
    only_gold = sorted(gold_ids - seen_ids)
    only_seen = sorted(seen_ids - gold_ids)
    print(f"Overlap golden&seen: {len(overlap)}; only in golden: {only_gold}; only in rounds: {only_seen}")

    # ---- Per-subject summary split by label ----
    def summarize(values):
        if not values:
            return {"n": 0, "min": None, "max": None, "mean": None, "median": None, "std": None}
        arr = np.array(values, dtype=float)
        return {
            "n": int(arr.size),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "std": float(arr.std()),
        }

    metrics = [
        "n_threads",
        "n_target_submissions",
        "n_target_comments",
        "n_target_writings",  # derived
        "n_context_submissions",
        "n_context_comments",
        "n_comments_total",
        "n_context_authors",
        "target_words",
        "target_chars",
        "span_days",  # derived
    ]
    by_label_vals = {0: defaultdict(list), 1: defaultdict(list)}
    writings_per_subject = {0: [], 1: []}
    span_days_list = {0: [], 1: []}

    for sid, s in subj.items():
        lbl = labels.get(sid)
        if lbl is None:
            continue
        tw = s["n_target_submissions"] + s["n_target_comments"]
        span = None
        if s["first_date"] is not None and s["last_date"] is not None:
            span = (s["last_date"] - s["first_date"]) / 86400.0
            span_days_list[lbl].append(span)
        rec = {
            "n_threads": s["n_threads"],
            "n_target_submissions": s["n_target_submissions"],
            "n_target_comments": s["n_target_comments"],
            "n_target_writings": tw,
            "n_context_submissions": s["n_context_submissions"],
            "n_context_comments": s["n_context_comments"],
            "n_comments_total": s["n_comments_total"],
            "n_context_authors": len(s["n_context_authors"]),
            "target_words": s["target_words"],
            "target_chars": s["target_chars"],
            "span_days": span if span is not None else 0.0,
        }
        for m in metrics:
            by_label_vals[lbl][m].append(rec[m])
        writings_per_subject[lbl].append(tw)

    per_subject_summary = {
        "positive": {m: summarize(by_label_vals[1][m]) for m in metrics},
        "control": {m: summarize(by_label_vals[0][m]) for m in metrics},
        "all": {m: summarize(by_label_vals[0][m] + by_label_vals[1][m]) for m in metrics},
    }

    # ---- Text length summary ----
    text_len_summary = {
        "words_per_writing": {
            "positive": summarize(len_words[1]),
            "control": summarize(len_words[0]),
        },
        "chars_per_writing": {
            "positive": summarize(len_chars[1]),
            "control": summarize(len_chars[0]),
        },
    }

    # ---- Distinctive terms via log-odds (with +0.5 smoothing) ----
    def distinctive(target_lbl, other_lbl, top=30, min_count=10):
        ct = vocab[target_lbl]
        co = vocab[other_lbl]
        nt = total_tokens[target_lbl] or 1
        no = total_tokens[other_lbl] or 1
        scores = []
        candidate = set(w for w, c in ct.items() if c >= min_count) | set(
            w for w, c in co.items() if c >= min_count
        )
        for w in candidate:
            a = ct.get(w, 0) + 0.5
            b = co.get(w, 0) + 0.5
            lo = math.log((a / nt) / (b / no))
            scores.append((w, lo, ct.get(w, 0), co.get(w, 0)))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top]

    distinctive_pos = distinctive(1, 0)
    distinctive_ctrl = distinctive(0, 1)

    vocab_summary = {
        "top_content_words_positive": [[w, c] for w, c in vocab[1].most_common(30)],
        "top_content_words_control": [[w, c] for w, c in vocab[0].most_common(30)],
        "distinctive_positive_logodds": [
            {"term": w, "log_odds": round(lo, 4), "count_pos": cp, "count_ctrl": cc}
            for w, lo, cp, cc in distinctive_pos
        ],
        "distinctive_control_logodds": [
            {"term": w, "log_odds": round(lo, 4), "count_ctrl": cp, "count_pos": cc}
            for w, lo, cp, cc in distinctive_ctrl
        ],
    }

    # ---- Linguistic rates ----
    lex_summary = {
        "first_person_rate_per_token": {
            "positive": first_person_hits[1] / (all_tokens_for_lex[1] or 1),
            "control": first_person_hits[0] / (all_tokens_for_lex[0] or 1),
        },
        "neg_emotion_rate_per_token": {
            "positive": neg_emo_hits[1] / (all_tokens_for_lex[1] or 1),
            "control": neg_emo_hits[0] / (all_tokens_for_lex[0] or 1),
        },
        "first_person_hits": {"positive": first_person_hits[1], "control": first_person_hits[0]},
        "neg_emotion_hits": {"positive": neg_emo_hits[1], "control": neg_emo_hits[0]},
        "total_tokens_for_lex": {"positive": all_tokens_for_lex[1], "control": all_tokens_for_lex[0]},
    }

    # ---- Temporal ----
    import datetime

    def to_iso(ts):
        if ts is None:
            return None
        return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")

    temporal_summary = {
        "global_min_date": to_iso(global_min_date),
        "global_max_date": to_iso(global_max_date),
        "global_span_days": (global_max_date - global_min_date) / 86400.0
        if global_min_date and global_max_date
        else None,
        "span_days_per_subject": {
            "positive": summarize(span_days_list[1]),
            "control": summarize(span_days_list[0]),
        },
    }

    # ---- Comments / depth ----
    comments_summary = {
        "comments_per_thread": summarize(comments_per_thread),
        "thread_depth_distinct_context_authors": summarize(thread_depth),
    }

    # ---- Streaming dynamics ----
    streaming_summary = {
        "n_rounds": len(round_files),
        "active_subjects_per_round": active_subjects_per_round,
        "target_writings_per_round": target_writings_per_round,
        "max_active": max(active_subjects_per_round) if active_subjects_per_round else 0,
        "min_active": min(active_subjects_per_round) if active_subjects_per_round else 0,
        "active_round_0": active_subjects_per_round[0] if active_subjects_per_round else 0,
        "active_round_last": active_subjects_per_round[-1] if active_subjects_per_round else 0,
    }

    results = {
        "label_balance": {
            "total": n_total,
            "positive": n_pos,
            "control": n_ctrl,
            "positive_rate": n_pos / n_total,
        },
        "overlap_check": {
            "golden_subjects": len(gold_ids),
            "subjects_seen_in_rounds": len(seen_ids),
            "overlap": len(overlap),
            "only_in_golden": only_gold,
            "only_in_rounds": only_seen,
            "note": (
                "server_responses folder is named 'train' but its targetSubject set matches "
                "the 2026 golden test subjects (522/523). subject_sOPw3Ku is in golden but "
                "never receives a released thread (522-vs-523 off-by-one)."
            ),
        },
        "per_subject_activity": per_subject_summary,
        "text_length": text_len_summary,
        "vocabulary": vocab_summary,
        "linguistic": lex_summary,
        "temporal": temporal_summary,
        "comments_and_depth": comments_summary,
        "streaming_dynamics": streaming_summary,
    }

    out_json = OUT_DIR / "eda_task2.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out_json}")

    # ================= FIGURES =================
    POS_C = "#c0392b"
    CTRL_C = "#2980b9"

    # 1. label balance
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Positive", "Control"], [n_pos, n_ctrl], color=[POS_C, CTRL_C])
    for i, v in enumerate([n_pos, n_ctrl]):
        ax.text(i, v + 3, str(v), ha="center", fontweight="bold")
    ax.set_ylabel("Subjects")
    ax.set_title(f"Task 2 test label balance (n={n_total}, {n_pos/n_total:.1%} positive)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "label_balance.png", dpi=150)
    plt.close(fig)

    # 2. writings per subject (box + strip)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    data_box = [writings_per_subject[1], writings_per_subject[0]]
    bp = ax.boxplot(data_box, labels=["Positive", "Control"], patch_artist=True, showfliers=True)
    for patch, c in zip(bp["boxes"], [POS_C, CTRL_C]):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax.set_ylabel("Target-authored writings per subject")
    ax.set_title("Target writings per subject by label")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "writings_per_subject.png", dpi=150)
    plt.close(fig)

    # 3. text length distribution (words per writing) — capped for readability
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cap = 200
    pos_w = [min(x, cap) for x in len_words[1]]
    ctrl_w = [min(x, cap) for x in len_words[0]]
    bins = np.linspace(0, cap, 41)
    ax.hist(ctrl_w, bins=bins, density=True, alpha=0.55, color=CTRL_C, label=f"Control (n={len(ctrl_w)})")
    ax.hist(pos_w, bins=bins, density=True, alpha=0.55, color=POS_C, label=f"Positive (n={len(pos_w)})")
    ax.set_xlabel(f"Words per target writing (capped at {cap})")
    ax.set_ylabel("Density")
    ax.set_title("Text length distribution (words per target-authored writing)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "text_length_dist.png", dpi=150)
    plt.close(fig)

    # 4. activity over rounds
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    x = list(range(len(active_subjects_per_round)))
    ax1.plot(x, active_subjects_per_round, color="#27ae60", label="Active subjects")
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Active subjects", color="#27ae60")
    ax1.tick_params(axis="y", labelcolor="#27ae60")
    ax2 = ax1.twinx()
    ax2.plot(x, target_writings_per_round, color="#8e44ad", alpha=0.7, label="Target writings")
    ax2.set_ylabel("Target-authored writings", color="#8e44ad")
    ax2.tick_params(axis="y", labelcolor="#8e44ad")
    ax1.set_title("Streaming activity over rounds (coverage decay)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "activity_over_rounds.png", dpi=150)
    plt.close(fig)

    # 5. comments per thread
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cap2 = 60
    cpt = [min(x, cap2) for x in comments_per_thread]
    ax.hist(cpt, bins=np.linspace(0, cap2, 31), color="#16a085", alpha=0.8)
    ax.set_xlabel(f"Comments per thread (capped at {cap2})")
    ax.set_ylabel("Threads")
    md = np.median(comments_per_thread) if comments_per_thread else 0
    ax.axvline(md, color="black", ls="--", label=f"median={md:.0f}")
    ax.set_title("Comments per thread distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "comments_per_thread.png", dpi=150)
    plt.close(fig)

    # 6. distinctive terms (positives, and controls)
    fig, axes = plt.subplots(1, 2, figsize=(11, 7))
    top_n = 20
    dp = distinctive_pos[:top_n][::-1]
    axes[0].barh([w for w, *_ in dp], [lo for _, lo, *_ in dp], color=POS_C)
    axes[0].set_title("Distinctive terms — POSITIVE")
    axes[0].set_xlabel("log-odds (pos vs control)")
    dc = distinctive_ctrl[:top_n][::-1]
    axes[1].barh([w for w, *_ in dc], [lo for _, lo, *_ in dc], color=CTRL_C)
    axes[1].set_title("Distinctive terms — CONTROL")
    axes[1].set_xlabel("log-odds (control vs pos)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "distinctive_terms.png", dpi=150)
    plt.close(fig)

    # verify figures
    expected = [
        "label_balance.png",
        "writings_per_subject.png",
        "text_length_dist.png",
        "activity_over_rounds.png",
        "comments_per_thread.png",
        "distinctive_terms.png",
    ]
    for fn in expected:
        p = OUT_DIR / fn
        assert p.exists() and p.stat().st_size > 0, f"figure missing/empty: {p}"
        print(f"  figure OK: {fn} ({p.stat().st_size} bytes)")

    print("EDA complete.")


if __name__ == "__main__":
    main()
