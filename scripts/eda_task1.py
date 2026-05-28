"""Exploratory Data Analysis (EDA) of the eRisk 2026 Task 1 official test set.

The Task 1 golden data consists of 20 LLM personas, each with a BDI-II total
score (0-63) and a list of 4 "key symptoms". This script:

  * normalizes the free-text key symptoms to the 21 canonical BDI-II item names
    via the organizer-provided ``symptom_mappings.json``;
  * computes descriptive statistics on BDI totals, official severity bands,
    key-symptom prevalence, symptom-by-band breakdown, symptom co-occurrence,
    and the mean BDI total per symptom;
  * writes all numbers to ``analysis/eda_task1/eda_task1.json``;
  * renders five figures (dpi=150) to ``analysis/eda_task1/*.png``;
  * is fully re-runnable and verifies every figure was actually written.

Run:  python scripts/eda_task1.py
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend, must precede pyplot import
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_ROOT = (
    REPO_ROOT
    / "data"
    / "eRisk-2026"
    / "eRisk26-datasets-20260519T175618Z-3-001"
    / "eRisk26-datasets"
    / "task1-llms"
    / "golden-data"
)
PATIENTS_PATH = GOLDEN_ROOT / "patients_data.jsonl"
BDI_ITEMS_PATH = GOLDEN_ROOT / "bdi_symptoms_list.json"
MAPPINGS_PATH = GOLDEN_ROOT / "symptom_mappings.json"

OUT_DIR = REPO_ROOT / "analysis" / "eda_task1"
JSON_OUT = OUT_DIR / "eda_task1.json"

# Official eRisk severity bands (inclusive bounds).
BANDS = [
    ("minimal", 0, 9),
    ("mild", 10, 18),
    ("moderate", 19, 29),
    ("severe", 30, 63),
]
BAND_ORDER = [b[0] for b in BANDS]
BAND_THRESHOLDS = [9.5, 18.5, 29.5]  # plot lines between bands


def band_for(score: int) -> str:
    for name, lo, hi in BANDS:
        if lo <= score <= hi:
            return name
    raise ValueError(f"BDI score {score} out of range 0-63")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_personas() -> list[dict]:
    rows = []
    with PATIENTS_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_bdi_items() -> list[str]:
    data = json.loads(BDI_ITEMS_PATH.read_text(encoding="utf-8"))
    return [item["name"] for item in data["items"]]


def build_normalizer(bdi_items: list[str], mappings: dict) -> dict[str, str]:
    """Build a case-insensitive map from any symptom variant -> canonical BDI name.

    For each mapping group, the canonical target is whichever group member (or
    the group key) matches a BDI item name (case-insensitive). Every member and
    the key are mapped to that target. We also seed the map with the BDI items
    themselves so exact matches always resolve.
    """
    bdi_lower = {name.lower(): name for name in bdi_items}
    normalizer: dict[str, str] = {}

    # Seed with canonical BDI item names.
    for name in bdi_items:
        normalizer[name.lower()] = name

    groups = mappings["symptom_mappings"]
    for key, variants in groups.items():
        members = [key] + list(variants)
        # Find the canonical BDI item among the members.
        target = None
        for m in members:
            if m.lower() in bdi_lower:
                target = bdi_lower[m.lower()]
                break
        if target is None:
            # Group maps to no BDI item (e.g. "Social Withdrawal"); skip — its
            # members that ARE BDI items will still resolve via other groups or
            # the seeded exact matches.
            continue
        for m in members:
            normalizer.setdefault(m.lower(), target)
    return normalizer


def normalize_symptom(raw: str, normalizer: dict[str, str]) -> str:
    key = raw.strip().lower()
    if key in normalizer:
        return normalizer[key]
    raise KeyError(f"Could not normalize symptom: {raw!r}")


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    personas_raw = load_personas()
    bdi_items = load_bdi_items()
    mappings = json.loads(MAPPINGS_PATH.read_text(encoding="utf-8"))
    normalizer = build_normalizer(bdi_items, mappings)

    # Normalize each persona's key symptoms.
    personas = []
    for p in personas_raw:
        norm_syms = [normalize_symptom(s, normalizer) for s in p["patient_key_symptoms"]]
        personas.append(
            {
                "patient_id": p["patient_id"],
                "patient_name": p["patient_name"],
                "bdi_score": int(p["bdi_score"]),
                "band": band_for(int(p["bdi_score"])),
                "key_symptoms_raw": list(p["patient_key_symptoms"]),
                "key_symptoms": norm_syms,
            }
        )
    personas.sort(key=lambda x: x["patient_id"])

    scores = np.array([p["bdi_score"] for p in personas], dtype=float)

    # ---- BDI total stats ----
    bdi_stats = {
        "count": int(scores.size),
        "min": float(scores.min()),
        "max": float(scores.max()),
        "mean": round(float(scores.mean()), 4),
        "median": round(float(np.median(scores)), 4),
        "std": round(float(scores.std(ddof=1)), 4),  # sample std
        "sorted_personas": [
            {"patient_id": p["patient_id"], "patient_name": p["patient_name"], "bdi_score": p["bdi_score"]}
            for p in sorted(personas, key=lambda x: x["bdi_score"])
        ],
    }

    # ---- Band distribution ----
    band_counts = {name: 0 for name in BAND_ORDER}
    band_members = {name: [] for name in BAND_ORDER}
    for p in personas:
        band_counts[p["band"]] += 1
        band_members[p["band"]].append(
            {"patient_id": p["patient_id"], "patient_name": p["patient_name"], "bdi_score": p["bdi_score"]}
        )
    band_distribution = {
        "counts": band_counts,
        "members": band_members,
        "n_personas": len(personas),
        "note": (
            "Class balance is uneven: the test set is dominated by minimal/mild/"
            "moderate personas with a small severe class. This imbalance — "
            "especially the small severe band — is a primary source of difficulty "
            "for band-classification metrics (DCHR)."
        ),
    }

    # ---- Key-symptom prevalence (80 slots = 20 personas x 4) ----
    n_slots = sum(len(p["key_symptoms"]) for p in personas)
    sym_counter: Counter[str] = Counter()
    # personas-with-symptom = count of distinct personas having symptom at least once
    sym_persona_presence: dict[str, set[int]] = defaultdict(set)
    for p in personas:
        for s in p["key_symptoms"]:
            sym_counter[s] += 1
            sym_persona_presence[s].add(p["patient_id"])

    # Frequency for every one of the 21 BDI items (0 if never appears).
    symptom_frequency = {item: int(sym_counter.get(item, 0)) for item in bdi_items}
    symptom_persona_count = {item: len(sym_persona_presence.get(item, set())) for item in bdi_items}
    never_appear = [item for item in bdi_items if symptom_frequency[item] == 0]
    appearing = [item for item in bdi_items if symptom_frequency[item] > 0]
    max_freq = max(symptom_frequency.values())
    most_common = [item for item in bdi_items if symptom_frequency[item] == max_freq]

    key_symptom_prevalence = {
        "n_symptom_slots": n_slots,
        "frequency_all_items": symptom_frequency,
        "persona_count_all_items": symptom_persona_count,
        "frequency_sorted": sorted(
            ({"symptom": k, "count": v} for k, v in symptom_frequency.items()),
            key=lambda d: (-d["count"], d["symptom"]),
        ),
        "never_appear": never_appear,
        "most_common_items": most_common,
        "most_common_count": int(max_freq),
        "n_distinct_symptoms_used": len(appearing),
    }

    # ---- Symptom prevalence by severity band ----
    symptom_by_band = {item: {b: 0 for b in BAND_ORDER} for item in bdi_items}
    for p in personas:
        for s in p["key_symptoms"]:
            symptom_by_band[s][p["band"]] += 1
    # Reduce to symptoms that appear at all, keep raw counts.
    symptom_by_band_appearing = {
        item: symptom_by_band[item] for item in appearing
    }

    # ---- Mean BDI total of personas who have each symptom ----
    mean_bdi_per_symptom = {}
    for item in bdi_items:
        pids = sym_persona_presence.get(item, set())
        if pids:
            vals = [p["bdi_score"] for p in personas if p["patient_id"] in pids]
            mean_bdi_per_symptom[item] = {
                "n_personas": len(pids),
                "mean_bdi": round(float(np.mean(vals)), 4),
                "min_bdi": int(min(vals)),
                "max_bdi": int(max(vals)),
            }
    mean_bdi_per_symptom_sorted = sorted(
        ({"symptom": k, **v} for k, v in mean_bdi_per_symptom.items()),
        key=lambda d: -d["mean_bdi"],
    )

    # ---- Co-occurrence matrix (21x21, ordered by BDI item) ----
    idx = {item: i for i, item in enumerate(bdi_items)}
    n = len(bdi_items)
    cooc = np.zeros((n, n), dtype=int)
    for p in personas:
        # Distinct symptoms within a persona (a persona may list a symptom twice,
        # e.g. Daniel's "Loss of interest" x2 -> treat as one for co-occurrence).
        present = sorted(set(p["key_symptoms"]), key=lambda s: idx[s])
        for a in present:
            cooc[idx[a], idx[a]] += 1  # diagonal = persona presence count
            for b in present:
                if idx[b] > idx[a]:
                    cooc[idx[a], idx[b]] += 1
                    cooc[idx[b], idx[a]] += 1
    # Top co-occurring pairs (off-diagonal).
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if cooc[i, j] > 0:
                pairs.append(
                    {"symptom_a": bdi_items[i], "symptom_b": bdi_items[j], "count": int(cooc[i, j])}
                )
    pairs.sort(key=lambda d: (-d["count"], d["symptom_a"], d["symptom_b"]))
    cooccurrence = {
        "items_order": bdi_items,
        "matrix": cooc.tolist(),
        "diagonal_note": "diagonal = number of personas listing that symptom (distinct within persona)",
        "top_pairs": pairs[:20],
    }

    # ---- Per-persona table ----
    per_persona = [
        {
            "patient_id": p["patient_id"],
            "patient_name": p["patient_name"],
            "bdi_score": p["bdi_score"],
            "band": p["band"],
            "key_symptoms": p["key_symptoms"],
            "key_symptoms_raw": p["key_symptoms_raw"],
        }
        for p in personas
    ]

    results = {
        "dataset": "eRisk 2026 Task 1 — official test golden data (20 LLM personas)",
        "source_files": {
            "patients": str(PATIENTS_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
            "bdi_items": str(BDI_ITEMS_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
            "symptom_mappings": str(MAPPINGS_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
        },
        "bands_definition": {name: [lo, hi] for name, lo, hi in BANDS},
        "bdi_items": bdi_items,
        "bdi_total_stats": bdi_stats,
        "band_distribution": band_distribution,
        "key_symptom_prevalence": key_symptom_prevalence,
        "symptom_by_band": symptom_by_band_appearing,
        "mean_bdi_per_symptom": mean_bdi_per_symptom,
        "mean_bdi_per_symptom_sorted": mean_bdi_per_symptom_sorted,
        "cooccurrence": cooccurrence,
        "per_persona": per_persona,
    }

    JSON_OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[json]   wrote {JSON_OUT.relative_to(REPO_ROOT)}")

    # ----------------------------------------------------------------------- #
    # Figures
    # ----------------------------------------------------------------------- #
    band_colors = {
        "minimal": "#4daf4a",
        "mild": "#ffb300",
        "moderate": "#ff7f00",
        "severe": "#e41a1c",
    }
    written = []

    # 1) BDI distribution: sorted bar with band threshold lines.
    fig, ax = plt.subplots(figsize=(11, 5.5))
    sorted_p = sorted(personas, key=lambda x: x["bdi_score"])
    names = [f"{p['patient_name']}\n({p['bdi_score']})" for p in sorted_p]
    vals = [p["bdi_score"] for p in sorted_p]
    cols = [band_colors[p["band"]] for p in sorted_p]
    ax.bar(range(len(vals)), vals, color=cols, edgecolor="black", linewidth=0.5)
    for thr in BAND_THRESHOLDS:
        ax.axhline(thr, color="grey", linestyle="--", linewidth=1)
    for name, lo, hi in BANDS:
        mid = (lo + min(hi, 63)) / 2
        ax.text(len(vals) - 0.4, mid, name, va="center", ha="left",
                fontsize=9, color=band_colors[name], fontweight="bold")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=8)
    ax.set_ylabel("BDI-II total (0-63)")
    ax.set_ylim(0, 63)
    ax.set_title("eRisk 2026 Task 1 — BDI-II totals by persona (sorted), with band thresholds")
    fig.tight_layout()
    f1 = OUT_DIR / "bdi_distribution.png"
    fig.savefig(f1, dpi=150)
    plt.close(fig)
    written.append(f1)
    print(f"[figure] bdi_distribution.png — {len(vals)} personas, "
          f"range {int(min(vals))}-{int(max(vals))}, band thresholds at {BAND_THRESHOLDS}")

    # 2) Band distribution bar.
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = [band_counts[b] for b in BAND_ORDER]
    bars = ax.bar(BAND_ORDER, counts, color=[band_colors[b] for b in BAND_ORDER],
                  edgecolor="black", linewidth=0.5)
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, c + 0.1, str(c), ha="center", va="bottom",
                fontweight="bold")
    ax.set_ylabel("number of personas")
    ax.set_xlabel("official eRisk severity band")
    ax.set_ylim(0, max(counts) + 1.5)
    ax.set_title(f"Persona count per severity band (n={len(personas)})")
    fig.tight_layout()
    f2 = OUT_DIR / "band_distribution.png"
    fig.savefig(f2, dpi=150)
    plt.close(fig)
    written.append(f2)
    print(f"[figure] band_distribution.png — counts {dict(zip(BAND_ORDER, counts))}")

    # 3) Symptom frequency horizontal bar (all 21 items, sorted ascending for plot).
    fig, ax = plt.subplots(figsize=(9, 8))
    items_sorted = sorted(bdi_items, key=lambda it: symptom_frequency[it])
    freqs = [symptom_frequency[it] for it in items_sorted]
    bar_cols = ["#bdbdbd" if f == 0 else "#377eb8" for f in freqs]
    ax.barh(range(len(items_sorted)), freqs, color=bar_cols, edgecolor="black", linewidth=0.4)
    ax.set_yticks(range(len(items_sorted)))
    ax.set_yticklabels(items_sorted, fontsize=9)
    for i, f in enumerate(freqs):
        if f > 0:
            ax.text(f + 0.05, i, str(f), va="center", fontsize=8)
    ax.set_xlabel("times listed as a key symptom (80 slots = 20 personas x 4)")
    ax.set_title("Key-symptom frequency across the 21 BDI-II items\n(grey = never appears)")
    fig.tight_layout()
    f3 = OUT_DIR / "symptom_frequency.png"
    fig.savefig(f3, dpi=150)
    plt.close(fig)
    written.append(f3)
    print(f"[figure] symptom_frequency.png — {len(appearing)}/21 items used; "
          f"top={most_common} (x{max_freq}); never={never_appear}")

    # 4) Co-occurrence heatmap (only appearing symptoms).
    appear_idx = [idx[it] for it in appearing]
    sub = cooc[np.ix_(appear_idx, appear_idx)].astype(float)
    sub_display = sub.copy()
    np.fill_diagonal(sub_display, np.nan)  # de-emphasize diagonal in color scale
    fig, ax = plt.subplots(figsize=(10, 9))
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="#dddddd")
    im = ax.imshow(sub_display, cmap=cmap, aspect="equal")
    ax.set_xticks(range(len(appearing)))
    ax.set_yticks(range(len(appearing)))
    ax.set_xticklabels(appearing, rotation=90, fontsize=8)
    ax.set_yticklabels(appearing, fontsize=8)
    # annotate off-diagonal counts > 0
    maxv = np.nanmax(sub_display) if np.isfinite(np.nanmax(sub_display)) else 0
    for i in range(len(appearing)):
        for j in range(len(appearing)):
            if i != j and sub[i, j] > 0:
                ax.text(j, i, int(sub[i, j]), ha="center", va="center",
                        fontsize=7,
                        color="white" if sub[i, j] < maxv * 0.6 else "black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("# personas sharing both as key symptoms")
    ax.set_title("Symptom co-occurrence (off-diagonal; diagonal hidden)")
    fig.tight_layout()
    f4 = OUT_DIR / "symptom_cooccurrence.png"
    fig.savefig(f4, dpi=150)
    plt.close(fig)
    written.append(f4)
    top_pair = pairs[0] if pairs else None
    print(f"[figure] symptom_cooccurrence.png — {len(appearing)}x{len(appearing)} matrix; "
          f"top pair={top_pair}")

    # 5) Symptom x band heatmap (appearing symptoms, ordered by total frequency desc).
    items_by_freq = sorted(appearing, key=lambda it: -symptom_frequency[it])
    band_mat = np.array(
        [[symptom_by_band[it][b] for b in BAND_ORDER] for it in items_by_freq], dtype=float
    )
    fig, ax = plt.subplots(figsize=(7.5, 9))
    im = ax.imshow(band_mat, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(BAND_ORDER)))
    ax.set_xticklabels(BAND_ORDER)
    ax.set_yticks(range(len(items_by_freq)))
    ax.set_yticklabels(items_by_freq, fontsize=9)
    bmax = band_mat.max() if band_mat.size else 0
    for i in range(len(items_by_freq)):
        for j in range(len(BAND_ORDER)):
            v = int(band_mat[i, j])
            if v > 0:
                ax.text(j, i, v, ha="center", va="center", fontsize=8,
                        color="black" if band_mat[i, j] < bmax * 0.6 else "white")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("# key-symptom occurrences")
    ax.set_xlabel("severity band")
    ax.set_title("Key-symptom frequency by severity band\n(rows ordered by total frequency)")
    fig.tight_layout()
    f5 = OUT_DIR / "symptom_by_band.png"
    fig.savefig(f5, dpi=150)
    plt.close(fig)
    written.append(f5)
    print(f"[figure] symptom_by_band.png — {len(items_by_freq)} symptoms x {len(BAND_ORDER)} bands")

    # Verify all figures written.
    for f in written:
        assert f.exists() and f.stat().st_size > 0, f"figure not written: {f}"
    print(f"\nAll {len(written)} figures verified on disk. JSON + figures in "
          f"{OUT_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
