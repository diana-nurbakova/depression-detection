"""Select demonstration-example candidates for the Task 2 eRisk write-up.

Three pre-committed selection rules, applied to the official eRisk-2026 Task 2
*test* data (523 subjects / 91 positives) using the live-run artefacts in
``runs/task2/train/`` and the golden labels in
``data/eRisk-2026/.../golden-data/risk_golden_truth_t2_2026.txt``.

  1. **Intro community-signal example.** Among Run-0 true-positive users, the
     subset with (target words/round below the TP median) AND (concern-phrase
     density above the TP median).  Pick the member nearest the subset
     centroid in z-scored feature space.
  2. **Multi-scale Wasserstein example.** A user whose mean-symptom W1
     trajectory crosses the running mu+2sigma threshold at one scale
     (short / long) but not the other.
  3. **Early-alert example.** Subjects (true positives) where Run 1 alerts
     >=5 rounds earlier than Run 0.  Pick the case nearest the median gap.

Output: ``analysis/eda_task2/outputs/example_candidates.json``.
"""

from __future__ import annotations

import glob
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from erisk_task2.distances.wasserstein import compute_wasserstein_1d  # noqa: E402

CHECKPOINT = ROOT / "runs" / "task2" / "train" / "checkpoint" / "round_0500_state.pkl"
DECISIONS_DIR = ROOT / "runs" / "task2" / "train" / "decisions"
GOLDEN = (
    ROOT
    / "data"
    / "eRisk-2026"
    / "eRisk26-datasets-20260519T175618Z-3-001"
    / "eRisk26-datasets"
    / "task2-contextualized-depression"
    / "golden-data"
    / "risk_golden_truth_t2_2026.txt"
)
OUT_DIR = ROOT / "analysis" / "eda_task2" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# I/O


def load_gold() -> dict[str, int]:
    gold: dict[str, int] = {}
    with open(GOLDEN, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                gold[parts[0]] = int(parts[1])
    return gold


def load_profiles() -> dict[str, object]:
    with open(CHECKPOINT, "rb") as f:
        state = pickle.load(f)
    return state["profiles"]


def load_first_alerts(run_id: int) -> dict[str, int]:
    """For each subject, the (first) round index where decision==1 in run_id."""
    paths = sorted(DECISIONS_DIR.glob(f"run_{run_id}_round_*.json"))
    first: dict[str, int] = {}
    for p in paths:
        # round index from filename: run_0_round_0007.json -> 7
        round_idx = int(p.stem.split("_")[-1])
        with open(p, "r", encoding="utf-8") as f:
            recs = json.load(f)
        for r in recs:
            if r.get("decision") == 1 and r["nick"] not in first:
                first[r["nick"]] = round_idx
    return first


# ---------------------------------------------------------------------------
# Per-user features


def per_user_features(profiles: dict[str, object]) -> dict[str, dict]:
    """target_words_per_round and concern_density per subject."""
    feats: dict[str, dict] = {}
    for uid, p in profiles.items():
        word_counts = list(getattr(p, "text_word_counts", []) or [])
        concern = list(getattr(p, "concern_flags", []) or [])
        target_rounds = len(word_counts)
        if target_rounds == 0:
            twpr = 0.0
        else:
            twpr = float(np.mean(word_counts))
        if len(concern) == 0:
            cden = 0.0
        else:
            cden = float(np.mean(concern))
        feats[uid] = {
            "target_words_per_round": twpr,
            "concern_density": cden,
            "rounds_with_target": int(target_rounds),
            "rounds_with_concern": int(sum(1 for c in concern if c)),
        }
    return feats


# ---------------------------------------------------------------------------
# Wasserstein trajectories


def w1_trajectory_meanover_symptoms(
    activations: list[np.ndarray],
    n: int,
    *,
    short_window: int = 5,
    medium_window: int = 25,
) -> dict[str, list[float]]:
    """For a per-round (21,) activation series, build *running* W1 trajectories
    at three scales (short / medium / long), aggregated across symptoms by mean.

    For each round k>=warmup we compute the 3-scale 63-d W1 vector of
    `compute_symptom_wasserstein` (well, its underlying 1D distance per symptom)
    and average across symptoms to produce a single scalar per scale.
    """
    if not activations:
        return {"short": [], "medium": [], "long": []}
    arr = np.stack(activations)  # (n, 21)
    short_traj: list[float] = []
    med_traj: list[float] = []
    long_traj: list[float] = []
    for k in range(1, len(arr) + 1):
        # short: half=5 -> needs k>=10
        if k >= 10:
            half = short_window
            early = arr[max(0, k - 2 * half) : k - half]
            recent = arr[k - half : k]
            vals = [
                compute_wasserstein_1d(early[:, s], recent[:, s]) for s in range(21)
            ]
            short_traj.append(float(np.mean(vals)))
        else:
            short_traj.append(float("nan"))
        # medium: half=25 -> needs k>=50
        if k >= 50:
            half = medium_window
            early = arr[max(0, k - 2 * half) : k - half]
            recent = arr[k - half : k]
            vals = [
                compute_wasserstein_1d(early[:, s], recent[:, s]) for s in range(21)
            ]
            med_traj.append(float(np.mean(vals)))
        else:
            med_traj.append(float("nan"))
        # long: split-at-mid -> needs k>=20
        if k >= 20:
            mid = k // 2
            early = arr[:mid]
            recent = arr[mid:k]
            vals = [
                compute_wasserstein_1d(early[:, s], recent[:, s]) for s in range(21)
            ]
            long_traj.append(float(np.mean(vals)))
        else:
            long_traj.append(float("nan"))
    return {"short": short_traj, "medium": med_traj, "long": long_traj}


def running_threshold_fires(traj: list[float]) -> dict:
    """Apply the trial-style running mu+2*sigma threshold.

    Skips warmup (the first three non-nan rounds, mirroring the trial spec).
    Returns the rounds that fired and the max overshoot.
    """
    fired_rounds: list[int] = []
    max_overshoot = float("-inf")
    past: list[float] = []
    for k, v in enumerate(traj):
        if np.isnan(v):
            continue
        # build threshold from prior non-nan w1's
        if len(past) >= 2:  # need at least 2 prior points (matches trial spec warmup)
            mu = float(np.mean(past))
            sigma = float(np.std(past))
            thr = mu + 2 * sigma
            if v > thr and len(past) >= 3:  # k<3 in trial spec is warmup
                fired_rounds.append(k)
                max_overshoot = max(max_overshoot, v - thr)
        past.append(v)
    return {
        "fired": len(fired_rounds) > 0,
        "fired_rounds": fired_rounds[:20],
        "n_fires": len(fired_rounds),
        "max_overshoot": max_overshoot if max_overshoot > float("-inf") else None,
    }


# ---------------------------------------------------------------------------
# Selection


def select_intro_signal(
    feats: dict[str, dict],
    gold: dict[str, int],
    run0_alerts: dict[str, int],
) -> dict:
    """Rule 1: Run-0 TPs with (twpr < median) AND (concern_density > median);
    pick member nearest subset centroid in z-scored space."""
    tp_ids = [uid for uid, lbl in gold.items() if lbl == 1 and uid in run0_alerts]
    rows = [(uid, feats[uid]["target_words_per_round"], feats[uid]["concern_density"])
            for uid in tp_ids if uid in feats]
    if not rows:
        return {"error": "no Run-0 TPs found in profiles"}
    twprs = np.array([r[1] for r in rows])
    cdens = np.array([r[2] for r in rows])
    med_t = float(np.median(twprs))
    med_c = float(np.median(cdens))
    mask = (twprs < med_t) & (cdens > med_c)
    if mask.sum() == 0:
        return {"error": "empty subset", "median_twpr": med_t, "median_cden": med_c}
    sub_ids = [rows[i][0] for i in np.where(mask)[0]]
    sub_t = twprs[mask]
    sub_c = cdens[mask]
    # z-score within the subset
    sub_t_z = (sub_t - sub_t.mean()) / (sub_t.std() + 1e-9)
    sub_c_z = (sub_c - sub_c.mean()) / (sub_c.std() + 1e-9)
    dists = np.sqrt(sub_t_z**2 + sub_c_z**2)
    pick_idx = int(np.argmin(dists))
    chosen = sub_ids[pick_idx]
    return {
        "rule": "Run-0 TPs, target_words_per_round < TP median AND concern_density > TP median; nearest z-scored centroid.",
        "tp_cohort_size": int(len(rows)),
        "subset_size": int(mask.sum()),
        "tp_median_target_words_per_round": med_t,
        "tp_median_concern_density": med_c,
        "chosen": {
            "subject_id": chosen,
            "target_words_per_round": float(sub_t[pick_idx]),
            "concern_density": float(sub_c[pick_idx]),
            "rounds_with_target": feats[chosen]["rounds_with_target"],
            "rounds_with_concern": feats[chosen]["rounds_with_concern"],
            "first_alert_round_run0": run0_alerts.get(chosen),
        },
        "subset_members": [
            {
                "subject_id": uid,
                "target_words_per_round": float(t),
                "concern_density": float(c),
            }
            for uid, t, c in sorted(
                zip(sub_ids, sub_t, sub_c), key=lambda x: -x[2]
            )[:10]
        ],
    }


def select_w1_scale_split(
    profiles: dict[str, object],
    gold: dict[str, int],
    run0_alerts: dict[str, int],
) -> dict:
    """Rule 2: a user whose short-scale W1 fires the running mu+2sigma threshold
    but whose long-scale W1 does not (or vice versa). Prefer TPs."""
    candidates_short_only = []
    candidates_long_only = []
    for uid, p in profiles.items():
        acts = getattr(p, "symptom_activations", None)
        if not acts or len(acts) < 50:
            continue
        traj = w1_trajectory_meanover_symptoms(list(acts), len(acts))
        s = running_threshold_fires(traj["short"])
        l = running_threshold_fires(traj["long"])
        if s["fired"] and not l["fired"]:
            candidates_short_only.append((uid, s, l))
        elif l["fired"] and not s["fired"]:
            candidates_long_only.append((uid, s, l))
    # Prefer TPs (gold==1 and Run-0 alerted), score by max overshoot
    def best(cands, label):
        tp = [c for c in cands if gold.get(c[0]) == 1 and c[0] in run0_alerts]
        cands = tp if tp else cands
        if not cands:
            return None
        cands.sort(key=lambda c: -(c[1]["max_overshoot"] or 0)
                   if label == "short" else -(c[2]["max_overshoot"] or 0))
        uid, sres, lres = cands[0]
        return {
            "subject_id": uid,
            "label": label,
            "is_true_positive": gold.get(uid) == 1 and uid in run0_alerts,
            "gold": gold.get(uid),
            "first_alert_round_run0": run0_alerts.get(uid),
            "short_scale": sres,
            "long_scale": lres,
        }

    return {
        "rule": "User where mean-symptom W1 trajectory fires running mu+2sigma at exactly one scale (short or long), not the other. Prefer Run-0 TPs.",
        "counts": {
            "short_only": len(candidates_short_only),
            "long_only": len(candidates_long_only),
        },
        "chosen_short_only": best(candidates_short_only, "short"),
        "chosen_long_only": best(candidates_long_only, "long"),
    }


def select_early_alert(
    run0: dict[str, int],
    run1: dict[str, int],
    gold: dict[str, int],
    min_gap: int = 5,
) -> dict:
    """Rule 3: Run 1 alerts >= min_gap rounds before Run 0. TPs only. Pick median gap."""
    pairs = []  # (uid, gap, r0, r1)
    for uid, lbl in gold.items():
        if lbl != 1:
            continue
        if uid not in run0 or uid not in run1:
            continue
        gap = run0[uid] - run1[uid]
        if gap >= min_gap:
            pairs.append((uid, gap, run0[uid], run1[uid]))
    if not pairs:
        return {"error": "no TPs with gap>=5"}
    gaps = np.array([p[1] for p in pairs])
    median_gap = float(np.median(gaps))
    # nearest to median
    pairs_sorted = sorted(pairs, key=lambda p: (abs(p[1] - median_gap), p[1]))
    uid, gap, r0, r1 = pairs_sorted[0]
    return {
        "rule": "TPs (gold==1) where Run 1 alerts >=5 rounds before Run 0; pick member with gap nearest the median.",
        "cohort_size": int(len(pairs)),
        "median_gap": median_gap,
        "mean_gap": float(np.mean(gaps)),
        "gap_distribution": {
            "min": int(gaps.min()),
            "p25": float(np.percentile(gaps, 25)),
            "p50": median_gap,
            "p75": float(np.percentile(gaps, 75)),
            "max": int(gaps.max()),
        },
        "chosen": {
            "subject_id": uid,
            "gap": int(gap),
            "first_alert_round_run0": int(r0),
            "first_alert_round_run1": int(r1),
        },
        "all_qualifying": [
            {"subject_id": u, "gap": int(g), "run0_round": int(a), "run1_round": int(b)}
            for u, g, a, b in sorted(pairs, key=lambda p: -p[1])
        ],
    }


# ---------------------------------------------------------------------------


def main() -> None:
    gold = load_gold()
    print(f"Loaded {len(gold)} gold labels ({sum(gold.values())} positives)")
    profiles = load_profiles()
    print(f"Loaded {len(profiles)} profiles from checkpoint")
    feats = per_user_features(profiles)
    print(f"Computed per-user features for {len(feats)} subjects")

    run0 = load_first_alerts(0)
    run1 = load_first_alerts(1)
    print(f"Run 0 first-alerts: {len(run0)}; Run 1 first-alerts: {len(run1)}")

    out = {
        "data_source": {
            "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
            "decisions_dir": str(DECISIONS_DIR.relative_to(ROOT)),
            "golden": str(GOLDEN.relative_to(ROOT)),
        },
        "n_subjects": len(gold),
        "n_positives": int(sum(gold.values())),
        "rules": {
            "1_intro_signal": select_intro_signal(feats, gold, run0),
            "2_w1_scale_split": select_w1_scale_split(profiles, gold, run0),
            "3_early_alert": select_early_alert(run0, run1, gold),
        },
    }

    out_path = OUT_DIR / "example_candidates.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path}")
    # Print a short human-readable summary
    print()
    for key, payload in out["rules"].items():
        print(f"--- {key} ---")
        if "chosen" in payload:
            print(json.dumps(payload["chosen"], indent=2))
        else:
            print(json.dumps({k: v for k, v in payload.items()
                              if k.startswith("chosen") or k == "error"}, indent=2))
        print()


if __name__ == "__main__":
    main()
