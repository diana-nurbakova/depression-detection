"""Diagnose whether the long-scale Wasserstein W1 belongs in the Task 2 feature vector.

Builds per-subject W1 statistics at each of the 3 symptom-W1 scales (short / medium /
long) for all 523 official test subjects, then evaluates:

  - Marginal discriminativeness: AUC of (max W1, mean W1, n_fires, max_overshoot)
    against the official golden labels, per scale.
  - Pairwise rank correlation between scales (Spearman, max W1).
  - Conditional info: of users who fire short, what does long add?
  - Classifier evidence: feature-importance share for the 21 long-scale symptom-W1
    features in the trained XGBoost (Run 0/4).

Writes:
  analysis/eda_task2/outputs/w1_scale_ablation.json
  analysis/eda_task2/outputs/w1_scale_perscale_perdim.csv  (XGB importance per W slot)
"""

from __future__ import annotations

import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from erisk_task2.distances.wasserstein import compute_wasserstein_1d  # noqa: E402

CHECKPOINT = ROOT / "runs" / "task2" / "train" / "checkpoint" / "round_0500_state.pkl"
XGB_PICKLE = ROOT / "runs" / "task2" / "train" / "classifier_xgboost.pkl"
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

# Feature-vector offsets (see pipeline.py line 564)
W_BLOCK_OFFSET = 2191  # absolute offset of the 72-d wasserstein block in X
# Within the 72-d W block:
#   0-62  symptom (3 scales x 21):  0-20 short, 21-41 medium, 42-62 long
#   63-65 emotion (3):              63 short, 64 medium, 65 long
#   66-68 embedding (3):            66 short, 67 medium, 68 long
#   69-71 topic (3):                69 short, 70 medium, 71 long

SHORT_SYMPTOM_IDX = list(range(W_BLOCK_OFFSET + 0, W_BLOCK_OFFSET + 21))
MED_SYMPTOM_IDX = list(range(W_BLOCK_OFFSET + 21, W_BLOCK_OFFSET + 42))
LONG_SYMPTOM_IDX = list(range(W_BLOCK_OFFSET + 42, W_BLOCK_OFFSET + 63))
SHORT_OTHER_IDX = [W_BLOCK_OFFSET + 63, W_BLOCK_OFFSET + 66, W_BLOCK_OFFSET + 69]
MED_OTHER_IDX = [W_BLOCK_OFFSET + 64, W_BLOCK_OFFSET + 67, W_BLOCK_OFFSET + 70]
LONG_OTHER_IDX = [W_BLOCK_OFFSET + 65, W_BLOCK_OFFSET + 68, W_BLOCK_OFFSET + 71]


def load_gold() -> dict[str, int]:
    gold: dict[str, int] = {}
    with open(GOLDEN, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                gold[parts[0]] = int(parts[1])
    return gold


def w1_trajectory_meanover_symptoms(
    activations: list[np.ndarray],
    *,
    short_window: int = 5,
    medium_window: int = 25,
) -> dict[str, list[float]]:
    if not activations:
        return {"short": [], "medium": [], "long": []}
    arr = np.stack(activations)
    short_traj: list[float] = []
    med_traj: list[float] = []
    long_traj: list[float] = []
    for k in range(1, len(arr) + 1):
        if k >= 10:
            half = short_window
            early = arr[max(0, k - 2 * half) : k - half]
            recent = arr[k - half : k]
            vals = [compute_wasserstein_1d(early[:, s], recent[:, s]) for s in range(21)]
            short_traj.append(float(np.mean(vals)))
        else:
            short_traj.append(float("nan"))
        if k >= 50:
            half = medium_window
            early = arr[max(0, k - 2 * half) : k - half]
            recent = arr[k - half : k]
            vals = [compute_wasserstein_1d(early[:, s], recent[:, s]) for s in range(21)]
            med_traj.append(float(np.mean(vals)))
        else:
            med_traj.append(float("nan"))
        if k >= 20:
            mid = k // 2
            early = arr[:mid]
            recent = arr[mid:k]
            vals = [compute_wasserstein_1d(early[:, s], recent[:, s]) for s in range(21)]
            long_traj.append(float(np.mean(vals)))
        else:
            long_traj.append(float("nan"))
    return {"short": short_traj, "medium": med_traj, "long": long_traj}


def running_threshold_stats(traj: list[float]) -> dict:
    fired_rounds = 0
    max_overshoot = 0.0
    past: list[float] = []
    fired = False
    for v in traj:
        if np.isnan(v):
            continue
        if len(past) >= 3:
            mu = float(np.mean(past))
            sigma = float(np.std(past))
            thr = mu + 2 * sigma
            if v > thr:
                fired = True
                fired_rounds += 1
                max_overshoot = max(max_overshoot, v - thr)
        past.append(v)
    return {"fired": fired, "n_fires": fired_rounds, "max_overshoot": max_overshoot}


def per_scale_summary(traj: list[float]) -> dict:
    arr = np.array([v for v in traj if not np.isnan(v)])
    if arr.size == 0:
        return {"max_w1": 0.0, "mean_w1": 0.0, "p95_w1": 0.0, "n_rounds": 0}
    return {
        "max_w1": float(arr.max()),
        "mean_w1": float(arr.mean()),
        "p95_w1": float(np.percentile(arr, 95)),
        "n_rounds": int(arr.size),
    }


# --------------------------- AUC + correlations ---------------------------


def roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    # Mann-Whitney form, equal to ROC AUC
    pos = score[y == 1]
    neg = score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # rank-based for ties
    all_scores = np.concatenate([pos, neg])
    ranks = np.empty_like(all_scores, dtype=float)
    order = np.argsort(all_scores)
    ranks[order] = np.arange(len(all_scores)) + 1
    # average ties
    sorted_scores = all_scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i]] + ranks[order[j]]) / 2
            for k in range(i, j + 1):
                ranks[order[k]] = avg
        i = j + 1
    rank_pos = ranks[: len(pos)]
    auc = (rank_pos.sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return float(auc)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    def rankdata(a):
        order = np.argsort(a)
        ranks = np.empty_like(a, dtype=float)
        ranks[order] = np.arange(len(a)) + 1
        sa = a[order]
        i = 0
        while i < len(sa):
            j = i
            while j + 1 < len(sa) and sa[j + 1] == sa[i]:
                j += 1
            if j > i:
                avg = (i + j + 2) / 2
                for k in range(i, j + 1):
                    ranks[order[k]] = avg
            i = j + 1
        return ranks

    rx, ry = rankdata(x), rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(np.sqrt((rx**2).sum() * (ry**2).sum()))
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def main() -> None:
    gold = load_gold()
    with open(CHECKPOINT, "rb") as f:
        state = pickle.load(f)
    profiles = state["profiles"]

    rows = []  # per-subject stats
    print(f"Computing W1 trajectories for {len(profiles)} subjects...")
    for i, (uid, p) in enumerate(profiles.items(), 1):
        acts = getattr(p, "symptom_activations", None) or []
        traj = w1_trajectory_meanover_symptoms(list(acts))
        row = {"subject_id": uid, "gold": gold.get(uid)}
        for scale in ("short", "medium", "long"):
            row[f"{scale}_summary"] = per_scale_summary(traj[scale])
            row[f"{scale}_threshold"] = running_threshold_stats(traj[scale])
        row["n_activations"] = len(acts)
        rows.append(row)
        if i % 50 == 0:
            print(f"  {i}/{len(profiles)}")

    # ---- AUC per scale per metric ----
    y = np.array([r["gold"] for r in rows if r["gold"] is not None])
    aucs = {}
    for scale in ("short", "medium", "long"):
        scale_aucs = {}
        for key in ("max_w1", "mean_w1", "p95_w1"):
            scores = np.array([r[f"{scale}_summary"][key] for r in rows if r["gold"] is not None])
            scale_aucs[key] = roc_auc(y, scores)
        scores_nfires = np.array([r[f"{scale}_threshold"]["n_fires"] for r in rows if r["gold"] is not None])
        scale_aucs["n_fires"] = roc_auc(y, scores_nfires)
        scores_overshoot = np.array([r[f"{scale}_threshold"]["max_overshoot"] for r in rows if r["gold"] is not None])
        scale_aucs["max_overshoot"] = roc_auc(y, scores_overshoot)
        scores_fired = np.array([1.0 if r[f"{scale}_threshold"]["fired"] else 0.0 for r in rows if r["gold"] is not None])
        scale_aucs["fired_flag"] = roc_auc(y, scores_fired)
        aucs[scale] = scale_aucs

    # ---- Pairwise correlations (max_w1) ----
    s = np.array([r["short_summary"]["max_w1"] for r in rows])
    m = np.array([r["medium_summary"]["max_w1"] for r in rows])
    l = np.array([r["long_summary"]["max_w1"] for r in rows])
    correlations = {
        "spearman_short_long": spearman(s, l),
        "spearman_short_medium": spearman(s, m),
        "spearman_medium_long": spearman(m, l),
    }

    # ---- Conditional contingency: short_fired x long_fired x gold ----
    sf = np.array([r["short_threshold"]["fired"] for r in rows])
    lf = np.array([r["long_threshold"]["fired"] for r in rows])
    g = np.array([r["gold"] for r in rows])
    cells = {}
    for sv in (False, True):
        for lv in (False, True):
            mask = (sf == sv) & (lf == lv)
            cells[f"short={sv}, long={lv}"] = {
                "n": int(mask.sum()),
                "n_pos": int((g[mask] == 1).sum()),
                "n_neg": int((g[mask] == 0).sum()),
                "pos_rate": float((g[mask] == 1).mean()) if mask.sum() else None,
            }

    # ---- XGBoost feature-importance evidence (Run 0/4) ----
    with open(XGB_PICKLE, "rb") as f:
        clf_pkl = pickle.load(f)
    model = clf_pkl["model"]
    importances = model.feature_importances_  # shape (n_features,)
    total_imp = float(importances.sum())
    short_imp = float(importances[SHORT_SYMPTOM_IDX].sum())
    med_imp = float(importances[MED_SYMPTOM_IDX].sum())
    long_imp = float(importances[LONG_SYMPTOM_IDX].sum())
    short_other = float(importances[SHORT_OTHER_IDX].sum())
    med_other = float(importances[MED_OTHER_IDX].sum())
    long_other = float(importances[LONG_OTHER_IDX].sum())
    # All-W block
    w_block_imp = float(importances[W_BLOCK_OFFSET : W_BLOCK_OFFSET + 72].sum())
    xgb_evidence = {
        "n_features": int(importances.shape[0]),
        "total_importance": total_imp,
        "wasserstein_block_share": w_block_imp / total_imp,
        "symptom_short_share": short_imp / total_imp,
        "symptom_medium_share": med_imp / total_imp,
        "symptom_long_share": long_imp / total_imp,
        "other_short_share": short_other / total_imp,
        "other_medium_share": med_other / total_imp,
        "other_long_share": long_other / total_imp,
        "symptom_short_nonzero_dims": int((importances[SHORT_SYMPTOM_IDX] > 0).sum()),
        "symptom_medium_nonzero_dims": int((importances[MED_SYMPTOM_IDX] > 0).sum()),
        "symptom_long_nonzero_dims": int((importances[LONG_SYMPTOM_IDX] > 0).sum()),
    }

    # ---- Per-dim CSV ----
    csv_path = OUT_DIR / "w1_scale_perscale_perdim.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["scale", "dim", "symptom_index", "importance"])
        for s_label, idxs in (
            ("short", SHORT_SYMPTOM_IDX),
            ("medium", MED_SYMPTOM_IDX),
            ("long", LONG_SYMPTOM_IDX),
        ):
            for d, gi in enumerate(idxs):
                wr.writerow([s_label, d, gi, float(importances[gi])])

    out = {
        "n_subjects": len(rows),
        "n_positive": int((g == 1).sum()),
        "auc_per_scale": aucs,
        "scale_correlations_spearman_maxw1": correlations,
        "contingency_short_x_long_fired": cells,
        "xgb_importance": xgb_evidence,
        "interpretation_keys": {
            "auc_0.5_means_no_discrimination": True,
            "fired_flag_AUC_summarizes_running_threshold_signal": True,
        },
    }
    out_path = OUT_DIR / "w1_scale_ablation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path}")
    print(f"Wrote {csv_path}")

    # Print summary
    print()
    print("=== AUC vs gold ===")
    for scale in ("short", "medium", "long"):
        print(f"  {scale:6s}", end="")
        for key, val in aucs[scale].items():
            print(f"  {key}={val:.3f}", end="")
        print()
    print()
    print("=== Spearman (max_w1) ===")
    for k, v in correlations.items():
        print(f"  {k}: {v:.3f}")
    print()
    print("=== Short x Long contingency ===")
    for k, v in cells.items():
        print(f"  {k}: {v}")
    print()
    print("=== XGB importance (Run 0/4) ===")
    for k, v in xgb_evidence.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
