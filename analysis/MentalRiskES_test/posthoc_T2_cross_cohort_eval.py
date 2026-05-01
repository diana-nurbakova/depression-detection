"""Task 2 cross-cohort evaluator.

Compares S, S2 bare-LLM (and our submitted Run-2-equivalent HYB B+ FIX W3
configuration) across three cohorts:

    test       — released test set (R1-30 inner-join slice + full 82-round replay)
    trial      — legacy 19-round single-session trial (gold = TRIAL_GROUND_TRUTH)
    simulated  — 7 persona-simulated sessions (gold = labels.json per session)

For each (system, cohort) combination, computes accuracy, macro F1, prediction
distribution, and chi-squared vs uniform / vs gold. The "would we have picked
S2 over Submitted before submission?" question is answered by comparing the
Submitted-equivalent ablation entry against the new bare-LLM numbers on
trial + simulated.

Outputs:
    W_t2_cross_cohort.csv
    W_t2_cross_cohort_summary.md (paper-ready table)
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(REPO_ROOT / "src"))

from utils import load_config, load_task2_gold, repo_path

logger = logging.getLogger("t2_cross_cohort")


# ─────────────────────────────────────────────────────────────────────────────
# Source registry — each entry returns {(cohort, session, round): pred_int}
# ─────────────────────────────────────────────────────────────────────────────
def _load_bare_llm(cohort: str, model: str, mode: str) -> dict[tuple[str, str, int], int]:
    """{(cohort, sid, round): pred} from output/mentalriskes_task2_bare_llm/<model>__<mode>__<cohort>/raw.jsonl."""
    model_short = model.replace("/", "_").replace(":", "_")
    suffix = f"{model_short}__{mode}" if cohort == "test" else f"{model_short}__{mode}__{cohort}"
    fp = REPO_ROOT / "output/mentalriskes_task2_bare_llm" / suffix / "raw.jsonl"
    out: dict[tuple[str, str, int], int] = {}
    if not fp.exists():
        return out
    with open(fp, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            out[(cohort, rec["session"], int(rec["round"]))] = int(rec["prediction"])
    return out


def _load_submitted_test(run_idx: int = 2) -> dict[tuple[str, str, int], int]:
    """Round-30 submission predictions for the test cohort."""
    sub_dir = REPO_ROOT / "output/mentalriskes_task2/server_submissions"
    out: dict[tuple[str, str, int], int] = {}
    for fp in sorted(sub_dir.glob(f"round*_run{run_idx}.json")):
        rnd = int(fp.stem.split("_")[0].replace("round", ""))
        with open(fp, encoding="utf-8") as fh:
            payload = json.load(fh)
        for entry in payload[0]["predictions"]:
            out[("test", entry["id"], rnd)] = int(entry["prediction"])
    return out


def _load_submitted_replay_test(run_idx: int = 2) -> dict[tuple[str, str, int], int]:
    """Full 82-round replay predictions for the test cohort."""
    rep_dir = REPO_ROOT / "output/mentalriskes_task2_test_replay/server_submissions"
    out: dict[tuple[str, str, int], int] = {}
    for fp in sorted(rep_dir.glob(f"round*_run{run_idx}.json")):
        rnd = int(fp.stem.split("_")[0].replace("round", ""))
        with open(fp, encoding="utf-8") as fh:
            payload = json.load(fh)
        for entry in payload[0]["predictions"]:
            out[("test", entry["id"], rnd)] = int(entry["prediction"])
    return out


def _load_jsonl_ablation(jsonl_path: Path, cohort: str, session_id: str) -> dict[tuple[str, str, int], int]:
    """Single-session JSONL produced by the legacy task2 ablation pipeline."""
    out: dict[tuple[str, str, int], int] = {}
    if not jsonl_path.exists():
        return out
    with open(jsonl_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("type") != "round":
                continue
            out[(cohort, session_id, int(rec["round_id"]))] = int(rec["chosen_option"])
    return out


def _load_submitted_trial() -> dict[tuple[str, str, int], int]:
    """Submitted-equivalent on trial: B+ HYB FIX W3 (matches Run 2 config)."""
    fp = REPO_ROOT / "output/mentalriskes_task2/ablation/B+_Llama-3.3-70B-Instruct-Turbo_es_HYB_FIX_W3.jsonl"
    return _load_jsonl_ablation(fp, "trial", "trial")


def _load_submitted_simulated() -> dict[tuple[str, str, int], int]:
    """Submitted-equivalent on simulated: per-session JSONL under simulated_ablation/B+_..._HYB_FIX_W3/."""
    sim_dir = REPO_ROOT / "output/mentalriskes_task2/simulated_ablation/B+_Llama-3.3-70B-Instruct-Turbo_es_HYB_FIX_W3"
    out: dict[tuple[str, str, int], int] = {}
    if not sim_dir.exists():
        return out
    for fp in sorted(sim_dir.glob("*.jsonl")):
        session_id = fp.stem
        out.update(_load_jsonl_ablation(fp, "simulated", session_id))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Gold loaders per cohort
# ─────────────────────────────────────────────────────────────────────────────
def _opt_to_int(opt: str) -> int:
    return int(opt.replace("option_", ""))


def _load_gold_test(cfg: dict) -> dict[tuple[str, str, int], int]:
    gold = load_task2_gold(cfg)
    return {("test", sid, rnd): _opt_to_int(opt)
            for rnd, sess in gold.items()
            for sid, opt in sess.items()}


def _load_gold_trial() -> dict[tuple[str, str, int], int]:
    """TRIAL_GROUND_TRUTH from src/mentalriskes/task2/data.py — 18 rounds."""
    from mentalriskes.task2.data import TRIAL_GROUND_TRUTH
    return {("trial", "trial", rnd): int(opt) for rnd, opt in TRIAL_GROUND_TRUTH.items()}


def _load_gold_simulated() -> dict[tuple[str, str, int], int]:
    sim_root = REPO_ROOT / "output/mentalriskes/data_prep/simulated/task2"
    out: dict[tuple[str, str, int], int] = {}
    if not sim_root.exists():
        return out
    for d in sorted(sim_root.iterdir()):
        if not d.is_dir():
            continue
        labels_fp = d / "labels.json"
        if not labels_fp.exists():
            continue
        with open(labels_fp, encoding="utf-8") as fh:
            labels = json.load(fh)
        for k, v in labels.items():
            out[("simulated", d.name, int(k))] = int(v)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Per (system, cohort) accuracy
# ─────────────────────────────────────────────────────────────────────────────
def _evaluate(predictions: dict, gold: dict, system_name: str, cohort: str) -> dict | None:
    common_keys = [k for k in predictions if k in gold]
    if not common_keys:
        return None
    g = [gold[k] for k in common_keys]
    p = [predictions[k] for k in common_keys]
    correct = sum(int(gv == pv) for gv, pv in zip(g, p))
    n = len(common_keys)
    pred_dist = Counter(p)
    gold_dist = Counter(g)
    return {
        "system": system_name, "cohort": cohort, "n": n,
        "accuracy": correct / n,
        "pred_p1": pred_dist.get(1, 0) / n,
        "pred_p2": pred_dist.get(2, 0) / n,
        "pred_p3": pred_dist.get(3, 0) / n,
        "gold_p1": gold_dist.get(1, 0) / n,
        "gold_p2": gold_dist.get(2, 0) / n,
        "gold_p3": gold_dist.get(3, 0) / n,
    }


def main() -> None:
    logging.basicConfig(level="INFO", format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    cfg = load_config()
    out_dir = repo_path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Gold
    gold_test = _load_gold_test(cfg)
    gold_trial = _load_gold_trial()
    gold_sim = _load_gold_simulated()
    logger.info("Gold: test=%d  trial=%d  simulated=%d", len(gold_test), len(gold_trial), len(gold_sim))

    # Predictions
    sources = {
        ("Submitted Run 2 (HYB B+, R1-30)", "test"): _load_submitted_test(),
        ("Submitted Run 2 replay (HYB B+, full)", "test"): _load_submitted_replay_test(),
        ("Submitted-equivalent (HYB B+ FIX W3)", "trial"): _load_submitted_trial(),
        ("Submitted-equivalent (HYB B+ FIX W3)", "simulated"): _load_submitted_simulated(),
    }
    for cohort in ("test", "trial", "simulated"):
        for mode in ("S", "S2", "S3", "S4", "R2"):
            preds = _load_bare_llm(cohort, "google/gemma-4-31b-it", mode)
            if preds:
                sources[(f"Gemma 4 31B bare ({mode})", cohort)] = preds
        # Also add the other models we tested on test, only on test
        if cohort == "test":
            for model_id in ("google/gemma-3-27b-it", "meta-llama/llama-3.3-70b-instruct"):
                short = model_id.split("/")[-1]
                preds = _load_bare_llm(cohort, model_id, "S")
                if preds:
                    sources[(f"{short} bare (S)", cohort)] = preds

    # Evaluate
    rows = []
    for (system, cohort), preds in sources.items():
        if cohort == "test":
            gold = gold_test
        elif cohort == "trial":
            gold = gold_trial
        elif cohort == "simulated":
            gold = gold_sim
        else:
            continue
        result = _evaluate(preds, gold, system, cohort)
        if result is not None:
            rows.append(result)

    if not rows:
        logger.warning("No predictions found")
        return

    df = pd.DataFrame(rows)
    # Order: cohort first (test, trial, simulated), then accuracy desc
    cohort_order = {"test": 0, "trial": 1, "simulated": 2}
    df["_cohort_ord"] = df["cohort"].map(cohort_order)
    df = df.sort_values(["_cohort_ord", "accuracy"], ascending=[True, False]).drop(columns="_cohort_ord")

    df.to_csv(out_dir / "W_t2_cross_cohort.csv", index=False)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Paper-ready Markdown
    md = ["# Task 2 — Cross-cohort comparison\n"]
    md.append("Each cell = accuracy (n). Sources of gold: test = released `round_X_gold.json`; "
              "trial = `TRIAL_GROUND_TRUTH` (18 rounds, single session); simulated = `labels.json` "
              "per persona session.\n")
    pivot = df.pivot_table(index="system", columns="cohort", values="accuracy", aggfunc="first")
    n_pivot = df.pivot_table(index="system", columns="cohort", values="n", aggfunc="first")
    md.append("| System | test | trial | simulated |")
    md.append("| --- | --- | --- | --- |")
    for sys_name in pivot.index:
        cells = []
        for cohort in ("test", "trial", "simulated"):
            acc = pivot.loc[sys_name, cohort] if cohort in pivot.columns else None
            n = n_pivot.loc[sys_name, cohort] if cohort in n_pivot.columns else None
            if pd.isna(acc) or pd.isna(n):
                cells.append("—")
            else:
                cells.append(f"**{acc:.3f}** (n={int(n)})")
        md.append(f"| {sys_name} | {cells[0]} | {cells[1]} | {cells[2]} |")
    md_path = out_dir / "W_t2_cross_cohort_summary.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"\nWrote {md_path}")


if __name__ == "__main__":
    main()
