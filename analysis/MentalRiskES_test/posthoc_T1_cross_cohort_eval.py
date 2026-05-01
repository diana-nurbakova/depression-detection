"""Task 1 Gemma GAD-7 cross-cohort evaluator.

Compares Gemma v1 / v2 prompts (and our submitted Llama assessor where
predictions are available) across:

    test       — released test set (gold = item-level GAD-7 per session in gold_label.json)
    trial      — single-session 19-round trial (gold = per-session GAD-7 *total* if available)
    simulated  — persona-simulated sessions (gold = target_scores.gad7_total in metadata.json,
                 total only — no item gold)

For test we report item-MAE + signed bias + band accuracy (rich metric set).
For trial/simulated we report total-MAE + band accuracy on totals (since
item-level gold is unavailable for those cohorts).

Outputs:
    W_t1_cross_cohort.csv
    W_t1_cross_cohort_summary.md
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(REPO_ROOT / "src"))

from utils import classify_band, load_config, load_task1_gold, mae, repo_path, total

logger = logging.getLogger("t1_cross_cohort")


GEMMA_ROOT = REPO_ROOT / "output/mentalriskes_gemma_gad7"


# ─────────────────────────────────────────────────────────────────────────────
# Prediction loaders — from raw.jsonl in cohort-suffixed dirs
# ─────────────────────────────────────────────────────────────────────────────
def _load_gemma_predictions(model: str, prompt_version: str, cohort: str) -> dict[tuple[str, str], list[int]]:
    """{(cohort, sid): [items at last round]} — Gemma per-session final-round prediction."""
    model_short = model.replace("/", "_").replace(":", "_")
    if cohort == "test" and prompt_version == "v1":
        suffix = model_short
    elif cohort == "test":
        suffix = f"{model_short}__{prompt_version}"
    else:
        suffix = f"{model_short}__{prompt_version}__{cohort}"
    fp = GEMMA_ROOT / suffix / "raw.jsonl"
    if not fp.exists():
        return {}
    last: dict[str, tuple[int, list[int]]] = {}
    with open(fp, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            sid = rec["session"]
            rnd = int(rec["round"])
            items = list(rec["scores"])
            if sid not in last or rnd > last[sid][0]:
                last[sid] = (rnd, items)
    return {(cohort, sid): items for sid, (_, items) in last.items()}


def _load_llama_test_predictions(run_idx: int = 2) -> dict[tuple[str, str], list[int]]:
    """Our Llama assessor's test predictions (full-replay last-round per session)."""
    rep_dir = REPO_ROOT / "output/mentalriskes_test_replay/predictions"
    last: dict[str, tuple[int, list[int]]] = {}
    for fp in sorted(rep_dir.glob(f"round*_run{run_idx}.json")):
        rnd = int(fp.stem.split("_")[0].replace("round", ""))
        with open(fp, encoding="utf-8") as fh:
            payload = json.load(fh)
        for entry in payload[0]["predictions"]:
            sid = entry["id"]
            items = list(entry["prediction"]["GAD-7"])
            if sid not in last or rnd > last[sid][0]:
                last[sid] = (rnd, items)
    return {("test", sid): items for sid, (_, items) in last.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Gold loaders per cohort
# ─────────────────────────────────────────────────────────────────────────────
def _load_gold_test(cfg: dict) -> dict[tuple[str, str], dict]:
    """{(cohort=test, sid): {items, total, band}} from gold_label.json (item-level)."""
    gold = load_task1_gold(cfg)
    out = {}
    for sid, instr_dict in gold.items():
        items = list(instr_dict["GAD-7"])
        out[("test", sid)] = {"items": items, "total": sum(items), "band": classify_band(sum(items), "GAD-7", cfg)}
    return out


def _load_gold_trial(cfg: dict) -> dict[tuple[str, str], dict]:
    """Trial Task 1 has no published per-item gold. Return empty unless we add one later."""
    return {}


def _load_gold_simulated(cfg: dict) -> dict[tuple[str, str], dict]:
    """{(cohort=simulated, sid): {total, band}} from metadata.json target_scores.gad7_total."""
    sim_root = REPO_ROOT / "output/mentalriskes/data_prep/simulated/task1"
    out = {}
    if not sim_root.exists():
        return out
    for d in sorted(sim_root.iterdir()):
        if not d.is_dir():
            continue
        meta_fp = d / "metadata.json"
        if not meta_fp.exists():
            continue
        with open(meta_fp, encoding="utf-8") as fh:
            meta = json.load(fh)
        total_score = meta.get("target_scores", {}).get("gad7_total")
        if total_score is None:
            continue
        out[("simulated", d.name)] = {
            "total": int(total_score),
            "band": classify_band(int(total_score), "GAD-7", cfg),
        }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────
def _evaluate_with_items(predictions: dict, gold: dict, system: str, cohort: str, cfg: dict) -> dict | None:
    common = sorted(k for k in predictions if k in gold and "items" in gold[k])
    if not common:
        return None
    item_maes = []
    total_diffs = []
    band_correct = 0
    for k in common:
        p = predictions[k]
        g = gold[k]["items"]
        item_maes.append(mae(p, g))
        total_diffs.append(sum(p) - sum(g))
        if classify_band(sum(p), "GAD-7", cfg) == gold[k]["band"]:
            band_correct += 1
    return {
        "system": system, "cohort": cohort, "n": len(common),
        "GAD7_MAE_items": sum(item_maes) / len(item_maes),
        "GAD7_signed_total_bias": sum(total_diffs) / len(total_diffs),
        "GAD7_band_acc": band_correct / len(common),
        "GAD7_MAE_total": sum(abs(d) for d in total_diffs) / len(total_diffs),
    }


def _evaluate_total_only(predictions: dict, gold: dict, system: str, cohort: str, cfg: dict) -> dict | None:
    common = sorted(k for k in predictions if k in gold)
    if not common:
        return None
    total_diffs = []
    band_correct = 0
    for k in common:
        p_total = sum(predictions[k])
        g_total = gold[k]["total"]
        total_diffs.append(p_total - g_total)
        if classify_band(p_total, "GAD-7", cfg) == gold[k]["band"]:
            band_correct += 1
    return {
        "system": system, "cohort": cohort, "n": len(common),
        "GAD7_MAE_items": None,
        "GAD7_signed_total_bias": sum(total_diffs) / len(total_diffs),
        "GAD7_band_acc": band_correct / len(common),
        "GAD7_MAE_total": sum(abs(d) for d in total_diffs) / len(total_diffs),
    }


def main() -> None:
    logging.basicConfig(level="INFO", format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    cfg = load_config()
    out_dir = repo_path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    gold_test = _load_gold_test(cfg)
    gold_trial = _load_gold_trial(cfg)  # currently empty
    gold_sim = _load_gold_simulated(cfg)
    logger.info("Gold: test=%d  trial=%d  simulated=%d",
                len(gold_test), len(gold_trial), len(gold_sim))

    # Predictions
    sources: list[tuple[str, str, dict]] = []
    sources.append(("Llama-3.3-70B (replay, our pipeline)", "test", _load_llama_test_predictions()))
    for cohort in ("test", "trial", "simulated"):
        for model in ("google/gemma-3-27b-it", "google/gemma-4-26b-a4b-it", "google/gemma-4-31b-it"):
            for pv in ("v1", "v2"):
                preds = _load_gemma_predictions(model, pv, cohort)
                if preds:
                    short = model.split("/")[-1]
                    sources.append((f"{short} {pv}", cohort, preds))

    # Evaluate
    rows = []
    for system, cohort, preds in sources:
        if cohort == "test":
            r = _evaluate_with_items(preds, gold_test, system, cohort, cfg)
        elif cohort == "trial":
            r = _evaluate_with_items(preds, gold_trial, system, cohort, cfg)
            if r is None:
                # Try totals if we add trial gold later; for now skip
                continue
        elif cohort == "simulated":
            r = _evaluate_total_only(preds, gold_sim, system, cohort, cfg)
        else:
            continue
        if r is not None:
            rows.append(r)

    if not rows:
        logger.warning("No predictions found")
        return

    df = pd.DataFrame(rows)
    cohort_order = {"test": 0, "trial": 1, "simulated": 2}
    df["_cohort_ord"] = df["cohort"].map(cohort_order)
    df = df.sort_values(["_cohort_ord", "GAD7_MAE_total"], ascending=[True, True]).drop(columns="_cohort_ord")
    df.to_csv(out_dir / "W_t1_cross_cohort.csv", index=False)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)))

    # Markdown summary
    md = ["# Task 1 GAD-7 — Cross-cohort comparison\n"]
    md.append("Gold sources: test = item-level `gold_label.json`; trial = (none, no item gold); "
              "simulated = `target_scores.gad7_total` (totals only).\n")
    md.append("**MAE_total** is reported uniformly across cohorts; item-level MAE is in the `GAD7_MAE_items` "
              "column for the test cohort only.\n")
    md.append("| System | Cohort | n | item-MAE | total MAE | signed bias | band acc |")
    md.append("| --- | --- | --- | --- | --- | --- | --- |")
    for _, r in df.iterrows():
        item_mae = f"{r['GAD7_MAE_items']:.3f}" if pd.notna(r['GAD7_MAE_items']) else "—"
        md.append(f"| {r['system']} | {r['cohort']} | {int(r['n'])} | {item_mae} | "
                  f"{r['GAD7_MAE_total']:.2f} | {r['GAD7_signed_total_bias']:+.2f} | "
                  f"{r['GAD7_band_acc']:.2f} |")
    (out_dir / "W_t1_cross_cohort_summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\nWrote", out_dir / "W_t1_cross_cohort_summary.md")


if __name__ == "__main__":
    main()
