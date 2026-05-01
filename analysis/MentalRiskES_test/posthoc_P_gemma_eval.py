"""Evaluator for the Gemma GAD-7 post-hoc experiment.

Reads `output/mentalriskes_gemma_gad7/<model>/raw.jsonl` and produces:

  W_gemma_per_session.csv      per-(model, session) Gemma final-round vs gold
  W_gemma_summary.csv          per-model headline stats
  W_gemma_per_item.csv         per-(model, item) signed bias / MAE / exact match
  W_gemma_confidence.csv       per-(model, confidence_level) MAE — calibration check
  W_gemma_hybrid.csv           hybrid MAE_combined: submitted/replay PHQ-9 +
                               CompACT-10 (all 3 runs) x each Gemma model's GAD-7
  W_gemma_rank_projection.csv  projected leaderboard rank for each hybrid

When two or more Gemma models are present, additional cross-model files:
  W_gemma_model_ranking.csv      models sorted by best (lowest) hybrid MAE
  W_gemma_per_item_pivot.csv     per-item MAE in wide format (model x item)
  W_gemma_per_item_delta.csv     vs reference model: signed and absolute deltas
  W_gemma_best_per_session.csv   oracle: best-model GAD-7 per session

All numbers are computed on the 10 sessions we received (the same set the
leaderboard scored). Sessions with no Gemma predictions are skipped.

Runs in seconds; no LLM calls. Re-run any time more `raw.jsonl` rows land.

Usage:
  python analysis/MentalRiskES_test/posthoc_P_gemma_eval.py
  python analysis/MentalRiskES_test/posthoc_P_gemma_eval.py --reference-model google_gemma-3-27b-it
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from utils import (
    classify_band,
    load_config,
    load_task1_gold,
    mae,
    repo_path,
    task1_last_round_predictions,
    total,
)


GEMMA_ROOT_REL = "output/mentalriskes_gemma_gad7"
SUBMITTED_DIR_REL = "output/mentalriskes/predictions"
REPLAY_DIR_REL = "output/mentalriskes_test_replay/predictions"


# ---------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------
def discover_gemma_models(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / "raw.jsonl").exists())


def load_gemma_raw(jsonl_path: Path) -> list[dict]:
    out = []
    with open(jsonl_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def gemma_last_round_per_session(records: list[dict]) -> dict[str, dict]:
    """Pick the highest-round Gemma prediction per session."""
    last: dict[str, tuple[int, dict]] = {}
    for rec in records:
        sid = rec["session"]
        rnd = int(rec["round"])
        if sid not in last or rnd > last[sid][0]:
            last[sid] = (rnd, rec)
    return {sid: rec for sid, (_, rec) in last.items()}


def load_predictions_dir(predictions_dir: Path, run_idx: int) -> dict[int, dict[str, dict]]:
    """{round: {sid: prediction_dict}} for one run."""
    out: dict[int, dict[str, dict]] = {}
    for fp in sorted(predictions_dir.glob(f"round*_run{run_idx}.json")):
        rnd = int(fp.stem.split("_")[0].replace("round", ""))
        with open(fp, encoding="utf-8") as fh:
            payload = json.load(fh)
        out[rnd] = {entry["id"]: entry["prediction"] for entry in payload[0]["predictions"]}
    return out


def last_per_session(rounds: dict[int, dict[str, dict]]) -> dict[str, dict]:
    last: dict[str, tuple[int, dict]] = {}
    for rnd, sessions in rounds.items():
        for sid, pred in sessions.items():
            if sid not in last or rnd > last[sid][0]:
                last[sid] = (rnd, pred)
    return {sid: pred for sid, (_, pred) in last.items()}


# ---------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------
def evaluate_gemma_model(
    model_short: str,
    records: list[dict],
    gold: dict[str, dict],
    cfg: dict,
) -> tuple[pd.DataFrame, dict]:
    """Per-session table + summary stats for one Gemma model."""
    last = gemma_last_round_per_session(records)
    common = sorted(set(gold.keys()) & set(last.keys()))

    rows = []
    for sid in common:
        rec = last[sid]
        items = list(rec["scores"])
        confidences = list(rec.get("confidences", []))
        g = gold[sid]["GAD-7"]
        rows.append({
            "model": model_short,
            "session": sid,
            "gemma_last_round": int(rec["round"]),
            "gemma_items": items,
            "gemma_total": sum(items),
            "gemma_band": classify_band(sum(items), "GAD-7", cfg),
            "gold_items": g,
            "gold_total": sum(g),
            "gold_band": classify_band(sum(g), "GAD-7", cfg),
            "MAE_items": mae(items, g),
            "signed_total_bias": sum(items) - sum(g),
            "n_HIGH": confidences.count("HIGH"),
            "n_MEDIUM": confidences.count("MEDIUM"),
            "n_LOW": confidences.count("LOW"),
            "scoring_notes": rec.get("scoring_notes"),
        })
    df = pd.DataFrame(rows)

    summary = {
        "model": model_short,
        "n_sessions": len(rows),
        "n_records_total": len(records),
        "GAD7_MAE_items": df["MAE_items"].mean() if not df.empty else None,
        "GAD7_mean_signed_total_bias": df["signed_total_bias"].mean() if not df.empty else None,
        "GAD7_band_acc": ((df["gemma_band"] == df["gold_band"]).mean() if not df.empty else None),
    }
    return df, summary


def per_item_profile(model_short: str, df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    for i in range(7):
        diffs = []
        preds = []
        golds = []
        for _, r in df.iterrows():
            preds.append(r["gemma_items"][i])
            golds.append(r["gold_items"][i])
            diffs.append(preds[-1] - golds[-1])
        rows.append({
            "model": model_short,
            "item_idx_0based": i,
            "item_label": f"item_{i + 1}",
            "n": len(diffs),
            "mean_signed_error": sum(diffs) / len(diffs),
            "MAE": sum(abs(d) for d in diffs) / len(diffs),
            "exact_match_rate": sum(1 for d in diffs if d == 0) / len(diffs),
            "mean_gold": sum(golds) / len(golds),
            "mean_pred": sum(preds) / len(preds),
        })
    return pd.DataFrame(rows)


def confidence_profile(model_short: str, records: list[dict], gold: dict[str, dict]) -> pd.DataFrame:
    """Item-level MAE bucketed by Gemma's per-item confidence (HIGH/MEDIUM/LOW).

    Uses ALL Gemma records (not just final per-session) for higher n.
    """
    bucket = defaultdict(lambda: [0, 0])  # confidence -> [|err|_sum, count]
    for rec in records:
        sid = rec["session"]
        if sid not in gold:
            continue
        items = rec["scores"]
        confs = rec.get("confidences", [])
        g = gold[sid]["GAD-7"]
        for i in range(7):
            c = confs[i] if i < len(confs) else "UNKNOWN"
            bucket[c][0] += abs(items[i] - g[i])
            bucket[c][1] += 1
    rows = []
    for c, (sum_err, n) in bucket.items():
        rows.append({
            "model": model_short,
            "confidence_level": c,
            "n_items": n,
            "item_MAE": sum_err / n if n else None,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Hybrid MAE_Combined
# ---------------------------------------------------------------------
def _hybrid_for_run(
    src_label: str,
    run_idx: int,
    gold: dict,
    our_preds_last: dict[str, dict],
    gemma_preds_last: dict[str, dict],
) -> dict | None:
    """Per-instrument MAE using our PHQ-9 + Gemma GAD-7 + our CompACT-10."""
    common = sorted(set(gold.keys()) & set(our_preds_last.keys()) & set(gemma_preds_last.keys()))
    if not common:
        return None
    phq, gad, com = [], [], []
    for sid in common:
        ours = our_preds_last[sid]
        g = gold[sid]
        phq.append(mae(ours["PHQ-9"], g["PHQ-9"]))
        gad.append(mae(gemma_preds_last[sid]["scores"], g["GAD-7"]))
        com.append(mae(ours["CompACT-10"], g["CompACT-10"]))
    phq_mae = sum(phq) / len(phq)
    gad_mae = sum(gad) / len(gad)
    com_mae = sum(com) / len(com)
    return {
        "source": src_label,
        "run": run_idx,
        "n_sessions": len(common),
        "PHQ9_MAE": phq_mae,
        "GAD7_MAE_gemma": gad_mae,
        "CompACT10_MAE": com_mae,
        "MAE_Combined_hybrid": (phq_mae + gad_mae + com_mae) / 3,
    }


def _gemma_records_to_last_dict(records: list[dict]) -> dict[str, dict]:
    last = gemma_last_round_per_session(records)
    return {sid: {"scores": rec["scores"]} for sid, rec in last.items()}


# ---------------------------------------------------------------------
# Cross-model comparison helpers
# ---------------------------------------------------------------------
def build_model_ranking(summary_df: pd.DataFrame, hybrid_df: pd.DataFrame) -> pd.DataFrame:
    """One row per model: standalone GAD-7 MAE + best hybrid + rank stats."""
    if summary_df.empty:
        return pd.DataFrame()
    rows = []
    for _, s in summary_df.iterrows():
        model = s["model"]
        sub = hybrid_df[hybrid_df["model"] == model]
        if sub.empty:
            best_hybrid = None
            best_source = None
            best_run = None
        else:
            best_idx = sub["MAE_Combined_hybrid"].idxmin()
            best = sub.loc[best_idx]
            best_hybrid = best["MAE_Combined_hybrid"]
            best_source = best["source"]
            best_run = int(best["run"])
        rows.append({
            "model": model,
            "n_sessions": s["n_sessions"],
            "GAD7_MAE_items": s["GAD7_MAE_items"],
            "GAD7_signed_total_bias": s["GAD7_mean_signed_total_bias"],
            "GAD7_band_acc": s["GAD7_band_acc"],
            "best_hybrid_MAE_combined": best_hybrid,
            "best_hybrid_source": best_source,
            "best_hybrid_run": best_run,
        })
    return pd.DataFrame(rows).sort_values("GAD7_MAE_items").reset_index(drop=True)


def build_per_item_pivot(per_item_df: pd.DataFrame) -> pd.DataFrame:
    """Wide format: rows = items, cols = models, values = MAE."""
    if per_item_df.empty or per_item_df["model"].nunique() < 2:
        return pd.DataFrame()
    return (
        per_item_df.pivot(index="item_idx_0based", columns="model", values="MAE")
        .reset_index()
        .rename_axis(None, axis=1)
    )


def build_per_item_delta(per_item_df: pd.DataFrame, ref_model: str | None) -> pd.DataFrame:
    """Per-item signed-error delta vs a reference model (default: best by GAD-7 MAE)."""
    if per_item_df.empty or per_item_df["model"].nunique() < 2:
        return pd.DataFrame()

    if ref_model is None or ref_model not in per_item_df["model"].unique():
        # default ref = model with lowest mean MAE across items
        means = per_item_df.groupby("model")["MAE"].mean().sort_values()
        ref_model = means.index[0]

    ref_table = per_item_df[per_item_df["model"] == ref_model].set_index("item_idx_0based")
    rows = []
    for model, sub in per_item_df.groupby("model"):
        if model == ref_model:
            continue
        sub = sub.set_index("item_idx_0based")
        for i in sub.index:
            rows.append({
                "ref_model": ref_model,
                "model": model,
                "item_idx_0based": int(i),
                "item_label": sub.loc[i, "item_label"],
                "ref_MAE": ref_table.loc[i, "MAE"],
                "model_MAE": sub.loc[i, "MAE"],
                "delta_MAE": sub.loc[i, "MAE"] - ref_table.loc[i, "MAE"],
                "ref_signed_error": ref_table.loc[i, "mean_signed_error"],
                "model_signed_error": sub.loc[i, "mean_signed_error"],
                "delta_signed_error": sub.loc[i, "mean_signed_error"] - ref_table.loc[i, "mean_signed_error"],
            })
    return pd.DataFrame(rows)


def build_best_per_session(per_session_dfs: list[pd.DataFrame], gold: dict, cfg: dict) -> pd.DataFrame:
    """For each session, pick the best (lowest item-MAE) Gemma model — oracle bound."""
    if len(per_session_dfs) < 2:
        return pd.DataFrame()
    combined = pd.concat(per_session_dfs, ignore_index=True)
    rows = []
    for sid, sub in combined.groupby("session"):
        best = sub.loc[sub["MAE_items"].idxmin()]
        rows.append({
            "session": sid,
            "best_model": best["model"],
            "best_MAE_items": best["MAE_items"],
            "gold_total": int(best["gold_total"]),
            "best_pred_total": int(best["gemma_total"]),
            "n_models_evaluated": sub["model"].nunique(),
            "MAE_range_across_models": sub["MAE_items"].max() - sub["MAE_items"].min(),
        })
    return pd.DataFrame(rows).sort_values("session")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-model", default=None,
        help="Model to use as reference for per-item delta (e.g. google_gemma-3-27b-it). "
             "Default: model with lowest mean per-item MAE.",
    )
    args = parser.parse_args()

    cfg = load_config()
    out_dir = repo_path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    gemma_root = repo_path(GEMMA_ROOT_REL)
    models = discover_gemma_models(gemma_root)
    if not models:
        print(f"No Gemma model dirs found under {gemma_root}")
        return

    gold = load_task1_gold(cfg)

    per_session_frames = []
    summary_rows = []
    per_item_frames = []
    confidence_frames = []
    hybrid_rows = []

    # Pre-load our PHQ-9 + CompACT-10 for both sources (submitted + replay)
    sources = {}
    for run_idx in (0, 1, 2):
        # Submitted: round-30 snapshot via existing util (reads server-format JSONs)
        sub = task1_last_round_predictions(cfg, run_idx)
        sources[("submitted", run_idx)] = sub if sub else None
        # Replay: full 82-round, server-format output produced by the JSONL converter
        rep_rounds = load_predictions_dir(repo_path(REPLAY_DIR_REL), run_idx)
        sources[("replay", run_idx)] = last_per_session(rep_rounds) if rep_rounds else None

    for model_dir in models:
        model_short = model_dir.name
        records = load_gemma_raw(model_dir / "raw.jsonl")
        if not records:
            continue
        df_sess, summary = evaluate_gemma_model(model_short, records, gold, cfg)
        per_session_frames.append(df_sess)
        summary_rows.append(summary)
        per_item_frames.append(per_item_profile(model_short, df_sess))
        confidence_frames.append(confidence_profile(model_short, records, gold))

        gemma_last_dict = _gemma_records_to_last_dict(records)
        for (src_label, run_idx), our_preds in sources.items():
            if our_preds is None:
                continue
            row = _hybrid_for_run(src_label, run_idx, gold, our_preds, gemma_last_dict)
            if row:
                row["model"] = model_short
                hybrid_rows.append(row)

    # ------------- write outputs -------------
    if per_session_frames:
        per_session = pd.concat(per_session_frames, ignore_index=True)
        # The list columns don't round-trip through CSV cleanly; stringify them.
        per_session_out = per_session.copy()
        per_session_out["gemma_items"] = per_session_out["gemma_items"].apply(json.dumps)
        per_session_out["gold_items"] = per_session_out["gold_items"].apply(json.dumps)
        per_session_out.to_csv(out_dir / "W_gemma_per_session.csv", index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "W_gemma_summary.csv", index=False)

    if per_item_frames:
        per_item = pd.concat(per_item_frames, ignore_index=True)
        per_item.to_csv(out_dir / "W_gemma_per_item.csv", index=False)
    if confidence_frames:
        conf = pd.concat(confidence_frames, ignore_index=True)
        conf.to_csv(out_dir / "W_gemma_confidence.csv", index=False)

    hybrid_df = pd.DataFrame(hybrid_rows)
    if not hybrid_df.empty:
        hybrid_df = hybrid_df[
            ["model", "source", "run", "n_sessions", "PHQ9_MAE",
             "GAD7_MAE_gemma", "CompACT10_MAE", "MAE_Combined_hybrid"]
        ].sort_values(["model", "source", "run"])
        hybrid_df.to_csv(out_dir / "W_gemma_hybrid.csv", index=False)

    # Rank projection: hybrid MAE_Combined onto leaderboard
    rank_rows = []
    try:
        xlsx = repo_path(cfg["paths"]["leaderboard_xlsx"])
        leaderboard = pd.read_excel(xlsx, sheet_name="Task1")
    except Exception as e:
        print(f"(skipping rank projection: {e})")
        leaderboard = None
    if leaderboard is not None and not hybrid_df.empty:
        for _, r in hybrid_df.iterrows():
            n_below = int((leaderboard["MAE_Combined"] < r["MAE_Combined_hybrid"]).sum())
            rank_rows.append({
                "model": r["model"],
                "source": r["source"],
                "run": int(r["run"]),
                "MAE_Combined_hybrid": r["MAE_Combined_hybrid"],
                "GAD7_MAE_gemma": r["GAD7_MAE_gemma"],
                "projected_rank": n_below + 1,
            })
        if rank_rows:
            rank_df = pd.DataFrame(rank_rows).sort_values(["model", "MAE_Combined_hybrid"])
            rank_df.to_csv(out_dir / "W_gemma_rank_projection.csv", index=False)

    # ------------- console summary -------------
    print("=" * 70)
    print("Gemma GAD-7 evaluation")
    print("=" * 70)
    print("\nW_gemma_summary.csv:")
    print(summary_df.to_string(index=False))

    if per_item_frames:
        print("\nW_gemma_per_item.csv (item-level error profile):")
        print(pd.concat(per_item_frames, ignore_index=True).to_string(index=False))

    if confidence_frames:
        print("\nW_gemma_confidence.csv (item-MAE by confidence):")
        print(pd.concat(confidence_frames, ignore_index=True).to_string(index=False))

    if not hybrid_df.empty:
        print("\nW_gemma_hybrid.csv:")
        print(hybrid_df.to_string(index=False))

    if rank_rows:
        print("\nW_gemma_rank_projection.csv:")
        print(pd.DataFrame(rank_rows).sort_values(["model", "MAE_Combined_hybrid"]).to_string(index=False))

    # ---------------- cross-model comparison ----------------
    n_models = len(summary_rows)
    if n_models < 2:
        print(f"\n(Only {n_models} model present; skipping cross-model comparison.)")
        return

    print("\n" + "=" * 70)
    print(f"Cross-model comparison ({n_models} models)")
    print("=" * 70)

    if not hybrid_df.empty:
        ranking_df = build_model_ranking(summary_df, hybrid_df)
        ranking_df.to_csv(out_dir / "W_gemma_model_ranking.csv", index=False)
        print("\nW_gemma_model_ranking.csv (sorted by GAD-7 MAE):")
        print(ranking_df.to_string(index=False))

    if per_item_frames:
        per_item_concat = pd.concat(per_item_frames, ignore_index=True)
        pivot_df = build_per_item_pivot(per_item_concat)
        if not pivot_df.empty:
            pivot_df.to_csv(out_dir / "W_gemma_per_item_pivot.csv", index=False)
            print("\nW_gemma_per_item_pivot.csv (per-item MAE by model):")
            print(pivot_df.to_string(index=False))

        delta_df = build_per_item_delta(per_item_concat, args.reference_model)
        if not delta_df.empty:
            delta_df.to_csv(out_dir / "W_gemma_per_item_delta.csv", index=False)
            print(f"\nW_gemma_per_item_delta.csv (vs ref={delta_df['ref_model'].iloc[0]}):")
            print(delta_df.to_string(index=False))

    if len(per_session_frames) >= 2:
        best_df = build_best_per_session(per_session_frames, gold, cfg)
        if not best_df.empty:
            best_df.to_csv(out_dir / "W_gemma_best_per_session.csv", index=False)
            print("\nW_gemma_best_per_session.csv (oracle Gemma per session):")
            print(best_df.to_string(index=False))
            print(f"\n  Mean best-MAE_items across sessions: {best_df['best_MAE_items'].mean():.4f}")
            print(f"  (vs single-best-model {summary_df['GAD7_MAE_items'].min():.4f}; "
                  f"oracle headroom = {summary_df['GAD7_MAE_items'].min() - best_df['best_MAE_items'].mean():+.4f})")


if __name__ == "__main__":
    main()
