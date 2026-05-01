"""Post-Hoc Analysis W — Submitted (round 30) vs Full Replay comparison.

Quantifies the cost of stopping at round 30. For each (patient, run,
instrument), compares the prediction we actually submitted to the live server
(round 30 snapshot in `output/mentalriskes/predictions/`) with the prediction
the system *would* have submitted at the patient's true last round, taken
from the full-test replay in `output/mentalriskes_test_replay/predictions/`.

Task 1 outputs:
    W_per_session_comparison.csv    per-(session, run, instrument) detail
    W_per_run_aggregate.csv         headline table for the paper
    W_per_run_band_acc.csv          severity-band accuracy delta
    W_rank_projection.csv           projected leaderboard rank using replay MAE
    figures/W_trajectory_{sid}_{instrument}.png   per-patient score arcs

Task 2 outputs (new):
    W_t2_round_decomposition.csv     per-run accuracy on R1-30 vs full
    W_t2_round_tercile.csv           accuracy by round tercile (early/mid/late)
    W_t2_rank_projection.csv         projected Task 2 rank with replay accuracy

Runs in seconds; no LLM calls. Pulls whatever is on disk, so it can be
re-run incrementally as the replay progresses.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from utils import (
    classify_band,
    load_config,
    load_task1_gold,
    load_task1_predictions,
    load_task2_gold,
    load_task2_predictions,
    mae,
    repo_path,
    task1_last_round_predictions,
    total,
)


# Round-30 snapshot lives in the live-submission directory the analysis
# scripts already point at.
SUBMITTED_DIR_REL = "output/mentalriskes/predictions"
SUBMITTED_T2_DIR_REL = "output/mentalriskes_task2/server_submissions"
# The replay output lands here (server-format JSONs after running the
# JSONL converter).
REPLAY_DIR_REL = "output/mentalriskes_test_replay/predictions"
REPLAY_T2_DIR_REL = "output/mentalriskes_task2_test_replay/server_submissions"

# Task 2 leaderboard reference values (best per-team accuracies seen so far)
T2_LEADERBOARD = {0: 0.210000, 1: 0.236667, 2: 0.246667}
T2_LEADERBOARD_TOP = 0.392606  # NLP Innovators Run 1 — winner
T2_LEADERBOARD_RANDOM = 0.362676  # random baseline


def _load_predictions_from_dir(predictions_dir: Path, run_idx: int) -> dict[int, dict[str, dict]]:
    """Return {round: {sid: prediction_dict}} for one run, reading server-format JSONs."""
    out: dict[int, dict[str, dict]] = {}
    for fp in sorted(predictions_dir.glob(f"round*_run{run_idx}.json")):
        rnd = int(fp.stem.split("_")[0].replace("round", ""))
        with open(fp, encoding="utf-8") as fh:
            payload = json.load(fh)
        out[rnd] = {entry["id"]: entry["prediction"] for entry in payload[0]["predictions"]}
    return out


def _load_t2_predictions_from_dir(predictions_dir: Path, run_idx: int) -> dict[int, dict[str, int]]:
    """Return {round: {sid: chosen_option_int}} for Task 2 server-format JSONs."""
    out: dict[int, dict[str, int]] = {}
    for fp in sorted(predictions_dir.glob(f"round*_run{run_idx}.json")):
        rnd = int(fp.stem.split("_")[0].replace("round", ""))
        with open(fp, encoding="utf-8") as fh:
            payload = json.load(fh)
        out[rnd] = {entry["id"]: int(entry["prediction"]) for entry in payload[0]["predictions"]}
    return out


def _opt_to_int(opt: str) -> int:
    return int(opt.replace("option_", ""))


def _last_per_session(rounds: dict[int, dict[str, dict]]) -> dict[str, tuple[int, dict]]:
    """For each session, return the highest-round prediction available."""
    last: dict[str, tuple[int, dict]] = {}
    for rnd, sessions in rounds.items():
        for sid, pred in sessions.items():
            if sid not in last or rnd > last[sid][0]:
                last[sid] = (rnd, pred)
    return last


def run() -> None:
    cfg = load_config()
    gold = load_task1_gold(cfg)
    out_dir = repo_path(cfg["paths"]["output_dir"])
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    submitted_dir = repo_path(SUBMITTED_DIR_REL)
    replay_dir = repo_path(REPLAY_DIR_REL)

    print("=== Post-Hoc Analysis W: Submitted (R30) vs Full Replay ===\n")
    print(f"  Submitted dir : {submitted_dir}")
    print(f"  Replay dir    : {replay_dir}\n")

    rows = []
    trajectory_records: list[dict] = []

    for run_meta in cfg["team"]["runs"]:
        run_idx = run_meta["idx"]
        run_label = run_meta["label"]

        sub_rounds = _load_predictions_from_dir(submitted_dir, run_idx)
        rep_rounds = _load_predictions_from_dir(replay_dir, run_idx)
        if not sub_rounds:
            print(f"[run {run_idx}] no submitted predictions on disk; skipping")
            continue
        if not rep_rounds:
            print(f"[run {run_idx}] no replay predictions on disk yet; skipping")
            continue

        sub_last = _last_per_session(sub_rounds)
        rep_last = _last_per_session(rep_rounds)

        common_sessions = sorted(set(gold.keys()) & set(sub_last.keys()) & set(rep_last.keys()))

        for sid in common_sessions:
            sub_round, sub_pred = sub_last[sid]
            rep_round, rep_pred = rep_last[sid]
            for instr in ("PHQ-9", "GAD-7", "CompACT-10"):
                g = gold[sid][instr]
                gtot = total(g)
                gband = classify_band(gtot, instr, cfg)

                sub_items = sub_pred[instr]
                rep_items = rep_pred[instr]
                sub_total = total(sub_items)
                rep_total = total(rep_items)

                rows.append({
                    "run": run_idx,
                    "run_label": run_label,
                    "session": sid,
                    "instrument": instr,
                    "submitted_round": sub_round,
                    "replay_round": rep_round,
                    "gold_total": gtot,
                    "submitted_total": sub_total,
                    "replay_total": rep_total,
                    "submitted_bias": sub_total - gtot,
                    "replay_bias": rep_total - gtot,
                    "bias_shift": (rep_total - gtot) - (sub_total - gtot),
                    "submitted_MAE_items": mae(sub_items, g),
                    "replay_MAE_items": mae(rep_items, g),
                    "delta_MAE_items": mae(rep_items, g) - mae(sub_items, g),
                    "gold_band": gband,
                    "submitted_band": classify_band(sub_total, instr, cfg),
                    "replay_band": classify_band(rep_total, instr, cfg),
                })

            # Trajectory: per-round totals across the full replay for this session
            for rnd in sorted(rep_rounds.keys()):
                if sid not in rep_rounds[rnd]:
                    continue
                p = rep_rounds[rnd][sid]
                for instr in ("PHQ-9", "GAD-7", "CompACT-10"):
                    trajectory_records.append({
                        "run": run_idx,
                        "session": sid,
                        "instrument": instr,
                        "round": rnd,
                        "pred_total": total(p[instr]),
                        "gold_total": total(gold[sid][instr]),
                    })

    if not rows:
        print("No comparable predictions yet (replay still warming up). Re-run when the replay has progressed.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "W_per_session_comparison.csv", index=False)
    print(f"Wrote {out_dir / 'W_per_session_comparison.csv'} ({len(df)} rows)")

    # Per-run aggregate
    agg = (
        df.groupby(["run", "instrument"])
        .agg(
            n=("session", "nunique"),
            submitted_MAE=("submitted_MAE_items", "mean"),
            replay_MAE=("replay_MAE_items", "mean"),
            delta_MAE=("delta_MAE_items", "mean"),
            mean_submitted_bias=("submitted_bias", "mean"),
            mean_replay_bias=("replay_bias", "mean"),
        )
        .reset_index()
    )
    agg.to_csv(out_dir / "W_per_run_aggregate.csv", index=False)
    print(f"\nW_per_run_aggregate.csv:")
    print(agg.to_string(index=False))

    # Band accuracy
    df["submitted_band_correct"] = (df["submitted_band"] == df["gold_band"]).astype(int)
    df["replay_band_correct"] = (df["replay_band"] == df["gold_band"]).astype(int)
    band_acc = (
        df[df["instrument"].isin(["PHQ-9", "GAD-7"])]
        .groupby(["run", "instrument"])
        .agg(
            submitted_band_acc=("submitted_band_correct", "mean"),
            replay_band_acc=("replay_band_correct", "mean"),
        )
        .reset_index()
    )
    band_acc.to_csv(out_dir / "W_per_run_band_acc.csv", index=False)
    print(f"\nW_per_run_band_acc.csv:")
    print(band_acc.to_string(index=False))

    # Rank projection: project replay-based MAE_Combined onto the leaderboard
    rank_rows = []
    try:
        xlsx = repo_path(cfg["paths"]["leaderboard_xlsx"])
        leaderboard = pd.read_excel(xlsx, sheet_name="Task1")
    except Exception as e:
        print(f"\n(skipping rank projection: {e})")
        leaderboard = None

    if leaderboard is not None:
        for run_idx in sorted(df["run"].unique()):
            sub = agg[agg["run"] == run_idx]
            if len(sub) < 3:
                continue
            replay_combined = sub["replay_MAE"].mean()
            submitted_combined = sub["submitted_MAE"].mean()
            n_below = int((leaderboard["MAE_Combined"] < replay_combined).sum())
            rank_rows.append({
                "run": run_idx,
                "submitted_MAE_combined": submitted_combined,
                "replay_MAE_combined": replay_combined,
                "delta": replay_combined - submitted_combined,
                "projected_rank": n_below + 1,
            })
        rank_df = pd.DataFrame(rank_rows)
        rank_df.to_csv(out_dir / "W_rank_projection.csv", index=False)
        print(f"\nW_rank_projection.csv:")
        print(rank_df.to_string(index=False))

    # ---------------------------------------------------------------------
    # Task 2 comparison: submitted (R1-30) vs replay (R1-82)
    # ---------------------------------------------------------------------
    print("\n=== Task 2 comparison (submitted R1-30 vs replay R1-82) ===\n")
    t2_gold = load_task2_gold(cfg)
    t2_submitted_dir = repo_path(SUBMITTED_T2_DIR_REL)
    t2_replay_dir = repo_path(REPLAY_T2_DIR_REL)

    t2_rows = []
    t2_tercile_rows = []
    full_rounds = sorted(t2_gold.keys())
    max_round = max(full_rounds) if full_rounds else 0
    tercile_bounds = (max_round // 3, 2 * max_round // 3) if max_round else (0, 0)

    def _tercile(rnd: int) -> str:
        if rnd <= tercile_bounds[0]: return "early"
        if rnd <= tercile_bounds[1]: return "mid"
        return "late"

    for run_idx in (0, 1, 2):
        sub_t2 = _load_t2_predictions_from_dir(t2_submitted_dir, run_idx)
        rep_t2 = _load_t2_predictions_from_dir(t2_replay_dir, run_idx)
        if not sub_t2 and not rep_t2:
            continue

        # Submitted: how many correct out of R1-30 inner-join with predictions
        sub_correct, sub_total = 0, 0
        rep_correct_r1_30, rep_total_r1_30 = 0, 0
        rep_correct_full, rep_total_full = 0, 0
        rep_tercile = {"early": [0, 0], "mid": [0, 0], "late": [0, 0]}  # [correct, total]

        for rnd, sess_gold in t2_gold.items():
            for sid, opt_str in sess_gold.items():
                g = _opt_to_int(opt_str)
                if rnd in sub_t2 and sid in sub_t2[rnd]:
                    sub_total += 1
                    if g == sub_t2[rnd][sid]:
                        sub_correct += 1
                if rnd in rep_t2 and sid in rep_t2[rnd]:
                    rep_total_full += 1
                    correct = int(g == rep_t2[rnd][sid])
                    if correct:
                        rep_correct_full += 1
                    if rnd <= 30:
                        rep_total_r1_30 += 1
                        if correct:
                            rep_correct_r1_30 += 1
                    bucket = _tercile(rnd)
                    rep_tercile[bucket][0] += correct
                    rep_tercile[bucket][1] += 1

        sub_acc = sub_correct / sub_total if sub_total else None
        rep_acc_r1_30 = rep_correct_r1_30 / rep_total_r1_30 if rep_total_r1_30 else None
        rep_acc_full = rep_correct_full / rep_total_full if rep_total_full else None
        t2_rows.append({
            "run": run_idx,
            "submitted_n": sub_total,
            "submitted_acc": sub_acc,
            "leaderboard_acc": T2_LEADERBOARD.get(run_idx),
            "replay_n_r1_30": rep_total_r1_30,
            "replay_acc_r1_30": rep_acc_r1_30,
            "replay_n_full": rep_total_full,
            "replay_acc_full": rep_acc_full,
            "delta_replay_full_minus_submitted": (rep_acc_full - sub_acc) if rep_acc_full and sub_acc else None,
        })
        for tname in ("early", "mid", "late"):
            corr, tot = rep_tercile[tname]
            t2_tercile_rows.append({
                "run": run_idx,
                "tercile": tname,
                "n": tot,
                "acc": corr / tot if tot else None,
            })

    if t2_rows:
        t2_df = pd.DataFrame(t2_rows)
        t2_df.to_csv(out_dir / "W_t2_round_decomposition.csv", index=False)
        print("W_t2_round_decomposition.csv:")
        print(t2_df.to_string(index=False))
    if t2_tercile_rows:
        t2_terc_df = pd.DataFrame(t2_tercile_rows)
        t2_terc_df.to_csv(out_dir / "W_t2_round_tercile.csv", index=False)
        print("\nW_t2_round_tercile.csv:")
        print(t2_terc_df.to_string(index=False))

    # Task 2 rank projection
    if t2_rows:
        try:
            xlsx = repo_path(cfg["paths"]["leaderboard_xlsx"])
            lb2 = pd.read_excel(xlsx, sheet_name="Task2")
        except Exception as e:
            print(f"\n(skipping Task 2 rank projection: {e})")
            lb2 = None
        if lb2 is not None:
            rank_t2 = []
            for r in t2_rows:
                if r["replay_acc_full"] is None:
                    continue
                # Leaderboard ranks by Overall Accuracy descending
                n_above = int((lb2["Overall Accuracy"] > r["replay_acc_full"]).sum())
                rank_t2.append({
                    "run": r["run"],
                    "submitted_acc": r["submitted_acc"],
                    "replay_acc_full": r["replay_acc_full"],
                    "delta": r["delta_replay_full_minus_submitted"],
                    "projected_rank": n_above + 1,
                    "vs_random_baseline": r["replay_acc_full"] - T2_LEADERBOARD_RANDOM,
                    "vs_top_team": r["replay_acc_full"] - T2_LEADERBOARD_TOP,
                })
            if rank_t2:
                rank_df = pd.DataFrame(rank_t2)
                rank_df.to_csv(out_dir / "W_t2_rank_projection.csv", index=False)
                print("\nW_t2_rank_projection.csv:")
                print(rank_df.to_string(index=False))

    # Per-patient trajectory plots
    if trajectory_records:
        traj = pd.DataFrame(trajectory_records)
        primary_run = 2
        traj_r2 = traj[traj["run"] == primary_run]
        for sid in sorted(traj_r2["session"].unique()):
            for instr in ("PHQ-9", "GAD-7", "CompACT-10"):
                sub = traj_r2[(traj_r2["session"] == sid) & (traj_r2["instrument"] == instr)].sort_values("round")
                if sub.empty:
                    continue
                fig, ax = plt.subplots(figsize=(7, 3))
                ax.plot(sub["round"], sub["pred_total"], marker="o", linewidth=1, label="replay total")
                ax.axhline(sub["gold_total"].iloc[0], color="black", linestyle="--", label="gold total")
                ax.axvline(30, color="red", linestyle=":", label="submitted boundary (R30)")
                ax.set_xlabel("Round")
                ax.set_ylabel("Total score")
                ax.set_title(f"{sid} — {instr} — Run {primary_run}")
                ax.legend(fontsize=8)
                fig.tight_layout()
                fp = figures_dir / f"W_trajectory_{sid}_{instr.replace('-', '')}_run{primary_run}.png"
                fig.savefig(fp, dpi=120)
                plt.close(fig)
        print(f"\nWrote per-patient trajectory plots to {figures_dir}")


if __name__ == "__main__":
    run()
