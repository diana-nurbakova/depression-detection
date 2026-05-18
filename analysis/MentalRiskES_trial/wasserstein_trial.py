"""Compute per-round Wasserstein (W1) anomaly trajectories on the trial
conversation, populating the extraction spec at
`specs/MentalRiskES/wasserstein-extraction-spec.md`.

Uses the same algorithm as `src/mentalriskes/task1/temporal.py`:
  - Per-instrument convention: treat each round's per-item score vector as
    an empirical distribution over items (normalised by sum).
  - For round k, compute W1 between the round-k distribution and the
    *running barycenter* (mean across rounds 1..k), under the instrument's
    clinical ground metric M.
  - Threshold: mu_{<r} + 2 * sigma_{<r}, where mu/sigma are running stats
    over the W1 trajectory up to (but not including) the current round.
    For k < 3 the implementation does not flag (warmup window).

Outputs:
  - analysis/MentalRiskES_trial/outputs/trial_wasserstein.csv (long format)
  - analysis/MentalRiskES_trial/outputs/trial_wasserstein_session_stats.csv
  - analysis/MentalRiskES_trial/outputs/trial_wasserstein_firing.json
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

# Make sure src/ is importable
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mentalriskes.task1.temporal import (
    INSTRUMENT_SPECS,
    SessionPredictionHistory,
    compute_w1_trajectory,
    detect_anomalous_rounds,
    get_ground_metric,
)


TRIAL_LOG = ROOT / "output" / "mentalriskes" / "trial_results_all.json"
OUT_DIR = ROOT / "analysis" / "MentalRiskES_trial" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_runs() -> dict:
    with open(TRIAL_LOG, "r", encoding="utf-8") as f:
        return json.load(f)


def build_history(run_rounds: list[dict]) -> SessionPredictionHistory:
    history = SessionPredictionHistory(session_id="trial")
    for r in run_rounds:
        history.add_round(
            r["round"], r["phq9"], r["gad7"], r["compact10"]
        )
    return history


def main() -> None:
    data = load_runs()
    gold = data["gold"]
    runs = data["runs"]

    rows = []
    session_rows = []
    firing = {}

    for run_id, payload in runs.items():
        history = build_history(payload["rounds"])
        firing[run_id] = {}

        for inst, matrix in history.matrices.items():
            traj = compute_w1_trajectory(matrix)
            anomalies = detect_anomalous_rounds(traj, threshold_factor=2.0)

            # Per-round rows: include w1, running mu/sigma/threshold computed
            # the same way detect_anomalous_rounds does internally, plus the
            # firing flag.
            running_firing = []
            for k, (_, w1, is_anom) in enumerate(anomalies):
                round_n = matrix.round_numbers[k]
                past = traj[:k]
                if k < 3:
                    mu = float("nan") if k == 0 else float(np.mean(past))
                    sigma = float("nan") if k <= 1 else float(np.std(past))
                    threshold = float("nan") if k <= 1 else mu + 2 * sigma
                else:
                    mu = float(np.mean(past))
                    sigma = float(np.std(past))
                    threshold = mu + 2 * sigma
                rows.append({
                    "run": run_id,
                    "instrument": inst,
                    "round": round_n,
                    "w1": round(w1, 4),
                    "running_mu_through_prev": round(mu, 4) if not np.isnan(mu) else "",
                    "running_sigma_through_prev": round(sigma, 4) if not np.isnan(sigma) else "",
                    "threshold_mu_plus_2sigma": round(threshold, 4) if not np.isnan(threshold) else "",
                    "fired": bool(is_anom),
                })
                if is_anom:
                    running_firing.append(round_n)

            firing[run_id][inst] = running_firing

            # Session-level statistics (excluding the round-1 degenerate 0)
            #  - mu_session over W1(2..19)
            #  - sigma_session over W1(2..19)
            #  - session-level threshold mu + 2*sigma
            traj_arr = np.array(traj[1:], dtype=float)
            mu_sess = float(np.mean(traj_arr))
            sigma_sess = float(np.std(traj_arr))
            thr_sess = mu_sess + 2 * sigma_sess
            # Also report including round 1 for transparency
            traj_full = np.array(traj, dtype=float)
            mu_full = float(np.mean(traj_full))
            sigma_full = float(np.std(traj_full))
            thr_full = mu_full + 2 * sigma_full
            session_rows.append({
                "run": run_id,
                "instrument": inst,
                "rounds_used_for_session_stats": "2-19",
                "mu_session": round(mu_sess, 4),
                "sigma_session": round(sigma_sess, 4),
                "threshold_session": round(thr_sess, 4),
                "mu_session_incl_round1": round(mu_full, 4),
                "sigma_session_incl_round1": round(sigma_full, 4),
                "threshold_session_incl_round1": round(thr_full, 4),
                "rounds_firing_running_threshold": ", ".join(
                    str(r) for r in running_firing
                ) if running_firing else "none",
            })

    # Write long-format CSV
    with open(OUT_DIR / "trial_wasserstein.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Write session-stats CSV
    with open(OUT_DIR / "trial_wasserstein_session_stats.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(session_rows[0].keys()))
        writer.writeheader()
        writer.writerows(session_rows)

    # Write firing rounds JSON
    with open(OUT_DIR / "trial_wasserstein_firing.json", "w", encoding="utf-8") as f:
        json.dump(firing, f, indent=2)

    print(f"Wrote {len(rows)} per-round rows across {len(runs)} runs x 3 instruments")
    print(f"Per-round CSV   : {OUT_DIR / 'trial_wasserstein.csv'}")
    print(f"Session stats   : {OUT_DIR / 'trial_wasserstein_session_stats.csv'}")
    print(f"Firing rounds   : {OUT_DIR / 'trial_wasserstein_firing.json'}")
    print()
    print("Firing rounds by run/instrument:")
    print(json.dumps(firing, indent=2))


if __name__ == "__main__":
    main()
