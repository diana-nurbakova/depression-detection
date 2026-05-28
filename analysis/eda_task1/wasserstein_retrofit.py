"""Retrofit per-round Wasserstein (W1) anomaly trajectories for eRisk 2026 Task 1.

CONTEXT / HONESTY NOTE
----------------------
During the official Task 1 runs the POT library was NOT installed
(`pot_available: false` in every internal_*.json log), so NO Wasserstein
distance was ever computed live. Moreover `W_self` is never read by the
orchestrator (`get_orchestrator_context` only consumes coverage gaps and
`W_align`). The live ToM guidance used symptom-COVERAGE signals, not W1.

This script therefore performs a *post-hoc reconstruction*: it reloads the
per-turn expressed symptom profiles E(t) that WERE logged (they need no POT)
and recomputes W1 trajectories + firing flags. These results are diagnostic
(belong in a results/analysis section), NOT a description of the live policy.

COVERAGE LIMITATION
-------------------
Only the ToM-era logs (personas 12-19, Run 3 = internal_3.json) saved
per-turn E_profiles. Personas 1-11 (incl. Daniel, pid 6) and Runs 1-2 logged
empty ToM, so W1 cannot be retrofitted for them without re-running the
assessor pipeline (LLM). Those persona/run combos are reported as
"not_retrofittable".

TWO CONVENTIONS (computed side by side)
---------------------------------------
1. barycenter : W1(E(t), running-mean(E(1..t)))   -- matches
   src/mentalriskes/task1/temporal.compute_w1_trajectory and the
   MentalRiskES trial CSV. Designed as a temporal anomaly detector.
2. w_self     : W1(E(t), E(t-1))                   -- the turn-over-turn
   shift the eRisk Task 1 tracker (tom.compute_wasserstein_metrics) computes
   as W_self[t][1].

Both use the same clinical ground metric (tom.get_cost_matrix), L1-normalise
each profile (shape-only), and the same firing rule as
temporal.detect_anomalous_rounds: fire when W1 > running_mu + 2*running_sigma,
stats over rounds strictly before the current one, no firing for the first 3
trajectory points (warmup), std floored with +1e-6.

Outputs:
  analysis/eda_task1/wasserstein_retrofit.json   (full + example slices)
"""

from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from erisk_task1.tom import BDI_SHORT, get_cost_matrix, wasserstein_balanced  # noqa: E402

import ot  # noqa: E402  (POT is available now, post-hoc)

OUT = ROOT / "analysis" / "eda_task1" / "wasserstein_retrofit.json"
GOLD_PATH = (
    ROOT / "data" / "eRisk-2026" / "eRisk26-datasets-20260519T175618Z-3-001"
    / "eRisk26-datasets" / "task1-llms" / "golden-data" / "patients_data.jsonl"
)
SUBMISSION = ROOT / "runs" / "task1" / "official_submission" / "task1-llms-results"

THRESHOLD_FACTOR = 2.0
WARMUP = 3  # first WARMUP trajectory points never fire

# Normalise curly punctuation in logged transcripts to ASCII for paper-ready quotes.
_PUNCT = {
    "’": "'", "‘": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "…": "...", "�": "'",
}


def clean(text: str) -> str:
    for k, v in _PUNCT.items():
        text = text.replace(k, v)
    return text


def label(j: int) -> str:
    return BDI_SHORT.get(j + 1, str(j + 1))


def emd(a: np.ndarray, b: np.ndarray, M: np.ndarray) -> float:
    """L1-normalised balanced W1 under cost matrix M (shape-only). 0 if degenerate."""
    sa, sb = float(a.sum()), float(b.sum())
    if sa <= 1e-12 or sb <= 1e-12:
        return 0.0
    return float(ot.emd2(a / sa, b / sb, M.astype(np.float64)))


def fire_flags(traj: list[float]) -> list[dict]:
    """Running mu+2*sigma firing, matching temporal.detect_anomalous_rounds."""
    rows = []
    for k in range(len(traj)):
        if k < WARMUP:
            mu = None if k == 0 else float(np.mean(traj[:k]))
            sigma = None if k <= 1 else float(np.std(traj[:k]))
            thr = None if (mu is None or sigma is None) else mu + THRESHOLD_FACTOR * sigma
            fired = False
        else:
            past = traj[:k]
            mu = float(np.mean(past))
            sigma = float(np.std(past)) + 1e-6
            thr = mu + THRESHOLD_FACTOR * sigma
            fired = bool(traj[k] > thr)
        rows.append({
            "running_mu_through_prev": None if mu is None else round(mu, 4),
            "running_sigma_through_prev": None if sigma is None else round(sigma, 4),
            "threshold_mu_plus_2sigma": None if thr is None else round(thr, 4),
            "fired": fired,
        })
    return rows


def load_gold() -> tuple[dict, dict, dict]:
    name, gold_bdi, gold_syms = {}, {}, {}
    for line in open(GOLD_PATH, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        pid = int(d["patient_id"])
        name[pid] = d["patient_name"]
        gold_bdi[pid] = d["bdi_score"]
        gold_syms[pid] = d.get("patient_key_symptoms")
    return name, gold_bdi, gold_syms


def discover_eprofile_logs() -> dict:
    """Map (persona_folder, run) -> best internal_*.json path with E_profiles."""
    have: dict[tuple[str, str], tuple[str, int]] = {}
    for f in glob.glob(str(ROOT / "runs" / "task1" / "**" / "internal_*.json"), recursive=True):
        base = os.path.basename(f)
        run = base.split("_")[1].split(".")[0]
        parts = Path(f).parts
        persona = [p for p in parts if p.startswith("persona")][-1]
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        tom = d.get("tom") or d.get("tom_summary") or {}
        n = len(tom.get("E_profiles", {})) if tom else 0
        if n > 0:
            key = (persona, run)
            if key not in have or n >= have[key][1]:
                have[key] = (f, n)
    return have


def load_transcript(pid: int, run: str) -> dict[int, str]:
    """assistant turn index (1-based) -> message text, from official submission."""
    p = SUBMISSION / f"persona{pid}" / f"interactions_{run}.json"
    out: dict[int, str] = {}
    if not p.exists():
        return out
    conv = json.load(open(p, encoding="utf-8"))[0]["conversation"]
    t = 0
    for m in conv:
        if m["role"] == "assistant":
            t += 1
            out[t] = clean(m["message"])
    return out


def symptom_delta(prev: np.ndarray, cur: np.ndarray) -> dict[str, list[int]]:
    return {
        label(j): [int(prev[j]), int(cur[j])]
        for j in range(21) if cur[j] > prev[j]
    }


# W_align interpretation thresholds, from tom.get_orchestrator_context.
def align_hint(w):
    if w is None:
        return None
    if w > 1.0:
        return "high_misalignment"
    if w > 0.5:
        return "moderate_misalignment"
    return "good_alignment"


def process_persona(path: str, pid: int, run: str, transcript: dict[int, str], M):
    d = json.load(open(path, encoding="utf-8"))
    tom = d["tom"]
    ep = {int(k): np.array(v, dtype=float) for k, v in tom["E_profiles"].items()}
    ip = {int(k): np.array(v, dtype=float) for k, v in tom.get("I_profiles", {}).items()}
    turns = sorted(ep)

    bary, wself, walign = [], [], []
    for i, t in enumerate(turns):
        sub = np.array([ep[u] for u in turns[: i + 1]])
        bary.append(emd(sub.mean(axis=0), ep[t], M))
        wself.append(0.0 if i == 0 else emd(ep[turns[i - 1]], ep[t], M))
        walign.append(emd(ip[t], ep[t], M) if t in ip else None)

    fb = fire_flags(bary)
    fw = fire_flags(wself)

    rounds = []
    for i, t in enumerate(turns):
        prev = ep[turns[i - 1]] if i > 0 else np.zeros(21)
        rounds.append({
            "round": t,
            "expressed_mass": round(float(ep[t].sum()), 2),
            "new_or_increased_symptoms": symptom_delta(prev, ep[t]),
            "persona_utterance": transcript.get(t),
            "barycenter": {"w1": round(bary[i], 4), **fb[i]},
            "w_self": {"w1": round(wself[i], 4), **fw[i]},
            "w_align": {
                "w1": None if walign[i] is None else round(walign[i], 4),
                "hint": align_hint(walign[i]),
            },
        })

    mass = [float(ep[t].sum()) for t in turns]
    n = len(turns)
    return {
        "persona_id": pid,
        "predicted_bdi": d.get("bdi-score"),
        "predicted_band": d.get("severity_band"),
        "source_log": os.path.relpath(path, ROOT),
        "n_rounds": n,
        "fired_rounds_barycenter": [turns[i] for i in range(n) if fb[i]["fired"]],
        "fired_rounds_w_self": [turns[i] for i in range(n) if fw[i]["fired"]],
        # Counterfactual signal (reconstructed from logs; never live):
        "w_align_misalignment_rounds": [
            turns[i] for i in range(n)
            if walign[i] is not None and walign[i] > 0.5
        ],
        "expressed_mass_trajectory": [int(m) for m in mass],
        "mass_rising_at_final_round": bool(n >= 2 and mass[-1] > mass[-2]),
        "rounds": rounds,
    }


def main():
    M = get_cost_matrix()
    name, gold_bdi, gold_syms = load_gold()
    have = discover_eprofile_logs()

    personas = {}
    for (persona_folder, run), (path, _n) in sorted(have.items()):
        pid = int(persona_folder.replace("persona", ""))
        transcript = load_transcript(pid, run)
        rec = process_persona(path, pid, run, transcript, M)
        rec["persona_name"] = name.get(pid)
        rec["gold_bdi"] = gold_bdi.get(pid)
        rec["gold_band"] = band_of(gold_bdi.get(pid))
        rec["run"] = int(run)
        personas[f"{rec['persona_name']}_run{run}"] = rec

    retrofittable_pids = sorted({r["persona_id"] for r in personas.values()})
    not_retro = [
        {"persona_id": pid, "persona_name": name.get(pid), "gold_bdi": gold_bdi.get(pid)}
        for pid in sorted(name) if pid not in retrofittable_pids
    ]

    out = {
        "title": "eRisk 2026 Task 1 - retrofitted Wasserstein (W1) trajectories",
        "honesty_note": (
            "POST-HOC reconstruction. POT was unavailable during the official runs "
            "(pot_available=false in every log) so NO W1 was computed live; the eRisk "
            "tracker's W_self is never read by the orchestrator either (only coverage "
            "gaps + W_align feed get_orchestrator_context, and W_align was empty too). "
            "These W1 values informed analysis, not the live interviewing policy."
        ),
        "method": {
            "ground_metric": "tom.get_cost_matrix (clinical BDI-II item cost)",
            "normalisation": "each E(t) L1-normalised -> profile shape only (severity-mass invariant)",
            "barycenter": "W1(E(t), running-mean(E(1..t)))  [temporal.compute_w1_trajectory / trial CSV]",
            "w_self": "W1(E(t), E(t-1)) = eRisk W_self[t][1]  [tom.compute_wasserstein_metrics]",
            "firing_rule": "fire iff W1 > running_mu + 2*running_sigma over rounds strictly before t",
            "warmup": "first 3 trajectory points never fire; sigma floored with +1e-6",
            "threshold_factor": THRESHOLD_FACTOR,
        },
        "coverage": {
            "retrofittable_persona_ids": retrofittable_pids,
            "note": (
                "Only Run-3 ToM-era logs (personas 12-19) saved per-turn E_profiles. "
                "Personas 1-11 (incl. Daniel pid 6) and Runs 1-2 logged empty ToM and "
                "are not retrofittable from logs without re-running the assessor pipeline."
            ),
            "not_retrofittable": not_retro,
        },
        "counterfactual": {
            "question": "What could W1 have changed in the live system?",
            "answerable": (
                "PARTIALLY. The signal the live policy would have seen is reconstructable "
                "from logs (fact). Its effect on final predictions is NOT measurable without "
                "re-running, since acting on it changes the questions and the personas would "
                "respond differently (hypothesis)."
            ),
            "live_levers": [
                "W_align hint -> orchestrator question selection (already plumbed into "
                "get_orchestrator_context / orchestrator.py:246; only needs POT).",
                "A W1 fire -> defer early termination (should_terminate allows early stop "
                "after min_turns=5 once band stable; orchestrator.py:305). Not currently wired.",
            ],
            "w_align_signal_fact": {
                key: {
                    "misalignment_rounds": r["w_align_misalignment_rounds"],
                    "hints": [rd["w_align"]["hint"] for rd in r["rounds"]],
                }
                for key, r in personas.items()
            },
            "termination_timing_fact": {
                key: {
                    "n_rounds": r["n_rounds"],
                    "min_turns": 5, "max_turns": 10,
                    "fired_rounds_barycenter": r["fired_rounds_barycenter"],
                    "mass_trajectory": r["expressed_mass_trajectory"],
                    "mass_rising_at_final_round": r["mass_rising_at_final_round"],
                }
                for key, r in personas.items()
            },
            "verdict": (
                "W_align cleanly separates the two pathological conversations (Javier loop, "
                "Priya over-probe) from the six healthy ones -> real diagnostic value. "
                "Plausible-but-unverified live wins: break Javier's loop, curb Priya's "
                "over-prediction, defer Linda's premature cutoff (fired t4, terminated t5 at "
                "min_turns with mass still rising 15->19). NO effect on the dominant error "
                "(severe under-prediction Marco 38->18, Maria 40->14) -> that is an assessor "
                "calibration problem orthogonal to W1."
            ),
        },
        "highlighted_examples": build_highlights(personas),
        "personas": personas,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # Console summary
    print(f"Wrote {OUT.relative_to(ROOT)}")
    print(f"Retrofittable personas: {retrofittable_pids}")
    print(f"Not retrofittable (no logged E_profiles): {[r['persona_id'] for r in not_retro]}")
    print("\nFiring summary (barycenter / w_self):")
    for key, r in personas.items():
        print(f"  {key:16s} gold={r['gold_bdi']:>2} pred={r['predicted_bdi']:>2} "
              f"rounds={r['n_rounds']}  bary_fires={r['fired_rounds_barycenter']}  "
              f"wself_fires={r['fired_rounds_w_self']}")


def band_of(bdi):
    if bdi is None:
        return None
    return ("minimal" if bdi <= 9 else "mild" if bdi <= 18
            else "moderate" if bdi <= 29 else "severe")


def build_highlights(personas: dict) -> dict:
    """Curate the paper-ready slices."""
    def slice_round(rec, rnd):
        return next((r for r in rec["rounds"] if r["round"] == rnd), None)

    hi = {}
    marco = personas.get("Marco_run3")
    if marco:
        hi["primary_positive"] = {
            "persona": "Marco", "persona_id": 15, "run": 3,
            "why": ("Named Daniel-substitute (Daniel pid6 predates ToM logging, not "
                    "retrofittable). Clean monotonic disclosure over 7 rounds. Barycenter "
                    "W1 fires at t4 on an interpretable anhedonia/worthlessness disclosure; "
                    "W_self never fires (it misses the disclosure -> the eRisk-native "
                    "metric's failure mode, shown side by side)."),
            "fired_rounds_barycenter": marco["fired_rounds_barycenter"],
            "fired_rounds_w_self": marco["fired_rounds_w_self"],
            "disclosure_round": 4,
            "cause": ("Rounds 1-2 are purely somatic (energy/sleep/fatigue); t3 adds "
                      "sadness/grief (lost sister); at t4 Marco discloses anhedonia + "
                      "withdrawal + worthlessness ('I just focus on work and staying "
                      "tired. It's easier than trying to pretend I'm okay'), shifting the "
                      "symptom-distribution centre of mass -> barycenter W1 spikes."),
            "note_on_strength": ("Marginal fire (W1 just clears the threshold): the early "
                                 "rounds 2-3 disclosure inflates the running baseline, so a "
                                 "genuine mid-conversation disclosure only barely trips "
                                 "mu+2sigma. This is the honest behaviour on clean trajectories."),
            "rounds": marco["rounds"],
        }
    linda = personas.get("Linda_run3")
    if linda:
        hi["secondary_positive"] = {
            "persona": "Linda", "persona_id": 14, "run": 3,
            "why": "Second named substitute; barycenter W1 fires at t4 on a self-identity/cognitive disclosure.",
            "fired_rounds_barycenter": linda["fired_rounds_barycenter"],
            "disclosure_round": 4,
            "cause": ("After somatic/anhedonia rounds 1-3, t4 discloses loss of competence "
                      "and indecisiveness ('I used to be the one with the answers... Now I "
                      "don't know. I feel like I've lost that part of myself') -> Past "
                      "failure, Self-criticalness, Indecisiveness, Concentration appear."),
            "rounds": linda["rounds"],
        }
    javier = personas.get("Javier_run3")
    if javier:
        hi["failure_mode_spurious_fire"] = {
            "persona": "Javier", "persona_id": 12, "run": 3,
            "why": ("Honest negative result for the results section. Barycenter W1 fires "
                    "STRONGLY at t4 (and W_self at t7) but NOT because of disclosure: the "
                    "interviewer loops (t5/t6 are near-identical questions and answers) and "
                    "the assessor drops then re-adds whole symptom clusters, so the "
                    "expressed mass oscillates 15->6->15->6->24. The spikes track assessor "
                    "instability, not the persona -> a false positive (fails the "
                    "interpretable-cause criterion)."),
            "fired_rounds_barycenter": javier["fired_rounds_barycenter"],
            "fired_rounds_w_self": javier["fired_rounds_w_self"],
            "expressed_mass_trajectory": [r["expressed_mass"] for r in javier["rounds"]],
            "rounds": javier["rounds"],
        }
    return hi


if __name__ == "__main__":
    main()
