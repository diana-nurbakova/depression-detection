"""Exploratory data analysis of the MentalRiskES 2026 trial + simulated data.

Computes descriptive statistics across:
  - Task 1 trial: 19-round single-session transcript (Spanish therapy dialogue).
  - Task 2 trial: same conversation with 3 candidate therapist responses per round.
  - Task 1 simulated personas: 6 personas (15 rounds each) with target PHQ-9 / GAD-7
    totals + CompACT profile.
  - Task 2 simulated personas: 7 personas (~14 rounds each) with gold response labels.

Outputs (in analysis/MentalRiskES_trial/outputs/):
  trial_t1_round_stats.csv       per-round word counts, role, etc.
  trial_t2_option_stats.csv      per-round option length, gold label
  trial_t2_phase_distribution.csv per-phase gold distribution
  simulated_t1_personas.csv      one row per Task 1 persona with target scores
  simulated_t2_sessions.csv      per-session round count + gold distribution

  eda_trial_data.md              standalone Markdown report consolidating all
                                  the above into paper-section-quality tables.

Run:
  python analysis/MentalRiskES_trial/eda_trial_data.py

The script is idempotent: re-running overwrites everything in
analysis/MentalRiskES_trial/outputs/.
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "analysis/MentalRiskES_trial/outputs"


# Per the Task 2 solution doc §2.1 — the trial's 19 rounds map to therapeutic phases.
TASK2_TRIAL_PHASES: dict[int, str] = {
    1: "crisis/engagement",
    2: "committed_action", 3: "committed_action",
    4: "acceptance/defusion", 5: "acceptance/defusion",
    6: "defusion_deepening", 7: "defusion_deepening", 8: "defusion_deepening",
    9: "behavioral_activation", 10: "behavioral_activation", 11: "behavioral_activation", 12: "behavioral_activation",
    13: "integration", 14: "integration", 15: "integration",
    16: "self_as_context", 17: "self_as_context",
    18: "closing", 19: "closing",
}

# From src/mentalriskes/task2/data.py:TRIAL_GROUND_TRUTH — 18 labelled rounds (round 19 unlabelled).
TRIAL_GROUND_TRUTH: dict[int, int] = {
    1: 2, 2: 3, 3: 3, 4: 3, 5: 2, 6: 1, 7: 3, 8: 3, 9: 3,
    10: 2, 11: 1, 12: 3, 13: 1, 14: 3, 15: 1, 16: 1, 17: 3, 18: 2,
}

# Task 1 trial gold (from Task 1 doc §3.1)
T1_TRIAL_GOLD = {
    "PHQ-9":  {"items": [1, 2, 1, 2, 1, 2, 2, 2, 0], "total": 13, "band": "moderate"},
    "GAD-7":  {"items": [3, 2, 2, 2, 2, 1, 2], "total": 14, "band": "moderate"},
    "CompACT-10": {"items": [3, 3, 4, 3, 3, 3, 4, 3, 3, 4], "total": 33, "OtE_total": 11, "BA_total": 9, "VA_total": 14},
}


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────
def _load_trial_rounds(trial_dir: Path) -> list[dict]:
    """Load round_*.json with single-session 'trial'-keyed format."""
    out = []
    for fp in sorted(trial_dir.glob("round_*.json"), key=lambda p: int(p.stem.split("_")[1])):
        with open(fp, encoding="utf-8") as fh:
            payload = json.load(fh).get("trial", {})
        payload["_round"] = int(fp.stem.split("_")[1])
        out.append(payload)
    return out


def _load_simulated_t1() -> list[dict]:
    sim_root = REPO_ROOT / "output/mentalriskes/data_prep/simulated/task1"
    out = []
    if not sim_root.exists():
        return out
    for d in sorted(sim_root.iterdir()):
        if not d.is_dir() or not (d / "metadata.json").exists():
            continue
        with open(d / "metadata.json", encoding="utf-8") as fh:
            meta = json.load(fh)
        ts = meta.get("target_scores", {})
        out.append({
            "session_id": meta.get("session_id", d.name),
            "presentation": meta.get("profile", {}).get("description", ""),
            "profile_id": meta.get("profile", {}).get("id", ""),
            "phq9_target": ts.get("phq9_total"),
            "gad7_target": ts.get("gad7_total"),
            "compact_profile": ts.get("compact10_profile"),
            "personality": meta.get("profile", {}).get("personality", ""),
            "n_rounds": meta.get("n_rounds"),
        })
    return out


def _load_simulated_t2() -> list[dict]:
    sim_root = REPO_ROOT / "output/mentalriskes/data_prep/simulated/task2"
    out = []
    if not sim_root.exists():
        return out
    for d in sorted(sim_root.iterdir()):
        if not d.is_dir():
            continue
        labels_fp = d / "labels.json"
        meta_fp = d / "metadata.json"
        if not labels_fp.exists():
            continue
        with open(labels_fp, encoding="utf-8") as fh:
            labels = {int(k): int(v) for k, v in json.load(fh).items()}
        meta = {}
        if meta_fp.exists():
            with open(meta_fp, encoding="utf-8") as fh:
                meta = json.load(fh)
        counts = Counter(labels.values())
        n = sum(counts.values())
        out.append({
            "session_id": d.name,
            "presentation": meta.get("profile", {}).get("description", ""),
            "profile_id": meta.get("profile", {}).get("id", ""),
            "n_rounds": n,
            "opt1_count": counts.get(1, 0),
            "opt2_count": counts.get(2, 0),
            "opt3_count": counts.get(3, 0),
            "opt1_pct": counts.get(1, 0) / n if n else None,
            "opt2_pct": counts.get(2, 0) / n if n else None,
            "opt3_pct": counts.get(3, 0) / n if n else None,
        })
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────────────
def _word_count(text: str) -> int:
    return len((text or "").split())


def _task1_round_stats(rounds: list[dict]) -> pd.DataFrame:
    rows = []
    for r in rounds:
        pi = r.get("patient_input", "") or ""
        tr = r.get("therapist_response", "") or ""
        rows.append({
            "round": r["_round"],
            "patient_chars": len(pi),
            "patient_words": _word_count(pi),
            "therapist_chars": len(tr),
            "therapist_words": _word_count(tr),
            "has_therapist": bool(tr),
        })
    return pd.DataFrame(rows).sort_values("round").reset_index(drop=True)


def _task2_option_stats(rounds: list[dict]) -> pd.DataFrame:
    rows = []
    for r in rounds:
        rnd = r["_round"]
        opt1 = r.get("option_1", "") or ""
        opt2 = r.get("option_2", "") or ""
        opt3 = r.get("option_3", "") or ""
        pi = r.get("patient_input", "") or ""
        gold = TRIAL_GROUND_TRUTH.get(rnd)
        rows.append({
            "round": rnd,
            "phase": TASK2_TRIAL_PHASES.get(rnd, "unknown"),
            "patient_words": _word_count(pi),
            "opt1_words": _word_count(opt1),
            "opt2_words": _word_count(opt2),
            "opt3_words": _word_count(opt3),
            "mean_option_words": (_word_count(opt1) + _word_count(opt2) + _word_count(opt3)) / 3,
            "gold": gold,
            "gold_label": ("gold_option_1" if gold == 1 else
                           "gold_option_2" if gold == 2 else
                           "gold_option_3" if gold == 3 else "unlabelled"),
        })
    return pd.DataFrame(rows).sort_values("round").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Markdown report
# ─────────────────────────────────────────────────────────────────────────────
def _build_markdown(t1_round: pd.DataFrame, t2_option: pd.DataFrame,
                    sim_t1: list[dict], sim_t2: list[dict]) -> str:
    md: list[str] = []
    md.append("# MentalRiskES 2026 — Trial + Simulated Data EDA")
    md.append("")
    md.append("Auto-generated by [analysis/MentalRiskES_trial/eda_trial_data.py](../analysis/MentalRiskES_trial/eda_trial_data.py). "
              "Re-run that script after any data changes; this file is overwritten in place.")
    md.append("")

    # Task 1 trial — single session ----------------------------------------
    md.append("## 1. Task 1 trial corpus")
    md.append("")
    md.append("Single Spanish therapy conversation, 19 rounds, university student presenting "
              "with academic anxiety and parental pressure. The same transcript underlies both "
              "Task 1 and Task 2 trial corpora.")
    md.append("")
    md.append(f"- **Rounds:** {len(t1_round)}")
    md.append(f"- **Patient turns:** {len(t1_round)} (one per round)")
    n_therapist = int(t1_round["has_therapist"].sum())
    md.append(f"- **Therapist turns observed:** {n_therapist}  (rounds 2–{n_therapist + 1 if n_therapist else 0} carry the prior round's therapist response)")
    md.append(f"- **Total patient words:** {int(t1_round['patient_words'].sum())}")
    md.append(f"- **Total therapist words:** {int(t1_round['therapist_words'].sum())}")
    md.append(f"- **Patient words / round:** mean {t1_round['patient_words'].mean():.1f}, "
              f"median {t1_round['patient_words'].median():.0f}, "
              f"min {int(t1_round['patient_words'].min())}, "
              f"max {int(t1_round['patient_words'].max())}")
    md.append(f"- **Therapist words / round (when present):** "
              f"mean {t1_round[t1_round['has_therapist']]['therapist_words'].mean():.1f}, "
              f"median {t1_round[t1_round['has_therapist']]['therapist_words'].median():.0f}")
    md.append("")
    md.append("### Per-round transcript size")
    md.append("")
    md.append("| Round | Patient words | Therapist words (prev round) |")
    md.append("| --- | --- | --- |")
    for _, r in t1_round.iterrows():
        tr_cell = str(int(r["therapist_words"])) if r["has_therapist"] else "—"
        md.append(f"| {int(r['round'])} | {int(r['patient_words'])} | {tr_cell} |")
    md.append("")
    md.append("### Gold standard (Task 1 trial)")
    md.append("")
    md.append("| Instrument | Items | Total | Severity band |")
    md.append("| --- | --- | --- | --- |")
    for instr, gold in T1_TRIAL_GOLD.items():
        md.append(f"| {instr} | {gold['items']} | {gold['total']} | {gold.get('band', '—')} |")
    md.append("")
    md.append("**CompACT-10 subscales:** OtE total = 11 (items 3, 5, 8), BA total = 9 "
              "(items 1, 6, 9), VA total = 14 (items 2, 4, 7, 10).")
    md.append("")

    # Task 2 trial — same transcript + options ------------------------------
    md.append("## 2. Task 2 trial corpus")
    md.append("")
    md.append("Same 19-round transcript with three candidate therapist responses per round. "
              "Gold derived from matching the next-round therapist response against the three "
              "options (Task 2 solution doc §2.1); 18 of 19 rounds are labelled (round 19 has "
              "no successor round).")
    md.append("")
    labelled = t2_option[t2_option["gold"].notna()].copy()
    counts = labelled["gold"].astype(int).value_counts().sort_index()
    n_lab = len(labelled)
    md.append("### Gold class distribution")
    md.append("")
    md.append("| Gold option | Count | Share |")
    md.append("| --- | --- | --- |")
    for k in (1, 2, 3):
        c = int(counts.get(k, 0))
        share = c / n_lab if n_lab else 0
        md.append(f"| {k} | {c} | {share:.1%} |")
    md.append(f"| **Total labelled** | **{n_lab}** | 100 % |")
    md.append(f"| Unlabelled (round 19) | 1 | — |")
    md.append("")
    md.append("**Note:** option 3 is the majority gold class at 50 %; option 1 = 28 %, option 2 = 22 %. "
              "The trial gold is **not balanced** — a majority-class baseline (always option 3) "
              "would score 50 %, exceeding random (33.3 %).")
    md.append("")
    md.append("### Option lengths")
    md.append("")
    md.append(f"- **Patient turn words / round (mean):** {t2_option['patient_words'].mean():.1f}")
    md.append(f"- **Option words / round (mean across all three):** {t2_option['mean_option_words'].mean():.1f}")
    md.append(f"- **Option 1 words (mean ± sd):** {t2_option['opt1_words'].mean():.1f} ± {t2_option['opt1_words'].std():.1f}")
    md.append(f"- **Option 2 words (mean ± sd):** {t2_option['opt2_words'].mean():.1f} ± {t2_option['opt2_words'].std():.1f}")
    md.append(f"- **Option 3 words (mean ± sd):** {t2_option['opt3_words'].mean():.1f} ± {t2_option['opt3_words'].std():.1f}")
    md.append("")
    md.append("### Per-phase gold distribution")
    md.append("")
    phase_rows = (labelled.groupby("phase")["gold"]
                  .value_counts().unstack(fill_value=0).reset_index())
    md.append("| Phase | n | gold=1 | gold=2 | gold=3 |")
    md.append("| --- | --- | --- | --- | --- |")
    phase_n = labelled.groupby("phase").size()
    for phase in (
        "crisis/engagement", "committed_action", "acceptance/defusion",
        "defusion_deepening", "behavioral_activation", "integration",
        "self_as_context", "closing",
    ):
        if phase not in phase_n.index:
            continue
        n = int(phase_n.loc[phase])
        c1 = int(phase_rows.loc[phase_rows["phase"] == phase, 1].iloc[0]) if 1 in phase_rows.columns else 0
        c2 = int(phase_rows.loc[phase_rows["phase"] == phase, 2].iloc[0]) if 2 in phase_rows.columns else 0
        c3 = int(phase_rows.loc[phase_rows["phase"] == phase, 3].iloc[0]) if 3 in phase_rows.columns else 0
        md.append(f"| {phase} | {n} | {c1} | {c2} | {c3} |")
    md.append("")

    # Simulated Task 1 -----------------------------------------------------
    md.append("## 3. Simulated Task 1 corpus")
    md.append("")
    md.append(f"**{len(sim_t1)} personas**, each ~15 rounds, generated by [src/mentalriskes/data_prep/simulator.py](../src/mentalriskes/data_prep/simulator.py) "
              "with target PHQ-9 / GAD-7 totals and a CompACT flexibility profile.")
    md.append("")
    md.append("| Persona | Presentation | PHQ-9 target | GAD-7 target | CompACT profile | Rounds |")
    md.append("| --- | --- | --- | --- | --- | --- |")
    for p in sim_t1:
        md.append(f"| `{p['session_id']}` | {p['presentation'][:60]} | "
                  f"{p['phq9_target']} | {p['gad7_target']} | {p['compact_profile']} | {p['n_rounds']} |")
    md.append("")
    if sim_t1:
        phq_vals = [p["phq9_target"] for p in sim_t1 if p["phq9_target"] is not None]
        gad_vals = [p["gad7_target"] for p in sim_t1 if p["gad7_target"] is not None]
        md.append(f"**Target-score ranges:** PHQ-9 [{min(phq_vals)}, {max(phq_vals)}], "
                  f"GAD-7 [{min(gad_vals)}, {max(gad_vals)}]. "
                  f"PHQ-9 mean = {sum(phq_vals)/len(phq_vals):.1f}, GAD-7 mean = {sum(gad_vals)/len(gad_vals):.1f}.")
        md.append("")

    # Simulated Task 2 -----------------------------------------------------
    md.append("## 4. Simulated Task 2 corpus")
    md.append("")
    md.append(f"**{len(sim_t2)} personas**, generated alongside Task 1 with three candidate "
              "therapist responses per round (one correct + two error-typed distractors).")
    md.append("")
    md.append("| Session | Presentation | Rounds | opt 1 | opt 2 | opt 3 |")
    md.append("| --- | --- | --- | --- | --- | --- |")
    total_n = 0; total_1 = 0; total_2 = 0; total_3 = 0
    for s in sim_t2:
        md.append(f"| `{s['session_id']}` | {s['presentation'][:60]} | {s['n_rounds']} | "
                  f"{s['opt1_count']} ({s['opt1_pct']:.0%}) | "
                  f"{s['opt2_count']} ({s['opt2_pct']:.0%}) | "
                  f"{s['opt3_count']} ({s['opt3_pct']:.0%}) |")
        total_n += s["n_rounds"]
        total_1 += s["opt1_count"]; total_2 += s["opt2_count"]; total_3 += s["opt3_count"]
    if total_n:
        md.append(f"| **Total** | | **{total_n}** | "
                  f"**{total_1} ({total_1/total_n:.0%})** | "
                  f"**{total_2} ({total_2/total_n:.0%})** | "
                  f"**{total_3} ({total_3/total_n:.0%})** |")
    md.append("")
    md.append("**Note:** simulated gold distribution is roughly balanced (option 1 ≈ 37 %, "
              "option 2 ≈ 33 %, option 3 ≈ 30 %), unlike the trial corpus which favours option 3 "
              "at 50 %. This contrast is part of why pre-submission ablation on trial alone "
              "does not predict the test-set winner — see post-hoc §9 in the Task 2 solution "
              "description.")
    md.append("")

    return "\n".join(md)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    logging.basicConfig(level="INFO", format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    t1_rounds = _load_trial_rounds(REPO_ROOT / "data/MentalRiskES-2026/task1_trial/data")
    t2_rounds = _load_trial_rounds(REPO_ROOT / "data/MentalRiskES-2026/task2_trial/data")
    sim_t1 = _load_simulated_t1()
    sim_t2 = _load_simulated_t2()

    t1_round_df = _task1_round_stats(t1_rounds)
    t2_option_df = _task2_option_stats(t2_rounds)

    t1_round_df.to_csv(OUT_DIR / "trial_t1_round_stats.csv", index=False)
    t2_option_df.to_csv(OUT_DIR / "trial_t2_option_stats.csv", index=False)
    pd.DataFrame(sim_t1).to_csv(OUT_DIR / "simulated_t1_personas.csv", index=False)
    pd.DataFrame(sim_t2).to_csv(OUT_DIR / "simulated_t2_sessions.csv", index=False)

    md = _build_markdown(t1_round_df, t2_option_df, sim_t1, sim_t2)
    md_path = OUT_DIR / "eda_trial_data.md"
    md_path.write_text(md, encoding="utf-8")

    print(f"Wrote {md_path}")
    print(f"  Task 1 trial rounds: {len(t1_round_df)}")
    print(f"  Task 2 trial rounds: {len(t2_option_df)} (labelled: {int(t2_option_df['gold'].notna().sum())})")
    print(f"  Simulated Task 1 personas: {len(sim_t1)}")
    print(f"  Simulated Task 2 sessions: {len(sim_t2)} (rounds: {sum(s['n_rounds'] for s in sim_t2)})")


if __name__ == "__main__":
    main()
