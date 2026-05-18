"""Exploratory data analysis of the MentalRiskES 2026 released test corpus.

Computes descriptive statistics across:
  - Task 1 test: 10 sessions × 30–82 rounds, with item-level gold for 17 patients
    (10 of which appear in the released test data; 7 absent — leaderboard quirk).
  - Task 2 test: same 10 sessions, three candidate responses per patient turn,
    per-round gold labels in round_X_gold.json.

Outputs (in analysis/MentalRiskES_test/outputs/):
  test_t1_session_stats.csv     per-session round count, word counts, gold totals + bands
  test_t1_band_distribution.csv per-instrument band counts across the 10 sessions
  test_t2_session_stats.csv     per-session round count, gold class distribution
  test_t2_round_stats.csv       per-round option lengths, gold class
  test_t2_class_by_tercile.csv  gold class breakdown by early / mid / late round terciles

  eda_test_data.md              standalone Markdown report consolidating the above.

Run:
  python analysis/MentalRiskES_test/eda_test_data.py
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "analysis/MentalRiskES_test/outputs"

T1_TEST_DIR = REPO_ROOT / "data/MentalRiskES-2026/test/task1/test/data"
T1_GOLD_PATH = REPO_ROOT / "data/MentalRiskES-2026/test/task1/test/gold_label.json"
T2_TEST_DIR = REPO_ROOT / "data/MentalRiskES-2026/test/task2/test/data"
T2_GOLD_DIR = REPO_ROOT / "data/MentalRiskES-2026/test/task2/test/gold"

# Severity bands from utils.py — duplicated here so the script is standalone.
PHQ9_BANDS = [(0, 4, "minimal"), (5, 9, "mild"), (10, 14, "moderate"),
              (15, 19, "moderately_severe"), (20, 27, "severe")]
GAD7_BANDS = [(0, 4, "minimal"), (5, 9, "mild"), (10, 14, "moderate"),
              (15, 21, "severe")]


def _band(total: int, bands) -> str:
    for lo, hi, name in bands:
        if lo <= total <= hi:
            return name
    return "out_of_range"


# ─────────────────────────────────────────────────────────────────────────────
# Task 1 loaders + stats
# ─────────────────────────────────────────────────────────────────────────────
def _walk_task1() -> tuple[dict[str, dict], dict[str, dict[int, dict]]]:
    """Return (last_round_by_sid, all_rounds_by_sid_round_dict).

    all_rounds_by_sid is {sid: {round: payload_with_word_counts}} aggregating
    patient_input + therapist_response counts per (sid, round).
    """
    last_round: dict[str, int] = {}
    all_rounds: dict[str, dict[int, dict]] = {}
    for fp in sorted(T1_TEST_DIR.glob("round_*.json"), key=lambda p: int(p.stem.split("_")[1])):
        rnd = int(fp.stem.split("_")[1])
        with open(fp, encoding="utf-8") as fh:
            payload = json.load(fh)
        for sid, turn in payload.items():
            last_round[sid] = max(last_round.get(sid, 0), rnd)
            pi_words = len((turn.get("patient_input", "") or "").split())
            tr_words = len((turn.get("therapist_response", "") or "").split())
            all_rounds.setdefault(sid, {})[rnd] = {
                "patient_words": pi_words,
                "therapist_words": tr_words,
            }
    return last_round, all_rounds


def _task1_session_stats() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    last_round, rounds_by_sid = _walk_task1()
    with open(T1_GOLD_PATH, encoding="utf-8") as fh:
        gold = json.load(fh)

    rows = []
    for sid in sorted(rounds_by_sid):
        rounds = rounds_by_sid[sid]
        n_rounds = len(rounds)
        pi_total = sum(r["patient_words"] for r in rounds.values())
        tr_total = sum(r["therapist_words"] for r in rounds.values())
        g = gold.get(sid, {})
        phq9 = g.get("PHQ-9", [])
        gad7 = g.get("GAD-7", [])
        compact = g.get("CompACT-10", [])
        row = {
            "session": sid,
            "first_round": min(rounds),
            "last_round": last_round[sid],
            "n_rounds": n_rounds,
            "patient_words_total": pi_total,
            "therapist_words_total": tr_total,
            "patient_words_per_round_mean": pi_total / n_rounds if n_rounds else None,
            "phq9_total": sum(phq9) if phq9 else None,
            "phq9_band": _band(sum(phq9), PHQ9_BANDS) if phq9 else None,
            "gad7_total": sum(gad7) if gad7 else None,
            "gad7_band": _band(sum(gad7), GAD7_BANDS) if gad7 else None,
            "compact10_total": sum(compact) if compact else None,
        }
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("session").reset_index(drop=True)

    # Band distribution
    band_rows = []
    for instr, col in (("PHQ-9", "phq9_band"), ("GAD-7", "gad7_band")):
        counts = df[col].value_counts(dropna=True).to_dict()
        for band, n in counts.items():
            band_rows.append({"instrument": instr, "band": band, "count": int(n)})
    band_df = pd.DataFrame(band_rows)

    # Patient-set mismatch
    test_sids = set(rounds_by_sid)
    gold_sids = set(gold)
    extras_in_gold = sorted(gold_sids - test_sids)

    return df, band_df, {
        "n_test_sessions": len(test_sids),
        "n_gold_listed": len(gold_sids),
        "extras_in_gold": extras_in_gold,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Task 2 loaders + stats
# ─────────────────────────────────────────────────────────────────────────────
def _opt_to_int(opt: str) -> int:
    return int(opt.replace("option_", ""))


def _task2_round_stats() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (per_session_df, per_round_df, by_tercile_df)."""
    round_rows = []

    # Load test data + gold
    for fp in sorted(T2_TEST_DIR.glob("round_*.json"), key=lambda p: int(p.stem.split("_")[1])):
        rnd = int(fp.stem.split("_")[1])
        with open(fp, encoding="utf-8") as fh:
            payload = json.load(fh)
        gold_fp = T2_GOLD_DIR / f"round_{rnd}_gold.json"
        gold = {}
        if gold_fp.exists():
            with open(gold_fp, encoding="utf-8") as fh:
                gold_raw = json.load(fh)
            gold = {sid: _opt_to_int(g["correct_option"]) for sid, g in gold_raw.items()}
        for sid, turn in payload.items():
            opts = [turn.get(f"option_{k}", "") or "" for k in (1, 2, 3)]
            round_rows.append({
                "session": sid,
                "round": rnd,
                "patient_words": len((turn.get("patient_input", "") or "").split()),
                "opt1_words": len(opts[0].split()),
                "opt2_words": len(opts[1].split()),
                "opt3_words": len(opts[2].split()),
                "gold": gold.get(sid),
            })

    round_df = pd.DataFrame(round_rows).sort_values(["session", "round"]).reset_index(drop=True)

    # Per-session aggregates
    sess_rows = []
    for sid, sub in round_df.groupby("session"):
        gold_counts = sub["gold"].value_counts(dropna=True).to_dict()
        n_lab = int(sub["gold"].notna().sum())
        sess_rows.append({
            "session": sid,
            "n_rounds": len(sub),
            "n_labelled": n_lab,
            "gold_opt1": int(gold_counts.get(1, 0)),
            "gold_opt2": int(gold_counts.get(2, 0)),
            "gold_opt3": int(gold_counts.get(3, 0)),
            "pct_opt1": gold_counts.get(1, 0) / n_lab if n_lab else None,
            "pct_opt2": gold_counts.get(2, 0) / n_lab if n_lab else None,
            "pct_opt3": gold_counts.get(3, 0) / n_lab if n_lab else None,
            "mean_patient_words": sub["patient_words"].mean(),
            "mean_option_words": sub[["opt1_words", "opt2_words", "opt3_words"]].mean(axis=1).mean(),
        })
    sess_df = pd.DataFrame(sess_rows).sort_values("session").reset_index(drop=True)

    # Tercile breakdown — use the max round across all sessions for tercile cutoffs
    max_round = round_df["round"].max() or 1
    third = max_round / 3

    def _tercile(r: int) -> str:
        if r <= third: return "early"
        if r <= 2 * third: return "mid"
        return "late"

    round_df["tercile"] = round_df["round"].apply(_tercile)
    by_tercile = (round_df[round_df["gold"].notna()]
                  .groupby(["tercile", "gold"]).size().unstack(fill_value=0)
                  .reindex(["early", "mid", "late"])
                  .reset_index()).rename_axis(None, axis=1)

    return sess_df, round_df, by_tercile


# ─────────────────────────────────────────────────────────────────────────────
# Markdown report
# ─────────────────────────────────────────────────────────────────────────────
def _build_markdown(t1_sess: pd.DataFrame, t1_band: pd.DataFrame, t1_meta: dict,
                    t2_sess: pd.DataFrame, t2_round: pd.DataFrame, t2_tercile: pd.DataFrame) -> str:
    md: list[str] = []
    md.append("## Test Data EDA")
    md.append("")
    md.append("*Auto-generated by [analysis/MentalRiskES_test/eda_test_data.py](eda_test_data.py); "
              "re-run that script after any data changes.*")
    md.append("")

    # Task 1 ----------------------------------------------------------
    md.append("### Task 1 test corpus")
    md.append("")
    md.append(f"- **Sessions in test data:** {t1_meta['n_test_sessions']} "
              f"({', '.join(t1_sess['session'].tolist())})")
    md.append(f"- **Sessions in `gold_label.json`:** {t1_meta['n_gold_listed']}")
    if t1_meta["extras_in_gold"]:
        md.append(f"- **Patients in gold but NOT in released test data:** "
                  f"{', '.join(t1_meta['extras_in_gold'])} "
                  f"(presumably scored against zero defaults by the leaderboard — "
                  f"this is the ~±0.10 gap between local R30 item-MAE and the official numbers).")
    md.append(f"- **Rounds per session:** mean {t1_sess['n_rounds'].mean():.1f}, "
              f"min {int(t1_sess['n_rounds'].min())}, max {int(t1_sess['n_rounds'].max())}.")
    md.append(f"- **Patient words per session:** mean {t1_sess['patient_words_total'].mean():.0f}, "
              f"median {t1_sess['patient_words_total'].median():.0f}.")
    md.append("")
    md.append("#### Per-session statistics")
    md.append("")
    md.append("| Session | Rounds | Patient words | PHQ-9 total / band | GAD-7 total / band | CompACT-10 total |")
    md.append("| --- | --- | --- | --- | --- | --- |")
    for _, r in t1_sess.iterrows():
        phq_cell = f"{int(r['phq9_total'])} / {r['phq9_band']}" if pd.notna(r['phq9_total']) else "—"
        gad_cell = f"{int(r['gad7_total'])} / {r['gad7_band']}" if pd.notna(r['gad7_total']) else "—"
        com_cell = f"{int(r['compact10_total'])}" if pd.notna(r['compact10_total']) else "—"
        md.append(f"| {r['session']} | {int(r['n_rounds'])} | "
                  f"{int(r['patient_words_total'])} | {phq_cell} | {gad_cell} | {com_cell} |")
    md.append("")

    md.append("#### Severity-band distribution across the 10 test sessions")
    md.append("")
    md.append("| Instrument | Band | Sessions |")
    md.append("| --- | --- | --- |")
    band_order = {"PHQ-9": ["minimal", "mild", "moderate", "moderately_severe", "severe"],
                  "GAD-7": ["minimal", "mild", "moderate", "severe"]}
    for instr in ("PHQ-9", "GAD-7"):
        for band in band_order[instr]:
            row = t1_band[(t1_band["instrument"] == instr) & (t1_band["band"] == band)]
            n = int(row["count"].iloc[0]) if len(row) else 0
            md.append(f"| {instr} | {band} | {n} |")
    md.append("")
    md.append("**Cohort skew:** the test cohort is heavily weighted to moderate-to-severe "
              "presentations — 8 of 10 sessions land in the GAD-7 severe band (gold total ≥ 15). "
              "This is meaningfully different from the trial cohort (single moderate patient) "
              "and the simulated personas (target totals span the full range). Under-prediction "
              "by anchored systems is the dominant failure mode on this cohort, as documented in "
              "[analysis/MentalRiskES_test/SUMMARY.md §2.1](../analysis/MentalRiskES_test/SUMMARY.md).")
    md.append("")

    # Task 2 ----------------------------------------------------------
    md.append("### Task 2 test corpus")
    md.append("")
    n_total_rounds = int(t2_round["round"].notna().sum())
    n_labelled = int(t2_round["gold"].notna().sum())
    md.append(f"- **Patient-round pairs:** {n_total_rounds} across {len(t2_sess)} sessions "
              f"(rounds 1–{int(t2_round['round'].max())}).")
    md.append(f"- **Labelled rounds:** {n_labelled} (every patient turn in the test data has "
              "an associated `correct_option` in `round_X_gold.json`).")
    md.append(f"- **Mean patient words per round:** {t2_round['patient_words'].mean():.1f}.")
    md.append(f"- **Mean option length:** "
              f"opt1 {t2_round['opt1_words'].mean():.1f}, "
              f"opt2 {t2_round['opt2_words'].mean():.1f}, "
              f"opt3 {t2_round['opt3_words'].mean():.1f} words.")
    md.append("")
    md.append("#### Per-session statistics")
    md.append("")
    md.append("| Session | Rounds | gold=1 | gold=2 | gold=3 | Mean patient words | Mean option words |")
    md.append("| --- | --- | --- | --- | --- | --- | --- |")
    total_n = 0; total_1 = 0; total_2 = 0; total_3 = 0
    for _, r in t2_sess.iterrows():
        md.append(f"| {r['session']} | {int(r['n_rounds'])} | "
                  f"{int(r['gold_opt1'])} ({r['pct_opt1']:.0%}) | "
                  f"{int(r['gold_opt2'])} ({r['pct_opt2']:.0%}) | "
                  f"{int(r['gold_opt3'])} ({r['pct_opt3']:.0%}) | "
                  f"{r['mean_patient_words']:.1f} | {r['mean_option_words']:.1f} |")
        total_n += int(r["n_labelled"])
        total_1 += int(r["gold_opt1"]); total_2 += int(r["gold_opt2"]); total_3 += int(r["gold_opt3"])
    md.append(f"| **Total** | | "
              f"**{total_1} ({total_1/total_n:.1%})** | "
              f"**{total_2} ({total_2/total_n:.1%})** | "
              f"**{total_3} ({total_3/total_n:.1%})** | | |")
    md.append("")

    md.append("#### Gold class by round tercile")
    md.append("")
    md.append("| Tercile | gold=1 | gold=2 | gold=3 |")
    md.append("| --- | --- | --- | --- |")
    for _, r in t2_tercile.iterrows():
        cells = []
        for k in (1, 2, 3):
            val = r[k] if k in r.index else 0
            cells.append(str(int(val)))
        md.append(f"| {r['tercile']} | {' | '.join(cells)} |")
    md.append("")
    md.append("**Gold distribution is roughly uniform across classes** (35.7 / 32.8 / 31.5 % overall), "
              "in contrast to the trial corpus where option 3 is 50 %. The consensus-failure analysis in "
              "[SUMMARY.md §5.8](../analysis/MentalRiskES_test/SUMMARY.md) shows that despite this near-uniform "
              "distribution, gold=3 rounds are wrong-by-every-system 38.5 % of the time — a categorical "
              "blind spot across the LLM family rather than a class-imbalance artefact.")
    md.append("")

    return "\n".join(md).rstrip() + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    logging.basicConfig(level="INFO", format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    t1_sess, t1_band, t1_meta = _task1_session_stats()
    t2_sess, t2_round, t2_tercile = _task2_round_stats()

    t1_sess.to_csv(OUT_DIR / "test_t1_session_stats.csv", index=False)
    t1_band.to_csv(OUT_DIR / "test_t1_band_distribution.csv", index=False)
    t2_sess.to_csv(OUT_DIR / "test_t2_session_stats.csv", index=False)
    t2_round.to_csv(OUT_DIR / "test_t2_round_stats.csv", index=False)
    t2_tercile.to_csv(OUT_DIR / "test_t2_class_by_tercile.csv", index=False)

    md = _build_markdown(t1_sess, t1_band, t1_meta, t2_sess, t2_round, t2_tercile)
    md_path = OUT_DIR / "eda_test_data.md"
    md_path.write_text(md, encoding="utf-8")

    print(f"Wrote {md_path}")
    print(f"  Task 1 sessions: {len(t1_sess)}  (gold lists {t1_meta['n_gold_listed']}, extras: {t1_meta['extras_in_gold']})")
    print(f"  Task 2 sessions: {len(t2_sess)}  total labelled rounds: {int(t2_round['gold'].notna().sum())}")


if __name__ == "__main__":
    main()
