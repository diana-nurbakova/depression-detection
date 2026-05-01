"""Qualitative case-study generator for MentalRiskES test analysis.

Produces self-contained Markdown reports with full transcript context for the
prediction errors most worth reading. Two modes:

    --task t1   Picks the top-K patients per instrument by |replay_MAE -
                submitted_MAE| and dumps each as a case study with submitted
                vs replay items, LLM CoT excerpts, and a transcript slice.
                Output: outputs/qualitative_T1_case_studies.md

    --task t2   Samples K errors weighted by the dominant confusion-matrix
                cells (gold=3->pred=2 first, then gold=1->pred=2, etc.) and
                dumps the full transcript, the three candidate responses, our
                pick, and the gold pick. A "Disagreement type" slot is
                pre-filled with the four taxonomy options from Analysis V.
                Output: outputs/qualitative_T2_disagreement_taxonomy.md

Usage:
    python analysis/MentalRiskES_test/qualitative_case_studies.py \
        --task t1 --top 5 --run 2

    python analysis/MentalRiskES_test/qualitative_case_studies.py \
        --task t2 --n 30 --run 2

Both modes are pure read-only data wrangling; no LLM calls. The script can
be re-run as the replay produces more data.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

from utils import (
    classify_band,
    load_config,
    load_task1_gold,
    load_task2_gold,
    load_task2_predictions,
    load_task2_test,
    mae,
    repo_path,
    total,
)

logger = logging.getLogger("qualitative")

REPLAY_DIR_REL = "output/mentalriskes_test_replay/predictions"
REPLAY_LOGS_DIR_REL = "output/mentalriskes_test_replay/logs"
SUBMITTED_DIR_REL = "output/mentalriskes/predictions"
DEEPL_CACHE_REL = "analysis/MentalRiskES_test/outputs/deepl_cache_es_en.json"


# ─────────────────────────────────────────────────────────────────────────────
# DeepL EN gloss with on-disk cache (free key allows ~500K chars/month)
# ─────────────────────────────────────────────────────────────────────────────
class DeepLTranslator:
    def __init__(self, api_key: str, cache_path: Path) -> None:
        self.api_key = api_key
        self.cache_path = cache_path
        self.cache: dict[str, str] = {}
        if cache_path.exists():
            with open(cache_path, encoding="utf-8") as fh:
                self.cache = json.load(fh)
        # Free keys end with ":fx" and use api-free.deepl.com; pro uses api.deepl.com.
        self.endpoint = (
            "https://api-free.deepl.com/v2/translate"
            if api_key.endswith(":fx")
            else "https://api.deepl.com/v2/translate"
        )

    def translate(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return ""
        if text in self.cache:
            return self.cache[text]
        # Try modern header auth first; fall back to body auth on auth-style mismatch.
        last_err = None
        for endpoint in (self.endpoint, self._other_endpoint()):
            for auth_style in ("header", "body"):
                try:
                    if auth_style == "header":
                        resp = requests.post(
                            endpoint,
                            headers={"Authorization": f"DeepL-Auth-Key {self.api_key}"},
                            data={"text": text, "source_lang": "ES", "target_lang": "EN"},
                            timeout=30,
                        )
                    else:
                        resp = requests.post(
                            endpoint,
                            data={
                                "auth_key": self.api_key,
                                "text": text,
                                "source_lang": "ES",
                                "target_lang": "EN",
                            },
                            timeout=30,
                        )
                    resp.raise_for_status()
                    translated = resp.json()["translations"][0]["text"]
                    self.cache[text] = translated
                    return translated
                except Exception as e:
                    last_err = e
        logger.warning("DeepL failed (%s); leaving Spanish-only. Check DEEPL_AUTH_KEY validity.", last_err)
        self.cache[text] = ""
        return ""

    def _other_endpoint(self) -> str:
        if "api-free" in self.endpoint:
            return "https://api.deepl.com/v2/translate"
        return "https://api-free.deepl.com/v2/translate"

    def flush(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as fh:
            json.dump(self.cache, fh, ensure_ascii=False, indent=2)


def _make_deepl(enabled: bool) -> DeepLTranslator | None:
    if not enabled:
        return None
    key = os.getenv("DEEPL_AUTH_KEY", "").strip()
    if not key:
        logger.warning("--gloss requested but DEEPL_AUTH_KEY not set; skipping")
        return None
    return DeepLTranslator(api_key=key, cache_path=Path(DEEPL_CACHE_REL))


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────
def _load_round_json(predictions_dir: Path, run_idx: int) -> dict[int, dict[str, dict]]:
    out: dict[int, dict[str, dict]] = {}
    for fp in sorted(predictions_dir.glob(f"round*_run{run_idx}.json")):
        rnd = int(fp.stem.split("_")[0].replace("round", ""))
        with open(fp, encoding="utf-8") as fh:
            payload = json.load(fh)
        out[rnd] = {entry["id"]: entry["prediction"] for entry in payload[0]["predictions"]}
    return out


def _last_per_session(rounds: dict[int, dict[str, dict]]) -> dict[str, tuple[int, dict]]:
    last: dict[str, tuple[int, dict]] = {}
    for rnd, sessions in rounds.items():
        for sid, pred in sessions.items():
            if sid not in last or rnd > last[sid][0]:
                last[sid] = (rnd, pred)
    return last


def _load_test_transcripts(cfg: dict) -> dict[int, dict[str, dict]]:
    """Load the multi-session test rounds for Task 1 (also has therapist_response after round 1)."""
    out: dict[int, dict[str, dict]] = {}
    for fp in sorted(repo_path(cfg["paths"]["task1_data_dir"]).glob("round_*.json")):
        rnd = int(fp.stem.replace("round_", ""))
        with open(fp, encoding="utf-8") as fh:
            out[rnd] = json.load(fh)
    return out


def _load_replay_jsonl(run_idx: int) -> dict[tuple[str, int], dict]:
    """Load Task 1 replay JSONL keyed by (session, round). Maps run_idx 0/1/2 to run0_A5/run1_A3/run2_A1."""
    name_map = {0: "run0_A5", 1: "run1_A3", 2: "run2_A1"}
    fname = f"predictions_{name_map[run_idx]}.jsonl"
    fp = repo_path(REPLAY_LOGS_DIR_REL) / fname
    out: dict[tuple[str, int], dict] = {}
    if not fp.exists():
        return out
    with open(fp, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            out[(entry["session_id"], int(entry["round"]))] = entry
    return out


def _md_escape(text: str) -> str:
    if text is None:
        return ""
    return text.replace("|", "\\|").replace("\n", " / ").strip()


def _items_table(name: str, items: list[int], gold: list[int]) -> str:
    header = "| Item | " + " | ".join(str(i + 1) for i in range(len(items))) + " | total |"
    sep = "| --- |" + " --- |" * (len(items) + 1)
    pred_row = f"| pred ({name}) | " + " | ".join(str(x) for x in items) + f" | {sum(items)} |"
    gold_row = "| gold | " + " | ".join(str(x) for x in gold) + f" | {sum(gold)} |"
    return "\n".join([header, sep, pred_row, gold_row])


def _format_cot_excerpt(steps: dict, item_label_map: dict[int, str], max_items: int = 7) -> str:
    """Pull the per-item endorsement scores + a short evidence snippet from the JSONL CoT."""
    if not steps:
        return "_(no CoT logged)_"
    lines = []
    detect = steps.get("step_1_detection", {}) or steps.get("step_1_extraction", {})
    score_step = steps.get("step_2_temporal", {}) or steps.get("step_2_endorsement", {})
    for i in range(1, max_items + 1):
        key = f"item_{i}"
        d = detect.get(key, {})
        s = score_step.get(key, {})
        evidence = d.get("evidence", "") if isinstance(d, dict) else ""
        score = s.get("score") if isinstance(s, dict) else None
        label = item_label_map.get(i - 1, f"item {i}")
        lines.append(f"- **{label}** (score={score}): {evidence[:140].strip() or '_no evidence_'}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Task 1 mode
# ─────────────────────────────────────────────────────────────────────────────
def _build_t1_case_studies(cfg: dict, top: int, run_idx: int, out_path: Path) -> None:
    gold = load_task1_gold(cfg)
    transcripts = _load_test_transcripts(cfg)

    submitted = _load_round_json(repo_path(SUBMITTED_DIR_REL), run_idx)
    replay = _load_round_json(repo_path(REPLAY_DIR_REL), run_idx)
    if not submitted or not replay:
        out_path.write_text(
            f"# Qualitative T1 case studies\n\n"
            f"_Replay output not yet available (replay dir: `{REPLAY_DIR_REL}`, "
            f"submitted dir: `{SUBMITTED_DIR_REL}`). Re-run after the replay finishes._\n",
            encoding="utf-8",
        )
        logger.warning("Missing predictions; wrote stub to %s", out_path)
        return

    sub_last = _last_per_session(submitted)
    rep_last = _last_per_session(replay)
    common = sorted(set(gold.keys()) & set(sub_last.keys()) & set(rep_last.keys()))

    jsonl_index = _load_replay_jsonl(run_idx)

    # Build a per-(session, instrument) ranking by |delta MAE_items|
    rows = []
    for sid in common:
        for instr in ("PHQ-9", "GAD-7", "CompACT-10"):
            g = gold[sid][instr]
            sub_round, sub_pred = sub_last[sid]
            rep_round, rep_pred = rep_last[sid]
            sub_mae = mae(sub_pred[instr], g)
            rep_mae = mae(rep_pred[instr], g)
            rows.append({
                "session": sid,
                "instrument": instr,
                "submitted_round": sub_round,
                "replay_round": rep_round,
                "submitted_total": total(sub_pred[instr]),
                "replay_total": total(rep_pred[instr]),
                "gold_total": total(g),
                "submitted_items": sub_pred[instr],
                "replay_items": rep_pred[instr],
                "gold_items": g,
                "submitted_MAE": sub_mae,
                "replay_MAE": rep_mae,
                "abs_delta": abs(rep_mae - sub_mae),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        out_path.write_text("# Qualitative T1 case studies\n\n_No matching predictions._\n", encoding="utf-8")
        return

    item_labels = {
        instr: cfg["instruments"][instr]["item_labels"] if "item_labels" in cfg["instruments"][instr] else [f"item {i+1}" for i in range(cfg["instruments"][instr]["n_items"])]
        for instr in ("PHQ-9", "GAD-7", "CompACT-10")
    }

    md = [
        "# Qualitative Task 1 Case Studies",
        "",
        f"Run: `{cfg['team']['runs'][run_idx]['label']}` (run_idx={run_idx})",
        f"Selection: top {top} per instrument by |replay_MAE − submitted_MAE|",
        "",
    ]

    for instr in ("PHQ-9", "GAD-7", "CompACT-10"):
        sub = df[df["instrument"] == instr].sort_values("abs_delta", ascending=False).head(top)
        md.append(f"## {instr}\n")
        for _, r in sub.iterrows():
            sid = r["session"]
            md.append(f"### {sid} — {instr}\n")
            md.append(f"- Gold band: `{classify_band(int(r['gold_total']), instr, cfg)}` (total {r['gold_total']})")
            md.append(f"- Submitted (round {int(r['submitted_round'])}): total {int(r['submitted_total'])}, MAE_items={r['submitted_MAE']:.3f}")
            md.append(f"- Replay (round {int(r['replay_round'])}): total {int(r['replay_total'])}, MAE_items={r['replay_MAE']:.3f}")
            md.append(f"- |ΔMAE| = {r['abs_delta']:.3f}\n")
            md.append("**Items (pred at last available round vs gold):**\n")
            md.append(_items_table("submitted", r["submitted_items"], r["gold_items"]))
            md.append("")
            md.append(_items_table("replay", r["replay_items"], r["gold_items"]))
            md.append("")

            # CoT excerpts at submitted_round and replay_round
            sub_jsonl = jsonl_index.get((sid, int(r["submitted_round"])), {})
            rep_jsonl = jsonl_index.get((sid, int(r["replay_round"])), {})
            md.append(f"**LLM reasoning at submitted round {int(r['submitted_round'])} (CoT excerpt):**\n")
            md.append(_format_cot_excerpt(sub_jsonl.get(f"{instr}_steps", {}), {i: item_labels[instr][i] for i in range(len(item_labels[instr]))}, max_items=cfg["instruments"][instr]["n_items"]))
            md.append("")
            md.append(f"**LLM reasoning at replay round {int(r['replay_round'])} (CoT excerpt):**\n")
            md.append(_format_cot_excerpt(rep_jsonl.get(f"{instr}_steps", {}), {i: item_labels[instr][i] for i in range(len(item_labels[instr]))}, max_items=cfg["instruments"][instr]["n_items"]))
            md.append("")

            # Transcript slice: rounds 25–35 + final 5 rounds for this session
            slice_rounds = sorted(set(list(range(25, 36)) + list(range(int(r["replay_round"]) - 4, int(r["replay_round"]) + 1))))
            md.append("**Transcript slice (patient turns; therapist turns where logged):**\n")
            for rnd in slice_rounds:
                if rnd < 1 or rnd not in transcripts:
                    continue
                if sid not in transcripts[rnd]:
                    continue
                turn = transcripts[rnd][sid]
                pi = _md_escape(turn.get("patient_input", ""))[:300]
                tr = _md_escape(turn.get("therapist_response", ""))[:300]
                md.append(f"- **R{rnd}** patient: _{pi}_")
                if tr:
                    md.append(f"  - therapist (prev): _{tr}_")
            md.append("")
            md.append("**Commentary:** _(fill in)_\n")
            md.append("---\n")

    out_path.write_text("\n".join(md), encoding="utf-8")
    logger.info("Wrote %s (%d patients × 3 instruments)", out_path, top)


# ─────────────────────────────────────────────────────────────────────────────
# Task 2 mode
# ─────────────────────────────────────────────────────────────────────────────
def _build_t2_disagreement(cfg: dict, n: int, run_idx: int, out_path: Path, seed: int = 42, gloss: bool = False) -> None:
    gold = load_task2_gold(cfg)
    test = load_task2_test(cfg)
    preds = load_task2_predictions(cfg, run_idx)
    translator = _make_deepl(gloss)

    # Sample errors weighted by dominant confusion cells.
    error_records = []
    for rnd, gold_round in gold.items():
        if rnd not in preds:
            continue
        for sid, opt_str in gold_round.items():
            if sid not in preds[rnd] or rnd not in test or sid not in test[rnd]:
                continue
            g_int = int(opt_str.replace("option_", ""))
            p_int = preds[rnd][sid]
            if g_int == p_int:
                continue
            error_records.append({"round": rnd, "session": sid, "gold": g_int, "pred": p_int})

    if not error_records:
        out_path.write_text("# Qualitative T2 disagreement taxonomy\n\n_No errors yet (replay still warming up)._\n", encoding="utf-8")
        return

    rng = random.Random(seed)
    df = pd.DataFrame(error_records)
    cell_counts = df.groupby(["gold", "pred"]).size().reset_index(name="count").sort_values("count", ascending=False)

    # Stratified sampling: take from top cells in proportion to their share, capped at n
    chosen: list[dict] = []
    remaining = n
    for _, row in cell_counts.iterrows():
        if remaining <= 0:
            break
        cell_total = int(row["count"])
        share = round(n * cell_total / len(df))
        take = min(remaining, max(1, share), cell_total)
        cell_errors = df[(df["gold"] == row["gold"]) & (df["pred"] == row["pred"])].to_dict("records")
        rng.shuffle(cell_errors)
        chosen.extend(cell_errors[:take])
        remaining -= take

    md = [
        "# Qualitative Task 2 Disagreement Taxonomy",
        "",
        f"Run: `{cfg['team']['runs'][run_idx]['label']}` (run_idx={run_idx})",
        f"Sample: {len(chosen)} errors stratified by confusion-matrix cell",
        "",
        "**Disagreement taxonomy** (Analysis V):",
        "- **sophistication**: we chose a complex intervention, gold chose simple validation",
        "- **phase**: we chose a working-phase response, gold was engagement-building",
        "- **safety**: we avoided the direct/confrontational option that gold picked",
        "- **pragmatics**: we optimized for ACT process, gold for conversational flow",
        "- **other**: doesn't fit above",
        "",
        "Confusion-cell distribution of the full error set:",
        "",
        "| gold | pred | count |",
        "| --- | --- | --- |",
        *[f"| {int(r['gold'])} | {int(r['pred'])} | {int(r['count'])} |" for _, r in cell_counts.iterrows()],
        "",
        "---",
        "",
    ]

    for i, e in enumerate(chosen, 1):
        rnd = e["round"]
        sid = e["session"]
        g_int = e["gold"]
        p_int = e["pred"]
        round_data = test[rnd][sid]

        md.append(f"## #{i} — {sid} round {rnd}  (gold={g_int}, pred={p_int})\n")

        # Recent transcript for context: last 4 patient turns of this session up to and including current
        ctx_lines = []
        for r_back in range(max(1, rnd - 3), rnd + 1):
            if r_back in test and sid in test[r_back]:
                turn = test[r_back][sid]
                pi_raw = turn.get("patient_input", "")
                pi = _md_escape(pi_raw)[:300]
                line = f"- **R{r_back} patient (es)**: _{pi}_"
                if translator and pi_raw.strip():
                    en = translator.translate(pi_raw)
                    if en:
                        line += f"\n  - **(en)**: _{_md_escape(en)[:300]}_"
                ctx_lines.append(line)
        md.append("**Recent transcript (last 4 turns):**")
        md.extend(ctx_lines)
        md.append("")

        for k in (1, 2, 3):
            label = round_data.get(f"option_{k}", "")
            tag = []
            if k == g_int:
                tag.append("**GOLD**")
            if k == p_int:
                tag.append("**OUR PICK**")
            tag_str = f" — {' / '.join(tag)}" if tag else ""
            md.append(f"- **Option {k}**{tag_str} (es): _{_md_escape(label)[:500]}_")
            if translator and label.strip():
                en = translator.translate(label)
                if en:
                    md.append(f"  - **(en)**: _{_md_escape(en)[:500]}_")
        md.append("")
        md.append("**Disagreement type:** [ ] sophistication  [ ] phase  [ ] safety  [ ] pragmatics  [ ] other")
        md.append("**Notes:** _(fill in)_\n")
        md.append("---\n")

    out_path.write_text("\n".join(md), encoding="utf-8")
    if translator:
        translator.flush()
    logger.info("Wrote %s (%d errors sampled from %d total)", out_path, len(chosen), len(error_records))


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("t1", "t2"), required=True, help="Which task to generate cases for.")
    parser.add_argument("--top", type=int, default=5, help="(t1) cases per instrument")
    parser.add_argument("--n", type=int, default=30, help="(t2) error sample size")
    parser.add_argument("--run", type=int, default=2, help="Run index (0/1/2)")
    parser.add_argument("--out", type=Path, default=None, help="Output Markdown path")
    parser.add_argument("--gloss", action="store_true",
                        help="(t2) Add English gloss for each Spanish option/transcript line via DeepL.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    cfg = load_config()
    out_dir = repo_path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.task == "t1":
        out_path = args.out or (out_dir / "qualitative_T1_case_studies.md")
        _build_t1_case_studies(cfg, top=args.top, run_idx=args.run, out_path=out_path)
    else:
        out_path = args.out or (out_dir / "qualitative_T2_disagreement_taxonomy.md")
        _build_t2_disagreement(cfg, n=args.n, run_idx=args.run, out_path=out_path, gloss=args.gloss)


if __name__ == "__main__":
    main()
