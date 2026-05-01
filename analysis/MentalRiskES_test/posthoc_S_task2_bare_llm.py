"""Task 2 post-hoc — bare-LLM and variants per task2_improvement_spec.md.

Modes:
  S    bare LLM single-pass selection (Experiment S)
  S2   bare LLM with anti-bias guardrails (Experiment S2)
  R2   bare LLM full 3-way ranking (Experiment R2)
  S3   bare LLM with permutation averaging (Experiment S3)
  S4   bare LLM pairwise (3 comparisons + Condorcet) (Experiment S4)

For each mode the runner saves predictions in server-format
(`output/mentalriskes_task2_bare_llm/<model>__<mode>/round{N}.json`) plus
a `raw.jsonl` with the full LLM response per (session, round).

The evaluator [posthoc_S_task2_bare_llm_eval.py] turns these into
W_t2_bare_llm.csv with accuracy, macro-F1, distribution, chi^2 and
per-tercile breakdown.

Usage examples (all 82 rounds, full 10-session set):
  python ... --mode S  --model google/gemma-3-27b-it
  python ... --mode S3 --model google/gemma-3-27b-it
  python ... --mode S4 --model google/gemma-3-27b-it
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from itertools import permutations
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
load_dotenv(REPO_ROOT / ".env")

logger = logging.getLogger("task2_bare")


# ─────────────────────────────────────────────────────────────────────────────
# Prompts (verbatim from task2_improvement_spec.md §2.x)
# ─────────────────────────────────────────────────────────────────────────────
BARE_LLM_SYSTEM = """You are an expert psychotherapist conducting an ACT (Acceptance and Commitment Therapy) session in Spanish.

Read the following conversation between a therapist and a patient. Then choose which of the three candidate responses the therapist should say next.

Consider:
- Which response best matches what the patient needs RIGHT NOW?
- Which response feels most natural as a continuation of the conversation?
- Which response maintains the therapeutic alliance while being helpful?

Do not overthink this. Trust your clinical intuition based on the conversation flow.
"""

BARE_LLM_S_USER = """## CONVERSATION

{transcript}

## CANDIDATE RESPONSES

Option 1: {option_1}
Option 2: {option_2}
Option 3: {option_3}

## YOUR CHOICE

Respond with ONLY a JSON object:
{{
  "choice": 1,
  "brief_reason": "one sentence explaining why"
}}
"""

BARE_LLM_S2_GUARDRAILS = """

IMPORTANT:
- Do NOT prefer longer or more elaborate responses. Sometimes the best response is the shortest and most direct.
- Do NOT assume the middle option (Option 2) is the safest choice. Evaluate all three equally.
- Sometimes the most therapeutically effective response is simple validation or a direct question, not a complex intervention.
- Consider what a skilled therapist would ACTUALLY say in this moment, not what sounds most impressive.
"""

BARE_LLM_R2_USER = """## CONVERSATION

{transcript}

## CANDIDATE RESPONSES

Option 1: {option_1}
Option 2: {option_2}
Option 3: {option_3}

Rank all three options from best (1st) to worst (3rd) as continuations of this therapeutic conversation.

Respond with ONLY a JSON object:
{{
  "ranking": [2, 1, 3],
  "reasons": {{
    "1st": "why this is the best response",
    "2nd": "why this is second",
    "3rd": "why this is worst"
  }}
}}
"""

BARE_LLM_S4_USER = """## PATIENT'S RECENT CONTEXT (last 3 exchanges)

{recent_context}

## OPTION A
{option_a}

## OPTION B
{option_b}

Which response is better as a continuation of this therapeutic conversation?
Respond with ONLY: {{"better": "A"}} or {{"better": "B"}}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Test data loader (multi-session round files; current released test format)
# ─────────────────────────────────────────────────────────────────────────────
def load_test_rounds(test_dir: Path) -> list[tuple[int, dict]]:
    out = []
    for fp in sorted(test_dir.glob("round_*.json"), key=lambda p: int(p.stem.split("_")[1])):
        with open(fp, encoding="utf-8") as fh:
            out.append((int(fp.stem.split("_")[1]), json.load(fh)))
    return out


def build_session_turns(rounds: list[tuple[int, dict]]) -> dict[str, list[tuple[int, str, str, dict]]]:
    """{session_id: [(round, role, text, options_dict), ...]} sorted by round.

    options_dict is non-empty only for the patient turn that has 3 candidates
    in the test data (i.e. the round being evaluated).
    """
    sessions: dict[str, list[tuple[int, str, str, dict]]] = {}
    for rnd, data in rounds:
        for sid, payload in data.items():
            therapist = payload.get("therapist_response")
            patient = payload.get("patient_input", "")
            options = {k: payload.get(k, "") for k in ("option_1", "option_2", "option_3")}
            if therapist:
                sessions.setdefault(sid, []).append((rnd, "therapist", therapist, {}))
            if patient:
                sessions.setdefault(sid, []).append((rnd, "patient", patient, options))
    for sid in sessions:
        sessions[sid].sort(key=lambda r: (r[0], 0 if r[1] == "therapist" else 1))
    return sessions


# ─────────────────────────────────────────────────────────────────────────────
# Trial / simulated loaders — both use the legacy "trial"-keyed single-session
# round_X.json format (one trial cohort = single session id "trial";
# simulated cohort = one dir per session, each holding round_X.json files).
# ─────────────────────────────────────────────────────────────────────────────
def load_trial_single_session(trial_dir: Path, session_id: str) -> list[tuple[int, str, str, dict]]:
    """Read round_*.json with {"trial": {round, patient_input, option_1/2/3, ...}} format.

    Returns a turn list compatible with build_session_turns output for one session.
    """
    turns: list[tuple[int, str, str, dict]] = []
    for fp in sorted(trial_dir.glob("round_*.json"), key=lambda p: int(p.stem.split("_")[1])):
        rnd = int(fp.stem.split("_")[1])
        with open(fp, encoding="utf-8") as fh:
            payload = json.load(fh).get("trial", {})
        therapist = payload.get("therapist_response")
        if therapist:
            turns.append((rnd, "therapist", therapist, {}))
        patient = payload.get("patient_input", "")
        options = {k: payload.get(k, "") for k in ("option_1", "option_2", "option_3")}
        if patient:
            turns.append((rnd, "patient", patient, options))
    turns.sort(key=lambda r: (r[0], 0 if r[1] == "therapist" else 1))
    return turns


def load_simulated_sessions(sim_root: Path) -> dict[str, list[tuple[int, str, str, dict]]]:
    """{session_id: turns} for every simulated session dir under sim_root."""
    out: dict[str, list[tuple[int, str, str, dict]]] = {}
    if not sim_root.exists():
        return out
    for d in sorted(sim_root.iterdir()):
        if not d.is_dir():
            continue
        if not (d / "labels.json").exists():
            # Task 2 simulated dirs all have labels.json; skip those that don't
            continue
        out[d.name] = load_trial_single_session(d, d.name)
    return out


def format_transcript_up_to(turns: list[tuple[int, str, str, dict]], up_to_round: int) -> str:
    lines = []
    for r, role, text, _ in turns:
        if r > up_to_round:
            break
        lines.append(f"[Round {r} - {role.upper()}]: {text}")
    return "\n\n".join(lines)


def format_recent_context(turns: list[tuple[int, str, str, dict]], up_to_round: int, last_n: int = 6) -> str:
    """Last N turn lines up to and including up_to_round (used by S4 pairwise)."""
    sub = [t for t in turns if t[0] <= up_to_round]
    sub = sub[-last_n:]
    return "\n\n".join(f"[Round {r} - {role.upper()}]: {text}" for r, role, text, _ in sub)


# ─────────────────────────────────────────────────────────────────────────────
# OpenRouter client
# ─────────────────────────────────────────────────────────────────────────────
def _openrouter_client():
    from openai import OpenAI
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL)


def _strip_fences(text: str) -> str:
    text = (text or "").strip()
    m = _FENCE_RE.match(text)
    if m:
        return m.group(1).strip()
    return text


def _is_gemma(model: str) -> bool:
    return "gemma" in model.lower()


def _build_messages(model: str, system: str, user: str) -> list[dict]:
    if _is_gemma(model):
        return [{"role": "user", "content": system + "\n\n---\n\n" + user}]
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _call(client, model: str, system: str, user: str, max_tokens: int) -> str:
    is_gemma = _is_gemma(model)
    kwargs = {
        "model": model,
        "messages": _build_messages(model, system, user),
        "temperature": 0.1,
        "max_tokens": max_tokens,
    }
    if not is_gemma:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""


# ─────────────────────────────────────────────────────────────────────────────
# Mode handlers — each returns (selected_option_int, raw_payload_dict)
# ─────────────────────────────────────────────────────────────────────────────
def mode_S(client, model: str, transcript: str, options: dict, mode: str = "S") -> tuple[int, dict]:
    system = BARE_LLM_SYSTEM + (BARE_LLM_S2_GUARDRAILS if mode == "S2" else "")
    user = BARE_LLM_S_USER.format(transcript=transcript, **options)
    raw = _call(client, model, system, user, max_tokens=200)
    parsed = json.loads(_strip_fences(raw))
    choice = int(parsed["choice"])
    if choice not in (1, 2, 3):
        raise ValueError(f"choice out of range: {choice}")
    return choice, {"raw": parsed}


def mode_R2(client, model: str, transcript: str, options: dict) -> tuple[int, dict]:
    user = BARE_LLM_R2_USER.format(transcript=transcript, **options)
    raw = _call(client, model, BARE_LLM_SYSTEM, user, max_tokens=400)
    parsed = json.loads(_strip_fences(raw))
    ranking = [int(x) for x in parsed["ranking"]]
    if sorted(ranking) != [1, 2, 3]:
        raise ValueError(f"invalid ranking: {ranking}")
    return ranking[0], {"raw": parsed, "ranking": ranking}


def mode_S3(client, model: str, transcript: str, options: dict) -> tuple[int, dict]:
    """All 6 permutations of the 3 candidates; majority vote on the original-numbering selection."""
    votes: list[int] = []
    perm_records = []
    for perm in permutations((1, 2, 3)):  # 6 orderings
        permuted_options = {f"option_{i + 1}": options[f"option_{p}"] for i, p in enumerate(perm)}
        user = BARE_LLM_S_USER.format(transcript=transcript, **permuted_options)
        raw = _call(client, model, BARE_LLM_SYSTEM, user, max_tokens=200)
        parsed = json.loads(_strip_fences(raw))
        # The model picks 1/2/3 in the PERMUTED order; map back to original
        permuted_choice = int(parsed["choice"])
        if permuted_choice not in (1, 2, 3):
            raise ValueError(f"choice out of range: {permuted_choice}")
        original_choice = perm[permuted_choice - 1]
        votes.append(original_choice)
        perm_records.append({"perm": perm, "permuted_choice": permuted_choice, "original": original_choice})
    # Majority vote (ties broken by lowest option number — matches what perm_voting did in submission)
    counts = {1: votes.count(1), 2: votes.count(2), 3: votes.count(3)}
    winner = max(counts, key=lambda k: (counts[k], -k))
    return winner, {"votes": votes, "counts": counts, "perm_records": perm_records}


def mode_S4(client, model: str, recent_context: str, options: dict) -> tuple[int, dict]:
    """Pairwise: A vs B, B vs C, A vs C; Condorcet winner else most-wins."""
    pairs = [(1, 2), (2, 3), (1, 3)]
    wins = {1: 0, 2: 0, 3: 0}
    pair_records = []
    for a, b in pairs:
        user = BARE_LLM_S4_USER.format(
            recent_context=recent_context,
            option_a=options[f"option_{a}"],
            option_b=options[f"option_{b}"],
        )
        raw = _call(client, model, BARE_LLM_SYSTEM, user, max_tokens=80)
        parsed = json.loads(_strip_fences(raw))
        pick = str(parsed.get("better", "")).strip().upper()
        if pick == "A":
            wins[a] += 1
            winner_pair = a
        elif pick == "B":
            wins[b] += 1
            winner_pair = b
        else:
            raise ValueError(f"invalid pairwise: {parsed}")
        pair_records.append({"pair": (a, b), "winner": winner_pair})
    overall = max(wins, key=lambda k: (wins[k], -k))
    return overall, {"pair_records": pair_records, "wins": wins}


MODE_REGISTRY = {
    "S": mode_S,
    "S2": lambda c, m, t, o: mode_S(c, m, t, o, mode="S2"),
    "R2": mode_R2,
    "S3": mode_S3,
    "S4": mode_S4,  # signature differs (recent_context); handled below
}


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────
def _resolve_cohort(args: argparse.Namespace) -> tuple[dict[str, list], str]:
    """Returns ({session_id: turns}, cohort_label_for_output_dir)."""
    if args.cohort == "test":
        test_dir = REPO_ROOT / "data/MentalRiskES-2026/test/task2/test/data"
        rounds = load_test_rounds(test_dir)
        sessions = build_session_turns(rounds)
        logger.info("Loaded %d rounds, %d sessions from %s (cohort=test)", len(rounds), len(sessions), test_dir)
        return sessions, "test"
    if args.cohort == "trial":
        trial_dir = REPO_ROOT / "data/MentalRiskES-2026/task2_trial/data"
        turns = load_trial_single_session(trial_dir, "trial")
        sessions = {"trial": turns}
        logger.info("Loaded %d turns from %s (cohort=trial)", len(turns), trial_dir)
        return sessions, "trial"
    if args.cohort == "simulated":
        sim_root = REPO_ROOT / "output/mentalriskes/data_prep/simulated/task2"
        sessions = load_simulated_sessions(sim_root)
        logger.info("Loaded %d simulated sessions from %s", len(sessions), sim_root)
        return sessions, "simulated"
    raise ValueError(f"Unknown --cohort {args.cohort!r}")


def run(args: argparse.Namespace) -> None:
    sessions, cohort_label = _resolve_cohort(args)
    sess_ids = list(sessions.keys())
    if args.max_sessions:
        sess_ids = sess_ids[: args.max_sessions]

    out_root = REPO_ROOT / "output/mentalriskes_task2_bare_llm"
    model_short = args.model.replace("/", "_").replace(":", "_")
    # 'test' keeps the legacy bare-model__mode path so existing analysis still works;
    # other cohorts go into a cohort-suffixed subdir.
    suffix = f"{model_short}__{args.mode}" if cohort_label == "test" else f"{model_short}__{args.mode}__{cohort_label}"
    run_dir = out_root / suffix
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_path = run_dir / "raw.jsonl"

    # Resume
    done: set[tuple[str, int]] = set()
    if raw_path.exists():
        with open(raw_path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    done.add((rec["session"], rec["round"]))
                except Exception:
                    continue
        logger.info("Resume: %d (session, round) pairs already done", len(done))

    client = _openrouter_client()
    raw_fh = open(raw_path, "a", encoding="utf-8")
    per_round_preds: dict[int, list[dict]] = {}

    started = time.monotonic()
    n_calls = 0
    n_failed = 0

    handler = MODE_REGISTRY[args.mode]

    try:
        for sid in sess_ids:
            turns = sessions[sid]
            patient_round_ids = sorted({r for r, role, _, _ in turns if role == "patient"})
            for rnd_id in patient_round_ids:
                if (sid, rnd_id) in done:
                    continue
                # The candidate options for this round are attached to the patient turn
                options_dict = next((opts for (r, role, _, opts) in turns
                                     if r == rnd_id and role == "patient" and opts), None)
                if options_dict is None or not all(options_dict.get(f"option_{k}") for k in (1, 2, 3)):
                    continue
                transcript = format_transcript_up_to(turns, rnd_id)
                recent_context = format_recent_context(turns, rnd_id)

                t0 = time.monotonic()
                last_err = None
                for attempt in range(1, args.max_retries + 1):
                    try:
                        if args.mode == "S4":
                            choice, payload = mode_S4(client, args.model, recent_context, options_dict)
                        else:
                            choice, payload = handler(client, args.model, transcript, options_dict)
                        elapsed = time.monotonic() - t0
                        record = {
                            "model": args.model,
                            "mode": args.mode,
                            "cohort": cohort_label,
                            "session": sid,
                            "round": rnd_id,
                            "elapsed_s": round(elapsed, 2),
                            "prediction": choice,
                            **payload,
                        }
                        raw_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                        raw_fh.flush()
                        per_round_preds.setdefault(rnd_id, []).append({
                            "id": sid, "round": rnd_id, "prediction": choice,
                        })
                        n_calls += 1
                        if n_calls % 5 == 0:
                            logger.info("done=%d  last=%s/R%d -> %d  elapsed=%.1fs",
                                        n_calls, sid, rnd_id, choice, elapsed)
                        break
                    except Exception as e:
                        last_err = e
                        wait = min(60, 5 * attempt)
                        logger.warning("call failed for %s R%d (attempt %d/%d): %s - sleep %ds",
                                       sid, rnd_id, attempt, args.max_retries, e, wait)
                        time.sleep(wait)
                else:
                    n_failed += 1
                    logger.error("giving up on %s R%d after %d attempts: %s",
                                 sid, rnd_id, args.max_retries, last_err)
                if args.rate_limit_delay > 0:
                    time.sleep(args.rate_limit_delay)
    finally:
        raw_fh.close()

    if per_round_preds:
        for rnd_id, preds in per_round_preds.items():
            with open(run_dir / f"round{rnd_id}.json", "w", encoding="utf-8") as fh:
                json.dump([{"predictions": preds, "emissions": {}}], fh, ensure_ascii=False, indent=2)

    elapsed = time.monotonic() - started
    logger.info("Run complete: %d new calls in %.1fs (model=%s, mode=%s, failed=%d)",
                n_calls, elapsed, args.model, args.mode, n_failed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="OpenRouter model id (e.g. google/gemma-3-27b-it)")
    parser.add_argument("--mode", required=True, choices=list(MODE_REGISTRY))
    parser.add_argument("--cohort", default="test", choices=("test", "trial", "simulated"),
                        help="Which cohort to evaluate on. test = released test set (10 patients × up to 82 rounds); "
                             "trial = legacy single-session trial transcript (1 patient × 19 rounds); "
                             "simulated = persona-simulated dialogues with labels.json (~7 sessions × ~14 rounds).")
    parser.add_argument("--max-sessions", type=int, default=0)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--rate-limit-delay", type=float, default=2.0)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    run(args)


if __name__ == "__main__":
    main()
