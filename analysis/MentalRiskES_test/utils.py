"""Shared utilities for MentalRiskES 2026 test set analysis.

Loads gold/prediction data, classifies severity bands, computes MCID.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config() -> dict:
    cfg_path = Path(__file__).parent / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def repo_path(rel: str) -> Path:
    return REPO_ROOT / rel


# ---------------------------------------------------------------------
# Task 1 loaders
# ---------------------------------------------------------------------
def load_task1_gold(cfg: dict) -> dict[str, dict]:
    """Returns {session_id: {GAD-7: [...], PHQ-9: [...], CompACT-10: [...]}}."""
    with open(repo_path(cfg["paths"]["task1_gold"]), "r", encoding="utf-8") as f:
        return json.load(f)


def load_task1_predictions(cfg: dict, run_idx: int) -> dict[int, dict[str, dict]]:
    """Returns {round: {session_id: {GAD-7: [...], PHQ-9: [...], CompACT-10: [...]}}}."""
    pred_dir = repo_path(cfg["paths"]["task1_predictions_dir"])
    out: dict[int, dict[str, dict]] = {}
    for fp in sorted(pred_dir.glob(f"round*_run{run_idx}.json")):
        # round{N}_run{R}.json
        rnd = int(fp.stem.split("_")[0].replace("round", ""))
        with open(fp, "r", encoding="utf-8") as f:
            payload = json.load(f)
        # payload structure: [{"predictions": [...], "emissions": {}}]
        round_preds: dict[str, dict] = {}
        for entry in payload[0]["predictions"]:
            sid = entry["id"]
            round_preds[sid] = entry["prediction"]
        out[rnd] = round_preds
    return out


def task1_last_round_predictions(cfg: dict, run_idx: int) -> dict[str, dict]:
    """For each session, return the prediction at its highest available round."""
    rounds = load_task1_predictions(cfg, run_idx)
    last: dict[str, tuple[int, dict]] = {}
    for rnd, sessions in rounds.items():
        for sid, pred in sessions.items():
            if sid not in last or rnd > last[sid][0]:
                last[sid] = (rnd, pred)
    return {sid: pred for sid, (_, pred) in last.items()}


def task1_last_round_per_session(cfg: dict, run_idx: int) -> dict[str, int]:
    rounds = load_task1_predictions(cfg, run_idx)
    last: dict[str, int] = {}
    for rnd, sessions in rounds.items():
        for sid in sessions:
            last[sid] = max(last.get(sid, 0), rnd)
    return last


# ---------------------------------------------------------------------
# Task 2 loaders
# ---------------------------------------------------------------------
def load_task2_gold(cfg: dict) -> dict[int, dict[str, str]]:
    """Returns {round: {session_id: 'option_X'}}."""
    gold_dir = repo_path(cfg["paths"]["task2_gold_dir"])
    out: dict[int, dict[str, str]] = {}
    for fp in sorted(gold_dir.glob("round_*_gold.json")):
        rnd = int(fp.stem.replace("round_", "").replace("_gold", ""))
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        out[rnd] = {sid: payload["correct_option"] for sid, payload in data.items()}
    return out


def load_task2_test(cfg: dict) -> dict[int, dict[str, dict]]:
    """Returns {round: {session_id: {patient_input, option_1, option_2, option_3}}}."""
    data_dir = repo_path(cfg["paths"]["task2_data_dir"])
    out: dict[int, dict[str, dict]] = {}
    for fp in sorted(data_dir.glob("round_*.json")):
        rnd = int(fp.stem.replace("round_", ""))
        with open(fp, "r", encoding="utf-8") as f:
            out[rnd] = json.load(f)
    return out


def load_task2_predictions(cfg: dict, run_idx: int) -> dict[int, dict[str, int]]:
    """Returns {round: {session_id: predicted_option_int (1/2/3)}}."""
    pred_dir = repo_path(cfg["paths"]["task2_predictions_dir"])
    out: dict[int, dict[str, int]] = {}
    for fp in sorted(pred_dir.glob(f"round*_run{run_idx}.json")):
        rnd = int(fp.stem.split("_")[0].replace("round", ""))
        with open(fp, "r", encoding="utf-8") as f:
            payload = json.load(f)
        round_preds: dict[str, int] = {}
        for entry in payload[0]["predictions"]:
            round_preds[entry["id"]] = int(entry["prediction"])
        out[rnd] = round_preds
    return out


# ---------------------------------------------------------------------
# Severity-band classification
# ---------------------------------------------------------------------
def classify_band(total: int, instrument: str, cfg: dict) -> str:
    cutoffs = cfg["instruments"][instrument]["band_cutoffs"]
    if not cutoffs:
        return "n/a"
    for lo, hi, name in cutoffs:
        if lo <= total <= hi:
            return name
    return "out_of_range"


def band_index(band: str, instrument: str, cfg: dict) -> int:
    """Ordinal index for severity band (0 = least severe)."""
    cutoffs = cfg["instruments"][instrument]["band_cutoffs"]
    for i, (_, _, name) in enumerate(cutoffs):
        if name == band:
            return i
    return -1


# ---------------------------------------------------------------------
# Convenience aggregators
# ---------------------------------------------------------------------
def mae(pred: list[int], gold: list[int]) -> float:
    return sum(abs(p - g) for p, g in zip(pred, gold)) / len(gold)


def signed_bias(pred: list[int], gold: list[int]) -> float:
    return sum(p - g for p, g in zip(pred, gold)) / len(gold)


def total(score_list: list[int]) -> int:
    return int(sum(score_list))


def per_item_signed(pred: list[int], gold: list[int]) -> list[int]:
    return [p - g for p, g in zip(pred, gold)]
