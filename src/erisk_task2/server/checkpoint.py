"""Checkpoint save/load for crash recovery (Spec Section 17.3.4)."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

from erisk_task2.models import RunUserState, UserProfile

logger = logging.getLogger(__name__)


def save_checkpoint(
    checkpoint_dir: str | Path,
    round_number: int,
    master_user_list: list[str],
    profiles: dict[str, UserProfile],
    run_states: dict[int, dict[str, RunUserState]],
) -> Path:
    """Save full pipeline state to disk.

    Returns path to saved checkpoint file.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    path = checkpoint_dir / f"round_{round_number:04d}_state.pkl"
    data = {
        "round_number": round_number,
        "master_user_list": master_user_list,
        "profiles": profiles,
        "run_states": run_states,
    }

    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info("Checkpoint saved: %s", path)

    # Keep only the last 2 checkpoints to avoid filling disk
    old = sorted(checkpoint_dir.glob("round_*_state.pkl"))[:-2]
    for f in old:
        f.unlink()
        logger.debug("Removed old checkpoint: %s", f)

    return path


def load_latest_checkpoint(
    checkpoint_dir: str | Path,
) -> tuple[int, list[str], dict[str, UserProfile], dict[int, dict[str, RunUserState]]] | None:
    """Load the most recent checkpoint.

    Returns (round_number, master_user_list, profiles, run_states) or None if no checkpoint.
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None

    checkpoints = sorted(checkpoint_dir.glob("round_*_state.pkl"))
    if not checkpoints:
        return None

    latest = checkpoints[-1]
    logger.info("Loading checkpoint: %s", latest)

    with open(latest, "rb") as f:
        data = pickle.load(f)

    return (
        data["round_number"],
        data["master_user_list"],
        data["profiles"],
        data["run_states"],
    )
