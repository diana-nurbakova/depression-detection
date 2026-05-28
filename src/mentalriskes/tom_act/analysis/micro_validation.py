"""Micro-validation of ToM-tier coding (spec §11).

Draws 10 patient turns at random — one per session — for manual coding by the
researcher against the §5.4 operational definitions, blind to the Gemma
prediction. Produces a CSV with the sampled turns and the Gemma argmax in a
*separate* column to fill in manually; a small confusion matrix is computed
later once the manual column is filled. n=10 is descriptive only.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import pandas as pd

from . import common

logger = logging.getLogger(__name__)


def run(run_root, sessions, cfg) -> dict:
    seed = cfg.get("micro_validation_seed", 20260527)
    rng = random.Random(seed)
    tables = common.load_tables(run_root)
    tier = tables["tom_tier"]

    rows = []
    for sid, sess in sessions.items():
        if not sess.rounds:
            continue
        r = rng.choice(sess.rounds)
        gemma = ""
        if not tier.empty:
            sel = tier[(tier["session_id"] == sid) & (tier["round"] == r.round)]
            if not sel.empty:
                gemma = sel.iloc[0]["argmax"]
        rows.append({
            "session_id": sid,
            "round": r.round,
            "patient_turn": r.patient_input,
            "manual_tier": "",          # to be filled by the researcher
            "manual_rationale": "",
            "gemma_argmax": gemma,       # not consulted until manual coding is done
        })

    df = pd.DataFrame(rows)
    out_dir = Path(run_root) / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "micro_validation_tom_tier.csv"
    df.to_csv(path, index=False, encoding="utf-8")
    logger.info("Wrote micro-validation sample (seed=%d) -> %s", seed, path)
    return {"micro_validation": str(path), "n": len(df), "seed": seed}
