"""RQ5 — co-occurrence of assessor-state ruptures with perspective / ACT
discontinuities (spec §4 RQ5, §7.4, §8.3).

Unit: rupture event (rows where ``fired=True`` in the temporal Wasserstein
traces, consecutive variant). At each rupture round we test whether the
aggregate perspective gap or an ACT process score departs from its 5-round
rolling mean by more than 2 SD. Descriptive only; no directional prediction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..constants import ACT_PROCESS_KEYS
from . import common

ROLL_WINDOW = 5
ROLL_SIGMA = 2.0


def _rolling_outlier(series: pd.Series, round_idx: int) -> bool:
    """True if value at round_idx departs > 2 SD from its trailing 5-round mean."""
    s = series.dropna()
    if round_idx not in s.index:
        return False
    prior = s[s.index < round_idx].tail(ROLL_WINDOW)
    if len(prior) < 3:
        return False
    mu, sd = prior.mean(), prior.std()
    if sd == 0 or np.isnan(sd):
        return False
    return bool(abs(s.loc[round_idx] - mu) > ROLL_SIGMA * sd)


def run(run_root, sessions, cfg) -> dict:
    tables = common.load_tables(run_root)
    temporal = tables["temporal"]
    if temporal.empty:
        common.write_result(run_root, "rq5_rupture_overlay", pd.DataFrame())
        return {"rq": 5, "n_ruptures": 0}

    t = temporal.copy()
    if "variant" in t:
        t = t[t["variant"] == "consecutive"]
    fired = t[t["fired"].astype(str).str.lower().isin(["true", "1"])]

    gaps = common.headline_gap_series(tables["cross_perspective"])
    state = tables["llama_state"]

    rows = []
    for _, ev in fired.iterrows():
        sid, rnd, inst = ev["run"], int(ev["round"]), ev["instrument"]
        rec = {"session_id": sid, "round": rnd, "rupture_instrument": inst,
               "w1": ev.get("w1")}
        # Perspective-gap discontinuity.
        if not gaps.empty:
            gs = gaps[gaps["session_id"] == sid].set_index("round")
            for g in ("gap_conservative", "gap_realistic"):
                if g in gs.columns:
                    rec[f"{g}_outlier"] = _rolling_outlier(gs[g], rnd)
        # ACT-process discontinuities.
        if not state.empty:
            ss = state[state["session_id"] == sid].set_index("round")
            for k in ACT_PROCESS_KEYS:
                if k in ss.columns:
                    rec[f"{k}_outlier"] = _rolling_outlier(ss[k], rnd)
        rows.append(rec)

    result = pd.DataFrame(rows)
    common.write_result(run_root, "rq5_rupture_overlay", result)
    return {"rq": 5, "n_ruptures": len(result)}
