"""RQ3 — lagged therapist intervention → patient state (spec §4 RQ3, §8.4).

Unit: round transition t → t+1. IVs: ToM-stance and presencia of the gold
(delivered) therapist response at round t. DVs: change in procesos_act t→t+1
(esp. yo_como_contexto), change in perspective gap, ToM-tier at t+1.
Mixed-effects with categorical fixed effects + per-session random intercept.
Correlational with a fixed temporal ordering — not causal.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..constants import ACT_PROCESS_KEYS
from . import common

logger = logging.getLogger(__name__)


def _gold_candidate_signal(table: pd.DataFrame, sessions, field: str) -> pd.DataFrame:
    """Pick the gold-selected candidate's signal value per (session, round)."""
    if table.empty:
        return pd.DataFrame(columns=["session_id", "round", field])
    rows = []
    for sid, sess in sessions.items():
        for r in sess.rounds:
            if r.gold_option is None:
                continue
            sel = table[(table["session_id"] == sid) & (table["round"] == r.round)
                        & (table["candidate"] == r.gold_option)]
            if not sel.empty:
                rows.append({"session_id": sid, "round": r.round,
                             field: sel.iloc[0][field]})
    return pd.DataFrame(rows)


def _transition_frame(run_root, sessions, tables) -> pd.DataFrame:
    state = tables["llama_state"]
    if state.empty:
        return pd.DataFrame()
    gaps = common.headline_gap_series(tables["cross_perspective"])
    tier = tables["tom_tier"]

    stance = _gold_candidate_signal(tables["tom_stance"], sessions, "stance")
    presc = _gold_candidate_signal(tables["presencia"], sessions, "presencia")

    rows = []
    for sid, sess in sessions.items():
        srounds = {r.round for r in sess.rounds}
        st = state[state["session_id"] == sid].set_index("round")
        gp = gaps[gaps["session_id"] == sid].set_index("round") if not gaps.empty else None
        tr = tier[tier["session_id"] == sid].set_index("round") if not tier.empty else None
        for t in sorted(srounds):
            if (t + 1) not in srounds or t not in st.index or (t + 1) not in st.index:
                continue
            row = {"session_id": sid, "round": t}
            for k in ACT_PROCESS_KEYS:
                if k in st.columns:
                    row[f"d_{k}"] = st.loc[t + 1, k] - st.loc[t, k]
            if gp is not None and t in gp.index and (t + 1) in gp.index:
                for g in ("gap_conservative", "gap_realistic"):
                    if g in gp.columns:
                        row[f"d_{g}"] = gp.loc[t + 1, g] - gp.loc[t, g]
            if tr is not None and (t + 1) in tr.index:
                row["tier_next"] = tr.loc[t + 1, "argmax"]
            rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if not stance.empty:
        df = df.merge(stance, on=["session_id", "round"], how="left")
    if not presc.empty:
        df = df.merge(presc, on=["session_id", "round"], how="left")
    return df


def _mixedlm_categorical(df, dv, cat_iv) -> list[dict]:
    """Mixed-effects of a categorical IV on continuous dv (per-session intercept)."""
    import statsmodels.formula.api as smf

    sub = df[["session_id", dv, cat_iv]].copy()
    sub[dv] = pd.to_numeric(sub[dv], errors="coerce")
    sub = sub.dropna()
    if sub["session_id"].nunique() < 2 or len(sub) < 10 or sub[cat_iv].nunique() < 2:
        return [{"dv": dv, "iv": cat_iv, "level": "", "beta": float("nan"),
                 "p": float("nan"), "n": len(sub)}]
    try:
        md = smf.mixedlm(f"Q('{dv}') ~ C(Q('{cat_iv}'))", sub, groups=sub["session_id"])
        fit = md.fit(reml=False, method="lbfgs", disp=False)
        out = []
        for name in fit.params.index:
            if cat_iv in name and "Intercept" not in name and "Group" not in name:
                out.append({"dv": dv, "iv": cat_iv, "level": name,
                            "beta": float(fit.params[name]), "p": float(fit.pvalues[name]),
                            "n": len(sub)})
        return out or [{"dv": dv, "iv": cat_iv, "level": "", "beta": float("nan"),
                        "p": float("nan"), "n": len(sub)}]
    except Exception as e:  # pragma: no cover
        logger.warning("RQ3 MixedLM failed (%s~%s): %s", dv, cat_iv, e)
        return [{"dv": dv, "iv": cat_iv, "level": "", "beta": float("nan"),
                 "p": float("nan"), "n": len(sub)}]


def run(run_root, sessions, cfg) -> dict:
    tables = common.load_tables(run_root)
    df = _transition_frame(run_root, sessions, tables)
    if df.empty:
        common.write_result(run_root, "rq3_lagged_models", pd.DataFrame())
        return {"rq": 3, "n_transitions": 0}

    dvs = [c for c in df.columns if c.startswith("d_")]
    rows = []
    for cat_iv in ("stance", "presencia"):
        if cat_iv not in df.columns:
            continue
        for dv in dvs:
            rows.extend(_mixedlm_categorical(df, dv, cat_iv))
    result = common.add_fdr(pd.DataFrame(rows))
    common.write_result(run_root, "rq3_lagged_models", result)
    common.write_result(run_root, "rq3_transition_frame", df)
    return {"rq": 3, "n_transitions": len(df), "n_models": len(result)}
