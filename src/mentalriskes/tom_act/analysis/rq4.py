"""RQ4 — structure of expert clinical judgment (spec §4 RQ4, §8.5).

Unit: round (1 gold candidate vs 2 rejected). Characterises whether panels
systematically prefer specific ToM-stance / presencia profiles, conditional on
conversational phase. Candidate-level logistic regression
P(gold-selected | stance, presencia, phase) + descriptive cross-tabulations.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..constants import canonical_phase
from . import common

logger = logging.getLogger(__name__)


def _candidate_frame(run_root, sessions, tables) -> pd.DataFrame:
    stance = tables["tom_stance"]
    presc = tables["presencia"]
    state = tables["llama_state"]
    if stance.empty or presc.empty:
        return pd.DataFrame()

    phase = (state[["session_id", "round", "fase_terapeutica"]].copy()
             if not state.empty else pd.DataFrame())
    if not phase.empty:
        # Defensive normalisation in case the aggregator wasn't re-run.
        phase["fase_terapeutica"] = phase["fase_terapeutica"].map(canonical_phase)

    df = stance.merge(presc, on=["session_id", "round", "candidate"], how="outer")
    if not phase.empty:
        df = df.merge(phase, on=["session_id", "round"], how="left")

    # Gold flag.
    gold_map = {(sid, r.round): r.gold_option
                for sid, sess in sessions.items() for r in sess.rounds}
    df["gold_option"] = df.apply(lambda x: gold_map.get((x["session_id"], x["round"])), axis=1)
    df["is_gold"] = (df["candidate"] == df["gold_option"]).astype(int)
    return df


def run(run_root, sessions, cfg) -> dict:
    tables = common.load_tables(run_root)
    df = _candidate_frame(run_root, sessions, tables)
    if df.empty:
        common.write_result(run_root, "rq4_candidate_logit", pd.DataFrame())
        return {"rq": 4, "n_candidates": 0}

    # Descriptive cross-tabs.
    for col in ("stance", "presencia"):
        if col in df.columns:
            ct = pd.crosstab(df[col], df["is_gold"], normalize="index")
            common.write_result(run_root, f"rq4_crosstab_{col}", ct.reset_index())

    # Candidate-level logistic regression.
    result = pd.DataFrame()
    try:
        import statsmodels.formula.api as smf

        sub = df.dropna(subset=["stance", "presencia"]).copy()
        terms = ["C(stance)", "C(presencia)"]
        if "fase_terapeutica" in sub.columns and sub["fase_terapeutica"].nunique() > 1:
            terms.append("C(fase_terapeutica)")
        if len(sub) >= 20 and sub["is_gold"].nunique() == 2:
            formula = "is_gold ~ " + " + ".join(terms)
            fit = smf.logit(formula, sub).fit(disp=False)
            result = pd.DataFrame({
                "term": fit.params.index,
                "coef": fit.params.values,
                "p": fit.pvalues.values,
                "odds_ratio": np.exp(fit.params.values),
            })
    except Exception as e:  # pragma: no cover
        logger.warning("RQ4 logit failed: %s", e)

    common.write_result(run_root, "rq4_candidate_logit", result)
    common.write_result(run_root, "rq4_candidate_frame", df)
    return {"rq": 4, "n_candidates": len(df), "n_terms": len(result)}
