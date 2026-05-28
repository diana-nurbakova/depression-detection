"""RQ2 — round-level model-derived ACT × ToM (spec §4 RQ2, §8.2).

Unit: round (n ≤ 568). Two parallel Llama-derived operationalisations of
psychological flexibility:
  Set A — procesos_act hexaflex (6 dims, incl. yo_como_contexto).
  Set B — Llama-derived CompACT-10 → OE/BA/VA subscales.

Internal consistency between A and B is reported. Primary models are mixed
effects with a per-session random intercept; Spearman is the descriptive start.
Both sides are LLM-derived but from different model classes (Llama vs Gemma).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..constants import ACT_PROCESS_KEYS
from . import common

logger = logging.getLogger(__name__)

# A↔B internal-consistency pairs (spec §4 RQ2).
CONSISTENCY = [
    ("defusion+aceptacion", "OE"),
    ("momento_presente", "BA"),
    ("valores+accion_comprometida", "VA"),
]


def _round_frame(run_root, tables) -> pd.DataFrame:
    state = tables["llama_state"]
    if state.empty:
        return pd.DataFrame()
    df = state.copy()

    # Set B subscales (Llama CompACT-10).
    subB = common.llama_compact_subscales(tables["llama_assessors"])
    if not subB.empty:
        df = df.merge(subB, on=["session_id", "round"], how="left")

    # Composite Set-A signals for consistency.
    if {"defusion", "aceptacion"}.issubset(df.columns):
        df["defusion+aceptacion"] = df["defusion"] + df["aceptacion"]
    if {"valores", "accion_comprometida"}.issubset(df.columns):
        df["valores+accion_comprometida"] = df["valores"] + df["accion_comprometida"]

    # DV: headline gaps + tier soft scores + temporal W1.
    gaps = common.headline_gap_series(tables["cross_perspective"])
    if not gaps.empty:
        df = df.merge(gaps[["session_id", "round", "gap_conservative", "gap_realistic"]],
                      on=["session_id", "round"], how="left")
    tier = tables["tom_tier"]
    if not tier.empty:
        df = df.merge(tier[["session_id", "round", "somatico", "cognitivo", "afectivo"]],
                      on=["session_id", "round"], how="left")
    return df


def _mixedlm(df: pd.DataFrame, dv: str, iv: str) -> dict:
    """Fixed effect of iv on dv with random intercept per session."""
    import statsmodels.formula.api as smf

    sub = df[["session_id", dv, iv]].copy()
    sub[dv] = pd.to_numeric(sub[dv], errors="coerce")
    sub[iv] = pd.to_numeric(sub[iv], errors="coerce")
    sub = sub.dropna()
    if sub["session_id"].nunique() < 2 or len(sub) < 10 or sub[iv].nunique() < 2:
        return {"beta": float("nan"), "p": float("nan"),
                "ci_lo": float("nan"), "ci_hi": float("nan"), "n": len(sub)}
    try:
        md = smf.mixedlm(f"Q('{dv}') ~ Q('{iv}')", sub, groups=sub["session_id"])
        fit = md.fit(reml=False, method="lbfgs", disp=False)
        name = [p for p in fit.params.index if iv in p][-1]
        ci = fit.conf_int().loc[name]
        return {"beta": float(fit.params[name]), "p": float(fit.pvalues[name]),
                "ci_lo": float(ci[0]), "ci_hi": float(ci[1]), "n": len(sub)}
    except Exception as e:  # pragma: no cover
        logger.warning("MixedLM failed (%s~%s): %s", dv, iv, e)
        return {"beta": float("nan"), "p": float("nan"),
                "ci_lo": float("nan"), "ci_hi": float("nan"), "n": len(sub)}


def _gold_calibration(sessions, llama_assessors: pd.DataFrame, threshold: float = 0.4) -> pd.DataFrame:
    """Move C (spec §8.2): per-session L1 between Llama session-mean CompACT-10
    and the gold item vector, normalised by 60. Flags divergent sessions."""
    if llama_assessors.empty:
        return pd.DataFrame()
    comp = llama_assessors[llama_assessors["instrument"] == "CompACT-10"]
    rows = []
    for sid, sess in sessions.items():
        if len(sess.gold_compact10) != 10:
            continue
        g = comp[comp["session_id"] == sid]
        if g.empty:
            continue
        mean_vec = g.groupby("item")["score"].mean().reindex(range(1, 11))
        if mean_vec.isna().any():
            continue
        l1 = float(np.abs(mean_vec.values - np.array(sess.gold_compact10)).sum())
        rows.append({"session_id": sid, "l1_norm": l1 / 60.0,
                     "flagged": bool(l1 / 60.0 > threshold)})
    return pd.DataFrame(rows)


def run(run_root, sessions, cfg) -> dict:
    tables = common.load_tables(run_root)
    df = _round_frame(run_root, tables)
    if df.empty:
        common.write_result(run_root, "rq2_round_models", pd.DataFrame())
        return {"rq": 2, "n_rounds": 0}

    # Move C — gold-anchored calibration check (no extra LLM calls).
    calib = _gold_calibration(sessions, tables["llama_assessors"])
    common.write_result(run_root, "rq2_gold_calibration", calib)
    flagged = set(calib[calib["flagged"]]["session_id"]) if not calib.empty else set()

    n_boot = cfg.get("bootstrap_resamples", 10000)

    # Internal consistency (Set A vs Set B).
    cons_rows = []
    for a, b in CONSISTENCY:
        if a in df.columns and b in df.columns:
            res = common.bootstrap_spearman(df[a], df[b], n_boot, seed=2)
            cons_rows.append({"set_a": a, "set_b": b, **res})
    common.write_result(run_root, "rq2_internal_consistency",
                        common.add_fdr(pd.DataFrame(cons_rows)))

    # Mixed-effects: ACT processes (Set A + B) × ToM DVs.
    act_ivs = [k for k in ACT_PROCESS_KEYS if k in df.columns] + \
              [c for c in ["OE", "BA", "VA"] if c in df.columns]
    dvs = [c for c in ["gap_conservative", "gap_realistic",
                       "somatico", "cognitivo", "afectivo"] if c in df.columns]
    rows = []
    for iv in act_ivs:
        for dv in dvs:
            rows.append({"iv": iv, "dv": dv, **_mixedlm(df, dv, iv)})
    result = common.add_fdr(pd.DataFrame(rows))
    common.write_result(run_root, "rq2_round_models", result)

    # Move C: re-run excluding gold-divergent sessions for a robustness annotation.
    if flagged:
        df_excl = df[~df["session_id"].isin(flagged)]
        rows_excl = [{"iv": iv, "dv": dv, **_mixedlm(df_excl, dv, iv)}
                     for iv in act_ivs for dv in dvs]
        common.write_result(run_root, "rq2_round_models_excl_flagged",
                            common.add_fdr(pd.DataFrame(rows_excl)))

    common.write_result(run_root, "rq2_round_frame", df)
    return {"rq": 2, "n_rounds": len(df), "n_models": len(result),
            "n_consistency": len(cons_rows), "n_flagged_sessions": len(flagged)}
