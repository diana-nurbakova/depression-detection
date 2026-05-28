"""RQ1 — session-level, gold-anchored correlations (spec §4 RQ1, §8.1).

Unit: session (n ≤ 10). One side of every correlation is a clinician-administered
gold psychometric score. Spearman + bootstrap 95% CI + BH-FDR. Low power; the
interpretive weight is consistency of direction across granularities.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import common

# Predicted relations (iv, dv, expected_direction). See RQ1 table in the spec.
PREDICTED = [
    ("OE", "prop_cognitivo", "+"),
    ("OE", "prop_afectivo", "+"),
    ("BA", "prop_somatico", "+"),
    ("VA", "prop_cognitivo", "+ (weaker than OE)"),
]


def _session_frame(run_root, sessions, tables) -> pd.DataFrame:
    gaps = common.headline_gap_series(tables["cross_perspective"])
    tiers = common.tier_proportions(tables["tom_tier"])

    # Mean temporal W1 per session (consecutive variant, across instruments).
    temporal = tables["temporal"]
    temp_mean = pd.DataFrame()
    if not temporal.empty:
        t = temporal.copy()
        if "variant" in t:
            t = t[t["variant"] == "consecutive"]
        t["w1"] = pd.to_numeric(t["w1"], errors="coerce")
        temp_mean = (t.groupby("run")["w1"].mean()
                     .rename("mean_temporal_w1").reset_index()
                     .rename(columns={"run": "session_id"}))

    rows = []
    for sid, sess in sessions.items():
        sub = common.gold_subscales(sess)
        row = {
            "session_id": sid,
            "OE": sub["OE"], "BA": sub["BA"], "VA": sub["VA"],
            "phq9_total": sum(sess.gold_phq9) if sess.gold_phq9 else np.nan,
            "gad7_total": sum(sess.gold_gad7) if sess.gold_gad7 else np.nan,
        }
        for i, v in enumerate(sess.gold_compact10, 1):
            row[f"compact_item_{i}"] = v
        rows.append(row)
    df = pd.DataFrame(rows)

    if not gaps.empty:
        gm = gaps.groupby("session_id")[["gap_conservative", "gap_realistic"]].mean().reset_index()
        df = df.merge(gm, on="session_id", how="left")
    if not tiers.empty:
        df = df.merge(tiers, on="session_id", how="left")
    if not temp_mean.empty:
        df = df.merge(temp_mean, on="session_id", how="left")
    return df


def run(run_root, sessions, cfg) -> dict:
    tables = common.load_tables(run_root)
    sf = _session_frame(run_root, sessions, tables)

    ivs = ["OE", "BA", "VA", "phq9_total", "gad7_total"]
    dvs = [c for c in ["prop_somatico", "prop_cognitivo", "prop_afectivo",
                       "gap_conservative", "gap_realistic", "mean_temporal_w1"]
           if c in sf.columns]
    predicted = {(iv, dv): d for iv, dv, d in PREDICTED}

    n_boot = cfg.get("bootstrap_resamples", 10000)
    rows = []
    for iv in ivs:
        if iv not in sf.columns:
            continue
        for dv in dvs:
            res = common.bootstrap_spearman(sf[iv], sf[dv], n_boot, seed=1)
            rows.append({"iv": iv, "dv": dv, **res,
                         "predicted_direction": predicted.get((iv, dv), "")})
    result = common.add_fdr(pd.DataFrame(rows))
    common.write_result(run_root, "rq1_session_correlations", result)
    common.write_result(run_root, "rq1_session_frame", sf)
    return {"rq": 1, "n_sessions": len(sf), "n_correlations": len(result)}
