"""Wasserstein computations (spec §7).

Two W1 (earth-mover) signal families on instrument item-vector distributions,
both under the clinical ground metrics from ``task1/temporal.py``:

  - Cross-perspective (§7.1–7.2): six pairwise W1 between the four Gemma views
    per instrument per round, plus a scale-range-weighted aggregate
    "perspective gap" across instruments.
  - Temporal (§7.3): W1 between consecutive Llama assessor vectors with a
    running μ+2σ rupture alert. Emitted in **both** variants:
      * ``consecutive`` — round t vs round t-1 (spec §7.3 step 2).
      * ``barycenter``  — round t vs running mean of rounds 1..t (matches the
        existing ``trial_wasserstein.csv`` procedure, §14.2).

Core functions take plain Python structures and are unit-tested directly; the
parquet-driven orchestration (``*_from_tables``) wires them to aggregator output.
"""

from __future__ import annotations

import logging

import numpy as np

from ..task1.temporal import get_ground_metric
from .constants import INSTRUMENTS

logger = logging.getLogger(__name__)

# Six view pairs (§7.1). First two are the headline gaps.
VIEW_PAIRS = [
    ("self_a", "observer_p"),    # headline, conservative
    ("self_b", "observer_pt"),   # headline, realistic
    ("self_a", "self_b"),        # disclosure vs endorsement
    ("observer_p", "observer_pt"),  # therapist influence on observer
    ("self_a", "observer_pt"),   # diagnostic
    ("self_b", "observer_p"),    # diagnostic
]

try:
    import ot
    _HAVE_OT = True
except ImportError:  # pragma: no cover
    _HAVE_OT = False


def w1_between(vec_a: list[int] | np.ndarray, vec_b: list[int] | np.ndarray,
               instrument: str) -> float:
    """W1 between two item-vectors as item-index distributions (§7.1).

    Each vector is normalised to sum 1 (mass = item values); distance under the
    instrument's clinical ground metric. Falls back to normalised L1 if POT is
    unavailable.
    """
    a = np.asarray(vec_a, dtype=float)
    b = np.asarray(vec_b, dtype=float)
    sa, sb = a.sum(), b.sum()
    if sa <= 0 or sb <= 0:
        return 0.0
    a, b = a / sa, b / sb
    if _HAVE_OT:
        M = get_ground_metric(instrument)
        return float(ot.emd2(a, b, M))
    return float(np.sum(np.abs(a - b)))


# ---------------------------------------------------------------------------
# Cross-perspective (§7.1–7.2)
# ---------------------------------------------------------------------------

def cross_perspective_round(views: dict[str, list[int]], instrument: str) -> dict[str, float]:
    """Six pairwise W1 for one round / instrument. Missing views → skipped."""
    out: dict[str, float] = {}
    for va, vb in VIEW_PAIRS:
        if va in views and vb in views:
            out[f"{va}__{vb}"] = w1_between(views[va], views[vb], instrument)
    return out


def _instrument_weights() -> dict[str, float]:
    ranges = {k: v["scale_range"] for k, v in INSTRUMENTS.items()}
    total = sum(ranges.values())
    return {k: v / total for k, v in ranges.items()}


def aggregate_gap(per_instrument: dict[str, float]) -> float:
    """Scale-range-weighted aggregate across instruments (§7.2)."""
    w = _instrument_weights()
    num = sum(per_instrument[i] * w[i] for i in per_instrument if i in w)
    den = sum(w[i] for i in per_instrument if i in w)
    return num / den if den else float("nan")


def cross_perspective_from_tables(view_long) -> "pd.DataFrame":  # noqa: F821
    """Compute per-(session,round) cross-perspective gaps from a long view table.

    Input columns: session_id, round, view, instrument, item, score.
    Output: long rows session_id, round, instrument ('PHQ-9'…|'AGG'), pair, w1.
    """
    import pandas as pd

    rows: list[dict] = []
    grouped = view_long.groupby(["session_id", "round"])
    for (sid, rnd), g in grouped:
        # Per instrument: build {view: vector}.
        agg_by_pair: dict[str, dict[str, float]] = {}
        for inst in INSTRUMENTS:
            gi = g[g["instrument"] == inst]
            views = {}
            for view, gv in gi.groupby("view"):
                vec = gv.sort_values("item")["score"].tolist()
                if len(vec) == INSTRUMENTS[inst]["n_items"]:
                    views[view] = vec
            pairs = cross_perspective_round(views, inst)
            for pair, w1 in pairs.items():
                rows.append({"session_id": sid, "round": rnd, "instrument": inst,
                             "pair": pair, "w1": w1})
                agg_by_pair.setdefault(pair, {})[inst] = w1
        for pair, per_inst in agg_by_pair.items():
            rows.append({"session_id": sid, "round": rnd, "instrument": "AGG",
                         "pair": pair, "w1": aggregate_gap(per_inst)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Temporal (§7.3) — both variants
# ---------------------------------------------------------------------------

def temporal_traces(vectors_by_round: list[tuple[int, list[int]]], instrument: str,
                    variant: str, alert_sigma: float = 2.0) -> list[dict]:
    """Temporal W1 trace + running μ+kσ rupture alert for one (session, instrument).

    Args:
        vectors_by_round: ascending [(round, item_vector), …].
        variant: ``"consecutive"`` (t-1 vs t) or ``"barycenter"`` (t vs mean 1..t).
        alert_sigma: σ multiplier for the alert threshold.

    Returns rows matching the trial CSV schema (+ ``variant``).
    """
    ordered = sorted(vectors_by_round, key=lambda x: x[0])
    w1_traj: list[float] = []
    for k, (_, vec) in enumerate(ordered):
        if k == 0:
            w1_traj.append(0.0)
        elif variant == "consecutive":
            w1_traj.append(w1_between(ordered[k - 1][1], vec, instrument))
        elif variant == "barycenter":
            bary = np.mean([v for _, v in ordered[: k + 1]], axis=0)
            w1_traj.append(w1_between(bary, vec, instrument))
        else:
            raise ValueError(f"unknown variant {variant!r}")

    rows: list[dict] = []
    for k, (round_n, _) in enumerate(ordered):
        past = w1_traj[:k]
        if k < 3:
            mu = float("nan") if k == 0 else float(np.mean(past))
            sigma = float("nan") if k <= 1 else float(np.std(past))
            threshold = float("nan") if k <= 1 else mu + alert_sigma * sigma
            fired = False
        else:
            mu = float(np.mean(past))
            sigma = float(np.std(past))
            threshold = mu + alert_sigma * sigma
            fired = bool(w1_traj[k] > threshold)
        rows.append({
            "variant": variant,
            "instrument": instrument,
            "round": round_n,
            "w1": round(w1_traj[k], 4),
            # Keep NaN as float (parquet-safe); rendered as "" at CSV time.
            "running_mu_through_prev": float("nan") if np.isnan(mu) else round(mu, 4),
            "running_sigma_through_prev": float("nan") if np.isnan(sigma) else round(sigma, 4),
            "threshold_mu_plus_2sigma": float("nan") if np.isnan(threshold) else round(threshold, 4),
            "fired": fired,
        })
    return rows


def temporal_from_tables(assessor_long, variants: list[str],
                         alert_sigma: float = 2.0) -> "pd.DataFrame":  # noqa: F821
    """Compute temporal traces from a long assessor table.

    Input columns: session_id, round, instrument, item, score.
    Output: rows run(=session_id), variant, instrument, round, w1, running μ/σ,
    threshold, fired.
    """
    import pandas as pd

    rows: list[dict] = []
    for (sid, inst), g in assessor_long.groupby(["session_id", "instrument"]):
        by_round: list[tuple[int, list[int]]] = []
        for rnd, gr in g.groupby("round"):
            vec = gr.sort_values("item")["score"].tolist()
            if len(vec) == INSTRUMENTS[inst]["n_items"]:
                by_round.append((int(rnd), vec))
        for variant in variants:
            for row in temporal_traces(by_round, inst, variant, alert_sigma):
                rows.append({"run": sid, **row})
    return pd.DataFrame(rows)
