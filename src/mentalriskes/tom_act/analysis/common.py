"""Shared helpers for the RQ analyses."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..constants import compact10_subscale_scores

logger = logging.getLogger(__name__)

AGG_DIR = "outputs/aggregated"
WASS_DIR = "outputs/wasserstein_test"
XPERSP_DIR = "outputs/cross_perspective"
ANALYSIS_DIR = "outputs/analysis"

# Headline perspective-gap pairs (column form used in cross-perspective table).
HEADLINE_PAIRS = {
    "gap_conservative": "self_a__observer_p",
    "gap_realistic": "self_b__observer_pt",
}


def load_parquet(run_root: Path, rel: str) -> pd.DataFrame:
    path = Path(run_root) / rel
    if path.exists():
        return pd.read_parquet(path)
    logger.warning("Missing table %s", path)
    return pd.DataFrame()


def load_tables(run_root: str | Path) -> dict[str, pd.DataFrame]:
    run_root = Path(run_root)
    names = ["llama_assessors", "llama_state", "gemma_views", "tom_tier",
             "tom_stance", "presencia"]
    tables = {n: load_parquet(run_root, f"{AGG_DIR}/{n}.parquet") for n in names}
    tables["temporal"] = load_parquet(run_root, f"{WASS_DIR}/temporal.csv".replace(".csv", ".parquet"))
    if tables["temporal"].empty:
        csv = Path(run_root) / WASS_DIR / "temporal.csv"
        if csv.exists():
            tables["temporal"] = pd.read_csv(csv)
    tables["cross_perspective"] = load_parquet(run_root, f"{XPERSP_DIR}/gaps.parquet")
    return tables


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def bootstrap_spearman(x, y, n_resamples: int = 10000, seed: int = 0) -> dict:
    """Spearman rho + bootstrap 95% CI. Returns NaNs if < 4 paired points."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 4 or np.all(x == x[0]) or np.all(y == y[0]):
        return {"rho": float("nan"), "p": float("nan"),
                "ci_lo": float("nan"), "ci_hi": float("nan"), "n": n}
    rho, p = spearmanr(x, y)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        if np.all(x[idx] == x[idx][0]) or np.all(y[idx] == y[idx][0]):
            continue
        r, _ = spearmanr(x[idx], y[idx])
        if not np.isnan(r):
            boots.append(r)
    if boots:
        lo, hi = np.percentile(boots, [2.5, 97.5])
    else:
        lo = hi = float("nan")
    return {"rho": float(rho), "p": float(p), "ci_lo": float(lo), "ci_hi": float(hi), "n": n}


def bh_fdr(pvals: list[float]) -> list[float]:
    """Benjamini-Hochberg FDR correction; NaNs passed through."""
    from statsmodels.stats.multitest import multipletests
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    mask = ~np.isnan(p)
    if mask.sum() > 0:
        out[mask] = multipletests(p[mask], method="fdr_bh")[1]
    return out.tolist()


def add_fdr(df: pd.DataFrame, p_col: str = "p", out_col: str = "p_fdr") -> pd.DataFrame:
    if not df.empty and p_col in df:
        df[out_col] = bh_fdr(df[p_col].tolist())
    return df


# ---------------------------------------------------------------------------
# Derived signals
# ---------------------------------------------------------------------------

def gold_subscales(session) -> dict[str, float]:
    """OE/BA/VA from a session's gold CompACT-10 item array (reverse-scored)."""
    if len(session.gold_compact10) != 10:
        return {"OE": float("nan"), "BA": float("nan"), "VA": float("nan")}
    return compact10_subscale_scores(session.gold_compact10)


def headline_gap_series(cross_perspective: pd.DataFrame) -> pd.DataFrame:
    """Per (session, round) headline aggregate perspective gaps (wide)."""
    if cross_perspective.empty:
        return pd.DataFrame()
    agg = cross_perspective[cross_perspective["instrument"] == "AGG"]
    wide = agg.pivot_table(index=["session_id", "round"], columns="pair",
                           values="w1", aggfunc="first").reset_index()
    for name, pair in HEADLINE_PAIRS.items():
        wide[name] = wide[pair] if pair in wide else np.nan
    return wide


def tier_proportions(tom_tier: pd.DataFrame) -> pd.DataFrame:
    """Per-session proportion of rounds in each ToM tier."""
    if tom_tier.empty or "argmax" not in tom_tier:
        return pd.DataFrame()
    rows = []
    for sid, g in tom_tier.groupby("session_id"):
        n = len(g)
        vc = g["argmax"].value_counts()
        rows.append({"session_id": sid,
                     "prop_somatico": vc.get("somatico", 0) / n,
                     "prop_cognitivo": vc.get("cognitivo", 0) / n,
                     "prop_afectivo": vc.get("afectivo", 0) / n})
    return pd.DataFrame(rows)


def llama_compact_subscales(llama_assessors: pd.DataFrame) -> pd.DataFrame:
    """Per (session, round) OE/BA/VA from Llama-derived CompACT-10 item vectors."""
    if llama_assessors.empty:
        return pd.DataFrame()
    comp = llama_assessors[llama_assessors["instrument"] == "CompACT-10"]
    rows = []
    for (sid, rnd), g in comp.groupby(["session_id", "round"]):
        vec = g.sort_values("item")["score"].tolist()
        if len(vec) != 10:
            continue
        sub = compact10_subscale_scores(vec)
        rows.append({"session_id": sid, "round": rnd, **sub})
    return pd.DataFrame(rows)


def write_result(run_root: str | Path, name: str, df: pd.DataFrame) -> Path:
    out_dir = Path(run_root) / ANALYSIS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.parquet"
    df.to_parquet(path, index=False)
    logger.info("Wrote analysis result %s (%d rows)", path, len(df))
    return path
