"""Case-study trajectory figures (spec §8.6): S07 (minimal-PHQ) vs S09 (severe).

For each session: a trajectory plot with the 6 procesos_act lines (incl.
yo_como_contexto), the 3 Llama-derived CompACT-10 subscales (OE/BA/VA), and the
2 headline perspective gaps, on a shared round axis, with temporal-Wasserstein
rupture rounds overlaid. Figures are illustrative, not confirmatory.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..constants import ACT_PROCESS_KEYS  # noqa: E402
from . import common  # noqa: E402

logger = logging.getLogger(__name__)


def _rupture_rounds(temporal, sid) -> list[int]:
    if temporal.empty:
        return []
    t = temporal.copy()
    if "variant" in t:
        t = t[t["variant"] == "consecutive"]
    t = t[t["run"] == sid]
    fired = t[t["fired"].astype(str).str.lower().isin(["true", "1"])]
    return sorted(set(int(r) for r in fired["round"]))


def plot_session(run_root, sid, tables) -> Path | None:
    state = tables["llama_state"]
    ss = state[state["session_id"] == sid].sort_values("round") if not state.empty else None
    if ss is None or ss.empty:
        logger.warning("Case study %s: no state data; skipping", sid)
        return None

    subs = common.llama_compact_subscales(tables["llama_assessors"])
    subs = subs[subs["session_id"] == sid].sort_values("round") if not subs.empty else None
    gaps = common.headline_gap_series(tables["cross_perspective"])
    gaps = gaps[gaps["session_id"] == sid].sort_values("round") if not gaps.empty else None
    ruptures = _rupture_rounds(tables["temporal"], sid)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    ax = axes[0]
    for k in ACT_PROCESS_KEYS:
        if k in ss.columns:
            ax.plot(ss["round"], ss[k], label=k, linewidth=1.2)
    ax.set_ylabel("procesos_act (0–1)")
    ax.set_title(f"{sid} — ACT process trajectory")
    ax.legend(fontsize=7, ncol=3)

    ax = axes[1]
    if subs is not None and not subs.empty:
        for s in ("OE", "BA", "VA"):
            if s in subs.columns:
                ax.plot(subs["round"], subs[s], label=s, linewidth=1.4)
    ax.set_ylabel("CompACT-10 subscale (Llama)")
    ax.legend(fontsize=8)

    ax = axes[2]
    if gaps is not None and not gaps.empty:
        for g in ("gap_conservative", "gap_realistic"):
            if g in gaps.columns:
                ax.plot(gaps["round"], gaps[g], label=g, linewidth=1.4)
    ax.set_ylabel("perspective gap (W1)")
    ax.set_xlabel("round")
    ax.legend(fontsize=8)

    for ax in axes:
        for r in ruptures:
            ax.axvline(r, color="red", alpha=0.25, linestyle="--", linewidth=0.8)

    fig.tight_layout()
    out_dir = Path(run_root) / "outputs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"case_study_{sid}.png"
    fig.savefig(png, dpi=150)
    fig.savefig(out_dir / f"case_study_{sid}.pdf")
    plt.close(fig)
    logger.info("Wrote case-study figure %s (%d ruptures)", png, len(ruptures))
    return png


def run(run_root, sessions, cfg) -> dict:
    tables = common.load_tables(run_root)
    targets = cfg.get("case_study_sessions", ["S07", "S09"])
    made = [str(p) for s in targets if s in sessions and (p := plot_session(run_root, s, tables))]
    return {"case_studies": made}
