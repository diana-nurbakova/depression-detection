"""CLI for the ToM × ACT explanatory analysis (spec implementation).

Chainable subcommands let a single invocation run several passes in order, e.g.

    uv run mentalriskes-tom-act --config config/tom_act.yaml --dry-run \
        --limit-sessions S07 --limit-rounds 3 regen-llama gen-gemma wasserstein \
        aggregate analyze --rq all case-studies micro-val

``--dry-run`` redirects all output to ``<root>_dryrun`` so a validation run never
touches the real run's logs, and defaults to S07 / first 3 rounds if no limits
are given.
"""

from __future__ import annotations

import logging

import click
import yaml

from ..config import LLMConfig
from ..llm_client import create_llm_client
from . import aggregator, gemma_signals, llama_regen, reparse as reparse_mod, wasserstein
from .analysis import case_studies, micro_validation, rq1, rq2, rq3, rq4, rq5
from .data import load_sessions
from .dispatcher import Dispatcher
from .tiers import TIERS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("tom_act")

_RQ = {"1": rq1, "2": rq2, "3": rq3, "4": rq4, "5": rq5}


def _build_client(block: dict):
    cfg = LLMConfig(
        provider=block["provider"],
        model=block["model"],
        temperature=block.get("temperature", 0.0),
        max_tokens=block.get("max_tokens", 2048),
        timeout=block.get("timeout", 180),
    )
    client = create_llm_client(cfg)
    return client, block["model"], block["provider"]


@click.group(chain=True)
@click.option("--config", "config_path", default="config/tom_act.yaml", show_default=True)
@click.option("--tier", type=click.Choice(list(TIERS)), default=None,
              help="Priority tier (v0.6 §15): T0 pilot, T1, T2, T3. Gates signals/sessions/candidates.")
@click.option("--dry-run", is_flag=True, help="Redirect to <root>_dryrun; default S07/3 rounds.")
@click.option("--limit-sessions", default=None, help="Comma-separated session ids (overrides tier).")
@click.option("--limit-rounds", type=int, default=None, help="Cap rounds per session (overrides tier).")
@click.pass_context
def cli(ctx, config_path, tier, dry_run, limit_sessions, limit_rounds):
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    tier_obj = TIERS[tier] if tier else None
    # Tier-derived defaults (explicit --limit-* and --dry-run override).
    if tier_obj is not None:
        if limit_sessions is None and tier_obj.sessions is not None:
            limit_sessions = ",".join(tier_obj.sessions)
        if limit_rounds is None and tier_obj.max_rounds is not None:
            limit_rounds = tier_obj.max_rounds
        logger.info("TIER %s: %s", tier_obj.name, tier_obj.description)

    if dry_run:
        limit_sessions = limit_sessions or "S07"
        limit_rounds = limit_rounds if limit_rounds is not None else 3
        cfg["run"]["root"] = cfg["run"]["root"].rstrip("/") + "_dryrun"
        logger.info("DRY-RUN: root=%s sessions=%s rounds<=%s",
                    cfg["run"]["root"], limit_sessions, limit_rounds)

    sess_filter = [s.strip() for s in limit_sessions.split(",")] if limit_sessions else \
        cfg["data"].get("sessions")
    sessions = load_sessions(
        cfg["data"]["task1_dir"], cfg["data"]["task2_dir"], cfg["data"]["gold_dir"],
        cfg["data"]["session_gold"], sess_filter,
    )

    ctx.obj = {
        "cfg": cfg,
        "root": cfg["run"]["root"],
        "sessions": sessions,
        "limit_rounds": limit_rounds,
        "tier": tier_obj,
        # Per-tier meta file when --tier is set, so concurrent tier passes never
        # share a writer on meta.jsonl. Non-tier commands keep the default file.
        "dispatcher": Dispatcher(
            cfg["run"]["root"],
            cfg["run"].get("max_attempts", 3),
            meta_suffix=tier_obj.name if tier_obj else None,
        ),
    }


@cli.command("regen-llama")
@click.pass_context
def regen_llama(ctx):
    """Llama regeneration pass: procesos_act + assessor item-vectors per round."""
    o = ctx.obj
    tier = o["tier"]
    if tier is not None and not tier.run_llama:
        logger.info("Tier %s does not include the Llama regeneration; skipping.", tier.name)
        return
    client, model_id, provider = _build_client(o["cfg"]["llama"])
    lang = o["cfg"]["llama"].get("lang", "es")
    assessor_mode = o["cfg"]["llama"].get("assessor_mode", "combined")
    for sid, sess in o["sessions"].items():
        logger.info("=== Llama regen: %s (%d rounds) assessor_mode=%s ===",
                    sid, sess.n_rounds, assessor_mode)
        llama_regen.regenerate_session(
            o["dispatcher"], client, sess, model_id, provider,
            lang=lang, limit_rounds=o["limit_rounds"], assessor_mode=assessor_mode,
        )
    o["dispatcher"].meta("tier_pass_complete", tier=tier.name if tier else None,
                         pass_="regen-llama", sessions=list(o["sessions"]))


@cli.command("gen-gemma")
@click.pass_context
def gen_gemma(ctx):
    """Gemma signal-generation pass: views, tier, stance, presencia per round."""
    o = ctx.obj
    tier = o["tier"]
    signals = set(tier.gemma_signals) if tier is not None else None
    candidate_filter = tier.candidate_filter if tier is not None else "all"
    client, model_id, provider = _build_client(o["cfg"]["gemma"])
    for sid, sess in o["sessions"].items():
        logger.info("=== Gemma signals: %s (%d rounds) signals=%s cand=%s ===",
                    sid, sess.n_rounds, sorted(signals) if signals else "all", candidate_filter)
        gemma_signals.generate_session(
            o["dispatcher"], client, sess, model_id, provider,
            limit_rounds=o["limit_rounds"], signals=signals,
            candidate_filter=candidate_filter,
        )
    o["dispatcher"].meta("tier_pass_complete", tier=tier.name if tier else None,
                         pass_="gen-gemma",
                         signals=sorted(signals) if signals else "all",
                         candidate_filter=candidate_filter, sessions=list(o["sessions"]))


@cli.command("wasserstein")
@click.pass_context
def wasserstein_cmd(ctx):
    """Compute cross-perspective + temporal (both variants) Wasserstein from logs."""
    import pandas as pd
    from pathlib import Path

    o = ctx.obj
    root = Path(o["root"])
    wcfg = o["cfg"].get("wasserstein", {})
    variants = wcfg.get("temporal_variants", ["consecutive", "barycenter"])
    sigma = wcfg.get("alert_sigma", 2.0)

    assessor_long = aggregator.build_llama_assessors_long(root)
    view_long = aggregator.build_gemma_views_long(root)

    temporal = wasserstein.temporal_from_tables(assessor_long, variants, sigma) \
        if not assessor_long.empty else pd.DataFrame()
    xpersp = wasserstein.cross_perspective_from_tables(view_long) \
        if not view_long.empty else pd.DataFrame()

    (root / "outputs" / "wasserstein_test").mkdir(parents=True, exist_ok=True)
    (root / "outputs" / "cross_perspective").mkdir(parents=True, exist_ok=True)
    temporal.to_csv(root / "outputs" / "wasserstein_test" / "temporal.csv",
                    index=False, na_rep="")
    if not temporal.empty:
        temporal.to_parquet(root / "outputs" / "wasserstein_test" / "temporal.parquet", index=False)
    if not xpersp.empty:
        xpersp.to_parquet(root / "outputs" / "cross_perspective" / "gaps.parquet", index=False)
    logger.info("Wasserstein: %d temporal rows, %d cross-perspective rows",
                len(temporal), len(xpersp))


@cli.command("reparse")
@click.option("--signals", default=None,
              help="Comma-separated signal types to scope reparse to (e.g. self_b,observer_pt).")
@click.pass_context
def reparse_cmd(ctx, signals):
    """Re-parse previously-failed JSONL lines with updated recovery logic (no LLM calls)."""
    sigs = [s.strip() for s in signals.split(",")] if signals else None
    summary = reparse_mod.reparse(ctx.obj["root"], sigs)
    total = sum(summary.values())
    logger.info("reparse complete: recovered %d call(s) total: %s",
                total, {k: v for k, v in summary.items() if v})


@cli.command("aggregate")
@click.pass_context
def aggregate_cmd(ctx):
    """Aggregate JSONL logs into tidy parquet tables (+ recovery report)."""
    aggregator.aggregate_all(ctx.obj["root"])


@cli.command("analyze")
@click.option("--rq", default="all", help="1|2|3|4|5|all")
@click.pass_context
def analyze_cmd(ctx, rq):
    """Run RQ analyses against the aggregated tables + Wasserstein outputs."""
    o = ctx.obj
    acfg = o["cfg"].get("analysis", {})
    targets = list(_RQ) if rq == "all" else [rq]
    for k in targets:
        summary = _RQ[k].run(o["root"], o["sessions"], acfg)
        logger.info("RQ%s summary: %s", k, summary)


@cli.command("case-studies")
@click.pass_context
def case_studies_cmd(ctx):
    """Generate S07 vs S09 trajectory figures."""
    o = ctx.obj
    logger.info("Case studies: %s",
                case_studies.run(o["root"], o["sessions"], o["cfg"].get("analysis", {})))


@cli.command("micro-val")
@click.pass_context
def micro_val_cmd(ctx):
    """Sample 10 patient turns (one per session) for manual ToM-tier coding."""
    o = ctx.obj
    logger.info("Micro-validation: %s",
                micro_validation.run(o["root"], o["sessions"], o["cfg"].get("analysis", {})))


@cli.command("info")
@click.pass_context
def info_cmd(ctx):
    """Print loaded session/round counts and config summary."""
    o = ctx.obj
    total = sum(s.n_rounds for s in o["sessions"].values())
    click.echo(f"root: {o['root']}")
    click.echo(f"sessions: {len(o['sessions'])}  total rounds: {total}")
    for sid, s in o["sessions"].items():
        click.echo(f"  {sid}: {s.n_rounds} rounds  "
                   f"PHQ-9={sum(s.gold_phq9)} GAD-7={sum(s.gold_gad7)} "
                   f"CompACT-10={sum(s.gold_compact10)}")


if __name__ == "__main__":
    cli()
