"""CLI for MentalRiskES data preparation.

Entry point: `mentalriskes-dataprep <command>`
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("mentalriskes.data_prep")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging.")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """MentalRiskES 2026 Data Preparation: translate, load, simulate."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    _setup_logging(verbose)


# ---- ESConv commands ----

@cli.command()
@click.option("--input", "input_path", default="data/MentalRiskES-2026/ESConv/ESConv.json",
              help="Path to ESConv.json")
@click.option("--output", "output_dir", default="output/mentalriskes/data_prep",
              help="Output directory")
@click.option("--max-dialogues", default=None, type=int,
              help="Limit number of dialogues to translate")
@click.option("--emotions", default="anxiety,depression,sadness",
              help="Comma-separated emotion types to filter")
@click.option("--min-turns", default=10, type=int,
              help="Minimum turns per dialogue")
@click.pass_context
def translate_esconv(
    ctx: click.Context,
    input_path: str,
    output_dir: str,
    max_dialogues: int | None,
    emotions: str,
    min_turns: int,
) -> None:
    """Translate ESConv dialogues from English to Spanish via DeepL."""
    from .deepl_translator import DeepLTranslator
    from .esconv import (
        filter_therapeutic_dialogues,
        load_esconv,
        save_translated_dialogues,
        translate_dialogues,
    )

    # Load and filter
    dialogues = load_esconv(input_path)
    emotion_set = set(emotions.split(","))
    dialogues = filter_therapeutic_dialogues(dialogues, emotion_set, min_turns)

    if max_dialogues:
        dialogues = dialogues[:max_dialogues]

    click.echo(f"Translating {len(dialogues)} dialogues...")

    # Translate
    translator = DeepLTranslator(
        cache_dir=Path(output_dir) / "translation_cache",
    )

    # Show usage before
    try:
        usage = translator.get_usage()
        click.echo(f"DeepL usage: {usage.get('character_count', '?')}/{usage.get('character_limit', '?')} chars")
    except Exception as e:
        click.echo(f"Could not check DeepL usage: {e}")

    translated = translate_dialogues(dialogues, translator, max_dialogues)

    # Save
    out_path = Path(output_dir) / "esconv_translated.json"
    save_translated_dialogues(translated, out_path)

    stats = translator.stats()
    click.echo(f"Done. Translated {stats['chars_translated']} chars, "
               f"{stats['cache_hits']} cache hits, {stats['api_calls']} API calls")
    click.echo(f"Saved to {out_path}")


@cli.command()
@click.option("--input", "input_path", default="output/mentalriskes/data_prep/esconv_translated.json",
              help="Path to translated ESConv dialogues")
@click.option("--output", "output_path", default="output/mentalriskes/data_prep/esconv_mc.json",
              help="Output path for MC dataset")
@click.option("--n-instances", default=100, type=int,
              help="Number of MC instances to generate")
@click.option("--seed", default=42, type=int, help="Random seed")
@click.pass_context
def generate_mc(
    ctx: click.Context,
    input_path: str,
    output_path: str,
    n_instances: int,
    seed: int,
) -> None:
    """Generate synthetic MC dev set from translated ESConv dialogues."""
    from .esconv import (
        ESConvDialogue,
        ESConvTurn,
        generate_cross_dialogue_mc,
        save_mc_dataset,
    )

    # Load translated dialogues
    with open(input_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    dialogues = []
    for d in raw:
        turns = [
            ESConvTurn(
                speaker=t["speaker"],
                content=t["content_en"],
                content_es=t["content_es"],
                strategy=t.get("strategy", ""),
            )
            for t in d["turns"]
        ]
        dialogues.append(ESConvDialogue(
            dialogue_id=d["dialogue_id"],
            emotion_type=d["emotion_type"],
            problem_type=d["problem_type"],
            situation=d["situation_en"],
            situation_es=d.get("situation_es", ""),
            turns=turns,
        ))

    click.echo(f"Loaded {len(dialogues)} translated dialogues")

    instances = generate_cross_dialogue_mc(
        dialogues, n_instances=n_instances, use_spanish=True, seed=seed,
    )

    save_mc_dataset(instances, output_path)
    click.echo(f"Generated {len(instances)} MC instances → {output_path}")


# ---- MIDAS commands ----

@cli.command()
@click.option("--input", "input_path", default="data/MentalRiskES-2026/MIDAS/Spanish_MI.json",
              help="Path to Spanish_MI.json")
@click.option("--output", "output_dir", default="output/mentalriskes/data_prep",
              help="Output directory")
@click.pass_context
def extract_midas(ctx: click.Context, input_path: str, output_dir: str) -> None:
    """Extract counselor responses and dialogue segments from MIDAS."""
    from .midas import (
        extract_counselor_responses,
        extract_dialogue_segments,
        load_midas,
        save_counselor_responses,
    )

    sessions = load_midas(input_path)

    # Extract counselor responses
    responses = extract_counselor_responses(sessions, min_length=30, with_context=True)
    resp_path = Path(output_dir) / "midas_counselor_responses.json"
    save_counselor_responses(responses, resp_path)
    click.echo(f"Extracted {len(responses)} counselor responses → {resp_path}")

    # Extract dialogue segments
    segments = extract_dialogue_segments(sessions, segment_length=6, overlap=2)
    seg_path = Path(output_dir) / "midas_dialogue_segments.json"
    with open(seg_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    click.echo(f"Extracted {len(segments)} dialogue segments → {seg_path}")

    # Print stats
    click.echo(f"\nMIDAS Summary:")
    click.echo(f"  Sessions: {len(sessions)}")
    click.echo(f"  Total turns: {sum(len(s.turns) for s in sessions)}")
    click.echo(f"  Counselor responses (≥30 chars): {len(responses)}")
    click.echo(f"  Dialogue segments: {len(segments)}")

    # Tag distribution
    from collections import Counter
    tags = Counter()
    for r in responses:
        for t in r.get("tags", []):
            tags[t] += 1
    if tags:
        click.echo(f"  Tag distribution: {dict(tags)}")


# ---- HOPE commands ----

@cli.command()
@click.option("--repo-dir", default="data/MentalRiskES-2026/HOPE",
              help="Path to HOPE/READER repo")
@click.pass_context
def check_hope(ctx: click.Context, repo_dir: str) -> None:
    """Check HOPE dataset availability."""
    from .hope import check_hope_data

    status = check_hope_data(repo_dir)
    click.echo("HOPE data files:")
    for fname, exists in status.items():
        icon = "OK" if exists else "MISSING"
        click.echo(f"  {fname}: {icon}")

    if not any(status.values()):
        click.echo("\nHOPE CSVs not included in the repo. Dataset may need separate download.")


# ---- Simulator commands ----

@cli.command()
@click.option("--output", "output_dir", default="output/mentalriskes/data_prep/simulated",
              help="Output directory for simulated sessions")
@click.option("--profiles", default=None,
              help="Comma-separated profile IDs (default: all)")
@click.option("--rounds", default=15, type=int,
              help="Number of rounds per session")
@click.option("--generate-mc/--no-mc", default=True,
              help="Generate Task 2 MC options")
@click.option("--seed", default=42, type=int, help="Random seed")
@click.option("--config", "config_path", default="config/mentalriskes.yaml",
              help="Config file for LLM settings")
@click.pass_context
def simulate(
    ctx: click.Context,
    output_dir: str,
    profiles: str | None,
    rounds: int,
    generate_mc: bool,
    seed: int,
    config_path: str,
) -> None:
    """Generate TalkDep-style simulated therapeutic conversations."""
    from ..config import load_config
    from ..llm_client import LLMClient
    from .simulator import (
        PATIENT_PROFILES,
        generate_session,
        save_session_task1,
        save_session_task2,
    )

    config = load_config(config_path)
    client = LLMClient.from_config(config.simulation_llm)
    click.echo(f"LLM: {config.simulation_llm.provider} / {config.simulation_llm.model}")

    # Select profiles
    if profiles:
        profile_ids = set(profiles.split(","))
        selected = [p for p in PATIENT_PROFILES if p["id"] in profile_ids]
        if not selected:
            click.echo(f"No profiles matched: {profiles}", err=True)
            click.echo(f"Available: {[p['id'] for p in PATIENT_PROFILES]}", err=True)
            sys.exit(1)
    else:
        selected = PATIENT_PROFILES

    click.echo(f"Simulating {len(selected)} sessions ({rounds} rounds each)")

    out_dir = Path(output_dir)
    for i, profile in enumerate(selected):
        click.echo(f"\n[{i+1}/{len(selected)}] Profile: {profile['id']} — {profile['description']}")

        session = generate_session(
            profile=profile,
            llm_client=client,
            n_rounds=rounds,
            generate_mc=generate_mc,
            seed=seed + i,
        )

        # Save Task 1 format
        save_session_task1(session, out_dir / "task1")
        click.echo(f"  Task 1: {len(session.turns)} rounds saved")

        # Save Task 2 format
        if generate_mc:
            save_session_task2(session, out_dir / "task2")
            mc_rounds = sum(1 for t in session.turns if t.correct_option > 0)
            click.echo(f"  Task 2: {mc_rounds} MC rounds saved")

    click.echo(f"\nAll sessions saved to {out_dir}")


@cli.command()
@click.pass_context
def info(ctx: click.Context) -> None:
    """Show data preparation status and available resources."""
    from .hope import check_hope_data

    click.echo("MentalRiskES 2026 Data Preparation Status")
    click.echo("=" * 50)

    # Check ESConv
    esconv_path = Path("data/MentalRiskES-2026/ESConv/ESConv.json")
    click.echo(f"\nESConv: {'OK' if esconv_path.exists() else 'MISSING'}")
    if esconv_path.exists():
        with open(esconv_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        click.echo(f"  Dialogues: {len(data)}")

    # Check translated ESConv
    trans_path = Path("output/mentalriskes/data_prep/esconv_translated.json")
    click.echo(f"  Translated: {'OK' if trans_path.exists() else 'NOT YET'}")

    # Check MC dataset
    mc_path = Path("output/mentalriskes/data_prep/esconv_mc.json")
    click.echo(f"  MC dataset: {'OK' if mc_path.exists() else 'NOT YET'}")

    # Check MIDAS
    midas_path = Path("data/MentalRiskES-2026/MIDAS/Spanish_MI.json")
    click.echo(f"\nMIDAS: {'OK' if midas_path.exists() else 'MISSING'}")
    if midas_path.exists():
        with open(midas_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        click.echo(f"  Sessions: {len(data)}")

    # Check MIDAS extractions
    resp_path = Path("output/mentalriskes/data_prep/midas_counselor_responses.json")
    click.echo(f"  Responses extracted: {'OK' if resp_path.exists() else 'NOT YET'}")

    # Check HOPE
    click.echo(f"\nHOPE:")
    hope_status = check_hope_data("data/MentalRiskES-2026/HOPE")
    for fname, exists in hope_status.items():
        click.echo(f"  {fname}: {'OK' if exists else 'MISSING'}")

    # Check simulated data
    sim_dir = Path("output/mentalriskes/data_prep/simulated")
    if sim_dir.exists():
        task1_dirs = list((sim_dir / "task1").glob("sim_*")) if (sim_dir / "task1").exists() else []
        task2_dirs = list((sim_dir / "task2").glob("sim_*")) if (sim_dir / "task2").exists() else []
        click.echo(f"\nSimulated sessions:")
        click.echo(f"  Task 1: {len(task1_dirs)} sessions")
        click.echo(f"  Task 2: {len(task2_dirs)} sessions")
    else:
        click.echo(f"\nSimulated sessions: NOT YET")

    # DeepL usage
    click.echo(f"\nDeepL:")
    try:
        from .deepl_translator import DeepLTranslator
        translator = DeepLTranslator()
        usage = translator.get_usage()
        click.echo(f"  Character usage: {usage.get('character_count', '?')}/{usage.get('character_limit', '?')}")
    except Exception as e:
        click.echo(f"  Could not check: {e}")


@cli.command("run-all")
@click.option("--output", "output_dir", default="output/mentalriskes/data_prep",
              help="Output directory")
@click.option("--max-esconv", default=50, type=int,
              help="Max ESConv dialogues to translate")
@click.option("--n-mc", default=100, type=int,
              help="Number of MC instances")
@click.option("--skip-simulation", is_flag=True,
              help="Skip LLM-based simulation (requires Ollama)")
@click.option("--seed", default=42, type=int)
@click.pass_context
def run_all(
    ctx: click.Context,
    output_dir: str,
    max_esconv: int,
    n_mc: int,
    skip_simulation: bool,
    seed: int,
) -> None:
    """Run the full data preparation pipeline."""
    out = Path(output_dir)

    # Step 1: MIDAS extraction
    click.echo("\n=== Step 1: MIDAS extraction ===")
    ctx.invoke(extract_midas, output_dir=output_dir)

    # Step 2: ESConv translation
    click.echo("\n=== Step 2: ESConv translation ===")
    ctx.invoke(translate_esconv, output_dir=output_dir, max_dialogues=max_esconv)

    # Step 3: MC generation
    click.echo("\n=== Step 3: MC generation ===")
    ctx.invoke(
        generate_mc,
        input_path=str(out / "esconv_translated.json"),
        output_path=str(out / "esconv_mc.json"),
        n_instances=n_mc,
        seed=seed,
    )

    # Step 4: HOPE check
    click.echo("\n=== Step 4: HOPE data check ===")
    ctx.invoke(check_hope)

    # Step 5: Simulation (optional)
    if not skip_simulation:
        click.echo("\n=== Step 5: Conversation simulation ===")
        ctx.invoke(simulate, output_dir=str(out / "simulated"), seed=seed)
    else:
        click.echo("\n=== Step 5: Simulation skipped ===")

    # Final status
    click.echo("\n=== Data Preparation Complete ===")
    ctx.invoke(info)
