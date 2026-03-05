#!/usr/bin/env python3
"""Run the Task 1 ablation study against TalkDep conversations.

Usage:
  # Run all ablation configs on all personas
  uv run python scripts/run_ablation.py

  # Run specific configs
  uv run python scripts/run_ablation.py --configs A0,A1,A4

  # Run on specific personas (boundary cases)
  uv run python scripts/run_ablation.py --personas Noah,Alex,Linda,Maria

  # Quick smoke test (1 config, 2 personas)
  uv run python scripts/run_ablation.py --configs A1 --personas Noah,Maria --output runs/ablation_test
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from erisk_task1.ablation import ABLATION_CONFIGS, run_ablation, run_full_ablation_study
from erisk_task1.config import load_config
from erisk_task1.evaluation import (
    compute_component_contribution,
    format_comparison_table,
    format_error_analysis,
    load_talkdep_conversations,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run Task 1 ablation study against TalkDep"
    )
    parser.add_argument(
        "--configs",
        default=None,
        help="Comma-separated ablation configs to run (default: all A0-A7)",
    )
    parser.add_argument(
        "--personas",
        default=None,
        help="Comma-separated persona names (default: all 12)",
    )
    parser.add_argument(
        "--talkdep-dir",
        default="data/TalkDep",
        help="Path to TalkDep repo (default: data/TalkDep)",
    )
    parser.add_argument(
        "--config-file",
        default="config/task1.yaml",
        help="Pipeline config YAML (default: config/task1.yaml)",
    )
    parser.add_argument(
        "--output",
        default="runs/ablation",
        help="Output directory (default: runs/ablation)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Parse arguments
    configs = args.configs.split(",") if args.configs else None
    personas = args.personas.split(",") if args.personas else None
    output_dir = Path(args.output)

    # Load pipeline config
    pipeline_cfg = load_config(args.config_file)

    # List available configs
    if configs:
        print(f"Running ablation configs: {', '.join(configs)}")
        for c in configs:
            if c not in ABLATION_CONFIGS:
                print(f"ERROR: Unknown config '{c}'. Available: {list(ABLATION_CONFIGS.keys())}")
                sys.exit(1)
    else:
        print(f"Running all ablation configs: {list(ABLATION_CONFIGS.keys())}")

    if personas:
        print(f"Personas: {', '.join(personas)}")
    else:
        print("Personas: all 12 TalkDep")

    print(f"Output: {output_dir}")
    print(f"Model: {pipeline_cfg.assessor.model} via {pipeline_cfg.assessor.provider}")
    print()

    # Run ablation study
    results = run_full_ablation_study(
        pipeline_cfg=pipeline_cfg,
        talkdep_dir=args.talkdep_dir,
        configs=configs,
        personas=personas,
        output_dir=str(output_dir),
    )

    if not results:
        print("No results produced.")
        sys.exit(1)

    # Print comparison table
    print("\n" + "=" * 70)
    print("ABLATION STUDY RESULTS")
    print("=" * 70)
    print()
    print(format_comparison_table(results))

    # Print error analysis for each config
    print()
    for r in results:
        print(format_error_analysis(r))
        print()

    # Print component contributions (consecutive pairs)
    if len(results) > 1:
        print("=" * 70)
        print("COMPONENT CONTRIBUTIONS")
        print("=" * 70)
        for i in range(len(results) - 1):
            contrib = compute_component_contribution(results[i], results[i + 1])
            print(
                f"\n{contrib['baseline']} -> {contrib['enhanced']}:"
                f"\n  DCHR: {contrib['dchr_delta']:+.1%}"
                f"  MAD: {contrib['mad_delta']:+.1f}"
                f"  ADODL: {contrib['adodl_delta']:+.3f}"
                f"  ASHR: {contrib['ashr_delta']:+.1%}"
                f"\n  Improved: {contrib['improved']}/{contrib['improved']+contrib['worsened']+contrib['unchanged']}"
                f"  Worsened: {contrib['worsened']}"
            )
            if contrib["boundary_fixes"]:
                print(f"  Boundary fixes: {', '.join(contrib['boundary_fixes'])}")
            if contrib["boundary_breaks"]:
                print(f"  Boundary breaks: {', '.join(contrib['boundary_breaks'])}")

    # Save final summary
    summary_file = output_dir / "ablation_summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, "w") as f:
        json.dump([r.summary() for r in results], f, indent=2)
    print(f"\nFull results saved to {summary_file}")


if __name__ == "__main__":
    main()
