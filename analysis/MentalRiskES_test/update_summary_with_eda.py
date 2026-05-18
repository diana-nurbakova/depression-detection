"""Inject the test-data EDA section into analysis/MentalRiskES_test/SUMMARY.md.

Reads the Markdown produced by `eda_test_data.py` (which already starts
with a `## Test Data EDA` heading and the per-section sub-headings) and
either inserts it as a new section §0.7 before the existing §1, OR
replaces the previous marker-delimited block in place.

Run any time after eda_test_data.py:
  python analysis/MentalRiskES_test/update_summary_with_eda.py
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EDA_PATH = REPO_ROOT / "analysis/MentalRiskES_test/outputs/eda_test_data.md"
SUMMARY_PATH = REPO_ROOT / "analysis/MentalRiskES_test/SUMMARY.md"

BEGIN_MARK = "<!-- BEGIN AUTO-GENERATED TEST EDA -->"
END_MARK = "<!-- END AUTO-GENERATED TEST EDA -->"


def _adapt_eda(md: str) -> str:
    """Demote the EDA's H2 `## Test Data EDA` to `## 0.7 Test Data EDA`,
    H3 subsections become H3.1/H3.2 (using `### Task 1 test corpus` form
    as the inner level, kept as-is)."""
    lines = md.splitlines()
    out: list[str] = []
    for ln in lines:
        if ln.startswith("## Test Data EDA"):
            out.append("## 0.7 Test Data EDA")
        else:
            out.append(ln)
    return "\n".join(out).rstrip() + "\n"


def main() -> None:
    logging.basicConfig(level="INFO", format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    log = logging.getLogger("summary_updater")

    if not EDA_PATH.exists():
        raise FileNotFoundError(f"EDA file missing: {EDA_PATH}. Run eda_test_data.py first.")
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"SUMMARY missing: {SUMMARY_PATH}")

    eda_body = _adapt_eda(EDA_PATH.read_text(encoding="utf-8"))
    new_block = (
        f"{BEGIN_MARK}\n\n"
        f"{eda_body}\n"
        f"{END_MARK}"
    )

    summary = SUMMARY_PATH.read_text(encoding="utf-8")

    if BEGIN_MARK in summary and END_MARK in summary:
        pattern = re.compile(
            re.escape(BEGIN_MARK) + r".*?" + re.escape(END_MARK),
            re.DOTALL,
        )
        summary = pattern.sub(new_block, summary, count=1)
        log.info("Replaced existing test-EDA block in %s", SUMMARY_PATH)
    else:
        # Insert immediately before "## 1. Per-instrument leaderboard position"
        anchor = "## 1. Per-instrument leaderboard position"
        idx = summary.find(anchor)
        if idx == -1:
            raise ValueError(f"Anchor heading not found in SUMMARY.md: {anchor!r}")
        # Find the start of the line containing the anchor (look backwards for previous newline)
        line_start = summary.rfind("\n", 0, idx) + 1
        summary = (summary[:line_start].rstrip() + "\n\n"
                   + new_block + "\n\n---\n\n"
                   + summary[line_start:])
        log.info("Inserted new test-EDA block into %s before §1", SUMMARY_PATH)

    SUMMARY_PATH.write_text(summary, encoding="utf-8")
    log.info("Done: %s now contains the auto-generated test-EDA section", SUMMARY_PATH)


if __name__ == "__main__":
    main()
