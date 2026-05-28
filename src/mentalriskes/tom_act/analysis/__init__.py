"""Per-RQ analysis scripts (spec §8).

Each ``rq{n}.run(run_root, sessions, cfg)`` loads the aggregated parquet tables
+ Wasserstein outputs + gold, computes the analysis, writes a result table under
``outputs/analysis/``, and returns a short summary dict. All analyses are
exploratory; effect sizes + 95% CIs are primary, FDR-corrected p-values
secondary. Functions degrade gracefully on insufficient data (return empty /
NaN rather than raise) so they are smoke-testable on a dry-run.
"""
