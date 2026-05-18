# MentalRiskES 2026 — Reporting Inventory

**Purpose:** single index of every file we need to write the working-notes paper(s) for both MentalRiskES 2026 challenge tracks (Task 1 = psychometric scoring; Task 2 = therapist response selection).

**Status legend**

- ✅ paper-ready — citable as-is, with at most light editorial editing
- 🟡 needs update — content exists but lags the latest results
- ⚠ gap — nothing exists yet, should be produced before the paper
- 🛠 working artefact — useful supporting evidence but not narrative-ready

---

## 1. Detailed system description

| Task | File | Status | Notes |
|---|---|---|---|
| Task 1 | [docs/mentalriskes_task1_solution_description.md](mentalriskes_task1_solution_description.md) | ✅ | Submitted Llama-3.3-70B three-tier (A/B/C) calibrated pipeline + T0/T2/T3 temporal aggregation + ablation, plus **§7 Post-Submission Findings** added 2026-05-12: truncation disclosure (§7.1), full-replay results (§7.2), per-instrument ranking (§7.3), Gemma GAD-7 post-hoc (§7.4), hybrid combined → rank 4 (§7.5), cross-cohort lesson (§7.6), summary (§7.7). |
| Task 2 | [docs/mentalriskes_task2_solution_description.md](mentalriskes_task2_solution_description.md) | ✅ | Submitted state-tracking ACT-FM pipeline + B/B+/ENS + FUNC/HYB/TOM-B/TOM-C + v2.0 prompt updates, plus **§9 Post-Submission Findings** added 2026-05-12: truncation (§9.1), full-replay (§9.2), bare-LLM S (§9.3) / S2 (§9.4, 0.470 headline), R2/S3/S4 (§9.5), consensus failure (§9.6), cross-cohort lesson (§9.7), disagreement appendix pointer (§9.8), summary (§9.9). |
| Disambiguation | [docs/task1_solution_description.md](task1_solution_description.md) + [docs/task2_solution_description.md](task2_solution_description.md) | ℹ️ note | **These are for eRisk 2026, NOT MentalRiskES.** Do not confuse them with the files above. |

---

## 2. Additional data used in the system

Source material the submitted pipelines depend on beyond the challenge release.

### 2.1 External datasets

| Dataset | Used by | Path / reference | Citation |
|---|---|---|---|
| PRIMATE | Task 1 PHQ-9 few-shot examples | [data/MentalRiskES-2026/primate_dataset.json](../data/MentalRiskES-2026/primate_dataset.json) (also [specs/MentalRiskES/extract_primate_examples.py](../specs/MentalRiskES/extract_primate_examples.py)) | Gupta et al. 2022, CLPsych @ NAACL |
| DAIC-WOZ | Task 1 PHQ-9 conversational examples (cited only) | — | Gratch et al. 2014, LREC |
| ESConv (translated to ES) | Task 1 calibration sentences | [output/mentalriskes/data_prep/esconv_translated.json](../output/mentalriskes/data_prep/esconv_translated.json), [esconv_mc.json](../output/mentalriskes/data_prep/esconv_mc.json) | Liu et al. 2021 |
| MIDAS | Task 1 calibration dialogue segments | [output/mentalriskes/data_prep/midas_dialogue_segments.json](../output/mentalriskes/data_prep/midas_dialogue_segments.json), [midas_counselor_responses.json](../output/mentalriskes/data_prep/midas_counselor_responses.json) | Welivita et al. 2020 |
| ACT-FM | Task 2 selection rubric (25 items, 4 areas) | [data/MentalRiskES-2026/ACT-FM/](../data/MentalRiskES-2026/ACT-FM/) | O'Neill et al. 2019 |
| Psychometric literature | Task 1 calibration rules (Levels A/B/C) | Reference list in [mentalriskes_task1_solution_description.md §3.3](mentalriskes_task1_solution_description.md) | Francis 2016, Ryan 2022, Kroenke 2016, etc. |

### 2.2 Static prompt assets and calibration artefacts

| File | Purpose |
|---|---|
| [specs/MentalRiskES/assessor_prompts_v2.py](../specs/MentalRiskES/assessor_prompts_v2.py) | Task 1 chain-of-thought prompt templates for the three assessors |
| [specs/MentalRiskES/verbalizer_update_v2_1.py](../specs/MentalRiskES/verbalizer_update_v2_1.py) | Task 1 level-1/2/3 verbalizer system |
| [specs/MentalRiskES/act_vocabulary_es.json](../specs/MentalRiskES/act_vocabulary_es.json) | Task 2 ACT-FM Spanish-tag vocabulary (consistency + inconsistency tags) |
| [specs/MentalRiskES/calibration_config.json](../specs/MentalRiskES/calibration_config.json) | Task 1 anchor + rule constants |
| [specs/MentalRiskES/gad7_severe_examples.py](../specs/MentalRiskES/gad7_severe_examples.py) | Task 1 GAD-7 severe-anchor few-shot |
| [specs/MentalRiskES/hexaflex_quotes_fallback.json](../specs/MentalRiskES/hexaflex_quotes_fallback.json) | Task 2 hexaflex quote fallbacks |
| [specs/MentalRiskES/fallback_resources.py](../specs/MentalRiskES/fallback_resources.py) | Task 2 fallback assets when no LLM available |

### 2.3 LLM implementation configuration (provider, model, sampling parameters)

| File | Purpose |
|---|---|
| [docs/mentalriskes_llm_configuration.md](mentalriskes_llm_configuration.md) | ✅ paper-ready appendix — provider / model checkpoint / temperature / top-p / max_tokens / stop / seed / retry policy / rate-limit / streaming / JSON-mode / timeout for every system: submitted Task 1, submitted Task 2, full-test replay, Gemma GAD-7 post-hoc (v1+v2 × 3 Gemma models), Task 2 bare-LLM (S/S2/R2/S3/S4 × 3 models), pre-submission ablation (TogetherAI Turbo), simulator. Includes a quick-lookup table mapping each output directory to its provider and model. |

### 2.3 Challenge release (for completeness)

| File | Contents |
|---|---|
| [data/MentalRiskES-2026/task1_trial/data/](../data/MentalRiskES-2026/task1_trial/data/) | 19 trial rounds, single session, single-key `"trial"` schema |
| [data/MentalRiskES-2026/task2_trial/data/](../data/MentalRiskES-2026/task2_trial/data/) | 19 trial rounds, single session, candidate options included |
| [data/MentalRiskES-2026/test/task1/test/](../data/MentalRiskES-2026/test/task1/test/) | 82 test rounds, 10 patient sessions, + `gold_label.json` |
| [data/MentalRiskES-2026/test/task2/test/](../data/MentalRiskES-2026/test/task2/test/) | 82 test rounds with options, + per-round `round_X_gold.json` |
| [data/MentalRiskES-2026/MentalRiskES2026 - Results.xlsx](../data/MentalRiskES-2026/MentalRiskES2026%20-%20Results.xlsx) | Official leaderboard |

---

## 3. EDA of the challenge data (trial)

| File | Cohort | Status | Notes |
|---|---|---|---|
| [output/mentalriskes/trial_calibration_report.md](../output/mentalriskes/trial_calibration_report.md) | Task 1 trial | ✅ | Per-round trajectories, item-level bias, parse reliability, model-size comparison (Llama 70B vs Mistral 7B), recommendations. **Now also contains the auto-generated §0 Trial Data EDA block** (statistics on both tasks + simulated cohort) inserted between marker comments by the updater script below. |
| [analysis/MentalRiskES_trial/outputs/eda_trial_data.md](../analysis/MentalRiskES_trial/outputs/eda_trial_data.md) | Task 1 trial + Task 2 trial + simulated | ✅ | Standalone EDA Markdown: per-round word counts, T1 gold table, T2 gold class distribution + per-phase breakdown, simulated persona inventory (T1 + T2). |
| [analysis/MentalRiskES_trial/outputs/trial_t1_round_stats.csv](../analysis/MentalRiskES_trial/outputs/trial_t1_round_stats.csv) | Task 1 trial | 🛠 | Per-round word counts (patient + therapist). |
| [analysis/MentalRiskES_trial/outputs/trial_t2_option_stats.csv](../analysis/MentalRiskES_trial/outputs/trial_t2_option_stats.csv) | Task 2 trial | 🛠 | Per-round option lengths, phase tag, gold class. |
| [analysis/MentalRiskES_trial/outputs/simulated_t1_personas.csv](../analysis/MentalRiskES_trial/outputs/simulated_t1_personas.csv) | Simulated T1 | 🛠 | 6 personas × (presentation, target PHQ-9 / GAD-7, CompACT profile, n_rounds). |
| [analysis/MentalRiskES_trial/outputs/simulated_t2_sessions.csv](../analysis/MentalRiskES_trial/outputs/simulated_t2_sessions.csv) | Simulated T2 | 🛠 | 7 sessions × (n_rounds, gold class counts and percentages). |
| [analysis/MentalRiskES_trial/eda_trial_data.py](../analysis/MentalRiskES_trial/eda_trial_data.py) | generator | 🛠 script | Produces all five files above. Idempotent — re-run after any data changes. |
| [analysis/MentalRiskES_trial/update_trial_calibration_report.py](../analysis/MentalRiskES_trial/update_trial_calibration_report.py) | injector | 🛠 script | Inserts/replaces the marker-delimited EDA block in `trial_calibration_report.md`. Run after `eda_trial_data.py`. |
| [output/mentalriskes/data_prep/](../output/mentalriskes/data_prep/) — esconv_mc.json, esconv_translated.json, midas_dialogue_segments.json, primate-extracted JSON | External calibration data | 🛠 | Inventory of what was used to build the prompt anchors and verbalizer system. |
| [data/MentalRiskES-2026/task1_trial/data/](../data/MentalRiskES-2026/task1_trial/data/), [data/MentalRiskES-2026/task2_trial/data/](../data/MentalRiskES-2026/task2_trial/data/) | raw trial data | raw | Counts and gold are in the auto-generated EDA Markdown above. |

---

## 4. Synthetic data used for ablation

### 4.1 Description / generation pipeline

| File | Status | Notes |
|---|---|---|
| Task 1 doc §4 (Simulated Data for Development) | ✅ | Persona table, generation pipeline, intended use |
| Task 2 doc §2.2 (Simulated Data) | ✅ | Persona table, distractor types, gold-label generation |
| [src/mentalriskes/data_prep/simulator.py](../src/mentalriskes/data_prep/simulator.py) | 🛠 source | Persona generation simulator |
| [src/mentalriskes/data_prep/cli.py](../src/mentalriskes/data_prep/cli.py) | 🛠 source | CLI for generating simulated sessions |

### 4.2 Data artefacts

| File | Contents |
|---|---|
| [output/mentalriskes/data_prep/simulated/task1/](../output/mentalriskes/data_prep/simulated/task1/) | 6 personas × 15 rounds, metadata.json has `target_scores` (PHQ-9, GAD-7 totals; CompACT profile label) |
| [output/mentalriskes/data_prep/simulated/task2/](../output/mentalriskes/data_prep/simulated/task2/) | 7 personas × 14 rounds, labels.json with per-round gold option |

Total: 6 personas (T1, n=90 rounds) + 7 personas (T2, n=87 rounds). Persona inventory tables in both solution docs are current.

---

## 5. EDA of the released test data (post-submission)

| File | Cohort | Status | Notes |
|---|---|---|---|
| [analysis/MentalRiskES_test/SUMMARY.md §0.7](../analysis/MentalRiskES_test/SUMMARY.md) | Test cohort EDA | ✅ | **Auto-generated §0.7 Test Data EDA block** in SUMMARY.md between marker comments. Per-session round/word counts, item-level gold totals, PHQ-9 + GAD-7 band distribution, mismatch with `gold_label.json` (10 in test data vs 17 listed in gold). Heavily moderate-to-severe cohort: 7 of 10 sessions in the GAD-7 severe band. |
| [analysis/MentalRiskES_test/outputs/eda_test_data.md](../analysis/MentalRiskES_test/outputs/eda_test_data.md) | Test T1 + T2 | ✅ | Standalone version of the same Markdown (mirror of what's injected into SUMMARY.md). |
| [analysis/MentalRiskES_test/outputs/test_t1_session_stats.csv](../analysis/MentalRiskES_test/outputs/test_t1_session_stats.csv) | Test T1 | 🛠 | One row per session: round count, word totals, gold totals + severity bands per instrument. |
| [analysis/MentalRiskES_test/outputs/test_t1_band_distribution.csv](../analysis/MentalRiskES_test/outputs/test_t1_band_distribution.csv) | Test T1 | 🛠 | PHQ-9 + GAD-7 band counts across the 10 sessions. |
| [analysis/MentalRiskES_test/outputs/test_t2_session_stats.csv](../analysis/MentalRiskES_test/outputs/test_t2_session_stats.csv) | Test T2 | 🛠 | Per-session round count + gold class distribution + mean option / patient word counts. |
| [analysis/MentalRiskES_test/outputs/test_t2_round_stats.csv](../analysis/MentalRiskES_test/outputs/test_t2_round_stats.csv) | Test T2 | 🛠 | Per-round option lengths, gold class, tercile bucket. |
| [analysis/MentalRiskES_test/outputs/test_t2_class_by_tercile.csv](../analysis/MentalRiskES_test/outputs/test_t2_class_by_tercile.csv) | Test T2 | 🛠 | Gold class counts by early / mid / late round tercile. |
| [analysis/MentalRiskES_test/eda_test_data.py](../analysis/MentalRiskES_test/eda_test_data.py) | generator | 🛠 script | Produces all six files above. Idempotent. |
| [analysis/MentalRiskES_test/update_summary_with_eda.py](../analysis/MentalRiskES_test/update_summary_with_eda.py) | injector | 🛠 script | Inserts/replaces the marker-delimited test-EDA block in `SUMMARY.md`. Run after `eda_test_data.py`. |
| [analysis/MentalRiskES_test/outputs/truncation_verification_summary.md](../analysis/MentalRiskES_test/outputs/truncation_verification_summary.md) | Test, Phase −1 | ✅ | Scenario-A confirmation (evaluator scored only submitted rounds), per-task arithmetic. |
| [data/MentalRiskES-2026/test/task1/test/gold_label.json](../data/MentalRiskES-2026/test/task1/test/gold_label.json) | raw gold | raw | Item-level GAD-7 / PHQ-9 / CompACT-10 for 17 listed patients (10 in test data). |
| [data/MentalRiskES-2026/test/task2/test/gold/](../data/MentalRiskES-2026/test/task2/test/gold/) | raw gold | raw | Per-round `round_X_gold.json` with correct option per session. |

---

## 6. Ablation results on trial + synthetic data

### 6.1 Task 1 — calibration ablation (A0–A5 × T0/T2/T3)

| File | Cohort | Type | Notes |
|---|---|---|---|
| [runs/mentalriskes_ablation/ablation_summary.json](../runs/mentalriskes_ablation/ablation_summary.json) | trial | summary JSON, 6 configs | Per-config final metrics + 19-round trajectory |
| [runs/mentalriskes_ablation/ablation_report.txt](../runs/mentalriskes_ablation/ablation_report.txt) | trial | rough notes | Table skeleton only |
| [runs/mentalriskes_simulated_ablation/simulated_ablation_report.txt](../runs/mentalriskes_simulated_ablation/simulated_ablation_report.txt) | simulated | rough notes | Per-config mean RMSE + per-instrument breakdown |
| [runs/mentalriskes_simulated_ablation/sim_\*/ablation_summary.json](../runs/mentalriskes_simulated_ablation/) | simulated | per-persona JSON × 6 | Per-round trajectories per persona |
| Tables 1–3 in Task 1 doc §6 | trial + simulated | ✅ paper-ready | Final-round metrics, CompACT subscale, Kappa agreement, convergence trajectories, simulated mean RMSE |

### 6.2 Task 2 — selection ablation (B/B+/ENS × FUNC/HYB/TOM × ES/EN × W1/W3/W5 × FIX/PERM)

| File | Cohort | Type | Notes |
|---|---|---|---|
| [output/mentalriskes_task2/ablation/ablation_report.md](../output/mentalriskes_task2/ablation/ablation_report.md) | trial | ✅ paper-section quality | 9 configs, framings, pipeline variants, per-phase analysis, universal failure modes (rounds 3/9/17) |
| [output/mentalriskes_task2/ablation/ablation_summary.json](../output/mentalriskes_task2/ablation/ablation_summary.json) | trial | summary JSON | Latest two configs with bootstrap CIs |
| [output/mentalriskes_task2/ablation/B*_\*_\*.jsonl](../output/mentalriskes_task2/ablation/) (23 files) | trial | per-config JSONL | Round-by-round predictions for every config combination |
| [output/mentalriskes_task2/simulated_ablation/ablation_summary_simulated.json](../output/mentalriskes_task2/simulated_ablation/ablation_summary_simulated.json) | simulated | comprehensive summary | B = 92.0 %, B+ = 85.1 % (B+ hurts on simulated) |
| [output/mentalriskes_task2/simulated_ablation/\<config\>/sim_*.jsonl](../output/mentalriskes_task2/simulated_ablation/) | simulated | per-config per-persona JSONL | 7 personas per config |
| [output/mentalriskes_task2/ensemble/](../output/mentalriskes_task2/ensemble/) | trial + simulated | per-config JSONL | B+B ensemble outputs |
| Tables in Task 2 doc §5.4–5.6 | trial + simulated | ✅ paper-ready | v1.2 and v2.0 ablation tables, per-session simulated breakdown |

---

## 7. Test-set results — before and after the round-30 fix

The hard-coded `--max-rounds=30` default in [src/mentalriskes/combined_server.py](../src/mentalriskes/combined_server.py) (since fixed to default 200 with a hard-error if the cap is hit while the server is still serving) truncated the submission at round 30. Evidence:

| File | Coverage | Status | Notes |
|---|---|---|---|
| [analysis/MentalRiskES_test/outputs/truncation_verification_summary.md](../analysis/MentalRiskES_test/outputs/truncation_verification_summary.md) | both tasks | ✅ | Phase −1 finding: Scenario A — leaderboard scored only the submitted rounds |
| [analysis/MentalRiskES_test/verify_truncation_impact.py](../analysis/MentalRiskES_test/verify_truncation_impact.py) | both tasks | 🛠 script | Generates the above |

### 7.1 Pre-fix (submitted R1-30) — what the leaderboard saw

| File | Notes |
|---|---|
| [output/mentalriskes/predictions/round\{N\}_run\{R\}.json](../output/mentalriskes/predictions/) | Task 1 submitted predictions, rounds 1–30, 3 runs |
| [output/mentalriskes_task2/server_submissions/round\{N\}_run\{R\}.json](../output/mentalriskes_task2/server_submissions/) | Task 2 submitted predictions, rounds 1–30, 3 runs |
| [data/MentalRiskES-2026/MentalRiskES2026 - Results.xlsx](../data/MentalRiskES-2026/MentalRiskES2026%20-%20Results.xlsx) | Official leaderboard with our pre-fix scores (T1 Run 2 MAE_Combined 1.063, T2 Run 2 acc 0.247) |
| [analysis/MentalRiskES_test/outputs/Q_team_metric_ranks.csv](../analysis/MentalRiskES_test/outputs/Q_team_metric_ranks.csv) + [Q_balanced_rank.csv](../analysis/MentalRiskES_test/outputs/Q_balanced_rank.csv) | Per-instrument leaderboard rank — pre-fix |

### 7.2 Post-fix (full 82-round replay) — what we should have submitted

| File | Notes |
|---|---|
| [output/mentalriskes_test_replay/predictions/round\{N\}_run\{R\}.json](../output/mentalriskes_test_replay/predictions/) | Task 1 full replay, rounds 1–82, 3 runs |
| [output/mentalriskes_test_replay/logs/predictions_run\{N\}_\*.jsonl](../output/mentalriskes_test_replay/logs/) | Task 1 raw per-prediction JSONL with CoT |
| [output/mentalriskes_task2_test_replay/server_submissions/round\{N\}_run\{R\}.json](../output/mentalriskes_task2_test_replay/server_submissions/) | Task 2 full replay (Run 1 done, Runs 0 + 2 done) |
| [analysis/MentalRiskES_test/outputs/W_per_run_aggregate.csv](../analysis/MentalRiskES_test/outputs/W_per_run_aggregate.csv) | Task 1: submitted vs replay item-MAE per (run × instrument). **Surprise: replay is slightly WORSE than R30 snapshot on item-MAE for all 3 runs.** |
| [analysis/MentalRiskES_test/outputs/W_rank_projection.csv](../analysis/MentalRiskES_test/outputs/W_rank_projection.csv) | Task 1 projected leaderboard rank with the replay numbers |
| [analysis/MentalRiskES_test/outputs/W_t2_round_decomposition.csv](../analysis/MentalRiskES_test/outputs/W_t2_round_decomposition.csv) + [W_t2_round_tercile.csv](../analysis/MentalRiskES_test/outputs/W_t2_round_tercile.csv) | Task 2 replay accuracy + per-tercile breakdown |
| [analysis/MentalRiskES_test/outputs/figures/](../analysis/MentalRiskES_test/outputs/figures/) | Per-patient trajectory PNGs with R30 marker (Run 2) |
| [analysis/MentalRiskES_test/SUMMARY.md §4.5](../analysis/MentalRiskES_test/SUMMARY.md) | Narrative section on submitted-vs-replay |

---

## 8. Gemma-based solution description

| File | Status | Notes |
|---|---|---|
| [docs/mentalriskes_gemma_branch_description.md](mentalriskes_gemma_branch_description.md) | ✅ **standalone narrative** | Full two-arm system description: Arm A (Task 1 GAD-7 v1/v2 × 3 Gemma models) and Arm B (Task 2 bare-LLM S/S2/R2/S3/S4). Covers failure-mode diagnosis, prompt design, model selection rationale, results, shared infrastructure, cross-cohort lesson, limitations, future work. Cross-references the per-task solution descriptions and SUMMARY.md. |
| [specs/MentalRiskES/gemma_gad7_prompt_spec.md](../specs/MentalRiskES/gemma_gad7_prompt_spec.md) | ✅ design v1 | Original Gemma GAD-7 prompt rationale: removes Llama's severity-anchor inflation, adds anti-ceiling guidance for item 2, introduces per-item confidence estimation. |
| [specs/MentalRiskES/gemma_gad7_prompt_v2.md](../specs/MentalRiskES/gemma_gad7_prompt_v2.md) | ✅ design v2 | Adds severe-anxiety example, item-5/item-6 indirect-evidence guidance, soft severity calibration, refined confidence framing (frequency precision vs presence). |
| [specs/MentalRiskES/task2_improvement_spec.md](../specs/MentalRiskES/task2_improvement_spec.md) | ✅ design | Task 2 bare-LLM experiments S, S2, R2, S3, S4 — the bare-LLM with anti-bias guardrails (S2) is the headline winner. |
| [analysis/MentalRiskES_test/posthoc_P_gemma_gad7.py](../analysis/MentalRiskES_test/posthoc_P_gemma_gad7.py) | 🛠 runner | Embeds both v1 and v2 prompts; supports `--cohort {test,trial,simulated}`; OpenRouter via OpenAI SDK with retries + JSON-mode parsing. |
| [analysis/MentalRiskES_test/posthoc_S_task2_bare_llm.py](../analysis/MentalRiskES_test/posthoc_S_task2_bare_llm.py) | 🛠 runner | Modes S, S2, R2, S3 (perm × 6), S4 (pairwise + Condorcet); `--cohort` flag. |
| Task 1 solution doc §7 + Task 2 solution doc §9 | ✅ embedded summaries | Per-task condensed versions of the Gemma branch findings — cross-referenced from the stand-alone document. |
| [analysis/MentalRiskES_test/SUMMARY.md §5.5–§5.10](../analysis/MentalRiskES_test/SUMMARY.md) | ✅ narrative | Full evidence stack: prompt v1, prompt v2, bare-LLM, cross-cohort, consensus failure, Submitted-vs-S2, Task 1 cross-cohort. |

---

## 9. Gemma results across cohorts

### 9.1 Task 1 GAD-7 (Gemma 3 27B, Gemma 4 31B, Gemma 4 26B MoE × v1, v2)

| File | Cohort coverage | Status |
|---|---|---|
| [analysis/MentalRiskES_test/outputs/W_gemma_summary.csv](../analysis/MentalRiskES_test/outputs/W_gemma_summary.csv) | test, all 3 models, both prompt versions | ✅ |
| [W_gemma_per_item.csv](../analysis/MentalRiskES_test/outputs/W_gemma_per_item.csv), [W_gemma_per_item_pivot.csv](../analysis/MentalRiskES_test/outputs/W_gemma_per_item_pivot.csv), [W_gemma_per_item_delta.csv](../analysis/MentalRiskES_test/outputs/W_gemma_per_item_delta.csv) | test | ✅ |
| [W_gemma_confidence.csv](../analysis/MentalRiskES_test/outputs/W_gemma_confidence.csv) | test | ✅ — inverted confidence calibration |
| [W_gemma_hybrid.csv](../analysis/MentalRiskES_test/outputs/W_gemma_hybrid.csv) + [W_gemma_rank_projection.csv](../analysis/MentalRiskES_test/outputs/W_gemma_rank_projection.csv) | test | ✅ — all hybrids project to rank 4 |
| [W_gemma_best_per_session.csv](../analysis/MentalRiskES_test/outputs/W_gemma_best_per_session.csv) + [W_gemma_model_ranking.csv](../analysis/MentalRiskES_test/outputs/W_gemma_model_ranking.csv) | test | ✅ — oracle ensemble, single-best |
| [W_t1_cross_cohort.csv](../analysis/MentalRiskES_test/outputs/W_t1_cross_cohort.csv) + [W_t1_cross_cohort_summary.md](../analysis/MentalRiskES_test/outputs/W_t1_cross_cohort_summary.md) | test + simulated (trial has no item-gold) | ✅ |

### 9.2 Task 2 bare-LLM (S, S2, S3, S4, R2 — Gemma 3 27B, Gemma 4 31B, Llama-3.3-70B)

| File | Cohort coverage | Status |
|---|---|---|
| [W_t2_bare_summary.csv](../analysis/MentalRiskES_test/outputs/W_t2_bare_summary.csv) | test, all modes × 3 models | ✅ — S2 (Gemma 4 31B) = 0.470 |
| [W_t2_bare_per_run.csv](../analysis/MentalRiskES_test/outputs/W_t2_bare_per_run.csv) + [W_t2_bare_tercile.csv](../analysis/MentalRiskES_test/outputs/W_t2_bare_tercile.csv) + [W_t2_bare_confusion.csv](../analysis/MentalRiskES_test/outputs/W_t2_bare_confusion.csv) | test | ✅ |
| [W_t2_bare_R2_inversion.csv](../analysis/MentalRiskES_test/outputs/W_t2_bare_R2_inversion.csv) | test | ✅ — inversion hypothesis rejected |
| [W_t2_cross_cohort.csv](../analysis/MentalRiskES_test/outputs/W_t2_cross_cohort.csv) + [W_t2_cross_cohort_summary.md](../analysis/MentalRiskES_test/outputs/W_t2_cross_cohort_summary.md) | test + trial + simulated | ✅ — methodological finding: trial + simulated don't predict S2's test-set win |
| [SUMMARY.md §5.7](../analysis/MentalRiskES_test/SUMMARY.md) | narrative | ✅ |

---

## 10. Additional analysis reports

| File | Type | Status |
|---|---|---|
| [analysis/MentalRiskES_test/SUMMARY.md](../analysis/MentalRiskES_test/SUMMARY.md) | Master analysis report, v1.3 | ✅ — single most important document; cross-references everything else |
| [analysis/MentalRiskES_test/REPORT_T2_case_studies.md](../analysis/MentalRiskES_test/REPORT_T2_case_studies.md) | Stand-alone Task 2 case-study narrative | ✅ — built on disagreement + consensus-failure data |
| [analysis/MentalRiskES_test/outputs/qualitative_T2_submitted_vs_s2.md](../analysis/MentalRiskES_test/outputs/qualitative_T2_submitted_vs_s2.md) | 300-case head-to-head Submitted-vs-S2 disagreement Markdown (English-glossed) | ✅ appendix-ready |
| [analysis/MentalRiskES_test/outputs/qualitative_T2_disagreement_taxonomy.md](../analysis/MentalRiskES_test/outputs/qualitative_T2_disagreement_taxonomy.md) | 30 stratified Task 2 error cases with disagreement-taxonomy checkboxes | 🟡 awaiting manual labelling (~3 h of human work) |
| [analysis/MentalRiskES_test/outputs/qualitative_T1_case_studies.md](../analysis/MentalRiskES_test/outputs/qualitative_T1_case_studies.md) | Per-instrument Task 1 worst-case patient deep dives | 🟡 working artefact; small sample |
| [analysis/MentalRiskES_test/outputs/W_t2_consensus_failures.md](../analysis/MentalRiskES_test/outputs/W_t2_consensus_failures.md) + [W_t2_consensus_failure_stats.csv](../analysis/MentalRiskES_test/outputs/W_t2_consensus_failure_stats.csv) | Cases where every tested system was wrong; gold=3 is 38.5 % all-wrong | ✅ |
| [analysis/MentalRiskES_test/outputs/O_oracle_swap.csv](../analysis/MentalRiskES_test/outputs/O_oracle_swap.csv) | Layer 1 oracle component swap (Gemma baseline GAD-7) | ✅ |
| [analysis/MentalRiskES_test/outputs/P_gad7_corrections.csv](../analysis/MentalRiskES_test/outputs/P_gad7_corrections.csv) | Layer 2 principled GAD-7 corrections P1–P5 + INV_low | ✅ |
| [analysis/MentalRiskES_test/outputs/B_item_level_errors.csv](../analysis/MentalRiskES_test/outputs/B_item_level_errors.csv), [B_compact10_subscale_run2.csv](../analysis/MentalRiskES_test/outputs/B_compact10_subscale_run2.csv) | Item-level error decomposition (R30 baseline) | 🛠 |
| [analysis/MentalRiskES_test/outputs/C_run2_GAD-7_confusion.csv](../analysis/MentalRiskES_test/outputs/C_run2_GAD-7_confusion.csv), [C_run2_PHQ-9_confusion.csv](../analysis/MentalRiskES_test/outputs/C_run2_PHQ-9_confusion.csv), [C_run2_boundary_zone.csv](../analysis/MentalRiskES_test/outputs/C_run2_boundary_zone.csv) | Severity-band misclassification + ±SEM boundary analysis | 🛠 |
| [analysis/MentalRiskES_test/outputs/T_bias_long.csv](../analysis/MentalRiskES_test/outputs/T_bias_long.csv) | Task 2 position / length bias evidence | 🛠 |
| [analysis/MentalRiskES_test/outputs/H_task2_predictions_long.csv](../analysis/MentalRiskES_test/outputs/H_task2_predictions_long.csv), [H_run\{0,1,2\}_confusion.csv](../analysis/MentalRiskES_test/outputs/) | Task 2 confusion matrices per submitted run | 🛠 |

---

## Assessment of the two solution descriptions

**Both `docs/mentalriskes_task1_solution_description.md` and `docs/mentalriskes_task2_solution_description.md` are now up to date.** They accurately describe the submitted systems (design rationale, ablation evidence, pre-submission prompt-engineering decisions) *and* now carry a post-hoc section folding in everything we learned and built after April 20, 2026:

| Topic | Task 1 reference | Task 2 reference |
|---|---|---|
| Truncation disclosure (`--max-rounds=30`) | §7.1 | §9.1 |
| Full-replay results on 82 rounds | §7.2 | §9.2 |
| Per-instrument leaderboard ranking (Analysis Q) | §7.3 | — (not applicable) |
| Gemma GAD-7 post-hoc / bare-LLM Experiment S | §7.4 | §9.3 |
| Hybrid combined rank-4 projection / Experiment S2 (0.470) | §7.5 | §9.4 |
| Cross-cohort comparison (trial + simulated under-predict) | §7.6 | §9.7 |
| Consensus-failure analysis (gold=3 is hardest) | — | §9.6 |
| Submitted-vs-S2 disagreement Markdown pointer | — | §9.8 |
| R2/S3/S4 follow-up experiments | — | §9.5 |
| Overall summary | §7.7 | §9.9 |

Cross-references throughout the post-hoc sections point back to [analysis/MentalRiskES_test/SUMMARY.md](../analysis/MentalRiskES_test/SUMMARY.md) for the full evidence.

---

## Gaps worth filling before paper submission

1. ✅ ~~`docs/eda_trial_data.md`~~ — superseded by [analysis/MentalRiskES_trial/outputs/eda_trial_data.md](../analysis/MentalRiskES_trial/outputs/eda_trial_data.md) and the auto-generated §0 in [trial_calibration_report.md](../output/mentalriskes/trial_calibration_report.md).
2. ✅ ~~`docs/eda_test_data.md`~~ — superseded by [analysis/MentalRiskES_test/outputs/eda_test_data.md](../analysis/MentalRiskES_test/outputs/eda_test_data.md) and the auto-generated §0.7 in [SUMMARY.md](../analysis/MentalRiskES_test/SUMMARY.md).
3. 🟡 Manual labelling of the 30 cases in [qualitative_T2_disagreement_taxonomy.md](../analysis/MentalRiskES_test/outputs/qualitative_T2_disagreement_taxonomy.md) — ~3 hours of human work, anchors the theory–practice gap section of the Task 2 paper.
4. ✅ ~~Update the two task solution docs with the post-hoc section~~ — done 2026-05-12 (see assessment above).
5. ✅ ~~(Optional) `docs/mentalriskes_gemma_branch_description.md`~~ — written 2026-05-12. See [docs/mentalriskes_gemma_branch_description.md](mentalriskes_gemma_branch_description.md): stand-alone description of the two-arm Gemma branch (Arm A = Task 1 GAD-7 v1/v2 prompts × 3 Gemma models; Arm B = Task 2 bare-LLM S/S2/R2/S3/S4 experiments) with full results, shared infrastructure, cross-cohort lesson, limitations, and cross-references back to the per-task solution descriptions.

---

## EDA regeneration commands

After any data change (new trial labels, additional simulated personas, etc.), re-run the four scripts in order:

```bash
# Trial corpus + injection into trial_calibration_report.md
python analysis/MentalRiskES_trial/eda_trial_data.py
python analysis/MentalRiskES_trial/update_trial_calibration_report.py

# Test corpus + injection into SUMMARY.md
python analysis/MentalRiskES_test/eda_test_data.py
python analysis/MentalRiskES_test/update_summary_with_eda.py
```

Both injectors are idempotent (marker-delimited replacement) — no manual edits required.
