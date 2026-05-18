# MentalRiskES 2026 — Task 1 Comprehensive Results

**Task:** zero-shot multi-instrument psychometric scoring (PHQ-9, GAD-7, CompACT-10) from Spanish therapeutic conversations.
**Team:** INSALyon.
**Pipeline source:** [src/mentalriskes/task1/](../src/mentalriskes/task1/) — submitted Llama-3.3-70B three-tier calibrated pipeline.
**Submitted runs:** Run 0 = A5-T3, Run 1 = A3-T2, Run 2 = A1-T2 (definitions below).
**Date compiled:** 2026-05-14.

---

## 0. Configuration legend

### 0.1 Calibration levels (assessor + post-assessment stack)

| Level | Component | Purpose | Cost | Default in |
|---|---|---|---|---|
| A | Prompt anchors embedded in each instrument's chain-of-thought prompt | Steer raw item scores toward psychometrically valid ranges (e.g., CompACT VA anchor for moderate distress = 3–4; PHQ-9–GAD-7 totals within 4 points) | Zero (in-prompt) | A1, A3, A5 |
| B | 7 rule-based cross-instrument constraints (C1–C7), applied after the three assessors return scores | Detect / correct cross-instrument discordance, somatic mismatches, CompACT spread, item-9 suicidality flag | Deterministic, no LLM | A3, A5 |
| C | Conditional LLM calibration agent (~1.5K tokens) invoked when Level B fires high/medium violations or CompACT-10 ceiling +5 | Resolve nuanced contradictions Level B cannot (e.g., within- vs out-of-session valued action) | One extra LLM call per fire | A5 only |

### 0.2 Temporal aggregation across rounds

| Code | Aggregation rule | Use case |
|---|---|---|
| T0 | Last-round-only: submit the most recent per-round prediction | Baseline, maximally susceptible to recency bias |
| T2 | Early-weighted: linearly decaying weight, rounds 1–5 weighted 2×, rounds 6+ at 1× | PHQ-9 / GAD-7 (past-two-weeks reference window) |
| T3 | Stability-adaptive: keep the prediction once item-level Mahalanobis stability falls below a threshold | CompACT-10 (process measures stabilise mid-session) |

### 0.3 Configuration ID convention

`A{0,1,3,5}-T{0,2,3}` = (calibration level applied) × (temporal aggregation). Officially submitted runs are flagged explicitly.

| Config | Anchors (A) | Rules (B) | Agent (C) | Temporal | Status |
|---|:-:|:-:|:-:|:-:|---|
| A0-T0 | No | No | No | T0 | Offline baseline |
| A1-T0 | Yes | No | No | T0 | Offline |
| A1-T2 | Yes | No | No | T2 | **Submitted Run 2** |
| A3-T2 | Yes | Yes | No | T2 | **Submitted Run 1** |
| A3-T3 | Yes | Yes | No | T3 | Offline variant |
| A5-T2 | Yes | Yes | Yes | T2 | Offline variant |
| A5-T3 | Yes | Yes | Yes | T3 | **Submitted Run 0** |

### 0.4 LLM configuration

- **Submitted runs (trial + test):** Llama-3.3-70B-Instruct via DeepInfra OpenAI-compatible API; temperature 0.2; top_p 0.9; max_tokens 1024 per assessor call; JSON-mode parsing with retry.
- **Trial / simulated ablation (pre-submission):** Llama-3.3-70B-Instruct-Turbo via TogetherAI.
- **Post-hoc Gemma branch (test only):** Gemma 3 27B, Gemma 4 31B, Gemma 4 26B MoE via OpenRouter (OpenAI SDK). v1 and v2 GAD-7 prompts.
- Full provider/sampling/retry matrix: [docs/mentalriskes_llm_configuration.md](mentalriskes_llm_configuration.md).

### 0.5 Cohorts

| Cohort | Sessions | Rounds | Gold provided | Notes |
|---|---|---|---|---|
| Trial | 1 (Miguel) | 19 (rounds 1–19) | Item-level PHQ-9 (13) / GAD-7 (14) / CompACT-10 (33) | Single moderate-distress patient |
| Simulated | 6 personas | 15/session (90 total) | Target totals + CompACT profile in `metadata.json` | Anxiety × 3, depression × 3 |
| Test (R1–30, submitted) | 10 (S01, S03–S07, S09, S12, S15, S16) | 30 (truncation bug capped submission) | Item-level for 17 patients in `gold_label.json`; 10 present in released test data | What the leaderboard scored |
| Test (R1–82, full replay) | 10 | 30–82 per session (568 patient-rounds) | Same gold | Post-submission replay on same Llama pipeline |

---

## 1. Trial cohort — A0–A5 × T0/T2/T3 ablation (n = 19 rounds, 1 patient)

Source: [runs/mentalriskes_ablation/ablation_summary.json](../runs/mentalriskes_ablation/ablation_summary.json) (final-round metrics + 19-round trajectories per config); aggregated in [output/mentalriskes/trial_calibration_report.md](../output/mentalriskes/trial_calibration_report.md). Gold: PHQ-9 = 13, GAD-7 = 14, CompACT-10 = 33.

### 1.1 Final-round metrics (round 19)

| Config | PHQ-9 RMSE | GAD-7 RMSE | CompACT RMSE | Mean RMSE | Mean Pearson | PHQ-9 Total (pred / gold) | GAD-7 Total | CompACT Total | Bands correct? |
|---|---:|---:|---:|---:|---:|:-:|:-:|:-:|:-:|
| A0-T0 (baseline) | 0.577 | 0.378 | 0.548 | 0.501 | 0.643 | 14 / 13 | 13 / 14 | 34 / 33 | PHQ✓ GAD✓ |
| A1-T0 (anchors only) | 0.333 | 0.535 | 0.837 | 0.568 | 0.708 | 14 / 13 | 16 / 14 | 38 / 33 | PHQ✓ GAD✗ (over) |
| **A1-T2 (Run 2)** | **0.000** | 0.535 | 0.837 | 0.457 | 0.818 | 13 / 13 | 16 / 14 | 40 / 33 | PHQ✓ GAD✗ (over) |
| **A3-T2 (Run 1)** | **0.000** | 0.535 | 0.837 | 0.457 | 0.818 | 13 / 13 | 16 / 14 | 40 / 33 | PHQ✓ GAD✗ (over) |
| A3-T3 | **0.000** | 0.535 | 0.707 | 0.414 | 0.697 | 13 / 13 | 16 / 14 | 38 / 33 | PHQ✓ GAD✗ |
| A5-T2 | **0.000** | 0.655 | 0.837 | 0.497 | 0.808 | 13 / 13 | 17 / 14 | 40 / 33 | PHQ✓ GAD✗ |
| **A5-T3 (Run 0)** | **0.000** | **0.378** | 0.837 | **0.405** | **0.842** | 13 / 13 | 15 / 14 | 40 / 33 | PHQ✓ GAD✗ |

Reading: A5-T3 wins on Mean RMSE (0.405) and Mean Pearson (0.842); A1-T2 / A3-T2 tie on PHQ-9 and Pearson but lose GAD-7 by 0.157 RMSE.

### 1.2 CompACT-10 subscale decomposition (round 19)

| Config | OtE RMSE | BA RMSE | VA RMSE | VA mean (gold 3.5) | Level B violations (cumulative) | Level C corrections |
|---|---:|---:|---:|---:|:-:|:-:|
| A0-T0 | 0.577 | 0.000 | 0.707 | 4.0 | 0 | 0 |
| A1-T0 | 0.817 | 0.577 | 1.000 | 4.5 | 0 | 0 |
| A1-T2 | 0.817 | 0.577 | 1.000 | 4.5 | 0 | 0 |
| A3-T2 | 0.817 | 0.577 | 1.000 | 4.5 | 4 | 0 |
| A3-T3 | 0.817 | 0.577 | **0.707** | 4.0 | 5 | 0 |
| A5-T2 | 0.817 | 0.577 | 1.000 | 4.5 | 1 | 1 |
| A5-T3 | 0.817 | 0.577 | 1.000 | 4.5 | 4 | 2 |

The C4 self-contradiction guard fires four times on A3 / A5 configurations and triggers two LLM corrections on A5-T3 but never on A3 (rule-only).

### 1.3 Kappa agreement (quadratic, round 19)

| Config | PHQ-9 κ_q | GAD-7 κ_q | CompACT κ_q |
|---|---:|---:|---:|
| A0-T0 | 0.716 | 0.800 | 0.348 |
| A1-T0 | 0.883 | 0.667 | 0.314 |
| **A1-T2 (Run 2)** | **1.000** | 0.667 | 0.364 |
| **A3-T2 (Run 1)** | **1.000** | 0.667 | 0.364 |
| A3-T3 | **1.000** | 0.667 | 0.194 |
| A5-T2 | **1.000** | 0.571 | 0.364 |
| **A5-T3 (Run 0)** | **1.000** | **0.800** | 0.364 |

### 1.4 Convergence trajectory (mean RMSE across rounds)

| Config | R1 | R3 | R5 | R9 | R12 | R15 | R18 | R19 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A0-T0 | 0.494 | 0.525 | 0.211 | 0.211 | 0.436 | 0.211 | 0.432 | 0.501 |
| A1-T0 | 0.568 | 0.642 | 0.497 | 0.457 | 0.592 | 0.414 | 0.571 | 0.568 |
| A1-T2 | 0.464 | 0.361 | 0.414 | 0.414 | 0.414 | 0.414 | 0.457 | 0.457 |
| A3-T2 | 0.464 | 0.497 | 0.405 | 0.405 | 0.405 | 0.405 | 0.457 | 0.457 |
| A3-T3 | 0.642 | 0.497 | 0.414 | 0.476 | 0.429 | 0.454 | 0.361 | 0.414 |
| A5-T2 | 0.401 | 0.476 | 0.497 | 0.497 | 0.497 | 0.497 | 0.497 | 0.497 |
| A5-T3 | 0.529 | 0.405 | 0.405 | 0.405 | 0.362 | 0.405 | 0.309 | 0.405 |

A5-T3 reaches its final-round level by R3 and stays there; T2 configurations stabilise by R5.

---

## 2. Simulated cohort — A0–A5 × T0/T2/T3 ablation (n = 90 rounds, 6 personas)

Source: [runs/mentalriskes_simulated_ablation/](../runs/mentalriskes_simulated_ablation/) per-session JSON × 6 + [simulated_ablation_report.txt](../runs/mentalriskes_simulated_ablation/simulated_ablation_report.txt). Targets: per-persona `target_scores.phq9_total / gad7_total / compact10_profile`. The aggregator computes per-session metrics and averages across the 6 sessions.

### 2.1 Aggregate metrics (6 sessions; A5-T3 has 5 sessions — one persona failed late)

| Config | Mean RMSE | PHQ-9 RMSE | GAD-7 RMSE | CompACT RMSE |
|---|---:|---:|---:|---:|
| A0-T0 | 1.415 | 0.951 | 1.299 | 1.996 |
| A1-T0 | 1.412 | 0.944 | 1.245 | 2.047 |
| **A1-T2 (Run 2)** | **1.272** | 0.909 | **0.943** | 1.964 |
| **A3-T2 (Run 1)** | 1.311 | 0.916 | 1.021 | 1.996 |
| A3-T3 | 1.275 | 0.916 | 0.971 | **1.938** |
| A5-T2 | 1.288 | 0.916 | 0.993 | 1.954 |
| **A5-T3 (Run 0)** | 1.320 | 0.970 | 0.989 | 2.000 |

On simulated, **A1-T2 (Run 2) wins** — the inverse of the trial ranking. Adding Level B / Level C does not help on the persona dialogues.

### 2.2 Per-persona breakdown (mean RMSE)

| Persona | Profile | A0-T0 | A1-T0 | A1-T2 | A3-T2 | A3-T3 | A5-T2 | A5-T3 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| sim_anx_academic_42 | Academic anxiety, perfectionist | 1.110 | 1.391 | 1.365 | 1.429 | 1.466 | 1.399 | 1.432 |
| sim_anx_health_46 | Health anxiety, somatic | 1.687 | 1.752 | 1.489 | 1.530 | 1.487 | 1.603 | 1.276 |
| sim_anx_social_44 | Social anxiety, avoidant | 2.337 | 1.788 | 1.483 | 1.633 | 1.450 | 1.454 | 1.574 |
| sim_dep_burnout_45 | Burnout depression | 1.084 | 1.258 | 0.971 | 1.053 | 1.007 | 0.971 | — |
| sim_dep_loss_43 | Loss / grief depression | 1.306 | 1.090 | 1.235 | 1.210 | 1.210 | 1.210 | 1.183 |
| sim_dep_mild_47 | Mild depression | 0.967 | 1.192 | 1.088 | 1.013 | 1.030 | 1.088 | 1.133 |

Per-persona PHQ-9 / GAD-7 / CompACT RMSE available in each `runs/mentalriskes_simulated_ablation/sim_*/ablation_summary_sim_*.json`.

### 2.3 Per-persona mean Pearson

| Persona | A0-T0 | A1-T0 | A1-T2 | A3-T2 | A3-T3 | A5-T2 | A5-T3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| sim_anx_academic_42 | 0.544 | 0.621 | 0.361 | 0.337 | 0.421 | 0.315 | 0.440 |
| sim_anx_health_46 | 0.434 | 0.081 | 0.234 | −0.028 | −0.164 | −0.039 | 0.183 |
| sim_anx_social_44 | −0.059 | 0.034 | 0.097 | 0.193 | 0.049 | 0.127 | −0.180 |
| sim_dep_burnout_45 | 0.219 | 0.049 | 0.051 | 0.051 | −0.021 | 0.051 | — |
| sim_dep_loss_43 | 0.095 | 0.162 | −0.135 | −0.135 | −0.135 | −0.135 | −0.135 |
| sim_dep_mild_47 | 0.312 | 0.274 | 0.213 | 0.275 | 0.312 | 0.213 | 0.145 |

---

## 3. Test cohort — submitted Runs 0/1/2 on R1–30 (leaderboard) and full R1–82 replay

### 3.1 Official leaderboard (released by organisers, all 82 rounds scored under Scenario A — only submitted rounds count)

Source: [data/MentalRiskES-2026/MentalRiskES2026 - Results.xlsx](../data/MentalRiskES-2026/MentalRiskES2026%20-%20Results.xlsx). INSALyon rows extracted plus key per-instrument ranks.

| Run | Config | MZOE_GAD7 | MAE_GAD7 | Macro_MAE_GAD7 | MZOE_PHQ9 | MAE_PHQ9 | Macro_MAE_PHQ9 | MZOE_CompACT10 | MAE_CompACT10 | Macro_MAE_CompACT10 | MAE_Combined | Combined rank |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:-:|
| Run 0 | A5-T3 | 0.742 | 1.159 | 1.059 | 0.632 | **0.797** | 0.831 | 0.863 | 1.366 | **1.642** | 1.107 | 13 |
| Run 1 | A3-T2 | 0.773 | 1.191 | 1.056 | 0.635 | 0.817 | 0.850 | 0.809 | **1.302** | 1.673 | 1.103 | 12 |
| **Run 2** | A1-T2 | **0.709** | **1.036** | **0.918** | 0.672 | 0.829 | 0.853 | 0.833 | 1.324 | 1.657 | **1.063** | **10** |

Per-instrument leaderboard ranks (best of three runs) — see [Q_team_metric_ranks.csv](../analysis/MentalRiskES_test/outputs/Q_team_metric_ranks.csv):

| Metric | Best run | Value | Rank (of 19 teams + baselines) |
|---|---|---:|:-:|
| GAD-7 MAE | Run 2 | 1.036 | 8 |
| PHQ-9 MAE | Run 0 | 0.797 | 4 |
| **CompACT-10 MAE** | **Run 1** | **1.302** | **3** |
| GAD-7 Macro_MAE | Run 2 | 0.918 | 5 |
| PHQ-9 Macro_MAE | Run 0 | 0.831 | 3 |
| **CompACT-10 Macro_MAE** | **Run 0** | **1.642** | **2** |
| MAE_Combined | Run 2 | 1.063 | **10** |
| Balanced rank (mean of per-instrument best ranks) | — | — | **4.17** |

Top-of-leaderboard reference: FBKillers (Combined rank 1, MAE 0.879); VerbaNex AI (rank 2, 0.883); BASELINE-Gemma (rank 3, 1.002); BUAP-NLP Lab (rank 4, 1.039).

### 3.2 Local R1–30 per-instrument (matches the leaderboard within ±0.10)

Source: [analysis/MentalRiskES_test/outputs/A_per_run_summary.csv](../analysis/MentalRiskES_test/outputs/A_per_run_summary.csv).

| Run | Instrument | MAE_items | Mean signed total bias | MAE_total | Band acc (10 sessions) |
|---|---|---:|---:|---:|---:|
| Run 0 (A5-T3) | PHQ-9 | 0.778 | −3.6 | 4.6 | 0.30 |
| Run 0 | GAD-7 | 1.057 | −6.4 | 6.4 | 0.20 |
| Run 0 | CompACT-10 | 1.380 | −5.6 | 7.8 | 1.00 |
| Run 1 (A3-T2) | PHQ-9 | 0.778 | −3.8 | 4.4 | 0.20 |
| Run 1 | GAD-7 | 1.129 | −6.1 | 6.5 | 0.20 |
| Run 1 | CompACT-10 | 1.230 | −1.7 | 5.9 | 1.00 |
| Run 2 (A1-T2) | PHQ-9 | 0.778 | −3.4 | 4.6 | 0.20 |
| Run 2 | GAD-7 | 0.971 | −5.8 | 5.8 | 0.20 |
| Run 2 | CompACT-10 | 1.280 | −2.4 | 6.8 | 1.00 |

### 3.3 Submitted R1–30 vs full R1–82 replay (same pipeline, same model)

Source: [analysis/MentalRiskES_test/outputs/W_per_run_aggregate.csv](../analysis/MentalRiskES_test/outputs/W_per_run_aggregate.csv).

| Run | Instrument | Submitted MAE_items (R1–30) | Replay MAE_items (R1–82) | Δ | Submitted signed bias | Replay signed bias |
|---|---|---:|---:|---:|---:|---:|
| Run 0 | PHQ-9 | 0.778 | 0.889 | +0.111 | −3.6 | −4.4 |
| Run 0 | GAD-7 | 1.057 | 1.100 | +0.043 | −6.4 | −6.1 |
| Run 0 | CompACT-10 | 1.380 | 1.320 | −0.060 | −5.6 | −4.4 |
| Run 1 | PHQ-9 | 0.778 | 0.844 | +0.067 | −3.8 | −4.4 |
| Run 1 | GAD-7 | 1.129 | 1.086 | −0.043 | −6.1 | −6.0 |
| Run 1 | CompACT-10 | 1.230 | 1.240 | +0.010 | −1.7 | −0.8 |
| Run 2 | PHQ-9 | 0.778 | 0.878 | +0.100 | −3.4 | −4.3 |
| Run 2 | GAD-7 | 0.971 | 1.086 | +0.114 | −5.8 | −6.0 |
| Run 2 | CompACT-10 | 1.280 | 1.250 | −0.030 | −2.4 | −0.9 |

**Replay is slightly worse on item-MAE for all 3 runs:** Run 0 +0.031 mean Δ, Run 1 +0.011, Run 2 +0.061 (MAE_Combined). PHQ-9 and GAD-7 worsen as transcripts get longer; only CompACT-10 mildly improves. The truncation bug cost coverage, not quality.

### 3.4 Test-cohort severity-band confusion (Run 2, R1–30, 10 sessions)

GAD-7 (band accuracy = 0.20; 80 % of misclassifications are *under*-classifications):

```
gold \ pred   minimal  mild  moderate  severe
mild              0      1       0        0
moderate          0      1       1        0
severe            1      1       5        0
```

PHQ-9 (band accuracy = 0.20; no prediction ever lands in moderately_severe or severe):

```
gold \ pred           minimal  mild  moderate  moderately_severe  severe
minimal                  0       1       0           0              0
mild                     0       0       1           0              0
moderate                 0       1       2           0              0
moderately_severe        0       0       3           0              0
severe                   0       0       2           0              0
```

### 3.5 Item-level error highlights (Run 2)

- **GAD-7 #4 "Trouble relaxing"** — MAE 1.2, signed −1.2 (always under)
- **GAD-7 #6 "Irritability"** — MAE 1.3, signed −1.3 (always under)
- **CompACT-10 #1, 6, 9** (BA subscale) — MAE 1.5–1.7
- **CompACT-10 #7** — MAE 1.7, signed +0.9 (the one VA item we still over-score)
- **PHQ-9 #5 "Appetite"** — best exact-match (0.7)
- **PHQ-9 #9 "Suicidality"** — gold mean 0.9, predicted mean 0.0 (never above 0). Safety-relevant.

CompACT-10 subscale signed bias on test: OtE = −0.87 (under), BA = −0.43 (under), VA = +0.38 (over — the trial-diagnosed VA over-prediction holds qualitatively but with smaller magnitude).

---

## 4. Post-hoc Gemma GAD-7 branch (Layer 3) — test cohort, 10 sessions × 82 rounds

GAD-7 re-scored with three Gemma models × two prompt versions (v1, v2). v1: severity-anchor reframe + anti-ceiling for item 2 + per-item confidence. v2 (over v1): adds severe (gold=17) anchor example + indirect-evidence markers for items 5 & 6 + soft severity calibration. Source: [W_gemma_summary.csv](../analysis/MentalRiskES_test/outputs/W_gemma_summary.csv) + [W_gemma_hybrid.csv](../analysis/MentalRiskES_test/outputs/W_gemma_hybrid.csv).

### 4.1 Standalone GAD-7 (full 82 rounds)

| Model / prompt | GAD-7 MAE_items | Signed total bias | Band acc | Δ vs Llama replay (1.086) |
|---|---:|---:|---:|---:|
| Our Llama (submitted R1–30) | 0.971 | −5.8 | 0.20 | — |
| Our Llama (replay R1–82) | 1.086 | −6.0 | 0.20 | — |
| Gemma 3 27B v1 | 0.814 | −4.9 | 0.30 | −25 % |
| Gemma 3 27B v2 | 0.743 | −3.8 | 0.20 | −32 % |
| Gemma 4 31B v1 | 0.786 | −4.7 | 0.30 | −28 % |
| Gemma 4 26B MoE v1 | 0.743 | −4.0 | 0.20 | −32 % |
| **Gemma 4 26B MoE v2** | **0.714** | **−3.4** | **0.50** | **−34 %** |
| Competition Gemma baseline | 0.582 | unknown | unknown | (target) |

### 4.2 Hybrid combined MAE = our PHQ-9 + Gemma GAD-7 + our CompACT-10 (full table)

| GAD-7 model | PHQ-9 / CompACT base | Run | PHQ-9 MAE | GAD-7 MAE (Gemma) | CompACT-10 MAE | MAE_Combined | Projected leaderboard rank |
|---|---|:-:|---:|---:|---:|---:|:-:|
| Gemma 3 27B v1 | submitted (R1–30) | 0 | 0.778 | 0.814 | 1.380 | 0.991 | 6 |
| Gemma 3 27B v1 | submitted | 1 | 0.778 | 0.814 | 1.230 | 0.941 | 4 |
| Gemma 3 27B v1 | submitted | 2 | 0.778 | 0.814 | 1.280 | 0.957 | 4 |
| Gemma 3 27B v1 | replay (R1–82) | 0 | 0.889 | 0.814 | 1.320 | 1.008 | 6 |
| Gemma 3 27B v1 | replay | 1 | 0.844 | 0.814 | 1.240 | 0.966 | 4 |
| Gemma 3 27B v1 | replay | 2 | 0.878 | 0.814 | 1.250 | 0.981 | 4 |
| Gemma 3 27B v2 | submitted | 0 | 0.778 | 0.743 | 1.380 | 0.967 | 4 |
| Gemma 3 27B v2 | submitted | 1 | 0.778 | 0.743 | 1.230 | 0.917 | 4 |
| Gemma 3 27B v2 | submitted | 2 | 0.778 | 0.743 | 1.280 | 0.934 | 4 |
| Gemma 3 27B v2 | replay | 0 | 0.889 | 0.743 | 1.320 | 0.984 | 4 |
| Gemma 3 27B v2 | replay | 1 | 0.844 | 0.743 | 1.240 | 0.942 | 4 |
| Gemma 3 27B v2 | replay | 2 | 0.878 | 0.743 | 1.250 | 0.957 | 4 |
| Gemma 4 31B v1 | submitted | 0 | 0.778 | 0.786 | 1.380 | 0.981 | 4 |
| Gemma 4 31B v1 | submitted | 1 | 0.778 | 0.786 | 1.230 | 0.931 | 4 |
| Gemma 4 31B v1 | submitted | 2 | 0.778 | 0.786 | 1.280 | 0.948 | 4 |
| Gemma 4 31B v1 | replay | 0 | 0.889 | 0.786 | 1.320 | 0.998 | 4 |
| Gemma 4 31B v1 | replay | 1 | 0.844 | 0.786 | 1.240 | 0.957 | 4 |
| Gemma 4 31B v1 | replay | 2 | 0.878 | 0.786 | 1.250 | 0.971 | 4 |
| Gemma 4 26B MoE v1 | submitted | 0 | 0.778 | 0.743 | 1.380 | 0.967 | 4 |
| Gemma 4 26B MoE v1 | submitted | 1 | 0.778 | 0.743 | 1.230 | 0.917 | 4 |
| Gemma 4 26B MoE v1 | submitted | 2 | 0.778 | 0.743 | 1.280 | 0.934 | 4 |
| Gemma 4 26B MoE v1 | replay | 0 | 0.889 | 0.743 | 1.320 | 0.984 | 4 |
| Gemma 4 26B MoE v1 | replay | 1 | 0.844 | 0.743 | 1.240 | 0.942 | 4 |
| Gemma 4 26B MoE v1 | replay | 2 | 0.878 | 0.743 | 1.250 | 0.957 | 4 |
| **Gemma 4 26B MoE v2** | **submitted** | **1** | 0.778 | **0.714** | 1.230 | **0.907** | **4** |
| Gemma 4 26B MoE v2 | submitted | 0 | 0.778 | 0.714 | 1.380 | 0.957 | 4 |
| Gemma 4 26B MoE v2 | submitted | 2 | 0.778 | 0.714 | 1.280 | 0.924 | 4 |
| Gemma 4 26B MoE v2 | replay | 0 | 0.889 | 0.714 | 1.320 | 0.974 | 4 |
| Gemma 4 26B MoE v2 | replay | 1 | 0.844 | 0.714 | 1.240 | 0.933 | 4 |
| Gemma 4 26B MoE v2 | replay | 2 | 0.878 | 0.714 | 1.250 | 0.947 | 4 |

**Every hybrid configuration (28 of 30) projects to leaderboard rank 4**; the two that fall to rank 6 use replay PHQ-9 with the weaker Gemma 3 27B v1 GAD-7. Loading the GAD-7 component, not the rounds, is what moves rank.

### 4.3 Cross-cohort sanity — Gemma 4 26B MoE v1 vs v2 (test + simulated; trial has no item-gold for Task 1)

Source: [W_t1_cross_cohort.csv](../analysis/MentalRiskES_test/outputs/W_t1_cross_cohort.csv).

| Model / prompt | Cohort | n | item-MAE | total-MAE | signed total bias | band acc |
|---|---|:-:|---:|---:|---:|---:|
| Gemma 4 26B MoE v1 | test | 10 | 0.74 | 4.0 | −4.0 | 0.20 |
| Gemma 4 26B MoE v2 | test | 10 | **0.71** | **3.4** | −3.4 | **0.50** |
| Gemma 4 31B v1 | test | 10 | 0.79 | 4.7 | −4.7 | 0.30 |
| Gemma 3 27B v2 | test | 10 | 0.74 | 4.8 | −3.8 | 0.20 |
| Gemma 3 27B v1 | test | 10 | 0.81 | 5.1 | −4.9 | 0.30 |
| Llama-3.3-70B (our pipeline, replay) | test | 10 | 1.09 | 6.4 | −6.0 | 0.20 |
| Gemma 4 26B MoE v1 | simulated | 6 | — | 5.5 | −2.5 | 0.50 |
| Gemma 4 26B MoE v2 | simulated | 6 | — | 5.7 | −1.7 | 0.50 |

The v2 prompt clearly wins on test (band acc 0.20→0.50) but is **indistinguishable from v1 on the simulated cohort** — pre-submission ablation would not have selected v2.

### 4.4 Per-item Gemma deltas vs Llama (test)

| Item | Llama signed | Gemma 3 27B v1 | Gemma 4 31B v1 | Gemma 4 26B MoE v1 | Comment |
|---|---:|---:|---:|---:|---|
| 1. Nervousness | −0.8 | −0.2 | −0.3 | **−0.1** | well-calibrated |
| 2. Uncontrollable worry | −0.5 (trial showed +1) | −0.5 | −0.2 | **+0.2** | anti-ceiling worked across all Gemmas |
| 3. Excessive worry | −0.8 | −0.5 | −0.5 | −0.3 | improved |
| 4. Trouble relaxing | −1.2 | −0.7 | −0.4 | −0.5 | improved |
| 5. Restlessness | −0.9 | −1.4 | −1.3 | −1.3 | shared blind spot |
| 6. Irritability | −1.3 | −1.3 | −1.6 | −1.6 | shared blind spot |
| 7. Dread | −0.3 | −0.3 | −0.4 | −0.4 | unchanged |

Item 5 / item 6 v2-prompt fix on the MoE variant: item 5 signed bias −1.3 → −0.4, item 6 ~ −1.6 → ~ −1.0.

### 4.5 Confidence calibration is inverted (Gemma 3 27B)

| Self-rated confidence | Items | item-MAE |
|---|:-:|---:|
| HIGH | 1741 | 1.09 |
| MEDIUM | 2132 | **0.82** ← lowest |
| LOW | 103 | 1.29 |

HIGH-confidence items are *less* accurate than MEDIUM. The reflective confidence step does not produce expected calibration; LOW does correctly flag the worst errors.

### 4.6 Oracle ensemble per session (best Gemma per session)

| Session | Best Gemma | Best GAD-7 MAE_items |
|---|---|---:|
| S01 | Gemma 3 27B | 1.000 |
| S03 | Gemma 4 31B | 0.714 |
| S04 | Gemma 4 26B MoE | 0.429 |
| S05 | Gemma 4 26B MoE | 0.429 |
| S06 | Gemma 3 27B | 0.429 |
| S07 | Gemma 4 26B MoE | 0.286 |
| S09 | Gemma 4 26B MoE | 1.000 |
| S12 | Gemma 3 27B | 0.714 |
| S15 | Gemma 3 27B | 1.000 |
| S16 | Gemma 3 27B | 0.571 |
| **Oracle mean** | — | **0.657** |

Oracle headroom = +0.086 over the best single Gemma (4 26B MoE, 0.743), within 0.075 of the competition Gemma baseline (0.582).

---

## 5. Other post-hoc Task 1 corrections (Layer 2 — principled rule-based)

Source: [P_gad7_corrections.csv](../analysis/MentalRiskES_test/outputs/P_gad7_corrections.csv); rules P1–P5 derived from trial diagnosis. Applied to Run 2 GAD-7 R1–30.

| Correction | GAD-7 MAE_items | Δ vs baseline (0.971) |
|---|---:|---:|
| baseline | 0.971 | — |
| **P1** (cap item 2 at 2) | 0.957 | −0.014 |
| P1 + P3 | 0.957 | −0.014 |
| P1 + P4 | 0.957 | −0.014 |
| **INV_low** (add 1 if total < 5; test-derived) | 0.957 | −0.014 |
| P2 (subtract 1 if total ≥ 12) | 1.000 | +0.029 |
| P1 + P2 | 1.000 | +0.029 |

Trial-derived downward corrections fail on the test cohort because the test population systematically *under*-shoots, not over-shoots. Cohort-aware correction is the lesson.

---

## 6. Source-file map

| Result block | File |
|---|---|
| Trial ablation summary (A0–A5 × T0/T2/T3) | [runs/mentalriskes_ablation/ablation_summary.json](../runs/mentalriskes_ablation/ablation_summary.json) |
| Trial calibration report (narrative) | [output/mentalriskes/trial_calibration_report.md](../output/mentalriskes/trial_calibration_report.md) |
| Simulated ablation (6 persona summaries) | [runs/mentalriskes_simulated_ablation/sim_*/ablation_summary_*.json](../runs/mentalriskes_simulated_ablation/) |
| Simulated ablation aggregate | [runs/mentalriskes_simulated_ablation/simulated_ablation_report.txt](../runs/mentalriskes_simulated_ablation/simulated_ablation_report.txt) |
| Submitted R1–30 predictions | [output/mentalriskes/predictions/round{N}_run{R}.json](../output/mentalriskes/predictions/) |
| Full-replay predictions (R1–82) | [output/mentalriskes_test_replay/predictions/](../output/mentalriskes_test_replay/predictions/) |
| Per-run summary (R1–30) | [analysis/MentalRiskES_test/outputs/A_per_run_summary.csv](../analysis/MentalRiskES_test/outputs/A_per_run_summary.csv) |
| Run comparison (R1–30 vs replay) | [analysis/MentalRiskES_test/outputs/W_per_run_aggregate.csv](../analysis/MentalRiskES_test/outputs/W_per_run_aggregate.csv) |
| Per-instrument leaderboard ranks | [Q_team_metric_ranks.csv](../analysis/MentalRiskES_test/outputs/Q_team_metric_ranks.csv), [Q_balanced_rank.csv](../analysis/MentalRiskES_test/outputs/Q_balanced_rank.csv) |
| Official leaderboard XLSX | [data/MentalRiskES-2026/MentalRiskES2026 - Results.xlsx](../data/MentalRiskES-2026/MentalRiskES2026%20-%20Results.xlsx) |
| Gemma GAD-7 standalone | [analysis/MentalRiskES_test/outputs/W_gemma_summary.csv](../analysis/MentalRiskES_test/outputs/W_gemma_summary.csv) |
| Gemma GAD-7 hybrid (×30) | [analysis/MentalRiskES_test/outputs/W_gemma_hybrid.csv](../analysis/MentalRiskES_test/outputs/W_gemma_hybrid.csv) |
| Gemma per-item | [W_gemma_per_item.csv](../analysis/MentalRiskES_test/outputs/W_gemma_per_item.csv), [W_gemma_per_item_pivot.csv](../analysis/MentalRiskES_test/outputs/W_gemma_per_item_pivot.csv) |
| Gemma confidence calibration | [W_gemma_confidence.csv](../analysis/MentalRiskES_test/outputs/W_gemma_confidence.csv) |
| Gemma oracle-per-session | [W_gemma_best_per_session.csv](../analysis/MentalRiskES_test/outputs/W_gemma_best_per_session.csv) |
| Cross-cohort Gemma v1 vs v2 | [W_t1_cross_cohort.csv](../analysis/MentalRiskES_test/outputs/W_t1_cross_cohort.csv) |
| Principled GAD-7 corrections | [P_gad7_corrections.csv](../analysis/MentalRiskES_test/outputs/P_gad7_corrections.csv) |
| Item-level error decomposition | [B_item_level_errors.csv](../analysis/MentalRiskES_test/outputs/B_item_level_errors.csv), [B_compact10_subscale_run2.csv](../analysis/MentalRiskES_test/outputs/B_compact10_subscale_run2.csv) |
| Confusion matrices | [C_run2_GAD-7_confusion.csv](../analysis/MentalRiskES_test/outputs/C_run2_GAD-7_confusion.csv), [C_run2_PHQ-9_confusion.csv](../analysis/MentalRiskES_test/outputs/C_run2_PHQ-9_confusion.csv) |
| LLM configuration appendix | [docs/mentalriskes_llm_configuration.md](mentalriskes_llm_configuration.md) |
| System description (full) | [docs/mentalriskes_task1_solution_description.md](mentalriskes_task1_solution_description.md) |
| Master post-hoc narrative | [analysis/MentalRiskES_test/SUMMARY.md](../analysis/MentalRiskES_test/SUMMARY.md) |
