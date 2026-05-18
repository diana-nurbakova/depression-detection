# MentalRiskES 2026 — Task 2 Comprehensive Results

**Task:** therapist-response selection — pick the most clinically appropriate continuation among 3 candidates per round of a Spanish ACT therapeutic conversation.
**Team:** INSALyon.
**Pipeline source:** [src/mentalriskes/task2/](../src/mentalriskes/task2/) — submitted state-tracking ACT-FM pipeline.
**Date compiled:** 2026-05-14.

---

## 0. Configuration legend

### 0.1 Pipeline structures

| Code | Definition | LLM calls per round |
|---|---|---|
| **A** | Single-prompt, score all 3 options + select in one call | 1 |
| **B** | 2-step: state-tracker update → evaluator/selector (each one LLM call) | 2 |
| **B+** | 2.5-step: state update → patient-needs *characterization* → evaluator/selector | 3 |
| **ENS** | B + B+ ensemble (agreement → shared answer; disagreement → consistency tag tiebreaker; default B+) | 5 |

### 0.2 Evaluation framings (how the evaluator reasons about the 3 options)

| Code | Definition |
|---|---|
| **FUNC** | Functional analysis throughout — evaluates options by their therapeutic *function*, not surface patterns |
| **HYB** | Functional analysis for elimination + ToM reasoning for final selection |
| **TOM-B** | Full ToM (beliefs) reasoning for evaluation, elimination, and selection with structured ACT-FM criteria |
| **TOM-C** | Full ToM (compassion) reasoning chain — the most cognitively demanding framing |

### 0.3 Other axes

| Axis | Values |
|---|---|
| Prompt language | ES (Spanish), EN (English) |
| Lookback window | W1, W3, W5 — number of prior round messages included as context |
| Option ordering | FIX (1, 2, 3 as presented), PERM (3 permutations + majority vote), PERM6 (6 permutations / S3) |
| Calibration | (none), CAL = experiential tiebreaker calibration added at end |
| Model | Llama 3.3 70B Instruct Turbo (TogetherAI), Llama 3.3 70B (DeepInfra, submission/replay), Claude Sonnet 4 (API), Gemma 3 27B, Gemma 4 31B, Gemma 4 26B MoE (OpenRouter) |

### 0.4 Run / configuration naming

`<Pipeline>_<Model>_<lang>_<Framing>_<Ordering>_W<window>[_CAL]`. The ablation logged this string into the run filename (e.g. `B+_Llama-3.3-70B-Instruct-Turbo_es_HYB_FIX_W3.jsonl`).

### 0.5 Bare-LLM post-hoc modes

| Code | Description |
|---|---|
| **S** | Bare 100-token prompt: "which of these three best continues the therapeutic conversation?" |
| **S2** | S + 4 anti-bias guardrails (don't prefer longer, don't always pick opt 2, simplest validation OK, "what a skilled therapist would actually say") |
| **S3** | Permutation averaging — run S over 6 candidate orderings, majority-vote on the original numbering |
| **S4** | Pairwise Condorcet — 3 head-to-head A-vs-B / B-vs-C / A-vs-C comparisons |
| **R2** | Full 3-way ranking — pick rank-1, also log where gold lands |

### 0.6 Cohorts

| Cohort | Sessions | Rounds | Gold distribution | Notes |
|---|---|---|---|---|
| Trial | 1 (Miguel) | 18 (rounds 1–18; round 19 has no next-round therapist_response) | opt1 = 5, opt2 = 4, opt3 = 9 (50 % option 3 → majority baseline = 0.500) | Expert-curated near-miss distractors |
| Simulated | 7 personas (incl. 3-round sim_anx_social_99) | 87 patient-rounds (mostly 14 per session) | Roughly uniform | Synthetic distractors with explicit error types |
| Test (R1–30, submitted) | 10 (S01, S03–S07, S09, S12, S15, S16) | 300 patient-rounds | opt1 = 101, opt2 = 95, opt3 = 104 (~uniform) | What the leaderboard scored |
| Test (R1–82, full replay) | 10 | 568 patient-rounds | opt1 35.7 % / opt2 32.7 % / opt3 31.5 % | Post-submission replay |

---

## 1. Trial cohort — full ablation (n = 18 rounds, 1 patient)

Source: [output/mentalriskes_task2/ablation/ablation_report.md](../output/mentalriskes_task2/ablation/ablation_report.md) (v1.2, 12 configs) + [v2_results_summary.json](../output/mentalriskes_task2/v2_results_summary.json) (v2.0 ablation, 12 configs). Per-config round-by-round predictions in [output/mentalriskes_task2/ablation/B*_*_*.jsonl](../output/mentalriskes_task2/ablation/).

**Baselines:** random = 33.3 %; majority-class (always option 3) = 50.0 %.

### 1.1 v1.2 ablation — 12 configurations (Llama-3.3-70B-Instruct-Turbo unless noted)

| Rank | Config ID | Pipeline | Model | Lang | Framing | Window | Ordering | Accuracy | Cohen's κ | 95 % CI | Time |
|---|:-:|:-:|---|:-:|:-:|:-:|:-:|---:|---:|---|---:|
| 1 | **C8** | **B+** | Llama 3.3 70B | ES | FUNC | W3 | FIX | **55.6 %** (10/18) | **0.345** | [33.3 %, 77.8 %] | 7.8 min |
| 1 | **C5b** | **B+** | Llama 3.3 70B | ES | HYB | W3 | FIX | **55.6 %** (10/18) | **0.351** | [33.3 %, 77.8 %] | 8.2 min |
| 3 | C8-CAL | B+ | Llama 3.3 70B | ES | FUNC + CAL | W3 | FIX | 50.0 % (9/18) | 0.267 | [27.8 %, 72.2 %] | 8.1 min |
| 4 | C8-EN | B+ | Llama 3.3 70B | EN | FUNC | W3 | FIX | 44.4 % (8/18) | 0.174 | [22.2 %, 66.7 %] | 8.9 min |
| 5 | C1 | B | Llama 3.3 70B | ES | FUNC | W3 | FIX | 38.9 % (7/18) | 0.108 | [16.7 %, 61.1 %] | 4.6 min |
| 6 | C3 | B | Llama 3.3 70B | EN | FUNC | W3 | FIX | 38.9 % (7/18) | 0.104 | [16.7 %, 61.1 %] | 8.8 min |
| 7 | C11 | B | Llama 3.3 70B | ES | FUNC | W3 | PERM | 38.9 % (7/18) | 0.048 | [16.7 %, 61.1 %] | 12.2 min |
| 8 | C10 | B | Llama 3.3 70B | ES | FUNC | W5 | FIX | 33.3 % (6/18) | 0.027 | [11.1 %, 55.6 %] | 16.7 min |
| 9 | C6 | B | Llama 3.3 70B | ES | TOM-B | W3 | FIX | 33.3 % (6/18) | 0.014 | [11.1 %, 55.6 %] | 6.2 min |
| 10 | C5 | B | Llama 3.3 70B | ES | HYB | W3 | FIX | 27.8 % (5/18) | −0.054 | [11.1 %, 50.0 %] | 6.4 min |
| 11 | C9 | B | Llama 3.3 70B | ES | FUNC | W1 | FIX | 27.8 % (5/18) | −0.109 | [11.1 %, 50.0 %] | 5.2 min |
| 12 | C7 | B | Llama 3.3 70B | ES | TOM-C | W3 | FIX | 22.2 % (4/18) | −0.086 | [5.6 %, 44.4 %] | 6.3 min |

### 1.2 v2.0 ablation (re-run with prompt v2.0 — Trial + Simulated)

Source: [output/mentalriskes_task2/v2_results_summary.json](../output/mentalriskes_task2/v2_results_summary.json).

| Config ID | Pipeline | Model | Lang | Framing | Window | Ordering | Trial Acc | Trial κ | Trial 95 % CI | Sim Acc | Sim κ | Sim 95 % CI |
|:-:|:-:|---|:-:|:-:|:-:|:-:|---:|---:|---|---:|---:|---|
| C1 | B | Llama 3.3 70B | ES | FUNC | W3 | FIX | 0.500 | 0.264 | [0.278, 0.722] | **0.920** | **0.879** | [0.862, 0.977] |
| C2 | B | Claude Sonnet 4 | ES | FUNC | W3 | FIX | 0.444 | 0.174 | [0.222, 0.667] | 0.908 | 0.862 | [0.839, 0.966] |
| C3 | B | Llama 3.3 70B | EN | FUNC | W3 | FIX | 0.500 | 0.190 | [0.278, 0.722] | 0.908 | 0.861 | [0.839, 0.966] |
| C4 | B | Claude Sonnet 4 | EN | FUNC | W3 | FIX | 0.500 | 0.221 | [0.278, 0.722] | 0.908 | 0.861 | [0.839, 0.966] |
| C5 | B | Llama 3.3 70B | ES | HYB | W3 | FIX | 0.389 | 0.034 | [0.167, 0.611] | 0.897 | 0.844 | [0.828, 0.954] |
| C6 | B | Llama 3.3 70B | ES | TOM-B | W3 | FIX | 0.389 | 0.116 | [0.167, 0.611] | 0.667 | 0.491 | [0.563, 0.759] |
| C7 | B | Llama 3.3 70B | ES | TOM-C | W3 | FIX | 0.444 | 0.200 | [0.222, 0.667] | 0.678 | 0.508 | [0.586, 0.770] |
| C8 | B+ | Llama 3.3 70B | ES | FUNC | W3 | FIX | 0.389 | 0.043 | [0.167, 0.611] | 0.851 | 0.776 | [0.770, 0.920] |
| C9 | B | Llama 3.3 70B | ES | FUNC | W1 | FIX | 0.389 | 0.048 | [0.167, 0.611] | 0.897 | 0.845 | [0.828, 0.954] |
| C10 | B | Llama 3.3 70B | ES | FUNC | W5 | FIX | 0.389 | 0.000 | [0.167, 0.611] | 0.885 | 0.828 | [0.816, 0.954] |
| **C11** | B | Llama 3.3 70B | ES | FUNC | W3 | **PERM** | 0.389 | 0.083 | [0.167, 0.611] | **0.943** | **0.914** | [0.885, 0.989] |
| ENS | B + B+ | Llama 3.3 70B | ES | FUNC | W3 | FIX | 0.500 | 0.236 | [0.278, 0.722] | 0.920 | 0.879 | [0.862, 0.977] |

**v1.2 → v2.0 prompt updates flipped the ranking on trial.** B+/FUNC (C8) dropped from 55.6 % → 38.9 % as the v2.0 evaluator prompt reduced the characterization advantage. The v2.0 winner on trial is B/FUNC/ES with permutation voting on simulated (94.3 %), or any B/FUNC config (50.0 %) on trial.

### 1.3 v1.2 round-by-round prediction matrix (gold + 9 representative configs)

Source: [ablation_report.md §2.2](../output/mentalriskes_task2/ablation/ablation_report.md). 18 rounds; rows are gold + selected configs.

| Round | Gold | B+/FUNC | FUNC/ES | FUNC/EN | HYB | TOM-B | TOM-C | W1 | W5 | PERM |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | **2** | 2 | 1 | 3 | 1 | 3 | 1 | 1 | 3 | 3 |
| 2 | **3** | 3 | 3 | 3 | 3 | 1 | 1 | 3 | 3 | 3 |
| 3 | **3** | 2 | 2 | 2 | 2 | 1 | 1 | 2 | 2 | 2 |
| 4 | **3** | 1 | 3 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| 5 | **2** | 2 | 2 | 1 | 2 | 1 | 1 | 2 | 2 | 2 |
| 6 | **1** | 3 | 2 | 2 | 2 | 1 | 1 | 3 | 2 | 3 |
| 7 | **3** | 3 | 3 | 3 | 3 | 3 | 1 | 3 | 3 | 3 |
| 8 | **3** | 3 | 2 | 3 | 2 | 3 | 1 | 2 | 2 | 2 |
| 9 | **3** | 2 | 2 | 2 | 2 | 1 | 2 | 2 | 2 | 2 |
| 10 | **2** | 2 | 2 | 2 | 2 | 1 | 1 | 2 | 2 | 2 |
| 11 | **1** | 1 | 1 | 1 | 3 | 1 | 1 | 3 | 1 | 1 |
| 12 | **3** | 1 | 2 | 2 | 2 | 1 | 1 | 1 | 2 | 3 |
| 13 | **1** | 1 | 2 | 2 | 2 | 2 | 2 | 3 | 2 | 3 |
| 14 | **3** | 1 | 1 | 2 | 1 | 1 | 1 | 1 | 1 | 2 |
| 15 | **1** | 3 | 3 | 1 | 3 | 1 | 1 | 3 | 3 | 3 |
| 16 | **1** | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| 17 | **3** | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| 18 | **2** | 2 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 |
| **Correct** | | **10** | **7** | **7** | **5** | **6** | **4** | **5** | **6** | **7** |

Universal failure modes (rounds 3, 9, 17) — every config picks against the gold-3 organizer preference for gentler metaphorical interventions.

### 1.4 v1.2 per-phase accuracy (best config B+/FUNC, 10/18)

| Phase | Rounds | Gold | B+/FUNC predictions | Correct |
|---|---|---|---|:-:|
| Crisis / engagement | 1 | 2 | 2 | 1/1 (100 %) |
| Committed action | 2–3 | 3, 3 | 3, 2 | 1/2 (50 %) |
| Acceptance / defusion | 4–5 | 3, 2 | 1, 2 | 1/2 (50 %) |
| Defusion deepening | 6–8 | 1, 3, 3 | 3, 3, 3 | 2/3 (67 %) |
| Behavioral activation | 9–12 | 3, 2, 1, 3 | 2, 2, 1, 1 | 2/4 (50 %) |
| Integration | 13–15 | 1, 3, 1 | 1, 1, 3 | 1/3 (33 %) |
| Self-as-context | 16–17 | 1, 3 | 1, 1 | 1/2 (50 %) |
| Closing | 18 | 2 | 2 | 1/1 (100 %) |

---

## 2. Simulated cohort — per-config detail (n = 87 rounds, 7 personas)

Source: [output/mentalriskes_task2/simulated_ablation/ablation_summary_simulated.json](../output/mentalriskes_task2/simulated_ablation/ablation_summary_simulated.json) + v2_results_summary.json. Per-config per-persona JSONL in [output/mentalriskes_task2/simulated_ablation/<config>/sim_*.jsonl](../output/mentalriskes_task2/simulated_ablation/).

### 2.1 Aggregate by configuration (same 12-config grid as Trial in §1.2)

| Config | Pipeline | Model | Lang | Framing | Ordering | Window | n rounds | Sim Acc | Sim κ | Sim 95 % CI | Elapsed |
|:-:|:-:|---|:-:|:-:|:-:|:-:|:-:|---:|---:|---|---:|
| C1 | B | Llama 3.3 70B | ES | FUNC | FIX | W3 | 87 | 0.920 | 0.879 | [0.862, 0.977] | 729 s |
| C2 | B | Claude Sonnet 4 | ES | FUNC | FIX | W3 | 87 | 0.908 | 0.862 | [0.839, 0.966] | — |
| C3 | B | Llama 3.3 70B | EN | FUNC | FIX | W3 | 87 | 0.908 | 0.861 | [0.839, 0.966] | — |
| C4 | B | Claude Sonnet 4 | EN | FUNC | FIX | W3 | 87 | 0.908 | 0.861 | [0.839, 0.966] | — |
| C5 | B | Llama 3.3 70B | ES | HYB | FIX | W3 | 87 | 0.897 | 0.844 | [0.828, 0.954] | — |
| C6 | B | Llama 3.3 70B | ES | TOM-B | FIX | W3 | 87 | 0.667 | 0.491 | [0.563, 0.759] | — |
| C7 | B | Llama 3.3 70B | ES | TOM-C | FIX | W3 | 87 | 0.678 | 0.508 | [0.586, 0.770] | — |
| C8 | B+ | Llama 3.3 70B | ES | FUNC | FIX | W3 | 87 | 0.851 | 0.776 | [0.770, 0.920] | 1002 s |
| C9 | B | Llama 3.3 70B | ES | FUNC | FIX | W1 | 87 | 0.897 | 0.845 | [0.828, 0.954] | — |
| C10 | B | Llama 3.3 70B | ES | FUNC | FIX | W5 | 87 | 0.885 | 0.828 | [0.816, 0.954] | — |
| **C11** | B | Llama 3.3 70B | ES | FUNC | **PERM** | W3 | 87 | **0.943** | **0.914** | [0.885, 0.989] | — |
| ENS | B + B+ | Llama 3.3 70B | ES | FUNC | FIX | W3 | 87 | 0.920 | 0.879 | [0.862, 0.977] | — |

### 2.2 Per-session detail for the v2.0 baseline (C1: B/FUNC/ES/W3 FIX) and B+/FUNC (C8)

| Session | Presentation | Rounds | C1 (B) Acc | C8 (B+) Acc |
|---|---|:-:|---:|---:|
| sim_anx_academic_42 | Academic anxiety, perfectionist | 14 | 92.9 % | 78.6 % |
| sim_anx_health_46 | Health anxiety, somatic | 14 | 100.0 % | 85.7 % |
| sim_anx_social_44 | Social anxiety, avoidant | 14 | 100.0 % | 100.0 % |
| sim_anx_social_99 | Social anxiety (short session) | 3 | 100.0 % | 100.0 % |
| sim_dep_burnout_45 | Burnout depression | 14 | 92.9 % | 85.7 % |
| sim_dep_loss_43 | Loss / grief depression | 14 | 85.7 % | 71.4 % |
| sim_dep_mild_47 | Mild depression | 14 | 78.6 % | 85.7 % |
| **Aggregate** | | **87** | **92.0 %** | **85.1 %** |

B+ *hurts* on simulated (−6.9 pp vs B) — the characterization step adds noise when distractors are constructed with explicit error types. B+'s value is restricted to the trial's expert-curated near-miss design.

---

## 3. Test cohort — official leaderboard + replay + bare-LLM post-hoc

### 3.1 Official leaderboard (all 3 INSALyon submitted runs)

Source: [data/MentalRiskES-2026/MentalRiskES2026 - Results.xlsx](../data/MentalRiskES-2026/MentalRiskES2026%20-%20Results.xlsx) (Task2 sheet). Truncated at round 30 due to `--max-rounds=30` bug, but Scenario A scoring (only-submitted-rounds-count) was confirmed.

| Run | Config | Accuracy | Macro Precision | Macro Recall | Macro F1 | Error Recovery Rate | Combined rank |
|:-:|---|---:|---:|---:|---:|---:|:-:|
| Run 0 | B / FUNC / ES / W3 PERM | 0.2100 | 0.2098 | 0.2102 | 0.2099 | 0.2130 | 31 |
| Run 1 | B / FUNC / ES / W3 FIX | 0.2367 | 0.2421 | 0.2396 | 0.2336 | 0.2217 | 25 |
| **Run 2** | B+ / HYB / ES / W3 FIX | **0.2467** | 0.2535 | 0.2482 | 0.2432 | 0.2557 | **24** |

Top of leaderboard reference: NLP Innovators Run 1 = 0.3926 (rank 1); VerbaNex AI Run 0 = 0.3856 (rank 2); **BASELINE-Random = 0.3627** (rank 3 — our runs are below random); debju Run 2 = 0.3592 (rank 4).

### 3.2 Full R1–82 replay (same Llama 3.3 70B pipeline, DeepInfra)

Source: SUMMARY.md §4.5.2 + W_t2_round_decomposition.csv. Run 0 replay didn't complete (perm voting × 3 calls / round was in flight).

| Run | Config | Submitted Acc (R1–30) | Replay R1–30 | Replay full (R1–82) | Δ submitted → full |
|:-:|---|---:|---:|---:|---:|
| Run 0 | B / FUNC / ES / W3 PERM | 0.210 | — | — | — |
| Run 1 | B / FUNC / ES / W3 FIX | 0.237 | 0.200 | 0.220 | −0.017 |
| Run 2 | B+ / HYB / ES / W3 FIX | 0.247 | 0.227 | **0.255** | +0.008 |

Per-tercile replay accuracy:

| Run | Early (R1–27) | Mid (R28–54) | Late (R55–82) |
|:-:|---:|---:|---:|
| Run 1 | 0.204 | 0.220 | **0.288** |
| Run 2 | 0.230 | **0.280** | 0.273 |

Late-round accuracy is *higher* than early-round, rejecting the pre-submission "state-tracker degradation" hypothesis.

### 3.3 Run-by-run confusion matrices (R1–30 leaderboard slice)

Run 0 (FUNC PERM):
```
gold \ pred   1    2    3
1             23   40   38
2             34   20   41
3             45   39   20
```

Run 1 (FUNC FIX):
```
gold \ pred   1    2    3
1             21   49   31
2             23   33   39
3             26   61   17
```

Run 2 (HYB B+ FIX):
```
gold \ pred   1    2    3
1             17   47   37
2             22   31   42
3             21   57   26
```

Position-bias check (vs uniform):

| Run | Pred opt1 / opt2 / opt3 | χ² vs uniform | p-value |
|:-:|---|---:|---|
| 0 | 102 / 99 / 99 | 0.06 | 0.97 (uniform) |
| 1 | 70 / **143** / 87 | 29.18 | < 10⁻⁶ |
| 2 | 60 / **135** / 105 | 28.50 | < 10⁻⁶ |

Run 0's permutation voting eliminates position bias entirely; Run 1/Run 2 over-predict option 2 at ~45 %.

### 3.4 Bare-LLM post-hoc (Gemma + Llama × S/S2/S3/S4/R2; test = 568 rounds)

Source: [analysis/MentalRiskES_test/outputs/W_t2_bare_summary.csv](../analysis/MentalRiskES_test/outputs/W_t2_bare_summary.csv).

| Model | Mode | n | Accuracy | Macro F1 | F1 cls1 | F1 cls2 | F1 cls3 | Pred dist 1 / 2 / 3 (%) | χ² vs uniform | p-uniform |
|---|:-:|:-:|---:|---:|---:|---:|---:|:-:|---:|---|
| **Gemma 4 31B** | **S2** | 568 | **0.4701** | **0.454** | 0.549 | 0.401 | 0.411 | 53.3 / 22.5 / 24.1 | 102.6 | < 10⁻²² |
| Gemma 4 31B | S | 568 | 0.4120 | 0.402 | 0.478 | 0.372 | 0.356 | 48.2 / 26.9 / 24.8 | 57.2 | < 10⁻¹² |
| Gemma 4 31B | S3 (×6 perms) | 567 | 0.4004 | 0.399 | 0.434 | 0.393 | 0.369 | 36.7 / 33.7 / 29.6 | 4.27 | 0.119 (uniform) |
| Gemma 4 31B | S4 (Condorcet) | 559 | 0.3542 | 0.353 | 0.378 | 0.348 | 0.333 | 35.6 / 35.2 / 29.2 | 4.39 | 0.111 (uniform) |
| Gemma 3 27B | S | 568 | 0.2905 | 0.263 | 0.414 | 0.151 | 0.223 | 56.2 / 23.2 / 20.6 | 133.8 | < 10⁻²⁹ |
| Gemma 3 27B | R2 (rank-1 pick) | 568 | 0.2870 | 0.225 | 0.166 | 0.410 | 0.100 | 15.3 / 77.3 / 7.4 | 499.2 | < 10⁻¹⁰⁹ |
| Llama 3.3 70B | S | 568 | 0.2570 | 0.234 | 0.368 | 0.164 | 0.169 | 54.2 / 29.4 / 16.4 | 126.0 | < 10⁻²⁷ |

### 3.5 Bare-LLM deltas vs reference systems

Source: [W_t2_bare_per_run.csv](../analysis/MentalRiskES_test/outputs/W_t2_bare_per_run.csv).

| Bare config | vs Submitted Run 0 (0.210) | vs Submitted Run 1 (0.237) | vs Submitted Run 2 (0.247) | vs Replay Run 2 full (0.255) | vs BASELINE-Random (0.363) | vs Top team (0.393) |
|---|---:|---:|---:|---:|---:|---:|
| **Gemma 4 31B S2** | **+26.0 pp** | **+23.3 pp** | **+22.3 pp** | **+21.5 pp** | **+10.7 pp** | **+7.7 pp** |
| Gemma 4 31B S | +20.2 | +17.5 | +16.5 | +15.7 | +4.9 | +1.9 |
| Gemma 4 31B S3 | +19.0 | +16.3 | +15.3 | +14.5 | +3.7 | +0.7 |
| Gemma 4 31B S4 | +14.4 | +11.7 | +10.7 | +9.9 | −0.9 | −3.9 |
| Gemma 3 27B S | +8.0 | +5.3 | +4.3 | +3.5 | −7.3 | −10.3 |
| Gemma 3 27B R2 | +7.7 | +5.0 | +4.0 | +3.2 | −7.6 | −10.6 |
| Llama 3.3 70B S | +4.7 | +2.0 | +1.0 | +0.2 | −10.6 | −13.6 |

### 3.6 Bare-LLM tercile breakdown (Gemma 4 31B)

| Mode | Early (R1–27, n=270) | Mid (R28–54, n=232) | Late (R55–82, n=66) |
|:-:|---:|---:|---:|
| S | 0.389 | 0.457 | 0.348 |
| **S2** | 0.407 | **0.569** | 0.379 |
| S3 | 0.389 | 0.442 | 0.303 |
| S4 | 0.356 | 0.353 | 0.354 |

Mid-conversation S2 accuracy peaks at **0.569** — more than half of mid-session response selections correct.

### 3.7 R2 ranking inversion test (Gemma 3 27B)

| Where gold lands in 3-way ranking | Share |
|---|---:|
| Rank 1 (model's top pick) | 28.7 % |
| Rank 2 | 37.1 % |
| Rank 3 (model's bottom pick) | 34.2 % |

Roughly uniform → the "valid but inverted" hypothesis is **rejected**. The model has weak signal, not anti-correlated signal.

---

## 4. Cross-cohort table — same systems, three corpora

Source: [W_t2_cross_cohort.csv](../analysis/MentalRiskES_test/outputs/W_t2_cross_cohort.csv). Submitted-equivalent on trial/simulated = the `HYB B+ FIX W3` ablation entry matching submitted Run 2.

| System | Test (n=568) acc | Trial (n=18) acc | Simulated (n=87) acc | Test pred dist (1/2/3) |
|---|---:|---:|---:|---|
| Submitted Run 2 (R1–30) | 0.247 | — | — | 0.20 / 0.45 / 0.35 |
| Submitted Run 2 replay (R1–82) | 0.255 | — | — | 0.21 / 0.48 / 0.31 |
| Submitted-equivalent (HYB B+ FIX W3) | — | **0.444** (8/18) | 0.897 | — |
| Gemma 4 31B bare (S) | 0.412 | 0.333 | 0.931 | 0.48 / 0.27 / 0.25 |
| **Gemma 4 31B bare (S2)** | **0.470** | **0.444** (8/18) | **0.943** | 0.53 / 0.23 / 0.24 |
| Gemma 4 31B bare (S3) | 0.400 | — | — | 0.37 / 0.34 / 0.30 |
| Gemma 4 31B bare (S4) | 0.354 | — | — | 0.36 / 0.35 / 0.29 |
| Gemma 3 27B bare (S) | 0.290 | — | — | 0.56 / 0.23 / 0.21 |
| Llama 3.3 70B bare (S) | 0.257 | — | — | 0.54 / 0.29 / 0.16 |

**Methodological lesson:** on trial (n=18) S2 ties Submitted; on simulated all systems saturate above 0.90. Only the test corpus reveals the 21.5 pp gap. Pre-submission cohorts under-discriminate at our quality range.

---

## 5. Consensus-failure analysis (9 systems × 299 rounds)

Source: [SUMMARY.md §5.8](../analysis/MentalRiskES_test/SUMMARY.md). Systems: Submitted Run 2 (R1–30 + replay), Gemma 4 31B {S, S2, S3, S4, R2}, Gemma 3 27B bare, Llama 3.3 70B bare.

| Gold class | n | All-wrong rate | All-correct rate | Mean correct systems / 9 |
|:-:|:-:|---:|---:|---:|
| 1 | 101 | 17.8 % | 3.0 % | 3.14 |
| 2 | 94 | 21.3 % | 1.1 % | 2.47 |
| **3** | **104** | **38.5 %** | **0.0 %** | **1.71** |
| ALL | 299 | 26.1 % | 1.3 % | 2.43 |

**Gold-3 is categorically the hardest class.** Zero rounds had every system correct when gold = option 3. The 26 % all-wrong rate suggests an ~26 % task floor on Task 2.

---

## 6. Submitted Run 2 vs S2 — head-to-head disagreement (R1–30 inner join, 300 cases)

Source: [outputs/qualitative_T2_submitted_vs_s2.md](../analysis/MentalRiskES_test/outputs/qualitative_T2_submitted_vs_s2.md) + W_t2_submitted_vs_s2_summary.csv.

| Bucket | Count | Share |
|---|:-:|---:|
| Both correct | 34 | 11.3 % |
| **S2 wins** (S2 right, Submitted wrong) | **91** | **30.3 %** |
| Submitted wins (Submitted right, S2 wrong) | 40 | 13.3 % |
| Both wrong, same answer | 80 | 26.7 % |
| Both wrong, different answers | 55 | 18.3 % |

Per-class accuracy:

| Gold class | n | Submitted acc | S2 acc | Δ |
|:-:|:-:|---:|---:|---:|
| 1 | 101 | 0.317 | 0.396 | +0.079 |
| 2 | 95 | 0.222 | 0.453 | **+0.231** |
| 3 | 104 | 0.198 | 0.404 | +0.206 |

S2 wins **2.3× as often as Submitted wins**. Largest absolute gain on gold = 2 (+23 pp).

---

## 7. Dominant Task 2 error patterns (Run 2 confusion)

Errors by gold × pred cell on Run 2 (R1–30 leaderboard slice):

| gold | pred | count | share of errors |
|:-:|:-:|:-:|---:|
| 3 | 2 | 57 | **25.2 %** |
| 1 | 2 | 47 | 20.8 % |
| 2 | 3 | 42 | 18.6 % |
| 1 | 3 | 37 | 16.4 % |
| 2 | 1 | 22 | 9.7 % |
| 3 | 1 | 21 | 9.3 % |

"Always pick option 2" failure mode covers 46 % of all errors (gold=3→pred=2 + gold=1→pred=2). When Run 1/Run 2 are wrong, the chosen response is on average +12 to +15 words longer than gold (75 % of errors are longer-than-gold).

---

## 8. Source-file map

| Result block | File |
|---|---|
| Trial v1.2 ablation report (narrative + tables) | [output/mentalriskes_task2/ablation/ablation_report.md](../output/mentalriskes_task2/ablation/ablation_report.md) |
| Trial v1.2 ablation summary (raw) | [output/mentalriskes_task2/ablation/ablation_summary.json](../output/mentalriskes_task2/ablation/ablation_summary.json) |
| Trial v1.2 per-config JSONL × 23 | [output/mentalriskes_task2/ablation/B*_*_*.jsonl](../output/mentalriskes_task2/ablation/) |
| v2.0 ablation summary (trial + simulated) | [output/mentalriskes_task2/v2_results_summary.json](../output/mentalriskes_task2/v2_results_summary.json) |
| Simulated ablation summary | [output/mentalriskes_task2/simulated_ablation/ablation_summary_simulated.json](../output/mentalriskes_task2/simulated_ablation/ablation_summary_simulated.json) |
| Simulated per-config per-persona JSONL | [output/mentalriskes_task2/simulated_ablation/<config>/sim_*.jsonl](../output/mentalriskes_task2/simulated_ablation/) |
| Ensemble (trial + simulated) | [output/mentalriskes_task2/ensemble/](../output/mentalriskes_task2/ensemble/) |
| Submitted R1–30 predictions | [output/mentalriskes_task2/server_submissions/](../output/mentalriskes_task2/) |
| Full replay predictions (R1–82) | [output/mentalriskes_task2_test_replay/server_submissions/](../output/mentalriskes_task2_test_replay/server_submissions/) |
| Official leaderboard XLSX | [data/MentalRiskES-2026/MentalRiskES2026 - Results.xlsx](../data/MentalRiskES-2026/MentalRiskES2026%20-%20Results.xlsx) |
| Submitted-vs-replay aggregate | [analysis/MentalRiskES_test/outputs/W_t2_round_decomposition.csv](../analysis/MentalRiskES_test/outputs/W_t2_round_decomposition.csv) |
| Replay per-tercile | [analysis/MentalRiskES_test/outputs/W_t2_round_tercile.csv](../analysis/MentalRiskES_test/outputs/W_t2_round_tercile.csv) |
| Bare-LLM summary | [analysis/MentalRiskES_test/outputs/W_t2_bare_summary.csv](../analysis/MentalRiskES_test/outputs/W_t2_bare_summary.csv) |
| Bare-LLM deltas vs reference | [W_t2_bare_per_run.csv](../analysis/MentalRiskES_test/outputs/W_t2_bare_per_run.csv) |
| Bare-LLM confusion matrices | [W_t2_bare_confusion.csv](../analysis/MentalRiskES_test/outputs/W_t2_bare_confusion.csv) |
| Bare-LLM tercile breakdown | [W_t2_bare_tercile.csv](../analysis/MentalRiskES_test/outputs/W_t2_bare_tercile.csv) |
| Ranking inversion R2 | [W_t2_bare_R2_inversion.csv](../analysis/MentalRiskES_test/outputs/W_t2_bare_R2_inversion.csv) |
| Cross-cohort | [W_t2_cross_cohort.csv](../analysis/MentalRiskES_test/outputs/W_t2_cross_cohort.csv) + [summary md](../analysis/MentalRiskES_test/outputs/W_t2_cross_cohort_summary.md) |
| Consensus failure stats | [W_t2_consensus_failure_stats.csv](../analysis/MentalRiskES_test/outputs/W_t2_consensus_failure_stats.csv) |
| Submitted vs S2 disagreement (300 cases) | [outputs/qualitative_T2_submitted_vs_s2.md](../analysis/MentalRiskES_test/outputs/qualitative_T2_submitted_vs_s2.md) + [W_t2_submitted_vs_s2_summary.csv](../analysis/MentalRiskES_test/outputs/W_t2_submitted_vs_s2_summary.csv) |
| Submitted per-run confusion (R1–30) | [H_run0_confusion.csv](../analysis/MentalRiskES_test/outputs/H_run0_confusion.csv), [H_run1_confusion.csv](../analysis/MentalRiskES_test/outputs/H_run1_confusion.csv), [H_run2_confusion.csv](../analysis/MentalRiskES_test/outputs/H_run2_confusion.csv) |
| Position / length bias evidence | [T_bias_long.csv](../analysis/MentalRiskES_test/outputs/T_bias_long.csv) |
| Predictions long format (R1–30) | [H_task2_predictions_long.csv](../analysis/MentalRiskES_test/outputs/H_task2_predictions_long.csv) |
| LLM configuration appendix | [docs/mentalriskes_llm_configuration.md](mentalriskes_llm_configuration.md) |
| System description (full) | [docs/mentalriskes_task2_solution_description.md](mentalriskes_task2_solution_description.md) |
| Task 2 case-study narrative | [analysis/MentalRiskES_test/REPORT_T2_case_studies.md](../analysis/MentalRiskES_test/REPORT_T2_case_studies.md) |
| Master post-hoc narrative | [analysis/MentalRiskES_test/SUMMARY.md](../analysis/MentalRiskES_test/SUMMARY.md) |
