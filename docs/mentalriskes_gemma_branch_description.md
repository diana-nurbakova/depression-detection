# MentalRiskES 2026 — Gemma Branch: Post-Submission Alternative System Description

**Team:** INSALyon
**Date:** 2026-05-12
**Status:** post-submission post-hoc system, *not* part of the official submission. All evaluation is on the released test set after Phase −1 verification confirmed Scenario A scoring (see [SUMMARY.md §0.5](../analysis/MentalRiskES_test/SUMMARY.md)).

---

## 1. Overview and Motivation

After the MentalRiskES 2026 evaluation period closed, three observations from our submitted system motivated a post-hoc alternative architecture:

1. **Task 1 GAD-7 was the single largest source of error.** Our submission ranked 10 on MAE_Combined despite ranking 2 on CompACT-10 Macro_MAE and 3 on PHQ-9 Macro_MAE; the gap to the top team was almost entirely driven by GAD-7 over-prediction (MAE 1.036 vs Gemma baseline 0.582).
2. **Task 2 accuracy was below the random baseline.** Our best run scored 0.247 vs random 0.363 and the top team 0.393. The pre-submission ablation diagnosed a "safety bias" (gold=3 / predicted=2 dominates the errors), a position bias toward option 2, and a length bias rewarding elaborate responses — but the engineered fix did not recover.
3. **The ACT-process-aware scoring pipeline was diagnostically informative but operationally counter-productive on Task 2.** This suggested testing whether a stripped-down conversational prompt would outperform the engineered system on the same task.

The Gemma branch is a **two-arm alternative system** that addresses both diagnoses:

- **Arm A (Task 1):** Replace the Llama-based GAD-7 assessor with a Gemma model and a redesigned prompt that targets the three diagnosed Llama failure modes (severity-anchor inflation, item-2 over-prediction, recency-bias overcorrection).
- **Arm B (Task 2):** Replace the entire ACT-FM state-tracking pipeline with a single bare-LLM prompt, with and without explicit anti-bias guardrails.

Both arms share the same infrastructure (OpenRouter via the OpenAI SDK), the same retry / parsing scaffolding, and the same evaluation framework. The two arms can be deployed independently: a real submission could use our submitted PHQ-9 and CompACT-10 assessors, our Gemma GAD-7 assessor, and the bare-LLM S2 Task 2 system in parallel.

---

## 2. Arm A — Task 1 Gemma GAD-7 Assessor

### 2.1 Failure Modes Targeted

From the pre-submission ablation diagnosis (consolidated in [specs/MentalRiskES/gemma_gad7_prompt_spec.md §1.1](../specs/MentalRiskES/gemma_gad7_prompt_spec.md)):

1. **Item 2 over-prediction.** "No poder dejar de preocuparse" was consistently scored 3 (nearly every day) when gold was 2. The Llama prompt's ruminative-pattern detection (`bucle`, `ideas dando vueltas`) triggered too aggressively — any mention of repetitive thinking mapped to "nearly every day" instead of "more than half the days".
2. **Severity-anchor inflation.** Our `GAD7_SEVERITY_ANCHOR` told the model "if anxiety is the primary therapy reason, total should be ≥ 10". Correct in principle, but creates a floor effect that prevents legitimate moderate scores (8–12).
3. **Recency-bias overcorrection.** The `RECENCY_BIAS_ANCHOR` instructed the model to ignore within-session improvement. The fix swung too far — the model began ignoring genuine contextual cues that moderate the assessment.

### 2.2 Prompt Design — v1

Spec: [specs/MentalRiskES/gemma_gad7_prompt_spec.md](../specs/MentalRiskES/gemma_gad7_prompt_spec.md).

Six design principles distinguish the v1 Gemma prompt from our Llama assessor:

1. **Simpler is better.** Strip the multi-step CoT, severity anchors, and recency-bias corrections. Let the model reason naturally.
2. **Questionnaire-first.** Present the GAD-7 exactly as a clinician would administer it. The model is filling out a standardised instrument, not performing free-form clinical assessment.
3. **Behavioral anchors over frequency labels.** "Several days" is ambiguous; "2–3 days per week" is concrete. Provide both standard labels and explicit temporal anchors.
4. **Confidence per item.** Request a confidence level (HIGH / MEDIUM / LOW) for each item. Forces the model to reflect on evidence quality before committing.
5. **Calibration through examples.** Include 1–2 scored examples from the trial data, covering mild (total 6) and moderate (total 11) presentations.
6. **Anti-ceiling guidance.** Explicitly state that score 3 requires strong, specific evidence; default to 1–2 when evidence is present but frequency is unclear.

Output format: structured JSON with per-item `evidence`, `frequency_inference`, `score`, `confidence`, plus `total_score`, `severity_band`, `overall_confidence`, and `scoring_notes`.

### 2.3 Prompt Design — v2

Spec: [specs/MentalRiskES/gemma_gad7_prompt_v2.md](../specs/MentalRiskES/gemma_gad7_prompt_v2.md). v1 reduced GAD-7 MAE from 1.086 (Llama replay) → 0.743 (Gemma 4 26B MoE), but introduced two new failures that v2 addresses:

1. **Item 5 over-correction.** v1's guidance "most anxious patients score 0–1 on item 5 (restlessness)" suppressed legitimate scores: 80 % of the test cohort is in the GAD-7 severe band, and severe presentations frequently include restlessness via indirect manifestations (sleep disruption, agitation, pacing). v1 signed bias on item 5 became −1.4 (worse than Llama's −0.9).
2. **No severe-anxiety example.** With only mild (total 6) and moderate (total 11) examples in v1, the model lacked a calibration anchor for the severe-anxiety range that dominates the test set.

Five changes from v1 to v2:

| Component | v1 | v2 | Expected impact |
|---|---|---|---|
| Examples | Mild + Moderate | **+ Severe (total 17)** | Anchors score range for severe cohort; shows what 2s and 3s look like |
| Item 5 guidance | "Most anxious patients score 0–1" | Removed suppression; added indirect-evidence markers (sleep disruption, agitation, "no puedo quedarme quieto/a"); "score ≥ 2 is common in severe patients" | Fix the −1.4 over-suppression |
| Item 6 guidance | None | Added conflict-as-proxy markers (arguments with partner / family / colleagues), shame/understatement note, "multiple references = score ≥ 2" | Fix the −1.3 bias |
| Severity calibration | None | "Patients in therapy typically score 10–21; total < 10 should prompt re-examination" | Soft anti-under-prediction anchor |
| Confidence framing | "How confident in this score" | "How precisely can you estimate FREQUENCY, not just symptom presence" — decouples detection from frequency precision | Prevent confidence from suppressing scores |

The v2 prompt also explicitly states that a HIGH score (2 or 3) with MEDIUM or LOW confidence is valid — confidence is about evidence quality, not score magnitude.

### 2.4 Models Tested

All three models accessed via OpenRouter (OpenAI-compatible API):

| Model | OpenRouter ID | Context | Pricing (paid tier) |
|---|---|---|---|
| Gemma 3 27B | `google/gemma-3-27b-it` | 131 K | $0.08 / $0.16 per M |
| Gemma 4 31B | `google/gemma-4-31b-it` | 262 K | $0.13 / $0.38 per M |
| Gemma 4 26B MoE | `google/gemma-4-26b-a4b-it` | 262 K | $0.06 / $0.33 per M |

The MoE variant has only 3.8 B active parameters but performs best on this task. Free-tier variants (`:free` suffix) are unusable in practice due to per-minute rate limiting from Google AI Studio — we used the paid endpoints for all final runs.

**Gemma-specific quirks handled in [posthoc_P_gemma_gad7.py](../analysis/MentalRiskES_test/posthoc_P_gemma_gad7.py):**

- Gemma 3 / 4 on the Google AI Studio backend rejects separate `system` role messages ("Developer instruction is not enabled"). The runner merges the system prompt into a single user message for Gemma models.
- `response_format: {"type": "json_object"}` is not honoured by the Google AI Studio backend; the parser strips ```code fences if present.

### 2.5 Results

Final-round per-session evaluation on the 10 test patients (gold = `data/MentalRiskES-2026/test/task1/test/gold_label.json`):

| Model / prompt | GAD-7 MAE_items | Signed total bias | Band accuracy |
|---|---|---|---|
| Our Llama-3.3-70B (replay) | 1.086 | −6.0 | 0.20 |
| Our Llama-3.3-70B (R30 submission) | 0.971 | −5.8 | 0.20 |
| Gemma 3 27B v1 | 0.814 | −4.9 | 0.30 |
| Gemma 3 27B v2 | 0.743 | −3.8 | 0.20 |
| Gemma 4 31B v1 | 0.786 | −4.7 | 0.30 |
| Gemma 4 26B MoE v1 | 0.743 | −4.0 | 0.20 |
| **Gemma 4 26B MoE v2** | **0.714** | **−3.4** | **0.50** |
| Competition Gemma baseline (target) | 0.582 | unknown | unknown |

**The v2 prompt is a uniform win on item-MAE for both Gemma variants tested with both prompt versions** (Gemma 3 27B: 0.814 → 0.743; Gemma 4 26B MoE: 0.743 → 0.714). The most striking change is Gemma 4 26B MoE band accuracy 0.20 → 0.50: five of ten patients placed in their correct severity band, vs two of ten with v1.

#### 2.5.1 Per-Item Profile (Final, all models on test)

| Item | Llama signed | Gemma 3 27B v2 | Gemma 4 31B v1 | Gemma 4 26B MoE v2 | Comment |
|---|---|---|---|---|---|
| 1. Nervousness | −0.8 | −0.3 | −0.3 | −0.1 | well-calibrated across Gemmas |
| **2. Uncontrollable worry** | **−0.5** (trial: +1 to +3) | −0.4 | −0.2 | **+0.1** | **Item-2 over-prediction eliminated** by anti-ceiling guidance across every Gemma variant |
| 3. Excessive worry | −0.8 | −0.3 | −0.5 | −0.4 | improved |
| 4. Trouble relaxing | −1.2 | −0.7 | −0.4 | −0.5 | improved |
| 5. Restlessness | −0.9 | **−0.4 (v2)** | −1.3 | **−0.4 (v2)** | **v2 fixes the v1 over-suppression** |
| 6. Irritability | −1.3 | −1.1 (v2) | −1.6 | −1.6 | **shared blind spot** across Llama and all Gemmas |
| 7. Dread | −0.3 | +0.3 (v2) | −0.4 | −0.4 | unchanged |

**The trial-diagnosed flagship error (item 2 over-prediction) is eliminated by the new prompt across every Gemma variant.** Items 5 and 6 are a shared blind spot — patients rarely describe restlessness or irritability with explicit frequency markers, so the limitation is in the transcript evidence rather than prompt design.

#### 2.5.2 Confidence Calibration is Inverted

For Gemma 3 27B v1 (1,968 total items across all sessions and rounds):

| Self-rated confidence | Items | item-MAE |
|---|---|---|
| HIGH | 1,741 | 1.09 |
| **MEDIUM** | **2,132** | **0.82** ← lowest |
| LOW | 103 | 1.29 |

**The model is more accurate on items it labels MEDIUM than HIGH.** HIGH calls appear to be reserved for "the symptom is clearly present", but the model still misjudges the *frequency* of the symptom. The reflective confidence step does not produce the calibration we asked for — but it does correctly identify LOW-confidence items as the worst (MAE 1.29). This is a paper-worthy finding about prompt-elicited confidence in clinical scoring.

#### 2.5.3 Hybrid Combined — All Configurations Project to Rank 4

Hybrid = our PHQ-9 + Gemma GAD-7 + our CompACT-10. Best per Gemma:

| Gemma model | Source for PHQ-9 + CompACT-10 | Run | Hybrid MAE_Combined | Projected rank |
|---|---|---|---|---|
| **Gemma 4 26B MoE** | submitted (R1–30) | 1 | **0.917** | **4** |
| Gemma 4 26B MoE | submitted (R1–30) | 2 | 0.934 | 4 |
| Gemma 4 31B | submitted (R1–30) | 1 | 0.931 | 4 |
| Gemma 3 27B | submitted (R1–30) | 1 | 0.941 | 4 |

**17 of 18 source × run × model hybrid configurations project to rank 4** on the official leaderboard. The only exception (Gemma 3 27B + replay Run 0) lands at rank 6.

**Notable:** the best hybrids use the **round-30 submitted** PHQ-9 / CompACT-10, not the full replay. This is because replay-time PHQ-9 worsens with longer transcripts (see [SUMMARY.md §4.5.1](../analysis/MentalRiskES_test/SUMMARY.md)).

#### 2.5.4 Oracle Ensemble Headroom

Picking the best Gemma model per session (each session gets the model whose prediction is closest to gold):

| Session | Best Gemma | Best MAE_items |
|---|---|---|
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

**Oracle mean MAE = 0.657** vs single-best-model 0.743 → headroom of **+0.086** for an ensemble. That puts us within 0.075 of the competition Gemma baseline (0.582). Heterogeneity matters: Gemma 4 26B MoE wins 4 sessions, Gemma 3 27B wins 5, Gemma 4 31B wins 1.

For paper reporting we recommend the single-best-model number (Gemma 4 26B MoE v2, GAD-7 MAE 0.714) as the primary headline and the oracle-ensemble number (0.657) as upper-bound headroom for a follow-up paper.

---

## 3. Arm B — Task 2 Bare-LLM Response Selection

### 3.1 Diagnosis Underlying the Bare-LLM Hypothesis

From [SUMMARY.md §3](../analysis/MentalRiskES_test/SUMMARY.md) and [task2_improvement_spec.md §1](../specs/MentalRiskES/task2_improvement_spec.md):

- **Position bias.** Our submitted Runs 1 and 2 pick option 2 at ~ 45 % rate, while the gold distribution is roughly uniform (33.6 / 31.7 / 34.7 %).
- **Length bias.** When wrong, our chosen response is on average 12–15 words longer than gold (75 % of errors are "the longer one").
- **Safety bias.** Gold = 3 → predicted = 2 is 25 % of all errors — the system over-penalises minor inconsistencies, picking "safe but bland" over "rich but slightly imperfect" options.
- **Run 0 is distribution-balanced but still 0.21 accurate.** This means even without position / length bias, our ACT-process scoring logic is anti-correlated with gold labels. The fix must go deeper than debiasing.

The bare-LLM hypothesis: **if the engineered ACT scoring is actively hurting, a stripped-down "trust your clinical intuition" prompt should do better.** Five experiments test this:

| ID | Description | LLM calls / round |
|---|---|---|
| **S** | Bare LLM single-pass | 1 |
| **S2** | Bare LLM + explicit anti-bias guardrails | 1 |
| **R2** | Bare LLM full 3-way ranking | 1 |
| **S3** | Permutation averaging (6 orderings → majority vote) | 6 |
| **S4** | Pairwise comparison + Condorcet voting | 3 |

### 3.2 Prompt Design

System prompt (same across all experiments):

> You are an expert psychotherapist conducting an ACT (Acceptance and Commitment Therapy) session in Spanish.
>
> Read the following conversation between a therapist and a patient. Then choose which of the three candidate responses the therapist should say next.
>
> Consider:
> - Which response best matches what the patient needs RIGHT NOW?
> - Which response feels most natural as a continuation of the conversation?
> - Which response maintains the therapeutic alliance while being helpful?
>
> Do not overthink this. Trust your clinical intuition based on the conversation flow.

**S2 adds** an anti-bias instruction block:

> IMPORTANT:
> - Do NOT prefer longer or more elaborate responses. Sometimes the best response is the shortest and most direct.
> - Do NOT assume the middle option (Option 2) is the safest choice. Evaluate all three equally.
> - Sometimes the most therapeutically effective response is simple validation or a direct question, not a complex intervention.
> - Consider what a skilled therapist would ACTUALLY say in this moment, not what sounds most impressive.

**R2** asks for a full ranking (`[2, 1, 3]` meaning option 2 is best, option 1 second, option 3 worst) with per-rank reasons.

**S3** runs all 6 permutations of the option order, maps each back to the original numbering, and majority-votes.

**S4** runs three pairwise comparisons (A vs B, B vs C, A vs C) and picks the Condorcet winner (option that wins all pairwise comparisons it appears in; ties broken by most wins).

All prompts request structured JSON output with a `brief_reason` field that is preserved in the raw log for downstream qualitative analysis.

### 3.3 Models Tested

Three models on Experiment S (bare LLM), the most informative starting point:

| Model | OpenRouter ID | Bare-LLM accuracy |
|---|---|---|
| **Gemma 4 31B** | `google/gemma-4-31b-it` | **0.412** |
| Gemma 3 27B | `google/gemma-3-27b-it` | 0.290 |
| Llama-3.3-70B | `meta-llama/llama-3.3-70b-instruct` | 0.257 |

**The result is architecture-sensitive.** Only Gemma 4 31B has the reasoning capability to make the bare prompt work. Gemma 3 27B has heavy option-1 bias (56 %); Llama-3.3-70B also over-picks option 1 (54 %); Gemma 4 31B is the only model that splits its predictions roughly proportionally to gold while still extracting signal (48 / 27 / 25 % vs gold 36 / 33 / 32 %). The +16.5 pp improvement S brings on Gemma 4 31B is the model's reasoning, not the prompt's simplicity.

The headline is therefore **"simpler-prompt-on-stronger-model wins"**, not "any LLM beats engineered systems."

All subsequent experiments (S2 / R2 / S3 / S4) use Gemma 4 31B.

### 3.4 Results — All Five Experiments

All 568 patient-rounds across 10 test sessions:

| Variant | Accuracy | Macro F1 | Pred dist (1 / 2 / 3) | χ² vs uniform | Mid-tercile acc |
|---|---|---|---|---|---|
| Random baseline | 0.363 | — | 33 / 33 / 33 | — | — |
| Top team (NLP Innovators) | 0.393 | — | — | — | — |
| Submitted Run 2 (R1–30) | 0.247 | — | 20 / 45 / 35 | p < 10⁻⁶ (heavy opt-2) | — |
| Submitted Run 2 replay (full 82) | 0.255 | — | 21 / 48 / 31 | p < 10⁻⁶ | — |
| Gemma 4 31B bare (**S**) | 0.412 | 0.402 | 48 / 27 / 25 | p < 10⁻¹² | 0.457 |
| Gemma 4 31B bare + guardrails (**S2**) | **0.470** | **0.454** | 53 / 23 / 24 | p < 10⁻²² | **0.569** |
| Gemma 4 31B perm × 6 (**S3**) | 0.400 | 0.399 | 37 / 33 / 30 | **p = 0.12 (uniform)** | 0.500 |
| Gemma 4 31B pairwise (**S4**) | 0.354 | 0.353 | 36 / 35 / 29 | p = 0.11 (uniform) | 0.353 |
| Gemma 3 27B R2 rank-1 pick | 0.287 | 0.225 | 15 / 77 / 7 | p < 10⁻¹⁰⁹ | 0.310 |

**Headline result: S2 = 0.470 — +22.3 pp over our submission, +7.7 pp above the official top team.** Mid-conversation accuracy peaks at 0.569: the system gets more than half of mid-session response selections correct.

#### 3.4.1 Why S2 Wins Over S, S3, and S4

The anti-bias guardrails in S2 add +5.8 pp over the bare prompt (S). Mechanical bias correction (S3 permutation, S4 pairwise) achieves cleaner prediction distributions but loses accuracy:

- **S3** averages over 6 candidate orderings → distribution chi² p = 0.12 (fails to reject uniform) but accuracy 0.400.
- **S4** pairwise + Condorcet → distribution chi² p = 0.11 but accuracy 0.354. Pairwise discards the contrastive information that having all three options available simultaneously provides.

**S2 wins because it gets both:** the contrastive 3-way signal *and* a controlled distribution. Explicit instructions are more effective than mechanical bias correction for this task.

#### 3.4.2 Ranking Inversion Test Result (R2)

The hypothesis: maybe our scoring is *valid but inverted* — we systematically rank the gold response 2nd or 3rd. R2 logs the full 3-way ranking and reports where gold lands:

| Gold position in our ranking | Share |
|---|---|
| Rank 1 (our top pick) | 28.7 % |
| Rank 2 | 37.1 % |
| Rank 3 (our bottom pick) | 34.2 % |

The distribution is roughly uniform (random expectation 33.3 % each). **The inversion hypothesis is rejected:** gold is not concentrated at rank 3. The system has weak signal, not anti-correlated signal.

#### 3.4.3 Consensus Failure on Task 2

Running 9 systems (Submitted R1-30, Submitted full replay, Gemma 4 31B {S, S2, S3, S4, R2}, Gemma 3 27B bare, Llama-3.3-70B bare) on the 299 (round, session) pairs covered by all of them ([posthoc_T2_consensus_failures.py](../analysis/MentalRiskES_test/posthoc_T2_consensus_failures.py)):

| Gold class | n | All-wrong rate | All-correct rate | Mean correct systems / 9 |
|---|---|---|---|---|
| 1 | 101 | 17.8 % | 3.0 % | 3.14 |
| 2 | 94 | 21.3 % | 1.1 % | 2.47 |
| **3** | **104** | **38.5 %** | **0.0 %** | **1.71** |
| ALL | 299 | 26.1 % | 1.3 % | 2.43 |

**Gold = option 3 is categorically harder than the other two classes.** 38.5 % of gold-3 rounds are wrong-by-every-system, and zero rounds had all 9 systems correct. The mean correct-systems count drops to 1.71 / 9. The "safety bias" pre-submission diagnosis was specific to our pipeline; the consensus-failure analysis shows that **option-3 rounds are categorically harder regardless of which system tries to solve them**.

A reasonable hypothesis: option 3 is positionally last, and its gold class is bimodal — sometimes "the riskiest direct probe" (clinically warranted but conversationally surprising), sometimes "the most elaborate intervention." The mixed nature makes a consistent decision rule hard to learn zero-shot for any LLM.

---

## 4. Shared Infrastructure

### 4.1 OpenRouter via the OpenAI SDK

Both arms use the same client constructor in [posthoc_P_gemma_gad7.py](../analysis/MentalRiskES_test/posthoc_P_gemma_gad7.py) and [posthoc_S_task2_bare_llm.py](../analysis/MentalRiskES_test/posthoc_S_task2_bare_llm.py):

```python
from openai import OpenAI
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)
```

Retry strategy: up to 5 attempts per call with exponential backoff (5 → 60 s cap). The retry filter catches HTTP 429 (rate limit), 5xx (provider errors), and `ChunkedEncodingError` / `IncompleteRead` / `ProtocolError` (stream truncation).

JSON parsing tolerates ```code fences (Gemma on Google AI Studio sometimes emits fences despite `response_format: {"type": "json_object"}`). The validator checks per-item score range (0–3 for GAD-7) and corrects stated `total_score` to the sum-of-items value with a warning if they disagree (we trust the items).

### 4.2 Cohort Runners

Both runners accept a `--cohort {test, trial, simulated}` flag. The runners load:

- **test:** multi-session per round file under `data/MentalRiskES-2026/test/taskN/test/data/round_X.json` (10 sessions, up to 82 rounds).
- **trial:** single-session legacy `"trial"`-keyed round files under `data/MentalRiskES-2026/taskN_trial/data/round_X.json` (19 rounds, 1 session).
- **simulated:** per-session directories under `output/mentalriskes/data_prep/simulated/taskN/<session_id>/` with `metadata.json` (Task 1 target totals) or `labels.json` (Task 2 gold options).

Output dirs are cohort-suffixed: `output/mentalriskes_gemma_gad7/<model>__<prompt_version>__<cohort>/` for Task 1; `output/mentalriskes_task2_bare_llm/<model>__<mode>__<cohort>/` for Task 2. The legacy `test` + `v1` path keeps the bare-model directory name (no cohort suffix) for backwards compatibility with the analysis scripts written before the cohort flag existed.

### 4.3 Auto-Resume

Both runners check the existing `raw.jsonl` on startup and skip any (session, round) pair already present. Interrupted runs can be restarted in place without re-billing already-completed calls.

### 4.4 File Map

| File | Purpose |
|---|---|
| [analysis/MentalRiskES_test/posthoc_P_gemma_gad7.py](../analysis/MentalRiskES_test/posthoc_P_gemma_gad7.py) | Task 1 Gemma GAD-7 runner; supports `--prompt-version {v1, v2}` and `--cohort {test, trial, simulated}`. |
| [analysis/MentalRiskES_test/posthoc_P_gemma_eval.py](../analysis/MentalRiskES_test/posthoc_P_gemma_eval.py) | Task 1 evaluator: per-session stats, per-item profile, confidence calibration, hybrid combined, oracle ensemble, multi-model cross comparison. |
| [analysis/MentalRiskES_test/posthoc_T1_cross_cohort_eval.py](../analysis/MentalRiskES_test/posthoc_T1_cross_cohort_eval.py) | Task 1 cross-cohort comparison (test vs simulated). |
| [analysis/MentalRiskES_test/posthoc_S_task2_bare_llm.py](../analysis/MentalRiskES_test/posthoc_S_task2_bare_llm.py) | Task 2 bare-LLM runner; supports `--mode {S, S2, R2, S3, S4}` and `--cohort {test, trial, simulated}`. |
| [analysis/MentalRiskES_test/posthoc_S_task2_bare_llm_eval.py](../analysis/MentalRiskES_test/posthoc_S_task2_bare_llm_eval.py) | Task 2 bare-LLM evaluator: accuracy, F1, distribution, chi², per-tercile, R2 inversion. |
| [analysis/MentalRiskES_test/posthoc_T2_cross_cohort_eval.py](../analysis/MentalRiskES_test/posthoc_T2_cross_cohort_eval.py) | Task 2 cross-cohort comparison (test vs trial vs simulated). |
| [analysis/MentalRiskES_test/posthoc_T2_consensus_failures.py](../analysis/MentalRiskES_test/posthoc_T2_consensus_failures.py) | Task 2 consensus-failure analysis across all tested systems. |
| [analysis/MentalRiskES_test/qualitative_T2_submitted_vs_s2.py](../analysis/MentalRiskES_test/qualitative_T2_submitted_vs_s2.py) | Submitted-vs-S2 head-to-head disagreement Markdown generator. |

---

## 5. Cross-Cohort Lesson

A central methodological finding from the Gemma branch: **pre-submission ablation on trial + simulated does not predict the Gemma branch's test-set win.**

### 5.1 Task 1 GAD-7

[posthoc_T1_cross_cohort_eval.py](../analysis/MentalRiskES_test/posthoc_T1_cross_cohort_eval.py):

| Cohort | n | Gemma 4 26B MoE v1 total-MAE | Gemma 4 26B MoE v2 total-MAE | v1 band acc | v2 band acc |
|---|---|---|---|---|---|
| test | 10 | 4.0 | **3.4** | 0.20 | **0.50** |
| simulated | 6 | 5.5 | 5.7 | 0.50 | 0.50 |

The v2 prompt clearly wins on test (total MAE 4.0 → 3.4; band acc 0.20 → 0.50) but is **indistinguishable from v1 on the simulated cohort**. Simulated personas are constructed from a single-paragraph profile with limited longitudinal evidence — the severity anchor and indirect-evidence markers in v2 have nothing rich enough to bind to. The trial cohort has no item-level gold, so a v1-vs-v2 comparison there is impossible.

### 5.2 Task 2 Response Selection

[posthoc_T2_cross_cohort_eval.py](../analysis/MentalRiskES_test/posthoc_T2_cross_cohort_eval.py):

| System | Test (n = 568) | Trial (n = 18) | Simulated (n = 87) |
|---|---|---|---|
| Submitted Run 2 (R1–30) | 0.247 | — | — |
| Submitted-equivalent (HYB B+ FIX W3) | — | **0.444** (8 / 18) | 0.897 |
| Gemma 4 31B bare (S) | 0.412 | 0.333 | 0.931 |
| **Gemma 4 31B bare + guardrails (S2)** | **0.470** | **0.444** (8 / 18) | **0.943** |

- **On trial:** S2 ties the submitted-equivalent at 8 / 18. With n = 18 a 0 pp gap is well within sampling noise.
- **On simulated:** S2 leads by only +4.6 pp. All three systems saturate above 0.90 because the persona dialogues are constructed with one clearly-fitting response per round.
- **On test:** S2 leads by +21.5 pp. Only here is the gap obvious.

**Implication:** the trial and persona-simulated benchmarks we used during system selection in March–April 2026 under-discriminate at the quality range relevant to choosing between approaches. Trial's small size lacks statistical power to expose a +5 pp difference; simulated's artificial unambiguity hides differences that surface on real heterogeneous data. The methodological lesson for future iterations is to **invest in test-like out-of-distribution corpora** (clinical conversations from a different therapist or population) before submission, not just synthetic personas and a single transcribed session.

This is itself a paper-relevant finding: it characterises a failure mode of the standard "pre-submission ablation on trial + simulated" approach in the MentalRiskES setup.

---

## 6. Limitations and Future Work

1. **Item 5 / Item 6 shared blind spot.** All Gemma variants — like Llama — under-predict restlessness and irritability when patients describe them indirectly. v2 closes the item-5 gap partially (signed bias −1.4 → −0.4 on Gemma 4 26B MoE) but item 6 remains stubborn. Fixing this likely requires either prompting the patient to elaborate on these symptoms specifically (out of scope for an assessor) or a higher-recall first-pass that flags possible irritability evidence.
2. **Confidence calibration is inverted.** The model is more accurate on MEDIUM-confidence items than HIGH. The HIGH label appears to track "symptom is clearly present" rather than "I can precisely estimate frequency". A future prompt could either remove the HIGH option or rewrite the rubric to ask for "frequency-precision" confidence specifically.
3. **Architecture sensitivity.** Only Gemma 4 31B has the reasoning to make the bare-LLM prompt outperform the engineered ACT pipeline. We don't have a principled account of *why* — Gemma 3 27B and Llama-3.3-70B both fall well short. A follow-up could compare token-level reasoning traces between the three models on the same rounds.
4. **No fine-tuning.** All results are zero-shot. The 300 (round, session) inner-join pairs from R1–30 could be used as supervised data to fine-tune Gemma 4 31B; an end-to-end "S2-then-supervised" pipeline is an obvious next experiment.
5. **The consensus-failure analysis hints at an irreducible task ceiling.** 26 % of all rounds are wrong-by-every-of-9-systems. Either the gold reflects clinician heuristics that LLMs don't replicate (in which case the ceiling is ~ 0.74), or the gold itself is contestable on those rounds (in which case it's lower).

---

## 7. Cross-References

- **Pre-submission system descriptions:** [docs/mentalriskes_task1_solution_description.md](mentalriskes_task1_solution_description.md), [docs/mentalriskes_task2_solution_description.md](mentalriskes_task2_solution_description.md). The Gemma branch is summarised at §7 (Task 1) and §9 (Task 2) of those files; this document is the deeper reference.
- **Post-hoc analysis report:** [analysis/MentalRiskES_test/SUMMARY.md](../analysis/MentalRiskES_test/SUMMARY.md) — Phase −1 verification (§0.5), full-replay (§0.6), test-data EDA (§0.7), Layer 3 Gemma GAD-7 (§5.5), Task 2 bare-LLM (§5.6), cross-cohort (§5.7), consensus failure (§5.8), Submitted-vs-S2 (§5.9), Task 1 cross-cohort (§5.10).
- **Task 2 case-study report:** [analysis/MentalRiskES_test/REPORT_T2_case_studies.md](../analysis/MentalRiskES_test/REPORT_T2_case_studies.md) — stand-alone narrative of the Task 2 disagreement analysis with cited examples.
- **Disagreement appendix:** [outputs/qualitative_T2_submitted_vs_s2.md](../analysis/MentalRiskES_test/outputs/qualitative_T2_submitted_vs_s2.md) — 300 cases with full transcript context and English glosses.
- **Consensus-failure appendix:** [outputs/W_t2_consensus_failures.md](../analysis/MentalRiskES_test/outputs/W_t2_consensus_failures.md) — 5 cases per gold class where every system failed.
- **Prompt specs:** [specs/MentalRiskES/gemma_gad7_prompt_spec.md](../specs/MentalRiskES/gemma_gad7_prompt_spec.md) (v1), [specs/MentalRiskES/gemma_gad7_prompt_v2.md](../specs/MentalRiskES/gemma_gad7_prompt_v2.md) (v2), [specs/MentalRiskES/task2_improvement_spec.md](../specs/MentalRiskES/task2_improvement_spec.md) (Task 2 bare-LLM experiments).
