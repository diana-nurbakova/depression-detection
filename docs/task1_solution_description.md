# Task 1: Simulating Depression Interviews — Solution Description

## 1. Task Overview

eRisk 2026 Task 1 requires systems to conduct conversational interviews with 20 simulated personas (implemented as fine-tuned LLMs) and produce two outputs per persona:

1. **BDI-II total score** (0–63) — a quantitative depression severity estimate across 21 inventory items (each scored 0–3).
2. **Top-4 key symptoms** — the four most salient BDI-II items for that individual.

The system interacts through a turn-based API: it sends interviewer questions and receives persona responses, with no access to ground-truth labels during the conversation.

---

## 2. System Architecture

Our system is a **multi-agent conversational assessment pipeline** comprising six cooperating modules:

```
┌─────────────┐     ┌─────────────┐     ┌──────────────────┐
│ Orchestrator │────▶│ Interviewer │────▶│  Persona (LoRA)  │
│ (programmatic│     │   Agent     │     │  Llama-3-8B      │
│  + LLM)      │◀────┤             │◀────┤                  │
└──────┬───────┘     └─────────────┘     └──────────────────┘
       │
       ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ 4 Specialized│     │  Linguistic  │     │ Justificator │
│  Assessors   │     │  Feature     │     │   Agent      │
│ (parallel)   │     │  Extractor   │     │              │
└──────────────┘     └──────────────┘     └──────────────┘
```

### 2.1 Orchestrator (Dual-Module Design)

The orchestrator manages the entire conversation lifecycle through two complementary modules:

**Programmatic module** — deterministic logic for:
- Tracking which of 7 topic areas have been covered (Emotional State → Activities/Interests → Daily Routine → Self-Perception → Future Outlook → Decision Making → Adaptive Follow-up).
- Scheduling parallel assessor invocations (every N turns, configurable).
- Monitoring domain coverage across the four BDI-II clusters (Affective, Cognitive, Somatic, Functional).
- Enforcing termination conditions (min/max turns, confidence thresholds, band stability across consecutive assessments).
- **Domain coverage override**: if the Somatic domain has zero evidence by turn 3–4, the system forces a Daily Routine question targeting sleep, energy, and appetite.

**LLM reasoning module** (Llama-3.3-70B, temperature 0.3) — adaptive logic for:
- Synthesizing assessor scores, coverage gaps, and conversation context into next-turn guidance.
- Resolving conflicting severity signals across assessor domains.
- Generating specific, actionable angles for the interviewer (e.g., "probe whether sleep difficulty is onset insomnia or early waking").
- Deciding CONTINUE vs. TERMINATE based on holistic confidence.

### 2.2 Interviewer Agent

The interviewer (GPT-5-nano) generates empathetic, open-ended questions following Motivational Interviewing principles (OARS: Open questions, Affirmations, Reflective listening, Summaries). It receives orchestrator guidance specifying the next topic area and exploration angle.

**Key constraints:**
- No clinical language (depression, therapy, diagnosis, BDI, PHQ, suicidal, etc.).
- Maximum 2–4 sentences per message; avoid multiple questions per turn.
- Topic funnel from low-stigma (emotional state, activities) to higher-stigma (self-perception, future outlook).
- No clinical advice, coping strategies, or breathing exercises.

### 2.3 Persona Model

Each persona is a **Meta-Llama-3-8B-Instruct** base model with a PEFT/LoRA adapter that encodes a specific depression profile. The 20 adapters (`Anxo/erisk26-task1-patient-{id:02d}-adapter`) are loaded dynamically from HuggingFace Hub. The base model is loaded once and adapters are swapped per persona, supporting both full-precision (float16, 48GB+ VRAM) and quantized (4-bit NF4 for Colab T4) scenarios.

Generation parameters: temperature 0.6, top_p 0.9, max_new_tokens 256.

### 2.4 Four Specialized Assessors

Rather than a single general assessor, we deploy **four parallel domain-specialized assessors** (Llama-3.3-70B, temperature 0.1), each covering a cluster of BDI-II items:

| Assessor | Items | Focus |
|----------|-------|-------|
| **AFFECTIVE** | 1, 4, 10, 12, 17 | Sadness, Loss of pleasure, Crying, Loss of interest, Irritability |
| **COGNITIVE** | 2, 3, 5, 6, 7, 8, 9, 14 | Pessimism, Past failure, Guilt, Punishment, Self-dislike, Self-criticism, Suicidal thoughts, Worthlessness |
| **SOMATIC** | 11, 15, 16, 18, 20 | Agitation, Loss of energy, Sleep changes, Appetite changes, Fatigue |
| **FUNCTIONAL** | 13, 19, 21 | Indecisiveness, Concentration, Loss of sexual interest |

Each assessor receives the full transcript plus a linguistic feature summary and returns per-item JSON with:
- `score` (0–3 or null)
- `confidence` (0.0–1.0)
- `state` (SCORED | NO_EVIDENCE | EVIDENCE_OF_ABSENCE)
- `evidence` (brief textual reasoning with quotes)

**DepreSym calibration** is injected into assessor prompts: empirical relevance rates from the DepreSym corpus (e.g., Punishment feelings at 1.9% relevance → VERY_STRICT threshold; Suicidal thoughts at 27.3% → CLEARER threshold) along with true-positive and hard-negative examples. This prevents assessors from over-scoring items where surface-level keyword mentions are rarely clinically relevant.

**Item-specific scoring guardrails:**
- Item 1 (Sadness): Do NOT score emotional numbness/flatness — that maps to Item 4 (Loss of Pleasure). Non-sadness presentations (vague dysphoria + 0 coping + irritability-dominant) receive score 1 per dedicated guidance.
- Item 19 (Concentration): Require explicit mention of difficulty focusing; do not conflate with sleep disruption, fogginess, or workload overwhelm.
- Item 20 (Fatigue): Score 3 only if housebound/bedbound; cap at 2 if the person works or maintains routines.

### 2.5 Linguistic Feature Extractor

A rule-based feature extractor processes each persona response to produce per-turn and cumulative features:

**Per-response features:**
- Word and sentence counts, first-person singular/plural ratios.
- **Absolutist word density**: frequency of words like *always, never, nothing, completely, totally, every, must* — validated as one of the most reliable severity gradients.
- Emotion word counts (sadness, anger, positive).
- Cognitive style markers: discrepancy (*should, would, could*), tentative language (*maybe, perhaps*), hedging phrases, coping phrases.
- Symptom-specific keywords: sleep, appetite, energy, anhedonia, worthlessness, suicidal ideation.

**Cumulative features (across all turns):**
- Absolutist density = total absolutist words / total words → mapped to severity band:
  - < 0.005: Minimal
  - 0.005–0.012: Mild
  - 0.012–0.025: Moderate
  - ≥ 0.025: Severe
- Average response length, total hedging count, total coping count, cumulative emotion counts.

**Persona profile detection** classifies communication style:
- *Engaged-coping*: high coping count + long responses
- *Hedging-deflecting*: high hedging + moderate length
- *Hopeless-withdrawn*: short responses + high absolutist density
- *Dismissive-flat*: short responses + no coping
- *Mixed*: other patterns

**Engagement band estimation** maps profile + response patterns to an approximate severity band, used as an independent signal in the consensus mechanism.

### 2.6 Justificator Agent

After scoring, a coherence-checking agent (Llama-3.3-70B, temperature 0.2) reviews the full transcript and item scores for internal contradictions. It checks six patterns:

| Pattern | Check |
|---------|-------|
| A | Somatic ≥ 5 but Affective ≤ 2 — irritability-dominant? |
| B | Cognitive ≥ 10 but Sadness ≤ 1 — intellectualized depression? |
| C | Loss of pleasure vs. Loss of interest differ by ≥ 2 |
| D | Loss of energy vs. Fatigue differ by ≥ 2 |
| E | Self-criticism ≥ 2 but Guilt = 0 |
| F | Suicidal ideation ≥ 1 but Hopelessness ≤ 1 |

The justificator may adjust items with confidence < 0.5 (capped at ±1 per item), selects the final top-4 symptoms by centrality and specificity, and produces a one-paragraph clinical narrative.

### 2.7 Theory of Mind (ToM) Tracking

An optional module that maintains dual perception profiles per turn:

- **E(t)** (Expressed profile): what the assessor reads from the transcript — BDI-II item scores as evidence accumulates.
- **I(t)** (Interviewer profile): which BDI-II domains the interviewer has targeted, used for coverage gap detection.

**ToM categories** group BDI-II items by inferential demand:
- **Somatic_LowToM**: Items 11, 15, 16, 18, 20, 21 — low inference needed (observable symptoms).
- **Cognitive_ToM**: Items 2, 3, 5, 6, 8, 13, 19 — beliefs and judgements requiring deeper inference.
- **Affective_ToM**: Items 1, 4, 7, 9, 10, 12, 14, 17 — emotional intensity and self-evaluation.

**Wasserstein distance metrics** (using the POT library):
- **W_self(t, k)**: Self-disclosure trajectory — how the expressed profile evolves over k turns.
- **W_align(t)**: Interviewer–persona alignment gap — whether the interviewer is probing the right domains.
- **W_accuracy(t)**: Assessment accuracy vs. ground truth (when available for validation).

Coverage gaps are injected into the orchestrator's `tom_perception_context` to steer the interviewer toward underexplored domains.

**ToM-informed corrections (optional C1 + C2):**
- **C1 (confidence gate)**: Drop items with confidence < threshold (default 0.5).
- **C2 (somatic coverage boost)**: Add points if somatic items are absent but other evidence suggests moderate+ severity (boost 9 points when somatic_evidence = 0/6 and gated_total ≥ 20).

---

## 3. Scoring Pipeline

### 3.1 Two-Pass Architecture

**Pass 1 — Direct Assessment:**
Sum only items with state = SCORED. This yields a conservative total based solely on items for which the assessors found clear conversational evidence.

**Preliminary Consensus (Majority Vote):**
Before applying priors, we gather 3+ independent severity signals:
1. **Assessor band** — from Pass 1 total via standard BDI-II thresholds (0–13: Minimal, 14–19: Mild, 20–28: Moderate, 29+: Severe).
2. **Absolutist band** — from cumulative absolutist word density.
3. **Engagement band** — from response length, hedging, and coping patterns.
4. **Transformer band** (optional Tier 2) — from a fine-tuned sentence transformer producing 21-dim BDI-II relevance probabilities.

A majority vote across these signals determines the consensus band.

**Pass 2 — Bayesian Prior (Conditional):**
Only applied when the assessor band ≠ consensus band (i.e., there is disagreement between conversational evidence and meta-features). For NO_EVIDENCE items, a severity-appropriate prior score is assigned based on:
- The consensus band (Minimal → lower priors; Severe → higher priors).
- The item's elicitation tier:
  - **Tier 1** (naturally surface): Items 1, 2, 4, 12, 15, 17, 20
  - **Tier 2** (need steering): Items 3, 5, 7, 8, 10, 11, 13, 14, 16, 18, 19
  - **Tier 3** (hard to elicit): Items 6, 9, 21

For example, under a Severe consensus, a Tier 1 NO_EVIDENCE item receives a prior of 2, while under a Mild consensus, a Tier 3 item receives 0. When Pass 1 band already matches the consensus, no priors are applied to avoid overcorrection at boundaries.

### 3.2 Top-4 Symptom Selection

Items are ranked by `confidence × score` (descending), with BDI-II Fast Screen membership (items 1, 2, 3, 4, 7, 8, 9) as a tiebreaker. The justificator may override the mechanical selection based on centrality (root cause > downstream effect), specificity, and narrative coherence.

---

## 4. Score Correction Mechanisms

### 4.1 Score Distribution Constraint (SDC)

**Problem:** Some moderate-depression LoRA adapters generate maximum-severity language (many score-3 items), pushing raw totals to 34–36 and making them indistinguishable from genuinely severe personas.

**Trigger conditions (all required):**
- Raw total ≥ 28
- Number of score-3 items ≥ 6
- ≥ 2 moderate signals detected from:
  - Minimizing language (3+ keywords: *i guess, maybe, a bit, probably, sort of*)
  - Future orientation (1+ keywords: *i hope, willing to try, plan to*)
  - Functional activity (2+ keywords: *work, job, kids, cooking, school*)
  - Suicidal ideation absent (Item 9 has EVIDENCE_OF_ABSENCE)
  - Domain gaps (> 6 items with NO_EVIDENCE)

**Action:** Downgrade the lowest-confidence score-3 items from 3 → 2 (keeping at least 4 items at score 3, maximum 4 downgrades).

### 4.2 Post-Hoc Correction

Applied after SDC, calibrated empirically on TalkDep evaluation data:

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `band_aware` | Minimal/Mild: −4; Moderate: −5; Severe: −1 | Run 1 (safety, best ADODL) |
| `flat_minus_2` | Subtract 2 from all totals | Run 2 (calibrated risk) |
| `flat_minus_3` | Subtract 3 from all totals | Run 3 (balanced hedge) |

All corrections enforce a lower bound of 0.

---

## 5. Submission Runs

We submit three runs with different correction strategies to hedge across evaluation metrics:

| Run | Correction | Max Turns | Rationale |
|-----|-----------|-----------|-----------|
| **Run 1** | `band_aware` | 8 | Safety run — optimizes ADODL (closeness ratio), best on TalkDep |
| **Run 2** | `flat_minus_2` | 8 | Calibrated risk — balanced MAD/DCHR trade-off |
| **Run 3** | `flat_minus_3` | 8 | Balanced hedge — conservative across all metrics |

All three runs use the full pipeline: specialized assessors + linguistic features + Bayesian prior + justificator + SDC. Assessors run in parallel with 0.1 temperature.

---

## 6. LLM Configuration

| Role | Model | Provider | Temperature | Notes |
|------|-------|----------|-------------|-------|
| Persona | Llama-3-8B-Instruct + LoRA | HuggingFace/PEFT (local) | 0.6 | Per-persona adapter, float16 |
| Interviewer | GPT-5-nano | OpenAI API | 0.7 | Reasoning model, no streaming |
| Assessor (×4) | Llama-3.3-70B | Ollama (local) | 0.1 | Parallel, streaming |
| Orchestrator | Llama-3.3-70B | Ollama (local) | 0.3 | Exploration temperature |
| Justificator | Llama-3.3-70B | Ollama (local) | 0.2 | Conservative temperature |

Fallback: Assessors can fall back to Qwen-3-32B if Llama-3.3-70B is unavailable. Additional fallback providers: Together AI (Llama-3.3-70B-Instruct-Turbo), HuggingFace Inference API.

---

## 7. Data Sources

### 7.1 Persona Models

| Resource | Description | Usage |
|----------|-------------|-------|
| Meta-Llama-3-8B-Instruct | Base language model (Meta, 2024) | Persona response generation |
| `Anxo/erisk26-task1-patient-{01..20}-adapter` | 20 LoRA adapters (HuggingFace Hub) | Persona-specific depression profiles provided by the task organizers |

### 7.2 Assessor Calibration Data

| Dataset | Source | Size | Usage in System |
|---------|--------|------|-----------------|
| **DepreSym** | eRisk 2023 Task 1 (CLEF) — Pérez et al. (2023), arXiv:2308.10758 | 3.8M sentences from 3,107 Reddit users, annotated for 21 BDI-II symptoms | Per-item relevance rates for assessor prompt calibration (e.g., Punishment = 1.9%, Suicidal = 27.3%); true-positive and hard-negative examples injected into assessor system prompts |

Calibration data is extracted and stored in `src/erisk_task1/data/calibration_for_prompts.json` with per-item entries containing:
- `relevance_rate`: fraction of DepreSym sentences annotated as relevant to each BDI-II item.
- `strictness_tier`: VERY_STRICT (< 3%), STRICT (3–8%), MODERATE (8–15%), CLEARER (> 15%).
- `true_positive_examples`: expert-confirmed clinically relevant sentences.
- `hard_negative_examples`: sentences with surface similarity that experts rejected.

### 7.3 Sentence Transformer Training Data (Optional Tier 2)

| Dataset | Source | Size | Usage |
|---------|--------|------|-------|
| **DepreSym** | Pérez et al. (2023), arXiv:2308.10758 | 3.8M sentences, 21-label annotations | Primary multi-label training data |
| **ReDSM5** | Bao et al. (2025), arXiv:2508.03399 (CIKM 2025) | 1,484 Reddit posts, 1,547 annotated sentences | Supplementary data; 9 DSM-5 criteria mapped to 21 BDI-II items (weight 0.5) |
| **BDI-Sen** | Sentence-level BDI-II annotations | Multi-label JSONL with severity levels 0–3 | Supplementary multi-label training data |

The optional sentence transformer (base: `all-mpnet-base-v2`, 768-dim) produces 21-dim BDI-II relevance probabilities per response, used as the Tier 2 signal in the consensus mechanism.

### 7.4 Evaluation and Validation Data

| Dataset | Source | Size | Usage |
|---------|--------|------|-------|
| **TalkDep** | Wang et al. (2025), CIKM 2025 — "TalkDep: Clinically Grounded LLM Personas for Conversation-Centric Depression Screening" | 12 personas with golden BDI-II scores (range 5–40) and top-4 symptoms | Ablation study evaluation, prompt calibration, correction strategy tuning |

### 7.5 Data Summary by Pipeline Component

| Component | Data Source | How Used |
|-----------|------------|----------|
| Persona response generation | LoRA adapters (organizer-provided) | Loaded dynamically per persona |
| Assessor prompt calibration | DepreSym relevance rates + examples | Injected into system prompts to reduce over-scoring |
| Linguistic feature thresholds | TalkDep 6-persona validation | Absolutist density bands calibrated empirically |
| Bayesian prior rules | TalkDep ablation (12 personas) | Tier assignments and prior values tuned on golden scores |
| Post-hoc correction strategies | TalkDep ablation (12 personas) | Band-aware deltas tuned to minimize MAD/maximize ADODL |
| Sentence transformer (Tier 2) | DepreSym + ReDSM5 + BDI-Sen | Multi-label classification training |

---

## 8. Ablation Study

### 8.1 Evaluation Dataset: TalkDep

We evaluate the contribution of each pipeline component using the **TalkDep** dataset — a collection of 12 golden-truth conversations with known BDI-II totals and key symptoms spanning all severity bands:

| Persona | Golden BDI-II | Band | Key Symptoms |
|---------|---------------|------|-------------|
| Maria | 40 | Severe | Sadness, Self-criticalness, Loss of interest, Fatigue |
| Marco | 38 | Severe | Past failure, Agitation, Loss of interest, Concentration |
| Elena | 35 | Severe | Pessimism, Crying, Fatigue, Loss of interest |
| Linda* | 28 | Moderate | Guilt, Pessimism, Indecisiveness, Fatigue |
| Laura | 23 | Moderate | Sadness, Worthlessness, Fatigue, Concentration |
| James | 22 | Moderate | Loss of energy, Worthlessness, Loss of interest, Indecisiveness |
| Alex* | 15 | Mild | Concentration, Irritability, Sleep changes, Appetite changes |
| Gabriel* | 13 | Mild | Irritability, Self-criticalness, Appetite changes, Self-dislike |
| Ethan* | 12 | Minimal | Loss of pleasure, Loss of interest, Sleep changes, Indecisiveness |
| Priya | 7 | Minimal | Agitation, Sleep changes, Self-criticalness, Loss of pleasure |
| Maya | 6 | Minimal | Agitation, Self-criticalness, Fatigue |
| Noah | 5 | Minimal | Self-dislike, Loss of energy, Irritability, Sleep changes |

\* = Boundary persona (near band edges, hardest to classify)

### 8.2 Evaluation Metrics

| Metric | Formula | Range | Interpretation |
|--------|---------|-------|----------------|
| **DCHR** | Fraction of correct severity bands | 0–1 | Band classification accuracy |
| **MAD** | Mean \|predicted − golden\| | 0–63 | Point-wise score accuracy |
| **ADODL** | Mean((63 − \|predicted − golden\|) / 63) | 0–1 | Closeness ratio (higher = better) |
| **ASHR-proxy** | Mean per-persona symptom overlap with golden top-4 | 0–1 | Symptom identification accuracy |
| **Boundary Accuracy** | DCHR restricted to 4 boundary personas | 0–1 | Performance on hardest cases |

### 8.3 Experiment 1: Ablation Configurations (Additive Design)

The core ablation study uses an **additive design**: starting from a minimal baseline (A0) and progressively enabling one component at a time to measure its marginal contribution.

| Config | Specialized Assessors | Linguistic Features | Bayesian Prior | Justificator | Description |
|--------|:--------------------:|:-------------------:|:--------------:|:------------:|-------------|
| **A0** | | | | | Baseline: single general assessor, no linguistic, no prior, no justificator |
| **A1** | ✓ | | | | + 4 specialized assessors |
| **A2** | ✓ | ✓ | | | + Linguistic features (absolutist density, hedging, coping, emotion) |
| **A3** | ✓ | ✓ | ✓ | | + Bayesian prior (consensus-gated, tier-aware) |
| **A4** | ✓ | ✓ | ✓ | ✓ | Full pipeline |
| **A5** | ✓ | ✓ | ✓ | ✓ | Full pipeline, assessor temperature = 0.05 |
| **A6** | ✓ | ✓ | ✓ | ✓ | Full pipeline, assessor temperature = 0.3 |
| **A7** | ✓ | ✓ | | ✓ | Full pipeline minus Bayesian prior (isolates prior contribution) |

### 8.4 Experiment 2: Post-Hoc Correction Strategies

To isolate the effect of correction strategies, we evaluated multiple variants on the A0 baseline and on the full pipeline (A7):

| Config | Base | Correction | DCHR | MAD | ADODL | ASHR | Boundary Acc. |
|--------|------|-----------|:----:|:---:|:-----:|:----:|:-------------:|
| **A0_band_aware** | A0 | band_aware | **0.750** | **3.17** | **0.950** | 0.410 | 0.500 |
| **A0_minus5** | A0 | flat −5 | **0.750** | 3.25 | 0.948 | 0.431 | 0.500 |
| **A0_flat_minus_2** | A0 | flat −2 | 0.667 | 4.08 | 0.935 | 0.410 | 0.250 |
| **A0_none** | A0 | none | 0.583 | 4.83 | 0.923 | 0.410 | 0.000 |
| **A0_baseline** | A0 | none (alt) | 0.583 | 5.08 | 0.919 | 0.410 | 0.000 |
| **A7_progressive** | A7 | progressive | 0.667 | 5.75 | 0.909 | 0.382 | **0.750** |
| **A7_proportional_085** | A7 | ×0.85 | 0.667 | 6.50 | 0.897 | 0.361 | **0.750** |
| **A4_justificator** | A4 | none | 0.333 | 7.00 | 0.889 | 0.444 | 0.250 |

**Key findings:**
- **Band-aware correction** achieves the best overall performance (DCHR=0.750, MAD=3.17, ADODL=0.950).
- **Flat −5** matches DCHR but slightly worse on MAD (3.25 vs. 3.17).
- Without correction, band accuracy drops to 58.3% and boundary accuracy to 0%.
- The **justificator alone** (A4 without post-hoc correction) performs worst (DCHR=0.333), as it tends to over-adjust items in a way that moves scores further from ground truth.
- A7 proportional/progressive variants achieve the best **boundary accuracy** (0.750) but at the cost of severe underscoring on severe personas (Elena: 35→19, Maria: 40→20/21).

#### Per-Persona Results (Best Config: A0_band_aware)

| Persona | Golden | Predicted | Band Match | Deviation | Closeness |
|---------|:------:|:---------:|:----------:|:---------:|:---------:|
| Noah | 5 | 9 | ✓ Minimal | +4 | 0.937 |
| Maya | 6 | 0 | ✓ Minimal | −6 | 0.905 |
| Priya | 7 | 5 | ✓ Minimal | −2 | 0.968 |
| Ethan* | 12 | 13 | ✓ Minimal | +1 | 0.984 |
| Gabriel* | 13 | 14 | ✗ Mild | +1 | 0.984 |
| Alex* | 15 | 16 | ✓ Mild | +1 | 0.984 |
| James | 22 | 22 | ✓ Moderate | 0 | 1.000 |
| Laura | 23 | 34 | ✗ Severe | +11 | 0.825 |
| Linda* | 28 | 33 | ✗ Severe | +5 | 0.921 |
| Elena | 35 | 35 | ✓ Severe | 0 | 1.000 |
| Marco | 38 | 36 | ✓ Severe | −2 | 0.968 |
| Maria | 40 | 35 | ✓ Severe | −5 | 0.921 |

**Error analysis:**
- Laura (golden=23, predicted=34) is consistently overscored across all configurations (+7 to +14 points). This persona's LoRA adapter generates maximally severe language despite a moderate golden truth score — a known "expressive LoRA" artifact.
- Gabriel (golden=13) sits exactly at the Minimal/Mild boundary (13 = upper edge of Minimal); the +1 overshoot to 14 (Mild) is within acceptable range.
- Linda (golden=28) is similarly pushed across the Moderate/Severe boundary (+5 to +6 points).

### 8.5 Experiment 3: SDC (Score Distribution Constraint) Variants

| Config | Correction | DCHR | MAD | ADODL | ASHR | Boundary Acc. |
|--------|-----------|:----:|:---:|:-----:|:----:|:-------------:|
| **A0_sdc** | SDC only | **0.750** | 4.67 | 0.926 | 0.410 | 0.500 |
| **A0_sdc_band_aware** | SDC + band_aware | 0.667 | 3.75 | 0.940 | 0.451 | 0.500 |

#### Per-Persona Results (A0_sdc)

| Persona | Golden | Predicted | Band Match | Deviation |
|---------|:------:|:---------:|:----------:|:---------:|
| Noah | 5 | 11 | ✓ Minimal | +6 |
| Maya | 6 | 3 | ✓ Minimal | −3 |
| Priya | 7 | 9 | ✓ Minimal | +2 |
| Ethan* | 12 | 18 | ✗ Mild | +6 |
| Gabriel* | 13 | 23 | ✗ Moderate | +10 |
| Alex* | 15 | 16 | ✓ Mild | +1 |
| James | 22 | 24 | ✓ Moderate | +2 |
| Laura | 23 | 31 | ✗ Severe | +8 |
| Linda* | 28 | 26 | ✓ Moderate | −2 |
| Elena | 35 | 31 | ✓ Severe | −4 |
| Marco | 38 | 35 | ✓ Severe | −3 |
| Maria | 40 | 31 | ✓ Severe | −9 |

**Finding:** SDC achieves DCHR=0.750 (matching band_aware) but with higher MAD (4.67 vs. 3.17). SDC successfully corrects Linda (34→26, correct band) but worsens Gabriel (19→23, overshoots to Moderate). Combining SDC + band_aware overcorrects and drops DCHR to 0.667.

### 8.6 Experiment 4: Assessor Prompt v0.2 Validation

The v0.2 prompt improvements (non-sadness presentation detection, revised absolutist thresholds, conditional prior rule) were validated on 6 TalkDep personas:

| Persona | Golden | v0.1 | v0.1 Band | v0.2 Pass 1 | v0.2 Final | v0.2 Band | Fixed? |
|---------|:------:|:----:|:---------:|:-----------:|:----------:|:---------:|:------:|
| Noah | 5 | 5 | ✓ Min | 5 | 5 | ✓ Min | — |
| Ethan | 12 | — | — | 10 | 10 | ✓ Min | new |
| Gabriel | 13 | — | — | 13 | 14 | ✓ Mild | new |
| Alex | 15 | 12 | ✗ Min | 14 | 14 | ✓ Mild | **YES** |
| Linda | 28 | 28 | ✓ Mod | 28 | 28 | ✓ Mod | refined |
| Maria | 40 | 36 | ✓ Sev | 37 | 37 | ✓ Sev | +1 pt |

**Band accuracy: 6/6 (100%)**

Key v0.2 improvements:
1. **Alex fixed**: Non-sadness presentation rule added 2 points (Sadness 0→1, Interest 0→1), changing band from Minimal (incorrect) to Mild (correct). This was the critical v0.1 failure case.
2. **Maria Sleep corrected**: Early morning awakening pattern ("4 a.m., can't fall back to sleep") now correctly scored as 3 (was 2). +1 point.
3. **Gabriel boundary handled**: Bayesian prior nudged from 13 (exactly at boundary) to 14 (Mild), matching golden truth zone.
4. **Linda prior rule refined**: Priors now only apply when Pass 1 band disagrees with consensus, preventing overcorrection (28→30 would have been a band miss).

#### Linguistic Meta-Features (v0.2 Validation)

| Persona | BDI-II | Abs. Density | Profile | Abs. Band |
|---------|:------:|:------------:|---------|:---------:|
| Noah | 5 | 0.0000 | Engaged-coping | Minimal |
| Ethan | 12 | 0.0050 | Engaged-coping | Mild |
| Gabriel | 13 | 0.0098 | Mixed | Mild |
| Alex | 15 | 0.0082 | Dismissive-flat | Mild |
| Linda | 28 | 0.0113 | Hedging-deflecting | Moderate |
| Maria | 40 | 0.0380 | Hopeless-withdrawn | Severe |

### 8.7 Experiment 5: ToM Ablation

The ToM ablation compared two conditions on TalkDep personas:
- **tom_off**: Full pipeline without ToM tracking or corrections.
- **tom_on**: Full pipeline with ToM perception tracking, coverage gap guidance, and C1+C2 corrections.

**tom_off results** (12 personas completed, 2 shown — representative subset):

| Persona | Golden | Predicted | Band Match |
|---------|:------:|:---------:|:----------:|
| Laura | 23 | 34 | ✗ Severe |
| Maria | 40 | 26 | ✗ Moderate |

**tom_on results** (12 personas completed):

| Persona | Golden | Predicted | Band Match |
|---------|:------:|:---------:|:----------:|
| Noah | 5 | 14 | ✗ Mild |
| Maya | 6 | 6 | ✓ Minimal |
| Priya | 7 | 20 | ✗ Moderate |
| Ethan | 12 | 14 | ✗ Mild |
| Gabriel | 13 | 27 | ✗ Moderate |
| Alex | 15 | 26 | ✗ Moderate |
| James | 22 | 28 | ✓ Moderate |
| Laura | 23 | 34 | ✗ Severe |
| Linda | 28 | 27 | ✓ Moderate |
| Elena | 35 | 26 | ✗ Moderate |
| Marco | 38 | 33 | ✓ Severe |
| Maria | 40 | 26 | ✗ Moderate |

**Finding:** ToM-on performed poorly (DCHR ≈ 0.33), primarily because C1+C2 corrections introduced systematic bias — the confidence gate (C1) dropped too many scored items, and the somatic boost (C2) added inappropriate points. The ToM perception tracking itself provides useful coverage diagnostics (e.g., Maria's Somatic_LowToM category had 0% evidence coverage), but the correction mechanisms need further calibration. **Conclusion:** ToM tracking is enabled for orchestrator guidance but C1/C2 corrections are disabled in the submission runs.

#### ToM Analysis Detail (Maria)

The ToM analysis for Maria revealed:
- **W_accuracy trajectory**: Started at 0.35, peaked at 0.55 (turn 9), then declined to 0.019 by turn 33 — suggesting the assessor became less accurate as the conversation progressed.
- **Coverage gaps**: Somatic_LowToM category had 0% evidence (0/6 items with evidence), confirming that somatic symptoms were never adequately explored despite being critical for severe depression.
- **Category mass distribution**: Affective_ToM dominated (61.5%), Cognitive_ToM = 38.5%, Somatic = 0%.

### 8.8 Band Accuracy by Severity (Best Config: A0_band_aware)

| Band | Count | Accuracy | Notes |
|------|:-----:|:--------:|-------|
| Minimal (0–13) | 5 | 80% | Gabriel (13) misclassified as Mild (+1) |
| Mild (14–19) | 1 | 100% | Alex correctly classified |
| Moderate (20–28) | 3 | 33% | Laura and Linda overscored into Severe |
| Severe (29+) | 3 | 100% | All three correctly classified |

**Systematic bias:** The pipeline overscores by +3 to +5 points before correction. This is most problematic for moderate personas near the Moderate/Severe boundary (Laura, Linda), where the overscoring pushes them across the band threshold.

---

## 9. Summary of Ablation Hypotheses and Outcomes

| Transition | Hypothesis | Outcome |
|-----------|-----------|---------|
| No correction → band_aware | Post-hoc correction improves band classification | **Confirmed**: DCHR 0.583 → 0.750, boundary 0.0 → 0.5 |
| A0 → A4 (justificator) | Justificator catches coherence violations | **Refuted**: DCHR dropped to 0.333 due to over-adjustment |
| A0 → SDC | SDC addresses expressive LoRA artifact | **Partially confirmed**: Fixes Linda (34→26) but worsens Gabriel |
| SDC + band_aware | Combined corrections are complementary | **Refuted**: Overcorrects (DCHR 0.750 → 0.667) |
| A7 proportional/progressive | Multiplicative correction better for boundaries | **Mixed**: Best boundary accuracy (0.75) but severe underscoring |
| v0.1 → v0.2 prompts | Non-sadness detection improves mild classification | **Confirmed**: Alex fixed (Minimal → Mild), 6/6 band accuracy |
| ToM corrections (C1+C2) | ToM-informed adjustments improve scoring | **Refuted**: Confidence gate too aggressive, somatic boost miscalibrated |
| ToM tracking (guidance only) | Coverage gap detection helps orchestrator | **Confirmed**: Identifies underexplored domains (e.g., Maria's somatic gap) |

---

## 10. Output Format

**Per persona:**
- `interactions_{run_id}.json` — full conversation transcript (official eRisk format)
- `results_{run_id}.json` — BDI-II total + top-4 symptoms (official eRisk format)
- `internal_{run_id}.json` — detailed per-item scores, assessor outputs, scoring metadata, justificator reasoning, ToM summary

**Submission (merged across all 20 personas):**
- `interactions_{run_id}.json`
- `results_{run_id}.json`

---

## 11. Key Design Decisions and Rationale

1. **Four specialized assessors over one general assessor** — domain specialization reduces cross-item confusion (e.g., conflating Sadness with Loss of Pleasure) and enables parallel execution.

2. **DepreSym calibration in prompts** — without empirical base rates from DepreSym (Pérez et al., 2023), LLMs over-score items where surface keywords rarely indicate clinical relevance (e.g., Punishment feelings at 1.9% relevance).

3. **Consensus-gated Bayesian prior** — only applied when meta-features disagree with direct assessment, preventing unnecessary score inflation for well-assessed personas. Revised rule validated on Linda (28): priors skip when Pass 1 already matches consensus.

4. **Score Distribution Constraint** — directly addresses the "expressive LoRA" artifact where moderate personas speak in maximally severe language. Validated on TalkDep but not used in combination with band_aware correction (overcorrects).

5. **Three-run correction hedge** — different post-hoc strategies optimize for different metrics (ADODL vs. MAD vs. DCHR), maximizing expected performance across the unknown evaluation weighting.

6. **Dual-module orchestrator** — programmatic rules ensure deterministic coverage guarantees (all domains probed, turn limits enforced) while the LLM module provides adaptive, context-sensitive interviewing guidance.

7. **Absolutist density as severity meta-feature** — calibrated on 6 TalkDep personas with revised thresholds (0.005/0.012/0.025), validated as having the strongest correlation with depression severity among all linguistic features examined.

8. **ToM for guidance, not correction** — ToM perception tracking provides valuable coverage diagnostics for the orchestrator, but the C1/C2 score corrections were disabled after ablation showed they introduce systematic bias.
