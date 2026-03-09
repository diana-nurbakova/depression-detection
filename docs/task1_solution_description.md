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

Each persona is a **Meta-Llama-3-8B-Instruct** base model with a PEFT/LoRA adapter that encodes a specific depression profile. The 20 adapters are loaded from a HuggingFace collection. The persona generates naturalistic conversational responses reflecting its encoded severity and symptom pattern.

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
- Item 1 (Sadness): Do NOT score emotional numbness/flatness — that maps to Item 4 (Loss of Pleasure).
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

For example, under a Severe consensus, a Tier 1 NO_EVIDENCE item receives a prior of 2, while under a Mild consensus, a Tier 3 item receives 0.

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
| `proportional_085` | Multiply by 0.85 | Comparison variant |
| `minus_5` | Subtract 5 from all totals | Comparison variant |
| `progressive` | Band-aware with different deltas | Comparison variant |

---

## 5. Submission Runs

We submit three runs with different correction strategies to hedge across evaluation metrics:

| Run | Correction | Rationale |
|-----|-----------|-----------|
| **Run 1** | `band_aware` | Safety run — optimizes ADODL (closeness ratio) |
| **Run 2** | `flat_minus_2` | Calibrated risk — balanced MAD/DCHR trade-off |
| **Run 3** | `flat_minus_3` | Balanced hedge — conservative across all metrics |

All three runs use the full pipeline (specialized assessors + linguistic features + Bayesian prior + justificator + SDC).

---

## 6. LLM Configuration

| Role | Model | Provider | Temperature | Notes |
|------|-------|----------|-------------|-------|
| Persona | Llama-3-8B-Instruct + LoRA | HuggingFace/PEFT | — | Per-persona adapter |
| Interviewer | GPT-5-nano | OpenAI | — | Reasoning model, no streaming |
| Assessor (×4) | Llama-3.3-70B | Ollama | 0.1 | Parallel, streaming |
| Orchestrator | Llama-3.3-70B | Ollama | 0.3 | Exploration temperature |
| Justificator | Llama-3.3-70B | Ollama | 0.2 | Conservative temperature |

Fallback providers: Together AI (Llama-3.3-70B-Instruct-Turbo), HuggingFace Inference API.

---

## 7. Ablation Study Design

We evaluate the contribution of each pipeline component using the **TalkDep** dataset — a collection of 12 golden-truth conversations with known BDI-II totals and key symptoms spanning all severity bands:

### 7.1 TalkDep Golden-Truth Personas

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

### 7.2 Evaluation Metrics

| Metric | Formula | Range | Interpretation |
|--------|---------|-------|----------------|
| **DCHR** | Fraction of correct severity bands | 0–1 | Band classification accuracy |
| **MAD** | Mean |predicted − golden| | 0–63 | Point-wise score accuracy |
| **ADODL** | Mean((63 − |predicted − golden|) / 63) | 0–1 | Closeness ratio (higher = better) |
| **ASHR-proxy** | Mean per-persona symptom overlap with golden top-4 | 0–1 | Symptom identification accuracy |
| **Boundary Accuracy** | DCHR restricted to 4 boundary personas | 0–1 | Performance on hardest cases |

### 7.3 Ablation Configurations (Additive Design)

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

### 7.4 Additional Ablation Variants

Beyond the core additive study, we evaluate **correction strategy variants** and **SDC combinations**:

**Post-hoc correction variants** (on A0 baseline to isolate correction effects):
- `A0_none` — raw scores, no correction
- `A0_band_aware` — band-aware correction
- `A0_flat_minus_2` — flat −2
- `A0_flat_minus_3` — flat −3
- `A0_minus5` — flat −5

**SDC variants:**
- `A0_sdc` — SDC only, no post-hoc correction
- `A0_sdc_band_aware` — SDC + band-aware correction
- `A0_sdc_minus5` — SDC + flat −5
- `A0_sdc_flat_minus_3` — SDC + flat −3

**Cross-component variants:**
- `A7_proportional_085` — full pipeline (no prior) + proportional ×0.85
- `A7_progressive` — full pipeline (no prior) + progressive correction

### 7.5 Ablation Execution

Each configuration is evaluated against all 12 TalkDep conversations. The ablation framework:

1. **Loads TalkDep conversations** with golden BDI-II totals and key symptoms.
2. **For each (config, persona) pair:**
   - Extracts linguistic features from persona responses (if enabled).
   - Runs either a single general assessor (A0) or four specialized assessors (A1+).
   - Executes the scoring pipeline (Pass 1, optional consensus + Pass 2).
   - Applies justificator coherence check (if enabled).
   - Applies SDC (if enabled).
   - Applies post-hoc correction (if configured).
3. **Evaluates** predicted total and top-4 against golden truth.
4. **Aggregates** per-persona results into DCHR, MAD, ADODL, ASHR-proxy, and boundary accuracy.

### 7.6 Component Contribution Analysis

For each consecutive pair (A(n-1) → A(n)), we compute:
- **Delta metrics**: ΔDCHR, ΔMAD, ΔADODL, ΔASHR, Δboundary accuracy.
- **Consistency**: number of personas improved vs. worsened vs. unchanged.
- **Boundary impact**: which boundary personas were fixed or broken by the component addition.

This allows us to determine which components provide the most reliable improvements and which have risk of regression on specific persona types.

### 7.7 Key Ablation Hypotheses

| Transition | Hypothesis |
|-----------|-----------|
| A0 → A1 | Specialized assessors improve domain coverage and reduce cross-item confusion |
| A1 → A2 | Linguistic features provide independent severity signal, especially for boundary personas |
| A2 → A3 | Bayesian prior recovers information for unprobed items, helping moderate/severe personas |
| A3 → A4 | Justificator catches coherence violations, fine-tunes scores for edge cases |
| A4 vs A5/A6 | Temperature sweep identifies optimal assessor calibration |
| A4 vs A7 | Isolates Bayesian prior contribution within the full pipeline |

---

## 8. Output Format

**Per persona:**
- `interactions_{run_id}.json` — full conversation transcript (official eRisk format)
- `results_{run_id}.json` — BDI-II total + top-4 symptoms (official eRisk format)
- `internal_{run_id}.json` — detailed per-item scores, assessor outputs, scoring metadata, justificator reasoning

**Submission (merged across all 20 personas):**
- `interactions_{run_id}.json`
- `results_{run_id}.json`

---

## 9. Key Design Decisions and Rationale

1. **Four specialized assessors over one general assessor** — domain specialization reduces cross-item confusion (e.g., conflating Sadness with Loss of Pleasure) and enables parallel execution.

2. **DepreSym calibration in prompts** — without empirical base rates, LLMs over-score items where surface keywords rarely indicate clinical relevance (e.g., Punishment feelings).

3. **Consensus-gated Bayesian prior** — only applied when meta-features disagree with direct assessment, preventing unnecessary score inflation for well-assessed personas.

4. **Score Distribution Constraint** — directly addresses the "expressive LoRA" artifact where moderate personas speak in maximally severe language.

5. **Three-run correction hedge** — different post-hoc strategies optimize for different metrics (ADODL vs. MAD vs. DCHR), maximizing expected performance across the unknown evaluation weighting.

6. **Dual-module orchestrator** — programmatic rules ensure deterministic coverage guarantees (all domains probed, turn limits enforced) while the LLM module provides adaptive, context-sensitive interviewing guidance.

7. **Absolutist density as severity meta-feature** — validated on TalkDep as having the strongest correlation with depression severity among all linguistic features examined.
