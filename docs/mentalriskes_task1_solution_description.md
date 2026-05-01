# MentalRiskES 2026 Task 1: Zero-Shot Multi-Instrument Psychometric Assessment from Therapeutic Conversations — System Description

## 1. Task Overview

MentalRiskES 2026 Task 1 requires systems to predict a patient's responses to three standardized psychometric instruments from an unfolding Spanish therapeutic conversation, with predictions submitted after each patient turn across multiple rounds:

- **PHQ-9** (Patient Health Questionnaire-9): 9 items, scale 0--3 (depression severity)
- **GAD-7** (Generalized Anxiety Disorder-7): 7 items, scale 0--3 (anxiety severity)
- **CompACT-10** (Comprehensive Assessment of ACT processes): 10 items, scale 0--6 (psychological flexibility)

The task is **zero-shot**: no labeled training data is provided. Systems must rely on pretrained language models, prompting strategies, or external psychometric resources. The evaluation campaign ran from April 13--20, 2026, with 3 runs allowed per team.

### 1.1 Corpus Characteristics

The corpus consists of simulated yet realistic therapeutic conversations in Spanish. Five licensed psychologists interacted through a custom application with simulated patients (played by psychology students). Each student was assigned a patient profile with specific characteristics and key symptoms derived from PHQ-9, GAD-7, and CompACT-10. Therapist responses were AI-suggested but could be edited or replaced entirely. Key properties include human-in-the-loop generation (patient turns from psychology students, not pure LLM output), expert review for clinical validity, naturalistic informal Spanish therapeutic register, and progressive symptom revelation across rounds.

### 1.2 Evaluation Metrics

The official evaluation uses item-level RMSE (Root Mean Squared Error) and Pearson correlation per instrument, as well as mean RMSE and mean Pearson across all three instruments.

---

## 2. System Architecture

Our system is a **three-tier calibrated LLM assessment pipeline** that processes each conversation round through parallel instrument-specific assessors, followed by multi-level psychometric calibration and temporal aggregation.

```
Patient Spanish Text (per round)
  │
  ├─→ Context Accumulator (full dialogue history, sliding window)
  │
  ├─→ Three Parallel Chain-of-Thought Assessors
  │     ├─→ PHQ-9 Assessor  → [9 scores: 0-3]
  │     ├─→ GAD-7 Assessor  → [7 scores: 0-3]
  │     └─→ CompACT-10 Assessor → [10 scores: 0-6]
  │
  ├─→ Level A: Prompt Anchors (embedded in assessor prompts)
  │
  ├─→ Level B: Rule-Based Cross-Instrument Constraints (7 rules, no LLM)
  │
  ├─→ Level C: LLM Calibration Agent (conditional invocation)
  │
  ├─→ Temporal Aggregation (T2 early-weighted / T3 stability-adaptive)
  │
  └─→ Submission
```

### 2.1 Context Accumulation

A per-session `ConversationStore` maintains the full dialogue history with both patient and therapist turns. For prompt construction, a sliding window strategy is applied:

- **Always include** the first patient turn (initial presenting complaint)
- **Always include** the last 3 complete exchanges (6 turns)
- **Skip intermediate turns** if context exceeds ~20 turns (~4K characters)

This ensures the LLM has access to both the initial presentation (most diagnostic for past-two-weeks symptoms) and the most recent exchanges, while staying within the model's effective context window.

### 2.2 Chain-of-Thought Assessors

Three separate assessment prompts are used, one per instrument. Each follows a structured **three-step chain-of-thought** protocol:

**Step 0 — Category-Level Evidence Scan**: Broad symptom categories (not individual items) are rated as STRONG, MODERATE, WEAK, or NONE.
- PHQ-9 categories: Somatic (items 3,4,5,8), Cognitive (items 6,7), Affective (items 1,2,9)
- GAD-7 categories: Somatic Anxiety (items 1,4,5), Cognitive Anxiety (items 2,3), Emotional Reactivity (items 6,7)
- CompACT-10 categories: Openness to Experience, Behavioral Awareness, Valued Action

**Step 1 — Per-Item Detection with Disambiguation**: For evidenced categories, each item is detected as PRESENT, ABSENT, or INSUFFICIENT. Disambiguation notes separate co-occurring symptoms (e.g., PHQ-9 item 4 energy fatigue vs. item 8 observable psychomotor changes; GAD-7 item 2 worry controllability vs. item 3 worry pervasiveness).

**Step 2 — Temporal Inference / Endorsement Level**: Maps temporal evidence to the frequency/agreement scale. PHQ-9/GAD-7 use behavioral frequency anchors (0 = "para nada", 3 = "casi todos los dias"). CompACT-10 uses agreement anchors (0 = "totalmente en desacuerdo", 6 = "totalmente de acuerdo").

The assessor outputs a structured JSON containing reasoning steps and final scores, enabling both submission and post-hoc error analysis.

**Prompt design principles**:
1. Chain-of-thought before scores (prevents pattern-matching to severity levels)
2. Few-shot examples from real clinical data (grounds calibration)
3. English reasoning, Spanish content (stronger LLM reasoning in English; instrument items in original Spanish)
4. Structured JSON output with per-item reasoning

#### 2.2.1 Verbalizer System (v2.1)

A verbalizer system links numeric scores to clinically meaningful Spanish labels for consistency checking. Three levels of verbalization are provided:

- **Level 1 — Score Verbalizers**: Generic frequency labels for PHQ-9/GAD-7 and agreement labels for CompACT-10, plus item-specific behavioral descriptors (e.g., PHQ-9 item 1 anhedonia: 0 = "mantiene interes", 3 = "perdida casi total")
- **Level 2 — Category Evidence Tags**: Replace abstract STRONG/MODERATE/WEAK/NONE with clinically meaningful descriptions
- **Level 3 — Detection Tags**: Per-item process-specific labels

If the LLM outputs a label that does not match its numeric score, the system trusts the label (semantic reasoning over mechanical transcription).

#### 2.2.2 Recency Bias Defense

A recency bias warning is injected into all assessor prompts, reminding the LLM that PHQ-9/GAD-7 ask about the **past two weeks**, not the current therapy session. Within-session therapeutic improvement (patient calming down, engaging with techniques) should not reduce past-two-weeks severity estimates.

### 2.3 Three-Tier Calibration Architecture

#### Level A: Prompt Anchors (Zero Cost)

Psychometric calibration guidance embedded directly in assessor prompts:

- **CompACT-10 Valued Action anchor**: "For a moderately distressed patient (PHQ-9 10--14), typical Valued Action scores are 3--4. Score 5+ only with strong behavioral evidence of values-aligned action OUTSIDE the therapy session, not just within-session willingness."
- **CompACT-10 Openness to Experience anchor**: "If the therapist teaches acceptance/defusion techniques, this implies the patient has avoidance/fusion patterns. Score OtE items at 3--4 (moderate avoidance), not 0--1."
- **GAD-7 items 2 vs 3 anchor**: Disambiguates worry controllability (item 2, the loop) from worry pervasiveness (item 3, the breadth).
- **Cross-instrument anchor**: "PHQ-9 and GAD-7 totals are typically within 4 points of each other."
- **GAD-7 severity examples**: Additional anchoring for severe anxiety presentations.

#### Level B: Rule-Based Post-Assessment Constraints (7 Rules, No LLM)

Applied after all three assessors produce scores. Seven constraint rules enforce cross-instrument psychometric consistency:

| Rule | Constraint | Severity | Action |
|------|-----------|----------|--------|
| C1 | PHQ-9/GAD-7 normalized discordance > 0.40 | High | Flag |
| C2 | \|PHQ-9 total - GAD-7 total\| > 8 points | Medium | Flag |
| C3 | PHQ-9 somatic vs GAD-7 somatic mean diff > 1.5 | Medium | Flag |
| C4 | CompACT VA mean > expected + 1.0 AND not self-contradiction | High | Correct VA items -1 |
| C5 | CompACT OtE mean < expected - 0.5 for distress level | Medium | Flag only |
| C6 | CompACT within-subprocess spread > 3 | Medium | Flag |
| C7 | PHQ-9 item 9 (suicidality) > 0 with total < 10 | High | Flag for verification |

**Self-Contradiction Guard (Rule C4)** — the most clinically nuanced rule. Literature from latent profile analysis (N=1,769 Chinese college students) shows that approximately 19% of patients exhibit a **self-contradiction profile**: high Valued Action combined with high distress and low Openness to Experience. These patients act on their values but cannot face difficult internal experiences. The guard logic:

1. Is VA mean above expected range for the distress level?
2. If yes, is OtE mean < 2.5? (indicating genuine low avoidance — the self-contradiction pattern)
3. If OtE is low: do NOT correct (real clinical pattern)
4. If OtE is moderate-to-high: apply -1 to VA items (LLM conflation artifact)

#### Level C: LLM Calibration Agent (Conditional)

A separate LLM call (~1,500 tokens) that receives raw scores, assessor reasoning summaries, and Level B violation reports. Invoked only when Level B detects high/medium severity violations or CompACT-10 total exceeds expected ceiling + 5.

Agent capabilities beyond Level B:
- Distinguishes "patient demonstrates values behavior outside therapy" from "within-session engagement only"
- Evaluates internal consistency of the assessor's reasoning vs. scores
- Integrates evidence type classifications (explicit vs. implicit vs. absent)

Agent constraints:
- When in doubt, do NOT correct
- Never correct > 2 points on any single item
- Never override Level B's self-contradiction guard
- All corrections logged with rationale

### 2.4 Temporal Aggregation

Analysis of trial and simulated data revealed that within-session therapeutic improvement systematically distorts assessments: anxiety evidence concentrates in rounds 1--5 (keyword density 3--7 per turn), while later rounds shift to mindfulness and values work (density 0--1). The LLM anchors on recent rounds, producing under-estimates for past-two-weeks symptoms.

**Architectural solution**: Store per-item predictions at every round in a `PredictionMatrix` (rounds x items), then aggregate across rounds rather than submitting the latest round alone.

Three temporal aggregation methods were developed and tested:

**T0 — Last-Round Only** (baseline): Submit latest round's prediction. No temporal information. Maximally susceptible to recency bias.

**T2 — Early-Weighted Median**: Weight early rounds more heavily. The `step` decay variant doubles the weight of the first 5 rounds (weight = 2.0 for rounds 1--5, 1.0 for rounds 6+). Per-item weighted median is computed.

**T3 — Stability-Adaptive Aggregation**: Per-item adaptive choice based on prediction stability. For stable items (std < 0.5 across rounds), use the latest round (model is confident). For unstable items (std >= 0.5), use the early-weighted median (model uncertain, lean on initial presentation).

**Wasserstein Anomaly Detection**: W1 (Earth Mover's) distance between each round's prediction distribution and the running mean is computed using the Python Optimal Transport library. Clinically-grounded ground metrics define inter-item distances:
- PHQ-9: within-category (affective/cognitive/somatic) distance 0.2, adjacent 0.5, far 0.8
- GAD-7: within-category 0.2, between-category 0.6
- CompACT-10: within-hexaflex 0.2, OtE--BA 0.4, BA--VA 0.6, OtE--VA 0.8

Rounds where W1 > mean + 2.0*sigma are flagged as anomalous. For PHQ-9/GAD-7, anomalous rounds are discarded before aggregation (symptoms are constant over two weeks). For CompACT-10, anomalous rounds are flagged but retained (psychological flexibility may genuinely evolve within a therapy session).

### 2.5 LLM Configuration

| Role | Model | Provider | Temperature |
|------|-------|----------|-------------|
| Assessor (x3) | Llama-3.3-70B-Instruct | HuggingFace Inference API | 0.1 |
| Level C Agent | Llama-3.3-70B-Instruct | HuggingFace Inference API | 0.1 |
| Simulation (data prep) | Llama-3.3-70B-Instruct-Turbo | Together AI | 0.7 |

All official runs use the same model (Llama-3.3-70B-Instruct). Max tokens: 8192 (CompACT-10 chain-of-thought outputs require ~5--6K tokens). Timeout: 180 seconds.

---

## 3. Data Sources and External Resources

### 3.1 Task-Provided Data

- **Trial data**: 19 rounds of a single Spanish therapeutic conversation with gold-standard item-level scores (released March 27, 2026). Used for system development, calibration, and ablation analysis.
- **Test data**: Multi-session evaluation conversations (released April 13, 2026).

**Trial Gold Standard** (from task specification Appendix C):
- PHQ-9: [1, 2, 1, 2, 1, 2, 2, 2, 0] — total 13 (moderate depression)
- GAD-7: [3, 2, 2, 2, 2, 1, 2] — total 14 (moderate anxiety)
- CompACT-10: [3, 3, 4, 3, 3, 3, 4, 3, 3, 4] — total 33

CompACT-10 subscales (gold):
- Openness to Experience (items 3, 5, 8): [4, 3, 4] — total 11
- Behavioral Awareness (items 1, 6, 9): [3, 3, 3] — total 9
- Valued Action (items 2, 4, 7, 10): [3, 3, 4, 4] — total 14

### 3.2 External Datasets Used for Few-Shot Examples

| Dataset | Use | Items | Reference |
|---------|-----|-------|-----------|
| **PRIMATE** | PHQ-9 few-shot examples | 2,003 Reddit posts with per-item PHQ-9 binary labels | Gupta et al. (2022). "Learning to Automate Follow-up Question Generation using Process Knowledge for Depression Triage on Reddit Posts." *CLPsych @ NAACL 2022*. |
| **DAIC-WOZ** | PHQ-9 conversational examples | 189 clinical interviews with PHQ-8 total scores | Gratch et al. (2014). "The Distress Analysis Interview Corpus of human and computer interviews." *LREC 2014*. |

**PRIMATE extraction**: Best positive and negative examples per PHQ-9 symptom are selected using a scoring function: keyword clarity x 2.0 + (1 / (1 + other_positives)) x 3.0. Posts between 50--250 words are preferred (conversational, not too sparse or verbose). Each post is assigned to its single most illustrative symptom.

### 3.3 Psychometric Literature Used for Calibration

The three-tier calibration system is grounded in published psychometric correlations:

**PHQ-9 x GAD-7 relationship**:
- Ryan et al. (2022), N=31,974 concurrent pairs: Spearman rho = 0.74; 78.4% within 4 points; 56.4% same severity class. PHQ-9 mean = 8.4, GAD-7 mean = 7.5.
- Melbye et al. (2022), N=1,588 Norwegian psychiatric outpatients: Bifactor analysis shows poor discriminant validity of somatic factors (PHQ-9 sleep/fatigue/appetite/psychomotor vs. GAD-7 nervousness/relaxation/restlessness measure nearly the same construct).
- Kroenke et al. (2016), N=896: PHQ-ADS composite analysis; 31.9% score >= 10 on both; only 2.3% on GAD-7 alone without PHQ-9.

**CompACT-10 x Distress**:
- Francis et al. (2016): CompACT-23 validation. CompACT total x depression r ~ -0.55; CompACT total x anxiety r ~ -0.50.
- Golijani-Moghaddam et al. (2023): CompACT-10 validation study.
- Baker & Berghoff (2021), N=447: Network analysis confirming experiential avoidance (OtE) as central node connecting inflexibility to distress.

**Self-contradiction profile**:
- Latent profile analysis, N=1,769 Chinese college students (CompACT-18 x DASS-21): 19.2% show high Valued Action + low Openness + moderate-to-high distress. This is the critical empirical basis for the self-contradiction guard (Rule C4).

**CompACT structural analysis**:
- Francis et al. (2016); Ruiz et al. (2024): AAQ-II loads on CompACT Openness to Experience factor, confirming OtE as primary bridge between flexibility measurement and distress (r ~ 0.45 with distress; VA weaker at r ~ -0.30; BA weakest at r ~ -0.30).

### 3.4 Therapist Technique Recognition

The therapist's ACT technique usage provides direct signal for CompACT-10. A heuristic mapping was developed:

| Therapist Technique | ACT Process | CompACT-10 Implication |
|-------------------|-------------|----------------------|
| "Dejalo pasar como un coche" | Cognitive defusion | Item 3 — patient shows fusion if technique is needed |
| "Tiene una forma? Un color?" | Present-moment contact | Items 1,6,9 — patient may lack awareness |
| "Observar como una nube" | Acceptance | Items 5,8 — patient shows avoidance |
| "Moverte hacia algo que te importe" | Values clarification | Items 2,4 — patient may lack values clarity |
| "Un evento que ocurre en tu mente" | Self-as-context | Item 3 — defusion indicator |

**Heuristic**: If the therapist introduces an ACT technique, the patient NEEDS it (has deficit in that area). If the patient ENGAGES with the technique, score moderately (3--4) rather than at extremes.

---

## 4. Simulated Data for Development

To extend development beyond the single-session trial data, we generated **6 simulated patient-therapist conversations** using LLM-based persona simulation.

### 4.1 Persona Profiles

Six personas spanning diverse clinical presentations were defined:

| Persona ID | Profile | PHQ-9 Target | GAD-7 Target | CompACT Profile | Personality |
|-----------|---------|:------------:|:------------:|:---------------:|-------------|
| sim_anx_academic_42 | University student, academic anxiety | 6 (mild) | 10 (moderate) | Low flexibility | Perfeccionista, autocritico/a, evitativo/a |
| sim_anx_health_46 | Health anxiety / hypochondria | 10 (moderate) | 15 (severe) | Low flexibility | Hipervigilante, catastrofista |
| sim_anx_social_44 | Social anxiety, low self-esteem | 14 (moderate) | 20 (severe) | Low flexibility | Timido/a, sensible a evaluacion |
| sim_dep_burnout_45 | Professional burnout | 14 (moderate) | 11 (moderate) | Moderate flexibility | Responsable, dificultad poner limites |
| sim_dep_loss_43 | Grief / significant loss | 8 (mild) | 8 (mild) | Low flexibility | Introspectivo/a |
| sim_dep_mild_47 | Mild emerging depression | 7 (mild) | 6 (mild) | Moderate flexibility | Reservado/a |

### 4.2 Generation Pipeline

Each simulated session consists of 15 rounds generated through an LLM-driven conversation loop:

1. **Patient LLM** (Llama-3.3-70B-Instruct-Turbo, temperature 0.7): Generates naturalistic Spanish responses consistent with the assigned profile, distress level, and personality traits.
2. **Therapist LLM**: Produces ACT-informed therapeutic responses adapted to the current therapeutic phase (10 phases mapped across 15 rounds: engagement → creative_hopelessness → acceptance → defusion → present_moment → self_as_context → values → committed_action → integration → closing).
3. **Metadata**: Each persona includes target PHQ-9/GAD-7 totals, CompACT flexibility profile, personality descriptors in Spanish, and round-by-round internal state.

### 4.3 Use of Simulated Data

The simulated data served three purposes:
1. **Recency bias analysis**: Confirmed that within-session therapeutic improvement distorts late-round assessments (e.g., sim_anx_health_46: GAD-7 gold = 20, but round 15 assessment = 7--11). This motivated the temporal aggregation layer.
2. **Calibration generalization testing**: Ablation configurations tested on simulated data assess whether calibration rules generalize beyond the single trial patient.
3. **Temporal aggregation tuning**: The 6 x 15-round simulated sessions provided sufficient data to compare T0/T2/T3 aggregation methods.

---

## 5. Submission Runs

Three official runs were submitted, forming a **calibration ablation staircase**:

| Run | Config | Level A (Anchors) | Level B (Rules) | Level C (Agent) | Temporal | Optimization Target |
|-----|--------|:-:|:-:|:-:|----------|-------------------|
| **Run 0** | A5-T3 | Yes | Yes | Yes | T3 (stability-adaptive) for CompACT-10; T2 for PHQ-9/GAD-7 | Best RMSE (full stack) |
| **Run 1** | A3-T2 | Yes | Yes | No | T2 (early-weighted) for all | Safety hedge (no agent risk) |
| **Run 2** | A1-T2 | Yes | No | No | T2 (early-weighted) for all | Ranking preservation (Pearson) |

**Rationale**: All three runs use the strongest available model (Llama-3.3-70B-Instruct). The three runs form a clean ablation:
- Run 0 vs Run 1: marginal value of the LLM calibration agent
- Run 1 vs Run 2: marginal value of rule-based constraints
- Run 0 vs Run 2: total value of the full post-assessment calibration stack

Every possible outcome (Run 0 > 1 > 2, or any other ordering) produces a clear analytical conclusion for the working notes paper.

---

## 6. Ablation Study

### 6.1 Experimental Design

A 2x2 factorial design crossing prompt anchors (on/off) and post-assessment calibration level (none/B/C), combined with temporal aggregation variants:

| Config | Anchors (A) | Rules (B) | Agent (C) | Temporal | Status |
|--------|:-:|:-:|:-:|:--------:|--------|
| **A0-T0** | No | No | No | T0 | Offline baseline |
| **A1-T0** | Yes | No | No | T0 | Offline |
| **A1-T2** | Yes | No | No | T2 | = Official Run 2 |
| **A3-T2** | Yes | Yes | No | T2 | = Official Run 1 |
| **A3-T3** | Yes | Yes | No | T3 | Offline variant |
| **A5-T2** | Yes | Yes | Yes | T2 | Offline variant |
| **A5-T3** | Yes | Yes | Yes | T3 | = Official Run 0 |

### 6.2 Trial Data Results (19 Rounds, Single Patient)

All ablation configurations were evaluated against the trial gold standard. Results from the final ablation summary:

#### Table 1: Final-Round Metrics (Round 19)

| Config | PHQ-9 RMSE | GAD-7 RMSE | CompACT RMSE | Mean RMSE | Mean Pearson | PHQ-9 Total | GAD-7 Total | CompACT Total |
|--------|:----------:|:----------:|:------------:|:---------:|:------------:|:-----------:|:-----------:|:-------------:|
| A0-T0 (baseline) | 0.577 | 0.378 | 0.548 | 0.501 | 0.643 | 14 | 13 | 34 |
| A1-T0 (anchors only) | 0.333 | 0.535 | 0.837 | 0.568 | 0.708 | 14 | 16 | 38 |
| **A1-T2 (Run 2)** | **0.000** | 0.535 | 0.837 | 0.457 | 0.818 | 13 | 16 | 40 |
| **A3-T2 (Run 1)** | **0.000** | 0.535 | 0.837 | 0.457 | 0.818 | 13 | 16 | 40 |
| A3-T3 | **0.000** | 0.535 | 0.707 | 0.414 | 0.697 | 13 | 16 | 38 |
| A5-T2 | **0.000** | 0.655 | 0.837 | 0.497 | 0.808 | 13 | 17 | 40 |
| **A5-T3 (Run 0)** | **0.000** | **0.378** | 0.837 | **0.405** | **0.842** | 13 | 15 | 40 |

Gold: PHQ-9 = 13, GAD-7 = 14, CompACT-10 = 33.

#### Table 2: CompACT-10 Subscale Analysis

| Config | OtE RMSE | BA RMSE | VA RMSE | VA Mean (gold: 3.5) | Level B Violations | Level C Corrections |
|--------|:--------:|:-------:|:-------:|:-------------------:|:------------------:|:-------------------:|
| A0-T0 | 0.577 | 0.000 | 0.707 | 4.0 | 0 | 0 |
| A1-T0 | 0.817 | 0.577 | 1.000 | 4.5 | 0 | 0 |
| A1-T2 | 0.817 | 0.577 | 1.000 | 4.5 | 0 | 0 |
| A3-T2 | 0.817 | 0.577 | 1.000 | 4.5 | 4 | 0 |
| A3-T3 | 0.817 | 0.577 | **0.707** | 4.0 | 5 | 0 |
| A5-T2 | 0.817 | 0.577 | 1.000 | 4.5 | 1 | 1 |
| A5-T3 | 0.817 | 0.577 | 1.000 | 4.5 | 4 | 2 |

#### Table 3: Kappa Agreement (Final Round)

| Config | PHQ-9 kappa_q | GAD-7 kappa_q | CompACT kappa_q |
|--------|:------------:|:------------:|:---------------:|
| A0-T0 | 0.716 | 0.800 | 0.348 |
| A1-T2 | 1.000 | 0.667 | 0.364 |
| A3-T2 | 1.000 | 0.667 | 0.364 |
| A5-T3 | 1.000 | 0.800 | 0.364 |

### 6.3 Convergence Trajectories

Per-round mean RMSE trajectories show how quickly each configuration stabilizes:

| Config | Round 1 | Round 3 | Round 5 | Round 9 | Round 15 | Round 19 |
|--------|:-------:|:-------:|:-------:|:-------:|:--------:|:--------:|
| A0-T0 | 0.494 | 0.525 | 0.211 | 0.211 | 0.211 | 0.501 |
| A1-T2 | 0.464 | 0.361 | 0.414 | 0.414 | 0.414 | 0.457 |
| A3-T2 | 0.464 | 0.497 | 0.405 | 0.405 | 0.405 | 0.457 |
| A5-T3 | 0.529 | 0.405 | 0.405 | 0.405 | 0.405 | 0.405 |

A5-T3 achieves its best performance by round 4 and maintains it consistently. The temporal aggregation layer (T2/T3) stabilizes predictions from round 5 onward, preventing late-round degradation that affects the T0 baseline (which deteriorates from 0.211 at round 5 to 0.501 at round 19).

### 6.4 Simulated Persona Ablation Results (6 Sessions, 15 Rounds Each)

| Config | Mean RMSE | PHQ-9 RMSE | GAD-7 RMSE | CompACT RMSE |
|--------|:---------:|:----------:|:----------:|:------------:|
| A0-T0 | 1.415 | 0.951 | 1.299 | 1.996 |
| A1-T0 | 1.412 | 0.944 | 1.245 | 2.047 |
| **A1-T2** | **1.272** | 0.909 | **0.943** | 1.964 |
| A3-T2 | 1.311 | 0.916 | 1.021 | 1.996 |
| A3-T3 | 1.275 | 0.916 | 0.971 | **1.938** |
| A5-T2 | 1.288 | 0.916 | 0.993 | 1.954 |
| A5-T3 | 1.320 | 0.970 | 0.989 | 2.000 |

### 6.5 Effect Decomposition

**Prompt Anchors (Level A)**: A0 vs A1 shows anchors improve PHQ-9 dramatically (0.577 → 0.333 RMSE on trial) and Pearson correlation (+0.065). On simulated data, anchors provide a modest RMSE reduction.

**Temporal Aggregation (T2 vs T0)**: Early-weighted aggregation reduces mean RMSE by 0.10--0.16 across configurations. T2 prevents the late-round degradation that the baseline suffers, which is the single largest source of error. On simulated data, temporal aggregation provides the most consistent improvement (-0.14 mean RMSE).

**Rule-Based Constraints (Level B)**: A3 vs A1 shows negligible effect on trial data (same RMSE). Level B acts defensively: it rarely fires (0--5 violations per 19-round session) but catches occasional outliers. On simulated data, adding Level B to A1-T2 slightly increases RMSE (1.272 → 1.311), suggesting over-correction risk with limited data.

**LLM Calibration Agent (Level C)**: A5-T3 vs A3-T2 shows -0.052 mean RMSE improvement on trial. Level C fires 2 corrections across 19 rounds. The main benefit is on GAD-7 (0.535 → 0.378 RMSE), where the agent corrects context-sensitive discordances that rules cannot. On simulated data, the agent provides no additional benefit, suggesting its value is in handling edge cases.

**Interaction effects**: Level A + temporal aggregation is the most impactful combination. The full stack (A5-T3) wins on trial data by combining Level A's detection improvement with temporal stabilization and occasional Level C corrections. On simulated data, the simpler A1-T2 configuration generalizes best.

### 6.6 Key Findings

1. **PHQ-9 is essentially solved** by prompt anchors + temporal aggregation: perfect RMSE (0.000) and Pearson (1.000) on trial data from A1-T2 onward.

2. **GAD-7 has a variance problem** from within-category item discrimination failures. Level C is the only component that substantially improves GAD-7 (from 0.535 to 0.378 RMSE on trial).

3. **CompACT-10 remains the most challenging instrument** (RMSE 0.71--0.84 across all configurations, ~2.0 on simulated data). The Valued Action subscale is systematically over-scored due to the LLM conflating within-session therapeutic engagement with general life patterns.

4. **Temporal aggregation is the single highest-value component**, preventing the late-round recency bias degradation that pushes baseline RMSE from 0.211 (round 5) to 0.501 (round 19).

5. **The self-contradiction guard (Rule C4) preserves clinical validity**: 19% of patients genuinely exhibit high VA + high distress, and the guard prevents false corrections of this real pattern.

---

## 7. References

- Baker, K. D., & Berghoff, C. R. (2021). A network analysis of psychological flexibility processes and depression. *Journal of Contextual Behavioral Science*, 19, 56--62.
- Francis, A. W., Dawson, D. L., & Golijani-Moghaddam, N. (2016). The development and validation of the Comprehensive assessment of Acceptance and Commitment Therapy processes (CompACT). *Journal of Contextual Behavioral Science*, 5(3), 134--145.
- Golijani-Moghaddam, N., Dawson, D. L., & Sherwood, L. (2023). The CompACT-10: Development of a short-form of the Comprehensive assessment of Acceptance and Commitment Therapy processes. *Journal of Contextual Behavioral Science*, 30, 57--67.
- Gratch, J., Artstein, R., Lucas, G., Stratou, G., Scherer, S., Nazarian, A., ... & Morency, L.-P. (2014). The Distress Analysis Interview Corpus of human and computer interviews. *Proceedings of LREC 2014*.
- Gupta, D., Suman, S., & Ekbal, A. (2022). Learning to Automate Follow-up Question Generation using Process Knowledge for Depression Triage on Reddit Posts. *Proceedings of the Eighth Workshop on Computational Linguistics and Clinical Psychology (CLPsych)*, 137--147. NAACL 2022.
- Kroenke, K., Wu, J., Yu, Z., Bair, M. J., Kean, J., Stump, T., & Monahan, P. O. (2016). Patient Health Questionnaire Anxiety and Depression Scale: Initial validation in three clinical trials. *Psychosomatic Medicine*, 78(6), 716--727.
- Melbye, S., Kessing, L. V., Bardram, J. E., & Faurholt-Jepsen, M. (2022). Validation of the Patient Health Questionnaire-9 and the Generalized Anxiety Disorder-7 in outpatients with mood disorders using item response theory and network analysis. *Journal of Affective Disorders*, 312, 162--168.
- Ruiz, F. J., Garcia-Martin, M. B., Suarez-Falcon, J. C., & Odriozola-Gonzalez, P. (2024). The hierarchical factor structure of the Spanish Acceptance and Action Questionnaire-II (AAQ-II). *Journal of Contextual Behavioral Science*, 31, 100724.
- Ryan, T. A., Bailey, A., Fearon, P., & King, J. (2022). Factorial invariance of the Patient Health Questionnaire and Generalized Anxiety Disorder Questionnaire: a large-scale study. *The British Journal of Clinical Psychology*, 61(1), 245--260.
