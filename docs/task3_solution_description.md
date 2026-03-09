# Task 3: ADHD Symptom Sentence Ranking — Solution Description (HiPerT)

## 1. Task Overview

eRisk 2026 Task 3 requires systems to rank sentences from a large Reddit corpus by their relevance to each of 18 ASRS-v1.1 (Adult ADHD Self-Report Scale) symptoms. For each symptom, systems produce a ranked list of up to 1,000 sentences, each scored on a 0–3 relevance scale:

| Score | Meaning |
|-------|---------|
| **0** | Irrelevant — no connection to the symptom |
| **1** | Marginally relevant — vague, indirect, or ambiguous connection |
| **2** | Moderately relevant — clear symptom reference + personal experience, but lacks specificity |
| **3** | Highly relevant — explicit personal experience with concrete behavioral detail |

**Corpus:** 4,521 TREC files containing ~4.17 million sentences from Reddit, each with contextual triplets (PRE, TEXT, POST). Sentence identifiers follow the format `{userId}_{contextId}_{sentenceIdx}`.

---

## 2. ASRS-v1.1 Symptom Structure

The 18 ASRS items are organized into a **three-factor bifactor model** (Panagiotidi et al. 2024, Stanton et al. 2018) with seven subclusters:

| Factor | Subcluster | Items | Example Question |
|--------|-----------|-------|-----------------|
| **Inattention (F_IN)** | Organization/Planning | 1–2 | "How often do you have difficulty wrapping up the final details of a project?" |
| | Memory/Avoidance | 3–4 | "How often do you have problems remembering appointments or obligations?" |
| | Sustained Attention | 7–11 | "How often do you have difficulty concentrating on what people say to you?" |
| **Motor H/I (F_MH)** | Fidgeting/Restlessness | 5, 13 | "How often do you fidget or squirm with your hands or feet?" |
| | Internal Drive/Settling | 6, 12, 14 | "How often do you feel overly active and compelled to do things?" |
| **Verbal H/I (F_VH)** | Output Control | 15–16 | "How often do you find yourself talking too much?" |
| | Turn-Taking | 17–18 | "How often do you finish the sentences of people you are talking to?" |

### 2.1 Four-Layer Clinical Definitions

Each symptom is grounded in a four-layer clinical definition framework totaling ~12,000 words across all 18 items:

1. **L1 — Clinical Definition (DSM-5-TR):** Formal diagnostic criterion from the DSM-5-TR (APA 2022, pp. 68–70), with adult-specific adaptations distinguishing childhood from adult presentations.

2. **L2 — Adult Behavioral Manifestation (Barkley BAARS-IV / Ramsay & Rostain):** Concrete daily-life examples across work, home, and social domains — e.g., "90% completion paralysis on administrative tasks" for Item 1, "scrolling or channel-surfing at bedtime" for Item 7.

3. **L3 — Empirical Discussion Topics:** Language patterns from Reddit r/ADHD (372K posts analyzed via BERTopic, Kang et al. JMIR 2025) and Twitter (Guntuku et al. 2019) — how symptoms manifest in self-report social media language. E.g., "my brain doesn't have an off switch" for Item 7.

4. **L4 — Differential Markers:** Distinguishing ADHD from depression, anxiety, autism, OCD, and normal variation. E.g., for Item 7: GAD = worry-driven inability to relax; PTSD = trauma triggers; ADHD = boredom intolerance and understimulation.

### 2.2 Token Budget Strategy

Not all items require the same elaboration depth. We allocate prompt space based on cross-diagnostic ambiguity:

| Budget | Layers Included | Items | Rationale |
|--------|----------------|-------|-----------|
| **full_4** (~400–500 tokens) | L1 + L2 + L3 + L4 | 7–11 | Highest cross-diagnostic overlap with depression/anxiety |
| **compressed_3** (~250–350 tokens) | L1 + L3 + L4 | 1–4, 6, 13–14 | Moderate ambiguity |
| **minimal_2** (~150–200 tokens) | L1 + L3 | 5, 12, 15–18 | Concrete observable behaviors, few confounders |

---

## 3. System Architecture

```
INPUT CORPUS (4.17M Reddit sentences)
         │
    ┌────▼─────────────────────────────────────────────────────┐
    │  STAGE 1: CANDIDATE SELECTION                            │
    │  Bi-encoder (all-mpnet-base-v2) → top-K candidates       │
    │  + query expansion + first-person filter + keyword boost  │
    └────┬─────────────────────────────────────────────────────┘
         │  ~5K–10K candidates per symptom
    ┌────▼─────────────────────────────────────────────────────┐
    │  STAGE 2: LLM SCORING CASCADE                            │
    │  Primary LLM scores ALL candidates                       │
    │  → 5 escalation rules → GPT re-scores ~20–25% of cases  │
    │  → confidence-weighted silver labels                     │
    └────┬─────────────────────────────────────────────────────┘
         │  ~30K–36K silver label triples
    ┌────▼─────────────────────────────────────────────────────┐
    │  STAGE 3: TRAINING                                       │
    │  Stage A: Depression pre-training (BDI-Sen + eRisk 2025) │
    │  Stage B: ADHD fine-tuning (silver labels + curriculum)  │
    │  3 backbone ensemble (mental-roberta, clinical-bert, mpnet) │
    └────┬─────────────────────────────────────────────────────┘
         │
    ┌────▼─────────────────────────────────────────────────────┐
    │  STAGE 4: INFERENCE & ENSEMBLE                           │
    │  3-backbone averaged expected scores                     │
    │  Per-symptom temperature scaling + Dirichlet calibration │
    └────┬─────────────────────────────────────────────────────┘
         │
    ┌────▼─────────────────────────────────────────────────────┐
    │  OUTPUT: 5 SUBMISSION RUNS                               │
    │  R1=Encoder, R2=LLM, R3=RRF Ensemble, R4=Transfer, R5=BiEnc │
    └──────────────────────────────────────────────────────────┘
```

---

## 4. Stage 1: Candidate Selection (Retrieval)

### 4.1 Bi-Encoder Retrieval

- **Model:** `all-mpnet-base-v2` (sentence transformer)
- **Query expansion:** 2–3 paraphrases per ASRS symptom, concatenated with the original question text
- **Top-K:** 5,000 candidates per symptom (×2 internal over-retrieval, then filtered)
- **Similarity:** Cosine similarity between query and sentence embeddings

### 4.2 First-Person Filter

Retains only sentences containing first-person markers (`I`, `me`, `my`, `mine`, `myself`, `I'm`, `I've`, `I'd`, `I'll`). This eliminates ~60% of the corpus — third-person references, advice, and general statements are rarely relevant at score ≥ 2.

### 4.3 Keyword Boost

Each symptom has 16–18 curated keywords (from Layer 3 empirical vocabulary). A `+0.05` boost per keyword match is added to the cosine similarity score. Example keywords for Item 1 (Wrapping up projects): *organize, finish, procrastinate, deadline, last-minute, follow-through, incomplete*.

### 4.4 Output

Per symptom: a ranked list of ~5,000–10,000 candidate sentences with `combined_score = cosine_similarity + keyword_boost`. These candidates feed into both the LLM scoring cascade (Stage 2) and the BiEnc baseline run (Run 5).

---

## 5. Stage 2: LLM Scoring Cascade

### 5.1 Primary Scoring

Every candidate receives a structured LLM assessment. The prompt consists of:

1. **System prompt:** Clinical psychologist specializing in adult ADHD assessment, with the 0–3 scoring rubric.
2. **Symptom definition:** Clinical elaboration respecting the token budget (full_4 / compressed_3 / minimal_2).
3. **Four few-shot examples:** One per score level (0, 1, 2, 3) from curated annotations.
4. **Sentence context:** PRE, [TARGET], POST markers around the sentence to score.

**Structured output template (7 fields):**
```
SYMPTOM_MATCH: [YES|PARTIAL|NO]
SELF_REFERENCE: [DIRECT|INDIRECT|NONE]
DETAIL_LEVEL: [HIGH|MEDIUM|LOW|NONE]
CONFOUNDERS: [list alternatives or "NONE"]
SCORE: [0|1|2|3]
CONFIDENCE: [1|2|3|4|5]
REASONING: [1–2 sentences]
```

### 5.2 Few-Shot Examples

72 annotated examples (4 per symptom × 18 symptoms), sourced from:

| Source | Use | Count |
|--------|-----|-------|
| **score0_random** | Irrelevant control sentences | ~18 |
| **BDI-Sen** | Depression confounders at score 1–2 | ~25 |
| **Retrieval candidates** | High-scoring corpus sentences | ~20 |
| **RedSM5** | Clinician-annotated depression sentences | ~9 |

Each example includes the full context triplet and a structured annotation matching the 7-field template. Score-1 examples are deliberately chosen from depression datasets to calibrate the boundary between "depression-related" and "ADHD-related" language.

### 5.3 Escalation Rules

Five rules trigger re-scoring by a stronger model (GPT-4o):

| Rule | Trigger | Target Rate |
|------|---------|-------------|
| **1. Low Confidence** | CONFIDENCE ≤ 2 | ~5–10% |
| **2. Internal Inconsistency** | Contradictions between fields (e.g., SYMPTOM_MATCH=NO but SCORE ≥ 2; SELF_REFERENCE=NONE but SCORE ≥ 1; DETAIL_LEVEL=HIGH but SCORE ≤ 1) | ~5% |
| **3. Confounder + Borderline** | CONFOUNDERS ≠ "NONE" AND SCORE ∈ {1, 2} | ~8–12% |
| **4. Inattention Cross-Diagnostic** | Items 7–11 (Sustained Attention) with SCORE ≥ 2 AND cross-diagnostic keywords (depression, anxiety, fatigue, sleep, etc.) | ~3–5% |
| **5. Moderate Confidence on Boundary** | SCORE ∈ {1, 2} AND CONFIDENCE = 3 | ~5–8% |

**Expected overall escalation rate:** ~20–25% (concentrated on items 7–11 at 30–45%, minimal for Motor/Verbal H/I at 10–15%).

### 5.4 Resolution Logic

After primary scoring (and optional GPT re-scoring), each (sentence, symptom) pair receives a final label and confidence weight:

**Base symptom weights** (reflecting cross-diagnostic reliability):
- Motor H/I (items 5–6, 12–14): **1.0** — concrete, observable, few confounders
- Verbal H/I (items 15–18): **0.9** — mostly observable
- Organization/Memory (items 1–4): **0.7** — moderate depression overlap
- Sustained Attention (items 7–11): **0.5** — strongest depression-ADHD overlap

**Resolution weight** (from scoring confidence):

| Scenario | Resolution Weight |
|----------|-------------------|
| Not escalated, confidence ≥ 4 | 0.80 |
| Not escalated, confidence = 3 | 0.60 |
| Escalated, GPT agrees (diff = 0) | 0.85 |
| Escalated, GPT differs by 1 | 0.65 |
| Escalated, GPT differs by ≥ 2 | 0.45 (trust GPT) |
| Both uncertain + diff ≥ 2 | 0.30 (use floor(mean)) |

**Final confidence_weight = resolution_weight × symptom_weight** (range: 0.15–0.85)

---

## 6. Data Sources and Cross-Dataset Mappings

A key innovation of HiPerT is leveraging **depression datasets as calibrated confounders and pre-training sources** for ADHD symptom ranking. This is motivated by ~30% ADHD-depression comorbidity and strong surface-level symptom overlap (concentration difficulty, fatigue, agitation, sleep disruption).

### 6.1 Data Source Inventory

| Dataset | Size | Format | Labels | Role in Pipeline |
|---------|------|--------|--------|-----------------|
| **eRisk 2026 Task 3 Corpus** | 4,521 files, ~4.17M sentences | TREC (PRE/TEXT/POST) | Unlabeled | Primary ranking corpus |
| **BDI-Sen 2.0** | 2,529 sentences, 5,003 annotations | JSONL | Graded 0–3, 21 BDI-II symptoms | Stage A1 pre-training; few-shot confounders |
| **eRisk 2025 Task 1** | 11,042 judgments | CSV + TREC | Binary (majority + consensus) | Stage A2 pre-training; boundary candidates |
| **eRisk 2023 Task 1** | 21,580 judgments | CSV + TREC | Binary (majority + consensus) | Optional additional pre-training |
| **RedSM5** | 1,484 posts, 2,058 annotations | CSV | Binary + DSM-5 categories | Few-shot confounders; cross-validation |
| **Few-shot annotations** | 72 examples (4 × 18 symptoms) | JSON | Graded 0–3 + structured fields | LLM scoring calibration |

### 6.2 BDI-II → ASRS Symptom Mappings

Depression symptoms from BDI-II (items 1–21) are mapped to overlapping ASRS items based on shared behavioral manifestations:

| BDI-II Query | Depression Symptom | → ASRS Items | ADHD Manifestation |
|-------------|-------------------|-------------|-------------------|
| Q19 | Concentration Difficulty | 8, 9, 10, 11 | Sustained Attention/Distractibility |
| Q11 | Agitation | 5, 6, 12, 13 | Motor Hyperactivity/Impulsivity |
| Q15 | Loss of Energy | 4, 7 | Memory/Avoidance, Difficulty Unwinding |
| Q16 | Changes in Sleep | 7 | Difficulty Unwinding |
| Q13 | Indecisiveness | 1, 2 | Organization/Planning |

**Unmapped ASRS items:** 3, 14, 15, 16, 17, 18 — these have no direct BDI-II analog and rely solely on ADHD-specific retrieval and scoring.

### 6.3 BDI-Sen Symptom Mappings

Six BDI-Sen symptoms map to ASRS items:

| BDI-Sen Symptom | → ASRS Items | Usage |
|-----------------|-------------|-------|
| Concentration_difficulty | 8–11 | Stage A training + score-1 confounders |
| Agitation | 5–6, 12–13 | Stage A training + score-1 confounders |
| Loss_of_energy | 4, 7 | Stage A training |
| Tiredness_or_fatigue | 4, 7 | Stage A training |
| Indecision | 1–2 | Stage A training |
| Change_of_sleep | 7 | Stage A training |

**Confounder strategy:** BDI-Sen sentences at severity=1 (mildest depression relevance) are used as calibrated score-1 few-shot examples — they represent the maximal ADHD ambiguity zone where depression language most closely resembles ADHD symptoms.

### 6.4 RedSM5 DSM-5 → ASRS Mappings

Five DSM-5 depression categories from RedSM5 (clinician-annotated) map to ASRS items:

| DSM-5 Category | → ASRS Items |
|---------------|-------------|
| COGNITIVE_ISSUES | 7, 8, 9, 10, 11 |
| PSYCHOMOTOR | 5, 6, 12, 13 |
| FATIGUE | 4, 7 |
| SLEEP_ISSUES | 7 |
| ANHEDONIA | 4, 11 |

### 6.5 Boundary Candidate Extraction

From eRisk 2023 and 2025, we extract **disagreement cases** — sentences where majority vote labeled as relevant but consensus labeled as irrelevant (or vice versa). These ~2,000–2,700 boundary cases per dataset represent the hardest scoring decisions and are particularly valuable as score-1 training examples.

### 6.6 Coverage Summary

| ASRS Item | BDI-Sen | BDI-II Qrels | RedSM5 | External Sources |
|-----------|:-------:|:------------:|:------:|:----------------:|
| 1–2 | ✓ (Indecision) | ✓ (Q13) | — | 2 |
| 3 | — | — | — | 0 (ADHD-only) |
| 4 | ✓ (Energy, Fatigue) | ✓ (Q15) | ✓ (FATIGUE) | 3 |
| 5–6 | ✓ (Agitation) | ✓ (Q11) | ✓ (PSYCHOMOTOR) | 3 |
| 7 | ✓ (Energy, Sleep) | ✓ (Q15, Q16) | ✓ (FATIGUE, SLEEP) | 4 |
| 8–11 | ✓ (Concentration) | ✓ (Q19) | ✓ (COGNITIVE) | 3 |
| 12–13 | ✓ (Agitation) | ✓ (Q11) | ✓ (PSYCHOMOTOR) | 3 |
| 14–18 | — | — | — | 0 (ADHD-only) |

13 of 18 ASRS items have at least one external depression data source; the remaining 5 (items 3, 14, 15–18) rely solely on corpus retrieval and LLM scoring.

---

## 7. Stage 3: Training

### 7.1 Encoder Architecture

**SymptomConditionedEncoder** — a symptom-aware sentence relevance model:

```
Input: (text, pre_context, post_context, symptom_id)
   │
   ▼
Backbone Encoder (one of 3 choices)
   ├─ h_pre  = CLS(encode(PRE))
   ├─ h_text = CLS(encode(TEXT))
   └─ h_post = CLS(encode(POST))
   │
   ▼
Context Fusion (Multi-Head Attention, 4 heads)
   Q = h_text,  K/V = [h_pre, h_text, h_post]
   h_fused = LayerNorm(h_text + Attn(Q, K, V))
   │
   ▼
Symptom Conditioning
   e_q = SymptomEmbedding(symptom_id)   # learnable, 18 or 21 entries
   z = [h_fused ⊙ e_q ; h_fused + e_q]  # Hadamard + additive fusion
   │
   ▼
Projection: Linear(2d → d) → LayerNorm → GELU → Dropout(0.1)
   │
   ▼
Ordinal Classifier: Linear(d → 4)  →  logits for {0, 1, 2, 3}
```

**Three backbone models** (trained independently, ensembled at inference):

| Backbone | Model ID | Parameters | Domain |
|----------|---------|-----------|--------|
| **mental-roberta** | `mental/mental-roberta-base` | 110M | Mental health text |
| **clinical-bert** | `emilyalsentzer/Bio_ClinicalBERT` | 110M | Clinical notes |
| **mpnet** | `sentence-transformers/all-mpnet-base-v2` | 420M | General-purpose |

### 7.2 Loss Functions

**Composite loss:** `L_total = λ₁·L_ord + λ₂·L_rank + λ₃·L_hier` (default λ = 1.0, 0.5, 0.05)

| Component | Description | Key Detail |
|-----------|-------------|------------|
| **Ordinal Cross-Entropy (L_ord)** | Soft-target CE with label smoothing toward adjacent levels | ε=0.2: core label gets 0.8, each neighbor gets 0.1 |
| **Symmetric Cross-Entropy (L_sce)** | Bidirectional CE for noise robustness (Stage B only) | α·CE(p,q) + β·CE(q,p), α=1.0, β=0.5 |
| **Margin Ranking (L_rank)** | Pairwise ordinal consistency enforcement | For pairs where label_i > label_j: max(0, γ − (score_i − score_j)), γ=0.3 |
| **Hierarchy Regularization (L_hier)** | Symptom embeddings respect clinical hierarchy | Same-factor attraction + weak cross-factor repulsion (λ_repel=0.01) |
| **Silhouette Contrastive (L_sil)** | Ordinal-weighted separation in embedding space | w(r,r') = \|r−r'\|/3 — novel ordinal silhouette loss |

### 7.3 Stage A: Depression Pre-Training

Pre-trains on labeled depression data with BDI-II symptom IDs (1–21), leveraging the structural similarity between BDI-II and ASRS assessments.

#### Stage A1: BDI-Sen Graded Training

| Parameter | Value |
|-----------|-------|
| **Data** | 2,529 sentences, 5,003 (sentence, symptom) pairs |
| **Labels** | Graded 0–3 (severity) |
| **Context** | None (BDI-Sen has text only) |
| **Symptoms** | 21 BDI-II items |
| **Split** | 90/10 train/val (2,276 / 253) |
| **LR** | 2e-5 |
| **Epochs** | 10 |
| **Batch** | 32 |
| **Backbone freezing** | 6 layers frozen → unfreeze at epoch 3 |
| **Loss** | Ordinal CE + Margin Ranking + Hierarchy Reg |
| **Early stopping** | patience=3, min_delta=0.001 |

#### Stage A2: eRisk 2025 Contextual Fine-Tuning (Optional)

| Parameter | Value |
|-----------|-------|
| **Data** | 11,042 binary judgments with PRE/TEXT/POST context |
| **Labels** | Binary 0/1 (majority vote), with consensus-weighted confidence (1.0 if consensus, 0.8 if majority-only) |
| **Context** | Full PRE/TEXT/POST triplets |
| **Symptoms** | 21 BDI-II items |
| **Split** | 90/10 train/val (9,938 / 1,104) |
| **LR** | 1e-5 (half of A1) |
| **Epochs** | 5 |
| **Initialization** | A1 best checkpoint |
| **Early stopping** | patience=2 |

**Key contribution of A2:** Introduces contextual encoding — the model learns to use surrounding sentences (PRE/POST) for disambiguation, which is critical for the target TREC corpus where context is always available.

### 7.4 Stage B: ADHD Fine-Tuning with Curriculum Learning

| Parameter | Value |
|-----------|-------|
| **Data** | ~30K–36K silver labels from LLM scoring cascade |
| **Labels** | Graded 0–3 with confidence weights (0.3–1.0) |
| **Context** | Full PRE/TEXT/POST triplets |
| **Symptoms** | 18 ASRS items (resized from 21 BDI-II) |
| **Split** | 90/10 train/val (stratified by symptom) |
| **LR** | 1e-5 (conservative) |
| **Epochs** | 15 |
| **Initialization** | Stage A best checkpoint |
| **Loss** | Symmetric CE (noise-robust) + Margin Ranking + Hierarchy Reg |
| **Early stopping** | patience=5 |

#### Curriculum Learning

Silver labels have heterogeneous noise levels — Motor/Verbal H/I labels are more reliable (concrete behaviors) while Sustained Attention labels are noisier (depression-ADHD overlap). The curriculum introduces examples in order of reliability:

**Difficulty priors** (per symptom):
- Motor H/I (items 5, 6, 12–14): π = 0.2–0.25 (easy)
- Verbal H/I (items 15–18): π = 0.3 (easy-medium)
- Organization/Memory (items 1–4): π = 0.5 (medium)
- Sustained Attention (items 7–11): π = 0.8 (hard)

**Competence function:** c(t) = min(1, (0.01^p + t/T·(1−0.01^p))^(1/p)) with p=2 (quadratic growth)

**Inclusion probability:** P(s,t) = σ(β(t)·(c(t) − difficulty(s))), where β(t) = 5.0·c(t)

**Three natural phases:**
1. **Phase 1** (c < 0.35): Motor H/I + Verbal H/I only — the model first learns to recognize concrete, observable ADHD behaviors.
2. **Phase 2** (0.35 ≤ c < 0.65): Organization/Memory added — introduces moderate cross-diagnostic ambiguity.
3. **Phase 3** (c ≥ 0.65): Full symptom set including Sustained Attention — the hardest items are introduced last, after the model has a strong foundation in unambiguous ADHD features.

### 7.5 Calibration

Post-training, a two-stage calibration is fitted on the validation set:

1. **Per-symptom temperature scaling:** Learns T_j for each symptom j. Calibrated probabilities: p_cal(r|s,q_j) = softmax(logits/T_j). Temperatures clamped to [0.1, 10.0], fitted via L-BFGS.

2. **Dirichlet calibration:** Learns a linear transform in log-probability space: p_cal = softmax(W·log(p) + b), with W initialized to identity.

**Final expected score:** φ(s,q) = Σ_r r · p_cal(r|s,q) for r ∈ {0,1,2,3}

### 7.6 Checkpointing Strategy

- Every 500 steps (Stage A) / 300 steps (Stage B)
- Every epoch
- Best validation metric
- On interrupt (SIGINT/SIGTERM)
- Final checkpoint
- Keep last 5 (prevents disk overflow)
- Format: `{stage}_{backbone}_{tag}.pt` (e.g., `stage_b_mpnet_best.pt`)

---

## 8. Stage 4: Inference & Ensemble

### 8.1 Single-Backbone Inference

For each backbone and each of 18 symptoms:
1. Load best checkpoint and calibration parameters
2. Encode all candidates with context (PRE/TEXT/POST)
3. Compute calibrated expected scores: φ(s,q) = Σ_r r · p_cal(r|s,q)
4. Rank by score, retain top 1,000

### 8.2 Three-Backbone Ensemble

Scores are averaged across all three backbones:

```
φ_ensemble(s,q) = (1/3) · [φ_mental-roberta(s,q) + φ_clinical-bert(s,q) + φ_mpnet(s,q)]
```

Union of docnos across backbones (missing scores treated as 0), sorted descending, top 1,000 per symptom.

---

## 9. Submission Runs

We submit five complementary runs, each using a different combination of system components:

| Run | System Name | Method | Dependencies | Fallback |
|-----|-----------|--------|--------------|----------|
| **R1** | HiPerTHiPerT_full | 3-backbone trained encoder ensemble | Stage A + B training, inference | — |
| **R2** | HiPerTLLM_cascade | LLM scores + confidence + cosine tie-breaking | LLM scoring complete | — |
| **R3** | HiPerTEnsemble | RRF fusion of R1 + R2 | R1 + R2 | RRF(R5 + R2) if R1 unavailable |
| **R4** | HiPerTDepTransfer | Stage A only, BDI-II → ASRS mapping | Stage A training | R5 for unmapped items (3, 14–18) |
| **R5** | HiPerTBiEnc_baseline | Cosine similarity from retrieval | Retrieval complete | — |

### 9.1 Run 1: HiPerT_full (Encoder Ensemble)

The full pipeline. Ranking by calibrated ensemble expected score:

```
φ(s,q) = (1/3) · Σ_k Σ_r r · p̂_cal,k(r | s, q)
```

### 9.2 Run 2: LLM_cascade (LLM Scoring Only)

No trained encoder — purely LLM-based with hierarchical tie-breaking:

```
composite_score = label × 10.0              # Primary: final LLM label (0–3)
                + confidence_weight × 5.0    # Secondary: resolution confidence
                + llm_confidence × 0.01      # Tertiary: LLM internal confidence (1–5)
                + cosine × 0.001             # Quaternary: retrieval similarity
```

### 9.3 Run 3: Ensemble (RRF Fusion)

Reciprocal Rank Fusion combining encoder and LLM rankings:

```
score_rrf(s) = 1/(60 + rank_encoder(s)) + 1/(60 + rank_llm(s))
```

If R1 is unavailable, falls back to RRF(R5, R2). Parameter-free, no tuning needed.

### 9.4 Run 4: DepTransfer (Cross-Condition Transfer)

Uses only Stage A (depression pre-training) without ADHD fine-tuning. The BDI-II → ASRS mapping transfers depression knowledge to 12/18 ASRS items. Unmapped items (3, 14, 15–18) fall back to Run 5 (cosine baseline).

**Motivation:** Tests whether depression pre-training alone provides competitive ADHD symptom ranking, given the ~30% comorbidity and shared executive function deficits.

### 9.5 Run 5: BiEnc_baseline (Cosine Similarity)

Simplest system — no LLM, no training. Ranks by combined retrieval score (cosine similarity + keyword boost + first-person indicator). Serves as baseline and fallback.

---

## 10. Validation Strategy

### 10.1 Internal Validation

| Method | Description | Threshold |
|--------|-------------|-----------|
| **BDI-II cross-validation** | Run LLM pipeline on BDI-Sen sentences with known labels | κ ≥ 0.5 |
| **Silver label holdout** | 80/20 train/val split on silver labels | MAP, nDCG@100 |
| **Qualitative audit** | 20 sentences × 3 score levels per symptom | Face validity |

### 10.2 Per-Symptom Quality Monitoring

For each of the 18 symptoms:
- **Escalation rate** and primary triggers (which rules fire most frequently)
- **Score distribution** across {0, 1, 2, 3} — detect systematic over/under-scoring
- **Primary-escalation agreement rate** (target: >50% agreement on escalated cases)
- **Reliability flag** if agreement < 50% — triggers manual review

### 10.3 Cross-Dataset Calibration

BDI-Sen severity levels (0–3) provide ground truth for calibrating the LLM scoring pipeline on 6 overlapping symptoms. This allows measuring:
- LLM overestimation bias (expected: 0.3–0.5 points on average)
- Per-symptom calibration curves
- Isotonic regression parameters for bias correction

---

## 11. LLM Configuration

| Role | Model | Provider | Temperature | Notes |
|------|-------|----------|-------------|-------|
| **Primary scorer** | GPT-4o-mini | OpenAI | 0.1 | Structured output, all candidates |
| **Escalation scorer** | GPT-4o | OpenAI | 0.1 | Re-scores ~20–25% of candidates |
| **Local alternative** | Llama-3.1-70B | Ollama | 0.1 | Self-hosted, cost-free |

Rate limit: 0.1s delay between calls, max 5 retries, 120s read timeout, batch size 50.

---

## 12. Key Design Decisions and Rationale

1. **Depression pre-training for ADHD** — BDI-II and ASRS share 12/18 overlapping behavioral constructs (concentration, agitation, energy, sleep, indecision). Pre-training on labeled depression data provides a strong initialization for ADHD symptom recognition, especially for the hardest items (7–11) where both conditions manifest as attention deficits.

2. **Curriculum learning by clinical difficulty** — Silver labels from LLM scoring have heterogeneous noise. Motor/Verbal items are straightforward (concrete behaviors), while Sustained Attention items are systematically noisier due to depression-ADHD overlap. Training on easy items first builds a reliable foundation before exposing the model to ambiguous cases.

3. **Symmetric cross-entropy for noise robustness** — Silver labels are imperfect (LLM-generated). The reverse KL term in symmetric CE penalizes overconfident predictions on noisy labels, preventing the model from memorizing LLM errors.

4. **Four-layer clinical definitions with token budgets** — Not all symptoms need the same elaboration. Items 7–11 (highest cross-diagnostic ambiguity) receive full 4-layer definitions (~500 tokens), while Items 5, 12, 15–18 (concrete observable behaviors) need only 2 layers (~150 tokens). This optimizes prompt space allocation.

5. **Escalation cascade (primary → GPT)** — The primary LLM handles 75–80% of cases autonomously. GPT is reserved for structurally uncertain cases (low confidence, internal inconsistency, cross-diagnostic confounders), concentrating compute budget where it matters most.

6. **Five complementary runs** — Each run provides a different signal type (learned representations, LLM reasoning, retrieval similarity, cross-condition transfer, fusion). This hedges against failure of any single component and provides ablation insights.

7. **Depression few-shot examples as confounders** — Score-1 examples are deliberately sourced from BDI-Sen (severity=1) and RedSM5. These sentences describe depression symptoms that superficially resemble ADHD (e.g., "I can't focus on anything anymore" — depression concentration difficulty), teaching the LLM to distinguish mood-congruent from task-selective attention deficits.

8. **Three-backbone ensemble** — Mental-roberta (mental health domain), Clinical-BERT (clinical notes), and MPNet (general-purpose) capture complementary linguistic patterns. Averaging expected scores reduces variance without tuning ensemble weights.

---

## 13. Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| LLM overestimation (scoring too high) | HIGH | MEDIUM | Isotonic regression, "choose lower" prompt rule, BDI-II cross-validation |
| Low reliability on items 7–11 | HIGH | HIGH | Full 4-layer definitions, symptom weight 0.5, GPT escalation concentrated (30–45% rate) |
| Silver labels miss patterns | MEDIUM | HIGH | Layer 3 empirical vocabulary, keyword boost, manual audit per symptom |
| Primary LLM quality insufficient | MEDIUM | MEDIUM | Structured template, few-shot examples, escalation safety net |
| Training fails / insufficient GPU | MEDIUM | MEDIUM | Run 2 (LLM-only) and Run 5 (BiEnc) require no training |
| Unmapped ASRS items (3, 14–18) underperform | MEDIUM | LOW | Run 5 fallback provides retrieval-based baseline for all items |
