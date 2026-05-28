# Task 3: ADHD Symptom Sentence Ranking — Solution Description (HiPerT)

**Team:** INSALyon
**Competition:** eRisk 2026, Task 3
**System:** HiPerT (Hierarchical Prompt Engineering with Transfer learning)

---

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

Each symptom is grounded in a four-layer clinical definition framework totaling ~12,000 words across all 18 items (see `specs/asrs_four_layer_definitions.md`):

1. **L1 — Clinical Definition (DSM-5-TR):** Formal diagnostic criterion from the DSM-5-TR (APA 2022, pp. 68–70), with adult-specific adaptations distinguishing childhood from adult presentations.

2. **L2 — Adult Behavioral Manifestation (Barkley BAARS-IV / Ramsay & Rostain):** Concrete daily-life examples across work, home, and social domains — e.g., "90% completion paralysis on administrative tasks" for Item 1, "scrolling or channel-surfing at bedtime" for Item 7.

3. **L3 — Empirical Discussion Topics:** Language patterns from Reddit r/ADHD (372K posts analyzed via BERTopic, Kang et al. JMIR 2025) and Twitter (Guntuku et al. 2019) — how symptoms manifest in self-report social media language. E.g., "my brain doesn't have an off switch" for Item 7.

4. **L4 — Differential Markers:** Distinguishing ADHD from depression, anxiety, autism, OCD, and normal variation. E.g., for Item 7: GAD = worry-driven inability to relax; PTSD = trauma triggers; ADHD = boredom intolerance and understimulation.

### 2.2 Token Budget Strategy

Not all items require the same elaboration depth. Prompt space is allocated based on cross-diagnostic ambiguity:

| Budget | Layers Included | Items | Rationale |
|--------|----------------|-------|-----------|
| **full_4** (~400–500 tokens) | L1 + L2 + L3 + L4 | 7–11 | Highest cross-diagnostic overlap with depression/anxiety |
| **compressed_3** (~250–350 tokens) | L1 + L3 + L4 | 1–4, 6, 13–14 | Moderate ambiguity |
| **minimal_2** (~150–200 tokens) | L1 + L3 | 5, 12, 15–18 | Concrete observable behaviors, few confounders |

---

## 3. System Architecture

The HiPerT system is a multi-stage pipeline that combines bi-encoder retrieval, LLM-based scoring, and neural reranking. Two major architecture versions were developed and evaluated:

- **v1** used a bi-encoder (`SymptomConditionedEncoder`) with silhouette-contrastive loss and curriculum learning. This version suffered from representation collapse (Section 10.1) and was superseded by v2.
- **v2** replaced the bi-encoder with a cross-encoder (`CrossEncoderReranker`) using CORAL ordinal regression or ListMLE listwise ranking loss. The LLM cascade was promoted to the primary submission.

```
INPUT CORPUS (4.17M Reddit sentences)
         │
    ┌────▼─────────────────────────────────────────────────────┐
    │  STAGE 1: CANDIDATE SELECTION                            │
    │  Bi-encoder (all-mpnet-base-v2) → top-K candidates       │
    │  + query expansion + first-person filter + keyword boost  │
    └────┬─────────────────────────────────────────────────────┘
         │  ~5K candidates per symptom
    ┌────▼─────────────────────────────────────────────────────┐
    │  STAGE 2: LLM SCORING CASCADE                            │
    │  Primary LLM scores ALL candidates                       │
    │  (AS SUBMITTED: Llama-3.1-8B-4bit via HuggingFace;       │
    │   designed for GPT-4o-mini + GPT-4o escalation — see §5) │
    │  → 5 escalation rules → confidence-weighted labels (0–3) │
    └────┬─────────────────────────────────────────────────────┘
         │  ~90K silver label triples
    ┌────▼─────────────────────────────────────────────────────┐
    │  STAGE 3: CROSS-ENCODER RERANKER (v2)                    │
    │  [CLS] symptom [SEP] sentence [SEP]                      │
    │  CORAL ordinal regression or ListMLE                     │
    │  3 backbones × 5 folds ensemble                          │
    └────┬─────────────────────────────────────────────────────┘
         │
    ┌────▼─────────────────────────────────────────────────────┐
    │  OUTPUT: 5 SUBMISSION RUNS                               │
    │  R1=LLM cascade, R2=CrossEncoder, R3=RRF Ensemble,      │
    │  R4=DepTransfer, R5=BiEnc baseline                       │
    └──────────────────────────────────────────────────────────┘
```

---

## 4. Stage 1: Candidate Selection (Retrieval)

### 4.1 Bi-Encoder Retrieval

- **Model:** `sentence-transformers/all-mpnet-base-v2` (768-dim, 110M parameters)
- **Source:** Pre-trained general-purpose sentence transformer from Hugging Face
- **Query expansion:** Each ASRS symptom query is expanded by concatenating the item text with Layer 3 empirical discussion topics from `config/symptoms.yaml`
- **Top-K:** 5,000 candidates per symptom
- **Similarity:** Cosine similarity between query and sentence embeddings
- **Caching:** Corpus embeddings are cached to disk as `.npy` files for efficiency; progressive encoding with checkpointing supports interrupted runs

### 4.2 First-Person Filter

Retains only sentences containing first-person markers (`I`, `me`, `my`, `mine`, `myself`, `I'm`, `I've`, `I'd`, `I'll`). This eliminates ~60% of the corpus — third-person references, advice, and general statements are rarely relevant at score >= 2.

### 4.3 Keyword Boost

Each symptom has 16–18 curated keywords drawn from Layer 3 empirical vocabulary (defined in `config/symptoms.yaml`). A `+0.05` boost per keyword match is added to the cosine similarity score. Example keywords for Item 1 (Wrapping up projects): *organize, finish, procrastinate, deadline, last-minute, follow-through, incomplete*.

### 4.4 Output

Per symptom: a ranked list of ~5,000 candidate sentences with `combined_score = cosine_similarity + keyword_boost`. Saved to `output/candidates/symptom_{id}.json`. These candidates feed into both the LLM scoring cascade (Stage 2) and the BiEnc baseline run (Run 5).

---

## 5. Stage 2: LLM Scoring Cascade

### 5.1 Primary Scoring

Every candidate receives a structured LLM assessment. The prompt consists of:

1. **System prompt:** Clinical psychologist specializing in adult ADHD assessment, with the 0–3 scoring rubric.
2. **Symptom definition:** Clinical elaboration respecting the token budget (full_4 / compressed_3 / minimal_2), drawing from the four-layer definitions.
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

**LLM Configuration — as submitted.** All submitted Task 3 runs were generated on Google Colab via `notebooks/task3_colab.ipynb`, which patches the LLM client (`make_hf_client`) to use a single open model for every scoring call:

| Role | Model | Provider | Temperature | Max Tokens |
|------|-------|----------|-------------|------------|
| **Primary + escalation scorer (as submitted)** | **`meta-llama/Llama-3.1-8B-Instruct` (4-bit)** | **HuggingFace** | 0.1 | 512 |

> **As-submitted vs as-designed.** `config/pipeline.yaml` and earlier drafts of this document specify **GPT-4o-mini (OpenAI)** as the primary scorer and **GPT-4o (OpenAI)** as the escalation scorer, with **Llama-3.1-70B (Ollama)** as a local alternative. **None of these produced the submission.** The Colab notebook overrode the client to `meta-llama/Llama-3.1-8B-Instruct` quantized to 4-bit, served via HuggingFace, and generated all five runs (`generate_run`) with it. Consequently the official LLM-cascade results (AP 0.158 / 0.134, P@10 0.344 / 0.272) were achieved by a small 4-bit open model — not GPT-4o-mini. When reading §8 and the figures, interpret every "GPT-4o-mini" reference as the *designed* configuration; the *as-run* scorer was Llama-3.1-8B-4bit. The 5-rule escalation cascade collapses to the same model under this override (no stronger re-scoring model was available on Colab).

Rate limit: 0.1s delay between calls, max 5 retries with exponential backoff, 120s read timeout.

### 5.2 Few-Shot Examples

72 annotated examples (4 per symptom x 18 symptoms), sourced from:

| Source Dataset | Use | Count |
|----------------|-----|-------|
| **Random corpus sentences** | Irrelevant control (score 0) | ~18 |
| **BDI-Sen 2.0** | Depression confounders at score 1–2 | ~25 |
| **Retrieval candidates** | High-scoring corpus sentences (score 2–3) | ~20 |
| **RedSM5** | Clinician-annotated depression sentences (score 1) | ~9 |

Each example includes the full context triplet and a structured annotation matching the 7-field template. Score-1 examples are deliberately chosen from depression datasets to calibrate the boundary between "depression-related" and "ADHD-related" language.

### 5.3 Escalation Rules

Five rules trigger re-scoring by a stronger model:

| Rule | Trigger | Target Rate |
|------|---------|-------------|
| **1. Low Confidence** | CONFIDENCE <= 2 | ~5–10% |
| **2. Internal Inconsistency** | Contradictions between fields (e.g., SYMPTOM_MATCH=NO but SCORE >= 2) | ~5% |
| **3. Confounder + Borderline** | CONFOUNDERS != "NONE" AND SCORE in {1, 2} | ~8–12% |
| **4. Inattention Cross-Diagnostic** | Items 7–11 with SCORE >= 2 AND cross-diagnostic keywords | ~3–5% |
| **5. Moderate Confidence on Boundary** | SCORE in {1, 2} AND CONFIDENCE = 3 | ~5–8% |

### 5.4 Resolution Logic

After primary scoring (and optional escalation re-scoring), each (sentence, symptom) pair receives a final label and confidence weight.

**Base symptom weights** (reflecting cross-diagnostic reliability):
- Motor H/I (items 5–6, 12–14): **1.0** — concrete, observable, few confounders
- Verbal H/I (items 15–18): **0.9** — mostly observable
- Organization/Memory (items 1–4): **0.7** — moderate depression overlap
- Sustained Attention (items 7–11): **0.5** — strongest depression-ADHD overlap

**Resolution weight** (from scoring confidence):

| Scenario | Resolution Weight |
|----------|-------------------|
| Not escalated, confidence >= 4 | 0.80 |
| Not escalated, confidence = 3 | 0.60 |
| Escalated, GPT agrees (diff = 0) | 0.85 |
| Escalated, GPT differs by 1 | 0.65 |
| Escalated, GPT differs by >= 2 | 0.45 (trust GPT) |
| Both uncertain + diff >= 2 | 0.30 (use floor(mean)) |

**Final confidence_weight = resolution_weight x symptom_weight** (range: 0.15–0.85)

Silver labels are saved as per-symptom JSONL files at `output/silver_labels/symptom_{id}.jsonl`.

---

## 6. Data Sources and Cross-Dataset Mappings

### 6.1 Data Source Inventory

| Dataset | Size | Format | Labels | Role in Pipeline |
|---------|------|--------|--------|-----------------|
| **eRisk 2026 Task 3 Corpus** | 4,521 files, ~4.17M sentences | TREC (PRE/TEXT/POST) | Unlabeled | Primary ranking corpus (Stages 1–2) |
| **BDI-Sen 2.0** (Kayalvizhi et al., 2022) | 2,529 sentences, 5,003 annotations | JSONL | Graded 0–3, 21 BDI-II symptoms | v1 Stage A1 pre-training; few-shot confounders |
| **eRisk 2025 Task 1** | 11,042 judgments | CSV + TREC | Binary (majority + consensus) | v1 Stage A2 pre-training; boundary candidates |
| **eRisk 2023 Task 1** | 21,580 judgments | CSV + TREC | Binary (majority + consensus) | Optional additional pre-training |
| **RedSM5** (Sosa et al.) | 1,484 posts, 2,058 annotations | CSV | Binary + DSM-5 categories | Few-shot confounders; cross-validation |
| **Few-shot annotations** | 72 examples (4 x 18 symptoms) | JSON | Graded 0–3 + structured fields | LLM scoring calibration |

Data paths are configured in `config/pipeline.yaml`:

| Dataset | Path |
|---------|------|
| eRisk 2026 Corpus | `data/eRisk-2026/task3-adhd-symptom-ranking-.../output_trec_files/` |
| BDI-Sen 2.0 | `data/BDI-Sen/full_dataset/` |
| eRisk 2025 T1 | `data/eRisk-2025/eRisk25-datasets/t1-depression-symptom-ranking/` |
| eRisk 2023 T1 | `data/eRisk2023_T1/` |
| RedSM5 | `data/RedSM5/` |
| Few-shot examples | `annotations/symptom_*_examples.json` |

### 6.2 BDI-II → ASRS Symptom Mappings

Depression symptoms from BDI-II (items 1–21) are mapped to overlapping ASRS items based on shared behavioral manifestations. This mapping is used in Run 4 (DepTransfer) and for depression pre-training in v1:

| BDI-II Query | Depression Symptom | → ASRS Items | ADHD Manifestation |
|-------------|-------------------|-------------|-------------------|
| Q19 | Concentration Difficulty | 8, 9, 10, 11 | Sustained Attention/Distractibility |
| Q11 | Agitation | 5, 6, 12, 13 | Motor Hyperactivity/Impulsivity |
| Q15 | Loss of Energy | 4, 7 | Memory/Avoidance, Difficulty Unwinding |
| Q16 | Changes in Sleep | 7 | Difficulty Unwinding |
| Q13 | Indecisiveness | 1, 2 | Organization/Planning |

**Unmapped ASRS items:** 3, 14, 15, 16, 17, 18 — these have no direct BDI-II analog and rely solely on ADHD-specific retrieval and scoring.

### 6.3 BDI-Sen Symptom Mappings (for Few-Shot and Pre-Training)

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

### 6.4 Coverage Summary

| ASRS Item | BDI-Sen | BDI-II Qrels | RedSM5 | External Sources |
|-----------|:-------:|:------------:|:------:|:----------------:|
| 1–2 | Yes (Indecision) | Yes (Q13) | — | 2 |
| 3 | — | — | — | 0 (ADHD-only) |
| 4 | Yes (Energy, Fatigue) | Yes (Q15) | Yes (FATIGUE) | 3 |
| 5–6 | Yes (Agitation) | Yes (Q11) | Yes (PSYCHOMOTOR) | 3 |
| 7 | Yes (Energy, Sleep) | Yes (Q15, Q16) | Yes (FATIGUE, SLEEP) | 4 |
| 8–11 | Yes (Concentration) | Yes (Q19) | Yes (COGNITIVE) | 3 |
| 12–13 | Yes (Agitation) | Yes (Q11) | Yes (PSYCHOMOTOR) | 3 |
| 14–18 | — | — | — | 0 (ADHD-only) |

13 of 18 ASRS items have at least one external depression data source; the remaining 5 (items 3, 14, 15–18) rely solely on corpus retrieval and LLM scoring.

---

## 7. Stage 3: Neural Reranking

Two neural reranking architectures were developed: a bi-encoder (v1, deprecated) and a cross-encoder (v2, current).

### 7.1 v1: Bi-Encoder with Symptom Conditioning (Deprecated)

**Architecture: SymptomConditionedEncoder**

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

**v1 Training stages:**

| Stage | Data | Labels | Symptoms | Epochs | LR |
|-------|------|--------|----------|--------|-----|
| **A1** (BDI-Sen) | 2,529 sentences | Graded 0–3 | 21 BDI-II | 10 | 2e-5 |
| **A2** (eRisk 2025) | 11,042 judgments | Binary 0/1 | 21 BDI-II | 5 | 1e-5 |
| **B** (ADHD silver) | ~30K–36K silver labels | Graded 0–3 | 18 ASRS | 15 | 1e-5 |

**v1 Loss functions (composite):**
- **Ordinal Cross-Entropy:** Soft targets with label smoothing toward adjacent levels (epsilon=0.2)
- **Margin Ranking Loss:** Pairwise ordinal consistency (gamma=0.3)
- **Hierarchy Regularization:** Same-factor attraction + cross-factor repulsion (lambda=0.01)
- **Silhouette Contrastive Loss:** Ordinal-weighted separation in embedding space

**v1 Curriculum learning** ordered symptoms by clinical difficulty: Phase 1 (Motor/Verbal H/I, easy) → Phase 2 (Organization/Memory, medium) → Phase 3 (Sustained Attention, hard).

**Outcome:** The v1 encoder suffered from severe representation collapse (see Section 10.1) and was replaced by the v2 cross-encoder.

### 7.2 v2: Cross-Encoder Reranker (Current)

**Architecture: CrossEncoderReranker**

```
Input: [CLS] <symptom_text> [SEP] <sentence_text> [SEP]
   │
   ▼
Pre-trained Backbone (one of 3 choices)
   → [CLS] hidden state (768-dim)
   │
   ▼
Head: Dropout(0.3) → Linear(768, 256) → GELU → Dropout(0.1) → Linear(256, K)
   │
   ▼
Output: K=3 ordinal logits (CORAL) or K=1 scalar (ListMLE)
```

**Key design choice:** The cross-encoder processes symptom and sentence jointly via cross-attention, enabling valence and negation detection that bi-encoders miss. For example, on Item 1 ("trouble wrapping up final details"), the bi-encoder could not distinguish "I'm dedicated to finishing projects" (positive valence) from "I can never finish projects" (negative valence, symptom-relevant).

**Backbone models (trained independently, ensembled at inference):**

| Backbone | Model ID | Parameters | Domain |
|----------|---------|-----------|--------|
| **MentalRoBERTa** | `mental/mental-roberta-base` | 125M | Mental health Reddit |
| **ClinicalBERT** | `emilyalsentzer/Bio_ClinicalBERT` | 110M | Clinical notes |
| **mpnet** | `sentence-transformers/all-mpnet-base-v2` | 110M | General-purpose |

**Layer freezing:** Bottom layers frozen; only top 4 transformer layers + head are fine-tuned. Embeddings always frozen.

### 7.3 v2 Loss Functions

Two loss variants were implemented:

**Option A — CORAL Ordinal Regression** (Niu et al. 2016, Cao et al. 2020):

Models ordinal labels as a series of binary classification tasks: P(Y > k) for k = 0, 1, 2.

```
L_coral = -(1/N) Σ_i Σ_{k=1}^{K-1} [ y_ik · log(σ(f_k(x_i))) + (1 - y_ik) · log(1 - σ(f_k(x_i))) ]
```

Scoring: φ_coral(s, q) = σ(f_1) + σ(f_2) + σ(f_3) ∈ [0, 3]

Optional confidence weighting: samples weighted by LLM confidence in the silver label.

**Option B — ListMLE** (Xia et al. 2008):

Listwise learning-to-rank via the Plackett-Luce model. Directly optimizes the likelihood of the correct permutation of candidates per symptom.

```
L_listmle = -Σ_{i=1}^{n} [ f(s_π(i), q) - log( Σ_{j=i}^{n} exp(f(s_π(j), q)) ) ]
```

Uses sublist sampling (size 64, 8 sublists per batch) for computational efficiency with stochastic tie-breaking within grades (reshuffled each epoch).

### 7.4 v2 Training Procedure

**Training data:** 89,998 (symptom_text, sentence_text, score, confidence) triples extracted from LLM cascade silver labels (`output/training_v2/training_data.jsonl`, 41.7 MB).

**Label distribution (from LLM cascade):**

| Score | Meaning | Expected % |
|-------|---------|------------|
| 0 | Off-topic | ~35–45% |
| 1 | Wrong symptom | ~5–10% |
| 2 | Partial | ~20–30% |
| 3 | Relevant | ~20–30% |

**Cross-validation:** Leave-symptom-out 5-fold CV (splits by factor/subcluster):

| Fold | Train Symptoms | Validation Symptoms | Validation Factor |
|------|---------------|--------------------|--------------------|
| 1 | {1–14} | {15–18} | Verbal H/I |
| 2 | {1–4, 9–18} | {5–8} | Motor H/I (partial) |
| 3 | {5–18} | {1–4} | Organization/Planning |
| 4 | {1–8, 13–18} | {9–12} | Sustained Attention |
| 5 | {1–12} | {13–18} | Internal Drive + Verbal |

**Hyperparameters:**

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 2e-5 |
| Weight decay | 0.01 |
| LR schedule | CosineAnnealingLR |
| Batch size | 32 |
| Max sequence length | 256 tokens |
| Gradient clipping | max_norm = 1.0 |
| Early stopping patience | 3 epochs |
| Validation metric | nDCG@10 |

**Ensemble inference:** 3 backbones x 5 folds = 15 models. Final score per (sentence, symptom):

```
φ(s,q) = (1/3) Σ_backbone (1/5) Σ_fold predict_score(s, q)
```

---

## 8. Submission Runs

Five complementary runs are submitted, each using a different combination of system components:

| Run | System Name | Method | Primary Signal |
|-----|-----------|--------|----------------|
| **R1** | `INSALyon_LLM_cascade` | LLM scoring only (PRIMARY) | Llama-3.1-8B-4bit (HuggingFace) structured assessment — *as submitted; designed for GPT-4o-mini, see §5* |
| **R2** | `INSALyon_HiPerT_full` | Cross-encoder v2 reranker | 3-backbone × 5-fold ensemble |
| **R3** | `INSALyon_Ensemble` | RRF fusion of R1 + R2 | Reciprocal Rank Fusion (k=60) |
| **R4** | `INSALyon_DepTransfer` | Depression-only transfer | Stage A encoder + BDI→ASRS mapping |
| **R5** | `INSALyon_BiEnc_baseline` | Bi-encoder cosine similarity | Retrieval-only baseline |

### 8.1 Run 1: LLM_cascade (Primary Submission)

Pure LLM-based ranking with hierarchical tie-breaking:

```
φ(s,q) = label × 10.0                  # Primary: final LLM label (0–3)
        + confidence_weight × 5.0       # Secondary: resolution confidence
        + llm_confidence × 0.01         # Tertiary: LLM internal confidence (1–5)
        + cosine × 0.001               # Quaternary: retrieval similarity
```

Reads silver label JSONL files and candidate JSON files (for cosine scores).

### 8.2 Run 2: HiPerT_full (Cross-Encoder v2)

Cross-encoder reranker trained on LLM silver labels:

```
φ(s,q) = (1/3) Σ_backbone (1/5) Σ_fold [σ(f₁) + σ(f₂) + σ(f₃)]
```

Reads pre-computed scores from `output/encoder_scores_v2/`. Falls back to v1 encoder scores if v2 unavailable.

### 8.3 Run 3: Ensemble (RRF Fusion)

Reciprocal Rank Fusion combining Run 1 and Run 2:

```
φ_rrf(s,q) = 1/(60 + rank_llm(s)) + 1/(60 + rank_hipert(s))
```

Sentences appearing in only one ranking get 0 for the missing contribution. Falls back to RRF(Run 1, Run 5) if Run 2 is unavailable.

### 8.4 Run 4: DepTransfer (Cross-Condition Transfer)

Uses only Stage A depression pre-training (BDI-Sen + eRisk 2025 T1) without ADHD-specific fine-tuning. The BDI-II → ASRS mapping transfers depression knowledge to 12/18 ASRS items. Unmapped items (3, 14, 15–18) fall back to Run 5 (cosine baseline).

### 8.5 Run 5: BiEnc_baseline (Cosine Similarity)

Simplest system — no LLM, no training:

```
φ(s,q) = max_k cosine(emb(s), emb(q_k)) + 0.05 × first_person(s) + keyword_boost(s,q)
```

Serves as baseline and ablation anchor.

---

## 9. Experiments and Evaluation

### 9.1 Evaluation Methodology

All 5 runs were evaluated using an internal **LLM-as-Judge** protocol:

- **Judge models:** Llama 3.3 70B (primary, via Ollama) and Llama 3.1 8B (secondary, via HuggingFace Inference API)
- **Evaluation scope:** Top-10 sentences per symptom, 18 ASRS items, 5 runs = 900 judgments
- **Relevance scale:** 3 = Relevant (direct symptom description), 2 = Partial (indirect), 1 = Wrong symptom (ADHD but different item), 0 = Off-topic

### 9.2 Overall Results

| Run | System | Method | P@10 (score >= 2) | Mean Relevance (0–3) | Off-topic % | Wrong Symptom % |
|-----|--------|--------|----|----|----|-----|
| **R1** | INSALyon_LLM_cascade | LLM scoring | **0.944** | **2.74** | 1.7% | 3.9% |
| **R5** | INSALyon_BiEnc_baseline | Cosine similarity | **0.894** | **2.39** | 5.6% | 5.0% |
| **R3** | INSALyon_Ensemble | RRF (R1+R2) | 0.822 | 2.22 | 14.4% | 3.3% |
| **R4** | INSALyon_DepTransfer | Stage A only | 0.656 | 1.65 | 28.3% | 6.1% |
| **R2** | INSALyon_HiPerT_full | Cross-encoder | 0.450 | 1.10 | 49.4% | 5.6% |

### 9.3 Per-Factor Results (Precision@10)

| Factor | R2 (HiPerT) | R1 (LLM) | R3 (Ensemble) | R4 (DepTransfer) | R5 (BiEnc) |
|--------|:-----------:|:--------:|:-------------:|:----------------:|:----------:|
| Inattention | 0.589 | **0.933** | 0.833 | 0.644 | 0.956 |
| Motor H/I | 0.280 | **0.940** | 0.760 | 0.400 | 0.820 |
| Verbal H/I | 0.350 | **0.975** | 0.875 | **1.000** | 0.850 |

### 9.4 Per-Subcluster Results (Precision@10)

| Subcluster | R2 (HiPerT) | R1 (LLM) | R3 (Ens) | R4 (DepTr) | R5 (BiEnc) |
|------------|:-----------:|:--------:|:--------:|:----------:|:----------:|
| Organization/Planning | 0.600 | 0.950 | 0.750 | 0.450 | **0.950** |
| Memory/Avoidance | 0.500 | 0.850 | 0.800 | 0.750 | **0.950** |
| Sustained Attention | 0.620 | **0.960** | 0.880 | 0.680 | 0.960 |
| Fidgeting/Restlessness | 0.200 | 0.950 | 0.750 | 0.300 | **1.000** |
| Internal Drive/Settling | 0.333 | **0.933** | 0.767 | 0.467 | 0.700 |
| Output Control | 0.350 | **1.000** | 0.900 | **1.000** | **1.000** |
| Turn-Taking | 0.350 | **0.950** | 0.850 | **1.000** | 0.700 |

### 9.5 Cross-Run Agreement (Jaccard Overlap @10)

| Pair | @5 | @10 | @20 |
|------|:--:|:---:|:---:|
| HiPerT-LLM | 0.006 | 0.003 | 0.006 |
| HiPerT-Ens | 0.159 | 0.237 | 0.275 |
| HiPerT-BiEnc | 0.000 | 0.000 | 0.004 |
| LLM-Ens | 0.120 | 0.225 | 0.296 |
| LLM-BiEnc | 0.227 | 0.170 | 0.130 |
| LLM-DepTr | 0.102 | 0.063 | 0.053 |
| Ens-BiEnc | 0.045 | 0.092 | 0.111 |
| DepTr-BiEnc | 0.340 | 0.339 | 0.341 |

### 9.6 Kendall's Tau (Rank Correlation on Shared Sentences, top-50)

| Pair | Mean Tau | Shared % |
|------|:--------:|:--------:|
| LLM-BiEnc | **0.923** | 10.8% |
| DepTr-BiEnc | **0.924** | 33.9% |
| LLM-Ens | 0.564 | 32.5% |
| HiPerT-Ens | 0.533 | 32.5% |
| HiPerT-LLM | 0.333 | 0.8% |

### 9.7 Diversity Analysis

Unique sentences per run in top-10 (out of 180 possible slots across 18 symptoms):

| Run | Unique Sentences | % of Slots |
|-----|:----------------:|:----------:|
| DepTransfer | 115 | 63.9% |
| HiPerT | 111 | 61.7% |
| LLM | 94 | 52.2% |
| BiEnc | 86 | 47.8% |
| Ensemble | 49 | 27.2% |

The union of all top-10 lists spans **652 distinct sentences** across all symptoms, indicating high diversity between runs.

---

## 10. Analysis and Lessons Learned

### 10.1 v1 Encoder Collapse (HiPerT v1 Failure)

The trained bi-encoder (v1) produced scores with coefficient of variation (CV) of 0.00–0.02 across all 18 symptoms. The score gap between top-10 and remaining candidates was negligible:

| Symptom | Top-10 Mean | Rest Mean | Gap | CV |
|---------|:-----------:|:---------:|:---:|:--:|
| 1 (Wrapping up) | 0.4573 | 0.4414 | 0.016 | 0.02 |
| 7 (Unwinding) | 0.6612 | 0.6474 | 0.014 | 0.01 |
| 9 (Concentrating) | 0.1166 | 0.1153 | 0.001 | 0.00 |
| 12 (Leaving seat) | 0.0429 | 0.0424 | 0.001 | 0.01 |
| 17 (Finishing sentences) | 0.0861 | 0.0843 | 0.002 | 0.00 |

For comparison, BiEnc achieves a 39% gap on symptom 1 (0.607 vs. 0.437) and LLM cascade achieves a 3.5x ratio on symptom 12.

**Root causes identified:**
1. **Silhouette-contrastive loss:** Optimized inter-cluster separation and intra-cluster cohesion, collapsing within-grade variance. Since ranking requires ordering *within* a grade, this destroyed the signal needed for top-K selection.
2. **Curriculum learning amplification:** Presenting easy (already well-separated) examples first reinforced the collapse direction before harder examples were introduced.
3. **Depression pre-training damage:** Progression: BiEnc (89.4%) → DepTransfer (65.6%) → HiPerT (45.0%). Depression supervision pushed representations toward BDI-II-relevant features.
4. **Valence blindness:** The bi-encoder architecture encodes symptom and sentence independently, matching on topic overlap rather than semantic fit. E.g., surfaced sentences about "dedication to projects" rather than "inability to finish projects" for Item 1.

### 10.2 Key Findings

1. **LLM reranking is the strongest signal** — 94.4% P@10. The structured prompt engineering with four-layer clinical definitions and depression confounders as few-shot examples effectively guides the LLM to distinguish ADHD-specific experiences.

2. **Bi-encoder baseline is surprisingly competitive** — 89.4% P@10. Good sentence transformers capture symptom relevance well when combined with first-person filtering and keyword boosting.

3. **Ensemble fusion is a mixed bag** — Mixing HiPerT's noisy signal via RRF degrades quality from 94% to 82%, despite adding genuine diversity (64% exclusive sentences in Ensemble top-10).

4. **DepTransfer is factor-specific** — Perfect on Verbal H/I (100%) but poor on Motor H/I (40%), showing BDI-II and ASRS capture partially overlapping constructs.

5. **Cross-encoder (v2) did not fully resolve collapse** — While CORAL/ListMLE losses are theoretically more appropriate than silhouette-contrastive, the v2 cross-encoder still underperformed the LLM cascade and even the simple bi-encoder baseline. Training on silver labels with limited diversity may be inherently limiting.

6. **Verbal/Impulsivity symptoms are easiest** — All runs achieve high precision on items 15–18 (distinctive vocabulary). Motor symptoms (fidgeting, leaving seat) and Internal Drive are hardest.

### 10.3 Configuration Variants Tested

| Configuration | Description | Status |
|---|---|---|
| **v1 + silhouette + curriculum** | Bi-encoder, composite loss (ordinal + ranking + hierarchy + silhouette), 3-phase curriculum | Failed (representation collapse, 45% P@10) |
| **v2 CORAL** | Cross-encoder, CORAL ordinal regression, leave-symptom-out 5-fold CV | Trained but underperformed LLM |
| **v2 ListMLE** | Cross-encoder, ListMLE listwise ranking, sublist sampling | Trained as alternative to CORAL |
| **Stage A only (DepTransfer)** | Depression pre-training (BDI-Sen + eRisk 2025), BDI→ASRS mapping, no ADHD fine-tuning | 65.6% P@10 |
| **LLM cascade (Llama-3.1-8B-4bit, HuggingFace) — AS SUBMITTED** | Single open model for primary + escalation (Colab), same prompt stack | Official P@10 0.344 (maj) / 0.272 (unan); the 94.4% figure below is an internal silver-label self-agreement metric, *not* the official qrels P@10 |
| **LLM cascade (GPT-4o-mini)** | Primary scorer = GPT-4o-mini, escalation = GPT-4o, max 25% escalation | **94.4% P@10** internal silver eval — *designed config, NOT the submission* |
| **LLM cascade (Llama 3.1:70b)** | Primary scorer = Llama 3.1 70B (Ollama), same prompt stack | Local alternative, not used |
| **BiEnc baseline** | all-mpnet-base-v2 cosine similarity + first-person filter + keyword boost | 89.4% P@10 (strong baseline) |
| **Escalation rate = 0%** | Disabled GPT escalation (single-LLM scoring) | Current production config |
| **Escalation rate = 20–25%** | Full 5-rule escalation with GPT-4o re-scoring | Original designed config |

---

## 11. Key Design Decisions and Rationale

1. **Depression datasets as calibrated confounders** — BDI-II and ASRS share 12/18 overlapping behavioral constructs (~30% ADHD-depression comorbidity). Depression sentences (BDI-Sen at severity=1) serve as calibrated score-1 few-shot examples, teaching the LLM to distinguish mood-congruent from task-selective attention deficits.

2. **Four-layer clinical definitions with token budgets** — Items 7–11 (highest cross-diagnostic ambiguity) receive full 4-layer definitions (~500 tokens), while Items 5, 12, 15–18 (concrete observable behaviors) need only 2 layers (~150 tokens). This optimizes prompt space allocation.

3. **Cross-encoder over bi-encoder** — After v1's valence blindness failure, the cross-encoder architecture was adopted. Joint processing of `[CLS] symptom [SEP] sentence [SEP]` enables cross-attention between symptom and sentence tokens, supporting valence and negation detection.

4. **CORAL ordinal regression** — Each sentence gets an independent prediction (unlike silhouette loss which operates across the batch), preventing representational collapse. The ordinal structure of 0–3 relevance is preserved without treating labels as unordered categories.

5. **Five complementary runs** — Each run provides a different signal type (LLM reasoning, learned representations, retrieval similarity, cross-condition transfer, fusion). This hedges against failure of any single component and provides ablation insights.

6. **LLM cascade as primary** — Empirical results (94.4% vs. 45–89.4%) led to reversing the initial architecture where HiPerT was primary. The LLM cascade exploits the structured 7-field output template and clinical prompt engineering more effectively than any trained model.

---

## 12. Software Architecture

The system is implemented as the `hipert` Python package (`src/hipert/`), installable via `uv` with the CLI entry point `uv run hipert <command>`.

### 12.1 Package Structure

```
src/hipert/
├── cli.py                   # Click CLI: parse, retrieve, score, output, runs, audit,
│                            #   train, infer, extract-v2, train-v2, infer-v2, diagnose-v2
├── config.py                # Config loading (pipeline.yaml, symptoms.yaml, annotations)
├── models.py                # Core dataclasses: Sentence, SymptomDefinition, LLMOutput,
│                            #   ScoringResult, CandidateScore, FewShotExample
├── data/
│   ├── trec_parser.py       # TREC corpus parser (full + simplified formats)
│   ├── corpus.py            # Corpus loader, statistics, first-person markers
│   ├── bdisen_loader.py     # BDI-Sen 2.0 data loader
│   ├── erisk2025_loader.py  # eRisk 2025 T1 data loader
│   ├── erisk2023_loader.py  # eRisk 2023 T1 data loader
│   ├── redsm5_loader.py     # RedSM5 data loader
│   ├── cross_dataset_mappings.py  # BDI-II → ASRS mappings
│   └── trec_writer.py       # TREC format output writer
├── retrieval/
│   ├── encoder.py           # BiEncoderRetriever (embedding, caching)
│   ├── candidate_selector.py # CandidateSelector (full retrieval pipeline)
│   ├── filters.py           # First-person filter, keyword boost
│   └── query_expansion.py   # Symptom query expansion with L3 topics
├── scoring/
│   ├── llm_client.py        # LLMClient (OpenAI-compatible, streaming, retry)
│   ├── hf_llm_client.py     # HuggingFace text-generation-inference client
│   ├── prompt_builder.py    # PromptBuilder (system/user prompts, token budgets)
│   ├── response_parser.py   # Robust 7-field regex parser
│   ├── escalation.py        # 5 escalation trigger rules
│   ├── resolution.py        # Label resolution with symptom weights
│   └── scorer.py            # ScoringCascade (full LLM→escalation→GPT orchestration)
├── pipeline/
│   ├── runner.py            # PipelineRunner (end-to-end orchestration)
│   └── checkpoint.py        # CheckpointManager (per-symptom JSONL, thread-safe)
├── training/
│   ├── encoder.py           # SymptomConditionedEncoder (v1 bi-encoder)
│   ├── cross_encoder.py     # CrossEncoderReranker (v2 cross-encoder)
│   ├── cross_encoder_dataset.py  # CrossEncoderDataset (leave-symptom-out CV)
│   ├── dataset.py           # ScoringDataset (v1, BDI-Sen/eRisk support)
│   ├── losses.py            # OrdinalCE, SymmetricCE, Ranking, Hierarchy, Silhouette
│   ├── trainer.py           # Trainer v1 (checkpointing, early stopping)
│   ├── trainer_v2.py        # TrainerV2 (per-fold, CORAL/ListMLE)
│   ├── stage_a.py           # Stage A depression pre-training (A1 BDI-Sen, A2 eRisk)
│   ├── stage_b.py           # Stage B ADHD fine-tuning (curriculum)
│   ├── inference.py         # v1 single-backbone and ensemble inference
│   ├── inference_v2.py      # v2 cross-encoder ensemble inference
│   ├── calibration.py       # Isotonic regression, temperature scaling
│   └── extract_training_data.py  # Silver label → cross-encoder training data
├── runs/
│   ├── registry.py          # Run registry, system names, dispatch
│   ├── run1_full.py         # LLM cascade ranking (PRIMARY)
│   ├── run2_llm.py          # Cross-encoder v2 ranking
│   ├── run3_ensemble.py     # RRF fusion
│   ├── run4_transfer.py     # Depression transfer
│   └── run5_bienc.py        # Bi-encoder baseline
├── quality/
│   ├── audit.py             # Per-symptom audit reports
│   ├── calibration.py       # Post-hoc score calibration
│   └── confidence.py        # Confidence weighting schemes
└── utils/
    └── logging.py           # Logging utilities (console, JSONL, LLM call logs)
```

### 12.2 Typical Execution Flow

```bash
# 1. Parse corpus and compute statistics
uv run hipert parse

# 2. Retrieve candidates (bi-encoder → filter → boost → top-K)
uv run hipert retrieve

# 3. Score candidates with LLM cascade (resumable)
uv run hipert score

# 4. (Optional) Train cross-encoder v2
uv run hipert extract-v2      # Extract training data from silver labels
uv run hipert train-v2         # Train 3 backbones × 5 folds
uv run hipert infer-v2         # Generate ensemble scores

# 5. Generate TREC output for all 5 runs
uv run hipert output --run-id 1  # LLM cascade
uv run hipert output --run-id 2  # Cross-encoder
uv run hipert output --run-id 3  # Ensemble
uv run hipert output --run-id 4  # DepTransfer
uv run hipert output --run-id 5  # BiEnc baseline

# 6. Quality audit
uv run hipert audit
```

---

## 13. Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| LLM overestimation (scoring too high) | HIGH | MEDIUM | Isotonic regression, depression confounders as few-shot, BDI-II cross-validation |
| Low reliability on items 7–11 | HIGH | HIGH | Full 4-layer definitions, symptom weight 0.5, escalation concentrated (30–45% rate) |
| Trained encoder collapse | REALIZED | HIGH | Replaced bi-encoder with cross-encoder (v2); promoted LLM cascade to primary |
| Silver labels miss patterns | MEDIUM | HIGH | Layer 3 empirical vocabulary, keyword boost, manual audit per symptom |
| Training fails / insufficient GPU | MEDIUM | MEDIUM | Run 1 (LLM-only) and Run 5 (BiEnc) require no training |
| Unmapped ASRS items (3, 14–18) underperform | MEDIUM | LOW | Run 5 fallback provides retrieval-based baseline for all items |

---

## 14. References

- **ASRS-v1.1:** Kessler et al. (2005). The World Health Organization Adult ADHD Self-Report Scale.
- **DSM-5-TR:** American Psychiatric Association (2022). Diagnostic and Statistical Manual of Mental Disorders, 5th ed., Text Revision.
- **Bifactor model:** Panagiotidi et al. (2024); Stanton et al. (2018). Three-factor structure of ASRS.
- **BAARS-IV:** Barkley (2011). Barkley Adult ADHD Rating Scale-IV.
- **BDI-Sen 2.0:** Kayalvizhi et al. (2022). BDI-Sen: A Sentence-level Dataset for Depression.
- **RedSM5:** Sosa et al. Clinician-annotated Reddit posts with DSM-5 categories.
- **BERTopic / Reddit ADHD:** Kang et al. (JMIR 2025). Topic modeling of r/ADHD (372K posts).
- **Social media ADHD:** Guntuku et al. (2019). Twitter language markers.
- **CORAL:** Niu et al. (2016); Cao et al. (2020). Consistent Rank Logits for ordinal regression.
- **ListMLE:** Xia et al. (2008). Listwise approach to learning to rank.
- **RRF:** Cormack et al. (2009). Reciprocal Rank Fusion.
- **MentalRoBERTa:** Ji et al. `mental/mental-roberta-base` — RoBERTa pre-trained on mental health Reddit.
- **ClinicalBERT:** Alsentzer et al. (2019). `Bio_ClinicalBERT` — BERT pre-trained on clinical notes.
- **MPNet:** Song et al. (2020). `all-mpnet-base-v2` — general-purpose sentence transformer.
