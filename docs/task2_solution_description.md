# Task 2: Contextualized Early Detection of Depression — Solution Description

## 1. Task Overview

eRisk 2026 Task 2 requires systems to monitor a stream of Reddit discussion threads delivered round-by-round via REST API and classify each target user as depressed (`decision=1`) or control (`decision=0`). Each round provides one thread per target user containing the full conversational context: submission title and body, all comments from all participants in chronological order, and the hierarchical reply structure. The target user (`targetSubject`) has one or more contributions in the thread but may or may not be the submission author.

At each round, the system must emit for each user and each run:
- **`decision`**: 0 (wait) or 1 (final alert — irreversible once emitted)
- **`score`**: a continuous depression probability in [0, 1]

The system is evaluated on ERDE (Early Risk Detection Error) at two operating points (o=5 and o=50), F1 classification accuracy, and F_latency (F1 discounted by alert speed). The core challenge is the **speed–accuracy trade-off**: alerting too early risks false positives, while waiting too long incurs exponential latency penalties.

---

## 2. Training Data

### 2.1 Source Dataset

The training data comes from the **eRisk 2025 Task 2** competition, stored as individual JSON files per user under `data/eRisk-2025/.../all_combined/`.

**Dataset statistics:**
- **909 users** total: 102 depressed (11.2%), 807 control (88.8%)
- Mean 306.5 threads per user, median 211
- Each thread represents one "round" of the simulated competition

**Ground truth labels** are loaded from `shuffled_ground_truth_labels.txt`, a tab-separated file mapping `subject_id → 0|1`.

### 2.2 Training Data Format

Each user has a JSON file containing a list of thread objects. Each thread includes:

```json
{
  "submission": {
    "submission_id": "...",
    "user_id": "target_subject",
    "title": "...",
    "body": "...",
    "created_utc": "2024-05-08 02:55:38 UTC"
  },
  "comments": [
    {
      "comment_id": "...",
      "user_id": "author",
      "body": "comment text",
      "parent_id": "...",
      "created_utc": "...",
      "target": true
    }
  ]
}
```

The `target` flag on each comment indicates whether the comment was authored by the target user. The submission's `user_id` indicates the submission author (may or may not be the target).

### 2.3 Server Data Format (Live Competition)

During live competition, threads arrive from the server in a different JSON format:

```json
{
  "submissionId": "...",
  "title": "...",
  "body": "...",
  "author": "...",
  "date": "2024-05-08T02:55:38.000+00:00",
  "number": 0,
  "targetSubject": "subject_xyz",
  "comments": [
    {
      "commentId": "...",
      "author": "...",
      "body": "...",
      "date": "...",
      "parent": "..."
    }
  ]
}
```

Both formats are normalized into a shared internal representation (`Thread`, `Comment` dataclasses) by the data loader, enabling unified processing across training and live modes.

### 2.4 Key Data Challenges

- **Extreme text sparsity**: median 14 words per target post, 22% of threads have NO target text at all.
- **Class imbalance**: only 11.2% depressed users.
- **Deleted submissions**: 12.2% of threads have deleted submission bodies.
- **Sparse community signal**: median 0 direct replies to target per thread.

---

## 3. System Architecture Overview

Our system is a **multi-layer incremental feature accumulation pipeline** that processes threads one at a time, maintains a per-user state profile, and runs classification at every round for each of 5 parallel run configurations.

```
┌────────────────────────────────────────────────────────────────────┐
│                        Per-Round Processing                        │
│                                                                    │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────────────┐    │
│  │ Thread       │──▶│ Layer 1      │──▶│ Embedding Running    │    │
│  │ (1 per user) │   │ Textual      │   │ Mean (1920d, λ=0.95) │    │
│  │              │   │              │──▶│ Symptom Scores (21d) │    │
│  │              │   │              │──▶│ Lexical (4d)         │    │
│  └─────────────┘   └──────────────┘   └──────────────────────┘    │
│        │                                                           │
│        │            ┌──────────────┐   ┌──────────────────────┐    │
│        ├───────────▶│ Layer 2      │──▶│ Reply Sentiment (3d) │    │
│        │            │ Conversational│──▶│ Concern (1d)         │    │
│        │            │              │──▶│ Conv Position (3d)   │    │
│        │            │              │──▶│ Thread Topic (21d)   │    │
│        │            └──────────────┘   └──────────────────────┘    │
│        │                                                           │
│        │            ┌──────────────┐   ┌──────────────────────┐    │
│        ├───────────▶│ Layer 3      │──▶│ Emotion (9d)         │    │
│        │            │ Semantic      │──▶│ BERTopic (41d)       │    │
│        │            └──────────────┘   └──────────────────────┘    │
│        │                                                           │
│        │            ┌──────────────┐                               │
│        └───────────▶│ ToM Module   │──▶ Dual Mentalizing (47d)     │
│                     └──────────────┘                               │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│                     Per-User Aggregation                           │
│                                                                    │
│  ┌──────────────────┐  ┌─────────────────┐  ┌──────────────────┐  │
│  │ Wasserstein      │  │ Mahalanobis     │  │ Thompson         │  │
│  │ Distances (72d)  │  │ Distances (3d)  │  │ Sampling (25d)   │  │
│  │ (temporal shift)  │  │ (anomaly)       │  │ (symptom weight) │  │
│  └──────────────────┘  └─────────────────┘  └──────────────────┘  │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
              ┌──────────────────────────────┐
              │  Feature Assembler (~2341d)   │
              └──────────────────────────────┘
                              │
                              ▼
              ┌──────────────────────────────┐
              │  Classifier (per-run)         │
              │  XGBoost / MLP / SVM / Ens.  │
              └──────────────────────────────┘
                              │
                              ▼
              ┌──────────────────────────────┐
              │  Adaptive Decision Policy     │
              │  ERDE-aware threshold decay   │
              └──────────────────────────────┘
                              │
                              ▼
                       decision ∈ {0, 1}
```

---

## 4. Feature Extraction

### 4.1 Layer 1: Target User Textual Features (~2092d)

#### 4.1.1 Sentence Transformer Ensemble (1920d)

We use a 3-model ensemble of pre-trained sentence transformers to produce dense semantic representations of the target user's text:

| Model | Dimension | Architecture |
|-------|-----------|-------------|
| `all-mpnet-base-v2` | 768 | MPNet |
| `all-MiniLM-L12-v2` | 384 | MiniLM |
| `all-distilroberta-v1` | 768 | DistilRoBERTa |
| **Total** | **1920** | Concatenated |

**Per-round encoding**: All target user texts within a thread are encoded by each model and averaged to produce a single (1920d) round embedding.

**Exponential-decay running mean**: Rather than a simple mean across all rounds, we maintain a weighted running sum with decay factor λ=0.95:

```
sum_{new} = λ × sum_{old} + embedding_{round}
weight_{new} = λ × weight_{old} + 1.0
running_mean = sum / weight
```

This gives exponentially more weight to recent posts while retaining long-term context. The running mean (1920d) enters the final feature vector.

**Rationale for 3-model ensemble**: Each model captures different aspects of semantic similarity — MPNet excels at paraphrase detection, MiniLM provides efficient broad coverage, and DistilRoBERTa captures distributional semantics. The concatenation produces a richer representation than any single model, at acceptable computational cost since encoding is batched.

#### 4.1.2 BDI-II Symptom Activation Scores (21d per round, 189d aggregated)

We define 21 textual descriptions of BDI-II symptoms (one per item) and compute their sentence transformer embeddings as **reference vectors**. The symptom descriptions use "Variant C" — detailed clinical formulations:

| BDI-II Item | Symptom | Description |
|-------------|---------|-------------|
| 1 | Sadness | "Persistent feelings of sorrow, unhappiness, or emotional pain." |
| 2 | Pessimism | "Discouragement and hopelessness about the future." |
| 3 | Past failure | "Feeling like a failure, seeing many disappointments." |
| 4 | Loss of pleasure | "Reduced enjoyment of activities, hobbies, social life." |
| 5 | Guilty feelings | "Excessive guilt, self-blame for bad things." |
| 6 | Punishment feelings | "Expectation of punishment, sense that bad things are deserved." |
| 7 | Self-dislike | "Self-criticism, disappointment in oneself as a person." |
| 8 | Self-criticalness | "Harsh self-judgment for all faults and mistakes." |
| 9 | Suicidal thoughts | "Thoughts of ending one's life, death wishes." |
| 10 | Crying | "Increased tearfulness, uncontrollable emotional outbursts." |
| 11 | Agitation | "Restlessness, irritability, inability to stay still or relax." |
| 12 | Loss of interest | "Social withdrawal, apathy, not caring about things." |
| 13 | Indecisiveness | "Difficulty making decisions, putting off choices." |
| 14 | Worthlessness | "Profound sense of having no value, being useless." |
| 15 | Loss of energy | "Fatigue, everything takes extra effort." |
| 16 | Sleep changes | "Insomnia, oversleeping, or disrupted sleep patterns." |
| 17 | Irritability | "Short temper, easily frustrated or angered." |
| 18 | Appetite changes | "Eating much more or less, weight gain or loss." |
| 19 | Concentration difficulty | "Brain fog, difficulty focusing, forgetfulness." |
| 20 | Tiredness/fatigue | "Constant exhaustion, lack of motivation due to tiredness." |
| 21 | Loss of interest in sex | "Reduced libido, no sexual desire." |

**Per-round scoring**: The round embedding is compared via cosine similarity to all 21 reference embeddings, producing a (21d) symptom activation vector with values in [-1, 1].

**Aggregated features (189d total)**:
- **Max-pool** across all rounds (21d): captures peak symptom severity ever observed.
- **Mean-pool** across all rounds (21d): captures average symptom activation level.
- **Distributional statistics** per symptom (21 × 7 = 147d): for each of the 21 symptoms, we compute mean, variance, skewness, kurtosis, Q25, Q50, Q75 over the time series of activations. This captures not just central tendency but the shape of temporal symptom evolution — e.g., a high-variance symptom suggests episodic presentation, while high kurtosis suggests rare but extreme spikes.

#### 4.1.3 Lexical Indicator Ratios (4d)

Four word-frequency ratios computed over all accumulated target texts:

| Feature | Word List | Count | Rationale |
|---------|-----------|-------|-----------|
| First-person singular ratio | i, me, my, mine, myself, i'm, i've, i'd, i'll | 9 | Self-focus correlates with depression severity |
| Negative emotion ratio | sad, depressed, hopeless, lonely, anxious, worthless, ... | 36 | Direct negative affect expression |
| Absolutist word ratio | always, never, nothing, everything, completely, totally, ... | 15 | Validated as one of the strongest linguistic markers of depression |
| Cognitive process ratio | think, know, believe, realize, remember, wonder, ... | 18 | Rumination and over-thinking patterns |

Each ratio = count of matching words / total words in accumulated texts.

### 4.2 Layer 2: Conversational Context Features (~28d)

These features capture the social context — how others respond to the target user.

#### 4.2.1 Reply Sentiment (3d)

We apply **VADER** sentiment analysis to direct replies to the target user (or branch comments as fallback). Over all rounds, we compute:
- **Mean compound sentiment**: average community sentiment toward target.
- **Standard deviation**: variability in community response tone.
- **Trend** (linear slope): whether community sentiment is shifting over time (e.g., increasing concern).

#### 4.2.2 Concern Detection (1d)

We scan all thread text for 40 concern phrases that indicate community worry about the target user, such as:
- "are you okay", "please get help", "reach out to someone"
- "suicide hotline", "therapist", "medication"
- "worried about you", "take care of yourself"

The feature is the ratio of rounds where at least one concern phrase was detected.

#### 4.2.3 Conversational Position (3d)

Structural features characterizing how the target user participates in discussions:
- **is_author_ratio**: fraction of threads where the target user is the submission author (initiator vs. commenter).
- **target_silent_ratio**: fraction of rounds with no target text at all (withdrawal signal).
- **reply_depth_mean**: average depth of target user's comments in the reply tree (surface-level participation vs. deep engagement).

#### 4.2.4 Thread Topic Similarity (21d)

For each thread, the title (or body if title is absent) is embedded using the same 3-model ensemble and compared via cosine similarity to the 21 BDI-II symptom reference embeddings. This captures whether the target user tends to participate in threads whose topics are depression-adjacent. The final feature is the mean of these (21d) vectors across all rounds.

### 4.3 Layer 3: Emotion and Topic Features (~50d)

#### 4.3.1 Emotion Classification (9d)

We use the `j-hartmann/emotion-english-distilroberta-base` model, a fine-tuned DistilRoBERTa classifier for Plutchik's 8 basic emotions:

| Emotion | Index |
|---------|-------|
| Anger | 0 |
| Anticipation | 1 |
| Disgust | 2 |
| Fear | 3 |
| Joy | 4 |
| Sadness | 5 |
| Surprise | 6 |
| Trust | 7 |

**Per-round**: Each target text is classified → (8d) distribution. Texts with fewer than 10 words receive a uniform (1/8) distribution to avoid noisy classifications on very short texts. Per-round distributions are averaged.

**Final features (9d)**: Mean emotion distribution across all rounds (8d) + Shannon entropy of the mean distribution (1d). Low entropy indicates dominant emotional state; high entropy indicates emotional variability.

#### 4.3.2 BERTopic Dynamic Topic Modeling (41d)

We fit a BERTopic model on all target user texts from the training set:

| Parameter | Value |
|-----------|-------|
| n_topics | 40 |
| n_neighbors (UMAP) | 15 |
| n_components (UMAP) | 5 |
| min_cluster_size (HDBSCAN) | 50 |
| min_samples (HDBSCAN) | 10 |

**Fitting**: Documents from all training users (with ≥10 words each) are pooled. Depression labels are used to identify which topics are disproportionately associated with depressed users (depression_proportion metric).

**Per-user transform**: A rolling buffer of the last 5 target texts is maintained. The concatenated buffer text is transformed into a topic distribution via the fitted model.

**Final features (41d)**:
- 40d topic distribution vector
- 1d mean depression proportion across all rounds

---

## 5. Distributional Distance Metrics

### 5.1 Wasserstein Distance (72d) — Within-User Temporal Shift Detection

The Wasserstein (Earth Mover's) distance measures how the distribution of a user's features shifts over time. We compute it at **3 temporal scales** to capture shifts at different time horizons:

| Scale | Division | Minimum Rounds | Captures |
|-------|----------|----------------|----------|
| **Short** (half-window=5) | [k-10, k-5] vs [k-5, k] | 10 | Acute recent changes |
| **Medium** (half-window=25) | [k-50, k-25] vs [k-25, k] | 50 | Sub-chronic shifts |
| **Long** (full history) | [0, n/2] vs [n/2, n] | 20 | Chronic trajectory |

**Components**:

- **Symptom Wasserstein (63d)**: For each of 21 symptoms and each of 3 scales, the 1D Wasserstein distance between the early and recent halves of that symptom's activation time series. Total: 3 × 21 = 63 features. A high value for a specific symptom at a specific scale means that symptom's prevalence is changing at that temporal resolution.

- **Emotion Wasserstein (3d)**: Same 3-scale Wasserstein applied to the scalar time series of emotion Shannon entropy. Captures whether the user's emotional variability is shifting.

- **Embedding Sliced Wasserstein (3d)**: For the high-dimensional (1920d) embedding history, we use **sliced Wasserstein distance** — projecting both early and recent embedding sets onto 50 random unit directions and averaging the 1D Wasserstein distances. This avoids the curse of dimensionality while retaining distributional sensitivity.

- **Topic Wasserstein (3d)**: Same 3-scale Wasserstein applied to the topic entropy time series. Captures whether the user's thematic focus is narrowing or broadening.

### 5.2 Mahalanobis Distance (3d) — Cross-User Anomaly Detection

The Mahalanobis distance provides an anomaly score measuring how far a user's feature vector is from the normative distribution, accounting for feature correlations.

**Fitting (on training data)**:
1. **PCA reduction**: The full feature vector (~2341d) is reduced to min(50, n_features, n_samples-1) principal components for numerical stability.
2. **Control distribution**: Mean and covariance (LedoitWolf shrinkage estimator for regularization) of PCA-reduced features for the 807 control users.
3. **Depressed distribution**: Same for the 102 depressed users.

**Per-user scoring (3d)**:
- **D_M_control**: Mahalanobis distance from the control distribution center. High values indicate deviation from typical non-depressed behavior.
- **D_M_depressed**: Mahalanobis distance from the depressed distribution center.
- **D_M_relative**: D_M_control − D_M_depressed. Positive values indicate the user is closer to the depressed distribution than to the control distribution.

**Combined distributional score (2d)**: Additionally, we include:
- D_M_control (duplicated for emphasis)
- Mean Wasserstein distance (average of all 72 Wasserstein features)

---

## 6. Theory of Mind Module (47d)

The Theory of Mind (ToM) module captures **dual perspectives** on the target user: how the target presents themselves (self-view) versus how the community perceives them (observer-view). The discrepancy between these perspectives is itself a diagnostic signal — depressed individuals often show reduced insight into how their behavior appears to others.

### 6.1 Implementation Options

Three implementation options are available, all producing the same 47d feature vector:

| Option | Method | LLM Required | Active |
|--------|--------|-------------|--------|
| **Option A** | Embedding-based | No | Fallback (always computed) |
| **Option B** | Response category classification | Yes | Optional |
| **Option C** | LLM dual mentalizing | Yes | Primary (when Ollama available) |

### 6.2 Option C: LLM-Based Dual Mentalizing (Primary)

Uses **Llama 3.3 70B** via Ollama (temperature 0.1, context window 8192 tokens) to generate structured JSON assessments.

**Prompt 1 — Self-View**: Analyzes ONLY the target user's own writings. The system prompt includes the full 21 BDI-II symptom definitions. Output:
- `active_symptoms`: dict of symptom name → {score: 1-3, evidence: "..."}
- `depression_probability`: 0.0–1.0
- `overall_impression`: free text

**Prompt 2a — Observer-View (Independent)**: Analyzes how OTHER PEOPLE in the thread perceive the target user, using the formatted thread with `[TARGET]` markers. Output:
- `perceived_symptoms`: dict of symptom name → {score: 1-3}
- `observer_concern_level`: 0–3 (none/mild/moderate/strong)
- `community_response_type`: concern/support/advice/mixed/normalization/casual
- `depression_probability`: 0.0–1.0

System prompts are pre-formatted and kept **byte-identical** across calls to maximize Ollama's KV cache reuse.

### 6.3 Option A: Embedding-Based (Fallback)

When no LLM is available, we compute:
- **Self-view**: Mean embedding of target user texts → embedding norm as depression_probability proxy.
- **Observer-view**: Mean embedding of all other comments → embedding norm.
- **Gap metric**: 1 − cosine_similarity(self, observer) as insight_gap proxy.

### 6.4 ToM Feature Vector (47d)

| Offset | Dimension | Feature |
|--------|-----------|---------|
| 0–20 | 21 | Self-view symptom scores (normalized to 0–1 from LLM scores 1–3) |
| 21–41 | 21 | Observer-view symptom scores |
| 42 | 1 | Self depression_probability |
| 43 | 1 | Observer depression_probability |
| 44 | 1 | Insight gap (mean absolute difference between self and observer symptom scores) |
| 45 | 1 | Observer concern level (0–3, normalized to 0–1) |
| 46 | 1 | Community response type encoded (concern→1.0, support→0.8, advice→0.6, mixed→0.5, normalization→0.3, casual→0.0) |

---

## 7. Thompson Sampling for Symptom Weighting (25d)

### 7.1 Motivation

Not all 21 BDI-II symptoms are equally informative for every user. Some users' depression manifests primarily as cognitive symptoms (pessimism, worthlessness, guilt), while others show primarily somatic symptoms (sleep, appetite, fatigue). Thompson Sampling provides a principled way to learn which symptoms are most consistently activated for each individual user, dynamically up-weighting the most informative symptoms.

### 7.2 Mechanism

We maintain a **21-armed bandit** with **Beta(α, β) posteriors** per symptom per user, initialized to Beta(1, 1) (uniform prior).

**Per-round update**:
1. Observe symptom activation vector (21d) from the current round.
2. For each symptom: if activation > τ_active (0.3), mark as **active**.
3. Update: α_i += (active ? 1 : 0), β_i += (¬active ? 1 : 0).

**Weight computation**: Expected weight = α_i / (α_i + β_i), normalized to sum to 1.

### 7.3 Thompson Sampling Features (25d)

| Feature | Dimension | Description |
|---------|-----------|-------------|
| Weighted symptom score | 1 | dot(weights, latest_activations) — personalized depression score |
| Symptom entropy | 1 | −Σ(w_i × log(w_i)) — concentration of symptom pattern |
| Active symptom count | 1 | Number of symptoms with activation > 0.3 |
| Weight vector | 21 | Full 21d weight distribution |
| Posterior uncertainty | 1 | Mean variance of Beta distributions = mean(αβ/(α+β)²(α+β+1)) |

Low entropy (concentrated weights) suggests a clear symptom profile; high uncertainty suggests insufficient evidence to differentiate symptoms.

---

## 8. Thread Formatting for LLM Prompts

When preparing threads for the ToM module's LLM prompts, we format the hierarchical thread into flat chronological text with a **2000-token budget** (~8000 characters at 4 chars/token).

### 8.1 Priority-Based Truncation

| Priority | Content | Treatment |
|----------|---------|-----------|
| **P1** (always full) | Target user posts + direct replies to target | Full text, never omitted |
| **P2** (full if budget) | Other posts in branches containing target | Full text, omitted only if over budget |
| **P3** (truncated) | Non-target posts in target-adjacent branches | First 100 chars + "...", omitted if severely over budget |
| **P4** (omitted) | Posts in branches with no target participation | Always omitted |

### 8.2 Output Format

```
=== THREAD (Round N) ===
Title: [thread title]

[POST] author [TARGET]: [submission body]
[REPLY to POST by author [TARGET]] replier: [reply text]
[REPLY to replier] author [TARGET]: [response text]

=== END THREAD ===
```

The `[TARGET]` tag marks the target user's contributions, enabling the LLM to distinguish self-expression from community response.

---

## 9. Feature Vector Assembly

All extracted features are concatenated into a single vector of approximately **2341 dimensions**:

| Component | Dimension | Source |
|-----------|-----------|--------|
| Embedding running mean | 1920 | Layer 1: 3-model ensemble |
| Symptom max-pool | 21 | Layer 1: peak activations |
| Symptom mean-pool | 21 | Layer 1: average activations |
| Symptom distributional stats | 147 | Layer 1: 21 × 7 stats |
| Lexical indicators | 4 | Layer 1: word ratios |
| Reply sentiment | 3 | Layer 2: mean, std, trend |
| Concern ratio | 1 | Layer 2: phrase detection |
| Conversational position | 3 | Layer 2: author/silent/depth |
| Thread topic similarity | 21 | Layer 2: vs. BDI-II references |
| Emotion | 9 | Layer 3: 8 emotions + entropy |
| BERTopic | 41 | Layer 3: 40 topics + dep. ratio |
| Wasserstein distances | 72 | 3 scales × (21 sym + emo + emb + topic) |
| Mahalanobis distances | 3 | control, relative, depressed |
| Combined distributional | 2 | D_M_control + W_mean |
| Theory of Mind | 47 | Self/observer/gap features |
| Thompson Sampling | 25 | Weighted score + weights + meta |
| Meta | 1 | rounds_seen |
| **Total** | **~2341** | |

**Feature masking**: For ablation runs, specific components can be zeroed out via a `feature_mask` parameter (e.g., `["no_tom"]` replaces the 47d ToM block with zeros).

---

## 10. Classification Layer

### 10.1 Classifiers

We train four classifier types, all sharing a common interface (`fit`, `predict_proba`, `save`/`load`):

#### XGBoost
- max_depth=6, n_estimators=300, learning_rate=0.1
- `scale_pos_weight = n_neg / n_pos` for class imbalance
- StandardScaler preprocessing
- Eval metric: log-loss

#### Neural Net (2-layer MLP)
- Architecture: Linear(input, 256) → BatchNorm → ReLU → Dropout(0.3) → Linear(256, 64) → BatchNorm → ReLU → Dropout(0.3) → Linear(64, 1)
- Loss: BCEWithLogitsLoss with pos_weight = n_neg / n_pos
- Optimizer: Adam(lr=1e-3)
- Early stopping: patience=10, monitoring validation loss
- Max epochs: 100, batch size: 64

#### SVM
- Kernel: RBF, probability=True, class_weight='balanced'
- StandardScaler preprocessing

#### Ensemble (Stacking)
- Base learners: XGBoost + MLP + SVM (all three above)
- Meta-learner: LogisticRegression(class_weight='balanced')
- Training: 5-fold CV to generate out-of-fold predictions → train meta-learner on stacked base predictions
- Prediction: base predictions → (3d) vector → meta-learner → final probability

### 10.2 Class Imbalance Handling

All classifiers explicitly handle the 11.2% depressed / 88.8% control imbalance through:
- **XGBoost**: `scale_pos_weight` = ~7.9
- **MLP**: `pos_weight` in BCE loss = ~7.9
- **SVM**: `class_weight='balanced'`
- **Ensemble meta-learner**: `class_weight='balanced'`

---

## 11. Training Procedure

### 11.1 User Processing (Feature Extraction Phase)

1. **Load** all 909 users and their labels from the training data directory.
2. **Initialize** models: 3 sentence transformers, symptom scorer (with 21 reference embeddings), Thompson sampler, emotion classifier, BERTopic (unfitted).
3. **For each user** (with checkpointing every 50 users):
   - **Subsample** threads: keep up to 100 evenly spaced threads (for computational efficiency — some users have 300+ threads).
   - **For each thread**: call `process_thread()` which updates the user's profile in-place with Layer 1/2/3 features, ToM Option A embeddings, and bandit posteriors.
4. **Fit BERTopic**: Collect all target texts from all users (≥10 words), fit the topic model using depression labels to identify depression-associated topics. Transform each user's rolling text buffer.

### 11.2 Classification Training Phase

5. **Extract feature vectors**: Call `compute_final_features()` for each user to produce the ~2341d vector from accumulated profile state.
6. **Fit Mahalanobis scorer**: PCA → LedoitWolf covariance estimation on control users (and separately on depressed users for relative distance). Inject Mahalanobis scores back into the feature matrix.
7. **5-fold stratified cross-validation**: For each classifier type (xgboost, neural_net, svm, ensemble):
   - StratifiedKFold split (preserving 11.2% class ratio per fold)
   - Train on fold training set, evaluate F1 on fold validation set
   - Compute ERDE5, ERDE50, F_latency on pooled out-of-fold predictions
8. **Train final models**: Train each classifier on the full dataset.
9. **Save** all artifacts: classifiers (pickle/torch), Mahalanobis model, BERTopic model, symptom reference embeddings, feature matrix (npz), training results (JSON).

### 11.3 NaN and Infinity Handling

After feature assembly, `np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)` is applied to handle edge cases (e.g., division by zero in sentiment trends, empty embedding histories).

---

## 12. Decision Policy (Alert Timing)

### 12.1 ERDE Latency Cost Function

The ERDE metric penalizes late alerts with a sigmoidal cost function:

```
lc_o(k) = 1 − 1 / (1 + e^(k − o))
```

Where `k` is the alert round and `o` is the ERDE operating point. This function:
- At k ≪ o: lc_o(k) ≈ 0 (low latency cost → early alert is cheap)
- At k = o: lc_o(k) = 0.5 (half penalty)
- At k ≫ o: lc_o(k) ≈ 1 (near-maximum penalty → late alert is expensive)

### 12.2 Adaptive Threshold

The decision threshold decays from an initial conservative value toward a floor as rounds progress, following the ERDE cost curve:

```
θ(k) = θ_init − (θ_init − θ_floor) × lc_o(k)
```

This means the system starts cautious (high threshold, only very confident predictions trigger alerts) and becomes progressively more willing to alert as the latency cost increases.

### 12.3 Consecutive Confirmation

An alert requires `t_con` consecutive rounds where `P(depressed) ≥ θ(k)`. This prevents single-round fluctuations from triggering irreversible alerts. If any round drops below threshold, the counter resets to 0.

### 12.4 Alert Irreversibility

Once `decision=1` is emitted for a user in a given run, it remains 1 for all subsequent rounds. This is a task requirement — the system cannot retract an alert.

---

## 13. Submission Runs (5 Official Runs)

Each run uses the same feature extraction pipeline but differs in classifier type, decision policy parameters, or feature mask:

| Run | Classifier | θ_init | θ_floor | ERDE o | t_con | Feature Mask | Strategy |
|-----|-----------|--------|---------|--------|-------|-------------|----------|
| **R0** | XGBoost | 0.85 | 0.45 | 50 | 2 | Full | Conservative baseline — high threshold, slow decay, double confirmation |
| **R1** | XGBoost | 0.70 | 0.35 | 5 | 1 | Full | Aggressive early — lower threshold, rapid decay, single confirmation (optimizes ERDE5) |
| **R2** | MLP | 0.85 | 0.45 | 50 | 2 | Full | Neural alternative — tests if MLP captures non-linear patterns missed by XGBoost |
| **R3** | Ensemble | 0.80 | 0.40 | 30 | 2 | Full | Stacked ensemble — XGB+MLP+SVM meta-learned, balanced decay (optimizes overall ranking) |
| **R4** | XGBoost | 0.85 | 0.45 | 50 | 2 | `["no_tom"]` | Ablation — identical to R0 but with ToM features zeroed, isolating ToM contribution |

**Run diversity rationale**: The 5 runs are designed to:
1. Hedge across ERDE5 vs ERDE50 operating points (R1 vs R0/R2/R3/R4).
2. Compare classifier architectures (R0 vs R2 vs R3).
3. Measure ToM contribution (R0 vs R4).
4. Provide a robust ensemble option (R3).

---

## 14. Server Interaction (Live Competition)

### 14.1 Protocol

The live pipeline operates as a GET/POST loop against the eRisk server:

1. **GET** `/getdiscussions/{token}` — retrieves one thread per user for the current round.
2. **Process** threads through the feature extraction pipeline.
3. **Classify + Decide** for each run × each user.
4. **POST** `/submit/{token}/{run_number}` — submits decisions in format `[{nick, decision, score}, ...]`.
5. **Checkpoint** full state (profiles, run states, master user list) to disk.
6. **Repeat** until empty server response indicates all rounds are complete.

### 14.2 Resilience

- **Retry with exponential backoff**: max_retries=5, initial_delay=2s, backoff_factor=2.0 (2s → 4s → 8s → 16s → 32s).
- **Crash recovery**: Full state is pickled after each round. On restart, the latest checkpoint is loaded and processing resumes from the next round.
- **Master user list**: Captured on round 0 and used to initialize profiles for all users, ensuring no user is missed even if they are absent from later rounds.

---

## 15. DSM-5 to BDI-II Mapping

The system includes a structured mapping from DSM-5 major depressive episode criteria to BDI-II items, used for clinical grounding of symptom references:

| DSM-5 Criterion | BDI-II Items |
|-----------------|-------------|
| Depressed mood | 1 (Sadness), 10 (Crying), 17 (Irritability) |
| Anhedonia | 4 (Loss of pleasure), 12 (Loss of interest), 21 (Loss of interest in sex) |
| Appetite/weight changes | 18 (Appetite changes) |
| Sleep disturbance | 16 (Sleep changes) |
| Psychomotor changes | 11 (Agitation) |
| Fatigue/loss of energy | 15 (Loss of energy), 20 (Tiredness/fatigue) |
| Worthlessness/guilt | 5 (Guilt), 6 (Punishment), 7 (Self-dislike), 8 (Self-criticalness), 14 (Worthlessness) |
| Diminished concentration | 13 (Indecisiveness), 19 (Concentration difficulty) |
| Suicidal ideation | 9 (Suicidal thoughts) |

---

## 16. External Models and Resources

| Resource | Usage | Source |
|----------|-------|--------|
| `all-mpnet-base-v2` | Sentence embedding (768d) | sentence-transformers (HuggingFace) |
| `all-MiniLM-L12-v2` | Sentence embedding (384d) | sentence-transformers (HuggingFace) |
| `all-distilroberta-v1` | Sentence embedding (768d) | sentence-transformers (HuggingFace) |
| `j-hartmann/emotion-english-distilroberta-base` | Emotion classification (8 classes) | HuggingFace |
| Llama 3.3 70B | ToM dual mentalizing (Option C) | Ollama (local) |
| VADER | Sentiment analysis | vaderSentiment (NLTK) |
| BERTopic | Dynamic topic modeling | bertopic (HuggingFace) |
| UMAP + HDBSCAN | Dimensionality reduction + clustering (BERTopic internals) | umap-learn, hdbscan |
| XGBoost | Gradient boosted classifier | xgboost |
| PyTorch | MLP classifier | torch |
| scikit-learn | SVM, PCA, LedoitWolf, StratifiedKFold, LogisticRegression | sklearn |

---

## 17. Key Design Decisions and Rationale

1. **Incremental state accumulation**: Rather than re-processing all historical threads at each round, we maintain a running profile per user with exponentially decayed embeddings, cumulative text, and per-round feature histories. This enables O(1) per-round processing regardless of history length.

2. **3-model embedding ensemble**: Captures complementary semantic aspects (paraphrase sensitivity, distributional similarity, broad coverage) in a single 1920d vector, without the computational cost of fine-tuning.

3. **Multi-scale Wasserstein distances**: By computing distributional shifts at 3 temporal scales (short/medium/long), we capture both acute deterioration and chronic trajectory changes without requiring a fixed window assumption.

4. **Mahalanobis with dual distributions**: Computing distance from both control and depressed reference distributions (with relative distance) provides a stronger anomaly signal than distance from controls alone, since it captures directionality.

5. **Theory of Mind module**: The self-view/observer-view discrepancy is a unique signal not captured by standard NLP features. Depressed users may exhibit "insight gap" — their self-expression may not match how the community perceives their situation.

6. **Thompson Sampling for symptom weighting**: Provides personalized, data-driven importance weighting that adapts to each user's specific depression profile as evidence accumulates, rather than treating all 21 BDI-II symptoms equally.

7. **ERDE-aware adaptive threshold**: The threshold decay directly mirrors the evaluation metric's latency cost, ensuring the system becomes more aggressive at alerting precisely when the cost of waiting exceeds the cost of a potential false positive.

8. **5-run diversity**: Different classifiers, ERDE operating points, and an ablation run maximize the chance of achieving strong performance on at least one official metric while providing scientific insight into component contributions.

9. **Priority-based thread truncation**: Ensures the most diagnostically relevant content (target user's own words + direct community responses) always reaches the LLM, even for threads with hundreds of comments.

10. **Checkpoint-and-resume design**: Full state serialization after every round ensures zero-loss recovery from crashes during the multi-day live competition.
