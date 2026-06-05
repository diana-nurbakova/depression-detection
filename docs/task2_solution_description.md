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

We scan all thread text (all non-target comments, plus the submission body when the target is not the author) for **42 concern phrases** (case-insensitive substring match) that indicate community worry about the target user, such as:
- "are you okay", "please get help", "reach out to someone"
- "suicide hotline", "crisis line", "988"
- "i'm worried", "take care of yourself", "hope you're doing better"

A round is flagged if any phrase matches (one match per comment is enough). The feature is the **ratio of rounds with at least one match** across the user's history. See [src/erisk_task2/features/layer2.py:22-97](../src/erisk_task2/features/layer2.py#L22-L97).

#### 4.2.3 Conversational Position (3d)

Structural features characterizing how the target user participates in discussions:
- **is_author_ratio**: fraction of threads where the target user is the submission author (initiator vs. commenter).
- **target_silent_ratio**: fraction of rounds with no target text at all (withdrawal signal).
- **reply_depth_mean**: average depth of target user's comments in the reply tree (surface-level participation vs. deep engagement).

#### 4.2.4 Thread Topic Similarity (21d)

For each thread, the title (or body if title is absent) is embedded using the same 3-model ensemble and compared via cosine similarity to the 21 BDI-II symptom reference embeddings. This captures whether the target user tends to participate in threads whose topics are depression-adjacent. The final feature is the mean of these (21d) vectors across all rounds.

### 4.3 Layer 3: Emotion and Topic Features (~50d)

#### 4.3.1 Emotion Classification (9d)

We use the `j-hartmann/emotion-english-distilroberta-base` model. Our intent at design time was Plutchik's 8 primary emotions; the **actual deployed pairing is mismatched** — see §4.3.1a immediately below. The wire shape of the feature block is unchanged (9d = 8 emotion slots + 1 entropy), but the *semantics* of those 8 slots differ from what the doc previously claimed.

**Slot layout in the feature vector (`PLUTCHIK_EMOTIONS` in [layer3.py:18-22](../src/erisk_task2/features/layer3.py#L18-L22)):**

| Slot | Name in code (Plutchik) | Populated at runtime? |
| --- | --- | --- |
| 0 | Anger | yes — matches model's `anger` |
| 1 | Anticipation | **never** — model has no `anticipation` class |
| 2 | Disgust | yes — matches model's `disgust` |
| 3 | Fear | yes — matches model's `fear` |
| 4 | Joy | yes — matches model's `joy` |
| 5 | Sadness | yes — matches model's `sadness` |
| 6 | Surprise | yes — matches model's `surprise` |
| 7 | Trust | **never** — model has no `trust` class |

Model's `neutral` class has no destination slot (no Plutchik name is a substring of `"neutral"`) and is silently dropped before renormalisation (see §4.3.1a).

**Per-round**: Each target text is classified → 8-d distribution buffer (slots 1 and 7 always 0). Texts with fewer than 10 words receive a uniform (1/8) distribution to avoid noisy classifications on very short texts. Per-round distributions are averaged.

**Final features (9d)**: Mean emotion distribution across all rounds (8d) + Shannon entropy of the mean distribution (1d). Low entropy indicates dominant emotional state; high entropy indicates emotional variability. Because two of the eight slots are structurally zero, the maximum achievable entropy is `log(6) ≈ 1.79` (not `log(8) ≈ 2.08`), but the entropy is *computed* over the full 8-d buffer, so the published value is on the smaller scale.

#### 4.3.1a Taxonomy mismatch: Plutchik-8 vs Ekman-6 + neutral (known bug)

The constant in [layer3.py:17-22](../src/erisk_task2/features/layer3.py#L17-L22) declares Plutchik's 8 emotions, but `j-hartmann/emotion-english-distilroberta-base` is trained on **Ekman's 6 emotions plus a `neutral` class** (7 classes total). The pairing was an authoring error; the two label sets agree on 6 emotions and diverge on the rest.

**Verified empirically.** Loading the model and running it on three sample inputs returns 7 labels per prediction. Its `config.id2label`:

```python
{0: 'anger', 1: 'disgust', 2: 'fear', 3: 'joy',
 4: 'neutral', 5: 'sadness', 6: 'surprise'}
```

The final classification head is `Linear(hidden_dim → 7)` — there is no neuron that can fire `anticipation` or `trust`. It is a structural impossibility, not a probabilistic rarity.

**The two vocabularies side-by-side:**

| Class | In Plutchik (code's assumption) | In Ekman + neutral (actual model) | Result at runtime |
| --- | --- | --- | --- |
| Anger | ✓ slot 0 | ✓ id 0 | mapped → slot 0 |
| Disgust | ✓ slot 2 | ✓ id 1 | mapped → slot 2 |
| Fear | ✓ slot 3 | ✓ id 2 | mapped → slot 3 |
| Joy | ✓ slot 4 | ✓ id 3 | mapped → slot 4 |
| Sadness | ✓ slot 5 | ✓ id 5 | mapped → slot 5 |
| Surprise | ✓ slot 6 | ✓ id 6 | mapped → slot 6 |
| **Anticipation** | ✓ slot 1 | ✗ not in model | **slot 1 always 0** |
| **Trust** | ✓ slot 7 | ✗ not in model | **slot 7 always 0** |
| **Neutral** | ✗ not in code | ✓ id 4 | **score discarded before renormalisation** |

**The dispatch loop ([layer3.py:65-83](../src/erisk_task2/features/layer3.py#L65-L83))**:

```python
for p in preds:                                  # 7 iterations (model labels)
    label = p["label"].lower()
    for j, emotion in enumerate(PLUTCHIK_EMOTIONS):
        if emotion in label:                     # first-match-wins substring test
            dist[j] = p["score"]
            break                                # nothing happens if no name matches
# ...
total = dist.sum()
if total > 0:
    dist = dist / total                          # renormalises over 6 non-zero slots
```

Worked example on the input `"I feel completely hopeless and empty inside, nothing matters anymore."` (model output sorted by score):

| Model output | Slot written | Comment |
| --- | --- | --- |
| `sadness 0.9822` | `dist[5] = 0.9822` | matches `sadness ∈ "sadness"` |
| `neutral 0.0068` | none | **dropped** — no Plutchik name is a substring of `"neutral"` |
| `disgust 0.0056` | `dist[2] = 0.0056` | |
| `fear 0.0019` | `dist[3] = 0.0019` | |
| `anger 0.0014` | `dist[0] = 0.0014` | |
| `surprise 0.0012` | `dist[6] = 0.0012` | |
| `joy 0.0010` | `dist[4] = 0.0010` | |

After renormalisation (total = 0.9933; neutral's 0.0068 is gone):

```text
dist = [0.00141,  0,       0.00564,  0.00191,  0.00101,  0.98886,  0.00121,  0     ]
        anger    anticip.  disgust   fear      joy       sadness   surprise  trust
```

**Three consequences for the trained pipeline.**

1. **Two wasted feature dimensions.** Slots 1 and 7 are exactly zero in every emotion vector ever produced — for every user, every round, every fold. XGBoost still has to consider them as split candidates (each split iteration scans every feature column); they will never be selected, but they cost a small amount of training-time overhead and pollute the feature-importance ranking with two guaranteed zeros.
2. **Distorted distributions for emotionally-bland posts.** When the model says `0.95 neutral + 0.01 × {anger, disgust, …}`, the renormalisation amplifies each surviving 0.01 to roughly 0.167 across the 6 active slots. The downstream Shannon entropy then reads "near-uniform emotional state" (high entropy) when the model's actual reading was "this text is mostly neutral" (which carries different prior weight on depression). The Reddit base rate of low-affect posts is high, so this distortion touches a non-trivial share of inputs.
3. **Entropy ceiling shift.** Because the support is 6 active dims, the achievable Shannon-entropy range is `[0, log 6 ≈ 1.79]`, not `[0, log 8 ≈ 2.08]`. The classifier learned its split thresholds on this restricted range, so the bug is internally consistent for training and inference — but the "entropy of an 8-emotion distribution" interpretation in earlier drafts of this doc is wrong.

**Fix options.** A correct mapping would replace `PLUTCHIK_EMOTIONS` with the Ekman+neutral set (`["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]`), changing the block dimension from 9d to 8d (7 emotion slots + 1 entropy). That would mean re-training every classifier on the corrected feature matrix; we did not do this for the submission because (a) the wasted-slot cost is bounded and (b) the entropy-ceiling shift is internally consistent across train and inference. The bug is recorded here and in [task2_results_analysis.md](task2_results_analysis.md) §6 as a known limitation of the submission run.

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

**Per-user transform**: A rolling buffer of the last 5 target texts is maintained. The concatenated buffer text is run through BERTopic's `transform([text])` — the **40d topic distribution is the soft (probabilistic) output**, while the depression-proportion uses the **hard top-1 topic** (`topics[0] ∈ depression_topic_ids`).

**Depression-topic identification (fit time)**: A topic is labeled "depression-related" if (a) total count > 10 and (b) `dep_count / total > 1.5 × base_rate` ([layer3.py:182-188](../src/erisk_task2/features/layer3.py#L182-L188)).

**Final features (41d)** ([feature_assembler.py:33](../src/erisk_task2/classification/feature_assembler.py#L33), [pipeline.py:258-275](../src/erisk_task2/pipeline.py#L258-L275)):

- 40d topic distribution vector (last round's `transform` output, zero-padded if BERTopic returned fewer than 40 topics).
- **1d mean depression-proportion across all rounds** (NB: the inline comment "40d distribution + entropy" at `pipeline.py:272` is misleading — the assembled slot 40 is the mean of `depression_proportion`, not the topic entropy. Topic entropy is computed and used elsewhere in the Wasserstein block, but not in this 41-d vector).

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

**Combined distributional score (2d)** ([feature_assembler.py:136-142](../src/erisk_task2/classification/feature_assembler.py#L136-L142)): a 2-scalar summary block produced only when both upstream blocks are present (else zeroed).

- **Slot [0]** = `mahalanobis_features[0]` = **D_M_control**, identical to the first Mahalanobis slot. A literal duplicate, not a re-weighted or rescaled version — XGBoost simply sees the same scalar twice. The duplication is what the code does; it is not load-bearing modelling-wise.
- **Slot [1]** = `wasserstein_features.mean()` = **unweighted arithmetic mean of the 72-d Wasserstein vector** (= flat average across 63 symptom × 3 scales + 3 emotion + 3 embedding + 3 topic). No length-normalisation, no scale-specific weighting, no per-block re-scaling beyond what each component natively produces.

Both slots are zeroed when either Mahalanobis or Wasserstein is unavailable for a user (e.g., insufficient round history).

---

## 6. Theory of Mind Module (47d)

The Theory of Mind (ToM) module captures **dual perspectives** on the target user: how the target presents themselves (self-view) versus how the community perceives them (observer-view). The discrepancy between these perspectives is itself a diagnostic signal — depressed individuals often show reduced insight into how their behavior appears to others.

### 6.1 Implementation Options

Three implementation options are available, all producing the same 47d feature vector:

| Option | Method | LLM Required | Where it was used |
|--------|--------|-------------|--------|
| **Option A** | Embedding-based | No | **Live test run** (`run_pipeline`, local machine) |
| **Option B** | Response category classification | Yes | Implemented, not used |
| **Option C** | LLM dual mentalizing | Yes | **Training feature build** (Colab notebook) |

> **Note — ToM train/test mismatch (important).** The two phases used *different* ToM implementations:
>
> - **Training features (Colab, `notebooks/task2.ipynb`)** were built with **Option C — LLM dual mentalizing** via `ToMModule(method="option_c", chained=False, symptom_variant="C")` using the **HuggingFace Inference API** (`huggingface_hub.InferenceClient`) running **`meta-llama/Llama-3.3-70B-Instruct`** — *not* Ollama (the config default `backend="ollama"` is misleading), and *not* TogetherAI (a Together key was set as a fallback but the ToM client was the HF one). The result was saved to `tom_features.npz` (47d × 909 users) and loaded by the notebook (cell 63) to train the shipped "with-ToM" classifiers. Confirmation the LLM ran: observer-view symptom dimensions in the npz hold exactly {0.333, 0.667, 1.0} (= LLM scores 1/2/3 ÷ 3), and the concern/community-response dims are populated — both impossible under Option A.
> - **The live test run** (executed locally via `run_pipeline` in `src/erisk_task2/pipeline.py`) computes ToM **inline as Option A** (embedding-based) via `_compute_tom_a_features()` — **no LLM** — and with community-comment (observer) encoding disabled for latency (see §6 runtime note in the results analysis). At inference, therefore, only the self-view embedding norm `[42]` is non-zero; the observer norm `[43]`, insight gap `[44]`, the 21+21 symptom-score slots `[0:42]`, and the concern/community slots `[45:46]` are all ~0.
>
> **Consequence:** the classifiers learned weights on rich LLM-derived ToM features (observer symptom scores, concern level, community-response type, insight gap) that are fed near-zeros at test time — a genuine train/test mismatch. It also explains why **R0 (full features) ≈ R4 (no-ToM ablation)** on the live submission: same XGBoost model, the `no_tom` mask just zeroes the 47d block, and R0's block is already ~all-zero live, so the two runs differ by a single scalar (1 subject / 1 FP in our re-score). The no-ToM ablation is thus confounded — it could not measure the ToM contribution that training actually relied on. `src/erisk_task2/pipeline.py` never loads `tom_features.npz`; that LLM path lives only in the Colab notebook and `src/erisk_task2/tom/`.
>
> **A second train/test heterogeneity — BERTopic topic-features (41-d).** The 41 Layer-3 BERTopic dimensions ([feature_assembler.py:124-128](../src/erisk_task2/classification/feature_assembler.py#L124-L128)) hold real topic-similarity values in training (the BERTopic model was fitted on training texts; see §4.3.2) but are **not uniform at test time**. The live run was produced in two regimes (boundary verified in `runs/task2/train/run.log`):
>
> - **Phase 1 (rounds 0–19, 2026-04-17 → 04-19):** BERTopic loaded and used — 41 real topic features per subject per round, the closest-to-training feature configuration.
> - **Phase 2 (rounds 20–500, 2026-04-19 → 04-23):** BERTopic skipped in live mode ([pipeline.py:936](../src/erisk_task2/pipeline.py#L936), `BERTopic skipped in live pipeline (memory optimization)`) — the 41 topic features are zeroed for memory reasons (BERTopic loads its own 4th sentence-transformer alongside the 3 encoders and OOM'd on the local box).
>
> So the early-round and late-round feature vectors come from *different distributions* in two slots (topic features and the ToM observer dimensions described above). Because ERDE5 / latency-TP weight *early* correct positives most heavily, and because **50 of Run 0's 101 final alerts fired in Phase 1**, a disproportionate share of the latency-aware score reflects the BERTopic-on / observer-view-on regime. See [docs/task2_results_analysis.md](task2_results_analysis.md) §6 for the per-phase timing detail and round counts. (The same heterogeneity affects every classifier-based run — R0–R4 — not just the ToM ablation.)

### 6.2 Option C: LLM-Based Dual Mentalizing (used to build the training features)

Uses **`meta-llama/Llama-3.3-70B-Instruct`** via the **HuggingFace Inference API** to generate structured JSON assessments. (The codebase also ships an `OllamaClient` with the same interface; the config default points at Ollama, but the Colab feature build used the HuggingFace `InferenceClient`.)

**Decoding & request parameters** ([hf_client.py:21-86](../src/erisk_task2/tom/hf_client.py#L21-L86), [llm_client.py:21-85](../src/erisk_task2/tom/llm_client.py#L21-L85)):

| Parameter | HF Inference (used for training features) | Ollama (config default) |
| --- | --- | --- |
| Temperature | 0.1 (all prompts, all attempts) | 0.1 |
| Max output tokens | 2048 (`max_tokens`) | uncapped (`num_predict` not set → model default) |
| Context window | n/a (chat-completion endpoint) | `num_ctx = 8192` |
| Timeout | 120 s | 120 s |
| Retries | `max_retries = 3` (up to 2 retries after first call) | `max_retries = 3` |
| Retry back-off | `2 ** attempt` seconds | `2 ** attempt` seconds |
| Retry temperature override | **none** (same 0.1 on every attempt) | **none** |
| JSON fallback | regex-based markdown/code-fence stripping in `_extract_json`; on parse failure, the slot is recorded as `None` and the corresponding 21-d sub-vector is left at zero — there is no second LLM call with a stricter temperature | same |

**Batching policy**: per-thread ToM is **sequential** — for each (user, round) we issue at most two calls (self-view then observer-view; chained mode would pipe the self-view JSON into the observer prompt, but the config default is `chained=False`, so they are independent). No batching across users or across rounds. System prompts are pre-formatted in `ToMModule.__init__` and reused byte-identically for KV-cache reuse on the Ollama side.

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

### 6.4 ToM Feature Vector (47d) — Training-Time Layout

This is the layout the classifiers were **trained** on (Option C, LLM dual mentalizing). Live-mode (Option A) actually populates only the highlighted row — see §6.5.

| Offset | Dimension | Feature (training, Option C) | Live (Option A) value |
| --- | --- | --- | --- |
| 0–20 | 21 | Self-view symptom scores (normalized to 0–1 from LLM scores 1–3) | **0** |
| 21–41 | 21 | Observer-view symptom scores | **0** |
| **42** | 1 | Self depression_probability (LLM, ∈ [0, 1]) | **`‖ mean_over_rounds( round_mean ) ‖₂`** — unbounded L2 norm |
| 43 | 1 | Observer depression_probability | **0** (observer embeddings not computed live) |
| 44 | 1 | Insight gap (mean abs diff between self and observer symptom scores) | **0** |
| 45 | 1 | Observer concern level (0–3, normalized to 0–1) | **0** |
| 46 | 1 | Community response type encoded (concern→1.0, support→0.8, advice→0.6, mixed→0.5, normalization→0.3, casual→0.0) | **0** |

### 6.5 ToM at inference (live mode) — what actually populated each slot

The live pipeline does **not** call any LLM. It computes Option A inline, and within Option A it skips encoding `other_comments` to avoid a ~10× sentence-transformer cost ([pipeline.py:995-996](../src/erisk_task2/pipeline.py#L995-L996), [pipeline.py:214-225](../src/erisk_task2/pipeline.py#L214-L225)). The consequence is that **only slot [42] is non-zero at inference**, and the value at slot [42] is not on the same scale as the training-time value.

**Slot [42], live**: `‖ mean_over_rounds( round_mean_target_text_embedding ) ‖₂` — the L2 norm of the **centroid** of the per-round mean target-text embeddings (`mean → norm`, in that order, not `norm → mean`).

**Three caveats to keep in mind when reading this slot:**

1. **Order matters.** `_compute_tom_a_features` ([pipeline.py:326-362](../src/erisk_task2/pipeline.py#L326-L362)) stacks per-round means, averages them, then takes a single norm. It is therefore the magnitude of the centroid — not the average magnitude of the rounds.
2. **Embeddings are not L2-normalised.** The 1920-d concat of mpnet-base-v2 + MiniLM-L12-v2 + distilroberta-v1 is the raw `encode()` output. Each model's `encode()` returns un-normalised vectors by default. The resulting norm is unbounded above and tracks "how much accumulated semantic content" the target's own posts carry, not a probability.
3. **Training–test distributional mismatch at slot [42].** During Colab training-feature build, slot [42] was `self_view["depression_probability"]` from Prompt 1, a Llama-3.3-70B-emitted scalar in `[0, 1]`. At inference, the same slot carries an unnormalised embedding norm. The XGBoost split thresholds learned around `[0, 1]` simply do not fire on the live distribution.

**Mechanism behind R0 ≈ R4.** Combined with `[0:42] = 0`, `[43] = 0`, `[44] = 0`, `[45] = 0`, `[46] = 0`, only one of 47 slots carries any live signal — and it is on the wrong scale. Zeroing the ToM block via `feature_mask=["no_tom"]` (R4) is therefore numerically almost identical to leaving it as-is (R0): R4 zeros 47 slots; R0 has 46 of them already at zero and the 47th miscalibrated. The ablation could not measure the ToM contribution that training actually relied on. See §18.4 / §18.5 and the train/test mismatch note in §6.1.

The full call-shape, retry policy, temperature, JSON-enforcement, batching, and output-token-cap details are in [docs/task2_prompts_extracted.md §9](task2_prompts_extracted.md#9-appendix-b--call-shape--runtime-details).

---

## 7. Thompson Sampling for Symptom Weighting (25d)

### 7.1 Motivation

Not all 21 BDI-II symptoms are equally informative for every user. Some users' depression manifests primarily as cognitive symptoms (pessimism, worthlessness, guilt), while others show primarily somatic symptoms (sleep, appetite, fatigue). Thompson Sampling provides a principled way to learn which symptoms are most consistently activated for each individual user, dynamically up-weighting the most informative symptoms.

### 7.2 Mechanism

We maintain a **21-armed bandit** with **Beta(α, β) posteriors** per symptom per user, initialized to Beta(1, 1) (uniform prior).

**Per-round update**:

1. Observe symptom activation vector (21d) from the current round.
2. For each symptom: if `activation > τ_active`, mark as **active**. The activation threshold is **τ_active = 0.3** (`config.symptom.activation_threshold` in [config/task2.yaml:19](../config/task2.yaml#L19); passed as `tau_active` to `ThompsonSampler` in [pipeline.py:411](../src/erisk_task2/pipeline.py#L411)). The same threshold also drives the `active_symptom_count` feature reported below.
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

- Kernel: RBF, `probability=True`, `class_weight='balanced'`, `random_state=42`
- `gamma='scale'` and `C=1.0` — left at sklearn defaults (not tuned)
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
| Llama 3.3 70B (`meta-llama/Llama-3.3-70B-Instruct`) | ToM dual mentalizing (Option C) — used to build **training** features; live run falls back to embedding Option A (see §6 note) | HuggingFace Inference API |
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

---

## 18. Experiments and Evaluation

### 18.1 Evaluation Methodology

All offline experiments use **5-fold stratified cross-validation** on the eRisk 2025 training set (909 users: 102 depressed, 807 control). Stratification preserves the 11.2% class ratio in each fold.

**Metrics**:
- **F1**: Harmonic mean of precision and recall on the binary depression label.
- **ERDE_o** (Early Risk Detection Error at operating point o): Penalizes late alerts via a sigmoidal cost function. Lower is better. Computed at o=5 (aggressive) and o=50 (conservative).
- **F_latency**: F1 discounted by median alert speed. Rewards both accuracy and timeliness.

For the evaluation pipeline, the system **simulates** round-by-round processing on the full training set (all threads in sequential order), applying the adaptive threshold and consecutive confirmation logic exactly as in the live competition.

### 18.2 Cross-Validation Results (Without ToM)

Training with the full feature set excluding ToM features (Option A embedding fallback used, producing near-zero ToM features):

| Classifier | F1 (mean ± std) | ERDE5 | ERDE50 | F_latency |
|------------|-----------------|-------|--------|-----------|
| **XGBoost** | 0.722 ± 0.077 | 0.070 | 0.064 | 0.718 |
| **MLP** | 0.686 ± 0.063 | 0.086 | 0.081 | 0.686 |
| **Ensemble** | 0.693 ± 0.043 | 0.085 | 0.078 | 0.693 |

XGBoost achieves the best F1 and lowest ERDE scores. The ensemble shows the lowest variance (std=0.043) but does not outperform the single XGBoost model — likely because the SVM base learner contributes noise in high dimensions.

### 18.3 Cross-Validation Results (With ToM — Option C, LLM dual mentalizing)

After adding the 47d Theory of Mind feature block, built with **Option C** — `meta-llama/Llama-3.3-70B-Instruct` via the **HuggingFace Inference API** (not Ollama; see §6 note). These `training_results_with_tom.json` numbers are computed on the Colab-built `tom_features.npz`. **Caveat:** the live submission does *not* reproduce this setting — `run_pipeline` falls back to embedding-based Option A with the observer view disabled, so the LLM ToM signal measured here is largely absent at test time (the train/test mismatch documented in §6).

| Classifier | F1 | ERDE5 | ERDE50 | F_latency |
|------------|------|-------|--------|-----------|
| **XGBoost** | 0.724 | 0.072 | 0.064 | 0.721 |
| **MLP** | 0.704 | 0.079 | 0.070 | 0.704 |
| **Ensemble** | 0.726 | 0.082 | 0.069 | 0.725 |

**ToM contribution** (Option C, LLM dual mentalizing via Llama-3.3-70B): adding the 47d ToM features provides a consistent improvement in cross-validation:
- XGBoost: F1 +0.2pp, ERDE5 −0.2pp (marginal)
- MLP: F1 +1.8pp, ERDE50 −1.1pp (moderate)
- Ensemble: F1 +3.3pp, ERDE50 −0.9pp (substantial)

The ensemble benefits most from ToM features, likely because the meta-learner can learn to weight the ToM signal appropriately across different user profiles. The MLP shows moderate gains, suggesting ToM features provide non-linear patterns that complement the neural network's capacity. **However, this gain does not transfer to the live submission:** the live run computes ToM with embedding-based Option A and the observer view disabled, so the rich LLM dimensions measured here (observer symptom scores, concern, community response) are ~zero at test time. The §18.4 round-by-round simulation — which also uses Option A — shows R0 and R4 (no-ToM) as numerically identical, confirming that under the live ToM regime the block contributes essentially nothing. The true value of LLM ToM therefore remains effectively untested on the submission.

### 18.4 Simulated Evaluation (Round-by-Round, Full Training Set)

The evaluation pipeline simulates live competition conditions by processing all 909 users sequentially through the full pipeline with all 5 run configurations. Results using models trained on the full dataset:

| Run | Classifier | F1 | Precision | Recall | ERDE5 | ERDE50 | Alerts | Mean Alert Round |
|-----|-----------|------|-----------|--------|-------|--------|--------|-----------------|
| **R0** | XGBoost | 0.701 | 0.827 | 0.608 | 0.127 | 0.114 | 75 | 158.3 |
| **R1** | XGBoost | 0.681 | 0.730 | 0.637 | 0.131 | 0.108 | 89 | 114.3 |
| **R2** | MLP | 0.646 | 0.535 | 0.814 | 0.191 | 0.151 | 155 | 128.1 |
| **R3** | Ensemble | 0.632 | 0.538 | 0.765 | 0.186 | 0.169 | 145 | 113.8 |
| **R4** | XGBoost (no ToM) | 0.701 | 0.827 | 0.608 | 0.127 | 0.114 | 75 | 158.7 |

**Key observations**:

1. **R0 vs R1 (conservative vs aggressive)**: R1 alerts earlier (mean round 114 vs 158) and achieves higher recall (0.637 vs 0.608) at the cost of lower precision (0.730 vs 0.827). R0 has better ERDE5 but R1 has better ERDE50, reflecting their respective optimization targets.

2. **R0 vs R2 (XGBoost vs MLP)**: The MLP classifier is much more aggressive — it fires 155 alerts (vs 75), achieving high recall (0.814) but at substantial precision cost (0.535). This results in significantly worse ERDE scores due to the many false positives.

3. **R0 vs R3 (XGBoost vs Ensemble)**: The stacked ensemble is similarly aggressive to the MLP (145 alerts), with intermediate recall (0.765) but poor precision (0.538). The meta-learner does not appear to effectively moderate the MLP's tendency toward false positives.

4. **R0 vs R4 (ToM ablation)**: In simulated evaluation, R0 and R4 produce nearly identical results (same F1=0.701, same alerts=75, mean alert round 158.3 vs 158.7). This suggests that the ToM features have minimal impact on the XGBoost's decision boundary for users near the classification threshold in this evaluation setup. The cross-validation improvement from ToM (+0.2pp) is absorbed into the classifier when trained on the full dataset.

5. **Speed–accuracy trade-off**: R1's aggressive ERDE5 decay (o=5) makes it alert ~44 rounds earlier on average than R0 (114 vs 158), trading 2pp precision for 3pp recall. This is the intended design — R1 optimizes for early detection when ERDE5 is the primary metric.

### 18.5 Ablation: ToM Feature Contribution

The ablation is evaluated at two levels:

**Cross-validation level** (comparing training with and without ToM):

| Classifier | F1 (no ToM) | F1 (with ToM) | Δ F1 | ERDE50 (no ToM) | ERDE50 (with ToM) | Δ ERDE50 |
|------------|-------------|---------------|------|-----------------|-------------------|----------|
| XGBoost | 0.722 | 0.724 | +0.002 | 0.064 | 0.064 | 0.000 |
| MLP | 0.686 | 0.704 | +0.018 | 0.081 | 0.070 | −0.011 |
| Ensemble | 0.693 | 0.726 | +0.033 | 0.078 | 0.069 | −0.009 |

**Run-level** (R0 full features vs R4 no_tom mask, same XGBoost classifier):

| Metric | R0 (full) | R4 (no ToM) | Δ |
|--------|-----------|-------------|---|
| F1 | 0.701 | 0.701 | 0.000 |
| Precision | 0.827 | 0.827 | 0.000 |
| Recall | 0.608 | 0.608 | 0.000 |
| ERDE5 | 0.127 | 0.127 | 0.000 |
| ERDE50 | 0.114 | 0.114 | 0.000 |
| Mean alert round | 158.3 | 158.7 | +0.4 |

**Interpretation**: ToM features improve cross-validation F1 (especially for ensemble: +3.3pp), but the improvement does not manifest in the full-dataset simulated evaluation for XGBoost. This suggests that ToM features help disambiguate borderline users during cross-validation (where training data is reduced by 20%), but their contribution is subsumed by other features when the full training set is available. The MLP and ensemble classifiers benefit more consistently, likely because they can model non-linear interactions between ToM features and other feature groups.

### 18.6 Ablation: Classifier Architecture Comparison

Holding the decision policy constant (θ_init=0.85, θ_floor=0.45, ERDE o=50, t_con=2):

| Classifier | CV F1 | Eval F1 | CV ERDE5 | Eval ERDE5 | Alerts | Behavior |
|------------|-------|---------|----------|------------|--------|----------|
| **XGBoost** | 0.722 | 0.701 | 0.070 | 0.127 | 75 | Conservative, high-precision |
| **MLP** | 0.686 | 0.646 | 0.086 | 0.191 | 155 | Aggressive, high-recall |
| **Ensemble** | 0.693 | 0.632 | 0.085 | 0.186 | 145 | Aggressive, moderate-recall |

XGBoost is the strongest classifier for this task. Its built-in feature selection (via gradient boosting) handles the high-dimensional (~2341d) sparse feature space more effectively than the MLP or ensemble.

### 18.7 Ablation: Decision Policy Parameters

Comparing R0 (conservative: θ_init=0.85, o=50, t_con=2) vs R1 (aggressive: θ_init=0.70, o=5, t_con=1), both using XGBoost:

| Parameter | R0 | R1 | Effect |
|-----------|----|----|--------|
| θ_init | 0.85 | 0.70 | R1 accepts lower confidence for initial alert |
| θ_floor | 0.45 | 0.35 | R1 decays to a lower minimum threshold |
| ERDE o | 50 | 5 | R1's threshold decays much faster (penalty ramps at round 5 vs 50) |
| t_con | 2 | 1 | R1 requires only 1 round above threshold (no confirmation) |
| **Alerts** | 75 | 89 | R1 alerts 19% more users |
| **Mean alert round** | 158.3 | 114.3 | R1 alerts **44 rounds earlier** on average |
| **Precision** | 0.827 | 0.730 | R1 trades 10pp precision |
| **Recall** | 0.608 | 0.637 | R1 gains 3pp recall |
| **ERDE50** | 0.114 | 0.108 | R1 has slightly better ERDE50 (earlier alerts help) |

### 18.8 Data Sources Summary

| Data / Resource | Used For | Source |
|----------------|----------|--------|
| eRisk 2025 Task 2 dataset (909 users) | Training features, labels, cross-validation | CLEF eRisk 2025 competition |
| `all-mpnet-base-v2` | Sentence embeddings (768d), BERTopic backbone | Reimers & Gurevych, 2019; HuggingFace |
| `all-MiniLM-L12-v2` | Sentence embeddings (384d) | Wang et al., 2020; HuggingFace |
| `all-distilroberta-v1` | Sentence embeddings (768d) | Reimers & Gurevych, 2019; HuggingFace |
| `j-hartmann/emotion-english-distilroberta-base` | 8-class emotion classification | Hartmann, 2022; HuggingFace |
| VADER sentiment lexicon | Reply sentiment analysis | Hutto & Gilbert, 2014 |
| BDI-II symptom descriptions (21 items) | Symptom reference embeddings, activation scoring | Beck et al., 1996 |
| DSM-5 criteria for Major Depressive Episode | Clinical grounding of BDI-II to DSM mapping | APA, 2013 |
| Llama 3.3 70B | Theory of Mind dual mentalizing prompts | Meta, 2024; via Ollama |
| BERTopic + UMAP + HDBSCAN | Dynamic topic modeling | Grootendorst, 2022 |
| XGBoost | Gradient boosted classification | Chen & Guestrin, 2016 |
| scikit-learn (SVM, PCA, LedoitWolf, LogisticRegression) | Classification, dimensionality reduction, covariance estimation | Pedregosa et al., 2011 |
| PyTorch | MLP classifier implementation | Paszke et al., 2019 |

### 18.9 Trained Artifacts

All trained models and intermediate artifacts are stored under `runs/task2/train/`:

| Artifact | Size | Description |
|----------|------|-------------|
| `classifier_xgboost.pkl` | ~350KB | Trained XGBoost model with StandardScaler |
| `classifier_neural_net.pkl` | ~2.5MB | Trained 2-layer MLP with BatchNorm |
| `classifier_ensemble.pkl` | ~8.7MB | Stacked ensemble (XGB+MLP+SVM + LogReg meta-learner) |
| `mahalanobis.pkl` | ~999KB | PCA projections + LedoitWolf covariances for control/depressed |
| `bertopic_model/` | ~2.5MB | Fitted BERTopic with vocabulary and topic representations |
| `symptom_references.npy` | ~162KB | 21×1920 symptom reference embedding matrix |
| `tom_features_checkpoint.pkl` | ~388KB | Cached LLM responses for ToM feature extraction |
| `features.npz` | ~9.1MB | Full 909×~2341 feature matrix with labels |
| `eval_embeddings_cache/` | ~3.8GB | Pre-computed sentence transformer embeddings per user |
| `training_results.json` | — | CV metrics (without ToM) |
| `training_results_with_tom.json` | — | CV metrics (with ToM) |
| `eval_results.json` | — | Full simulated evaluation metrics per run |

---

## Appendix A — Parameter Summary

A machine-readable copy of every numeric setting verified against the source is kept in [docs/task2_parameters.json](task2_parameters.json). The table below pins the values most likely to be quoted in a paper / system description; each row cites the canonical source file.

### A.1 Layer 1 — Target-user text features

| Block | Parameter | Value | Source |
| --- | --- | --- | --- |
| Embedding ensemble | Models | mpnet-base-v2 (768d) + MiniLM-L12-v2 (384d) + distilroberta-v1 (768d) → concat 1920d | [config.py:14-23](../src/erisk_task2/config.py#L14-L23) |
| Embedding ensemble | Batch size | 64 | [config.py:23](../src/erisk_task2/config.py#L23) |
| Running mean | Decay λ | 0.95 | [config/task2.yaml:12](../config/task2.yaml#L12) |
| BDI symptoms | n_symptoms / variant | 21 / "C" (clinical-formulation reference texts) | [layer1.py:48-71](../src/erisk_task2/features/layer1.py#L48-L71) |
| BDI distributional stats | Per symptom | mean, var, skew, kurtosis, Q25, Q50, Q75 (7 stats) → 147d | [layer1.py:207-239](../src/erisk_task2/features/layer1.py#L207-L239) |
| Lexical block | Total dim | 4 (one ratio per category) | [layer1.py:178-204](../src/erisk_task2/features/layer1.py#L178-L204) |
| Lexical wordlist sizes | first-person / neg-emotion / absolutist / cognitive | 9 / 36 / 15 / 18 | [layer1.py:26-46](../src/erisk_task2/features/layer1.py#L26-L46) |

### A.2 Layer 2 — Conversational context

| Block | Parameter | Value | Source |
| --- | --- | --- | --- |
| Reply sentiment | Dim | 3 (mean compound, std, linear-trend slope) | [feature_assembler.py:91-96](../src/erisk_task2/classification/feature_assembler.py#L91-L96) |
| Concern detection | Phrase count | **42** (outline previously said "~40") | [layer2.py:22-35](../src/erisk_task2/features/layer2.py#L22-L35) |
| Concern detection | Feature dim | 1 (cross-round flag ratio) | [feature_assembler.py:98-102](../src/erisk_task2/classification/feature_assembler.py#L98-L102) |
| Conv. position | Dim **in feature vector** | 3 (`is_author_ratio`, `target_silent_ratio`, `reply_depth_mean`) | [feature_assembler.py:104-110](../src/erisk_task2/classification/feature_assembler.py#L104-L110) |
| Conv. position | Computed but discarded | `text_volume_mean`, `text_volume_trend` | [layer2.py:106-130](../src/erisk_task2/features/layer2.py#L106-L130) |
| Thread topic similarity | Dim | 21 (title × 21 BDI references, averaged across rounds) | [layer2.py:133-150](../src/erisk_task2/features/layer2.py#L133-L150) |
| **Layer 2 total** | — | **28d** (3 + 1 + 3 + 21) | — |

### A.3 Layer 3 — Emotion + Topics

| Block | Parameter | Value | Source |
| --- | --- | --- | --- |
| Emotion | Model | `j-hartmann/emotion-english-distilroberta-base` | [config/task2.yaml:30](../config/task2.yaml#L30) |
| Emotion | Min words for classification | 10 (else uniform 1/8 fallback) | [layer3.py:47-83](../src/erisk_task2/features/layer3.py#L47-L83) |
| Emotion | Feature dim | 9 (8 mean-dist + 1 Shannon entropy) | [feature_assembler.py:118-122](../src/erisk_task2/classification/feature_assembler.py#L118-L122) |
| BERTopic | n_topics | 40 | [config/task2.yaml:22](../config/task2.yaml#L22) |
| BERTopic | UMAP n_neighbors / n_components | 15 / 5 | [config/task2.yaml:23-24](../config/task2.yaml#L23-L24) |
| BERTopic | HDBSCAN min_cluster_size / min_samples | 50 / 10 | [config/task2.yaml:25-26](../config/task2.yaml#L25-L26) |
| BERTopic | Rolling buffer | 5 last target texts | [config/task2.yaml:27](../config/task2.yaml#L27) |
| BERTopic | Test-time topic assignment | `model.transform()` → **soft (probabilistic)** for the 40d distribution; **hard top-1** for the depression-proportion slot | [layer3.py:195-227](../src/erisk_task2/features/layer3.py#L195-L227) |
| BERTopic | Feature dim | 41 = 40d distribution + 1d **mean depression-proportion** (not entropy) | [pipeline.py:258-275](../src/erisk_task2/pipeline.py#L258-L275) |

### A.4 Distributional distances

| Block | Parameter | Value | Source |
| --- | --- | --- | --- |
| Wasserstein | short_window / medium_window | 5 / 25 | [config/task2.yaml:34-35](../config/task2.yaml#L34-L35) |
| Sliced Wasserstein | n_projections | 50 (random unit directions in R^1920) | [config/task2.yaml:36](../config/task2.yaml#L36) |
| Wasserstein | Feature dim | 72 (63 symptom + 3 emotion + 3 embedding + 3 topic) | [wasserstein.py:142-196](../src/erisk_task2/distances/wasserstein.py#L142-L196) |
| Mahalanobis | n_pca_components | 50 (effective = min(50, n_features, n_control − 1)) | [config/task2.yaml:39](../config/task2.yaml#L39) |
| Mahalanobis | Regularization | `LedoitWolf` shrinkage covariance | [mahalanobis.py:51, 58](../src/erisk_task2/distances/mahalanobis.py#L51) |
| Mahalanobis | Feature dim | 3 (D_M_control, D_M_relative, D_M_depressed) | [mahalanobis.py:68-91](../src/erisk_task2/distances/mahalanobis.py#L68-L91) |

### A.5 ToM (LLM) operationalisation

| Block | Parameter | Value | Source |
| --- | --- | --- | --- |
| Backend (training features) | Provider / model | HuggingFace InferenceClient / `meta-llama/Llama-3.3-70B-Instruct` | [hf_client.py:23](../src/erisk_task2/tom/hf_client.py#L23) |
| HF max output tokens | `max_tokens` | 2048 | [config.py:85](../src/erisk_task2/config.py#L85) |
| Ollama context | `num_ctx` | 8192 | [config/task2.yaml:50](../config/task2.yaml#L50) |
| Ollama max output tokens | `num_predict` | **not set** → model default (uncapped from the client side) | [llm_client.py:50-59](../src/erisk_task2/tom/llm_client.py#L50-L59) |
| Temperature | All prompts, all attempts | 0.1 | [config/task2.yaml:52](../config/task2.yaml#L52) |
| Retry policy | `max_retries` / back-off | 3 / `2 ** attempt` seconds | [hf_client.py:64-83](../src/erisk_task2/tom/hf_client.py#L64-L83), [llm_client.py:64-82](../src/erisk_task2/tom/llm_client.py#L64-L82) |
| Retry temperature override | — | **none** (no temperature=0 fallback) | same |
| Batching | Across users / rounds | sequential, no batching | [tom_module.py:155-204](../src/erisk_task2/tom/tom_module.py#L155-L204) |
| Thread format | Max tokens | 2000 (~8000 chars) | [config/task2.yaml:67](../config/task2.yaml#L67) |
| ToM feature dim | — | 47 | [tom_module.py:207-251](../src/erisk_task2/tom/tom_module.py#L207-L251) |

### A.6 Thompson Sampling

| Parameter | Value | Source |
| --- | --- | --- |
| n_symptoms (arms) | 21 | [thompson.py:16](../src/erisk_task2/bandits/thompson.py#L16) |
| Activation threshold τ_active | **0.3** | [config/task2.yaml:19](../config/task2.yaml#L19) |
| Prior | **Beta(1, 1)** (uniform) | [thompson.py:20-25](../src/erisk_task2/bandits/thompson.py#L20-L25) |
| Feature dim | 25 (1 weighted score + 1 entropy + 1 active count + 21 weights + 1 uncertainty) | [thompson.py:68-105](../src/erisk_task2/bandits/thompson.py#L68-L105) |

### A.7 Classifiers

| Classifier | Hyperparameter | Value | Source |
| --- | --- | --- | --- |
| XGBoost | max_depth | 6 | [config/task2.yaml:72](../config/task2.yaml#L72) |
| XGBoost | n_estimators | 300 | [config/task2.yaml:73](../config/task2.yaml#L73) |
| XGBoost | learning_rate | 0.1 | [config/task2.yaml:74](../config/task2.yaml#L74) |
| XGBoost | `scale_pos_weight` | **auto = n_neg / n_pos** at fit time (~7.9 on training data) | [classifiers.py:66-71](../src/erisk_task2/classification/classifiers.py#L66-L71) |
| XGBoost | eval_metric / random_state | `logloss` / 42 | [classifiers.py:73-79](../src/erisk_task2/classification/classifiers.py#L73-L79) |
| MLP | Hidden widths | (256, 64) | [classifiers.py:104](../src/erisk_task2/classification/classifiers.py#L104) |
| MLP | Dropout | 0.3 | [classifiers.py:104](../src/erisk_task2/classification/classifiers.py#L104) |
| MLP | Optimizer / lr | Adam / 1e-3 | [classifiers.py:137](../src/erisk_task2/classification/classifiers.py#L137) |
| MLP | Batch size / max epochs / patience | 64 / 100 / 10 | [classifiers.py:107, 143, 147](../src/erisk_task2/classification/classifiers.py#L107) |
| MLP | Loss | `BCEWithLogitsLoss(pos_weight = n_neg / n_pos)` | [classifiers.py:121-138](../src/erisk_task2/classification/classifiers.py#L121-L138) |
| SVM | Kernel / γ / C | RBF / `'scale'` / 1.0 (sklearn defaults; not tuned) | [classifiers.py:223-227](../src/erisk_task2/classification/classifiers.py#L223-L227) |
| SVM | class_weight | balanced | same |
| Stacking meta | Model / weighting | LogisticRegression / `class_weight='balanced'` | [classifiers.py:272](../src/erisk_task2/classification/classifiers.py#L272) |
| Stacking meta | OOF strategy | 5-fold StratifiedKFold from {XGB, MLP, SVM} | [classifiers.py:262](../src/erisk_task2/classification/classifiers.py#L262) |

### A.8 Per-run decision-policy parameters (source of truth: [models.py:206-213](../src/erisk_task2/models.py#L206-L213))

| Run | Classifier | θ_init | θ_floor | ERDE o | t_con | Feature mask |
| --- | --- | --- | --- | --- | --- | --- |
| **R0** | XGBoost | 0.85 | 0.45 | 50 | 2 | — |
| **R1** | XGBoost | 0.70 | 0.35 | 5 | 1 | — |
| **R2** | MLP | 0.85 | 0.45 | 50 | 2 | — |
| **R3** | Ensemble | 0.80 | 0.40 | 30 | 2 | — |
| **R4** | XGBoost | 0.85 | 0.45 | 50 | 2 | `["no_tom"]` |

Threshold schedule: `θ(k) = θ_init − (θ_init − θ_floor) · σ(k − o)` ([policy.py:30-41](../src/erisk_task2/decision/policy.py#L30-L41)). Alert latches once `consecutive_positives ≥ t_con` ([policy.py:44-85](../src/erisk_task2/decision/policy.py#L44-L85)).
