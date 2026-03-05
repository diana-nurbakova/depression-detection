# Dataset Exploratory Analysis

> Generated: 2026-03-03

## 1. BDI-Sen-2.0

**Location:** `data/BDI-Sen/full_dataset/`

### 1.1 Overview

BDI-Sen-2.0 contains Reddit sentences annotated for the **21 symptoms of the Beck Depression Inventory-II (BDI-II)**. Each sentence is judged for relevance (`label`: 0/1) to a specific BDI-II symptom at a given severity level (0–3).

### 1.2 Files

| File | Records | Description |
|------|---------|-------------|
| `bdi_majority_vote.jsonl` | 5,003 | Flat: one row per (sentence, symptom) pair |
| `bdi_unified.jsonl` | 2,529 | Grouped: one row per sentence, annotations nested |
| `splits/train.jsonl` | 1,766 | Training set (BDI sentences only) |
| `splits/train-with-control.jsonl` | 2,363 | Training + 597 control sentences |
| `splits/val.jsonl` | 262 | Validation set |
| `splits/val-with-control.jsonl` | 347 | Validation + 85 control sentences |
| `splits/test.jsonl` | 501 | Test set |
| `splits/test-with-control.jsonl` | 1,355 | Test + 854 control sentences |

Split ratio: ~70/10/20 (train/val/test). All 21 symptoms appear in every split.

### 1.3 Schema

**Flat format** (`bdi_majority_vote.jsonl`):
```json
{"sentence": "I feel like a burden.", "symptom": "Feelings_of_worthlessness", "severity": 2, "label": 1}
```

**Grouped format** (`bdi_unified.jsonl`):
```json
{"sentence": "I feel like a burden.", "annotations": [
  {"symptom": "Feelings_of_worthlessness", "severity": 2, "label": 1},
  {"symptom": "Self-dislike", "severity": 1, "label": 0}
]}
```

Both files contain the same data in different shapes.

### 1.4 Key Statistics

| Metric | Value |
|--------|-------|
| **Total unique sentences** | **2,529** |
| **Total (sentence, symptom) annotations** | **5,003** |
| **Positive annotations (label=1)** | 852 (17.0%) |
| **Negative annotations (label=0)** | 4,151 (83.0%) |
| **Control sentences (across splits)** | 1,536 |
| Mean annotations per sentence | 1.98 |
| Median sentence length | 8 words |
| Max sentence length | 61 words |

### 1.5 Severity Distribution

| Severity | Count | % |
|----------|-------|---|
| 0 (none) | 1,356 | 27.1% |
| 1 (mild) | 1,785 | 35.7% |
| 2 (moderate) | 976 | 19.5% |
| 3 (severe) | 671 | 13.4% |
| null | 215 | 4.3% |

Distribution skews toward mild (severity 1), with progressively fewer annotations at higher severity.

### 1.6 Per-Symptom Breakdown

| Symptom | Annotations | Positive (label=1) | Positive rate |
|---------|-------------|---------------------|---------------|
| Loss_of_Pleasure | 739 | 140 | 18.9% |
| Sadness | 644 | 152 | 23.6% |
| Self-dislike | 505 | 60 | 11.9% |
| Guilty_feelings | 317 | 27 | 8.5% |
| Irritability | 305 | 55 | 18.0% |
| Sense_of_failure | 299 | 62 | 20.7% |
| Sense_of_punishment | 287 | 18 | 6.3% |
| Crying | 278 | 34 | 12.2% |
| Pessimism | 274 | 72 | 26.3% |
| Suicidal_ideas | 230 | 44 | 19.1% |
| Agitation | 161 | 26 | 16.1% |
| Feelings_of_worthlessness | 147 | 28 | 19.0% |
| Social_withdrawal | 143 | 25 | 17.5% |
| Concentration_difficulty | 141 | 13 | 9.2% |
| Tiredness_or_fatigue | 129 | 28 | 21.7% |
| Loss_of_energy | 124 | 16 | 12.9% |
| Self-incrimination | 118 | 10 | 8.5% |
| Indecision | 80 | 22 | 27.5% |
| Change_of_sleep | 31 | 9 | 29.0% |
| Changes_in_appetite | 27 | 8 | 29.6% |
| Loss_of_interest_in_sex | 24 | 3 | 12.5% |

**Most represented:** Loss_of_Pleasure (739), Sadness (644), Self-dislike (505).
**Least represented:** Loss_of_interest_in_sex (24), Changes_in_appetite (27), Change_of_sleep (31).
**Highest positive rate:** Changes_in_appetite (29.6%), Change_of_sleep (29.0%), Indecision (27.5%).
**Lowest positive rate:** Sense_of_punishment (6.3%), Guilty_feelings / Self-incrimination (8.5%).

---

## 2. eRisk-2025

**Location:** `data/eRisk-2025/eRisk25-datasets/`

Two sub-tasks:

### 2.1 Task 1 — Depression Symptom Ranking

**Location:** `t1-depression-symptom-ranking/`

#### Overview

An information-retrieval task: rank Reddit sentences by relevance to each of the 21 BDI-II depression symptoms. Structurally identical to the eRisk-2026 Task 3 ADHD dataset (same TREC format, same ranking paradigm).

#### Data Format

Each `.trec` file contains all sentences from one Reddit user in TREC XML:
```xml
<DOC>
  <DOCNO>mR9stN_528_0</DOCNO>
  <PRE>Previous sentence context</PRE>
  <TEXT>The target sentence to rank</TEXT>
  <POST>Following sentence context</POST>
</DOC>
```
DOCNO format: `{userID}_{postIndex}_{sentenceIndex}`

#### Corpus Statistics

| Metric | Value |
|--------|-------|
| **.trec files (users)** | **6,300** |
| **Total sentences** | **17,553,441** |
| Mean sentences/user | 2,786 |
| Median sentences/user | 1,491 |
| Min / Max | 1 / 73,722 |

#### Annotations (qrels)

Two annotation variants with **11,042 judgments** each across all 21 queries:

| Variant | Relevant (True) | Rate |
|---------|-----------------|------|
| **Consensus** (all annotators agree) | 3,410 | 30.9% |
| **Majority** (majority vote) | 6,117 | 55.4% |
| Agreement between variants | 8,335 (75.5%) | — |

2,707 borderline sentences where majority found relevant but consensus did not.

#### Per-Query Relevance Rates

| Query | Symptom | Consensus % | Majority % |
|-------|---------|-------------|------------|
| 1 | Sadness | 28.7% | 50.9% |
| 6 | Punishment Feelings | 6.0% | 20.1% |
| 9 | Suicidal Thoughts | 58.0% | 72.9% |
| 13 | Indecisiveness | 8.6% | 23.8% |
| 16 | Sleep Changes | 48.2% | 71.0% |
| 19 | Concentration | 38.8% | 63.3% |
| 21 | Interest in Sex | 45.1% | 66.0% |

Hardest for annotators to agree on: Punishment Feelings (6.0% consensus). Easiest: Suicidal Thoughts (58.0%).

#### Evaluation Metrics

Average Precision (AP), R-Precision, Precision@10, nDCG@1000.

---

### 2.2 Task 2 — Early Contextualized Depression Detection

**Location:** `t2-early-contextualized-depression/`

#### Overview

Binary classification: determine whether a Reddit user suffers from depression based on their full posting history, delivered in chronological rounds (early-detection scenario).

#### Data Format

Each `.json` file is a chronological array of submission+comments groups preserving full conversation structure:
```json
[{
  "submission": {"user_id": "subject_X", "target": true, "title": "...", "body": "...", "created_utc": "..."},
  "comments": [{"user_id": "...", "target": false, "body": "...", "created_utc": "..."}]
}]
```

#### Statistics

| Metric | Value |
|--------|-------|
| **Total subjects** | **909** |
| Control (label=0) | 807 (88.8%) |
| Depressed (label=1) | 102 (11.2%) |
| Total submissions | 278,596 |
| Total comments | 13,025,230 |
| **Total writings** | **13,303,826** |

Depressed users tend to have more posts on average (~985 target writings) vs control users (~433).

---

### 2.3 ADHD Content in eRisk-2025

**There are no ADHD-specific annotations or tasks in eRisk-2025.** Both tasks focus exclusively on BDI-II depression symptoms.

However, some observations relevant to cross-condition transfer:

- User posts in T2 naturally mention ADHD (1,091+ mentions in a subsample of 50 files), since Reddit users discuss multiple mental health conditions.
- The T1 task uses the same TREC format and ranking paradigm as eRisk-2026 Task 3 (ADHD), making T1 directly usable for **pre-training the bi-encoder** and **calibrating the LLM scoring pipeline** before transfer to ADHD.
- Several BDI-II symptoms overlap with ASRS (ADHD) symptom clusters:

| BDI-II Symptom | Overlapping ASRS Items |
|----------------|----------------------|
| Concentration Difficulty (Q19) | ASRS 8, 9, 10, 11 |
| Agitation (Q11) | ASRS 5, 6, 12, 13 |
| Loss of Energy (Q15) | ASRS 4, 7 |
| Sleep Changes (Q16) | ASRS 7 |
| Indecisiveness (Q13) | ASRS 1, 2 |

---

## 3. Cross-Dataset Summary

| | BDI-Sen-2.0 | eRisk-2025 T1 | eRisk-2025 T2 |
|---|---|---|---|
| **Task** | Sentence-symptom classification | Sentence ranking per symptom | User-level depression detection |
| **Format** | JSONL | TREC XML + qrels CSV | JSON + ground truth TXT |
| **Symptom inventory** | BDI-II (21 symptoms) | BDI-II (21 symptoms) | Binary (depression/control) |
| **Unique sentences/users** | 2,529 sentences | 17.5M sentences / 6,300 users | 13.3M writings / 909 users |
| **Annotated pairs** | 5,003 | 11,042 | 909 labels |
| **Severity granularity** | 0–3 per annotation | Binary relevance | Binary label |
| **ADHD annotations** | No | No | No |
