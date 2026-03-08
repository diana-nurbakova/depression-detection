# eRisk Task 2: Full Pipeline on Google Colab

## 1. Setup

### 1.1 Upload the project

```python
# Option A: Clone from Git
!git clone https://github.com/<your-repo>/depression-detection.git
%cd depression-detection

# Option B: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
!cp -r "/content/drive/MyDrive/depression-detection" /content/depression-detection
%cd /content/depression-detection
```

### 1.2 Check GPU

```python
!nvidia-smi
import torch
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
```

### 1.3 Install

```bash
!pip install -e . 2>&1 | tail -5
```

If PyTorch conflicts:
```bash
!pip install --upgrade torch torchvision
!pip install -e .
```

### 1.4 Install HuggingFace Inference Client (for ToM Options B/C)

```bash
!pip install huggingface_hub
```

### 1.5 Verify imports

```python
from erisk_task2.config import load_config
from erisk_task2.pipeline import train_pipeline
from erisk_task2.features.layer1 import EmbeddingEncoder
from erisk_task2.features.layer3 import EmotionClassifier, TopicModeler
from erisk_task2.tom.tom_module import ToMModule, extract_tom_features
print("All imports OK")
```

### 1.6 Restart runtime after install

```python
import IPython
IPython.Application.instance().kernel.do_shutdown(True)
```

---

## 2. Training Data

```python
# Point to your data (adjust paths as needed)
DATA_DIR = "/content/drive/MyDrive/eRisk-data/all_combined"
LABELS = "/content/drive/MyDrive/eRisk-data/shuffled_ground_truth_labels.txt"

# Verify
import os
n_files = len([f for f in os.listdir(DATA_DIR) if f.endswith('.json')])
print(f"Found {n_files} user JSON files")  # expect 909

with open(LABELS) as f:
    lines = f.readlines()
print(f"Found {len(lines)} label entries")  # expect 909
```

---

## 3. Configuration

### 3.1 Write Colab config

```python
config_content = """
embedding:
  models:
    - "all-mpnet-base-v2"
    - "all-MiniLM-L12-v2"
    - "all-distilroberta-v1"
  decay_lambda: 0.95
  device: "cuda"
  batch_size: 128

symptom:
  variant: "C"
  use_depresym_embeddings: true
  activation_threshold: 0.3

bertopic:
  n_topics: 40
  n_neighbors: 15
  n_components: 5
  min_cluster_size: 50
  min_samples: 10
  rolling_buffer_size: 5

emotion:
  model: "j-hartmann/emotion-english-distilroberta-base"
  min_words: 10

wasserstein:
  short_window: 5
  medium_window: 25
  n_projections: 50

mahalanobis:
  n_pca_components: 50
  regularization: "ledoit_wolf"

tom:
  enabled: true
  method: "option_a"
  chained: false

ollama:
  base_url: "http://localhost:11434"
  model: "meta-llama/Llama-3.3-70B-Instruct"
  num_ctx: 8192
  keep_alive: "24h"
  temperature: 0.1
  timeout_seconds: 120

server:
  base_url: "https://erisk.irlab.org/challenge-t2"
  max_retries: 5
  initial_delay: 2.0
  backoff_factor: 2.0
  timeout: 60

logging:
  output_dir: "/content/drive/MyDrive/erisk-task2-train"
  log_level: "INFO"

thread_format:
  max_tokens: 2000
  truncate_length: 100

cv_folds: 5
xgboost_params:
  max_depth: 6
  n_estimators: 300
  learning_rate: 0.1
"""

with open("config/task2_colab.yaml", "w") as f:
    f.write(config_content)
print("Config written")
```

---

## 4. Training (Full Pipeline)

### 4.1 Standard training (ToM Option A — no LLM needed)

This is the recommended first run. Uses embedding-based ToM which is fast and requires no external API.

```python
import logging, sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

from erisk_task2.config import load_config
from erisk_task2.pipeline import train_pipeline

cfg = load_config("config/task2_colab.yaml")
cfg.training_data_dir = DATA_DIR
cfg.labels_path = LABELS
cfg.logging.output_dir = "/content/drive/MyDrive/erisk-task2-train"

# ToM Option A (embedding-based, no LLM)
cfg.tom.enabled = True
cfg.tom.method = "option_a"

train_pipeline(cfg)
```

**What happens:**
1. Loads 909 users from JSON (~30s)
2. Loads 3 sentence transformers + emotion classifier (~2 min)
3. Per user (x909): embeds target texts, computes symptom scores, VADER sentiment, concern detection, emotion classification, ToM Option A embeddings, Thompson sampling (~60-90 min on T4 GPU)
4. Fits BERTopic on all target texts (~5 min)
5. Assembles ~2341d feature vectors
6. Fits Mahalanobis scorer on control users
7. 5-fold CV for XGBoost, MLP, Ensemble (~10 min)
8. Saves everything to Google Drive

**Checkpoints:** Saved every 50 users to `profiles_checkpoint.pkl`. If Colab disconnects, re-run the same cell — it resumes automatically.

---

## 5. Theory of Mind Options

The pipeline supports 3 ToM methods. Option A is always computed. Options B and C require an LLM.

### 5.1 Option A: Embedding-based (default, no LLM)

Computes:
- Self-view embedding = mean of target user texts
- Observer-view embedding = mean of other comments
- Insight gap = cosine distance between the two

Features (47d vector, 3 active signals):
- `[42]` self embedding norm
- `[43]` observer embedding norm
- `[44]` cosine distance (insight gap)

Already wired into the pipeline. No extra setup needed.

### 5.2 Option B: Response Category Classification (requires LLM)

Classifies each reply to the target user into 7 categories:
CONCERN, ADVICE, EMOTIONAL_SUPPORT, NORMALIZATION, SHARED_EXPERIENCE, PRACTICAL_SUPPORT, CASUAL

Returns a 7d distribution over categories.

### 5.3 Option C: Full LLM Mentalizing (requires LLM)

Two-stage assessment:
- **Prompt 1 (Self-view)**: Analyzes target user texts for BDI-II symptom indicators. Returns per-symptom scores (0-3), depression probability (0-1), overall impression.
- **Prompt 2a/2b (Observer-view)**: Analyzes how the community perceives the target user. Returns perceived symptoms, concern level (0-3), community response type.

Features (47d, all active):
- `[0:21]` self-view symptom scores (normalized 0-1)
- `[21:42]` observer-view symptom scores (normalized 0-1)
- `[42]` self depression probability
- `[43]` observer depression probability
- `[44]` insight gap (mean absolute difference between self and observer scores)
- `[45]` observer concern level (normalized 0-1)
- `[46]` community response type (encoded 0-1)

---

## 6. Using HuggingFace Inference API for ToM Options B/C

Since Colab doesn't have Ollama, you can use HuggingFace's Inference API with your Pro tier.

### 6.1 Create HF-compatible LLM client

```python
# Run this cell to create the HF client wrapper

import json
import logging
import time
from typing import Optional
from huggingface_hub import InferenceClient

logger = logging.getLogger(__name__)


class HuggingFaceClient:
    """Drop-in replacement for OllamaClient using HuggingFace Inference API."""

    def __init__(
        self,
        model: str = "meta-llama/Llama-3.3-70B-Instruct",
        token: str = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        timeout: int = 120,
        max_retries: int = 3,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.client = InferenceClient(model=model, token=token, timeout=timeout)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
    ) -> tuple[str, float]:
        """Generate text using HF Inference API."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        for attempt in range(self.max_retries):
            try:
                start = time.monotonic()
                response = self.client.chat_completion(
                    messages=messages,
                    temperature=temperature or self.temperature,
                    max_tokens=self.max_tokens,
                )
                elapsed = time.monotonic() - start
                text = response.choices[0].message.content
                return text, elapsed
            except Exception as e:
                logger.warning("HF request failed (attempt %d): %s", attempt + 1, e)
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)

        logger.error("All %d HF attempts failed", self.max_retries)
        return "", 0.0

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
    ) -> tuple[Optional[dict], float]:
        """Generate and parse JSON response."""
        text, elapsed = self.generate(system_prompt, user_prompt, temperature)
        if not text:
            return None, elapsed

        # Try parsing JSON
        text = text.strip()
        try:
            return json.loads(text), elapsed
        except json.JSONDecodeError:
            pass

        # Try markdown code block
        if "```" in text:
            for part in text.split("```"):
                cleaned = part.strip()
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:].strip()
                try:
                    return json.loads(cleaned), elapsed
                except json.JSONDecodeError:
                    continue

        # Try finding JSON boundaries
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end + 1]), elapsed
            except json.JSONDecodeError:
                pass

        logger.warning("Failed to parse JSON: %s", text[:200])
        return None, elapsed

    def is_available(self) -> bool:
        """Check if HF API is reachable."""
        try:
            self.client.chat_completion(
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1,
            )
            return True
        except Exception:
            return False


print("HuggingFaceClient defined")
```

### 6.2 Set your HuggingFace token

```python
import os
from google.colab import userdata

# Option A: Use Colab secrets (recommended)
# Go to the key icon in left sidebar > add HF_TOKEN
HF_TOKEN = userdata.get('HF_TOKEN')

# Option B: Set directly (less secure)
# HF_TOKEN = "hf_..."

os.environ["HF_TOKEN"] = HF_TOKEN
print("Token set")
```

### 6.3 Test the client

```python
client = HuggingFaceClient(
    model="meta-llama/Llama-3.3-70B-Instruct",
    token=HF_TOKEN,
    temperature=0.1,
)

# Quick test
response, elapsed = client.generate(
    "You are a helpful assistant. Respond with valid JSON only.",
    'Classify this text as positive or negative: "I feel terrible today." Output: {"sentiment": "..."}',
)
print(f"Response ({elapsed:.1f}s): {response}")
```

---

## 7. Running ToM Option C with HuggingFace

After the standard training completes (Section 4), you can enrich the ToM features with LLM-based assessment.

### 7.1 Post-training ToM enrichment

```python
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm

from erisk_task2.config import load_config
from erisk_task2.data.loader import load_training_data
from erisk_task2.formatting.thread_formatter import format_thread
from erisk_task2.tom.tom_module import ToMModule, extract_tom_features
from erisk_task2.tom.prompts import (
    PROMPT1_SYSTEM, PROMPT1_USER,
    PROMPT2A_SYSTEM, PROMPT2A_USER,
    get_symptom_variant,
)

cfg = load_config("config/task2_colab.yaml")
OUTPUT_DIR = Path("/content/drive/MyDrive/erisk-task2-train")

# Load training data
users, labels = load_training_data(DATA_DIR, LABELS)

# Create HF client
hf_client = HuggingFaceClient(
    model="meta-llama/Llama-3.3-70B-Instruct",
    token=HF_TOKEN,
    temperature=0.1,
    max_tokens=1024,
)

# Create ToM module with Option C
tom = ToMModule(
    method="option_c",
    chained=False,  # Use independent observer-view (Prompt 2a)
    symptom_variant="C",
    llm_client=hf_client,
)

# Process a subsample of users (LLM calls are slow — ~5s per prompt)
# For full 909 users x 2 prompts x ~100 threads = too many calls
# Strategy: assess 1 representative thread per user (the last one with target text)

user_ids = sorted(users.keys())
tom_features_dict = {}
tom_checkpoint = OUTPUT_DIR / "tom_features_checkpoint.pkl"

# Resume from checkpoint
if tom_checkpoint.exists():
    with open(tom_checkpoint, "rb") as f:
        tom_features_dict = pickle.load(f)
    print(f"Resumed: {len(tom_features_dict)} users already processed")

for i, uid in enumerate(tqdm(user_ids, desc="ToM Assessment")):
    if uid in tom_features_dict:
        continue

    threads = users[uid]

    # Find the last thread where target user has text
    representative_thread = None
    for t in reversed(threads):
        if t.target_texts:
            representative_thread = t
            break

    if representative_thread is None:
        tom_features_dict[uid] = np.zeros(47)
        continue

    # Create a minimal profile for assessment
    from erisk_task2.models import UserProfile
    profile = UserProfile(subject_id=uid)

    try:
        result = tom.assess(representative_thread, profile)
        features = extract_tom_features(result)
        tom_features_dict[uid] = features
    except Exception as e:
        print(f"ToM failed for {uid}: {e}")
        tom_features_dict[uid] = np.zeros(47)

    # Checkpoint every 25 users
    if (i + 1) % 25 == 0:
        with open(tom_checkpoint, "wb") as f:
            pickle.dump(tom_features_dict, f)
        print(f"  Checkpoint: {len(tom_features_dict)}/{len(user_ids)}")

# Final save
with open(tom_checkpoint, "wb") as f:
    pickle.dump(tom_features_dict, f)

# Save as numpy
np.savez_compressed(
    OUTPUT_DIR / "tom_features.npz",
    **{uid: feat for uid, feat in tom_features_dict.items()},
)
print(f"ToM features saved for {len(tom_features_dict)} users")
```

**Time estimate:** ~909 users x 2 prompts x 5s = ~2.5 hours with HF Pro tier.

### 7.2 Inject ToM features and retrain classifiers

```python
import json
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from erisk_task2.classification.classifiers import create_classifier
from erisk_task2.decision.policy import compute_erde, compute_f_latency
from erisk_task2.models import DEFAULT_RUNS

OUTPUT_DIR = Path("/content/drive/MyDrive/erisk-task2-train")

# Load existing features
data = np.load(OUTPUT_DIR / "features.npz", allow_pickle=True)
X = data["X"]
y = data["y"]
subject_ids = list(data["subject_ids"])

# Load ToM features
tom_data = np.load(OUTPUT_DIR / "tom_features.npz", allow_pickle=True)

# ToM feature offset in the feature vector:
# embedding(1920) + sym_max(21) + sym_mean(21) + sym_stats(147) + lex(4)
# + sent(3) + concern(1) + conv(3) + thread_topic(21) + emotion(9) + bertopic(41)
# + wasserstein(72) + mahalanobis(3) + combined_dist(2) = 2268
tom_offset = 1920 + 21 + 21 + 147 + 4 + 3 + 1 + 3 + 21 + 9 + 41 + 72 + 3 + 2

# Inject ToM features
for i, uid in enumerate(subject_ids):
    if uid in tom_data:
        tom_feat = tom_data[uid]
        X[i, tom_offset:tom_offset + 47] = tom_feat

X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
print(f"Injected ToM features. Matrix: {X.shape}")

# Retrain with CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
classifier_types = sorted(set(rc.classifier_type.value for rc in DEFAULT_RUNS))
results = {}

for clf_type in classifier_types:
    print(f"\n--- {clf_type} ---")
    fold_f1s = []
    all_val_decisions = {}
    all_val_alert_rounds = {}

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        clf = create_classifier(clf_type)
        clf.fit(X[train_idx], y[train_idx])
        probs = clf.predict_proba(X[val_idx])
        preds = (probs >= 0.5).astype(int)
        f1 = f1_score(y[val_idx], preds)
        fold_f1s.append(f1)

        for j, idx in enumerate(val_idx):
            uid = subject_ids[idx]
            all_val_decisions[uid] = int(preds[j])
            if preds[j] == 1:
                all_val_alert_rounds[uid] = max(1, int((1 - probs[j]) * 20))

        print(f"  Fold {fold+1}: F1={f1:.4f}")

    mean_f1 = float(np.mean(fold_f1s))
    # Compute ERDE (need labels dict)
    labels_dict = {subject_ids[i]: int(y[i]) for i in range(len(y))}
    eval_labels = {uid: labels_dict[uid] for uid in all_val_decisions}
    erde5 = compute_erde(all_val_decisions, all_val_alert_rounds, eval_labels, 5)
    erde50 = compute_erde(all_val_decisions, all_val_alert_rounds, eval_labels, 50)
    f_lat = compute_f_latency(all_val_alert_rounds, eval_labels, all_val_decisions, 50)

    results[clf_type] = {
        "mean_f1": mean_f1, "erde5": erde5, "erde50": erde50, "f_latency": f_lat,
    }
    print(f"  {clf_type}: F1={mean_f1:.4f} ERDE5={erde5:.4f} ERDE50={erde50:.4f}")

# Save updated results
with open(OUTPUT_DIR / "training_results_with_tom.json", "w") as f:
    json.dump(results, f, indent=2)

# Train final models on full data
for clf_type in classifier_types:
    clf = create_classifier(clf_type)
    clf.fit(X, y)
    clf.save(OUTPUT_DIR / f"classifier_{clf_type}.pkl")
    print(f"Saved {clf_type}")

# Save updated feature matrix
np.savez_compressed(OUTPUT_DIR / "features.npz", X=X, y=y, subject_ids=np.array(subject_ids))
print("Done. Models retrained with ToM Option C features.")
```

---

## 8. Evaluation

### 8.1 Compare ToM options

After running training with different ToM methods, compare results:

```python
import json
from pathlib import Path

OUTPUT_DIR = Path("/content/drive/MyDrive/erisk-task2-train")

# Load results
files = {
    "Option A (embedding)": "training_results.json",
    "Option C (LLM)": "training_results_with_tom.json",
}

for label, fname in files.items():
    path = OUTPUT_DIR / fname
    if not path.exists():
        print(f"{label}: not yet computed")
        continue
    with open(path) as f:
        results = json.load(f)
    print(f"\n{label}:")
    print(f"{'Classifier':<12} {'F1':>7} {'ERDE5':>8} {'ERDE50':>8} {'F_lat':>8}")
    print("-" * 45)
    for clf_type, m in results.items():
        print(f"{clf_type:<12} {m['mean_f1']:>7.4f} {m['erde5']:>8.4f} {m['erde50']:>8.4f} {m.get('f_latency', 0):>8.4f}")
```

### 8.2 Feature importance analysis

```python
import numpy as np
from erisk_task2.classification.classifiers import create_classifier

OUTPUT_DIR = Path("/content/drive/MyDrive/erisk-task2-train")

clf = create_classifier("xgboost")
clf.load(OUTPUT_DIR / "classifier_xgboost.pkl")

# Feature importance
importances = clf.model.feature_importances_
top_k = 30
top_indices = np.argsort(importances)[-top_k:][::-1]

# Feature names by offset
feature_ranges = [
    (0, 1920, "embedding"),
    (1920, 1941, "symptom_max"),
    (1941, 1962, "symptom_mean"),
    (1962, 2109, "symptom_stats"),
    (2109, 2113, "lexical"),
    (2113, 2116, "reply_sentiment"),
    (2116, 2117, "concern"),
    (2117, 2120, "conv_position"),
    (2120, 2141, "thread_topic"),
    (2141, 2150, "emotion"),
    (2150, 2191, "bertopic"),
    (2191, 2263, "wasserstein"),
    (2263, 2266, "mahalanobis"),
    (2266, 2268, "combined_dist"),
    (2268, 2315, "tom"),
    (2315, 2340, "bandit"),
    (2340, 2341, "meta"),
]

def get_feature_name(idx):
    for start, end, name in feature_ranges:
        if start <= idx < end:
            return f"{name}[{idx - start}]"
    return f"unknown[{idx}]"

print(f"\nTop {top_k} features by importance:")
for rank, idx in enumerate(top_indices):
    print(f"  {rank+1:2d}. {get_feature_name(idx):<25s} {importances[idx]:.4f}")
```

### 8.3 Round-by-round evaluation

```python
from erisk_task2.config import load_config
from erisk_task2.pipeline import evaluate_pipeline

cfg = load_config("config/task2_colab.yaml")
cfg.training_data_dir = DATA_DIR
cfg.labels_path = LABELS
cfg.logging.output_dir = "/content/drive/MyDrive/erisk-task2-train"

evaluate_pipeline(cfg)
```

Results saved to `eval_results.json`:

```python
with open(OUTPUT_DIR / "eval_results.json") as f:
    eval_results = json.load(f)

print(f"\n{'Run':<8} {'Classifier':<12} {'F1':>7} {'P':>7} {'R':>7} {'ERDE5':>8} {'ERDE50':>8} {'F_lat':>8} {'Alerts':>7}")
print("-" * 75)
for run_name, m in eval_results.items():
    print(f"{run_name:<8} {m['classifier']:<12} {m['f1']:>7.4f} {m['precision']:>7.3f} "
          f"{m['recall']:>7.3f} {m['erde5']:>8.4f} {m['erde50']:>8.4f} "
          f"{m['f_latency']:>8.4f} {m['alerts']:>7d}")
```

---

## 9. Output Files

All saved to the output directory on Google Drive:

| File | Description |
|------|-------------|
| `classifier_xgboost.pkl` | Trained XGBoost model |
| `classifier_neural_net.pkl` | Trained MLP model |
| `classifier_ensemble.pkl` | Trained ensemble (stacking) model |
| `mahalanobis.pkl` | Fitted Mahalanobis scorer |
| `symptom_references.npy` | 21 BDI-II symptom reference embeddings (21 x 1920d) |
| `features.npz` | Full feature matrix (X, y, subject_ids) |
| `training_results.json` | CV metrics per classifier (Option A) |
| `training_results_with_tom.json` | CV metrics with LLM ToM (Option C) |
| `bertopic_model/` | Fitted BERTopic model directory |
| `tom_features.npz` | Per-user ToM Option C features (if computed) |
| `tom_features_checkpoint.pkl` | ToM computation checkpoint |
| `profiles_checkpoint.pkl` | Feature extraction checkpoint (removed after completion) |
| `eval_results.json` | Round-by-round evaluation metrics per run |

---

## 10. 5 Competition Runs

| Run | Classifier | Threshold | ERDE | Description |
|-----|-----------|-----------|------|-------------|
| R0 | XGBoost | 0.85 -> 0.45 | ERDE50 | Full features, conservative |
| R1 | XGBoost | 0.70 -> 0.35 | ERDE5 | Full features, aggressive early detection |
| R2 | Neural Net | 0.85 -> 0.45 | ERDE50 | MLP classifier variant |
| R3 | Ensemble | 0.80 -> 0.40 | ERDE30 | Stacking (XGB + MLP + SVM) |
| R4 | XGBoost | 0.85 -> 0.45 | ERDE50 | Ablation: no ToM features |

---

## 11. Troubleshooting

### Out of memory (GPU)
```python
cfg.embedding.batch_size = 32
cfg.embedding.device = "cpu"  # fallback
```

### Out of memory (RAM) during BERTopic
```python
# Reduce BERTopic documents (sample instead of all)
cfg.bertopic.min_cluster_size = 100  # larger clusters = fewer topics
```

### HuggingFace API rate limits
The HF Pro tier allows ~1000 requests/hour for Llama 70B. At 2 prompts per user:
- 909 users = ~1818 requests = ~2 hours
- Checkpoint saves every 25 users, so disconnects are recoverable

### Session disconnects
- Output goes to Google Drive, so models persist
- Checkpoints allow resuming feature extraction and ToM computation
- After reconnect: remount Drive, `%cd` to project, re-run the training cell

### Emotion classifier device mismatch
```python
# If emotion classifier fails on CUDA, force CPU
cfg.emotion.model = "j-hartmann/emotion-english-distilroberta-base"
# The EmotionClassifier uses the embedding device setting
```
