# Training eRisk Task 2 on Google Colab

## 1. Setup

### 1.1 Upload the project

Option A — Clone from Git (if repo is on GitHub):
```python
!git clone https://github.com/<your-repo>/depression-detection.git
%cd depression-detection
```

Option B — Upload from Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')

# Copy project from Drive to Colab local storage (faster I/O)
!cp -r "/content/drive/MyDrive/depression-detection" /content/depression-detection
%cd /content/depression-detection
```

Option C — Upload a zip:
```python
from google.colab import files
uploaded = files.upload()  # upload depression-detection.zip
!unzip depression-detection.zip -d /content/depression-detection
%cd /content/depression-detection
```

### 1.2 Check GPU

```python
!nvidia-smi
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

### 1.3 Install the package

```bash
!pip install -e . 2>&1 | tail -5
```

This installs all dependencies: sentence-transformers, xgboost, bertopic, vaderSentiment, etc.

If you get PyTorch version conflicts:
```bash
!pip install --upgrade torch torchvision
!pip install -e .
```

### 1.4 Verify imports

```python
from erisk_task2.config import load_config
from erisk_task2.pipeline import train_pipeline
from erisk_task2.features.layer1 import EmbeddingEncoder
print("All imports OK")
```

---

## 2. Upload training data

The training data consists of:
- **`all_combined/`** — directory with one JSON file per user (909 files)
- **`shuffled_ground_truth_labels.txt`** — tab-separated labels file (`subject_id\t0/1`)

### Option A — From Google Drive

```python
DATA_DIR = "/content/drive/MyDrive/eRisk-data/all_combined"
LABELS = "/content/drive/MyDrive/eRisk-data/shuffled_ground_truth_labels.txt"
```

### Option B — Upload directly

```python
# Upload the zip containing training data
from google.colab import files
uploaded = files.upload()  # upload training_data.zip

!mkdir -p /content/data
!unzip training_data.zip -d /content/data

DATA_DIR = "/content/data/all_combined"
LABELS = "/content/data/shuffled_ground_truth_labels.txt"
```

### Verify data is accessible

```python
import os
n_files = len([f for f in os.listdir(DATA_DIR) if f.endswith('.json')])
print(f"Found {n_files} user JSON files")  # should be 909

with open(LABELS) as f:
    lines = f.readlines()
print(f"Found {len(lines)} label entries")  # should be 909
```

---

## 3. Configuration

### 3.1 Use GPU for embeddings

Create a modified config that uses CUDA:

```python
# Write a Colab-specific config
config_content = """
# eRisk Task 2 — Colab training config

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
  enabled: false
  method: "option_a"

ollama:
  base_url: "http://localhost:11434"
  model: "llama3.3:70b"
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
  output_dir: "/content/runs/task2/train"
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

print("Config written to config/task2_colab.yaml")
```

Key differences from default config:
- `device: "cuda"` — use GPU for sentence transformers
- `batch_size: 128` — larger batch for GPU
- `tom.enabled: false` — ToM requires Ollama (not available on Colab by default)
- `output_dir` points to `/content/runs/task2/train`

---

## 4. Run training

### Option A — CLI command

```bash
!erisk-task2 -c config/task2_colab.yaml train \
  --data-dir "$DATA_DIR" \
  --labels "$LABELS" \
  --output-dir /content/runs/task2/train
```

### Option B — Python API (recommended for Colab, gives more control)

```python
import logging
import sys

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
cfg.logging.output_dir = "/content/runs/task2/train"

train_pipeline(cfg)
```

---

## 5. Expected output

The pipeline runs 7 steps:

| Step | Description | Time estimate (T4 GPU) |
|------|-------------|----------------------|
| 1 | Load 909 users from JSON | ~30s |
| 2 | Load 3 sentence transformers | ~1-2 min (download on first run) |
| 3 | Process users (100 threads each, embed + symptoms + sentiment) | ~30-60 min |
| 4 | Extract feature vectors (~2300d per user) | ~1 min |
| 5 | Fit Mahalanobis scorer on control users | ~10s |
| 6 | 5-fold CV for each classifier (XGBoost, MLP, SVM, Ensemble) | ~5-10 min |
| 7 | Train final models on full data + save | ~2 min |

Total: approximately 40-75 minutes with GPU.

### Output files

All saved to `/content/runs/task2/train/`:

| File | Description |
|------|-------------|
| `classifier_xgboost.pkl` | Trained XGBoost model |
| `classifier_mlp.pkl` | Trained MLP model |
| `classifier_svm.pkl` | Trained SVM model |
| `classifier_ensemble.pkl` | Trained ensemble (stacking) model |
| `mahalanobis.pkl` | Fitted Mahalanobis scorer |
| `symptom_references.npy` | 21 BDI-II symptom reference embeddings |
| `features.npz` | Full feature matrix (X, y, subject_ids) |
| `training_results.json` | CV metrics per classifier |

### Expected metrics log

```
=== TRAINING COMPLETE ===
  xgboost:  F1=0.XXXX  ERDE5=0.XXXX  ERDE50=0.XXXX  F_lat=0.XXXX
  mlp:      F1=0.XXXX  ERDE5=0.XXXX  ERDE50=0.XXXX  F_lat=0.XXXX
  svm:      F1=0.XXXX  ERDE5=0.XXXX  ERDE50=0.XXXX  F_lat=0.XXXX
  ensemble: F1=0.XXXX  ERDE5=0.XXXX  ERDE50=0.XXXX  F_lat=0.XXXX
```

---

## 6. Save results to Google Drive

```python
import shutil
shutil.copytree("/content/runs/task2/train", "/content/drive/MyDrive/erisk-task2-models", dirs_exist_ok=True)
print("Models saved to Google Drive")
```

---

## 7. Inspect results

```python
import json
import numpy as np

# Load training results
with open("/content/runs/task2/train/training_results.json") as f:
    results = json.load(f)

for clf_type, metrics in results.items():
    print(f"\n{clf_type}:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

# Load feature matrix
data = np.load("/content/runs/task2/train/features.npz", allow_pickle=True)
print(f"\nFeature matrix: {data['X'].shape}")
print(f"Labels: {data['y'].sum():.0f} depressed / {(data['y']==0).sum():.0f} control")
```

---

## 8. Troubleshooting

### Out of memory
If you get OOM errors, reduce the number of models or batch size:
```python
# Use only 1 model instead of 3 (640d instead of 1920d)
cfg.embedding.models = ["all-MiniLM-L12-v2"]
cfg.embedding.batch_size = 32
```

### Session disconnects
Colab sessions can disconnect after ~90 min of inactivity. To prevent:
- Keep the browser tab active
- Save intermediate results to Google Drive
- The checkpoint system saves state, but the training pipeline runs as one batch (no mid-training checkpoints)

### Package version conflicts
If sentence-transformers or transformers have issues:
```bash
!pip install sentence-transformers>=3.0 transformers>=4.40 torch>=2.4
!pip install -e .
```

### CUDA out of memory
Lower the batch size:
```python
cfg.embedding.batch_size = 16
```
Or fall back to CPU:
```python
cfg.embedding.device = "cpu"
cfg.embedding.batch_size = 64
```
