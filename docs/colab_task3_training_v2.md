# Task 3 (HiPerT v2): Cross-Encoder Training on Google Colab

## Overview

HiPerT v2 replaces the bi-encoder (which collapsed, CV ≈ 0.01) with a **cross-encoder reranker** trained via **CORAL ordinal regression** or **ListMLE listwise ranking**. This is a single-stage distillation from LLM silver labels — no depression pre-training needed.

**Key changes from v1:**
- Cross-encoder: `[CLS] symptom [SEP] sentence [SEP]` (joint attention)
- CORAL/ListMLE losses (prevent representation collapse)
- Leave-symptom-out 5-fold cross-validation
- Score spread diagnostic (CV > 0.05 = healthy)
- Run reordering: LLM = Run 1 (primary), HiPerT = Run 2

---

## 1. Setup

### 1.1 Mount Drive and upload project

```python
from google.colab import drive
drive.mount('/content/drive')

# Copy project from Drive (or git clone)
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
    print(f"Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
```

### 1.3 Install dependencies

```bash
!pip install -e . 2>&1 | tail -5
```

If PyTorch needs CUDA upgrade:
```bash
!pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu121
!pip install -e .
```

### 1.4 Verify v2 imports

```python
from hipert.training.cross_encoder import CrossEncoderReranker, BACKBONES
from hipert.training.losses import CORALLoss, ListMLELoss
from hipert.training.cross_encoder_dataset import CrossEncoderDataset, create_cv_splits, SYMPTOM_CV_FOLDS
from hipert.training.extract_training_data import ASRS_ITEMS
from hipert.training.trainer_v2 import TrainerV2, TrainerV2Config, diagnose_score_spread, select_best_variant
from hipert.training.inference_v2 import run_v2_inference, run_v2_ensemble_inference
print("All v2 imports OK")
print(f"Backbones: {list(BACKBONES.keys())}")
print(f"CV folds: {list(SYMPTOM_CV_FOLDS.keys())}")
```

---

## 2. Data

### 2.1 Required data

| Data | Local path | Description |
|------|-----------|-------------|
| **Training data** | `output/training_v2/training_data.jsonl` | 89,998 (symptom, sentence, score) triples extracted from LLM cascade |
| **Candidates** | `output/candidates/symptom_{1-18}.json` | Pre-computed retrieval candidates (for inference) |
| **Silver labels** | `output/silver_labels/symptom_{1-18}.jsonl` | LLM scoring results (only needed if re-extracting) |

> **Note:** The training data was already extracted locally via `hipert extract-v2`. Upload the JSONL file to Colab. If not yet extracted, run Section 2.3 below.

### 2.2 Set paths

```python
import os
from pathlib import Path

DRIVE_BASE = Path("/content/drive/MyDrive")
PROJECT_DIR = Path("/content/depression-detection")

# Training data (extracted from LLM cascade)
TRAINING_DATA_PATH = PROJECT_DIR / "output/training_v2/training_data.jsonl"

# Candidates for inference
CANDIDATES_DIR = PROJECT_DIR / "output/candidates"

# Output (save to Drive for persistence across sessions)
CHECKPOINT_DIR = DRIVE_BASE / "hipert-v2-checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Verify
print(f"Training data: {TRAINING_DATA_PATH.exists()} ({TRAINING_DATA_PATH})")
print(f"Candidates: {len(list(CANDIDATES_DIR.glob('symptom_*.json')))} files")
```

### 2.3 Extract training data (if not already done)

```python
# Only needed if training_data.jsonl doesn't exist yet
from hipert.training.extract_training_data import extract_training_data, save_training_data

data = extract_training_data(
    silver_labels_dir=PROJECT_DIR / "output/silver_labels",
    candidates_dir=CANDIDATES_DIR,
)

TRAINING_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
save_training_data(data, TRAINING_DATA_PATH)
print(f"Extracted {len(data)} training examples")
```

### 2.4 Verify data distribution

```python
from collections import Counter
from hipert.training.extract_training_data import load_training_data

data = load_training_data(TRAINING_DATA_PATH)
dist = Counter(d["score"] for d in data)
total = len(data)
print(f"Total: {total} examples")
for score in sorted(dist.keys()):
    print(f"  Score {score}: {dist[score]:,} ({100*dist[score]/total:.1f}%)")

# Expected: ~73% score-0, 15% score-1, 9% score-2, 2% score-3
```

---

## 3. Training: CORAL (Recommended)

CORAL ordinal regression is the primary loss function. Each sentence gets independent predictions via cumulative binary thresholds P(score ≥ k), which prevents the representation collapse that killed v1.

### 3.1 Train single backbone (quick test)

```python
import logging, sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

from hipert.training.extract_training_data import load_training_data
from hipert.training.trainer_v2 import TrainerV2, TrainerV2Config

data = load_training_data(TRAINING_DATA_PATH)

config = TrainerV2Config(
    backbone_name="mpnet",
    head_type="coral",
    learning_rate=2e-5,
    max_epochs=20,
    batch_size=64,
    threshold_weights=[1.0, 1.5, 2.0],  # Upweight the 2/3 boundary
    use_confidence_weighting=True,
    checkpoint_dir=CHECKPOINT_DIR,
    num_folds=5,
    patience=3,
    num_unfrozen=4,
)

trainer = TrainerV2(config)
summary = trainer.train_all_folds(data)

print(f"\nResult: NDCG@10={summary['mean_ndcg@10']:.4f}±{summary['std_ndcg@10']:.4f}")
print(f"        P@10={summary['mean_p@10']:.4f}")
print(f"        Mean CV={summary['mean_cv']:.4f} (must be > 0.05)")
```

### 3.2 Train all 3 backbones

```python
from hipert.training.extract_training_data import load_training_data
from hipert.training.trainer_v2 import TrainerV2, TrainerV2Config

data = load_training_data(TRAINING_DATA_PATH)

backbones = ["mpnet", "mental-roberta", "clinical-bert"]
results = {}

for bb in backbones:
    print(f"\n{'='*60}")
    print(f"CORAL training: {bb}")
    print(f"{'='*60}")

    config = TrainerV2Config(
        backbone_name=bb,
        head_type="coral",
        learning_rate=2e-5,
        max_epochs=20,
        batch_size=64,
        threshold_weights=[1.0, 1.5, 2.0],
        use_confidence_weighting=True,
        checkpoint_dir=CHECKPOINT_DIR,
        num_folds=5,
        patience=3,
    )

    trainer = TrainerV2(config)
    summary = trainer.train_all_folds(data)
    results[bb] = summary

    print(f"  {bb}: NDCG@10={summary['mean_ndcg@10']:.4f} P@10={summary['mean_p@10']:.4f} CV={summary['mean_cv']:.4f}")

print("\n\nAll CORAL results:")
for bb, s in results.items():
    print(f"  {bb}: NDCG@10={s['mean_ndcg@10']:.4f}±{s['std_ndcg@10']:.4f} P@10={s['mean_p@10']:.4f}")
```

---

## 4. Training: ListMLE (Alternative)

ListMLE is the listwise alternative — directly optimizes ranking permutation likelihood. Train in parallel with CORAL (or after) to compare.

### 4.1 Train all 3 backbones with ListMLE

```python
from hipert.training.extract_training_data import load_training_data
from hipert.training.trainer_v2 import TrainerV2, TrainerV2Config

data = load_training_data(TRAINING_DATA_PATH)

backbones = ["mpnet", "mental-roberta", "clinical-bert"]
results_listmle = {}

for bb in backbones:
    print(f"\n{'='*60}")
    print(f"ListMLE training: {bb}")
    print(f"{'='*60}")

    config = TrainerV2Config(
        backbone_name=bb,
        head_type="listmle",
        learning_rate=1e-5,          # Lower LR for listwise (noisier gradients)
        max_epochs=30,               # More epochs (sees less data per step)
        batch_size=64,
        listmle_temperature=1.0,
        sublist_size=64,
        num_sublists_per_batch=8,
        steps_per_epoch=200,
        checkpoint_dir=CHECKPOINT_DIR,
        num_folds=5,
        patience=5,                  # More patience for ListMLE
    )

    trainer = TrainerV2(config)
    summary = trainer.train_all_folds(data)
    results_listmle[bb] = summary

    print(f"  {bb}: NDCG@10={summary['mean_ndcg@10']:.4f} P@10={summary['mean_p@10']:.4f} CV={summary['mean_cv']:.4f}")
```

---

## 5. Model Selection

### 5.1 Compare CORAL vs ListMLE

```python
from hipert.training.trainer_v2 import select_best_variant

# Compare for each backbone
for bb in ["mpnet", "mental-roberta", "clinical-bert"]:
    coral_s = results.get(bb, {})
    listmle_s = results_listmle.get(bb, {})

    if coral_s and listmle_s:
        best = select_best_variant(coral_s, listmle_s)
        print(f"{bb}: CORAL NDCG={coral_s['mean_ndcg@10']:.4f} vs ListMLE NDCG={listmle_s['mean_ndcg@10']:.4f} → {best}")
```

### 5.2 Score spread diagnostic

Run this on the best model to verify no collapse:

```python
from hipert.training.cross_encoder import CrossEncoderReranker
from hipert.training.trainer_v2 import diagnose_score_spread

# Load best model (adjust head_type and backbone as needed)
HEAD_TYPE = "coral"     # or "listmle" — whichever won
BACKBONE = "mpnet"      # or whichever performed best

ckpt_path = CHECKPOINT_DIR / HEAD_TYPE / BACKBONE / "fold_1" / "best.pt"
model = CrossEncoderReranker.load_checkpoint(ckpt_path)

report = diagnose_score_spread(
    model, data, model.tokenizer,
    device="cuda" if torch.cuda.is_available() else "cpu",
)

if report["healthy"]:
    print(f"\n✓ HEALTHY — Mean CV = {report['mean_cv']:.4f} (threshold: 0.05)")
else:
    print(f"\n✗ COLLAPSED — Mean CV = {report['mean_cv']:.4f}")
    print(f"  {report['n_collapsed']}/18 symptoms collapsed")
    print("  Do NOT submit. Investigate loss or hyperparameters.")
```

---

## 6. Inference

### 6.1 Ensemble inference (3 backbones × 5 folds)

```python
from hipert.training.inference_v2 import run_v2_ensemble_inference

HEAD_TYPE = "coral"  # or "listmle" — whichever won selection
ENCODER_SCORES_DIR = PROJECT_DIR / "output" / "encoder_scores_v2"

run_v2_ensemble_inference(
    checkpoint_dir=CHECKPOINT_DIR,
    candidates_dir=CANDIDATES_DIR,
    output_dir=ENCODER_SCORES_DIR,
    head_type=HEAD_TYPE,
    backbones=["mpnet", "mental-roberta", "clinical-bert"],
    num_folds=5,
    batch_size=128,
    top_n=1000,
)

print(f"Ensemble scores written to {ENCODER_SCORES_DIR}")
```

### 6.2 Single backbone inference (if only one backbone trained)

```python
from hipert.training.inference_v2 import run_v2_inference

ENCODER_SCORES_DIR = PROJECT_DIR / "output" / "encoder_scores_v2"

run_v2_inference(
    checkpoint_dir=CHECKPOINT_DIR,
    candidates_dir=CANDIDATES_DIR,
    output_dir=ENCODER_SCORES_DIR,
    head_type="coral",
    backbone_name="mpnet",
    num_folds=5,
    batch_size=128,
    top_n=1000,
)
```

---

## 7. Copy Results to Drive & Download

### 7.1 Copy encoder scores to Drive

```python
import shutil

# Copy scored rankings to Drive for download
src = PROJECT_DIR / "output" / "encoder_scores_v2"
dst = DRIVE_BASE / "hipert-v2-results" / "encoder_scores_v2"
dst.mkdir(parents=True, exist_ok=True)

for f in src.glob("symptom_*.json"):
    shutil.copy2(f, dst / f.name)
    print(f"Copied {f.name}")

print(f"\nScores saved to: {dst}")
print("Download this folder and place it at output/encoder_scores_v2/ locally")
```

### 7.2 Copy checkpoints to Drive (already there if CHECKPOINT_DIR is on Drive)

```python
# Verify checkpoints are persisted
import os
for head in ["coral", "listmle"]:
    head_dir = CHECKPOINT_DIR / head
    if head_dir.exists():
        for bb_dir in sorted(head_dir.iterdir()):
            if bb_dir.is_dir():
                ckpts = list(bb_dir.rglob("best.pt"))
                print(f"  {head}/{bb_dir.name}: {len(ckpts)} best checkpoints")
```

---

## 8. Generate Final Submissions (Local)

After downloading `encoder_scores_v2/` back to your local machine:

```bash
# Place scores at output/encoder_scores_v2/
# Then generate all 5 TREC runs:
unset VIRTUAL_ENV && uv run hipert output --run all
```

The 5 runs (v2 ordering):
- **Run 1** (`INSALyon_LLM_cascade`): LLM scoring — PRIMARY
- **Run 2** (`INSALyon_HiPerT_full`): Cross-encoder reranker
- **Run 3** (`INSALyon_Ensemble`): RRF fusion of Run 1 + Run 2
- **Run 4** (`INSALyon_DepTransfer`): Depression transfer (v1, unchanged)
- **Run 5** (`INSALyon_BiEnc_baseline`): Cosine baseline (unchanged)

---

## 9. Leave-Symptom-Out CV Folds

The 5 folds test cross-symptom generalization:

| Fold | Train Symptoms | Validation Symptoms | Tests |
|------|---------------|-------------------|-------|
| 1 | 1–14 | 15–18 | Verbal H/I generalization |
| 2 | 1–4, 9–18 | 5–8 | Motor H/I generalization |
| 3 | 5–18 | 1–4 | Organization/Planning generalization |
| 4 | 1–8, 13–18 | 9–12 | Sustained Attention generalization |
| 5 | 1–12 | 13–18 | Internal Drive + Verbal generalization |

If CV variance > 15%, fall back to sentence-level stratified splits.

---

## 10. Expected Timings (Colab)

| Task | A100 | T4 (free) |
|------|------|-----------|
| Extract training data | < 1 min | < 1 min |
| CORAL: 1 backbone × 5 folds × 20 epochs | ~1.5 hours | ~4 hours |
| CORAL: 3 backbones × 5 folds | ~4.5 hours | ~12 hours |
| ListMLE: 1 backbone × 5 folds × 30 epochs | ~2 hours | ~6 hours |
| Inference: 3 backbones × 5 folds × 18 symptoms | ~15 min | ~45 min |
| Score spread diagnostic | ~5 min | ~15 min |
| **Full pipeline (CORAL only, 3 backbones)** | **~5 hours** | **~13 hours** |
| **Full pipeline (both CORAL + ListMLE)** | **~9 hours** | **~25 hours** |

> For free Colab (T4, 12-hour limit): train 1 backbone per session. Checkpoints save to Drive automatically.

---

## 11. Troubleshooting

### Out of GPU memory
```python
# Reduce batch size
config = TrainerV2Config(batch_size=32, ...)  # instead of 64

# Or reduce max_length
config = TrainerV2Config(max_length=192, ...)  # instead of 256
```

### Colab disconnects mid-training
- Checkpoints save to Google Drive per fold — they persist
- Fold checkpoints: `{CHECKPOINT_DIR}/{head_type}/{backbone}/fold_{N}/best.pt`
- Re-run training — completed folds will have checkpoints; manually skip them or let the trainer overwrite

### Model has collapsed (CV < 0.05)
1. Check threshold_weights — try `[1.0, 2.0, 3.0]` to upweight the harder boundaries
2. Try unfreezing more layers: `num_unfrozen=6` instead of 4
3. Try ListMLE instead of CORAL (or vice versa)
4. Try sentence-level CV instead of symptom-level CV
5. Last resort: submit LLM cascade as primary, BiEnc as secondary

### ListMLE training is slow
ListMLE uses sublist sampling (64 sentences × 8 sublists = 512 per step, 200 steps/epoch). If too slow:
```python
config = TrainerV2Config(
    steps_per_epoch=100,         # halve steps
    num_sublists_per_batch=4,    # fewer sublists
)
```

### Pre-download backbones (avoid timeout during training)
```python
from transformers import AutoTokenizer, AutoModel

for name, model_id in [
    ("mpnet", "sentence-transformers/all-mpnet-base-v2"),
    ("mental-roberta", "mental/mental-roberta-base"),
    ("clinical-bert", "emilyalsentzer/Bio_ClinicalBERT"),
]:
    print(f"Downloading {name}...")
    AutoTokenizer.from_pretrained(model_id)
    AutoModel.from_pretrained(model_id)
    print(f"  {name} OK")
```

---

## 12. Quick Reference: CLI Commands (Local)

```bash
# Extract training data from LLM cascade outputs
uv run hipert extract-v2

# Train cross-encoder (CORAL, single backbone, 5 folds)
uv run hipert train-v2 --head-type coral --backbone mpnet

# Train all backbones
uv run hipert train-v2 --head-type coral --backbone all

# Train both CORAL and ListMLE
uv run hipert train-v2 --head-type both --backbone all

# Diagnose score spread
uv run hipert diagnose-v2 --head-type coral --backbone mpnet

# Run inference (ensemble)
uv run hipert infer-v2 --head-type coral --backbone all

# Generate all 5 TREC submission files
uv run hipert output --run all
```
