# Task 3 (HiPerT-ADHD): Encoder Training on Google Colab

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

### 1.4 Verify imports

```python
from hipert.training.encoder import SymptomConditionedEncoder, BACKBONES
from hipert.training.trainer import Trainer, TrainingConfig
from hipert.training.stage_a import train_stage_a
from hipert.training.stage_b import train_stage_b
from hipert.training.inference import run_inference, run_ensemble_inference
from hipert.training.calibration import CalibrationPipeline
print("All training imports OK")
print(f"Backbones: {list(BACKBONES.keys())}")
```

---

## 2. Upload Training Data

The encoder needs 3 data sources. Upload them to Google Drive beforehand.

### 2.1 Required data

| Data | Local path | Description |
|------|-----------|-------------|
| BDI-Sen 2.0 | `data/BDI-Sen/full_dataset/bdi_majority_vote.jsonl` | Stage A1: 5,003 graded annotations (0-3) across 21 BDI-II symptoms |
| eRisk 2025 T1 | `data/eRisk-2025/.../t1-depression-symptom-ranking/` | Stage A2: 11,042 binary judgments + 6,300 TREC files |
| Silver labels | `output/silver_labels/symptom_{1-18}.jsonl` | Stage B: LLM-generated ADHD labels from scoring step |
| Candidates | `output/candidates/symptom_{1-18}.json` | Inference: pre-computed retrieval candidates |

### 2.2 Set paths

```python
import os
from pathlib import Path

# Adjust these to your Drive layout
DRIVE_BASE = Path("/content/drive/MyDrive")
PROJECT_DIR = Path("/content/depression-detection")

# Stage A data
BDISEN_DIR = PROJECT_DIR / "data/BDI-Sen/full_dataset"
ERISK2025_DIR = PROJECT_DIR / "data/eRisk-2025/eRisk25-datasets/t1-depression-symptom-ranking"
ERISK2025_TREC_DIR = ERISK2025_DIR / "erisk25-t1-dataset/erisk25-t1-dataset"

# Stage B data
SILVER_LABELS_DIR = PROJECT_DIR / "output/silver_labels"

# Candidates for inference
CANDIDATES_DIR = PROJECT_DIR / "output/candidates"

# Output (save to Drive for persistence)
CHECKPOINT_DIR = DRIVE_BASE / "hipert-checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Verify data exists
print(f"BDI-Sen: {BDISEN_DIR.exists()}")
print(f"eRisk 2025: {ERISK2025_DIR.exists()}")
print(f"Silver labels: {len(list(SILVER_LABELS_DIR.glob('symptom_*.jsonl')))} files")
print(f"Candidates: {len(list(CANDIDATES_DIR.glob('symptom_*.json')))} files")
```

---

## 3. Stage A: Depression Pre-training

Stage A trains the encoder on gold-standard depression data before fine-tuning on ADHD.

- **A1**: BDI-Sen 2.0 — graded 0-3 labels, 21 BDI-II symptoms
- **A2** (optional): eRisk 2025 T1 — binary 0/1 labels

### 3.1 Train Stage A (single backbone)

```python
import logging, sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

from hipert.training.stage_a import train_stage_a

# Train mpnet first (fastest, best general-purpose)
stage_a_ckpt = train_stage_a(
    bdisen_dir=BDISEN_DIR,
    erisk2025_dir=ERISK2025_DIR,          # set to None to skip A2
    erisk2025_trec_dir=ERISK2025_TREC_DIR, # set to None to skip A2
    backbone_name="mpnet",
    checkpoint_dir=CHECKPOINT_DIR,
    max_epochs_a1=10,
    max_epochs_a2=5,
    batch_size=32,
    learning_rate=2e-5,
)

print(f"Stage A best checkpoint: {stage_a_ckpt}")
```

### 3.2 Train Stage A for all backbones

```python
from hipert.training.stage_a import train_stage_a

backbones = ["mpnet", "mental-roberta", "clinical-bert"]
stage_a_checkpoints = {}

for bb in backbones:
    print(f"\n{'='*60}")
    print(f"Stage A: {bb}")
    print(f"{'='*60}")

    ckpt = train_stage_a(
        bdisen_dir=BDISEN_DIR,
        erisk2025_dir=ERISK2025_DIR,
        erisk2025_trec_dir=ERISK2025_TREC_DIR,
        backbone_name=bb,
        checkpoint_dir=CHECKPOINT_DIR,
        max_epochs_a1=10,
        max_epochs_a2=5,
        batch_size=32,
        learning_rate=2e-5,
    )
    stage_a_checkpoints[bb] = ckpt
    print(f"  {bb} Stage A done: {ckpt}")

print("\nAll Stage A checkpoints:", stage_a_checkpoints)
```

---

## 4. Stage B: ADHD Fine-tuning

Stage B fine-tunes the Stage A encoder on LLM-generated silver labels with:
- Curriculum learning (easy symptoms first)
- Symmetric cross-entropy (noise-robust loss)
- Post-training calibration (per-symptom temperature scaling + Dirichlet)

### 4.1 Train Stage B (single backbone)

```python
from hipert.training.stage_b import train_stage_b

# Use the Stage A checkpoint from above (or specify path)
stage_a_ckpt = CHECKPOINT_DIR / "stage_a2" / "mpnet" / "stage_a2_best.pt"
if not stage_a_ckpt.exists():
    stage_a_ckpt = CHECKPOINT_DIR / "stage_a1" / "mpnet" / "stage_a1_best.pt"

stage_b_ckpt = train_stage_b(
    silver_labels_dir=SILVER_LABELS_DIR,
    stage_a_checkpoint=stage_a_ckpt,
    backbone_name="mpnet",
    checkpoint_dir=CHECKPOINT_DIR,
    max_epochs=15,
    batch_size=32,
    learning_rate=1e-5,
    min_confidence=0.3,
)

print(f"Stage B best checkpoint: {stage_b_ckpt}")
```

### 4.2 Train Stage B for all backbones

```python
from hipert.training.stage_b import train_stage_b

backbones = ["mpnet", "mental-roberta", "clinical-bert"]

for bb in backbones:
    print(f"\n{'='*60}")
    print(f"Stage B: {bb}")
    print(f"{'='*60}")

    # Find Stage A checkpoint
    stage_a_ckpt = CHECKPOINT_DIR / "stage_a2" / bb / "stage_a2_best.pt"
    if not stage_a_ckpt.exists():
        stage_a_ckpt = CHECKPOINT_DIR / "stage_a1" / bb / "stage_a1_best.pt"

    if not stage_a_ckpt.exists():
        print(f"  SKIP {bb}: no Stage A checkpoint found")
        continue

    stage_b_ckpt = train_stage_b(
        silver_labels_dir=SILVER_LABELS_DIR,
        stage_a_checkpoint=stage_a_ckpt,
        backbone_name=bb,
        checkpoint_dir=CHECKPOINT_DIR,
        max_epochs=15,
        batch_size=32,
        learning_rate=1e-5,
        min_confidence=0.3,
    )
    print(f"  {bb} Stage B done: {stage_b_ckpt}")
```

---

## 5. Inference

After training, score all candidates with the trained encoder(s).

### 5.1 Ensemble inference (Run 1: all 3 backbones averaged)

```python
from hipert.training.inference import run_ensemble_inference

ENCODER_SCORES_DIR = PROJECT_DIR / "output" / "encoder_scores"

run_ensemble_inference(
    checkpoint_dir=CHECKPOINT_DIR,
    candidates_dir=CANDIDATES_DIR,
    output_dir=ENCODER_SCORES_DIR,
    backbones=["mpnet", "mental-roberta", "clinical-bert"],
    stage="stage_b",
    batch_size=64,
    top_n=1000,
)

print(f"Encoder scores written to {ENCODER_SCORES_DIR}")
```

### 5.2 Transfer inference (Run 4: depression-only, Stage A checkpoint)

```python
from hipert.training.inference import run_inference

TRANSFER_SCORES_DIR = PROJECT_DIR / "output" / "transfer_scores"

run_inference(
    checkpoint_dir=CHECKPOINT_DIR,
    candidates_dir=CANDIDATES_DIR,
    output_dir=TRANSFER_SCORES_DIR,
    backbone_name="mpnet",
    stage="stage_a",    # Use depression-only checkpoint (no ADHD fine-tuning)
    batch_size=64,
    top_n=1000,
)

print(f"Transfer scores written to {TRANSFER_SCORES_DIR}")
```

---

## 6. Generate Final Submissions

### 6.1 Copy scores back to project and generate TREC files

```python
# If inference was done on Colab, copy scores back
import shutil

# Copy encoder scores
src = PROJECT_DIR / "output" / "encoder_scores"
dst = PROJECT_DIR / "output" / "encoder_scores"
# (already in place if inference ran on same machine)

# Generate all 5 runs
!cd /content/depression-detection && python -m hipert output --run all
```

### 6.2 Or generate runs individually

```python
# Runs 2, 3, 5 don't need training (already generated locally)
# Run 1 needs encoder_scores/
# Run 4 needs transfer_scores/
!cd /content/depression-detection && python -m hipert output --run 1,4
```

### 6.3 Download results

```python
# Copy TREC files to Drive for download
import shutil

rankings_dir = PROJECT_DIR / "output" / "rankings"
drive_out = DRIVE_BASE / "hipert-submissions"
drive_out.mkdir(parents=True, exist_ok=True)

for f in rankings_dir.glob("*.trec"):
    shutil.copy2(f, drive_out / f.name)
    print(f"Copied {f.name}")

print(f"\nSubmission files in: {drive_out}")
```

---

## 7. Checkpoints & Recovery

Checkpoints are saved to Google Drive automatically at multiple granularities:

| Checkpoint | When | Filename |
|-----------|------|----------|
| Step | Every 200-500 steps | `{stage}_{backbone}_step_000500.pt` |
| Epoch | Every epoch | `{stage}_{backbone}_epoch_003.pt` |
| Best | On validation improvement | `{stage}_{backbone}_best.pt` |
| Interrupt | On Ctrl+C / disconnect | `{stage}_{backbone}_interrupted.pt` |
| Final | After training completes | `{stage}_{backbone}_final.pt` |

Old step checkpoints are pruned (keeps last 5).

### Resume from checkpoint

```python
from hipert.training.stage_b import train_stage_b

# Resume from interrupted or epoch checkpoint
stage_b_ckpt = train_stage_b(
    silver_labels_dir=SILVER_LABELS_DIR,
    stage_a_checkpoint=CHECKPOINT_DIR / "stage_a2" / "mpnet" / "stage_a2_best.pt",
    backbone_name="mpnet",
    checkpoint_dir=CHECKPOINT_DIR,
    max_epochs=15,
    batch_size=32,
    learning_rate=1e-5,
    resume_from=CHECKPOINT_DIR / "stage_b" / "mpnet" / "stage_b_mpnet_interrupted.pt",
)
```

---

## 8. Expected Timings (Colab Pro A100)

| Stage | Backbone | Data size | Estimated time |
|-------|----------|-----------|---------------|
| A1 (BDI-Sen) | 1 backbone | ~5K examples, 10 epochs | ~15 min |
| A2 (eRisk) | 1 backbone | ~11K examples, 5 epochs | ~20 min |
| B (ADHD) | 1 backbone | ~90K silver labels, 15 epochs | ~45 min |
| Inference | 1 backbone | ~90K candidates | ~10 min |
| **Full pipeline** | **3 backbones** | **all stages** | **~4-5 hours** |

On free Colab T4, multiply by ~3x.

---

## 9. Troubleshooting

### Out of GPU memory
```python
# Reduce batch size
batch_size = 16  # instead of 32

# Or disable mixed precision (uses less peak memory at cost of speed)
# Edit TrainingConfig: mixed_precision=False
```

### Colab disconnects mid-training
- Checkpoints save to Google Drive — they persist
- Use the resume_from parameter to continue (Section 7)
- Check `CHECKPOINT_DIR / {stage} / {backbone} /` for latest checkpoint

### Model download fails
```python
# Pre-download backbones
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

### Stage A2 data not available
Stage A2 (eRisk 2025 T1) is optional. Pass `erisk2025_dir=None` to skip it. Stage A1 alone provides useful depression pre-training.
