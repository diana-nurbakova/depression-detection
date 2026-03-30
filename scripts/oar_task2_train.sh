#!/usr/bin/env bash
# =============================================================================
# OAR Job Script — Task 2 Training on GPU compute node
#
# Submit from the frontend:
#   oarsub -S ./scripts/oar_task2_train.sh
#
# Monitor:
#   oarstat -u                          # list your jobs
#   tail -f ~/depression-detection/runs/task2/train/training.log
#   oarstat -fj <JOB_ID>               # detailed job info
#
# Cancel:
#   oardel <JOB_ID>
# =============================================================================

# ── OAR directives ───────────────────────────────────────────────────────────
#OAR -n erisk-task2-train
#OAR -l /nodes=1/gpu=1,walltime=12:00:00
#OAR -p gpu='YES'
#OAR --stdout %jobid%-task2-train.stdout
#OAR --stderr %jobid%-task2-train.stderr
#OAR -d /home/%u/depression-detection

# =============================================================================
# NOTE: You may need to adjust the OAR directives above for your cluster.
# Common variations:
#   #OAR -l /nodes=1/gpunum=1,walltime=12:00:00
#   #OAR -p gpumodel='V100'
#   #OAR -t gpu
#   #OAR -q gpu
#
# Check your cluster's GPU properties with:
#   oarnodes -p gpu | head
#   oarstat --properties
# =============================================================================

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_DIR="${HOME}/depression-detection"
OUTPUT_DIR="${PROJECT_DIR}/runs/task2/train"
CONFIG_GPU="${PROJECT_DIR}/config/task2_gpu.yaml"
LOG_FILE="${OUTPUT_DIR}/training.log"
TRAIN_DATA_DIR="${PROJECT_DIR}/data/eRisk-2025/eRisk25-datasets/t2-early-contextualized-depression/final-eriskt2-dataset-with-ground-truth/final-eriskt2-dataset-with-ground-truth/all_combined"
LABELS_PATH="${PROJECT_DIR}/data/eRisk-2025/eRisk25-datasets/t2-early-contextualized-depression/final-eriskt2-dataset-with-ground-truth/final-eriskt2-dataset-with-ground-truth/shuffled_ground_truth_labels.txt"

export PATH="$HOME/.local/bin:$PATH"

cd "${PROJECT_DIR}"
mkdir -p "${OUTPUT_DIR}"

echo "[$(date)] Job ${OAR_JOB_ID:-unknown} started on $(hostname)" | tee "${LOG_FILE}"

# ── Environment setup ────────────────────────────────────────────────────────
# Load modules if your cluster uses them (uncomment/adjust as needed)
# module load cuda/12.x
# module load python/3.11

# Install uv if not present
if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Install deps
unset VIRTUAL_ENV 2>/dev/null || true
uv sync 2>&1 | tail -5

# ── GPU check ────────────────────────────────────────────────────────────────
echo "" | tee -a "${LOG_FILE}"
echo "=== GPU Info ===" | tee -a "${LOG_FILE}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | tee -a "${LOG_FILE}" || echo "nvidia-smi not found"

uv run python -c "
import torch
print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
" 2>&1 | tee -a "${LOG_FILE}"

# ── Create GPU config if needed ──────────────────────────────────────────────
if [ ! -f "${CONFIG_GPU}" ]; then
    cat > "${CONFIG_GPU}" <<'CFGEOF'
training_data_dir: "data/eRisk-2025/eRisk25-datasets/t2-early-contextualized-depression/final-eriskt2-dataset-with-ground-truth/final-eriskt2-dataset-with-ground-truth/all_combined"
labels_path: "data/eRisk-2025/eRisk25-datasets/t2-early-contextualized-depression/final-eriskt2-dataset-with-ground-truth/final-eriskt2-dataset-with-ground-truth/shuffled_ground_truth_labels.txt"

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

llm:
  backend: "hf"

hf_inference:
  model: "meta-llama/Llama-3.3-70B-Instruct"
  temperature: 0.1
  max_tokens: 2048
  timeout_seconds: 120
CFGEOF
fi

# ── Run training ─────────────────────────────────────────────────────────────
echo "" | tee -a "${LOG_FILE}"
echo "[$(date)] Starting training..." | tee -a "${LOG_FILE}"

uv run erisk-task2 --config "${CONFIG_GPU}" train \
    --data-dir "${TRAIN_DATA_DIR}" \
    --labels "${LABELS_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    2>&1 | tee -a "${LOG_FILE}"

echo "" | tee -a "${LOG_FILE}"
echo "[$(date)] Training complete." | tee -a "${LOG_FILE}"
