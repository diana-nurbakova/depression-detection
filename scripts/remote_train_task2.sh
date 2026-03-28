#!/usr/bin/env bash
# =============================================================================
# Task 2 Training — run directly on the GPU server
#
# Setup (once, from your local machine):
#   1. ssh user@server
#   2. git clone <repo-url> ~/depression-detection
#   3. scp -r /path/to/data/eRisk-2025 user@server:~/depression-detection/data/eRisk-2025
#   4. scp /path/to/.env user@server:~/depression-detection/.env
#
# Then on the server:
#   cd ~/depression-detection
#   ./scripts/remote_train_task2.sh            # setup + train (default)
#   ./scripts/remote_train_task2.sh --setup    # just install deps & verify GPU
#   ./scripts/remote_train_task2.sh --train    # just launch training
#   ./scripts/remote_train_task2.sh --pull     # git pull + train
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${PROJECT_DIR}/runs/task2/train"
CONFIG_GPU="${PROJECT_DIR}/config/task2_gpu.yaml"
LOG_FILE="${OUTPUT_DIR}/training.log"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── Setup: deps + GPU check ──────────────────────────────────────────────────
setup() {
    log "=== Setup ==="
    cd "${PROJECT_DIR}"

    # Install uv if not present
    if ! command -v uv &>/dev/null; then
        log "Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi
    log "uv: $(uv --version)"

    # Install project deps
    unset VIRTUAL_ENV 2>/dev/null || true
    uv sync
    log "Dependencies installed."

    # Create GPU config override if not present
    if [ ! -f "${CONFIG_GPU}" ]; then
        log "Creating GPU config: ${CONFIG_GPU}"
        cat > "${CONFIG_GPU}" <<'CFGEOF'
# GPU override — inherits everything from task2.yaml except these overrides
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

# Use HuggingFace Inference API instead of Ollama for ToM LLM calls
llm:
  backend: "hf"

hf_inference:
  model: "meta-llama/Llama-3.3-70B-Instruct"
  temperature: 0.1
  max_tokens: 2048
  timeout_seconds: 120
CFGEOF
    fi

    # Verify GPU
    log "Checking GPU..."
    uv run python -c "
import torch
print(f'  PyTorch:  {torch.__version__}')
print(f'  CUDA:     {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:      {torch.cuda.get_device_name(0)}')
    print(f'  VRAM:     {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
else:
    print('  WARNING: No GPU detected! Training will fall back to CPU.')
"

    # Verify training data
    TRAIN_DATA="${PROJECT_DIR}/data/eRisk-2025/eRisk25-datasets/t2-early-contextualized-depression/final-eriskt2-dataset-with-ground-truth/final-eriskt2-dataset-with-ground-truth/all_combined"
    if [ -d "${TRAIN_DATA}" ]; then
        FILE_COUNT=$(ls "${TRAIN_DATA}"/*.json 2>/dev/null | wc -l)
        log "Training data: ${FILE_COUNT} user files found."
    else
        log "ERROR: Training data not found!"
        log "Copy from local: scp -r /path/to/data/eRisk-2025 server:~/depression-detection/data/"
        exit 1
    fi

    # Verify .env
    if [ ! -f "${PROJECT_DIR}/.env" ]; then
        log "WARNING: No .env file found. Copy from local: scp .env server:~/depression-detection/"
    fi

    log "Setup complete."
}

# ── Train: launch in tmux ────────────────────────────────────────────────────
train() {
    log "=== Launching training ==="
    cd "${PROJECT_DIR}"
    mkdir -p "${OUTPUT_DIR}"
    unset VIRTUAL_ENV 2>/dev/null || true

    if command -v tmux &>/dev/null; then
        tmux kill-session -t task2_train 2>/dev/null || true

        tmux new-session -d -s task2_train "bash -c '
            set -euo pipefail
            export PATH=\"\$HOME/.local/bin:\$PATH\"
            cd ${PROJECT_DIR}
            unset VIRTUAL_ENV 2>/dev/null || true

            echo \"[\$(date)] Training started\" | tee ${LOG_FILE}

            uv run erisk-task2 train \
                --config ${CONFIG_GPU} \
                --output-dir ${OUTPUT_DIR} \
                2>&1 | tee -a ${LOG_FILE}

            echo \"\" | tee -a ${LOG_FILE}
            echo \"[\$(date)] Training complete.\" | tee -a ${LOG_FILE}
        '"

        log ""
        log "Training running in tmux. Safe to disconnect SSH."
        log ""
        log "  Attach:  tmux attach -t task2_train"
        log "  Logs:    tail -f ${LOG_FILE}"
        log "  Status:  tmux ls"
        log ""
        log "After training, download results from your local machine:"
        log "  scp -r server:~/depression-detection/runs/task2/train ./runs/task2/"
    else
        log "tmux not found — running with nohup..."
        nohup bash -c "
            set -euo pipefail
            export PATH=\"\$HOME/.local/bin:\$PATH\"
            cd ${PROJECT_DIR}
            unset VIRTUAL_ENV 2>/dev/null || true
            echo \"[\$(date)] Training started\" > ${LOG_FILE}
            uv run erisk-task2 train \
                --config ${CONFIG_GPU} \
                --output-dir ${OUTPUT_DIR} \
                2>&1 | tee -a ${LOG_FILE}
            echo \"[\$(date)] Training complete.\" | tee -a ${LOG_FILE}
        " &
        log "Training PID: $!"
        log "Logs: tail -f ${LOG_FILE}"
    fi
}

# ── Main ─────────────────────────────────────────────────────────────────────
FLAG="${1:---all}"

case "${FLAG}" in
    --setup)
        setup
        ;;
    --train)
        train
        ;;
    --pull)
        log "Pulling latest changes..."
        cd "${PROJECT_DIR}" && git pull
        setup
        train
        ;;
    --all)
        setup
        train
        ;;
    -h|--help)
        cat <<EOF
Usage: $0 [--setup|--train|--pull|--all|-h]

  --setup   Install deps, verify GPU & data
  --train   Launch training in tmux
  --pull    git pull + setup + train
  --all     setup + train (default)

Initial server setup:
  git clone <repo-url> ~/depression-detection
  scp -r local/data/eRisk-2025 server:~/depression-detection/data/eRisk-2025
  scp local/.env server:~/depression-detection/.env
  cd ~/depression-detection && ./scripts/remote_train_task2.sh
EOF
        ;;
    *)
        echo "Unknown flag: ${FLAG}. Use -h for help."
        exit 1
        ;;
esac
