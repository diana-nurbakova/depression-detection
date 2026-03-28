#!/usr/bin/env bash
# =============================================================================
# Remote Training Script for eRisk Task 2
# Syncs project + data to a GPU server, runs training, downloads results.
#
# Usage:
#   ./scripts/remote_train_task2.sh <user@host> [--setup-only] [--run-only] [--download-only]
#
# Prerequisites:
#   - SSH key-based auth configured for the remote server
#   - Python 3.11+ and uv available (or installable) on the remote
#   - NVIDIA GPU with CUDA drivers on the remote
# =============================================================================

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────────
REMOTE_HOST="${1:?Usage: $0 <user@host> [--setup-only|--run-only|--download-only]}"
FLAG="${2:-}"

REMOTE_PROJECT_DIR="~/depression-detection"
REMOTE_DATA_DIR="${REMOTE_PROJECT_DIR}/data"
REMOTE_OUTPUT_DIR="${REMOTE_PROJECT_DIR}/runs/task2/train"

LOCAL_PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOCAL_DATA_DIR="${LOCAL_PROJECT_DIR}/data/eRisk-2025"
LOCAL_OUTPUT_DIR="${LOCAL_PROJECT_DIR}/runs/task2/train"

# Training data subpath (relative to data/eRisk-2025/)
TRAIN_DATA_SUBPATH="eRisk25-datasets/t2-early-contextualized-depression/final-eriskt2-dataset-with-ground-truth/final-eriskt2-dataset-with-ground-truth/all_combined"

# ── Helpers ────────────────────────────────────────────────────────────────────
log()  { echo "[$(date '+%H:%M:%S')] $*"; }
rssh() { ssh -o ConnectTimeout=10 "${REMOTE_HOST}" "$@"; }

# ── Phase 1: Setup remote environment ─────────────────────────────────────────
setup_remote() {
    log "=== Phase 1: Setting up remote environment ==="

    log "Creating remote directories..."
    rssh "mkdir -p ${REMOTE_PROJECT_DIR} ${REMOTE_DATA_DIR}"

    log "Syncing project source code..."
    rsync -avz --progress \
        --exclude '.git' \
        --exclude '.venv' \
        --exclude '__pycache__' \
        --exclude 'data/' \
        --exclude 'runs/' \
        --exclude 'notebooks/' \
        --exclude 'specs/' \
        --exclude '.env' \
        --exclude '*.egg-info' \
        --exclude 'dist/' \
        --exclude '.vscode/' \
        --exclude '.idea/' \
        "${LOCAL_PROJECT_DIR}/" "${REMOTE_HOST}:${REMOTE_PROJECT_DIR}/"

    log "Syncing training data..."
    rsync -avz --progress \
        "${LOCAL_DATA_DIR}/" "${REMOTE_HOST}:${REMOTE_DATA_DIR}/eRisk-2025/"

    log "Creating remote .env file..."
    rssh "cat > ${REMOTE_PROJECT_DIR}/.env" <<'ENVEOF'
# Minimal .env for training (add tokens if needed for ToM LLM calls)
# ERISK_TOKEN=
# ERISK_USER=
# OLLAMA_BASE_URL=
# OLLAMA_API_KEY=
ENVEOF

    log "Creating GPU-enabled config override..."
    rssh "cat > ${REMOTE_PROJECT_DIR}/config/task2_gpu.yaml" <<'CFGEOF'
# GPU override for remote training — merged on top of task2.yaml
# Paths (relative to remote project root)
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
CFGEOF

    log "Installing uv and project dependencies on remote..."
    rssh <<'SETUPEOF'
set -euo pipefail
cd ~/depression-detection

# Install uv if not present
if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Create venv and install deps
unset VIRTUAL_ENV
uv sync

# Verify GPU
uv run python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available:  {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU:             {torch.cuda.get_device_name(0)}')
    print(f'VRAM:            {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
else:
    print('WARNING: No GPU detected! Training will be slow.')
"
SETUPEOF

    log "Setup complete."
}

# ── Phase 2: Launch training ──────────────────────────────────────────────────
run_training() {
    log "=== Phase 2: Launching training ==="

    # Use nohup + tmux so training survives SSH disconnects
    rssh <<'TRAINEOF'
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd ~/depression-detection

# Create output dir
mkdir -p runs/task2/train

# Kill any previous training tmux session
tmux kill-session -t task2_train 2>/dev/null || true

# Launch in tmux
tmux new-session -d -s task2_train "bash -c '
    set -euo pipefail
    export PATH=\"$HOME/.local/bin:$PATH\"
    cd ~/depression-detection
    unset VIRTUAL_ENV

    echo \"[$(date)] Starting Task 2 training...\" | tee runs/task2/train/training.log

    uv run erisk-task2 train \
        --config config/task2_gpu.yaml \
        --output-dir runs/task2/train \
        2>&1 | tee -a runs/task2/train/training.log

    echo \"[$(date)] Training complete.\" | tee -a runs/task2/train/training.log
'"

echo ""
echo "Training launched in tmux session 'task2_train'."
echo "Monitor with:  ssh $HOSTNAME -t 'tmux attach -t task2_train'"
echo "Or check logs: ssh $HOSTNAME 'tail -f ~/depression-detection/runs/task2/train/training.log'"
TRAINEOF

    log "Training launched in background tmux session."
    log ""
    log "To monitor progress:"
    log "  ssh ${REMOTE_HOST} -t 'tmux attach -t task2_train'"
    log ""
    log "To check logs:"
    log "  ssh ${REMOTE_HOST} 'tail -f ${REMOTE_PROJECT_DIR}/runs/task2/train/training.log'"
    log ""
    log "Once training is complete, run:"
    log "  $0 ${REMOTE_HOST} --download-only"
}

# ── Phase 3: Download results ────────────────────────────────────────────────
download_results() {
    log "=== Phase 3: Downloading results ==="

    mkdir -p "${LOCAL_OUTPUT_DIR}"

    rsync -avz --progress \
        "${REMOTE_HOST}:${REMOTE_OUTPUT_DIR}/" "${LOCAL_OUTPUT_DIR}/"

    log "Results downloaded to: ${LOCAL_OUTPUT_DIR}"
    log ""
    log "Expected artifacts:"
    ls -lh "${LOCAL_OUTPUT_DIR}/" 2>/dev/null || log "(directory listing failed)"
}

# ── Main ──────────────────────────────────────────────────────────────────────
case "${FLAG}" in
    --setup-only)    setup_remote ;;
    --run-only)      run_training ;;
    --download-only) download_results ;;
    "")
        setup_remote
        run_training
        ;;
    *)
        echo "Unknown flag: ${FLAG}"
        echo "Usage: $0 <user@host> [--setup-only|--run-only|--download-only]"
        exit 1
        ;;
esac
