#!/bin/bash
# setup_pascal.sh — One-time setup for autoresearch on GTX 1080 Ti (Pascal)
# Replaces the uv-based workflow with conda + pip for PyTorch 2.0.1 compatibility
set -e

ENV_NAME="autoresearch"

echo "=== Creating conda environment '$ENV_NAME' with Python 3.10 ==="
conda create -y -n "$ENV_NAME" python=3.10

echo "=== Activating environment ==="
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

echo "=== Installing PyTorch 2.0.1+cu118 ==="
pip install torch==2.0.1+cu118 --index-url https://download.pytorch.org/whl/cu118

echo "=== Installing remaining dependencies ==="
pip install rustbpe tiktoken matplotlib numpy pandas huggingface-hub pyarrow tqdm rich pyyaml

echo "=== Verifying CUDA ==="
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Arch list:', torch.cuda.get_arch_list())"

echo ""
echo "Setup complete. Activate with: conda activate $ENV_NAME"
