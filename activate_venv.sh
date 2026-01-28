#!/bin/bash
################################################################################
# Quick Activation Script for Persistent Venv
# Run this when you restart RunPod pod: source activate_venv.sh
################################################################################

VENV_PATH="/workspace/venvs/surg-audio"

if [ -d "$VENV_PATH" ]; then
    source "$VENV_PATH/bin/activate"
    echo "✓ Virtual environment activated: $VENV_PATH"
    echo "Python: $(which python)"
    echo "PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"
else
    echo "❌ Virtual environment not found at: $VENV_PATH"
    echo "Run setup_runpod_venv.sh first"
    exit 1
fi
