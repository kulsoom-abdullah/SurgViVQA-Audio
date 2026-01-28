#!/bin/bash
################################################################################
# Quick activation script for Lambda Labs
# Source this instead of typing the full path every time
#
# Usage: source activate_venv_lambdalabs.sh
################################################################################

# CONFIGURATION - Should match setup_lambdalabs_venv.sh
VENV_PATH="$HOME/venvs/surg-audio"

if [ -d "$VENV_PATH" ]; then
    source "$VENV_PATH/bin/activate"
    echo "✓ Virtual environment activated"
    echo "Python: $(which python)"
    echo "Location: $(pwd)"
else
    echo "❌ Virtual environment not found at: $VENV_PATH"
    echo "Run setup_lambdalabs_venv.sh first!"
    return 1  # Use return instead of exit when sourced
fi
