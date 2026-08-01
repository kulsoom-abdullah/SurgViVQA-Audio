#!/bin/bash
# Setup minimal transformers fork for audio-grafted Qwen2-VL on Lambda Labs

set -e

# Resolve the repo root from this script's location, so the script works from any
# checkout. AUDIO_GRAFT_ROOT points at a DIFFERENT repository (the audio-graft
# predecessor) and cannot be derived — override it or edit the default.
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
AUDIO_GRAFT_ROOT="${AUDIO_GRAFT_ROOT:-/path/to/Qwen2-VL-Audio-Graft}"

echo "Setting up transformers fork for audio-grafted model..."

# Create directory structure
mkdir -p transformers_local/transformers/models/qwen2_vl

# Copy modified qwen2_vl files from your fork
# You'll need to scp these from your Mac to Lambda Labs

echo "Directory structure created at: $(pwd)/transformers_local"
echo ""
echo "Next steps:"
echo "1. Copy the modified qwen2_vl files from your Mac:"
echo "   scp -r ${AUDIO_GRAFT_ROOT}/transformers_fork/src/transformers/models/qwen2_vl/* \\"
echo "       ubuntu@YOUR_IP:~/audiograft/SurgViVQA-Audio/baselines/transformers_local/transformers/models/qwen2_vl/"
echo ""
echo "2. Run baseline2 (it will automatically use the local fork)"
