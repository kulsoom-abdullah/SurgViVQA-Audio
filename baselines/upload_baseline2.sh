#!/bin/bash
# One-command setup for Baseline 2 on Lambda Labs

set -e

LAMBDA_IP="132-145-135-107"
LAMBDA_USER="ubuntu"
LAMBDA_DIR="~/audiograft/SurgViVQA-Audio/baselines"

echo "========================================="
echo "Setting up Baseline 2 on Lambda Labs"
echo "========================================="
echo ""

# Step 1: Create directory structure on Lambda
echo "1. Creating directory structure on Lambda..."
ssh ${LAMBDA_USER}@${LAMBDA_IP} "mkdir -p ${LAMBDA_DIR}/transformers_local/transformers/models"

# Step 2: Copy modified qwen2_vl module
echo "2. Copying modified qwen2_vl module..."
scp -r /Users/kulsoom/workspace/learning/Qwen2-VL-Audio-Graft/transformers_fork/src/transformers/models/qwen2_vl \
    ${LAMBDA_USER}@${LAMBDA_IP}:${LAMBDA_DIR}/transformers_local/transformers/models/

# Step 3: Copy updated baseline2 script
echo "3. Copying baseline2 script..."
scp baseline2_audio_image.py ${LAMBDA_USER}@${LAMBDA_IP}:${LAMBDA_DIR}/

echo ""
echo "========================================="
echo "✓ Setup complete!"
echo "========================================="
echo ""
echo "Now on Lambda Labs, run:"
echo "  cd ~/audiograft/SurgViVQA-Audio"
echo "  source ~/venvs/surg-audio/bin/activate"
echo "  python baselines/baseline2_audio_image.py \\"
echo "      --test_file test_set/in_002-001.jsonl \\"
echo "      --frames_dir dataset/frames \\"
echo "      --audio_dir audio/in_002-001 \\"
echo "      --output results/baseline2_in.jsonl"
echo ""
