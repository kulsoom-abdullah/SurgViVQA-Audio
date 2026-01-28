#!/bin/bash
# Download and setup transformers fork from HuggingFace (no scp needed!)

set -e

echo "========================================="
echo "Setting up transformers fork from HF"
echo "========================================="
echo ""

cd ~/audiograft/SurgViVQA-Audio/baselines

# Download the fork from your HF repo
echo "1. Downloading transformers_fork from HuggingFace..."
git clone https://huggingface.co/kulsoom-abdullah/Qwen2-Audio-7B-Transcription temp_download

# Move just the transformers_fork to the right place
echo "2. Extracting transformers_fork..."
mv temp_download/transformers_fork/src transformers_local

# Clean up
rm -rf temp_download

echo ""
echo "✓ Setup complete!"
echo ""
echo "Now you can run:"
echo "  python baselines/baseline2_audio_image.py \\"
echo "      --test_file test_set/in_002-001.jsonl \\"
echo "      --frames_dir dataset/frames \\"
echo "      --audio_dir audio/in_002-001 \\"
echo "      --output results/baseline2_in.jsonl"
