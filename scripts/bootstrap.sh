#!/bin/bash
# Bootstrap script for RunPod Network Volume setup

set -e

echo "🚀 Bootstrapping SurgViVQA-Audio environment..."

# Install dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Verify transformers fork
if [ -d "transformers_fork" ]; then
    echo "✅ Custom transformers fork found"
else
    echo "⚠️  transformers_fork/ not found - you may need to copy it"
fi

# Create data directories if they don't exist
mkdir -p data/frames
mkdir -p data/audio
mkdir -p checkpoints
mkdir -p results

echo ""
echo "✅ Bootstrap complete!"
echo ""
echo "Next steps:"
echo "1. Upload your data to data/ directory (audio, frames)"
echo "2. Run training: python src/train_vqa.py --train_data_path test_set/in_002-001.jsonl ..."
echo ""
