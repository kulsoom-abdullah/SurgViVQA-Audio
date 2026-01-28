#!/bin/bash
# Reorganize project structure for GitHub

set -e

echo "📁 Reorganizing SurgViVQA-Audio for GitHub..."

# Create new directory structure
mkdir -p src
mkdir -p scripts
mkdir -p docs
mkdir -p data  # Placeholder for data symlinks

# Move core training files to src/
echo "Moving training scripts to src/..."
mv train_vqa.py src/ 2>/dev/null || echo "  train_vqa.py already in src/"
mv evaluate_checkpoint.py src/ 2>/dev/null || echo "  evaluate_checkpoint.py already in src/"

# Move debug scripts to scripts/
echo "Moving utility scripts to scripts/..."
mv verify_tokenizer.sh scripts/ 2>/dev/null || true
mv train_with_eval_FINAL.sh scripts/ 2>/dev/null || true
mv check_actual_tokens.py scripts/ 2>/dev/null || true
mv debug_labels.py scripts/ 2>/dev/null || true
mv debug_predictions.py scripts/ 2>/dev/null || true
mv check_tokenizer_config.py scripts/ 2>/dev/null || true

# Move documentation to docs/
echo "Moving documentation to docs/..."
mv EVAL_STRATEGY.md docs/ 2>/dev/null || true
mv FIXES_APPLIED.md docs/ 2>/dev/null || true
mv RUN_TRAINING_A100.md docs/ 2>/dev/null || true
mv INSTRUCTIONS_BF16_TRAINING.md docs/ 2>/dev/null || true
mv RUN_BASELINE2.md docs/ 2>/dev/null || true

# Keep baselines/ as is (already organized)
# Keep transformers_fork/ as is (custom modifications)
# Keep test_set/ as is (small data, ok for Git)

# Create bootstrap script
cat > scripts/bootstrap.sh << 'BOOTSTRAP'
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
BOOTSTRAP

chmod +x scripts/bootstrap.sh

# Create data README
cat > data/README.md << 'DATAREADME'
# Data Directory

This directory should contain your surgical VQA data:

```
data/
├── frames/
│   ├── 002-001/
│   │   ├── 002-001_18743.jpg
│   │   └── ...
│   └── ...
├── audio/
│   ├── in_002-001/
│   │   ├── qa_000357.mp3
│   │   └── ...
│   └── out_002-001/
│       └── ...
```

## Data is NOT in Git

Large data files are excluded via `.gitignore`.

### On RunPod Network Volume:
Data lives at `/workspace/SurgViVQA-Audio/data/` and persists between pod sessions.

### First-time setup:
Upload data once:
```bash
# From your Mac
scp -r audio frames ubuntu@RUNPOD_IP:/workspace/SurgViVQA-Audio/data/
```

### Alternative: Cloud storage
For stateless infrastructure, sync from R2/S3:
```bash
rclone sync r2:surgvqa-data/audio ./data/audio
rclone sync r2:surgvqa-data/frames ./data/frames
```
DATAREADME

echo ""
echo "✅ Reorganization complete!"
echo ""
echo "New structure:"
echo "  src/          - Core training code"
echo "  baselines/    - Baseline experiments"
echo "  scripts/      - Utility scripts"
echo "  docs/         - Documentation"
echo "  data/         - Data directory (excluded from Git)"
echo ""
echo "Next steps:"
echo "1. Review .gitignore"
echo "2. git add ."
echo "3. git commit -m 'Initial commit: Audio-Grafted Surgical VQA'"
echo "4. git remote add origin https://github.com/your-username/SurgViVQA-Audio"
echo "5. git push -u origin main"
echo ""
