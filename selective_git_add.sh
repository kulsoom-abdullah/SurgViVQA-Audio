#!/bin/bash
# Selective git add - only essential files for training

echo "Adding essential files to git..."

# Core files
git add .gitignore
git add README.md
git add requirements.txt

# Training scripts
git add src/train_vqa.py
git add src/evaluate_checkpoint.py

# Baselines (complete baseline experiments)
git add baselines/baseline1_text_image.py
git add baselines/baseline2_audio_image.py
git add baselines/baseline3_asr_pipeline.py
git add baselines/utils.py

# Custom transformers fork (CRITICAL for audio support)
git add transformers_fork/

# Essential scripts
git add scripts/bootstrap.sh

# Test data (small, ok for Git)
git add test_set/

# Data placeholder
git add data/README.md

echo ""
echo "Files staged for commit:"
git status --short

echo ""
echo "Review the list above. If it looks good:"
echo "  git commit -m 'Initial commit: Audio-Grafted Surgical VQA'"
echo "  git remote add origin https://github.com/YOUR-USERNAME/SurgViVQA-Audio.git"
echo "  git push -u origin main"
