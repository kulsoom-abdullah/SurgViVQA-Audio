#!/bin/bash
# Complete GitHub setup - selective files only

set -e

echo "🚀 Setting up GitHub repository..."
echo ""

# Initialize git if needed
if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
    echo "✓ Git initialized"
else
    echo "✓ Git already initialized"
fi

echo ""
echo "📦 Adding essential files..."

# Core project files
git add .gitignore
git add README.md
git add requirements.txt

# Training scripts (essential)
git add src/train_vqa.py
git add src/evaluate_checkpoint.py

# Baselines (your experiments)
git add baselines/

# Custom transformers (CRITICAL - has audio support modifications)
git add transformers_fork/

# Bootstrap script (for RunPod setup)
git add scripts/bootstrap.sh

# Test data (small JSONL files, ok for Git)
git add test_set/

# Data directory placeholder
git add data/README.md

echo ""
echo "✓ Files staged"
echo ""
echo "📋 What's being committed:"
git status --short
echo ""

echo "File count by category:"
echo "  Core files: .gitignore, README.md, requirements.txt"
echo "  Training: src/train_vqa.py, src/evaluate_checkpoint.py"
echo "  Baselines: baselines/ (4 files)"
echo "  Custom transformers: transformers_fork/ (your audio modifications)"
echo "  Scripts: scripts/bootstrap.sh"
echo "  Test data: test_set/ (2 small JSONL files)"
echo ""

echo "NOT committing (excluded by .gitignore):"
echo "  ✗ audio/ (large MP3 files)"
echo "  ✗ dataset/frames/ (large image files)"
echo "  ✗ checkpoints/ (model weights)"
echo "  ✗ venv/ (Python environment)"
echo "  ✗ Debug scripts (scripts/debug_*.py, etc.)"
echo ""

read -p "Ready to commit? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "📝 Creating commit..."
    git commit -m "Initial commit: Audio-Grafted Surgical VQA

Core training pipeline:
- Audio grafting (Whisper + Qwen2-VL)
- QLoRA training with label masking fix
- Baseline experiments (text, audio, ASR)
- Custom transformers fork for audio support

Excludes: data files, checkpoints (will sync via RunPod volume)"

    echo ""
    echo "✅ Commit created!"
    echo ""
    echo "Next steps:"
    echo "1. Create private repo on GitHub: https://github.com/new"
    echo "   Name: SurgViVQA-Audio"
    echo "   Visibility: Private"
    echo ""
    echo "2. Add remote and push:"
    echo "   git remote add origin https://github.com/YOUR-USERNAME/SurgViVQA-Audio.git"
    echo "   git branch -M main"
    echo "   git push -u origin main"
    echo ""
else
    echo ""
    echo "Commit cancelled. Files are still staged."
    echo "You can:"
    echo "  - Review: git status"
    echo "  - Unstage: git reset"
    echo "  - Commit manually: git commit -m 'Your message'"
fi
