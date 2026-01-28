#!/bin/bash
# Safe git pull on RunPod - preserves your local multi-GPU changes

echo "🔄 Safely pulling updates from GitHub..."
echo ""

# 1. Stash your local changes
echo "1. Saving your local changes..."
git stash push -m "RunPod multi-GPU config"

# 2. Pull from GitHub
echo "2. Pulling from GitHub..."
git pull origin main

# 3. Re-apply your local changes
echo "3. Re-applying your local changes..."
git stash pop

echo ""
echo "✅ Done! If there are conflicts, resolve them and commit."
echo ""
echo "Your changes:"
echo "  - Commented out device_map='auto' in train_vqa.py (for multi-GPU)"
echo "  - Added NUM_GPUS logic in train_runpod.sh"
echo ""
