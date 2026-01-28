#!/bin/bash
# Quick test using simple text-only baseline (no audio)

set -e

echo "=================================================="
echo "QUICK VISION BASELINE TEST (10 samples)"
echo "Using BASE Qwen2-VL (no audio grafting)"
echo "=================================================="
echo ""

# Test on 10 samples
python baselines/baseline_text_only_simple.py \
    --test_file test_sample_10.jsonl \
    --frames_dir data/frames \
    --output results/quick_test/qwen2_text_only.jsonl \
    --model "Qwen/Qwen2-VL-7B-Instruct"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ TEST PASSED!"
    echo ""
    echo "Ready to run full experiment on 1,000 samples"
    echo "Estimated time on RTX 4090: ~2-3 hours"
else
    echo ""
    echo "❌ TEST FAILED"
    exit 1
fi
