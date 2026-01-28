#!/bin/bash
# Run baseline comparisons across Qwen 2.0, 2.5, and 3.0 VL models
# Purpose: Test if newer vision encoders improve temporal reasoning

set -e

echo "=================================================="
echo "Vision Encoder Comparison Study"
echo "Testing: Qwen 2.0 vs 2.5 vs 3.0 VL on full test set"
echo "=================================================="
echo ""

# Configuration
TEST_FILE="test_multivideo.jsonl"
FRAMES_DIR="data/frames"
RESULTS_DIR="results/vision_comparison"

# Create results directory
mkdir -p "$RESULTS_DIR"

# Baseline 1: Qwen 2.0 VL (Current)
echo "🔬 Running Baseline 1: Qwen2-VL-7B-Instruct (zero-shot)"
echo "=================================================="
python baselines/baseline1_text_image.py \
    --test_file "$TEST_FILE" \
    --frames_dir "$FRAMES_DIR" \
    --output "$RESULTS_DIR/baseline_qwen2.0_full_test.jsonl" \
    --model_path "Qwen/Qwen2-VL-7B-Instruct"

echo ""
echo "✅ Qwen 2.0 baseline complete"
echo ""

# Baseline 2: Qwen 2.5 VL
echo "🔬 Running Baseline 2: Qwen2.5-VL-7B-Instruct (zero-shot)"
echo "=================================================="
python baselines/baseline1_text_image.py \
    --test_file "$TEST_FILE" \
    --frames_dir "$FRAMES_DIR" \
    --output "$RESULTS_DIR/baseline_qwen2.5_full_test.jsonl" \
    --model_path "Qwen/Qwen2.5-VL-7B-Instruct"

echo ""
echo "✅ Qwen 2.5 baseline complete"
echo ""

# Baseline 3: Qwen 3.0 VL
echo "🔬 Running Baseline 3: Qwen3-VL-8B-Instruct (zero-shot)"
echo "=================================================="
python baselines/baseline1_text_image.py \
    --test_file "$TEST_FILE" \
    --frames_dir "$FRAMES_DIR" \
    --output "$RESULTS_DIR/baseline_qwen3.0_full_test.jsonl" \
    --model_path "Qwen/Qwen3-VL-8B-Instruct"

echo ""
echo "✅ Qwen 3.0 baseline complete"
echo ""

# Generate comparison report
echo "📊 Generating comparison report..."
python scripts/compare_vision_baselines.py \
    --qwen2 "$RESULTS_DIR/baseline_qwen2.0_full_test.jsonl" \
    --qwen25 "$RESULTS_DIR/baseline_qwen2.5_full_test.jsonl" \
    --qwen3 "$RESULTS_DIR/baseline_qwen3.0_full_test.jsonl" \
    --finetuned "results/final_test_002004.jsonl" \
    --output "$RESULTS_DIR/vision_comparison_report.txt"

echo ""
echo "=================================================="
echo "✅ All baselines complete!"
echo "=================================================="
echo "Results saved in: $RESULTS_DIR/"
echo "  - baseline_qwen2.0_full_test.jsonl"
echo "  - baseline_qwen2.5_full_test.jsonl"
echo "  - baseline_qwen3.0_full_test.jsonl"
echo "  - vision_comparison_report.txt"
echo ""
