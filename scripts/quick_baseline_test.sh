#!/bin/bash
# Quick pipeline test using 10 samples
# Run this before the full expensive experiment

set -e

echo "=================================================="
echo "QUICK BASELINE PIPELINE TEST (10 samples)"
echo "=================================================="
echo ""

# Configuration
TEST_FILE="test_sample_10.jsonl"
FRAMES_DIR="data/frames"
RESULTS_DIR="results/quick_test"

# Create results directory
mkdir -p "$RESULTS_DIR"

# Check if 10-sample file exists
if [ ! -f "$TEST_FILE" ]; then
    echo "❌ Error: $TEST_FILE not found"
    echo "Run: python scripts/test_vision_baselines.py first"
    exit 1
fi

echo "✅ Using test file: $TEST_FILE (10 samples)"
echo ""

# Test Baseline 1: Qwen 2.0
echo "🧪 Testing Qwen2-VL-7B-Instruct (10 samples)..."
echo "=================================================="

python baselines/baseline1_text_image.py \
    --test_file "$TEST_FILE" \
    --frames_dir "$FRAMES_DIR" \
    --output "$RESULTS_DIR/quick_qwen2.0.jsonl" \
    --model_path "Qwen/Qwen2-VL-7B-Instruct"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Qwen 2.0 test PASSED"
else
    echo ""
    echo "❌ Qwen 2.0 test FAILED"
    exit 1
fi

# Verify output
echo ""
echo "📊 Checking output format..."
if [ -f "$RESULTS_DIR/quick_qwen2.0.jsonl" ]; then
    lines=$(wc -l < "$RESULTS_DIR/quick_qwen2.0.jsonl")
    echo "  ✅ Output file created: $lines results"

    # Check first result has required fields
    first_result=$(head -1 "$RESULTS_DIR/quick_qwen2.0.jsonl")

    if echo "$first_result" | grep -q "question_type" && \
       echo "$first_result" | grep -q "exact_match" && \
       echo "$first_result" | grep -q "inference_time_ms"; then
        echo "  ✅ Required fields present"
    else
        echo "  ❌ Missing required fields in output"
        exit 1
    fi
else
    echo "  ❌ Output file not created"
    exit 1
fi

echo ""
echo "=================================================="
echo "✅ QUICK PIPELINE TEST PASSED!"
echo "=================================================="
echo ""
echo "Output saved to: $RESULTS_DIR/quick_qwen2.0.jsonl"
echo ""
echo "Next steps:"
echo "  1. If on local machine: Launch RunPod A100 80GB"
echo "  2. Upload code and run full experiment:"
echo "     bash scripts/run_all_vision_baselines.sh"
echo ""
echo "Expected runtime for full experiment (1,000 samples × 3 models):"
echo "  - A100 80GB: ~2-3 hours"
echo "  - RTX 4090:  ~4-6 hours"
echo ""
