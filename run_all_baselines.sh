#!/bin/bash
################################################################################
# Run All 3 Baselines on Both Test Sets
# This script runs all experiments sequentially
################################################################################

set -e  # Exit on error

# Activate virtual environment if it exists
VENV_PATH="/workspace/venvs/surg-audio"
if [ -d "$VENV_PATH" ]; then
    source "$VENV_PATH/bin/activate"
    echo "✓ Virtual environment activated"
fi

echo "================================================================================"
echo "SurgViVQA-Audio - Running All Baseline Experiments"
echo "================================================================================"
echo ""

# Create results directory
mkdir -p results

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
FRAMES_DIR="dataset/frames"
WHISPER_MODEL="medium"  # Change to "large-v3" if you have enough VRAM

# Model path (optional - defaults to kulsoom-abdullah/Qwen2-Audio-7B-Transcription)
# Uncomment and set if using a local checkpoint:
# MODEL_PATH="/workspace/checkpoints/my_model"
MODEL_PATH=""  # Empty = use default from utils.py

# Test sets
TEST_SETS=(
    "test_set/in_002-001.jsonl:audio/in_002-001:in"
    "test_set/out_002-001.jsonl:audio/out_002-001:out"
)

total_start=$(date +%s)

# ============================================================================
# Run experiments for each test set
# ============================================================================
for test_config in "${TEST_SETS[@]}"; do
    IFS=':' read -r test_file audio_dir prefix <<< "$test_config"

    echo ""
    echo "================================================================================"
    echo -e "${BLUE}Processing Test Set: $test_file${NC}"
    echo "================================================================================"
    echo ""

    # ------------------------------------------------------------------------
    # Baseline 1: Text + Image
    # ------------------------------------------------------------------------
    echo -e "${BLUE}[1/3] Running Baseline 1: Text + Image VQA${NC}"
    echo "-------------------------------------------"

    baseline1_start=$(date +%s)

    # Add model_path if specified
    MODEL_ARG=""
    if [ -n "$MODEL_PATH" ]; then
        MODEL_ARG="--model_path $MODEL_PATH"
    fi

    python baselines/baseline1_text_image.py \
        --test_file "$test_file" \
        --frames_dir "$FRAMES_DIR" \
        --output "results/baseline1_${prefix}.jsonl" \
        $MODEL_ARG

    baseline1_end=$(date +%s)
    baseline1_time=$((baseline1_end - baseline1_start))

    echo -e "${GREEN}✓ Baseline 1 complete (${baseline1_time}s)${NC}"
    echo ""

    # ------------------------------------------------------------------------
    # Baseline 2: Audio + Image (Direct Embedding)
    # ------------------------------------------------------------------------
    echo -e "${BLUE}[2/3] Running Baseline 2: Audio + Image VQA (Direct Audio)${NC}"
    echo "-------------------------------------------"

    baseline2_start=$(date +%s)

    python baselines/baseline2_audio_image.py \
        --test_file "$test_file" \
        --frames_dir "$FRAMES_DIR" \
        --audio_dir "$audio_dir" \
        --output "results/baseline2_${prefix}.jsonl" \
        $MODEL_ARG

    baseline2_end=$(date +%s)
    baseline2_time=$((baseline2_end - baseline2_start))

    echo -e "${GREEN}✓ Baseline 2 complete (${baseline2_time}s)${NC}"
    echo ""

    # ------------------------------------------------------------------------
    # Clear GPU cache before Baseline 3
    # ------------------------------------------------------------------------
    echo "Clearing GPU cache..."
    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    echo ""

    # ------------------------------------------------------------------------
    # Baseline 3: Audio → ASR → Text + Image
    # ------------------------------------------------------------------------
    echo -e "${BLUE}[3/3] Running Baseline 3: Audio → ASR → Text + Image${NC}"
    echo "-------------------------------------------"
    echo "Using Whisper model: $WHISPER_MODEL"
    echo ""

    baseline3_start=$(date +%s)

    python baselines/baseline3_asr_pipeline.py \
        --test_file "$test_file" \
        --frames_dir "$FRAMES_DIR" \
        --audio_dir "$audio_dir" \
        --output "results/baseline3_${prefix}_${WHISPER_MODEL}.jsonl" \
        --whisper_model "$WHISPER_MODEL" \
        $MODEL_ARG

    baseline3_end=$(date +%s)
    baseline3_time=$((baseline3_end - baseline3_start))

    echo -e "${GREEN}✓ Baseline 3 complete (${baseline3_time}s)${NC}"
    echo ""

    # ------------------------------------------------------------------------
    # Test Set Summary
    # ------------------------------------------------------------------------
    echo "================================================================================"
    echo -e "${GREEN}Test Set Complete: $prefix${NC}"
    echo "================================================================================"
    echo "Timing Summary:"
    echo "  Baseline 1 (Text+Image):     ${baseline1_time}s"
    echo "  Baseline 2 (Audio+Image):    ${baseline2_time}s"
    echo "  Baseline 3 (ASR→Text+Image): ${baseline3_time}s"
    echo "  Total:                       $((baseline1_time + baseline2_time + baseline3_time))s"
    echo ""
done

total_end=$(date +%s)
total_time=$((total_end - total_start))
total_minutes=$((total_time / 60))
total_seconds=$((total_time % 60))

# ============================================================================
# Final Summary
# ============================================================================
echo ""
echo "================================================================================"
echo -e "${GREEN}✓ ALL EXPERIMENTS COMPLETE${NC}"
echo "================================================================================"
echo ""
echo "Total Time: ${total_minutes}m ${total_seconds}s"
echo ""
echo "Results saved to:"
ls -lh results/*.jsonl
echo ""
echo "Next Steps:"
echo "  1. Analyze results: python analyze_results.py"
echo "  2. Download results: scp -P PORT -r root@HOST:/workspace/SurgViVQA-Audio/results/ ./results/"
echo "  3. Create visualizations and comparison tables"
echo ""
echo "================================================================================"
