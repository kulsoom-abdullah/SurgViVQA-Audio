#!/bin/bash
################################################################################
# RunPod Setup Script for SurgViVQA-Audio (Using VENV - Persistent)
# Run this script each time you start a new RunPod instance
# VENV is stored in /workspace so it persists across pod restarts
################################################################################

set -e  # Exit on error

echo "================================================================================"
echo "SurgViVQA-Audio - RunPod Environment Setup (VENV)"
echo "================================================================================"
echo ""

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ============================================================================
# 1. System Information
# ============================================================================
echo -e "${BLUE}[1/7] System Information${NC}"
echo "-------------------------------------------"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "Python version: $(python --version)"
echo "Python location: $(which python)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'PyTorch not installed yet')"
echo ""

# ============================================================================
# 2. Create Virtual Environment in /workspace (PERSISTENT)
# ============================================================================
echo -e "${BLUE}[2/7] Setting up Virtual Environment${NC}"
echo "-------------------------------------------"

VENV_PATH="/workspace/venvs/surg-audio"

if [ -d "$VENV_PATH" ]; then
    echo "✓ Virtual environment already exists at: $VENV_PATH"
else
    echo "Creating new virtual environment at: $VENV_PATH"
    mkdir -p /workspace/venvs
    python -m venv "$VENV_PATH"
fi

# Activate environment
source "$VENV_PATH/bin/activate"
echo -e "${GREEN}✓ Virtual environment activated${NC}"
echo "Python: $(which python)"
echo ""

# ============================================================================
# 3. Upgrade pip
# ============================================================================
echo -e "${BLUE}[3/7] Upgrading pip${NC}"
echo "-------------------------------------------"
pip install --upgrade pip -q
echo "✓ pip upgraded"
echo ""

# ============================================================================
# 4. Install PyTorch with CUDA
# ============================================================================
echo -e "${BLUE}[4/7] Installing PyTorch${NC}"
echo "-------------------------------------------"

if python -c "import torch" 2>/dev/null; then
    echo "✓ PyTorch already installed: $(python -c 'import torch; print(torch.__version__)')"
else
    echo "Installing PyTorch with CUDA 11.8..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
fi
echo ""

# ============================================================================
# 5. Install System Dependencies (ffmpeg for audio)
# ============================================================================
echo -e "${BLUE}[5/8] Installing System Dependencies${NC}"
echo "-------------------------------------------"

# Install ffmpeg for audio processing (librosa/whisper requirement)
if command -v ffmpeg &> /dev/null; then
    echo "✓ ffmpeg already installed"
else
    echo "Installing ffmpeg..."
    apt-get update -qq && apt-get install -y -qq ffmpeg
fi
echo ""
# ============================================================================
# 6. Install Python Dependencies
# ============================================================================
echo -e "${BLUE}[6/8] Installing Python Dependencies${NC}"
echo "-------------------------------------------"

# A. First, install the critical build tools
pip install wheel setuptools packaging ninja -q

# B. Link your local transformers fork (The 'Editable' way)
echo "🔗 Linking custom Transformers fork..."
cd /workspace/SurgViVQA-Audio/transformers_fork && pip install -e . && cd ..

# C. Install FlashAttention-2 (Required for Qwen2-VL performance)
echo "⚡ Installing FlashAttention-2..."
pip install flash-attn --no-build-isolation

# D. Install the remaining standard packages
packages=(
    "accelerate"
    "bitsandbytes"
    "librosa"
    "pillow"
    "edge-tts"
    "openai-whisper"
    "tqdm"
    "scipy"
    "soundfile"
    "optimum"
    "qwen-vl-utils"
)


for pkg in "${packages[@]}"; do
    pkg_import=$(echo "$pkg" | tr '-' '_')
    if python -c "import ${pkg_import}" 2>/dev/null; then
        echo "✓ $pkg already installed"
    else
        echo "Installing $pkg..."
        pip install -q "$pkg"
    fi
done

echo -e "${GREEN}✓ All core dependencies installed${NC}"
echo ""

# ============================================================================
# 7. Verify Model Access
# ============================================================================
echo -e "${BLUE}[7/8] Verifying Model Access${NC}"
echo "-------------------------------------------"

python - <<EOF
import sys
try:
    from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer
    print("✓ Qwen2-VL imports successful")
except ImportError as e:
    print(f"⚠️  Import error: {e}")
    sys.exit(1)

try:
    from transformers import WhisperFeatureExtractor
    print("✓ Whisper imports successful")
except ImportError as e:
    print(f"❌ Whisper import failed: {e}")
    sys.exit(1)

try:
    import whisper
    print("✓ OpenAI Whisper installed")
except ImportError:
    print("❌ OpenAI Whisper not found")
    sys.exit(1)

print("\n✓ All model dependencies verified")
EOF

echo ""

# ============================================================================
# 8. Verify Dataset Structure
# ============================================================================
echo -e "${BLUE}[8/8] Verifying Dataset Structure${NC}"
echo "-------------------------------------------"

# Check key directories and files
checks=(
    "test_set/in_002-001.jsonl:Test set (in-domain)"
    "test_set/out_002-001.jsonl:Test set (out-domain)"
    "audio/in_002-001:Audio files (in-domain)"
    "audio/out_002-001:Audio files (out-domain)"
    "dataset/frames/002-001:Video frames"
    "baselines/baseline1_text_image.py:Baseline 1 script"
    "baselines/baseline2_audio_image.py:Baseline 2 script"
    "baselines/baseline3_asr_pipeline.py:Baseline 3 script"
)

all_found=true
for check in "${checks[@]}"; do
    path="${check%%:*}"
    desc="${check##*:}"
    if [ -e "$path" ]; then
        echo "✓ $desc"
    else
        echo -e "${YELLOW}⚠️  Missing: $desc ($path)${NC}"
        all_found=false
    fi
done

if [ "$all_found" = false ]; then
    echo ""
    echo -e "${YELLOW}⚠️  Some files are missing. You may need to upload:${NC}"
    echo "   - Dataset frames (run download_frames_figshare.sh)"
    echo "   - Audio files (or regenerate with generate_audio_subset.py)"
fi
echo ""

# Optional: Automate W&B login if you have the key
wandb login "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"

# ============================================================================
# Setup Complete
# ============================================================================
echo "================================================================================"
echo -e "${GREEN}✓ RunPod Setup Complete!${NC}"
echo "================================================================================"
echo ""
echo "Virtual Environment: $VENV_PATH (PERSISTENT)"
echo "CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""
echo "To activate venv in future sessions:"
echo "  source /workspace/venvs/surg-audio/bin/activate"
echo ""
echo "Next Steps:"
echo "  1. Verify setup: python verify_setup.py --test_file test_set/in_002-001.jsonl --frames_dir dataset/frames --audio_dir audio/in_002-001"
echo "  2. Run baselines: bash run_all_baselines.sh"
echo ""
echo "================================================================================"
