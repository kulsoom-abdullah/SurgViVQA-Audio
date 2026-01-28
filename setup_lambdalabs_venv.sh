#!/bin/bash
################################################################################
# Lambda Labs Setup Script for SurgViVQA-Audio (Using VENV)
# Run this script when you start a new Lambda Labs instance
################################################################################

set -e  # Exit on error

# ============================================================================
# CONFIGURATION - Customize these paths if needed
# ============================================================================
# Project directory (auto-detected from script location)
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Virtual environment location (can be customized)
VENV_PATH="$HOME/venvs/surg-audio"

# Dataset paths (relative to PROJECT_DIR)
FRAMES_DIR="$PROJECT_DIR/dataset/frames"
AUDIO_DIR="$PROJECT_DIR/audio"
TEST_SET_DIR="$PROJECT_DIR/test_set"

echo "================================================================================"
echo "SurgViVQA-Audio - Lambda Labs Environment Setup (VENV)"
echo "================================================================================"
echo ""
echo "Project Directory: $PROJECT_DIR"
echo "Venv Location: $VENV_PATH"
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
echo "Python version: $(python3 --version)"
echo "Python location: $(which python3)"
echo "CUDA available: $(python3 -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'PyTorch not installed yet')"
echo ""

# ============================================================================
# 2. Create Virtual Environment in home directory
# ============================================================================
echo -e "${BLUE}[2/7] Setting up Virtual Environment${NC}"
echo "-------------------------------------------"

# Lambda Labs typically uses /home/ubuntu
VENV_PATH="$HOME/venvs/surg-audio"

if [ -d "$VENV_PATH" ]; then
    echo "✓ Virtual environment already exists at: $VENV_PATH"
else
    echo "Creating new virtual environment at: $VENV_PATH"
    mkdir -p "$HOME/venvs"
    python3 -m venv "$VENV_PATH"
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
    echo "Installing PyTorch with CUDA 12.1..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
fi
echo ""

# ============================================================================
# 5. Install System Dependencies (ffmpeg for audio)
# ============================================================================
echo -e "${BLUE}[5/7] Installing System Dependencies${NC}"
echo "-------------------------------------------"

# Install ffmpeg for audio processing (librosa/whisper requirement)
if command -v ffmpeg &> /dev/null; then
    echo "✓ ffmpeg already installed"
else
    echo "Installing ffmpeg..."
    sudo apt-get update -qq && sudo apt-get install -y -qq ffmpeg
fi
echo ""

# ============================================================================
# 6. Install Python Dependencies
# ============================================================================
echo -e "${BLUE}[6/7] Installing Python Dependencies${NC}"
echo "-------------------------------------------"

# Check and install each package
packages=(
    "transformers"
    "accelerate"
    "bitsandbytes"
    "librosa"
    "pillow"
    "edge-tts"
    "openai-whisper"
    "tqdm"
    "scipy"
    "soundfile"
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
echo -e "${BLUE}[7/7] Verifying Model Access${NC}"
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
# Setup Complete
# ============================================================================
echo "================================================================================"
echo -e "${GREEN}✓ Lambda Labs Setup Complete!${NC}"
echo "================================================================================"
echo ""
echo "Project Directory: $PROJECT_DIR"
echo "Virtual Environment: $VENV_PATH"
echo "CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""
echo "To activate venv in future sessions:"
echo "  source $VENV_PATH/bin/activate"
echo "  # OR use the helper script:"
echo "  cd $PROJECT_DIR && source activate_venv_lambdalabs.sh"
echo ""
echo "Next Steps:"
echo "  1. Extract frames: unzip frames_002-001.zip -d $FRAMES_DIR/"
echo "  2. Generate audio: python generate_audio_subset.py --input $TEST_SET_DIR/in_002-001.jsonl --output $AUDIO_DIR/in_002-001"
echo "  3. Generate audio: python generate_audio_subset.py --input $TEST_SET_DIR/out_002-001.jsonl --output $AUDIO_DIR/out_002-001"
echo "  4. Verify setup: python verify_setup.py --test_file $TEST_SET_DIR/in_002-001.jsonl --frames_dir $FRAMES_DIR --audio_dir $AUDIO_DIR/in_002-001"
echo "  5. Run baselines: bash run_all_baselines.sh"
echo ""
echo "================================================================================"
