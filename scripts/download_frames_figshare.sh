#!/bin/bash
################################################################################
# Download REAL-Colon Frames from Figshare Directly on RunPod
# Run this ON RUNPOD, not locally
################################################################################

set -e

echo "================================================================================"
echo "Downloading REAL-Colon Dataset from Figshare"
echo "================================================================================"

# Navigate to dataset directory
cd /workspace/SurgViVQA-Audio/dataset

# Download from Figshare (7.6GB)
echo "📥 Downloading from Figshare..."
echo "URL: https://plus.figshare.com/ndownloader/files/39530437"
echo "Size: 7.6GB"
echo "Estimated time: 2-5 minutes on cloud bandwidth"
echo ""

wget -O real_colon_frames.zip 'https://plus.figshare.com/ndownloader/files/39530437'

echo ""
echo "✓ Download complete!"
echo ""

# Check what we got
echo "📦 Extracting archive..."
unzip -l real_colon_frames.zip | head -20
echo ""
echo "Extracting to temp_extract/..."
unzip -q real_colon_frames.zip -d temp_extract/

echo ""
echo "📂 Archive contents:"
ls -la temp_extract/
echo ""

# Create frames directory if it doesn't exist
mkdir -p frames

# Check for 002-001 folder
echo "🔍 Looking for 002-001 folder..."
if [ -d "temp_extract/002-001" ]; then
    echo "✓ Found 002-001 at root level"
    mv temp_extract/002-001 frames/
elif [ -d "temp_extract/frames/002-001" ]; then
    echo "✓ Found 002-001 in frames subdirectory"
    mv temp_extract/frames/002-001 frames/
else
    echo "⚠️  002-001 folder not found in expected location"
    echo "Available folders:"
    find temp_extract -type d -maxdepth 3
    echo ""
    echo "Please manually move 002-001 to frames/ directory"
    echo "Example: mv temp_extract/path/to/002-001 frames/"
    exit 1
fi

# Verify frame count
echo ""
echo "✅ Verifying frames..."
FRAME_COUNT=$(ls frames/002-001/ | wc -l)
echo "Frame count: $FRAME_COUNT"

if [ "$FRAME_COUNT" -eq 25998 ]; then
    echo "✓ Correct! All 25,998 frames present"
else
    echo "⚠️  Expected 25,998 frames but found $FRAME_COUNT"
    echo "This might still be OK if the dataset structure is different"
fi

# Cleanup
echo ""
echo "🧹 Cleaning up..."
rm -rf temp_extract/ real_colon_frames.zip

echo ""
echo "================================================================================"
echo "✓ FRAMES READY!"
echo "================================================================================"
echo "Location: /workspace/SurgViVQA-Audio/dataset/frames/002-001/"
echo "Frame count: $FRAME_COUNT"
echo ""
echo "Next steps:"
echo "  cd /workspace/SurgViVQA-Audio"
echo "  python verify_setup.py --test_file test_set/in_002-001.jsonl --frames_dir dataset/frames --audio_dir audio/in_002-001"
echo "================================================================================"
