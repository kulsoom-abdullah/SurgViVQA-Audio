#!/bin/bash
# Generate audio for videos 002-002, 002-003, 002-004 on RunPod
# Uses edge-tts (already in requirements.txt)

set -e

echo "🎙️ Generating Audio for Multi-Video Dataset"
echo "==========================================="
echo ""
echo "This will generate TTS audio for OUT (natural) questions"
echo "Videos: 002-002, 002-003, 002-004"
echo ""

# Check if edge-tts is installed
if ! python3 -c "import edge_tts" 2>/dev/null; then
    echo "📦 Installing edge-tts..."
    pip install edge-tts
fi

# Create output directories
mkdir -p data/audio/out_002-002
mkdir -p data/audio/out_002-003
mkdir -p data/audio/out_002-004

# Extract questions for each video from out_template.jsonl
echo "📊 Extracting questions by video..."

python3 << 'EOF'
import json

with open('out_template.jsonl') as f:
    all_data = [json.loads(line) for line in f]

# Split by video
for vid in ['002-002', '002-003', '002-004']:
    video_data = [s for s in all_data if s['video_id'] == vid]
    output_file = f'out_{vid}_temp.jsonl'
    with open(output_file, 'w') as f:
        for sample in video_data:
            f.write(json.dumps(sample) + '\n')
    print(f"✓ {output_file}: {len(video_data)} samples")
EOF

# Generate audio for each video
for vid in 002-002 002-003 002-004; do
    echo ""
    echo "🎙️ Generating audio for video ${vid}..."
    python3 generate_audio_subset.py \
        --input out_${vid}_temp.jsonl \
        --output data/audio/out_${vid}
    echo "✓ Audio generated: data/audio/out_${vid}/"
done

# Cleanup temp files
rm -f out_*_temp.jsonl

echo ""
echo "✅ Audio generation complete!"
echo ""
echo "Generated audio for:"
echo "  - data/audio/out_002-002/ ($(ls data/audio/out_002-002/ | wc -l) files)"
echo "  - data/audio/out_002-003/ ($(ls data/audio/out_002-003/ | wc -l) files)"
echo "  - data/audio/out_002-004/ ($(ls data/audio/out_002-004/ | wc -l) files)"
echo ""
echo "Ready for multi-video training!"
