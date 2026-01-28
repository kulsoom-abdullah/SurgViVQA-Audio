#!/bin/bash
# Check frame naming patterns

echo "=== Checking frame naming patterns ==="
echo ""

echo "Total frames in 002-001:"
ls dataset/frames/002-001/ | wc -l

echo ""
echo "First 10 frames:"
ls dataset/frames/002-001/ | head -10

echo ""
echo "Last 10 frames:"
ls dataset/frames/002-001/ | tail -10

echo ""
echo "Checking if test set frame exists:"
echo "Looking for: 002-001_18743.jpg"
ls dataset/frames/002-001/002-001_18743.jpg 2>/dev/null && echo "✓ Found!" || echo "❌ Not found"

echo ""
echo "Pattern check - sequential vs frame numbers:"
echo "Sequential pattern (002-001_0.jpg to 002-001_25997.jpg):"
ls dataset/frames/002-001/002-001_25997.jpg 2>/dev/null && echo "✓ Found last sequential frame" || echo "❌ Not found"

echo ""
echo "Checking range around frame 18743:"
ls dataset/frames/002-001/ | grep -E "002-001_(1874[0-9]|1875[0-9])" | head -5
